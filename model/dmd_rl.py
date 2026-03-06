"""
DMD + RL (Reinforcement Learning) Model

This module implements the combination of Distribution Matching Distillation (DMD)
with Reinforcement Learning from video reward models.

Two RL modes are supported:

1. **Differentiable RL** (default, reward_forcing=false):
   Loss: loss_gen = dmd_loss + rl_loss_weight * (-reward)
   Gradients flow through the reward model back to the generator.

2. **Reward Forcing** (reward_forcing=true):
   Loss: loss_gen = exp(beta * reward) * dmd_loss
   Reward is computed with NO gradient (pure scalar weight on DMD loss).
   Inspired by: "Reward Forcing" (https://arxiv.org/abs/2512.04678)
   - Much lower VRAM (no reward model gradient chain)
   - No need for FSDP on reward model or gradient checkpointing
   - Supports per-dimension weighting: VQ, MQ, TA, overall, or all (logs all 4)

Features:
- RL loss computed from VAE-decoded videos using DifferentiableVideoReward
- Cold start: RL loss only participates after a specified number of training steps
- Reward normalization support for stable training
- VAE decoding with gradient checkpointing for memory efficiency
- Pixel-space frame sampling: only a configurable subset of decoded frames is sent to the reward model
- Gradient checkpointing on reward model (Qwen2-VL)
- FSDP support for the reward model
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from typing import Optional, Tuple
import math

from model.dmd import DMD


class DMDRL(DMD):
    """
    DMD + RL Model combining distribution matching distillation with
    reinforcement learning from video reward models.
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        # RL hyperparameters
        self.rl_loss_weight = getattr(args, "rl_loss_weight", 1.0)
        self.rl_cold_start_steps = getattr(args, "rl_cold_start_steps", 0)
        self.rl_enabled = False  # Will be enabled after cold start

        # Reward model configuration
        self.rl_reward_checkpoint = getattr(args, "rl_reward_checkpoint", None)
        self.rl_target_height = getattr(args, "rl_target_height", 336)  # Must be multiple of 28
        self.rl_target_width = getattr(args, "rl_target_width", 504)    # Must be multiple of 28
        self.rl_reward_type = getattr(args, "rl_reward_type", "overall")  # VQ, MQ, TA, overall

        # Memory optimization options
        self.rl_reward_fsdp = getattr(args, "rl_reward_fsdp", False)

        # Frame sampling: only send a subset of frames to the reward model
        # This dramatically reduces VRAM since Qwen2-VL attention is quadratic in token count
        # 0 = no sampling (use all frames)
        self.rl_reward_num_frames = getattr(args, "rl_reward_num_frames", 10)

        # Reward normalization: use inference_config from reward model checkpoint
        # (VQ_mean/std, MQ_mean/std, TA_mean/std) for proper z-score normalization
        # These are loaded lazily when the reward model is initialized
        self._reward_norm_mean = None  # [3] tensor: [VQ_mean, MQ_mean, TA_mean]
        self._reward_norm_std = None   # [3] tensor: [VQ_std, MQ_std, TA_std]

        # Running EMA for reward monitoring only (not used in loss computation)
        self.rl_reward_ema_mean = 0.0
        self.rl_reward_ema_std = 1.0
        self.rl_reward_ema_decay = getattr(args, "rl_reward_ema_decay", 0.99)
        self._reward_stats_initialized = False

        # =============================================
        # Reward Forcing mode (no-grad reward weighting)
        # =============================================
        self.reward_forcing = getattr(args, "reward_forcing", False)
        self.reward_forcing_beta = getattr(args, "reward_forcing_beta", 2.0)
        # Which score to use for weighting: "VQ", "MQ", "TA", "overall", or "all"
        # "all" means use overall for weighting but log all individual scores too
        self.reward_forcing_score_type = getattr(args, "reward_forcing_score_type", "overall")

        if self.reward_forcing:
            print(f"[DMDRL] Reward Forcing mode ENABLED: beta={self.reward_forcing_beta}, "
                  f"score_type={self.reward_forcing_score_type}")
            print(f"[DMDRL] Reward is a NO-GRAD scalar weight on DMD loss: "
                  f"loss = exp(beta * reward) * dmd_loss")

        # Validate target dimensions
        assert self.rl_target_height % 28 == 0, f"rl_target_height must be multiple of 28, got {self.rl_target_height}"
        assert self.rl_target_width % 28 == 0, f"rl_target_width must be multiple of 28, got {self.rl_target_width}"

        # Lazy initialization of reward model (to save memory during cold start)
        self._reward_model = None
        self._reward_model_initialized = False

    def _initialize_reward_model(self):
        """
        Lazily initialize the reward model when RL is first enabled.
        Enables gradient checkpointing and optional FSDP wrapping.
        """
        if self._reward_model_initialized:
            return

        if self.rl_reward_checkpoint is None:
            raise ValueError("rl_reward_checkpoint must be specified for RL training")

        from VideoAlign.inference import DifferentiableVideoReward

        print(f"[DMDRL] Initializing reward model from {self.rl_reward_checkpoint}")
        self._reward_model = DifferentiableVideoReward(
            load_from_pretrained=self.rl_reward_checkpoint,
            device=self.device,
            dtype=self.dtype
        )

        # Freeze reward model parameters
        self._reward_model.inferencer.model.requires_grad_(False)

        if self.reward_forcing:
            # Reward Forcing mode: no gradient through reward model at all.
            # Gradient checkpointing not needed (no backward through reward model).
            # FSDP is still useful to shard frozen Qwen2-VL weights across GPUs.
            print(f"[DMDRL] Reward Forcing: reward model in pure eval/no-grad mode")
            if self.rl_reward_fsdp:
                self._fsdp_wrap_reward_model()
        else:
            # Differentiable RL: gradients flow THROUGH the frozen reward model
            # Enable gradient checkpointing on the reward model (Qwen2VL supports this)
            reward_qwen_model = self._reward_model.inferencer.model
            if hasattr(reward_qwen_model, 'gradient_checkpointing_enable'):
                reward_qwen_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print(f"[DMDRL] Reward model gradient checkpointing enabled")
            else:
                print(f"[DMDRL] Warning: reward model does not support gradient_checkpointing_enable")

            # Optional FSDP wrapping for the reward model
            if self.rl_reward_fsdp:
                self._fsdp_wrap_reward_model()

        # Load normalization stats from inference_config in the checkpoint
        inference_config = self._reward_model.inferencer.inference_config
        if inference_config is not None:
            self._reward_norm_mean = torch.tensor(
                [inference_config['VQ_mean'], inference_config['MQ_mean'], inference_config['TA_mean']],
                device=self.device, dtype=torch.float32
            )
            self._reward_norm_std = torch.tensor(
                [inference_config['VQ_std'], inference_config['MQ_std'], inference_config['TA_std']],
                device=self.device, dtype=torch.float32
            )
            print(f"[DMDRL] Reward normalization loaded from inference_config: "
                  f"VQ({inference_config['VQ_mean']:.4f}±{inference_config['VQ_std']:.4f}), "
                  f"MQ({inference_config['MQ_mean']:.4f}±{inference_config['MQ_std']:.4f}), "
                  f"TA({inference_config['TA_mean']:.4f}±{inference_config['TA_std']:.4f})")
        else:
            print(f"[DMDRL] Warning: no inference_config found in checkpoint, "
                  f"using raw logits without normalization")

        self._reward_model_initialized = True
        print(f"[DMDRL] Reward model initialized and frozen successfully")
        print(f"[DMDRL] VAE decode: checkpoint(decode(all)), then sample "
              f"{self.rl_reward_num_frames} pixel frames")

    def _fsdp_wrap_reward_model(self):
        """
        Wrap the reward model with FSDP to shard its parameters across GPUs.
        Even though the reward model is frozen, FSDP sharding reduces per-GPU
        memory usage for the model weights.
        """
        import torch.distributed as dist
        if not dist.is_initialized():
            print(f"[DMDRL] Distributed not initialized, skipping reward model FSDP")
            return

        from utils.distributed import fsdp_wrap

        cpu_offload = getattr(self.args, "rl_reward_cpu_offload", False)
        print(f"[DMDRL] Wrapping reward model with FSDP (cpu_offload={cpu_offload})...")
        self._reward_model.inferencer.model = fsdp_wrap(
            self._reward_model.inferencer.model,
            sharding_strategy=getattr(self.args, "sharding_strategy", "full"),
            mixed_precision=getattr(self.args, "mixed_precision", True),
            wrap_strategy="size",
            cpu_offload=cpu_offload,
        )
        print(f"[DMDRL] Reward model FSDP wrapping complete")

    def enable_rl(self, current_step: int) -> bool:
        if not self.rl_enabled and current_step >= self.rl_cold_start_steps:
            print(f"[DMDRL] Enabling RL at step {current_step} (cold start: {self.rl_cold_start_steps})")
            self.rl_enabled = True
            self._initialize_reward_model()
        return self.rl_enabled

    def _update_reward_stats(self, reward_value: float):
        if not self._reward_stats_initialized:
            self.rl_reward_ema_mean = reward_value
            self.rl_reward_ema_std = 1.0
            self._reward_stats_initialized = True
        else:
            delta = reward_value - self.rl_reward_ema_mean
            self.rl_reward_ema_mean = self.rl_reward_ema_mean + (1 - self.rl_reward_ema_decay) * delta
            self.rl_reward_ema_std = self.rl_reward_ema_decay * self.rl_reward_ema_std + \
                                     (1 - self.rl_reward_ema_decay) * abs(delta)
            self.rl_reward_ema_std = max(self.rl_reward_ema_std, 0.1)

    def _normalize_logits(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw reward logits [1, 3] using the inference_config from the
        reward model checkpoint. This is a differentiable linear transform:
            normalized[i] = (raw[i] - mean[i]) / std[i]

        After normalization, each dimension (VQ, MQ, TA) is a z-score centered
        around 0 with unit variance, making them comparable and well-scaled for RL.

        Args:
            rewards: [1, 3] raw logits tensor (VQ, MQ, TA) with gradient

        Returns:
            normalized: [1, 3] z-score normalized tensor (gradient preserved)
        """
        if self._reward_norm_mean is None or self._reward_norm_std is None:
            return rewards

        # Differentiable z-score normalization: (x - mean) / std
        mean = self._reward_norm_mean.to(device=rewards.device, dtype=rewards.dtype)
        std = self._reward_norm_std.to(device=rewards.device, dtype=rewards.dtype)
        return (rewards - mean.unsqueeze(0)) / std.unsqueeze(0)

    def _sample_frames(self, video: torch.Tensor, num_frames: int = 0) -> torch.Tensor:
        """
        Uniformly sample a subset of frames from a video tensor.
        Ensures an even number of frames (reward model requires temporal_patch_size=2).

        Args:
            video: [T, C, H, W] video tensor (with gradient)
            num_frames: number of frames to sample (0 = use self.rl_reward_num_frames)

        Returns:
            sampled: [T_sampled, C, H, W] sampled frames (gradient preserved)
        """
        T = video.shape[0]
        if num_frames <= 0:
            num_frames = self.rl_reward_num_frames

        if num_frames <= 0 or num_frames >= T:
            if T % 2 != 0:
                video = video[:T - 1]
            return video

        num_frames = num_frames if num_frames % 2 == 0 else num_frames - 1
        num_frames = max(num_frames, 4)

        indices = torch.linspace(0, T - 1, num_frames).round().long()
        return video[indices]

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode all latent frames to pixel space.
        Each batch item is wrapped in its own gradient checkpoint to reduce
        peak activation memory during the backward pass.

        Args:
            latent: [B, T, C, H, W] full latent tensor (e.g., T=21)

        Returns:
            pixel_video: [B, T_pixel, C, H, W] decoded video (e.g., T_pixel=81)
        """
        batch_size = latent.shape[0]
        if batch_size == 1:
            # Single-item fast path: one checkpoint wrapping the whole decode
            def _decode(lat):
                return self.vae.decode_to_pixel(lat)
            return checkpoint_utils.checkpoint(_decode, latent, use_reentrant=False)

        # Multi-item path: checkpoint each item independently to cap peak memory
        chunks = []
        for i in range(batch_size):
            single = latent[i:i + 1]  # keep batch dim: [1, T, C, H, W]
            def _decode_single(lat):
                return self.vae.decode_to_pixel(lat)
            decoded = checkpoint_utils.checkpoint(_decode_single, single, use_reentrant=False)
            chunks.append(decoded)
        return torch.cat(chunks, dim=0)

    def compute_reward_forcing_weight(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[float, dict]:
        """
        Compute reward weight for Reward Forcing mode (NO gradient).

        The reward is used as a scalar multiplier on DMD loss:
            loss = exp(beta * reward) * dmd_loss

        Higher reward → larger weight → stronger DMD gradient signal
        Lower reward → smaller weight → weaker DMD gradient signal

        All computation is done with torch.no_grad().

        Args:
            latent: Generated latent tensor [B, T, C, H, W]
            text_prompts: List of text prompts

        Returns:
            reward_weight: float scalar = exp(beta * score)
            rf_log_dict: Dictionary containing all reward scores for logging
        """
        if not self.rl_enabled or self._reward_model is None:
            return 1.0, {
                "rf_weight": 1.0,
                "rf_reward_VQ": 0.0,
                "rf_reward_MQ": 0.0,
                "rf_reward_TA": 0.0,
                "rf_reward_overall": 0.0,
                "rl_enabled": False
            }

        batch_size = latent.shape[0]

        with torch.no_grad():
            # Decode latent to pixel space (no gradient needed)
            pixel_video = self.vae.decode_to_pixel(latent)

            all_vq, all_mq, all_ta = [], [], []
            all_vq_norm, all_mq_norm, all_ta_norm = [], [], []

            for i in range(batch_size):
                single_video = pixel_video[i]  # [T_decoded, 3, H, W]
                sampled_video = self._sample_frames(single_video)
                prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts

                raw_logits = self._reward_model.compute_reward_from_vae_output(
                    vae_output=sampled_video,
                    prompt=prompt,
                    target_height=self.rl_target_height,
                    target_width=self.rl_target_width
                )  # [1, 3] → VQ, MQ, TA

                # Raw scores
                vq_raw = raw_logits[0, 0].item()
                mq_raw = raw_logits[0, 1].item()
                ta_raw = raw_logits[0, 2].item()
                all_vq.append(vq_raw)
                all_mq.append(mq_raw)
                all_ta.append(ta_raw)

                # Normalized scores
                normalized = self._normalize_logits(raw_logits)
                all_vq_norm.append(normalized[0, 0].item())
                all_mq_norm.append(normalized[0, 1].item())
                all_ta_norm.append(normalized[0, 2].item())

        # Average across batch
        avg_vq = sum(all_vq_norm) / len(all_vq_norm)
        avg_mq = sum(all_mq_norm) / len(all_mq_norm)
        avg_ta = sum(all_ta_norm) / len(all_ta_norm)
        avg_overall = avg_vq + avg_mq + avg_ta

        # Select which score to use for weighting
        score_type = self.reward_forcing_score_type
        if score_type == "VQ":
            selected_score = avg_vq
        elif score_type == "MQ":
            selected_score = avg_mq
        elif score_type == "TA":
            selected_score = avg_ta
        else:  # "overall" or "all"
            selected_score = avg_overall

        # Compute weight: exp(beta * score)
        # Clamp to prevent numerical issues (exp overflow)
        clamped = max(min(self.reward_forcing_beta * selected_score, 20.0), -20.0)
        reward_weight = math.exp(clamped)

        # Update EMA for monitoring
        self._update_reward_stats(selected_score)

        rf_log_dict = {
            "rf_weight": reward_weight,
            "rf_beta_x_score": self.reward_forcing_beta * selected_score,
            "rf_reward_VQ_raw": sum(all_vq) / len(all_vq),
            "rf_reward_MQ_raw": sum(all_mq) / len(all_mq),
            "rf_reward_TA_raw": sum(all_ta) / len(all_ta),
            "rf_reward_VQ": avg_vq,
            "rf_reward_MQ": avg_mq,
            "rf_reward_TA": avg_ta,
            "rf_reward_overall": avg_overall,
            "rf_score_type": score_type,
            "rf_selected_score": selected_score,
            "rl_reward_ema_mean": self.rl_reward_ema_mean,
            "rl_reward_ema_std": self.rl_reward_ema_std,
            "rl_enabled": True
        }

        return reward_weight, rf_log_dict

    def compute_rl_loss(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute RL loss from generated latents (differentiable RL mode).

        Pipeline:
        1. Decode all latent frames to pixel space with checkpoint
        2. Uniformly sub-sample pixel frames to rl_reward_num_frames
        3. Compute reward, return negative reward as loss

        Gradients flow: reward → pixel frames → VAE → latent → generator

        Args:
            latent: Generated latent tensor [B, T, C, H, W]
            text_prompts: List of text prompts

        Returns:
            rl_loss: Scalar tensor representing the RL loss (negative reward)
            rl_log_dict: Dictionary containing logging information
        """
        if not self.rl_enabled or self._reward_model is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "rl_loss": 0.0,
                "rl_reward_raw": 0.0,
                "rl_reward_normalized": 0.0,
                "rl_enabled": False
            }

        batch_size = latent.shape[0]

        # Step 1: Decode all latent frames to pixel space with checkpoint
        pixel_video = self._decode_latent(latent)

        # Compute rewards for each sample in the batch
        total_reward = 0.0
        raw_rewards_list = []
        normalized_rewards_list = []

        for i in range(batch_size):
            single_video = pixel_video[i]  # [T_decoded, 3, H, W]

            # Step 2: Sub-sample pixel frames for reward model
            sampled_video = self._sample_frames(single_video)

            prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts

            raw_logits = self._reward_model.compute_reward_from_vae_output(
                vae_output=sampled_video,
                prompt=prompt,
                target_height=self.rl_target_height,
                target_width=self.rl_target_width
            )

            # Apply z-score normalization from inference_config (differentiable)
            normalized_logits = self._normalize_logits(raw_logits)

            if self.rl_reward_type == "VQ":
                reward = normalized_logits[0, 0]
                raw_reward = raw_logits[0, 0].detach().item()
            elif self.rl_reward_type == "MQ":
                reward = normalized_logits[0, 1]
                raw_reward = raw_logits[0, 1].detach().item()
            elif self.rl_reward_type == "TA":
                reward = normalized_logits[0, 2]
                raw_reward = raw_logits[0, 2].detach().item()
            else:  # overall
                reward = normalized_logits.sum()
                raw_reward = raw_logits.sum().detach().item()

            reward_value = reward.detach().item()
            raw_rewards_list.append(raw_reward)
            normalized_rewards_list.append(reward_value)

            # Update EMA for monitoring
            self._update_reward_stats(reward_value)

            total_reward = total_reward + reward

        avg_reward = total_reward / batch_size
        rl_loss = -avg_reward

        rl_log_dict = {
            "rl_loss": rl_loss.detach().item(),
            "rl_reward_raw": sum(raw_rewards_list) / len(raw_rewards_list),
            "rl_reward_normalized": sum(normalized_rewards_list) / len(normalized_rewards_list),
            "rl_reward_ema_mean": self.rl_reward_ema_mean,
            "rl_reward_ema_std": self.rl_reward_ema_std,
            "rl_reward_num_frames": self.rl_reward_num_frames,
            "rl_enabled": True
        }

        return rl_loss, rl_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        text_prompts: list = None,
        current_step: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        # Check if RL should be enabled
        self.enable_rl(current_step)

        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        # Step 3: Apply reward - two modes
        if self.reward_forcing and self.rl_enabled and text_prompts is not None:
            # === Reward Forcing mode ===
            # Reward is a NO-GRAD scalar weight on DMD loss
            reward_weight, rf_log_dict = self.compute_reward_forcing_weight(
                latent=pred_image,
                text_prompts=text_prompts
            )
            total_loss = reward_weight * dmd_loss

            generator_log_dict = dmd_log_dict.copy()
            generator_log_dict.update(rf_log_dict)
            generator_log_dict["dmd_loss"] = dmd_loss.detach().item()
            generator_log_dict["total_generator_loss"] = total_loss.detach().item()
            generator_log_dict["reward_forcing"] = True

        elif not self.reward_forcing and self.rl_enabled and text_prompts is not None:
            # === Differentiable RL mode (original) ===
            rl_loss, rl_log_dict = self.compute_rl_loss(
                latent=pred_image,
                text_prompts=text_prompts
            )
            total_loss = dmd_loss + self.rl_loss_weight * rl_loss

            generator_log_dict = dmd_log_dict.copy()
            generator_log_dict.update(rl_log_dict)
            generator_log_dict["dmd_loss"] = dmd_loss.detach().item()
            generator_log_dict["total_generator_loss"] = total_loss.detach().item()
            generator_log_dict["rl_loss_weight"] = self.rl_loss_weight
            generator_log_dict["reward_forcing"] = False

        else:
            # === RL not yet enabled (cold start) ===
            total_loss = dmd_loss
            generator_log_dict = dmd_log_dict.copy()
            generator_log_dict["dmd_loss"] = dmd_loss.detach().item()
            generator_log_dict["total_generator_loss"] = total_loss.detach().item()
            generator_log_dict["rl_enabled"] = False
            generator_log_dict["reward_forcing"] = self.reward_forcing

        return total_loss, generator_log_dict