import gc
import logging
from datetime import datetime

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, DMDRL
from model.base import load_generator_checkpoint
import torch
import wandb
import time
import os
import numpy as np


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        # Initialize local logging directory and log file
        self._init_local_logging(config)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_rl":
            self.model = DMDRL(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # Load generator checkpoint BEFORE LoRA/FSDP (original key names required)
        if getattr(config, "generator_ckpt", False):
            load_generator_checkpoint(self.model.generator, config.generator_ckpt)

        # Apply LoRA AFTER checkpoint loading but BEFORE FSDP wrapping
        # (PEFT changes state_dict keys; FSDP must shard LoRA params too)
        self.model._apply_lora_if_enabled(config)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        # VAE is needed for: visualization, raw video loading, dmd_rl (RL reward computation),
        # or checkpoint video generation
        self.checkpoint_num_videos = getattr(config, "checkpoint_num_videos", 0)
        if not config.no_visualize or config.load_raw_video or config.distribution_loss == "dmd_rl" or self.checkpoint_num_videos > 0:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. Generator checkpoint was already loaded before LoRA/FSDP wrapping (see above).
        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

    def _init_local_logging(self, config):
        """
        Initialize local logging directory and log file.
        Creates: output/{timestamp}_{experiment_name}/log.txt
        """
        # Initialize training history for plotting
        self.training_history = {
            "steps": [],
            "generator_loss": [],
            "critic_loss": [],
            "dmd_loss": [],
            "rl_loss": [],
            "rl_reward_raw": [],
            "rl_reward_normalized": [],
            # Reward Forcing specific
            "rf_weight": [],
            "rf_reward_VQ": [],
            "rf_reward_MQ": [],
            "rf_reward_TA": [],
            "rf_reward_overall": [],
        }

        if not self.is_main_process:
            self.exp_dir = None
            self.log_file = None
            return

        # Create experiment directory name: timestamp + experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use exp_name config if provided, otherwise fall back to config_name
        exp_name = getattr(config, "exp_name", None) or getattr(config, "config_name", "experiment")
        exp_dir_name = f"{timestamp}_{exp_name}"

        # Create output directory
        output_base = getattr(config, "output_dir", "output")
        self.exp_dir = os.path.join(output_base, exp_dir_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Create log file
        self.log_file = os.path.join(self.exp_dir, "log.txt")

        # Write header to log file
        with open(self.log_file, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Distribution loss: {config.distribution_loss}\n")
            f.write(f"Max training steps: {getattr(config, 'max_training_steps', 'unlimited')}\n")
            f.write("=" * 80 + "\n")
            # Write column headers
            if config.distribution_loss == "dmd_rl" and getattr(config, "reward_forcing", False):
                f.write(f"{'step':>8} | {'dmd_loss':>10} | {'rf_weight':>10} | {'VQ':>6} {'MQ':>6} {'TA':>6} {'OA':>6} | {'total_loss':>12}\n")
            elif config.distribution_loss == "dmd_rl":
                f.write(f"{'step':>8} | {'dmd_loss':>10} | {'rl_loss':>10} | {'reward_raw':>12} | {'reward_norm':>12} | {'total_loss':>12} | {'rl_enabled':>10}\n")
            else:
                f.write(f"{'step':>8} | {'generator_loss':>14} | {'critic_loss':>12}\n")
            f.write("-" * 80 + "\n")

        print(f"[Logging] Experiment directory: {self.exp_dir}")
        print(f"[Logging] Log file: {self.log_file}")

    def _log_to_file(self, step, generator_log_dict, critic_log_dict):
        """
        Write training metrics to local log file and store in history for plotting.
        """
        # Store in history for plotting
        self.training_history["steps"].append(step)

        is_reward_forcing = generator_log_dict.get("reward_forcing", False)

        if self.config.distribution_loss == "dmd_rl" and is_reward_forcing:
            # Reward Forcing mode logging
            dmd_loss = generator_log_dict.get("dmd_loss", 0.0)
            total_loss = generator_log_dict.get("total_generator_loss", 0.0)
            rf_weight = generator_log_dict.get("rf_weight", 1.0)
            rf_vq = generator_log_dict.get("rf_reward_VQ", 0.0)
            rf_mq = generator_log_dict.get("rf_reward_MQ", 0.0)
            rf_ta = generator_log_dict.get("rf_reward_TA", 0.0)
            rf_overall = generator_log_dict.get("rf_reward_overall", 0.0)

            self.training_history["dmd_loss"].append(dmd_loss)
            self.training_history["generator_loss"].append(total_loss)
            self.training_history["rf_weight"].append(rf_weight)
            self.training_history["rf_reward_VQ"].append(rf_vq)
            self.training_history["rf_reward_MQ"].append(rf_mq)
            self.training_history["rf_reward_TA"].append(rf_ta)
            self.training_history["rf_reward_overall"].append(rf_overall)

        elif self.config.distribution_loss == "dmd_rl":
            # Differentiable RL mode logging
            dmd_loss = generator_log_dict.get("dmd_loss", 0.0)
            rl_loss = generator_log_dict.get("rl_loss", 0.0)
            reward_raw = generator_log_dict.get("rl_reward_raw", 0.0)
            reward_norm = generator_log_dict.get("rl_reward_normalized", 0.0)
            total_loss = generator_log_dict.get("total_generator_loss", 0.0)
            rl_enabled = generator_log_dict.get("rl_enabled", False)

            self.training_history["dmd_loss"].append(dmd_loss)
            self.training_history["rl_loss"].append(rl_loss)
            self.training_history["rl_reward_raw"].append(reward_raw)
            self.training_history["rl_reward_normalized"].append(reward_norm)
            self.training_history["generator_loss"].append(total_loss)
        else:
            gen_loss = generator_log_dict.get("generator_loss", torch.tensor(0.0))
            if isinstance(gen_loss, torch.Tensor):
                gen_loss = gen_loss.mean().item()
            self.training_history["generator_loss"].append(gen_loss)

        critic_loss = critic_log_dict.get("critic_loss", torch.tensor(0.0))
        if isinstance(critic_loss, torch.Tensor):
            critic_loss = critic_loss.mean().item()
        self.training_history["critic_loss"].append(critic_loss)

        if not self.is_main_process or self.log_file is None:
            return

        with open(self.log_file, "a") as f:
            if self.config.distribution_loss == "dmd_rl" and is_reward_forcing:
                f.write(f"{step:>8} | dmd={dmd_loss:.4f} | w={rf_weight:.4f} | "
                        f"VQ={rf_vq:.3f} MQ={rf_mq:.3f} TA={rf_ta:.3f} "
                        f"OA={rf_overall:.3f} | total={total_loss:.4f}\n")
            elif self.config.distribution_loss == "dmd_rl":
                dmd_loss = generator_log_dict.get("dmd_loss", 0.0)
                rl_loss = generator_log_dict.get("rl_loss", 0.0)
                reward_raw = generator_log_dict.get("rl_reward_raw", 0.0)
                reward_norm = generator_log_dict.get("rl_reward_normalized", 0.0)
                total_loss = generator_log_dict.get("total_generator_loss", 0.0)
                rl_enabled = generator_log_dict.get("rl_enabled", False)

                f.write(f"{step:>8} | {dmd_loss:>10.4f} | {rl_loss:>10.4f} | {reward_raw:>12.4f} | {reward_norm:>12.4f} | {total_loss:>12.4f} | {str(rl_enabled):>10}\n")
            else:
                gen_loss = generator_log_dict.get("generator_loss", torch.tensor(0.0))
                if isinstance(gen_loss, torch.Tensor):
                    gen_loss = gen_loss.mean().item()
                critic_loss_val = critic_log_dict.get("critic_loss", torch.tensor(0.0))
                if isinstance(critic_loss_val, torch.Tensor):
                    critic_loss_val = critic_loss_val.mean().item()

                f.write(f"{step:>8} | {gen_loss:>14.4f} | {critic_loss_val:>12.4f}\n")

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            # Save to experiment directory (exp_dir) instead of output_path
            save_dir = self.exp_dir if self.exp_dir else self.output_path
            checkpoint_dir = os.path.join(save_dir, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, save_path)
            print(f"Model saved to {save_path}")

    def _plot_training_curves(self):
        """
        Plot training curves and save to experiment directory.
        """
        if not self.is_main_process or self.exp_dir is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] matplotlib not available, skipping curve plotting")
            return

        steps = self.training_history["steps"]
        if len(steps) == 0:
            return

        # Check if this is reward_forcing mode
        is_rf = getattr(self.config, "reward_forcing", False)

        # Create figure with subplots
        if self.config.distribution_loss == "dmd_rl" and is_rf:
            # Reward Forcing plots: 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Plot DMD Loss
            if self.training_history["dmd_loss"]:
                axes[0, 0].plot(steps, self.training_history["dmd_loss"], label='DMD Loss')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('DMD Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

            # Plot Reward Weight
            if self.training_history["rf_weight"]:
                axes[0, 1].plot(steps, self.training_history["rf_weight"], label='exp(beta*r)', color='orange')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Weight')
                axes[0, 1].set_title('Reward Forcing Weight')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

            # Plot Total Generator Loss (weighted)
            if self.training_history["generator_loss"]:
                axes[0, 2].plot(steps, self.training_history["generator_loss"], label='Total Loss', color='green')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].set_title('Total Generator Loss (w*dmd)')
                axes[0, 2].legend()
                axes[0, 2].grid(True)

            # Plot individual reward scores
            if self.training_history["rf_reward_VQ"]:
                axes[1, 0].plot(steps, self.training_history["rf_reward_VQ"], label='VQ', color='blue')
                axes[1, 0].plot(steps, self.training_history["rf_reward_MQ"], label='MQ', color='red')
                axes[1, 0].plot(steps, self.training_history["rf_reward_TA"], label='TA', color='green')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Score (normalized)')
                axes[1, 0].set_title('Reward Scores (VQ/MQ/TA)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Plot overall reward
            if self.training_history["rf_reward_overall"]:
                axes[1, 1].plot(steps, self.training_history["rf_reward_overall"], label='Overall', color='purple')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Score (normalized)')
                axes[1, 1].set_title('Overall Reward')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            # Plot Critic Loss
            if self.training_history["critic_loss"]:
                axes[1, 2].plot(steps, self.training_history["critic_loss"], label='Critic Loss', color='brown')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].set_title('Critic Loss')
                axes[1, 2].legend()
                axes[1, 2].grid(True)

        elif self.config.distribution_loss == "dmd_rl":
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Plot DMD Loss
            if self.training_history["dmd_loss"]:
                axes[0, 0].plot(steps, self.training_history["dmd_loss"], label='DMD Loss')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('DMD Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

            # Plot RL Loss
            if self.training_history["rl_loss"]:
                axes[0, 1].plot(steps, self.training_history["rl_loss"], label='RL Loss', color='orange')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('RL Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)

            # Plot Total Generator Loss
            if self.training_history["generator_loss"]:
                axes[0, 2].plot(steps, self.training_history["generator_loss"], label='Total Loss', color='green')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].set_title('Total Generator Loss')
                axes[0, 2].legend()
                axes[0, 2].grid(True)

            # Plot Raw Reward
            if self.training_history["rl_reward_raw"]:
                axes[1, 0].plot(steps, self.training_history["rl_reward_raw"], label='Raw Reward', color='purple')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Reward')
                axes[1, 0].set_title('Raw Reward')
                axes[1, 0].legend()
                axes[1, 0].grid(True)

            # Plot Normalized Reward
            if self.training_history["rl_reward_normalized"]:
                axes[1, 1].plot(steps, self.training_history["rl_reward_normalized"], label='Normalized Reward', color='red')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].set_title('Normalized Reward')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            # Plot Critic Loss
            if self.training_history["critic_loss"]:
                axes[1, 2].plot(steps, self.training_history["critic_loss"], label='Critic Loss', color='brown')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].set_title('Critic Loss')
                axes[1, 2].legend()
                axes[1, 2].grid(True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot Generator Loss
            if self.training_history["generator_loss"]:
                axes[0].plot(steps, self.training_history["generator_loss"], label='Generator Loss')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Generator Loss')
                axes[0].legend()
                axes[0].grid(True)

            # Plot Critic Loss
            if self.training_history["critic_loss"]:
                axes[1].plot(steps, self.training_history["critic_loss"], label='Critic Loss', color='orange')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Critic Loss')
                axes[1].legend()
                axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Plotting] Training curves saved to {plot_path}")

    def _generate_checkpoint_videos(self, num_videos=5):
        """
        Generate a small number of videos at checkpoint save time for quick quality checks.

        ALL distributed processes must participate (inference uses dist.broadcast).
        Only rank-0 writes files to disk.

        Returns:
            checkpoint_video_dir path (rank-0) or None (other ranks)
        """
        prompts_file = "prompts/MovieGenVideoBench.txt"
        if not os.path.exists(prompts_file):
            if self.is_main_process:
                print(f"[Checkpoint] Prompts file not found: {prompts_file}, skipping checkpoint videos")
            return None

        with open(prompts_file, "r") as f:
            all_prompts = [line.strip() for line in f if line.strip()]

        prompts = all_prompts[:num_videos]
        if len(prompts) == 0:
            return None

        if self.is_main_process:
            print(f"[Checkpoint] Generating {len(prompts)} checkpoint videos at step {self.step}...")

        # Create checkpoint video directory (rank-0 only)
        ckpt_video_dir = None
        if self.is_main_process:
            save_dir = self.exp_dir if self.exp_dir else self.output_path
            ckpt_video_dir = os.path.join(save_dir, f"checkpoint_model_{self.step:06d}", "videos")
            os.makedirs(ckpt_video_dir, exist_ok=True)

        try:
            import imageio
        except ImportError:
            imageio = None

        self.model.eval()
        for i, prompt in enumerate(prompts):
            try:
                video_batch = self.generate_video([prompt])  # [B, T, H, W, C] in [0,255]
                video = video_batch[0]  # [T, H, W, C]

                if self.is_main_process:
                    if imageio is not None:
                        video_path = os.path.join(ckpt_video_dir, f"video_{i:03d}.mp4")
                        imageio.mimwrite(video_path, video.astype(np.uint8),
                                         fps=8, codec='libx264')
                    else:
                        video_path = os.path.join(ckpt_video_dir, f"video_{i:03d}.npy")
                        np.save(video_path, video)

                    with open(os.path.join(ckpt_video_dir, f"video_{i:03d}_prompt.txt"), "w") as f:
                        f.write(prompt)

            except Exception as e:
                if self.is_main_process:
                    print(f"[Checkpoint] Error generating video {i}: {e}")
                continue

        if self.is_main_process:
            print(f"[Checkpoint] Videos saved to {ckpt_video_dir}")

        return ckpt_video_dir

    def _generate_demo_videos(self, num_videos=20):
        """
        Generate demo videos at the end of training.

        ALL distributed processes must call this method together because
        inference_with_trajectory uses dist.broadcast internally.
        Only rank-0 writes files to disk.

        Returns:
            demo_dir path (rank-0) or None (other ranks)
        """
        prompts_file = "prompts/MovieGenVideoBench.txt"
        if not os.path.exists(prompts_file):
            if self.is_main_process:
                print(f"[Warning] Prompts file not found: {prompts_file}, skipping demo video generation")
            return None

        with open(prompts_file, "r") as f:
            all_prompts = [line.strip() for line in f if line.strip()]

        prompts = all_prompts[:num_videos]
        if len(prompts) == 0:
            if self.is_main_process:
                print("[Warning] No prompts found, skipping demo video generation")
            return None

        if self.is_main_process:
            print(f"[Demo] Generating {len(prompts)} demo videos...")

        # Create demo directory (rank-0 only)
        demo_dir = None
        if self.is_main_process:
            demo_dir = os.path.join(self.exp_dir, "demo_videos")
            os.makedirs(demo_dir, exist_ok=True)

        try:
            import imageio
        except ImportError:
            imageio = None

        self.model.eval()
        for i, prompt in enumerate(prompts):
            if self.is_main_process:
                print(f"[Demo] Generating video {i+1}/{len(prompts)}: {prompt[:60]}...")
            try:
                # All processes participate in generation
                video_batch = self.generate_video([prompt])  # [B, T, H, W, C] in [0,255]
                video = video_batch[0]  # [T, H, W, C]

                # Only rank-0 saves to disk
                if self.is_main_process:
                    if imageio is not None:
                        video_path = os.path.join(demo_dir, f"video_{i:03d}.mp4")
                        imageio.mimwrite(video_path, video.astype(np.uint8),
                                         fps=8, codec='libx264')
                    else:
                        video_path = os.path.join(demo_dir, f"video_{i:03d}.npy")
                        np.save(video_path, video)

                    with open(os.path.join(demo_dir, f"video_{i:03d}_prompt.txt"), "w") as f:
                        f.write(prompt)

            except Exception as e:
                if self.is_main_process:
                    print(f"[Demo] Error generating video {i}: {e}")
                continue

        if self.is_main_process:
            print(f"[Demo] Demo videos saved to {demo_dir}")

        return demo_dir

    def _run_reward_evaluation(self, demo_dir, prompts):
        """
        Run VideoAlign reward evaluation (VQ / MQ / TA / Overall) on the generated
        demo videos and save results to eval_results.csv.
        Only called by rank-0.
        """
        if not self.is_main_process or demo_dir is None:
            return

        reward_checkpoint = getattr(self.config, "rl_reward_checkpoint", None)
        if reward_checkpoint is None:
            print("[Eval] No rl_reward_checkpoint specified, skipping reward evaluation")
            return

        try:
            from VideoAlign.inference import VideoVLMRewardInference
        except ImportError as e:
            print(f"[Eval] Cannot import VideoVLMRewardInference: {e}")
            return

        print(f"[Eval] Running VideoAlign reward evaluation on {len(prompts)} videos ...")
        try:
            inferencer = VideoVLMRewardInference(
                reward_checkpoint, device=self.device, dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"[Eval] Failed to load reward model: {e}")
            return

        results = []
        for i, prompt in enumerate(prompts):
            video_path = os.path.join(demo_dir, f"video_{i:03d}.mp4")
            if not os.path.exists(video_path):
                continue
            try:
                with torch.no_grad():
                    rewards = inferencer.reward([video_path], [prompt], use_norm=True)
                r = rewards[0]
                results.append({
                    "video": f"video_{i:03d}.mp4",
                    "prompt": prompt[:120],
                    "VQ": r["VQ"],
                    "MQ": r["MQ"],
                    "TA": r["TA"],
                    "Overall": r["Overall"],
                })
            except Exception as e:
                print(f"[Eval] Error evaluating video {i}: {e}")

        if results:
            try:
                import pandas as pd
                df = pd.DataFrame(results)
                eval_path = os.path.join(self.exp_dir, "eval_results.csv")
                df.to_csv(eval_path, index=False)
                print(f"[Eval] Results saved to {eval_path}")
                print(f"[Eval] Mean scores — VQ: {df['VQ'].mean():.4f}  "
                      f"MQ: {df['MQ'].mean():.4f}  "
                      f"TA: {df['TA'].mean():.4f}  "
                      f"Overall: {df['Overall'].mean():.4f}")

                # Also append summary to log file
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write("\n[VideoAlign Reward Evaluation]\n")
                        f.write(f"VQ: {df['VQ'].mean():.4f}  "
                                f"MQ: {df['MQ'].mean():.4f}  "
                                f"TA: {df['TA'].mean():.4f}  "
                                f"Overall: {df['Overall'].mean():.4f}\n")
            except Exception as e:
                print(f"[Eval] Error saving results: {e}")

    def _on_training_end(self):
        """
        Actions to perform at the end of training.
        ALL processes participate in demo video generation (distributed inference).
        Only rank-0 handles file I/O and evaluation.
        """
        # --- Step 1: Generate demo videos (all processes must participate) ---
        num_demo_videos = getattr(self.config, "num_demo_videos", 20)
        demo_dir = None
        if num_demo_videos > 0:
            if self.is_main_process:
                print(f"\n[Post-training] Generating {num_demo_videos} demo videos...")
            demo_dir = self._generate_demo_videos(num_videos=num_demo_videos)

        # --- Step 2: Rank-0 only: evaluate, plot, log ---
        if self.is_main_process:
            print("\n" + "=" * 50)
            print("Training completed!")
            print("=" * 50)

            # Plot training curves
            print("\n[Post-training] Plotting training curves...")
            self._plot_training_curves()

            # Run reward evaluation on generated videos
            if demo_dir is not None and num_demo_videos > 0:
                prompts_file = "prompts/MovieGenVideoBench.txt"
                if os.path.exists(prompts_file):
                    with open(prompts_file, "r") as pf:
                        eval_prompts = [l.strip() for l in pf if l.strip()][:num_demo_videos]
                    print("\n[Post-training] Running VideoAlign reward evaluation...")
                    self._run_reward_evaluation(demo_dir, eval_prompts)

            # Write final summary to log
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total steps: {self.step}\n")
                    f.write("=" * 80 + "\n")

            print(f"\n[Summary] Experiment directory: {self.exp_dir}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # Check if model is DMDRL (needs text_prompts and current_step for RL)
            if self.config.distribution_loss == "dmd_rl":
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None,
                    text_prompts=text_prompts,
                    current_step=self.step
                )
            else:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None
                )

            # Scale loss for gradient accumulation so the effective gradient
            # magnitude is the same regardless of accumulation_steps.
            if self.gradient_accumulation_steps > 1:
                generator_loss = generator_loss / self.gradient_accumulation_steps
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        if self.gradient_accumulation_steps > 1:
            critic_loss = critic_loss / self.gradient_accumulation_steps
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, prompts):
        """
        Generate demo videos using the training inference pipeline + VAE decode.
        Must be called by ALL distributed processes (inference_with_trajectory
        uses dist.broadcast internally).

        Returns:
            numpy array [B, T, H, W, C] in range [0, 255], only meaningful on rank-0
        """
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, self.model.num_training_frames, 16, 60, 104],
            device=self.device,
            dtype=self.dtype
        )

        # Lazily initialise the inference pipeline (same one used during training)
        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=prompts)

            # Run causal inference to get latents
            latents, _, _ = self.model.inference_pipeline.inference_with_trajectory(
                noise=sampled_noise,
                **conditional_dict,
            )

            # Decode latents -> pixels  [B, T_pixel, 3, H, W]  range [-1, 1]
            video = self.model.vae.decode_to_pixel(latents)
            # Normalise to [0, 1]
            video = (video * 0.5 + 0.5).clamp(0, 1)

        # [B, T, H, W, C] in [0, 255]
        return video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0

    def train(self):
        start_step = self.step
        max_training_steps = getattr(self.config, "max_training_steps", None)

        if self.is_main_process and max_training_steps:
            print(f"[Training] Will train for {max_training_steps} steps")

        while True:
            # Check if we've reached max training steps
            if max_training_steps and self.step >= max_training_steps:
                if self.is_main_process:
                    print(f"\n[Training] Reached max_training_steps ({max_training_steps}), stopping training...")
                # Save final checkpoint
                if not self.config.no_save:
                    self.save()
                # Run end-of-training tasks
                self._on_training_end()
                break

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for _accum in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for _accum in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model and optionally generate checkpoint videos
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                # Generate checkpoint videos (all processes participate)
                if self.checkpoint_num_videos > 0:
                    self._generate_checkpoint_videos(num_videos=self.checkpoint_num_videos)
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                    # Log RL-specific metrics if using dmd_rl
                    if self.config.distribution_loss == "dmd_rl":
                        rl_metrics = [
                            "rl_loss", "rl_reward_raw", "rl_reward_normalized",
                            "rl_reward_ema_mean", "rl_reward_ema_std", "rl_enabled",
                            "dmd_loss", "total_generator_loss",
                            # Reward Forcing metrics
                            "rf_weight", "rf_beta_x_score",
                            "rf_reward_VQ", "rf_reward_MQ", "rf_reward_TA",
                            "rf_reward_overall", "rf_reward_VQ_raw", "rf_reward_MQ_raw",
                            "rf_reward_TA_raw", "rf_selected_score",
                        ]
                        for key in rl_metrics:
                            if key in generator_log_dict:
                                value = generator_log_dict[key]
                                if isinstance(value, bool):
                                    wandb_loss_dict[key] = float(value)
                                elif isinstance(value, str):
                                    pass  # skip string values for wandb
                                else:
                                    wandb_loss_dict[key] = value

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                # Log to local file
                if TRAIN_GENERATOR:
                    self._log_to_file(self.step, generator_log_dict, critic_log_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time