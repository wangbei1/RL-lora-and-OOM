import argparse
import os

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from model.base import apply_lora_to_model
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file (base model or LoRA training checkpoint)")
parser.add_argument("--lora_checkpoint_path", type=str, default=None,
                    help="Optional checkpoint path only used for loading LoRA-tuned generator weights")
parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of generated latent frames")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters when loading checkpoint")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save video using index or prompt as filename")

# LoRA-related flags (will override config when provided)
parser.add_argument("--use_lora", action="store_true", help="Enable LoRA wrapping before loading checkpoint")
parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
parser.add_argument("--lora_target_modules", nargs="+", default=None,
                    help="Optional LoRA target module names")
args = parser.parse_args()


def _extract_generator_state_dict(checkpoint_path: str, use_ema: bool = False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state_dict, dict):
        key = "generator_ema" if use_ema else "generator"
        if key in state_dict:
            state_dict = state_dict[key]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

    cleaned = {}
    for k, v in state_dict.items():
        name = k.replace("_fsdp_wrapped_module.", "")
        cleaned[name] = v
    return cleaned


def _apply_lora_if_needed(pipeline_generator, config):
    use_lora = bool(getattr(config, "use_lora", False))
    if not use_lora:
        return

    print("[inference_lora] Applying LoRA wrapper to generator model...")
    pipeline_generator.model.requires_grad_(False)
    pipeline_generator.model = apply_lora_to_model(
        pipeline_generator.model,
        lora_rank=getattr(config, "lora_rank", 16),
        lora_alpha=getattr(config, "lora_alpha", 32),
        lora_dropout=getattr(config, "lora_dropout", 0.0),
        target_modules=getattr(config, "lora_target_modules", None),
    )


def _load_generator_weights(pipeline_generator, checkpoint_path: str, use_ema: bool):
    raw_state_dict = _extract_generator_state_dict(checkpoint_path, use_ema=use_ema)

    has_lora = hasattr(pipeline_generator.model, "base_model")
    ckpt_has_lora = any("base_model.model." in k for k in raw_state_dict.keys())

    if has_lora and not ckpt_has_lora:
        print("[inference_lora] LoRA model + non-LoRA checkpoint detected, loading base weights into base model")
        base_sd = {}
        prefix = "model."
        for k, v in raw_state_dict.items():
            base_sd[k[len(prefix):] if k.startswith(prefix) else k] = v
        pipeline_generator.model.base_model.model.load_state_dict(base_sd, strict=True)
    elif not has_lora and ckpt_has_lora:
        print("[inference_lora] Non-LoRA model + LoRA checkpoint detected, dropping LoRA weights")
        stripped_sd = {}
        for k, v in raw_state_dict.items():
            new_k = k.replace("base_model.model.", "")
            if "lora_" not in new_k:
                stripped_sd[new_k] = v
        pipeline_generator.load_state_dict(stripped_sd, strict=True)
    else:
        pipeline_generator.load_state_dict(raw_state_dict, strict=True)


# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    set_seed(args.seed)

print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# CLI override for LoRA options
if args.use_lora:
    config.use_lora = True
if args.lora_rank is not None:
    config.lora_rank = args.lora_rank
if args.lora_alpha is not None:
    config.lora_alpha = args.lora_alpha
if args.lora_dropout is not None:
    config.lora_dropout = args.lora_dropout
if args.lora_target_modules is not None:
    config.lora_target_modules = args.lora_target_modules

# Initialize pipeline
if hasattr(config, "denoising_step_list"):
    pipeline = CausalInferencePipeline(config, device=device)
else:
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

_apply_lora_if_needed(pipeline.generator, config)

checkpoint_to_load = args.lora_checkpoint_path or args.checkpoint_path
if checkpoint_to_load:
    _load_generator_weights(pipeline.generator, checkpoint_to_load, use_ema=args.use_ema)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for _, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

    if args.i2v:
        prompt = batch["prompts"][0]
        prompts = [prompt] * args.num_samples
        image = batch["image"].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        prompt = batch["prompts"][0]
        extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
        prompts = [extended_prompt if extended_prompt is not None else prompt] * args.num_samples
        initial_latent = None
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    video, _ = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )

    video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
    pipeline.vae.model.clear_cache()

    if idx < num_prompts:
        model_name = "lora" if bool(getattr(config, "use_lora", False)) else "regular"
        model_name = "ema_" + model_name if args.use_ema else model_name
        for seed_idx in range(args.num_samples):
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f"{idx}-{seed_idx}_{model_name}.mp4")
            else:
                output_path = os.path.join(args.output_folder, f"{prompt[:100]}-{seed_idx}_{model_name}.mp4")
            write_video(output_path, video[seed_idx], fps=16)
