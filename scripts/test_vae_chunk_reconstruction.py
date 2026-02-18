import argparse
import os
from typing import Tuple

import torch
import torch.utils.checkpoint as checkpoint_utils
from torchvision.io import read_video, write_video

from utils.wan_wrapper import WanVAEWrapper


def to_model_input(video_thwc: torch.Tensor) -> torch.Tensor:
    """Convert [T,H,W,C] uint8 to [B,C,T,H,W] float in [-1,1]."""
    video = video_thwc.float() / 255.0
    video = video * 2.0 - 1.0
    video = video.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return video


def to_uint8_video(video_btchw: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,C,H,W] in [-1,1] to [T,H,W,C] uint8 (first sample)."""
    video = video_btchw[0].detach().cpu().clamp(-1, 1)
    video = (video + 1.0) / 2.0
    video = (video * 255.0).round().to(torch.uint8)
    return video.permute(0, 2, 3, 1).contiguous()


def decode_latent_chunked(vae: WanVAEWrapper, latent: torch.Tensor, chunk_size: int, decode_dtype: torch.dtype) -> torch.Tensor:
    """Mirror DMDRL chunked decode behavior."""

    def _decode(lat):
        return vae.decode_to_pixel(lat.to(dtype=decode_dtype))

    if chunk_size <= 0 or chunk_size >= latent.shape[1]:
        return checkpoint_utils.checkpoint(_decode, latent, use_reentrant=False)

    chunks = []
    for start in range(0, latent.shape[1], chunk_size):
        end = min(start + chunk_size, latent.shape[1])
        lat_chunk = latent[:, start:end]
        pix_chunk = checkpoint_utils.checkpoint(_decode, lat_chunk, use_reentrant=False)
        chunks.append(pix_chunk)
    return torch.cat(chunks, dim=1)


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """Return MSE, MAE, PSNR on uint8 videos [T,H,W,C]."""
    n = min(a.shape[0], b.shape[0])
    a = a[:n].float()
    b = b[:n].float()
    mse = torch.mean((a - b) ** 2).item()
    mae = torch.mean(torch.abs(a - b)).item()
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * torch.log10(torch.tensor(255.0)) - 10.0 * torch.log10(torch.tensor(mse))
        psnr = psnr.item()
    return mse, mae, psnr


def main():
    parser = argparse.ArgumentParser(description="Test VAE chunked reconstruction quality")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/zdmaogroup/wubin/RL1-claude-optimize-memory-frame-sampling-FrPp8/videos/lora_videos/0-0_lora.mp4",
    )
    parser.add_argument("--chunk_size", type=int, default=1, help="Temporal latent chunk size for decode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        default="bf16",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for VAE encode/decode on CUDA",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/vae_chunk_test")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means use all frames")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Input video not found: {args.video_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    video, _, info = read_video(args.video_path, pts_unit="sec")
    fps = info.get("video_fps", 16)

    if args.max_frames > 0:
        video = video[: args.max_frames]

    device = torch.device(args.device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    autocast_enabled = (device.type == "cuda") and (args.autocast_dtype != "none")
    if args.autocast_dtype == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.autocast_dtype == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            model_input = to_model_input(video).to(device=device, dtype=torch.bfloat16)
            latent = vae.encode_to_latent(model_input)

            # WanVAEWrapper.encode_to_latent returns float tensor; ensure decode input dtype
            # matches VAE module dtype to avoid conv3d dtype mismatch (float vs bf16).
            decode_dtype = next(vae.model.parameters()).dtype
            latent = latent.to(dtype=decode_dtype)

            recon_full = vae.decode_to_pixel(latent)
            recon_chunk = decode_latent_chunked(vae, latent, args.chunk_size, decode_dtype=decode_dtype)

    recon_full_u8 = to_uint8_video(recon_full)
    recon_chunk_u8 = to_uint8_video(recon_chunk)

    out_full = os.path.join(args.out_dir, "recon_full.mp4")
    out_chunk = os.path.join(args.out_dir, f"recon_chunk{args.chunk_size}.mp4")
    write_video(out_full, recon_full_u8, fps=float(fps))
    write_video(out_chunk, recon_chunk_u8, fps=float(fps))

    mse_orig_chunk, mae_orig_chunk, psnr_orig_chunk = compute_metrics(video, recon_chunk_u8)
    mse_full_chunk, mae_full_chunk, psnr_full_chunk = compute_metrics(recon_full_u8, recon_chunk_u8)

    print("=== Reconstruction Metrics ===")
    print(f"Input video: {args.video_path}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Autocast: enabled={autocast_enabled}, dtype={args.autocast_dtype}")
    print(f"Saved full recon : {out_full}")
    print(f"Saved chunk recon: {out_chunk}")
    print("--- Original vs Chunked-Recon ---")
    print(f"MSE : {mse_orig_chunk:.6f}")
    print(f"MAE : {mae_orig_chunk:.6f}")
    print(f"PSNR: {psnr_orig_chunk:.4f} dB")
    print("--- Full-Recon vs Chunked-Recon ---")
    print(f"MSE : {mse_full_chunk:.6f}")
    print(f"MAE : {mae_full_chunk:.6f}")
    print(f"PSNR: {psnr_full_chunk:.4f} dB")


if __name__ == "__main__":
    main()
