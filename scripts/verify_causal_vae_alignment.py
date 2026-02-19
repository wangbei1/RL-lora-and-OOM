import argparse
import os
from typing import List

import torch
from torchvision.io import read_video, write_video

from utils.wan_wrapper import WanVAEWrapper


def to_model_input(video_thwc: torch.Tensor) -> torch.Tensor:
    video = video_thwc.float() / 255.0
    video = video * 2.0 - 1.0
    return video.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [1,C,T,H,W]


def to_uint8_video(video_btchw: torch.Tensor) -> torch.Tensor:
    video = video_btchw[0].detach().cpu().clamp(-1, 1)
    video = ((video + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    return video.permute(0, 2, 3, 1).contiguous()  # [T,H,W,C]


def psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return (20.0 * torch.log10(torch.tensor(255.0)) - 10.0 * torch.log10(torch.tensor(mse))).item()


def frame_metrics(a: torch.Tensor, b: torch.Tensor) -> List[dict]:
    # a/b: [T,H,W,C] uint8
    n = min(a.shape[0], b.shape[0])
    out = []
    for i in range(n):
        af = a[i].float()
        bf = b[i].float()
        mse = torch.mean((af - bf) ** 2).item()
        mae = torch.mean(torch.abs(af - bf)).item()
        out.append({"idx": i, "mse": mse, "mae": mae, "psnr": psnr_from_mse(mse)})
    return out


def main():
    parser = argparse.ArgumentParser("Verify causal VAE latent-window decoding alignment")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/zdmaogroup/wubin/RL1-claude-optimize-memory-frame-sampling-FrPp8/videos/lora_videos/0-0_lora.mp4",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_start", type=int, default=2, help="inclusive latent index")
    parser.add_argument("--latent_end", type=int, default=5, help="exclusive latent index")
    parser.add_argument("--full_start", type=int, default=10, help="full decode start frame index")
    parser.add_argument("--compare_len", type=int, default=4, help="number of frames to compare")
    parser.add_argument(
        "--auto_scan",
        action="store_true",
        help="Automatically scan full-decode offsets and pick best alignment for part tail clip",
    )
    parser.add_argument(
        "--scan_min_start",
        type=int,
        default=0,
        help="Minimum full-decode start index used when --auto_scan is enabled",
    )
    parser.add_argument(
        "--scan_max_start",
        type=int,
        default=-1,
        help="Maximum full-decode start index (inclusive) used when --auto_scan is enabled; -1 means auto",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/causal_vae_verify")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(args.video_path)
    os.makedirs(args.out_dir, exist_ok=True)

    video, _, info = read_video(args.video_path, pts_unit="sec")
    fps = float(info.get("video_fps", 16))

    device = torch.device(args.device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        model_input = to_model_input(video).to(device=device, dtype=torch.bfloat16)
        latent = vae.encode_to_latent(model_input)
        decode_dtype = next(vae.model.parameters()).dtype
        latent = latent.to(decode_dtype)

        full_pix = vae.decode_to_pixel(latent)  # [1,T,C,H,W]

        latent_slice = latent[:, args.latent_start:args.latent_end]
        part_pix = vae.decode_to_pixel(latent_slice)  # [1,T',C,H,W]

    full_u8 = to_uint8_video(full_pix)
    part_u8 = to_uint8_video(part_pix)

    part_clip = part_u8[-args.compare_len:]

    if args.auto_scan:
        max_valid_start = full_u8.shape[0] - args.compare_len
        scan_min = max(0, args.scan_min_start)
        scan_max = max_valid_start if args.scan_max_start < 0 else min(args.scan_max_start, max_valid_start)
        if scan_min > scan_max:
            raise ValueError(f"Invalid scan range: [{scan_min}, {scan_max}] for full length {full_u8.shape[0]}")

        best = None
        for start in range(scan_min, scan_max + 1):
            cand = full_u8[start:start + args.compare_len]
            cand_mse = torch.mean((cand.float() - part_clip.float()) ** 2).item()
            cand_mae = torch.mean(torch.abs(cand.float() - part_clip.float())).item()
            if best is None or cand_mse < best["mse"]:
                best = {"start": start, "mse": cand_mse, "mae": cand_mae}

        full_start = best["start"]
    else:
        full_start = args.full_start

    full_clip = full_u8[full_start:full_start + args.compare_len]

    metrics = frame_metrics(full_clip, part_clip)
    mse = torch.mean((full_clip.float() - part_clip.float()) ** 2).item()
    mae = torch.mean(torch.abs(full_clip.float() - part_clip.float())).item()
    psnr = psnr_from_mse(mse)

    write_video(os.path.join(args.out_dir, "full_decode.mp4"), full_u8, fps=fps)
    write_video(os.path.join(args.out_dir, "part_decode.mp4"), part_u8, fps=fps)
    write_video(os.path.join(args.out_dir, "full_clip_for_compare.mp4"), full_clip, fps=fps)
    write_video(os.path.join(args.out_dir, "part_clip_tail_for_compare.mp4"), part_clip, fps=fps)

    print("=== Causal VAE alignment verification ===")
    print(f"video_path     : {args.video_path}")
    print(f"latent range   : [{args.latent_start}, {args.latent_end})")
    print(f"auto_scan      : {args.auto_scan}")
    if args.auto_scan:
        max_valid_start = full_u8.shape[0] - args.compare_len
        scan_min = max(0, args.scan_min_start)
        scan_max = max_valid_start if args.scan_max_start < 0 else min(args.scan_max_start, max_valid_start)
        print(f"scan range     : [{scan_min}, {scan_max}]")
    print(f"full frame idx : [{full_start}, {full_start + args.compare_len})")
    print(f"compare        : part_tail(last {args.compare_len}) vs full segment")
    print(f"full_decode frames: {full_u8.shape[0]}")
    print(f"part_decode frames: {part_u8.shape[0]}")
    print("--- aggregate ---")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"PSNR : {psnr:.4f} dB")
    print("--- per-frame ---")
    for m in metrics:
        print(f"i={m['idx']}: mse={m['mse']:.6f}, mae={m['mae']:.6f}, psnr={m['psnr']:.4f} dB")


if __name__ == "__main__":
    main()
