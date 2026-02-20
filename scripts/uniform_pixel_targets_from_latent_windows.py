import argparse
import os

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


def decode_window_u8(vae: WanVAEWrapper, latent: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Decode latent[:, start:end] and return [T,H,W,C] uint8."""
    latent_slice = latent[:, start:end]
    pix = vae.decode_to_pixel(latent_slice)
    return to_uint8_video(pix)


def best_align_single_frame(full_u8: torch.Tensor, frame_u8: torch.Tensor):
    """Find best-matching full frame index for one candidate frame by MSE."""
    best_idx, best_mse = None, None
    for idx in range(full_u8.shape[0]):
        mse = torch.mean((full_u8[idx].float() - frame_u8.float()) ** 2).item()
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_idx = idx
    return best_idx, best_mse


def main():
    parser = argparse.ArgumentParser("Uniform pixel targets -> latent window decode demo")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/zdmaogroup/wubin/RL1-claude-optimize-memory-frame-sampling-FrPp8/videos/lora_videos/0-0_lora.mp4",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_target_frames", type=int, default=10)
    parser.add_argument("--window_back", type=int, default=4, help="Use latent [i-window_back, i]")
    parser.add_argument("--out_dir", type=str, default="artifacts/uniform_pixel_targets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    video, _, info = read_video(args.video_path, pts_unit="sec")
    fps = float(info.get("video_fps", 16))

    device = torch.device(args.device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        x = to_model_input(video).to(device=device, dtype=torch.bfloat16)
        latent = vae.encode_to_latent(x)
        latent = latent.to(next(vae.model.parameters()).dtype)
        full_u8 = to_uint8_video(vae.decode_to_pixel(latent))  # [81,H,W,C]

    # 1) 按均匀 pixel 帧定目标
    target_pixel_idx = torch.linspace(0, full_u8.shape[0] - 1, args.num_target_frames).round().long().tolist()

    selected_full = []
    selected_window = []
    selected_diff = []

    print("target_pixel | latent_i(rough) | window[start,end) | aligned_full | mse")
    print("-" * 78)

    for t in target_pixel_idx:
        # 2) 反推 latent 索引（粗略：pixel约4倍于latent）
        latent_i = int(round(t / 4.0))
        latent_i = max(0, min(latent_i, latent.shape[1] - 1))

        w_start = max(0, latent_i - args.window_back)
        w_end = latent_i + 1  # inclusive i

        part_u8 = decode_window_u8(vae, latent, w_start, w_end)
        candidate = part_u8[-1]  # 取窗口尾帧当目标帧

        # 3) 在 full 里自动对齐单帧
        best_idx, best_mse = best_align_single_frame(full_u8, candidate)

        full_frame = full_u8[t]
        aligned_frame = full_u8[best_idx]
        diff = (aligned_frame.float() - candidate.float()).abs().clamp(0, 255).round().to(torch.uint8)

        selected_full.append(full_frame)
        selected_window.append(candidate)
        selected_diff.append(diff)

        print(f"{t:11d} | {latent_i:14d} | [{w_start:2d},{w_end:2d})      | {best_idx:12d} | {best_mse:8.3f}")

    selected_full = torch.stack(selected_full, dim=0)
    selected_window = torch.stack(selected_window, dim=0)
    selected_diff = torch.stack(selected_diff, dim=0)

    write_video(os.path.join(args.out_dir, "target_full_uniform10.mp4"), selected_full, fps=fps)
    write_video(os.path.join(args.out_dir, "target_from_window_uniform10.mp4"), selected_window, fps=fps)
    write_video(os.path.join(args.out_dir, "target_abs_diff_uniform10.mp4"), selected_diff, fps=fps)

    print("\nSaved:")
    print(f"- {os.path.join(args.out_dir, 'target_full_uniform10.mp4')}")
    print(f"- {os.path.join(args.out_dir, 'target_from_window_uniform10.mp4')}")
    print(f"- {os.path.join(args.out_dir, 'target_abs_diff_uniform10.mp4')}")


if __name__ == "__main__":
    main()
