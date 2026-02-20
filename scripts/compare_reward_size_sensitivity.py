import argparse
import csv
import os
from typing import List, Tuple

import torch

from VideoAlign.inference import DifferentiableVideoReward, get_video_tensor_for_reward


def parse_size_list(size_str: str) -> List[Tuple[int, int]]:
    sizes = []
    if not size_str:
        return sizes
    for item in size_str.split(','):
        item = item.strip().lower()
        if not item:
            continue
        h, w = item.split('x')
        sizes.append((int(h), int(w)))
    return sizes


def nearest_28(x: float) -> int:
    v = int(round(x / 28.0) * 28)
    return max(v, 28)


def build_auto_sizes(base_h: int, base_w: int) -> List[Tuple[int, int]]:
    ratios = [0.75, 0.875, 1.0, 1.125, 1.25]
    out = []
    for r in ratios:
        h = nearest_28(base_h * r)
        w = nearest_28(base_w * r)
        out.append((h, w))
    uniq = []
    seen = set()
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def main():
    parser = argparse.ArgumentParser("Compare reward sensitivity to target resize in VideoAlign inference")
    parser.add_argument("--reward_ckpt", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=0, help="0 means using reward model default")
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Optional explicit sizes in HxW comma-list, e.g. '336x504,392x588'",
    )
    parser.add_argument("--out_csv", type=str, default="artifacts/reward_size_sensitivity.csv")
    args = parser.parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    reward_model = DifferentiableVideoReward(
        load_from_pretrained=args.reward_ckpt,
        device=args.device,
        dtype=dtype,
    )

    # Build a differentiable video tensor from file once; then only resize target changes.
    max_pixels = None if args.max_pixels <= 0 else args.max_pixels
    video_tensor, resized_h, resized_w = get_video_tensor_for_reward(
        video_path=args.video_path,
        num_frames=args.num_frames,
        max_pixels=max_pixels,
        sample_type=reward_model.inferencer.data_config.sample_type,
        device=args.device,
    )

    base_h, base_w = reward_model.compute_target_size(
        height=resized_h,
        width=resized_w,
        num_frames=video_tensor.shape[0],
        max_pixels=max_pixels,
    )

    sizes = parse_size_list(args.sizes)
    if not sizes:
        sizes = build_auto_sizes(base_h, base_w)
        if (base_h, base_w) not in sizes:
            sizes.append((base_h, base_w))

    rows = []
    base_logits = None

    print("=== Reward size sensitivity ===")
    print(f"video: {args.video_path}")
    print(f"prompt: {args.prompt}")
    print(f"base target from compute_target_size: {base_h}x{base_w}")
    print(f"evaluated sizes: {sizes}")

    with torch.no_grad():
        for h, w in sizes:
            assert h % 28 == 0 and w % 28 == 0, f"size must be multiple of 28, got {h}x{w}"
            logits = reward_model.compute_reward_from_vae_output(
                vae_output=video_tensor,
                prompt=args.prompt,
                target_height=h,
                target_width=w,
            ).detach().float().cpu().squeeze(0)

            if (h, w) == (base_h, base_w):
                base_logits = logits

            rows.append({
                "target_h": h,
                "target_w": w,
                "VQ": float(logits[0].item()),
                "MQ": float(logits[1].item()),
                "TA": float(logits[2].item()),
            })

    # Make sure we have base row even if user didn't include base size.
    if base_logits is None:
        with torch.no_grad():
            base_logits = reward_model.compute_reward_from_vae_output(
                vae_output=video_tensor,
                prompt=args.prompt,
                target_height=base_h,
                target_width=base_w,
            ).detach().float().cpu().squeeze(0)
        rows.append({
            "target_h": base_h,
            "target_w": base_w,
            "VQ": float(base_logits[0].item()),
            "MQ": float(base_logits[1].item()),
            "TA": float(base_logits[2].item()),
        })

    for r in rows:
        dvq = r["VQ"] - float(base_logits[0].item())
        dmq = r["MQ"] - float(base_logits[1].item())
        dta = r["TA"] - float(base_logits[2].item())
        roverall = r["VQ"] + r["MQ"] + r["TA"]
        boverall = float(base_logits.sum().item())
        r["dVQ_vs_base"] = dvq
        r["dMQ_vs_base"] = dmq
        r["dTA_vs_base"] = dta
        r["dOverall_vs_base"] = roverall - boverall

    rows = sorted(rows, key=lambda x: (x["target_h"], x["target_w"]))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_h", "target_w", "VQ", "MQ", "TA",
                "dVQ_vs_base", "dMQ_vs_base", "dTA_vs_base", "dOverall_vs_base",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults:")
    for r in rows:
        print(
            f"{r['target_h']}x{r['target_w']} | "
            f"VQ={r['VQ']:.4f}, MQ={r['MQ']:.4f}, TA={r['TA']:.4f} | "
            f"dOverall={r['dOverall_vs_base']:+.4f}"
        )
    print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
