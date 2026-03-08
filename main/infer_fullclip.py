"""
Infer full clip sequences window-by-window and plot GT vs Predicted.
V2: loads feature normalization stats from checkpoint, uses EMA weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv

from dataloader_fullclip import (
    extract_features,
    normalize_features,
    load_annotation_csv,
    load_audio,
    SR,
    HOP_LENGTH,
    VIDEO_FPS,
    PREDICT_COLUMNS,
)
from train_diffusion_fullclip import DiffusionRegressor, derive_angle_b, move_dict


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_shapes = ckpt["feature_shapes"]
    y_dim = ckpt["y_dim"]
    cfg = ckpt["config"]

    state_for_shape = ckpt.get("ema_state", ckpt.get("model_state", {}))
    user_embed = state_for_shape.get("denoiser.user_embed.weight", None)
    if user_embed is not None:
        num_users = int(user_embed.shape[0])
    else:
        num_users = int(ckpt.get("num_users", 1))

    model = DiffusionRegressor(
        feature_shapes=feature_shapes,
        y_dim=y_dim,
        steps=cfg["diffusion_steps"],
        t_embed_dim=cfg["t_embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        feature_embed_dim=cfg["feature_embed_dim"],
        branch_channels=cfg.get("branch_channels", 64),
        n_denoiser_blocks=cfg.get("n_denoiser_blocks", 4),
        dropout=cfg.get("dropout", 0.1),
        num_users=num_users,
    ).to(device)

    # Use EMA weights if available, else model weights
    if "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"])
        print("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state"])
        print("  Loaded model weights (no EMA)")
    model.eval()

    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    sample_steps = cfg["sample_steps"]

    # Feature normalization stats
    raw_stats = ckpt.get("feature_stats", None)
    if raw_stats is not None:
        feature_stats = {
            k: (np.array(m, dtype=np.float32), np.array(s, dtype=np.float32))
            for k, (m, s) in raw_stats.items()
        }
        print("  Loaded feature normalization stats")
    else:
        feature_stats = None
        print("  WARNING: No feature_stats in checkpoint — features will NOT be normalized")

    return model, y_mean, y_std, sample_steps, cfg, feature_stats


@torch.no_grad()
def infer_full_clip(
    model: DiffusionRegressor,
    y_full: np.ndarray,
    sr: int,
    window_sec: float,
    hop_sec: float,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_steps: int,
    device: torch.device,
    feature_stats: Dict | None = None,
    batch_size: int = 64,
    save_pred_csv: str = None,
):
    """Slide 1s windows, extract + normalize features, infer, return predictions."""
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_windows = max(0, (len(y_full) - window_samples) // hop_samples + 1)

    all_preds = []

    windows_meta = []
    for w_idx in range(n_windows):
        start_sample = w_idx * hop_samples
        center_sec = start_sample / sr + window_sec / 2
        windows_meta.append((w_idx, start_sample, center_sec))

    for batch_start in range(0, len(windows_meta), batch_size):
        batch_meta = windows_meta[batch_start : batch_start + batch_size]
        batch_features = {
            "mel_spectrogram": [],
            "mfcc": [],
            "rms_energy": [],
            "zero_crossing_rate": [],
            "spectral_contrast": [],
        }

        for _, start_sample, _ in batch_meta:
            y_window = y_full[start_sample : start_sample + window_samples]
            if len(y_window) < window_samples:
                y_window = np.pad(y_window, (0, window_samples - len(y_window)))

            feats = extract_features(y_window, sr=sr)
            # Normalize using training stats
            if feature_stats is not None:
                feats = normalize_features(feats, feature_stats)

            for k in batch_features:
                batch_features[k].append(torch.from_numpy(feats[k].copy()))

        x_dict = {k: torch.stack(v, dim=0).to(device) for k, v in batch_features.items()}

        # Supply user_id tensor (single user: 0)
        user_id = torch.zeros(x_dict["mel_spectrogram"].shape[0], dtype=torch.long, device=device)
        pred_norm = model.sample_ddim(x_dict, sample_steps=sample_steps, user_id=user_id)
        pred = pred_norm * y_std + y_mean

        pred_a = pred[:, 0].cpu().numpy()
        pred_c = pred[:, 1].cpu().numpy()
        pred_b = 180.0 - pred_a - pred_c

        for i, (_, _, center_sec) in enumerate(batch_meta):
            all_preds.append((center_sec, pred_a[i], pred_b[i], pred_c[i]))

    centers = np.array([p[0] for p in all_preds])
    pred_angles = np.array([[p[1], p[2], p[3]] for p in all_preds])

    # Save predicted angles to CSV if requested
    if save_pred_csv is not None:
        with open(save_pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time", "angle_a", "angle_b", "angle_c"])
            for idx, (center_sec, a, b, c) in enumerate(all_preds):
                writer.writerow([idx, center_sec, a, b, c])

    return centers, pred_angles


def plot_gt_vs_pred(
    gt_t: np.ndarray,
    gt_angles: np.ndarray,
    pred_t: np.ndarray,
    pred_angles: np.ndarray,
    clip_name: str,
    save_path: Path,
):
    """Plot GT vs Predicted for all 3 angles on a single plot."""
    angle_names = ["angle_a", "angle_b", "angle_c"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(18, 6))

    mae_texts = []
    for i, (name, color) in enumerate(zip(angle_names, colors)):
        ax.plot(
            gt_t, gt_angles[:, i],
            label=f"GT {name}", color=color, linewidth=1.0, alpha=0.8,
        )
        ax.plot(
            pred_t, pred_angles[:, i],
            label=f"Pred {name}", color=color, linewidth=1.0, alpha=0.8, linestyle="--",
        )
        gt_interp = np.interp(pred_t, gt_t, gt_angles[:, i])
        mae = np.mean(np.abs(gt_interp - pred_angles[:, i]))
        mae_texts.append(f"{name} MAE={mae:.2f}°")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(f"GT vs Predicted — {clip_name}   |   {',  '.join(mae_texts)}")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Infer full clips and plot GT vs Predicted")
    parser.add_argument(
        "--ckpt", type=str, default="main/checkpoints/diffusion_v2/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--clips_root", type=str, default="data/test_dataset/full_clips",
    )
    parser.add_argument("--clip_name", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="main/checkpoints/diffusion_v2/inference_plots",
    )
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, y_mean, y_std, sample_steps, cfg, feature_stats = load_model(args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}")

    clips_root = Path(args.clips_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_name:
        clip_dirs = [clips_root / args.clip_name]
    else:
        clip_dirs = sorted(
            d for d in clips_root.iterdir()
            if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
        )

    all_maes = {"angle_a": [], "angle_b": [], "angle_c": []}

    for idx, clip_dir in enumerate(clip_dirs, 1):
        clip_name = clip_dir.name
        videos = sorted((clip_dir / "video").glob("*.mp4"))
        annos = sorted((clip_dir / "annotation").glob("*.csv"))
        if not videos or not annos:
            print(f"[{idx}/{len(clip_dirs)}] Skipping {clip_name}: missing files")
            continue

        print(f"\n[{idx}/{len(clip_dirs)}] {clip_name}")

        y_full = load_audio(str(videos[0]), sr=SR)
        sr = SR
        print(f"  Audio: {len(y_full) / sr:.2f}s")

        gt_t, gt_angles = load_annotation_csv(annos[0])
        print(f"  GT: {len(gt_t)} frames, {gt_t[-1]:.2f}s")

        pred_csv_path = out_dir / f"{clip_name}_pred_angles.csv"
        pred_t, pred_angles = infer_full_clip(
            model=model,
            y_full=y_full,
            sr=sr,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            y_mean=y_mean,
            y_std=y_std,
            sample_steps=sample_steps,
            device=device,
            feature_stats=feature_stats,
            batch_size=args.batch_size,
            save_pred_csv=str(pred_csv_path),
        )
        print(f"  Predicted: {len(pred_t)} windows")

        for i, name in enumerate(["angle_a", "angle_b", "angle_c"]):
            gt_interp = np.interp(pred_t, gt_t, gt_angles[:, i])
            mae = np.mean(np.abs(gt_interp - pred_angles[:, i]))
            all_maes[name].append(mae)
            print(f"  {name} MAE: {mae:.2f}°")

        save_path = out_dir / f"{clip_name}_gt_vs_pred.png"
        plot_gt_vs_pred(gt_t, gt_angles, pred_t, pred_angles, clip_name, save_path)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary across {len(all_maes['angle_a'])} clips:")
    for name in ["angle_a", "angle_b", "angle_c"]:
        vals = all_maes[name]
        print(f"  {name}: mean MAE={np.mean(vals):.2f}°, std={np.std(vals):.2f}°, "
              f"min={np.min(vals):.2f}°, max={np.max(vals):.2f}°")
    overall = np.mean([np.mean(v) for v in all_maes.values()])
    print(f"  Overall mean MAE: {overall:.2f}°")
    print(f"\nAll plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
