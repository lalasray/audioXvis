"""
Infer all 1s rolling windows of a full clip sequence, reconstruct the full
timeline of predicted angles, and plot GT vs Predicted for all 3 angles.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader_fullclip import (
    FullClipRollingDataset,
    extract_features,
    load_annotation_csv,
    SR,
    HOP_LENGTH,
    VIDEO_FPS,
    PREDICT_COLUMNS,
)
from train_diffusion_fullclip import DiffusionRegressor, derive_angle_b, move_dict


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_shapes = ckpt["feature_shapes"]
    y_dim = ckpt["y_dim"]
    cfg = ckpt["config"]

    model = DiffusionRegressor(
        feature_shapes=feature_shapes,
        y_dim=y_dim,
        steps=cfg["diffusion_steps"],
        t_embed_dim=cfg["t_embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        feature_embed_dim=cfg["feature_embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    sample_steps = cfg["sample_steps"]

    return model, y_mean, y_std, sample_steps, cfg


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
    batch_size: int = 64,
):
    """
    Slide 1s windows over the full audio, extract features, run inference,
    and average overlapping predictions.
    Returns: (time_centers, pred_angles[N,3], window_starts, window_ends)
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_windows = max(0, (len(y_full) - window_samples) // hop_samples + 1)

    all_preds = []  # list of (center_sec, pred_a, pred_b, pred_c)

    # Collect windows in batches
    windows_meta = []
    for w_idx in range(n_windows):
        start_sample = w_idx * hop_samples
        start_sec = start_sample / sr
        center_sec = start_sec + window_sec / 2
        windows_meta.append((w_idx, start_sample, start_sec, center_sec))

    # Process in batches
    for batch_start in range(0, len(windows_meta), batch_size):
        batch_meta = windows_meta[batch_start:batch_start + batch_size]
        batch_features = {
            "mel_spectrogram": [],
            "mfcc": [],
            "rms_energy": [],
            "zero_crossing_rate": [],
            "spectral_contrast": [],
        }

        for _, start_sample, _, _ in batch_meta:
            y_window = y_full[start_sample:start_sample + window_samples]
            if len(y_window) < window_samples:
                y_window = np.pad(y_window, (0, window_samples - len(y_window)))
            feats = extract_features(y_window, sr=sr)
            for k in batch_features:
                batch_features[k].append(torch.from_numpy(feats[k]))

        x_dict = {k: torch.stack(v, dim=0).to(device) for k, v in batch_features.items()}

        pred_norm = model.sample_ddim(x_dict, sample_steps=sample_steps)
        pred = pred_norm * y_std + y_mean  # (B, 2) = [angle_a, angle_c]

        pred_a = pred[:, 0].cpu().numpy()
        pred_c = pred[:, 1].cpu().numpy()
        pred_b = 180.0 - pred_a - pred_c

        for i, (_, _, _, center_sec) in enumerate(batch_meta):
            all_preds.append((center_sec, pred_a[i], pred_b[i], pred_c[i]))

    centers = np.array([p[0] for p in all_preds])
    pred_angles = np.array([[p[1], p[2], p[3]] for p in all_preds])  # (N, 3) = a, b, c

    return centers, pred_angles


def plot_gt_vs_pred(
    gt_t: np.ndarray,
    gt_angles: np.ndarray,
    pred_t: np.ndarray,
    pred_angles: np.ndarray,
    clip_name: str,
    save_path: Path,
):
    """Plot GT vs Predicted for all 3 angles, stacked vertically."""
    angle_names = ["angle_a", "angle_b", "angle_c"]
    colors_gt = ["tab:blue", "tab:orange", "tab:green"]
    colors_pred = ["tab:red", "tab:purple", "tab:brown"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        ax.plot(gt_t, gt_angles[:, i], label=f"GT {name}", color=colors_gt[i],
                linewidth=0.8, alpha=0.8)
        ax.plot(pred_t, pred_angles[:, i], label=f"Pred {name}", color=colors_pred[i],
                linewidth=0.8, alpha=0.8, linestyle="--")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

        # Compute MAE for this angle
        gt_interp = np.interp(pred_t, gt_t, gt_angles[:, i])
        mae = np.mean(np.abs(gt_interp - pred_angles[:, i]))
        ax.text(0.02, 0.95, f"MAE = {mae:.2f}°", transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"GT vs Predicted Angles — {clip_name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Infer a full clip and plot GT vs Predicted angles")
    parser.add_argument("--ckpt", type=str, default="main/checkpoints/diffusion_fullclip/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--clips_root", type=str, default="data/test_dataset/full_clips",
                        help="Full clips root directory")
    parser.add_argument("--clip_name", type=str, default=None,
                        help="Specific clip name (e.g. s1_01__gt_s1_01). If None, infer all clips.")
    parser.add_argument("--output_dir", type=str, default="main/checkpoints/diffusion_fullclip/inference_plots",
                        help="Output directory for plots")
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, y_mean, y_std, sample_steps, cfg = load_model(args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}")

    clips_root = Path(args.clips_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect clip directories
    if args.clip_name:
        clip_dirs = [clips_root / args.clip_name]
    else:
        clip_dirs = sorted(
            d for d in clips_root.iterdir()
            if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
        )

    import librosa

    for idx, clip_dir in enumerate(clip_dirs, 1):
        clip_name = clip_dir.name
        videos = sorted((clip_dir / "video").glob("*.mp4"))
        annos = sorted((clip_dir / "annotation").glob("*.csv"))
        if not videos or not annos:
            print(f"[{idx}/{len(clip_dirs)}] Skipping {clip_name}: missing files")
            continue

        print(f"\n[{idx}/{len(clip_dirs)}] {clip_name}")

        # Load full audio
        y_full, sr = librosa.load(str(videos[0]), sr=SR)
        print(f"  Audio: {len(y_full)/sr:.2f}s")

        # Load GT
        gt_t, gt_angles = load_annotation_csv(annos[0])
        print(f"  GT: {len(gt_t)} frames, {gt_t[-1]:.2f}s")

        # Infer
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
            batch_size=args.batch_size,
        )
        print(f"  Predicted: {len(pred_t)} windows")

        # Per-angle MAE
        for i, name in enumerate(["angle_a", "angle_b", "angle_c"]):
            gt_interp = np.interp(pred_t, gt_t, gt_angles[:, i])
            mae = np.mean(np.abs(gt_interp - pred_angles[:, i]))
            print(f"  {name} MAE: {mae:.2f}°")

        # Plot
        save_path = out_dir / f"{clip_name}_gt_vs_pred.png"
        plot_gt_vs_pred(gt_t, gt_angles, pred_t, pred_angles, clip_name, save_path)

    print(f"\nAll inference plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
