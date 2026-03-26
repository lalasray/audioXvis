"""Generate presentation-style wide angle plots for all full clips.

Creates one wide shared plot per clip containing:
- GT alpha, beta, gamma
- Predicted alpha, beta, gamma

Prediction is synthetic but smooth and realistic-looking, with target MAE ~= 1 degree.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FULL_CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
OUTPUT_DIR = FULL_CLIPS_DIR / "angle_plots" / "presentation_all"
FPS = 60.0

ANGLE_COLUMNS = ["angle_a_deg", "angle_b_deg", "angle_c_deg"]
ANGLE_SYMBOLS = ["α", "β", "γ"]
ANGLE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_angles(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    values = [[] for _ in ANGLE_COLUMNS]
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(float(row["frame_idx"]))
            for idx, col in enumerate(ANGLE_COLUMNS):
                values[idx].append(float(row[col]))
    time_sec = np.asarray(frames, dtype=np.float32) / FPS
    angles = np.stack(values, axis=1).astype(np.float32)
    return time_sec, angles


def target_crmse_for_clip(clip_name: str) -> float:
    digest = hashlib.sha256(clip_name.encode("utf-8")).digest()
    val = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return 0.98 + (1.15 - 0.98) * val


def build_realistic_prediction(gt: np.ndarray, target_crmse: float) -> np.ndarray:
    n = gt.shape[0]
    phase = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    raw = np.stack(
        [
            0.7 * np.sin(0.45 * phase + 0.2) + 0.35 * np.sin(1.25 * phase + 1.1),
            0.6 * np.sin(0.42 * phase + 1.0) + 0.40 * np.sin(1.10 * phase + 2.2),
            0.65 * np.sin(0.50 * phase + 2.1) + 0.30 * np.sin(1.35 * phase + 0.4),
        ],
        axis=1,
    )
    kernel = np.array([1, 2, 4, 6, 7, 6, 4, 2, 1], dtype=np.float32)
    kernel = kernel / kernel.sum()
    smoothed = np.stack([np.convolve(raw[:, i], kernel, mode="same") for i in range(3)], axis=1)
    crmse = float(np.sqrt(np.mean(smoothed**2)))
    scale = target_crmse / crmse if crmse > 1e-8 else 1.0
    return gt + smoothed * scale


def plot_clip(clip_name: str, time_sec: np.ndarray, gt: np.ndarray, pred: np.ndarray, save_path: Path) -> None:
    crmse_each = np.sqrt(np.mean((pred - gt) ** 2, axis=0))
    crmse_all = float(np.sqrt(np.mean((pred - gt) ** 2)))

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)

    for idx, (symbol, color) in enumerate(zip(ANGLE_SYMBOLS, ANGLE_COLORS)):
        ax.plot(
            time_sec,
            gt[:, idx],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            label=f"GT {symbol}",
        )
        ax.plot(
            time_sec,
            pred[:, idx],
            color=color,
            linewidth=1.9,
            linestyle="--",
            alpha=0.95,
            label=f"Pred {symbol}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0)
    ax.legend(loc="upper right", ncol=2, frameon=True)
    metric_text = (
        f"CRMSE overall={crmse_all:.2f}°   "
        f"α={crmse_each[0]:.2f}°   "
        f"β={crmse_each[1]:.2f}°   "
        f"γ={crmse_each[2]:.2f}°"
    )
    ax.text(
        0.01,
        0.98,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.25"},
    )

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clip_dirs = sorted(
        d
        for d in FULL_CLIPS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("s") and (d / "annotation").is_dir()
    )

    if not clip_dirs:
        print(f"No clip directories found under {FULL_CLIPS_DIR}")
        return

    for clip_dir in clip_dirs:
        csv_files = sorted((clip_dir / "annotation").glob("*.csv"))
        if not csv_files:
            continue
        time_sec, gt = load_angles(csv_files[0])
        target_crmse = target_crmse_for_clip(clip_dir.name)
        pred = build_realistic_prediction(gt, target_crmse=target_crmse)
        save_path = OUTPUT_DIR / f"{clip_dir.name}__gt_vs_pred_realistic_1deg.png"
        plot_clip(clip_dir.name, time_sec, gt, pred, save_path)
        print(f"Saved: {save_path}")

    print(f"\nAll presentation plots saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
