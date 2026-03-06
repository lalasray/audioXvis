"""Plot angle_a_deg, angle_b_deg, angle_c_deg vs time for every full clip and save figures."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FULL_CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
FPS = 60.0  # all test videos are 60 fps


def load_angles(csv_path: Path):
    """Return (time_sec, angle_a, angle_b, angle_c) arrays from an annotation CSV."""
    frames, a, b, c = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(float(row["frame_idx"]))
            a.append(float(row["angle_a_deg"]))
            b.append(float(row["angle_b_deg"]))
            c.append(float(row["angle_c_deg"]))
    frames = np.asarray(frames)
    time_sec = frames / FPS
    return time_sec, np.asarray(a), np.asarray(b), np.asarray(c)


def plot_and_save(time_sec, angle_a, angle_b, angle_c, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_sec, angle_a, label="angle_a", linewidth=0.8)
    ax.plot(time_sec, angle_b, label="angle_b", linewidth=0.8)
    ax.plot(time_sec, angle_c, label="angle_c", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main():
    clip_dirs = sorted(
        d for d in FULL_CLIPS_DIR.iterdir() if d.is_dir() and (d / "annotation").is_dir()
    )
    if not clip_dirs:
        print(f"No clip directories found under {FULL_CLIPS_DIR}")
        return

    plots_dir = FULL_CLIPS_DIR / "angle_plots"
    plots_dir.mkdir(exist_ok=True)

    for clip_dir in clip_dirs:
        anno_dir = clip_dir / "annotation"
        csvs = sorted(anno_dir.glob("*.csv"))
        if not csvs:
            print(f"No annotation CSV in {anno_dir}, skipping.")
            continue

        for csv_path in csvs:
            clip_name = clip_dir.name  # e.g. s1_01__gt_s1_01
            time_sec, a, b, c = load_angles(csv_path)
            title = f"Angles vs Time — {clip_name}"
            save_path = plots_dir / f"{clip_name}_angles.png"
            plot_and_save(time_sec, a, b, c, title, save_path)

    print(f"\nAll plots saved under: {plots_dir}")


if __name__ == "__main__":
    main()
