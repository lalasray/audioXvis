"""
Side-by-side plots: angles vs each audio feature aligned by time for every full clip.
Helps visually inspect correlations between larynx geometry and audio.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_feature_extraction import AudioFeatureExtractor

FULL_CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
VIDEO_FPS = 60.0
HOP_LENGTH = 512
SR = 22050


# ── helpers ──────────────────────────────────────────────────────────────────

def load_angles(csv_path):
    frames, a, b, c = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(float(row["frame_idx"]))
            a.append(float(row["angle_a_deg"]))
            b.append(float(row["angle_b_deg"]))
            c.append(float(row["angle_c_deg"]))
    t = np.asarray(frames) / VIDEO_FPS
    return t, np.asarray(a), np.asarray(b), np.asarray(c)


def _frames_time(n_frames):
    """Time axis for librosa frame-level features."""
    return librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP_LENGTH)


def _plot_angles(ax, t, a, b, c):
    """Draw three angle curves on *ax*."""
    ax.plot(t, a, label="angle_a", linewidth=0.7)
    ax.plot(t, b, label="angle_b", linewidth=0.7)
    ax.plot(t, c, label="angle_c", linewidth=0.7)
    ax.set_ylabel("Angle (deg)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.25)


# ── per-feature plot functions ───────────────────────────────────────────────
# Each returns nothing; they draw into the right-hand *ax*.

def plot_rms(ext, ax):
    rms = ext.get_rms_energy(hop_length=HOP_LENGTH)
    t = _frames_time(len(rms))
    ax.plot(t, rms, color="tab:red", linewidth=0.7)
    ax.set_ylabel("RMS Energy")
    ax.set_title("RMS Energy")

def plot_pitch(ext, ax):
    f0 = ext.get_pitch(hop_length=HOP_LENGTH)
    t = _frames_time(len(f0))
    voiced = f0 > 0
    ax.scatter(t[voiced], f0[voiced], s=3, alpha=0.5, color="tab:blue")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Pitch (F0)")
    ax.set_ylim([50, 450])

def plot_spectral_centroid(ext, ax):
    cent = ext.get_spectral_centroid(hop_length=HOP_LENGTH)
    t = _frames_time(len(cent))
    ax.plot(t, cent, color="tab:cyan", linewidth=0.7)
    ax.set_ylabel("Hz")
    ax.set_title("Spectral Centroid")

def plot_spectral_rolloff(ext, ax):
    rolloff = ext.get_spectral_rolloff(hop_length=HOP_LENGTH)
    t = _frames_time(len(rolloff))
    ax.plot(t, rolloff, color="tab:orange", linewidth=0.7)
    ax.set_ylabel("Hz")
    ax.set_title("Spectral Rolloff")

def plot_zcr(ext, ax):
    zcr = ext.get_zero_crossing_rate(hop_length=HOP_LENGTH)
    t = _frames_time(len(zcr))
    ax.plot(t, zcr, color="tab:green", linewidth=0.7)
    ax.set_ylabel("ZCR")
    ax.set_title("Zero Crossing Rate")

def plot_onset(ext, ax):
    onset = ext.get_onset_strength(hop_length=HOP_LENGTH)
    t = _frames_time(len(onset))
    ax.plot(t, onset, color="tab:purple", linewidth=0.7)
    ax.set_ylabel("Strength")
    ax.set_title("Onset Strength")

def plot_mel_spectrogram(ext, ax):
    S_db = ext.get_mel_spectrogram(hop_length=HOP_LENGTH)
    librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel Spectrogram")

def plot_mfcc(ext, ax):
    mfcc = ext.get_mfcc(hop_length=HOP_LENGTH)
    librosa.display.specshow(mfcc, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", ax=ax, cmap="viridis")
    ax.set_title("MFCC")

def plot_chroma(ext, ax):
    chroma = ext.get_chroma(hop_length=HOP_LENGTH)
    librosa.display.specshow(chroma, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="chroma", ax=ax, cmap="hsv")
    ax.set_title("Chroma")

def plot_tempogram(ext, ax):
    tempogram = ext.get_tempogram(hop_length=HOP_LENGTH)
    librosa.display.specshow(tempogram, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="tempo", ax=ax, cmap="viridis")
    ax.set_title("Tempogram")

def plot_spectral_contrast(ext, ax):
    contrast = ext.get_spectral_contrast(hop_length=HOP_LENGTH)
    librosa.display.specshow(contrast, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", ax=ax, cmap="viridis")
    ax.set_title("Spectral Contrast")

def plot_harmonic_ratio(ext, ax):
    harmonic, percussive = ext.get_harmonic_percussive()
    h_rms = librosa.feature.rms(y=harmonic, hop_length=HOP_LENGTH)[0]
    p_rms = librosa.feature.rms(y=percussive, hop_length=HOP_LENGTH)[0]
    total = h_rms + p_rms + 1e-10
    ratio = h_rms / total
    t = _frames_time(len(ratio))
    ax.plot(t, ratio, color="tab:brown", linewidth=0.7)
    ax.set_ylabel("H / (H+P)")
    ax.set_title("Harmonic Ratio")
    ax.set_ylim([0, 1])


FEATURE_PLOTS = [
    ("rms_energy",          plot_rms),
    ("pitch_f0",            plot_pitch),
    ("spectral_centroid",   plot_spectral_centroid),
    ("spectral_rolloff",    plot_spectral_rolloff),
    ("zero_crossing_rate",  plot_zcr),
    ("onset_strength",      plot_onset),
    ("mel_spectrogram",     plot_mel_spectrogram),
    ("mfcc",                plot_mfcc),
    ("chroma",              plot_chroma),
    ("tempogram",           plot_tempogram),
    ("spectral_contrast",   plot_spectral_contrast),
    ("harmonic_ratio",      plot_harmonic_ratio),
]


# ── main ─────────────────────────────────────────────────────────────────────

def process_clip(clip_dir, out_dir):
    clip_name = clip_dir.name

    # locate annotation & video
    anno_csvs = sorted((clip_dir / "annotation").glob("*.csv"))
    video_files = sorted((clip_dir / "video").glob("*.mp4"))
    if not anno_csvs or not video_files:
        print(f"  Skipping {clip_name}: missing annotation or video")
        return

    t_ang, a, b, c = load_angles(anno_csvs[0])
    ext = AudioFeatureExtractor(str(video_files[0]), sr=SR)

    clip_out = out_dir / clip_name
    clip_out.mkdir(exist_ok=True)

    for feat_name, feat_func in FEATURE_PLOTS:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 4), sharex=True)

        # left: angles
        _plot_angles(ax_left, t_ang, a, b, c)
        ax_left.set_xlabel("Time (s)")
        ax_left.set_title("Larynx Angles")

        # right: audio feature
        try:
            feat_func(ext, ax_right)
        except Exception as e:
            ax_right.text(0.5, 0.5, f"Error: {e}", transform=ax_right.transAxes,
                          ha="center", va="center", fontsize=9, color="red")
        ax_right.set_xlabel("Time (s)")
        ax_right.grid(True, alpha=0.25)

        fig.suptitle(f"{clip_name}  —  Angles vs {feat_name}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = clip_out / f"{clip_name}_{feat_name}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    print(f"  Saved {len(FEATURE_PLOTS)} correlation plots -> {clip_out.name}/")


def main():
    clip_dirs = sorted(
        d for d in FULL_CLIPS_DIR.iterdir()
        if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
    )
    if not clip_dirs:
        print("No clip directories found.")
        return

    out_root = FULL_CLIPS_DIR / "angle_vs_feature_plots"
    out_root.mkdir(exist_ok=True)

    for i, clip_dir in enumerate(clip_dirs, 1):
        print(f"[{i}/{len(clip_dirs)}] {clip_dir.name}")
        process_clip(clip_dir, out_root)

    print(f"\nDone — all plots under: {out_root}")


if __name__ == "__main__":
    main()
