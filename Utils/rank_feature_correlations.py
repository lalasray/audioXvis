"""
Compute Pearson correlation between each audio feature and angle signals,
averaged across all full clips. Rank features by how well they follow angle changes.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_feature_extraction import AudioFeatureExtractor

FULL_CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
VIDEO_FPS = 60.0
HOP_LENGTH = 512
SR = 22050


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


def resample_to_common(t_angle, angle_vals, t_feat, feat_vals):
    """Interpolate both signals onto a common time grid and return aligned arrays."""
    t_min = max(t_angle[0], t_feat[0])
    t_max = min(t_angle[-1], t_feat[-1])
    n_pts = min(len(t_angle), len(t_feat), 2000)
    t_common = np.linspace(t_min, t_max, n_pts)
    ang_interp = np.interp(t_common, t_angle, angle_vals)
    feat_interp = np.interp(t_common, t_feat, feat_vals)
    return ang_interp, feat_interp


def safe_pearson(x, y):
    """Pearson correlation, return 0 if constant signal."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def frames_time(n_frames):
    return librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP_LENGTH)


def extract_1d_features(ext):
    """Return dict of {name: (time_array, value_array)} for 1-D features."""
    features = {}

    # RMS
    rms = ext.get_rms_energy(hop_length=HOP_LENGTH)
    features["rms_energy"] = (frames_time(len(rms)), rms)

    # Pitch
    f0 = ext.get_pitch(hop_length=HOP_LENGTH)
    features["pitch_f0"] = (frames_time(len(f0)), f0)

    # Spectral centroid
    cent = ext.get_spectral_centroid(hop_length=HOP_LENGTH)
    features["spectral_centroid"] = (frames_time(len(cent)), cent)

    # Spectral rolloff
    rolloff = ext.get_spectral_rolloff(hop_length=HOP_LENGTH)
    features["spectral_rolloff"] = (frames_time(len(rolloff)), rolloff)

    # ZCR
    zcr = ext.get_zero_crossing_rate(hop_length=HOP_LENGTH)
    features["zero_crossing_rate"] = (frames_time(len(zcr)), zcr)

    # Onset strength
    onset = ext.get_onset_strength(hop_length=HOP_LENGTH)
    features["onset_strength"] = (frames_time(len(onset)), onset)

    # Harmonic ratio
    harmonic, percussive = ext.get_harmonic_percussive()
    h_rms = librosa.feature.rms(y=harmonic, hop_length=HOP_LENGTH)[0]
    p_rms = librosa.feature.rms(y=percussive, hop_length=HOP_LENGTH)[0]
    ratio = h_rms / (h_rms + p_rms + 1e-10)
    features["harmonic_ratio"] = (frames_time(len(ratio)), ratio)

    # For 2-D features, reduce to meaningful 1-D summaries:

    # Mel spectrogram -> mean energy per frame
    mel = ext.get_mel_spectrogram(hop_length=HOP_LENGTH)
    mel_mean = np.mean(mel, axis=0)
    features["mel_spec_mean"] = (frames_time(mel.shape[1]), mel_mean)

    # MFCC -> first coefficient (overall energy shape)
    mfcc = ext.get_mfcc(hop_length=HOP_LENGTH)
    features["mfcc_0"] = (frames_time(mfcc.shape[1]), mfcc[0])
    # Also MFCC 1 (spectral slope/tilt)
    if mfcc.shape[0] > 1:
        features["mfcc_1"] = (frames_time(mfcc.shape[1]), mfcc[1])

    # Chroma -> max chroma bin energy per frame
    chroma = ext.get_chroma(hop_length=HOP_LENGTH)
    chroma_max = np.max(chroma, axis=0)
    features["chroma_max"] = (frames_time(chroma.shape[1]), chroma_max)

    # Spectral contrast -> mean across bands
    contrast = ext.get_spectral_contrast(hop_length=HOP_LENGTH)
    contrast_mean = np.mean(contrast, axis=0)
    features["spectral_contrast_mean"] = (frames_time(contrast.shape[1]), contrast_mean)

    # Tempogram -> dominant tempo strength per frame
    tempogram = ext.get_tempogram(hop_length=HOP_LENGTH)
    tempo_max = np.max(tempogram, axis=0)
    features["tempogram_max"] = (frames_time(tempogram.shape[1]), tempo_max)

    return features


def main():
    clip_dirs = sorted(
        d for d in FULL_CLIPS_DIR.iterdir()
        if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
    )
    if not clip_dirs:
        print("No clip directories found.")
        return

    angle_names = ["angle_a", "angle_b", "angle_c"]

    # Accumulate |correlation| per feature per angle across clips
    # {feat_name: {angle_name: [corr_values...]}}
    all_corrs = {}

    for i, clip_dir in enumerate(clip_dirs, 1):
        clip_name = clip_dir.name
        anno_csvs = sorted((clip_dir / "annotation").glob("*.csv"))
        video_files = sorted((clip_dir / "video").glob("*.mp4"))
        if not anno_csvs or not video_files:
            continue

        print(f"[{i}/{len(clip_dirs)}] {clip_name}", end=" ... ")
        t_ang, a, b, c = load_angles(anno_csvs[0])
        angles = {"angle_a": a, "angle_b": b, "angle_c": c}

        try:
            ext = AudioFeatureExtractor(str(video_files[0]), sr=SR)
        except Exception as e:
            print(f"ERROR loading audio: {e}")
            continue

        features = extract_1d_features(ext)

        for feat_name, (t_feat, feat_vals) in features.items():
            if feat_name not in all_corrs:
                all_corrs[feat_name] = {an: [] for an in angle_names}

            for an_name in angle_names:
                ang_aligned, feat_aligned = resample_to_common(
                    t_ang, angles[an_name], t_feat, feat_vals
                )
                r = safe_pearson(ang_aligned, feat_aligned)
                all_corrs[feat_name][an_name].append(abs(r))

        print("done")

    # Compute averages and rank
    print("\n" + "=" * 90)
    print("FEATURE CORRELATION RANKING (avg |Pearson r| across all clips)")
    print("=" * 90)

    rankings = []
    for feat_name, angle_dict in all_corrs.items():
        avg_per_angle = {}
        for an_name, vals in angle_dict.items():
            avg_per_angle[an_name] = np.mean(vals) if vals else 0.0
        overall = np.mean(list(avg_per_angle.values()))
        rankings.append((feat_name, overall, avg_per_angle))

    rankings.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Feature':<25} {'Avg |r|':<10} {'angle_a':<10} {'angle_b':<10} {'angle_c':<10}")
    print("-" * 70)
    for rank, (feat_name, overall, per_angle) in enumerate(rankings, 1):
        print(f"{rank:<5} {feat_name:<25} {overall:<10.4f} "
              f"{per_angle['angle_a']:<10.4f} {per_angle['angle_b']:<10.4f} {per_angle['angle_c']:<10.4f}")

    # Save to file
    out_path = FULL_CLIPS_DIR / "feature_correlation_ranking.txt"
    with open(out_path, "w") as f:
        f.write("FEATURE CORRELATION RANKING\n")
        f.write("Average |Pearson r| between each audio feature and larynx angles\n")
        f.write(f"Computed across {len(clip_dirs)} full clips\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Rank':<5} {'Feature':<25} {'Avg |r|':<10} {'angle_a':<10} {'angle_b':<10} {'angle_c':<10}\n")
        f.write("-" * 70 + "\n")
        for rank, (feat_name, overall, per_angle) in enumerate(rankings, 1):
            f.write(f"{rank:<5} {feat_name:<25} {overall:<10.4f} "
                    f"{per_angle['angle_a']:<10.4f} {per_angle['angle_b']:<10.4f} {per_angle['angle_c']:<10.4f}\n")
    print(f"\nRanking saved to: {out_path}")


if __name__ == "__main__":
    main()
