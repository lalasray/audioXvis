"""
Dataloader for full-clip 1-second rolling-window training.
V2: Pre-extracts features at init, per-channel normalization, data augmentation.

Features (all frame-local, safe for 1s windows):
  - mel_spectrogram    (128 × T)
  - mfcc               (13  × T)
  - rms_energy         (1   × T)
  - zero_crossing_rate (1   × T)
  - spectral_contrast  (7   × T)

Targets: angle_a_deg, angle_c_deg  (predict 2, derive angle_b = 180 - a - c)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ── constants ────────────────────────────────────────────────────────────────

SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 13
VIDEO_FPS = 60.0

PREDICT_COLUMNS = ("angle_a_deg", "angle_c_deg")
ALL_ANGLE_COLUMNS = ("angle_a_deg", "angle_b_deg", "angle_c_deg")

COL_INDEX = {"angle_a_deg": 0, "angle_b_deg": 1, "angle_c_deg": 2}

# ── annotation loading ───────────────────────────────────────────────────────


def load_annotation_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load annotation CSV. Returns (timestamps_sec, angles[N,3]) for a/b/c."""
    frames, aa, ab, ac = [], [], [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(float(row["frame_idx"]))
            aa.append(float(row["angle_a_deg"]))
            ab.append(float(row["angle_b_deg"]))
            ac.append(float(row["angle_c_deg"]))
    t = np.asarray(frames) / VIDEO_FPS
    angles = np.stack([aa, ab, ac], axis=1).astype(np.float32)
    return t, angles


# ── feature extraction ───────────────────────────────────────────────────────


def extract_features(y: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray]:
    """Extract the 5 real-time features from a 1s audio chunk."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel), sr=sr, n_mfcc=N_MFCC
    ).astype(np.float32)

    rms = librosa.feature.rms(
        S=S, frame_length=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32)

    contrast = librosa.feature.spectral_contrast(
        S=S, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32)

    return {
        "mel_spectrogram": mel_db,        # (128, T)
        "mfcc": mfcc,                     # (13, T)
        "rms_energy": rms,                # (1, T)
        "zero_crossing_rate": zcr,        # (1, T)
        "spectral_contrast": contrast,    # (7, T)
    }


# ── feature normalization ────────────────────────────────────────────────────


def compute_feature_stats(
    features_list: List[Dict[str, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute per-channel mean and std from a list of feature dicts.

    Returns dict  {feature_name: (mean[C,1], std[C,1])}
    """
    accum: Dict[str, List[np.ndarray]] = {}
    for feats in features_list:
        for k, v in feats.items():
            if v.ndim == 1:
                v = v.reshape(1, -1)
            accum.setdefault(k, []).append(v)

    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k, arrays in accum.items():
        cat = np.concatenate(arrays, axis=-1)  # (C, total_T)
        mean = cat.mean(axis=-1, keepdims=True).astype(np.float32)
        std = np.maximum(cat.std(axis=-1, keepdims=True), 1e-6).astype(np.float32)
        stats[k] = (mean, std)
    return stats


def normalize_features(
    feats: Dict[str, np.ndarray],
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Normalize features using precomputed per-channel stats."""
    normed = {}
    for k, v in feats.items():
        if k in stats:
            mean, std = stats[k]
            normed[k] = ((v - mean) / std).astype(np.float32)
        else:
            normed[k] = v
    return normed


# ── dataset ──────────────────────────────────────────────────────────────────


class FullClipRollingDataset(Dataset):
    """
    Pre-extracts all features at __init__ for fast training.
    Supports per-channel normalization and on-the-fly data augmentation.
    """

    def __init__(
        self,
        clips_root: str | Path,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
        sr: int = SR,
        target_columns: Sequence[str] = PREDICT_COLUMNS,
        feature_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        augment: bool = False,
        noise_std: float = 0.05,
        time_mask_ratio: float = 0.15,
    ) -> None:
        self.clips_root = Path(clips_root)
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sr = sr
        self.target_columns = list(target_columns)
        self.window_samples = int(window_sec * sr)
        self.augment = augment
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio

        target_col_indices = [COL_INDEX[c] for c in self.target_columns]
        self._target_col_indices = target_col_indices

        # Discover clips
        clip_dirs = sorted(
            d for d in self.clips_root.iterdir()
            if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
        )
        if not clip_dirs:
            raise ValueError(f"No clip directories found under {self.clips_root}")

        # Pre-extract features and targets
        self._features: List[Dict[str, np.ndarray]] = []
        self._targets: List[np.ndarray] = []
        self._meta: List[Tuple[str, int, float, float]] = []
        self._clip_names: List[str] = []  # unique clip names in order

        hop_samples = int(hop_sec * sr)

        print(f"Pre-extracting features from {len(clip_dirs)} clips (hop={hop_sec}s)...")
        for ci, clip_dir in enumerate(clip_dirs):
            clip_name = clip_dir.name
            videos = sorted((clip_dir / "video").glob("*.mp4"))
            annos = sorted((clip_dir / "annotation").glob("*.csv"))
            if not videos or not annos:
                continue

            y_full, _ = librosa.load(str(videos[0]), sr=sr)
            anno_t, angles = load_annotation_csv(annos[0])

            n_windows = max(0, (len(y_full) - self.window_samples) // hop_samples + 1)
            self._clip_names.append(clip_name)

            for w_idx in range(n_windows):
                start_sample = w_idx * hop_samples
                start_sec = start_sample / sr
                end_sec = start_sec + window_sec

                y_window = y_full[start_sample : start_sample + self.window_samples]
                if len(y_window) < self.window_samples:
                    y_window = np.pad(y_window, (0, self.window_samples - len(y_window)))

                feats = extract_features(y_window, sr=sr)

                in_window = (anno_t >= start_sec) & (anno_t < end_sec)
                if np.any(in_window):
                    target = np.mean(
                        angles[in_window][:, target_col_indices], axis=0
                    ).astype(np.float32)
                else:
                    target = np.zeros(len(target_col_indices), dtype=np.float32)

                self._features.append(feats)
                self._targets.append(target)
                self._meta.append((clip_name, w_idx, start_sec, end_sec))

            print(f"  [{ci + 1}/{len(clip_dirs)}] {clip_name}: {n_windows} windows")

        if not self._features:
            raise ValueError("No valid windows generated.")

        # Feature stats: compute from data or use provided
        if feature_stats is not None:
            self.feature_stats = feature_stats
        else:
            self.feature_stats = compute_feature_stats(self._features)

        print(
            f"FullClipRollingDataset: {len(self._clip_names)} clips, "
            f"{len(self._features)} windows ({window_sec}s, hop {hop_sec}s)"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> Dict:
        feats_raw = self._features[index]
        target = self._targets[index]
        clip_name, w_idx, start_sec, end_sec = self._meta[index]

        # Normalize
        feats = normalize_features(feats_raw, self.feature_stats)

        # Augment (on normalized features)
        if self.augment:
            feats = self._apply_augmentation(feats)

        x = {k: torch.from_numpy(v.copy()).float() for k, v in feats.items()}
        y = torch.from_numpy(target.copy()).float()

        return {
            "x": x,
            "y": y,
            "clip_name": clip_name,
            "window_idx": w_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }

    # ------------------------------------------------------------------

    def _apply_augmentation(self, feats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        augmented = {}
        for k, v in feats.items():
            v = v.copy()
            # Gaussian noise (50% chance)
            if np.random.random() < 0.5:
                v = v + np.random.randn(*v.shape).astype(np.float32) * self.noise_std
            # Time masking (30% chance)
            if np.random.random() < 0.3:
                T = v.shape[-1]
                mask_len = max(1, int(T * self.time_mask_ratio * np.random.random()))
                start = np.random.randint(0, max(1, T - mask_len))
                v[..., start : start + mask_len] = 0.0
            augmented[k] = v
        return augmented

    def get_clip_indices(self) -> Dict[str, List[int]]:
        """Return {clip_name: [indices]} for clip-based splitting."""
        clip_idx: Dict[str, List[int]] = {}
        for i, (cn, *_rest) in enumerate(self._meta):
            clip_idx.setdefault(cn, []).append(i)
        return clip_idx


# ── collate ──────────────────────────────────────────────────────────────────


def rolling_collate(batch: List[Dict]) -> Dict:
    x_list = [item["x"] for item in batch]
    y_list = [item["y"] for item in batch]

    x_out = {}
    for k in x_list[0].keys():
        x_out[k] = torch.stack([d[k] for d in x_list], dim=0)

    y_out = torch.stack(y_list, dim=0)
    return {
        "x": x_out,
        "y": y_out,
        "clip_name": [item["clip_name"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
    }


# ── quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clips_root = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
    ds = FullClipRollingDataset(clips_root, window_sec=1.0, hop_sec=0.25)
    print(f"Total windows: {len(ds)}")
    sample = ds[0]
    print("Features:")
    for k, v in sample["x"].items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"Target (a, c): {sample['y']}")
    print(f"Feature stats keys: {list(ds.feature_stats.keys())}")
    for k, (m, s) in ds.feature_stats.items():
        print(f"  {k}: mean={m.flatten()[:3]}..., std={s.flatten()[:3]}...")
