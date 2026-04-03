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

import numpy as np
import torch
import torchaudio
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
DATASET_SOURCES = (
    {
        "label": "dataset_a",
        "user_id": 0,
        "clip_roots": ("test_dataset/full_clips",),
        "song_roots": (),
        "gt_roots": (),
    },
    {
        "label": "dataset_b",
        "user_id": 1,
        "clip_roots": (),
        "song_roots": ("test_set_2/songs",),
        "gt_roots": ("test_set_2/gt",),
    },
    {
        "label": "dataset_c",
        "user_id": 2,
        "clip_roots": (),
        "song_roots": ("test_set_3/songs",),
        "gt_roots": ("test_set_3/gt",),
    },
)

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


def discover_training_clips(clips_root: Path) -> List[Dict]:
    """Discover clips from either:
    1) legacy full_clips root, or
    2) project data root containing the bundled dataset folders.
    """
    entries: List[Dict] = []

    # Case 1: legacy full_clips-style root
    legacy_clip_dirs = sorted(
        d for d in clips_root.iterdir()
        if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
    ) if clips_root.exists() else []
    if legacy_clip_dirs:
        for d in legacy_clip_dirs:
            videos = sorted((d / "video").glob("*.mp4"))
            annos = sorted((d / "annotation").glob("*.csv"))
            if not videos or not annos:
                continue
            entries.append(
                {
                    "clip_name": d.name,
                    "video_path": videos[0],
                    "anno_path": annos[0],
                    "user_id": 0,
                    "source": "legacy_full_clips",
                }
            )
        return entries

    # Case 2: project data-root style
    for source in DATASET_SOURCES:
        for clip_root_rel in source["clip_roots"]:
            clip_root = clips_root / clip_root_rel
            if not clip_root.is_dir():
                continue
            clip_dirs = sorted(
                d for d in clip_root.iterdir()
                if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
            )
            for d in clip_dirs:
                videos = sorted((d / "video").glob("*.mp4"))
                annos = sorted((d / "annotation").glob("*.csv"))
                if not videos or not annos:
                    continue
                entries.append(
                    {
                        "clip_name": f"{source['label']}__{d.name}",
                        "video_path": videos[0],
                        "anno_path": annos[0],
                        "user_id": source["user_id"],
                        "source": source["label"],
                    }
                )

        for song_root_rel, gt_root_rel in zip(source["song_roots"], source["gt_roots"]):
            song_root = clips_root / song_root_rel
            gt_root = clips_root / gt_root_rel
            if not song_root.is_dir() or not gt_root.is_dir():
                continue
            for video_path in sorted(song_root.glob("*.mp4")):
                stem = video_path.stem
                anno_path = gt_root / f"gt_{stem}.csv"
                if not anno_path.exists():
                    continue
                entries.append(
                    {
                        "clip_name": f"{source['label']}__{stem}",
                        "video_path": video_path,
                        "anno_path": anno_path,
                        "user_id": source["user_id"],
                        "source": source["label"],
                    }
                )

    return entries


# ── torchaudio transforms (created once, reused) ────────────────────────────

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, power=2.0,
)
_amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
_mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SR, n_mfcc=N_MFCC,
    melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
)


# ── audio loading ────────────────────────────────────────────────────────────

def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load audio from file, resample to target sr, return mono float32 numpy."""
    waveform, orig_sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    return waveform.squeeze(0).numpy()


# ── feature extraction ───────────────────────────────────────────────────────


def _spectral_contrast(
    S: np.ndarray, sr: int, n_bands: int = 6, fmin: float = 200.0,
) -> np.ndarray:
    """Compute spectral contrast from a magnitude spectrogram (n_fft//2+1, T).

    Splits the spectrum into `n_bands` sub-bands (log-spaced from fmin to sr/2)
    and returns the dB difference between peaks and valleys per band, plus a
    "valley" summary band → shape (n_bands+1, T) = (7, T) by default.
    """
    n_freq, n_frames = S.shape
    freqs = np.linspace(0, sr / 2, n_freq)

    # Band edges: log-spaced from fmin to sr/2
    edges = np.concatenate(
        [[fmin], np.exp(np.linspace(np.log(fmin), np.log(sr / 2), n_bands + 1)[1:])]
    )
    contrast = np.zeros((n_bands + 1, n_frames), dtype=np.float32)

    alpha = 0.02  # proportion of bins for peak/valley
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() < 2:
            continue
        band = S[mask]  # (n_bins_in_band, T)
        sorted_band = np.sort(band, axis=0)
        k = max(1, int(alpha * band.shape[0]))
        valley = sorted_band[:k].mean(axis=0)
        peak = sorted_band[-k:].mean(axis=0)
        contrast[b] = np.log1p(peak) - np.log1p(valley)
    # last row: mean valley across all bands
    contrast[n_bands] = np.log1p(S.mean(axis=0))
    return contrast


def extract_features(y: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray]:
    """Extract the 5 real-time features from a 1s audio chunk."""
    y_t = torch.from_numpy(y).float().unsqueeze(0)  # (1, samples)

    # Mel spectrogram → dB
    mel_power = _mel_transform(y_t)        # (1, n_mels, T)
    mel_db = _amp_to_db(mel_power)          # (1, n_mels, T)
    mel_db_np = mel_db.squeeze(0).numpy()   # (128, T)

    # MFCC
    mfcc = _mfcc_transform(y_t)             # (1, n_mfcc, T)
    mfcc_np = mfcc.squeeze(0).numpy()       # (13, T)

    # Magnitude spectrogram (for RMS, contrast)
    spec = torch.stft(
        y_t.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window=torch.hann_window(N_FFT),
        return_complex=True,
    )  # (n_fft//2+1, T)
    S_mag = spec.abs().numpy()  # (n_fft//2+1, T)

    # RMS energy from magnitude spectrogram
    rms = np.sqrt((S_mag ** 2).mean(axis=0, keepdims=True)).astype(np.float32)  # (1, T)

    # Zero crossing rate  (frame-based, matching librosa convention)
    sign_changes = np.abs(np.diff(np.signbit(y).astype(np.float32)))
    pad_width = N_FFT // 2
    padded = np.pad(sign_changes, (pad_width, pad_width), mode="constant")
    n_frames = S_mag.shape[1]
    zcr = np.zeros((1, n_frames), dtype=np.float32)
    for f in range(n_frames):
        start = f * HOP_LENGTH
        end = start + N_FFT
        if end <= len(padded):
            zcr[0, f] = padded[start:end].mean()
        else:
            zcr[0, f] = padded[start:].mean() if start < len(padded) else 0.0

    # Spectral contrast
    contrast = _spectral_contrast(S_mag, sr=sr)  # (7, T)

    return {
        "mel_spectrogram": mel_db_np,      # (128, T)
        "mfcc": mfcc_np,                   # (13, T)
        "rms_energy": rms,                 # (1, T)
        "zero_crossing_rate": zcr,         # (1, T)
        "spectral_contrast": contrast,     # (7, T)
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

        # Discover clips from supported dataset layouts
        clip_entries = discover_training_clips(self.clips_root)
        if not clip_entries:
            raise ValueError(
                "No valid clips found. Expected either:\n"
                "1) legacy full_clips layout under clips_root, or\n"
                "2) data-root layout with bundled dataset folders (for example test_dataset/full_clips and test_set_*/songs+gt).\n"
                f"clips_root={self.clips_root}"
            )

        # Pre-extract features and targets
        self._features: List[Dict[str, np.ndarray]] = []
        self._targets: List[np.ndarray] = []
        self._meta: List[Tuple[str, int, float, float, int]] = []
        self._clip_names: List[str] = []  # unique clip names in order
        self._user_ids: List[int] = []    # user_id per window

        hop_samples = int(hop_sec * sr)

        print(f"Pre-extracting features from {len(clip_entries)} clips (hop={hop_sec}s)...")
        for ci, entry in enumerate(clip_entries):
            clip_name = entry["clip_name"]
            video_path = entry["video_path"]
            anno_path = entry["anno_path"]
            user_id = int(entry["user_id"])

            y_full = load_audio(str(video_path), sr=sr)
            anno_t, angles = load_annotation_csv(anno_path)

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
                self._meta.append((clip_name, w_idx, start_sec, end_sec, user_id))
                self._user_ids.append(user_id)

            print(
                f"  [{ci + 1}/{len(clip_entries)}] {clip_name}: {n_windows} windows "
                f"(user_id={user_id})"
            )

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
        clip_name, w_idx, start_sec, end_sec, user_id = self._meta[index]

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
            "user_id": user_id,
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
    user_id_list = [item["user_id"] for item in batch]

    x_out = {}
    for k in x_list[0].keys():
        x_out[k] = torch.stack([d[k] for d in x_list], dim=0)

    y_out = torch.stack(y_list, dim=0)
    user_id_out = torch.tensor(user_id_list, dtype=torch.long)
    return {
        "x": x_out,
        "y": y_out,
        "user_id": user_id_out,
        "clip_name": [item["clip_name"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
    }


# ── quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clips_root = Path(__file__).resolve().parent.parent / "data"
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
