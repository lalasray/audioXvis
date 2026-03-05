"""
Dataloader for full-clip 1-second rolling-window training.

Takes full clip videos (with audio) and annotations, applies 1s rolling windows
on both, extracts 5 audio features on-the-fly, and returns (features, angles) pairs.

Features (all frame-local, safe for 1s windows):
  - mel_spectrogram  (128 × T)
  - rms_energy       (1 × T)
  - zero_crossing_rate (1 × T)
  - spectral_contrast (7 × T)
  - mfcc             (13 × T)

Targets: angle_a_deg, angle_c_deg  (predict 2, derive angle_b = 180 - a - c)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── constants ────────────────────────────────────────────────────────────────

SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 13
VIDEO_FPS = 60.0

PREDICT_COLUMNS = ("angle_a_deg", "angle_c_deg")   # most correlated pair
ALL_ANGLE_COLUMNS = ("angle_a_deg", "angle_b_deg", "angle_c_deg")


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
    angles = np.stack([aa, ab, ac], axis=1).astype(np.float32)  # (N, 3)
    return t, angles


# ── feature extraction ───────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray]:
    """Extract the 5 recommended real-time features from a 1s audio chunk."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr,
                                 n_mfcc=N_MFCC).astype(np.float32)

    rms = librosa.feature.rms(S=S, frame_length=N_FFT,
                               hop_length=HOP_LENGTH).astype(np.float32)  # (1, T)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT,
                                              hop_length=HOP_LENGTH).astype(np.float32)  # (1, T)

    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=N_FFT,
                                                  hop_length=HOP_LENGTH).astype(np.float32)  # (7, T)

    return {
        "mel_spectrogram": mel_db,       # (128, T)
        "mfcc": mfcc,                    # (13, T)
        "rms_energy": rms,               # (1, T)
        "zero_crossing_rate": zcr,       # (1, T)
        "spectral_contrast": contrast,   # (7, T)
    }


# ── dataset ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClipRecord:
    video_path: Path
    annotation_path: Path
    clip_name: str


class FullClipRollingDataset(Dataset):
    """
    Lazily loads full-clip audio + annotation, generates 1s rolling windows.

    At init time: discovers clips, loads audio and annotations, pre-computes
    the window indices. At __getitem__ time: slices audio, extracts features.
    """

    def __init__(
        self,
        clips_root: str | Path,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
        sr: int = SR,
        target_columns: Sequence[str] = PREDICT_COLUMNS,
        preload_audio: bool = True,
    ) -> None:
        self.clips_root = Path(clips_root)
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sr = sr
        self.target_columns = list(target_columns)
        self.window_samples = int(window_sec * sr)

        # Discover clip directories
        clip_dirs = sorted(
            d for d in self.clips_root.iterdir()
            if d.is_dir() and (d / "video").is_dir() and (d / "annotation").is_dir()
        )
        if not clip_dirs:
            raise ValueError(f"No clip directories found under {self.clips_root}")

        # Build index: for each clip, compute how many windows fit
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._anno_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._windows: List[Tuple[str, int, float, float]] = []  # (clip_name, window_idx, start_sec, end_sec)

        target_col_indices = []
        for col in self.target_columns:
            if col == "angle_a_deg":
                target_col_indices.append(0)
            elif col == "angle_b_deg":
                target_col_indices.append(1)
            elif col == "angle_c_deg":
                target_col_indices.append(2)
        self._target_col_indices = target_col_indices

        for clip_dir in clip_dirs:
            clip_name = clip_dir.name

            # Find video and annotation
            videos = sorted((clip_dir / "video").glob("*.mp4"))
            annos = sorted((clip_dir / "annotation").glob("*.csv"))
            if not videos or not annos:
                continue

            # Load audio
            y, _ = librosa.load(str(videos[0]), sr=sr)
            duration_sec = len(y) / sr

            # Load annotation
            anno_t, angles = load_annotation_csv(annos[0])

            if preload_audio:
                self._audio_cache[clip_name] = (y, sr)
            else:
                self._audio_cache[clip_name] = (videos[0], sr)  # store path

            self._anno_cache[clip_name] = (anno_t, angles)

            # Generate rolling windows
            hop_samples = int(hop_sec * sr)
            n_windows = max(0, (len(y) - self.window_samples) // hop_samples + 1)

            for w_idx in range(n_windows):
                start_sample = w_idx * hop_samples
                start_sec = start_sample / sr
                end_sec = start_sec + window_sec
                self._windows.append((clip_name, w_idx, start_sec, end_sec))

        if not self._windows:
            raise ValueError("No valid windows generated from clips.")

        print(f"FullClipRollingDataset: {len(clip_dirs)} clips, "
              f"{len(self._windows)} windows ({window_sec}s, hop {hop_sec}s)")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Dict:
        clip_name, w_idx, start_sec, end_sec = self._windows[index]

        # Get audio window
        y_full, sr = self._audio_cache[clip_name]
        if isinstance(y_full, Path):
            y_full, sr = librosa.load(str(y_full), sr=self.sr)

        start_sample = int(start_sec * sr)
        y_window = y_full[start_sample:start_sample + self.window_samples]

        # Pad if needed (shouldn't normally happen)
        if len(y_window) < self.window_samples:
            y_window = np.pad(y_window, (0, self.window_samples - len(y_window)))

        # Extract features
        features = extract_features(y_window, sr=sr)

        # Get target angles — mean of annotation frames within this window
        anno_t, angles = self._anno_cache[clip_name]
        in_window = (anno_t >= start_sec) & (anno_t < end_sec)
        if np.any(in_window):
            window_angles = angles[in_window]
            target = np.mean(window_angles[:, self._target_col_indices], axis=0).astype(np.float32)
        else:
            target = np.zeros(len(self._target_col_indices), dtype=np.float32)

        # Convert to tensors
        x = {k: torch.from_numpy(v) for k, v in features.items()}
        y = torch.from_numpy(target)

        return {
            "x": x,
            "y": y,
            "clip_name": clip_name,
            "window_idx": w_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }


def rolling_collate(batch: List[Dict]) -> Dict:
    """Collate function for dicts of feature tensors."""
    x_list = [item["x"] for item in batch]
    y_list = [item["y"] for item in batch]

    x_out = {}
    for k in x_list[0].keys():
        tensors = [d[k] for d in x_list]
        # All windows are same size so shapes match
        x_out[k] = torch.stack(tensors, dim=0)

    y_out = torch.stack(y_list, dim=0)

    return {
        "x": x_out,
        "y": y_out,
        "clip_name": [item["clip_name"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
    }


def create_rolling_dataloaders(
    clips_root: str | Path,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Tuple[int, ...]]]:
    """Create train/val dataloaders from full clips with rolling windows."""
    dataset = FullClipRollingDataset(
        clips_root=clips_root,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )

    total = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator)

    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    train_indices = perm[:train_size].tolist()
    val_indices = perm[train_size:].tolist()

    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rolling_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rolling_collate,
        pin_memory=True,
    )

    sample = dataset[0]["x"]
    feature_shapes = {k: tuple(v.shape) for k, v in sample.items()}
    return train_loader, val_loader, feature_shapes


if __name__ == "__main__":
    clips_root = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"
    dataset = FullClipRollingDataset(clips_root, window_sec=1.0, hop_sec=0.5)
    print(f"Total windows: {len(dataset)}")
    sample = dataset[0]
    print(f"Features:")
    for k, v in sample["x"].items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"Target (angle_a, angle_c): {sample['y']}")
    print(f"Clip: {sample['clip_name']}, window {sample['window_idx']}, "
          f"t=[{sample['start_sec']:.2f}, {sample['end_sec']:.2f}]s")
