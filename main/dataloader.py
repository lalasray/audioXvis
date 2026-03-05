from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
	import torch
	import torch.nn.functional as F
	from torch.utils.data import DataLoader, Dataset
	TORCH_AVAILABLE = True
except ImportError:
	torch = None
	F = None
	DataLoader = None
	Dataset = object
	TORCH_AVAILABLE = False


ANGLE_COLUMNS = ("angle_a_deg", "angle_b_deg", "angle_c_deg")


@dataclass(frozen=True)
class SampleRecord:
	feature_path: Path
	annotation_path: Path
	clip_id: str
	group_id: str


def _load_angles_csv(annotation_path: Path, angle_columns: Sequence[str]) -> np.ndarray:
	rows: List[List[float]] = []
	with annotation_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		missing = [c for c in angle_columns if c not in (reader.fieldnames or [])]
		if missing:
			raise ValueError(f"Missing columns {missing} in {annotation_path}")
		for row in reader:
			rows.append([float(row[c]) for c in angle_columns])
	if not rows:
		raise ValueError(f"No rows in annotation file: {annotation_path}")
	return np.asarray(rows, dtype=np.float32)


def _time_stats(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 1:
		return np.asarray([
			float(np.mean(arr)),
			float(np.std(arr)),
			float(np.min(arr)),
			float(np.max(arr)),
		], dtype=np.float32)

	time_axis = arr.ndim - 1
	mean = np.mean(arr, axis=time_axis)
	std = np.std(arr, axis=time_axis)
	minv = np.min(arr, axis=time_axis)
	maxv = np.max(arr, axis=time_axis)
	stacked = np.stack([mean, std, minv, maxv], axis=-1)
	return stacked.reshape(-1).astype(np.float32)


class AudioAnglesDataset(Dataset):
	def __init__(
		self,
		root_dir: str | Path,
		selected_features: Optional[Sequence[str]] = None,
		feature_mode: str = "dict",
		target_mode: str = "mean",
		angle_columns: Sequence[str] = ANGLE_COLUMNS,
		transform: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
	) -> None:
		self.root_dir = Path(root_dir)
		self.feature_mode = feature_mode
		self.target_mode = target_mode
		self.angle_columns = tuple(angle_columns)
		self.transform = transform

		if self.feature_mode not in {"dict", "concat", "stats"}:
			raise ValueError("feature_mode must be one of: dict, concat, stats")
		if self.target_mode not in {"mean", "last", "framewise"}:
			raise ValueError("target_mode must be one of: mean, last, framewise")

		self.samples = self._build_index(self.root_dir)
		if not self.samples:
			raise ValueError(f"No paired samples found under: {self.root_dir}")

		self.selected_features = tuple(selected_features) if selected_features else None

	@staticmethod
	def _build_index(root_dir: Path) -> List[SampleRecord]:
		records: List[SampleRecord] = []
		for feature_path in sorted(root_dir.glob("**/audio_features/*_features.npz")):
			group_dir = feature_path.parent.parent
			clip_name = feature_path.name.replace("_features.npz", "")
			annotation_path = group_dir / "annotation" / f"{clip_name}.csv"
			if annotation_path.exists():
				records.append(
					SampleRecord(
						feature_path=feature_path,
						annotation_path=annotation_path,
						clip_id=clip_name,
						group_id=group_dir.name,
					)
				)
		return records

	def __len__(self) -> int:
		return len(self.samples)

	def _select_feature_dict(self, npz_obj: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
		available = list(npz_obj.files)
		if self.selected_features is None:
			keys = sorted(available)
		else:
			missing = [k for k in self.selected_features if k not in available]
			if missing:
				raise KeyError(f"Requested features not found in npz: {missing}")
			keys = list(self.selected_features)

		feature_dict: Dict[str, np.ndarray] = {}
		for key in keys:
			feature_dict[key] = np.asarray(npz_obj[key], dtype=np.float32)
		return feature_dict

	def _pack_features(self, feature_dict: Dict[str, np.ndarray]):
		if self.feature_mode == "dict":
			if TORCH_AVAILABLE:
				return {k: torch.from_numpy(v) for k, v in feature_dict.items()}
			return feature_dict

		vectors: List[np.ndarray] = []
		for key in sorted(feature_dict.keys()):
			arr = feature_dict[key]
			if self.feature_mode == "concat":
				vectors.append(arr.reshape(-1).astype(np.float32))
			else:
				vectors.append(_time_stats(arr))

		packed = np.concatenate(vectors, axis=0).astype(np.float32)
		if TORCH_AVAILABLE:
			return torch.from_numpy(packed)
		return packed

	def _pack_target(self, angles: np.ndarray):
		if self.target_mode == "framewise":
			return torch.from_numpy(angles.astype(np.float32)) if TORCH_AVAILABLE else angles.astype(np.float32)
		if self.target_mode == "last":
			out = angles[-1].astype(np.float32)
			return torch.from_numpy(out) if TORCH_AVAILABLE else out
		out = np.mean(angles, axis=0).astype(np.float32)
		return torch.from_numpy(out) if TORCH_AVAILABLE else out

	def __getitem__(self, index: int):
		sample = self.samples[index]

		with np.load(sample.feature_path) as npz_obj:
			feature_dict = self._select_feature_dict(npz_obj)

		if self.transform is not None:
			feature_dict = self.transform(feature_dict)

		angles = _load_angles_csv(sample.annotation_path, self.angle_columns)

		return {
			"x": self._pack_features(feature_dict),
			"y": self._pack_target(angles),
			"clip_id": sample.clip_id,
			"group_id": sample.group_id,
			"feature_path": str(sample.feature_path),
			"annotation_path": str(sample.annotation_path),
		}


def dict_feature_collate(batch: List[Dict]) -> Dict:
	x_list = [item["x"] for item in batch]
	y_list = [item["y"] for item in batch]

	if isinstance(x_list[0], dict):
		x_out = {}
		for k in x_list[0].keys():
			vals = [d[k] for d in x_list]
			if TORCH_AVAILABLE and isinstance(vals[0], torch.Tensor):
				if vals[0].ndim == 1:
					max_t = max(v.shape[0] for v in vals)
					padded = []
					for v in vals:
						if v.shape[0] < max_t:
							v = F.pad(v, (0, max_t - v.shape[0]))
						padded.append(v)
					x_out[k] = torch.stack(padded, dim=0)
				elif vals[0].ndim == 2:
					max_t = max(v.shape[1] for v in vals)
					padded = []
					for v in vals:
						if v.shape[1] < max_t:
							v = F.pad(v, (0, max_t - v.shape[1]))
						padded.append(v)
					x_out[k] = torch.stack(padded, dim=0)
				else:
					raise ValueError(f"Unsupported feature ndim for key {k}: {vals[0].ndim}")
			else:
				if vals[0].ndim == 1:
					max_t = max(v.shape[0] for v in vals)
					padded = [np.pad(v, (0, max_t - v.shape[0])) if v.shape[0] < max_t else v for v in vals]
					x_out[k] = np.stack(padded, axis=0)
				elif vals[0].ndim == 2:
					max_t = max(v.shape[1] for v in vals)
					padded = [np.pad(v, ((0, 0), (0, max_t - v.shape[1]))) if v.shape[1] < max_t else v for v in vals]
					x_out[k] = np.stack(padded, axis=0)
				else:
					raise ValueError(f"Unsupported feature ndim for key {k}: {vals[0].ndim}")
	else:
		x_out = torch.stack(x_list, dim=0) if TORCH_AVAILABLE else np.stack(x_list, axis=0)

	y_ndim = y_list[0].ndim
	if y_ndim == 1:
		y_out = torch.stack(y_list, dim=0) if TORCH_AVAILABLE else np.stack(y_list, axis=0)
	else:
		y_out = y_list

	return {
		"x": x_out,
		"y": y_out,
		"clip_id": [item["clip_id"] for item in batch],
		"group_id": [item["group_id"] for item in batch],
		"feature_path": [item["feature_path"] for item in batch],
		"annotation_path": [item["annotation_path"] for item in batch],
	}


class NumpyDataLoader:
	def __init__(self, dataset: AudioAnglesDataset, batch_size: int, shuffle: bool):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __iter__(self):
		indices = list(range(len(self.dataset)))
		if self.shuffle:
			random.shuffle(indices)
		for start in range(0, len(indices), self.batch_size):
			batch_indices = indices[start:start + self.batch_size]
			batch = [self.dataset[i] for i in batch_indices]
			yield dict_feature_collate(batch)

	def __len__(self):
		return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_audio_angles_dataloader(
	root_dir: str | Path,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 0,
	selected_features: Optional[Sequence[str]] = None,
	feature_mode: str = "dict",
	target_mode: str = "mean",
) -> Tuple[AudioAnglesDataset, object]:
	dataset = AudioAnglesDataset(
		root_dir=root_dir,
		selected_features=selected_features,
		feature_mode=feature_mode,
		target_mode=target_mode,
	)

	if TORCH_AVAILABLE:
		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			collate_fn=dict_feature_collate,
		)
	else:
		dataloader = NumpyDataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
	return dataset, dataloader


if __name__ == "__main__":
	root = "/home/lala/Documents/GitHub/audioXvis/data/test_dataset/sliding_1s_hop_0p1_all"
	dataset, loader = create_audio_angles_dataloader(
		root_dir=root,
		batch_size=4,
		shuffle=True,
		selected_features=["mel_spectrogram", "mfcc", "pitch", "rms_energy"],
		feature_mode="stats",
		target_mode="mean",
	)
	print(f"Samples: {len(dataset)}")
	first = next(iter(loader))
	if TORCH_AVAILABLE and isinstance(first["x"], torch.Tensor):
		print(f"Batch x shape: {tuple(first['x'].shape)}")
	elif isinstance(first["x"], np.ndarray):
		print(f"Batch x shape: {first['x'].shape}")
	if TORCH_AVAILABLE and isinstance(first["y"], torch.Tensor):
		print(f"Batch y shape: {tuple(first['y'].shape)}")
	elif isinstance(first["y"], np.ndarray):
		print(f"Batch y shape: {first['y'].shape}")
	else:
		print("Batch y shape: framewise-list")
