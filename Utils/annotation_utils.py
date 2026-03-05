"""Shared annotation helpers for 5-point ultrasound landmark CSVs."""

import numpy as np


def load_points_csv(csv_path: str) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",")
    points = np.atleast_2d(points)
    if points.shape[1] != 10:
        raise ValueError(
            f"Annotation CSV must have 10 columns (5 x,y points). Got {points.shape[1]} columns."
        )
    return points.reshape(-1, 5, 2).astype(np.float32)


def interpolate_points_to_frame_count(points: np.ndarray, target_frames: int) -> np.ndarray:
    n_frames, n_pts, _ = points.shape
    if target_frames <= 1:
        return points[:1].copy()

    t_src = np.arange(n_frames, dtype=np.float32)
    t_dst = np.linspace(0.0, n_frames - 1, target_frames, dtype=np.float32)

    dense = np.zeros((target_frames, n_pts, 2), dtype=np.float32)

    for p in range(n_pts):
        valid = ~np.isclose(points[:, p, 0], 0.0) | ~np.isclose(points[:, p, 1], 0.0)
        if not np.any(valid):
            continue

        src_valid_t = t_src[valid]

        for axis in range(2):
            signal = points[:, p, axis]
            src_valid_v = signal[valid]

            filled = np.interp(t_src, src_valid_t, src_valid_v)
            interp = np.interp(t_dst, t_src, filled)

            interp[t_dst < src_valid_t[0]] = src_valid_v[0]
            interp[t_dst > src_valid_t[-1]] = src_valid_v[-1]

            dense[:, p, axis] = interp

    return dense
