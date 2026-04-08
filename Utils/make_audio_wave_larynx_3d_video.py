from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "main"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/audio2vis-mpl")
sys.path.append(str(MAIN_DIR))

from dataloader_fullclip import SR, load_audio  # noqa: E402
from infer_fullclip import infer_full_clip, load_model  # noqa: E402


ANGLE_COLORS = {
    "A": (235, 99, 74),
    "B": (65, 166, 246),
    "C": (83, 210, 128),
}


def load_audio_any(path: Path, sr: int = SR) -> np.ndarray:
    try:
        return load_audio(path, sr=sr).astype(np.float32)
    except Exception:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)


def save_predictions_csv(path: Path, times: np.ndarray, angles: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time", "angle_a", "angle_b", "angle_c"])
        for idx, (time_sec, row) in enumerate(zip(times, angles)):
            writer.writerow([idx, float(time_sec), float(row[0]), float(row[1]), float(row[2])])


def triangle_points_from_angles(angles_deg: np.ndarray) -> np.ndarray:
    a, b, c = np.radians(np.clip(angles_deg, 1.0, 178.0))
    side_ab = math.sin(c)
    side_ac = math.sin(b)
    pts = np.array(
        [
            [0.0, 0.0],
            [side_ab, 0.0],
            [side_ac * math.cos(a), side_ac * math.sin(a)],
        ],
        dtype=np.float32,
    )
    pts -= pts.mean(axis=0, keepdims=True)
    pts /= max(np.linalg.norm(pts, axis=1).max(), 1e-6)
    return pts


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ rx @ ry


def project(points_3d: np.ndarray, width: int, height: int, scale: float = 235.0) -> np.ndarray:
    z = points_3d[:, 2] + 4.0
    x = points_3d[:, 0] / z * scale + width * 0.5
    y = -points_3d[:, 1] / z * scale + height * 0.43
    return np.stack([x, y], axis=1).astype(np.int32)


def arc_points(vertex: np.ndarray, ray_1: np.ndarray, ray_2: np.ndarray, radius: float, steps: int = 36) -> np.ndarray:
    t1 = math.atan2(ray_1[1], ray_1[0])
    t2 = math.atan2(ray_2[1], ray_2[0])
    diff = (t2 - t1 + math.pi) % (2 * math.pi) - math.pi
    theta = t1 + np.linspace(0.0, diff, steps, dtype=np.float32)
    return vertex + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def draw_waveform(frame: np.ndarray, audio: np.ndarray, sr: int, now_sec: float, duration: float) -> None:
    height, width = frame.shape[:2]
    left, right = 90, width - 90
    top, bottom = height - 130, height - 46
    mid = (top + bottom) // 2
    cv2.rectangle(frame, (left, top), (right, bottom), (24, 28, 38), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), (82, 91, 111), 1, cv2.LINE_AA)

    samples = audio[: int(duration * sr)]
    bins = right - left
    if samples.size >= bins:
        idx = np.linspace(0, samples.size - 1, bins + 1).astype(np.int32)
        peaks = np.array([np.max(np.abs(samples[idx[i] : max(idx[i + 1], idx[i] + 1)])) for i in range(bins)])
    else:
        peaks = np.pad(np.abs(samples), (0, bins - samples.size))
    peaks = np.clip(peaks / max(float(np.percentile(peaks, 98)), 1e-4), 0.0, 1.0)
    for offset, amp in enumerate(peaks):
        x = left + offset
        y = int(amp * (bottom - top) * 0.45)
        color = (93, 201, 255) if x <= left + int((now_sec / duration) * bins) else (72, 86, 113)
        cv2.line(frame, (x, mid - y), (x, mid + y), color, 1, cv2.LINE_AA)
    play_x = left + int(np.clip(now_sec / max(duration, 1e-6), 0.0, 1.0) * bins)
    cv2.line(frame, (play_x, top - 10), (play_x, bottom + 10), (255, 255, 255), 2, cv2.LINE_AA)


def put_text(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    angles: np.ndarray,
    audio: np.ndarray,
    sr: int,
    now_sec: float,
    duration: float,
) -> None:
    height, width = frame.shape[:2]
    frame[:] = (13, 16, 24)
    for y in range(height):
        frame[y, :, :] = np.clip(frame[y, :, :] + int(22 * (1.0 - y / height)), 0, 255)

    pts_2d = triangle_points_from_angles(angles)
    z = ((angles - 60.0) / 90.0).astype(np.float32) * 0.45
    pts_3d = np.column_stack([pts_2d[:, 0] * 1.65, pts_2d[:, 1] * 1.65, z])
    rot = rotation_matrix(yaw=0.58 + 0.12 * math.sin(now_sec * 0.45), pitch=-0.55, roll=0.08 * math.sin(now_sec * 0.7))
    pts_proj = project(pts_3d @ rot.T, width, height)

    center_3d = np.zeros((1, 3), dtype=np.float32)
    center_proj = project(center_3d @ rot.T, width, height)[0]
    shadow = pts_proj + np.array([0, 28], dtype=np.int32)
    cv2.fillConvexPoly(frame, shadow, (5, 7, 12), cv2.LINE_AA)
    cv2.polylines(frame, [pts_proj], True, (238, 236, 223), 4, cv2.LINE_AA)

    labels = ("A", "B", "C")
    for idx, label in enumerate(labels):
        color = ANGLE_COLORS[label]
        cv2.circle(frame, tuple(pts_proj[idx]), 8, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(frame, tuple(center_proj), tuple(pts_proj[idx]), color, 2, cv2.LINE_AA, tipLength=0.08)
        put_text(frame, label, tuple((pts_proj[idx] + np.array([12, -12])).tolist()), 0.75, color, 2)

    arcs = [
        ("A", 0, pts_2d[1] - pts_2d[0], pts_2d[2] - pts_2d[0]),
        ("B", 1, pts_2d[2] - pts_2d[1], pts_2d[0] - pts_2d[1]),
        ("C", 2, pts_2d[0] - pts_2d[2], pts_2d[1] - pts_2d[2]),
    ]
    for label, idx, ray_1, ray_2 in arcs:
        arc_2d = arc_points(pts_2d[idx], ray_1, ray_2, radius=0.33)
        arc_3d = np.column_stack([arc_2d[:, 0] * 1.65, arc_2d[:, 1] * 1.65, np.full(len(arc_2d), z[idx])])
        arc_proj = project(arc_3d @ rot.T, width, height)
        cv2.polylines(frame, [arc_proj], False, ANGLE_COLORS[label], 3, cv2.LINE_AA)

    put_text(frame, "Audio2Vis predicted larynx angles", (54, 58), 0.9, (245, 247, 250), 2)
    put_text(frame, f"{now_sec:04.1f}s / {duration:.0f}s", (width - 190, 58), 0.7, (194, 203, 218), 1)
    for i, label in enumerate(labels):
        y = 108 + i * 36
        put_text(frame, f"angle {label}: {angles[i]:5.1f} deg", (58, y), 0.72, ANGLE_COLORS[label], 2)

    draw_waveform(frame, audio, sr, now_sec, duration)


def mux_audio(video_path: Path, audio_path: Path, output_path: Path, duration: float) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 3D larynx-angle video driven by audio inference.")
    parser.add_argument("--audio", type=Path, default=REPO_ROOT / "data/test_set_3/audio/s01.mp3")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "main/checkpoints/diffusion_v2/best.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/audio2vis_larynx_angles_3d_15s.mp4")
    parser.add_argument("--pred_csv", type=Path, default=REPO_ROOT / "outputs/audio2vis_larynx_angles_3d_15s_predictions.csv")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--window_sec", type=float, default=0.5)
    parser.add_argument("--hop_sec", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pred_csv.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/audio2vis-mpl")

    audio = load_audio_any(args.audio, sr=SR)
    audio = audio[: int((args.duration + args.window_sec) * SR)]
    if audio.shape[0] < int(args.window_sec * SR):
        raise ValueError(f"Audio is too short for a {args.window_sec:.2f}s inference window: {args.audio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, y_mean, y_std, sample_steps, _cfg, feature_stats = load_model(str(args.ckpt), device)
    times, pred_angles = infer_full_clip(
        model=model,
        y_full=audio,
        sr=SR,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        y_mean=y_mean,
        y_std=y_std,
        sample_steps=sample_steps,
        device=device,
        feature_stats=feature_stats,
        batch_size=args.batch_size,
    )
    save_predictions_csv(args.pred_csv, times, pred_angles)

    temp_video = Path(tempfile.mkstemp(prefix="audio2vis_larynx_", suffix=".mp4", dir=str(args.output.parent))[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_video}")

    frame = np.empty((args.height, args.width, 3), dtype=np.uint8)
    total_frames = int(round(args.duration * args.fps))
    interp_t = np.maximum(times, 0.0)
    for frame_idx in range(total_frames):
        now = frame_idx / args.fps
        angles = np.array([np.interp(now, interp_t, pred_angles[:, col]) for col in range(3)], dtype=np.float32)
        draw_frame(frame, angles, audio, SR, now, args.duration)
        writer.write(frame)
        if frame_idx and frame_idx % args.fps == 0:
            print(f"Rendered {frame_idx // args.fps:02d}s / {int(args.duration)}s")
    writer.release()

    if not mux_audio(temp_video, args.audio, args.output, args.duration):
        temp_video.replace(args.output)
        print("ffmpeg audio mux failed; saved silent video instead.")
    else:
        temp_video.unlink(missing_ok=True)

    print(f"Saved video: {args.output}")
    print(f"Saved predictions: {args.pred_csv}")


if __name__ == "__main__":
    main()
