from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/audio2vis-mpl")
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "main"))

from Utils.make_audio_wave_larynx_3d_video import (  # noqa: E402
    ANGLE_COLORS,
    draw_waveform,
    load_audio_any,
    mux_audio,
    put_text,
    save_predictions_csv,
)
from dataloader_fullclip import SR  # noqa: E402
from infer_fullclip import infer_full_clip, load_model  # noqa: E402


def natural_sort_key(path_str: str) -> list[int | str]:
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path_str)]


def map_angle_to_frame(angle_val: float, frame_count: int) -> float:
    min_deg = 0.0
    neutral_deg = 60.0
    max_deg = 90.0
    min_frame = 0.0
    neutral_frame = float((frame_count - 1) / 2.0)
    max_frame = float(frame_count - 1)
    if angle_val <= neutral_deg:
        t = np.clip((angle_val - min_deg) / max(neutral_deg - min_deg, 1e-6), 0.0, 1.0)
        return min_frame + float(t) * (neutral_frame - min_frame)
    t = np.clip((angle_val - neutral_deg) / max(max_deg - neutral_deg, 1e-6), 0.0, 1.0)
    return neutral_frame + float(t) * (max_frame - neutral_frame)


def exaggerate_pose(frame_float: float, frame_count: int, multiplier: float) -> float:
    neutral = float((frame_count - 1) / 2.0)
    return float(np.clip(neutral + (frame_float - neutral) * multiplier, 0.0, frame_count - 1))


def parse_mtl(mtl_path: Path | None) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    if mtl_path is None or not mtl_path.exists():
        return colors
    current = None
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1]
            elif current and line.startswith("Kd "):
                parts = line.split()
                rgb = tuple(int(np.clip(float(v) * 255.0, 0, 255)) for v in parts[1:4])
                colors[current] = (rgb[2], rgb[1], rgb[0])
    return colors


def parse_obj(obj_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], Path | None]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    materials: list[str] = []
    current_material = "default"
    mtllib = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                mtllib = (obj_path.parent / line.split(maxsplit=1)[1]).resolve()
            elif line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1]
            elif line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                refs = [part.split("/")[0] for part in line.split()[1:]]
                idx = [int(ref) - 1 for ref in refs if ref]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
                    materials.append(current_material)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), materials, mtllib


def load_mesh_sequence(mesh_glob: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, float]:
    paths = sorted((Path(p) for p in glob.glob(mesh_glob)), key=lambda p: natural_sort_key(str(p)))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 OBJ frames for model rendering: {mesh_glob}")

    base_vertices, faces, materials, mtllib = parse_obj(paths[0])
    mtl_colors = parse_mtl(mtllib)
    face_colors = [mtl_colors.get(name, (130, 157, 214)) for name in materials]
    points = []
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for path in paths:
        vertices, frame_faces, _frame_materials, _ = parse_obj(path)
        if vertices.shape != base_vertices.shape or frame_faces.shape != faces.shape:
            raise ValueError("Baked mesh sequence does not share topology, cannot blend frames.")
        points.append(vertices)
        mins = np.minimum(mins, vertices.min(axis=0))
        maxs = np.maximum(maxs, vertices.max(axis=0))

    center = (mins + maxs) * 0.5
    scale = float(np.maximum((maxs - mins).max(), 1e-6))
    print(f"Loaded {len(points)} mesh frames, {base_vertices.shape[0]} points, {faces.shape[0]} faces")
    return faces, np.asarray(face_colors, dtype=np.uint8), points, center, scale


def blended_points(points: list[np.ndarray], frame_float: float) -> np.ndarray:
    hi_max = len(points) - 1
    frame_float = float(np.clip(frame_float, 0.0, hi_max))
    lo = int(np.floor(frame_float))
    hi = min(lo + 1, hi_max)
    t = frame_float - lo
    if hi == lo or t <= 1e-6:
        return points[lo]
    return ((1.0 - t) * points[lo] + t * points[hi]).astype(np.float32)


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ rx @ ry


def render_model(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    center: np.ndarray,
    scale: float,
    width: int,
    height: int,
    now_sec: float,
) -> np.ndarray:
    frame = np.full((height, width, 3), (13, 17, 27), dtype=np.uint8)
    for y in range(height):
        frame[y, :, :] = np.clip(frame[y, :, :] + int(24 * (1.0 - y / height)), 0, 255)

    normalized = (vertices - center) / scale
    coords = np.column_stack([normalized[:, 0] * 2.9, normalized[:, 1] * 2.9, normalized[:, 2] * 2.9])
    rot = rotation_matrix(yaw=0.28 + 0.06 * np.sin(now_sec * 0.35), pitch=-0.03, roll=0.0)
    pts = coords @ rot.T
    projected = np.empty((pts.shape[0], 2), dtype=np.int32)
    model_px = min(width, height) * 0.25
    projected[:, 0] = (pts[:, 0] * model_px + width * 0.50).astype(np.int32)
    projected[:, 1] = (-pts[:, 1] * model_px + height * 0.45).astype(np.int32)

    tris_3d = pts[faces]
    normals = np.cross(tris_3d[:, 1] - tris_3d[:, 0], tris_3d[:, 2] - tris_3d[:, 0])
    norm_len = np.maximum(np.linalg.norm(normals, axis=1), 1e-6)
    normals = normals / norm_len[:, None]
    light = np.array([-0.35, -0.55, 0.75], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = np.clip(np.abs(normals @ light), 0.0, 1.0) * 0.55 + 0.58
    depths = tris_3d[:, :, 2].mean(axis=1)
    order = np.argsort(depths)[::-1]
    tri_2d = projected[faces]

    for face_idx in order:
        poly = tri_2d[face_idx]
        if (
            poly[:, 0].max() < 0
            or poly[:, 0].min() >= width
            or poly[:, 1].max() < 0
            or poly[:, 1].min() >= height
        ):
            continue
        color = np.clip(face_colors[face_idx].astype(np.float32) * intensity[face_idx], 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(frame, poly, tuple(int(v) for v in color), cv2.LINE_AA)
    return frame


def annotate_frame(
    frame: np.ndarray,
    angles: np.ndarray,
    frame_float: float,
    audio: np.ndarray,
    now_sec: float,
    duration: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (34, 28), (438, 178), (9, 12, 20), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (frame.shape[1] - 250, 28), (frame.shape[1] - 36, 88), (9, 12, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    put_text(frame, "Audio2Vis predicted larynx model", (54, 62), 0.85, (245, 247, 250), 2)
    labels = ("A", "B", "C")
    for i, label in enumerate(labels):
        y = 104 + i * 30
        put_text(frame, f"angle {label}: {angles[i]:5.1f} deg", (58, y), 0.62, ANGLE_COLORS[label], 2)
    put_text(frame, f"mesh pose {frame_float:04.1f}", (270, 164), 0.55, (185, 198, 218), 1)
    put_text(frame, f"{now_sec:04.1f}s / {duration:.0f}s", (frame.shape[1] - 220, 64), 0.68, (222, 228, 238), 1)
    draw_waveform(frame, audio, SR, now_sec, duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the baked 3D larynx model driven by audio inference.")
    parser.add_argument("--audio", type=Path, default=REPO_ROOT / "data/test_set_3/audio/s01.mp3")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "main/checkpoints/diffusion_v2/best.pt")
    parser.add_argument("--mesh_glob", default=str(REPO_ROOT / "main/_fbx_baked/fbx_frame_*.obj"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/audio2vis_larynx_model_3d_15s.mp4")
    parser.add_argument("--pred_csv", type=Path, default=REPO_ROOT / "outputs/audio2vis_larynx_model_3d_15s_predictions.csv")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--window_sec", type=float, default=0.5)
    parser.add_argument("--hop_sec", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--motion_multiplier", type=float, default=4.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pred_csv.parent.mkdir(parents=True, exist_ok=True)

    audio = load_audio_any(args.audio, sr=SR)
    audio = audio[: int((args.duration + args.window_sec) * SR)]
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

    faces, face_colors, mesh_points, mesh_center, mesh_scale = load_mesh_sequence(args.mesh_glob)

    temp_video = Path(tempfile.mkstemp(prefix="audio2vis_model_", suffix=".mp4", dir=str(args.output.parent))[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_video}")

    total_frames = int(round(args.duration * args.fps))
    interp_t = np.maximum(times, 0.0)
    for frame_idx in range(total_frames):
        now = frame_idx / args.fps
        angles = np.array([np.interp(now, interp_t, pred_angles[:, col]) for col in range(3)], dtype=np.float32)
        driver = float((angles[0] + angles[1]) * 0.5)
        pose = map_angle_to_frame(driver, len(mesh_points))
        model_pose = exaggerate_pose(pose, len(mesh_points), args.motion_multiplier)
        vertices = blended_points(mesh_points, model_pose)
        frame = render_model(vertices, faces, face_colors, mesh_center, mesh_scale, args.width, args.height, now)
        annotate_frame(frame, angles, pose, audio, now, args.duration)
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
