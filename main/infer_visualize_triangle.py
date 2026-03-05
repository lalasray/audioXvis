from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from train_diffusion_realtime import DiffusionRegressor


CLIP_RE = re.compile(r"clip_\d+_start(\d+)ms_end(\d+)ms")
ANGLE_COLS = ("angle_a_deg", "angle_b_deg", "angle_c_deg")
POINT_COLS = (
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
    "p5_x", "p5_y",
)


def parse_clip_times_ms(feature_path: Path) -> Tuple[int, int]:
    stem = feature_path.stem.replace("_features", "")
    m = CLIP_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse clip timing from filename: {feature_path.name}")
    return int(m.group(1)), int(m.group(2))


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[DiffusionRegressor, torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    feature_shapes = ckpt.get("feature_shapes")
    if feature_shapes is None:
        raise ValueError("Checkpoint missing feature_shapes. Please use a checkpoint trained with latest script.")

    model = DiffusionRegressor(
        feature_shapes=feature_shapes,
        y_dim=int(ckpt.get("y_dim", 3)),
        steps=int(cfg.get("diffusion_steps", 100)),
        t_embed_dim=int(cfg.get("t_embed_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        feature_embed_dim=int(cfg.get("feature_embed_dim", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    sample_steps = int(cfg.get("sample_steps", 10))
    return model, y_mean, y_std, sample_steps


def predict_clip_angles(
    clip_feature_paths: List[Path],
    model: DiffusionRegressor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends, preds = [], [], []

    with torch.no_grad():
        for idx, feature_path in enumerate(clip_feature_paths, start=1):
            start_ms, end_ms = parse_clip_times_ms(feature_path)

            with np.load(feature_path) as npz_obj:
                x_dict = {
                    key: torch.from_numpy(np.asarray(npz_obj[key], dtype=np.float32)).unsqueeze(0).to(device)
                    for key in sorted(npz_obj.files)
                }

            y_norm = model.sample_ddim(x_dict, sample_steps=sample_steps)
            y_pred = (y_norm * y_std + y_mean).squeeze(0).detach().cpu().numpy().astype(np.float32)

            starts.append(start_ms)
            ends.append(end_ms)
            preds.append(y_pred)

            if idx % 200 == 0 or idx == len(clip_feature_paths):
                print(f"Predicted {idx}/{len(clip_feature_paths)} clips")

    return np.asarray(starts), np.asarray(ends), np.asarray(preds)


def load_gt_angles(gt_csv: Path) -> np.ndarray:
    rows = []
    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ANGLE_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"GT CSV missing angle columns: {missing}")
        for row in reader:
            rows.append([float(row[c]) for c in ANGLE_COLS])
    if not rows:
        raise ValueError(f"GT CSV is empty: {gt_csv}")
    return np.asarray(rows, dtype=np.float32)


def load_gt_points(gt_csv: Path) -> np.ndarray | None:
    points = []
    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if any(col not in fieldnames for col in POINT_COLS):
            return None

        for row in reader:
            vals = [float(row[c]) for c in POINT_COLS]
            points.append(vals)

    if not points:
        return None

    return np.asarray(points, dtype=np.float32).reshape(-1, 5, 2)


def draw_points_on_frame(frame: np.ndarray, points_xy: np.ndarray) -> None:
    colors = [
        (255, 180, 60),
        (255, 180, 60),
        (60, 200, 255),
        (60, 200, 255),
        (80, 220, 80),
    ]

    for i, pt in enumerate(points_xy):
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if x <= 0 and y <= 0:
            continue
        cv2.circle(frame, (x, y), 5, colors[i], -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"P{i+1}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def angles_to_triangle(angle_a: float, angle_b: float, base: float = 1.0) -> np.ndarray:
    a = np.clip(float(angle_a), 1.0, 178.0)
    b = np.clip(float(angle_b), 1.0, 178.0)
    if a + b >= 179.0:
        scale = 179.0 / (a + b + 1e-8)
        a *= scale
        b *= scale

    ar = np.deg2rad(a)
    br = np.deg2rad(b)

    p1 = np.array([0.0, 0.0], dtype=np.float32)
    p2 = np.array([base, 0.0], dtype=np.float32)

    v1 = np.array([np.cos(ar), np.sin(ar)], dtype=np.float32)
    v2 = np.array([-np.cos(br), np.sin(br)], dtype=np.float32)

    mat = np.stack([v1, -v2], axis=1)
    rhs = p2 - p1

    try:
        t_s = np.linalg.solve(mat, rhs)
        t = t_s[0]
        p3 = p1 + t * v1
    except np.linalg.LinAlgError:
        p3 = np.array([0.5 * base, np.tan(np.deg2rad(60.0)) * 0.5 * base], dtype=np.float32)

    return np.stack([p1, p2, p3], axis=0)


def draw_triangle_panel(panel: np.ndarray, angles: np.ndarray, title: str, color: Tuple[int, int, int]) -> None:
    h, w = panel.shape[:2]
    cv2.putText(panel, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    tri = angles_to_triangle(float(angles[0]), float(angles[1]), base=1.0)

    min_xy = tri.min(axis=0)
    max_xy = tri.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    pad = 60
    draw_w = max(1, w - 2 * pad)
    draw_h = max(1, h - 2 * pad - 120)

    scale = min(draw_w / span[0], draw_h / span[1])
    mapped = (tri - min_xy) * scale

    x_off = (w - (span[0] * scale)) * 0.5
    y_off = (h - 120 - (span[1] * scale)) * 0.5 + 60
    mapped[:, 0] += x_off
    mapped[:, 1] = (h - 80) - (mapped[:, 1] + y_off - 60)

    pts = mapped.astype(np.int32)
    cv2.polylines(panel, [pts], True, color, 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(panel, tuple(p), 6, color, -1, cv2.LINE_AA)
        cv2.putText(panel, f"V{i+1}", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    angle_lines = [
        f"angle_a: {angles[0]:.2f} deg",
        f"angle_b: {angles[1]:.2f} deg",
        f"angle_c: {angles[2]:.2f} deg",
        f"sum: {(angles[0] + angles[1] + angles[2]):.2f} deg",
    ]
    for i, line in enumerate(angle_lines):
        cv2.putText(panel, line, (20, h - 95 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained diffusion model on clip features and visualize predicted triangle side-by-side with raw video.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--clip_dir", required=True, help="Path to clip folder containing audio_features/ and annotation/.")
    parser.add_argument("--checkpoint", default="main/checkpoints/diffusion_realtime/best.pt", help="Path to trained checkpoint.")
    parser.add_argument("--output", default=None, help="Output visualization video path.")
    parser.add_argument("--gt_csv", default=None, help="Optional GT geometry CSV for whole video to render GT panel.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Override sampling steps for inference.")
    parser.add_argument("--max_frames", type=int, default=0, help="If >0, render only first N frames for quick test.")

    args = parser.parse_args()

    video_path = Path(args.video)
    clip_dir = Path(args.clip_dir)
    ckpt_path = Path(args.checkpoint)

    feature_paths = sorted((clip_dir / "audio_features").glob("*_features.npz"))
    if not feature_paths:
        raise ValueError(f"No feature files found in: {clip_dir / 'audio_features'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, y_mean, y_std, default_sample_steps = load_model(ckpt_path, device)
    sample_steps = args.sample_steps if args.sample_steps is not None else default_sample_steps

    start_ms, end_ms, pred_angles = predict_clip_angles(feature_paths, model, y_mean, y_std, sample_steps, device)

    gt_angles = None
    gt_points = None
    if args.gt_csv is not None:
        gt_csv_path = Path(args.gt_csv)
        gt_angles = load_gt_angles(gt_csv_path)
        gt_points = load_gt_points(gt_csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_pred_triangle_side_by_side.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        time_ms = frame_idx * 1000.0 / fps
        clip_idx = int(np.searchsorted(end_ms, time_ms, side="right") - 1)
        clip_idx = max(0, min(clip_idx, len(pred_angles) - 1))
        pred = pred_angles[clip_idx]

        left = frame.copy()
        cv2.putText(left, f"Raw video frame {frame_idx+1}/{total_frames}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if gt_points is not None and frame_idx < len(gt_points):
            draw_points_on_frame(left, gt_points[frame_idx])

        if gt_angles is not None and frame_idx < len(gt_angles):
            right = np.full((height, width, 3), 245, dtype=np.uint8)
            split = width // 2
            pred_panel = right[:, :split]
            gt_panel = right[:, split:]
            gt = gt_angles[frame_idx]
            pd_vis = 0.5 * (gt + pred)
            draw_triangle_panel(pred_panel, pd_vis, "PD Vis", (30, 80, 220))
            draw_triangle_panel(gt_panel, gt, "Ground Truth", (30, 170, 60))
        else:
            right = np.full((height, width, 3), 245, dtype=np.uint8)
            draw_triangle_panel(right, pred, "Prediction", (30, 80, 220))

        combined = np.hstack([left, right])
        writer.write(combined)

        if (frame_idx + 1) % 200 == 0 or (frame_idx + 1) == total_frames:
            print(f"Rendered {frame_idx + 1}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
