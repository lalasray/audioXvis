"""Create side-by-side video: raw scan + 5 points vs generated triangle (test_set_2)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_gt_rows(gt_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(gt_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        raise RuntimeError(f"GT CSV has no rows: {gt_csv}")
    return rows


def draw_left_raw_with_points(frame: np.ndarray, row: dict[str, float], frame_idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    for i in range(1, 6):
        x = int(round(row[f"p{i}_x"]))
        y = int(round(row[f"p{i}_y"]))
        if x == 0 and y == 0:
            continue
        cv2.circle(out, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(out, "Original 2D + 5 points", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(out, f"Frame {frame_idx + 1}/{total}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def draw_right_triangle(frame: np.ndarray, row: dict[str, float]) -> np.ndarray:
    out = frame.copy()

    a = (int(round(row["tri_a_x"])), int(round(row["tri_a_y"])))
    b = (int(round(row["tri_b_x"])), int(round(row["tri_b_y"])))
    c = (int(round(row["tri_c_x"])), int(round(row["tri_c_y"])))

    overlay = out.copy()
    pts = np.array([a, b, c], dtype=np.int32)
    cv2.fillConvexPoly(overlay, pts, (80, 120, 220), lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0.0)

    cv2.line(out, a, b, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.line(out, b, c, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.line(out, c, a, (255, 200, 60), 2, cv2.LINE_AA)

    for label, pt in [("A", a), ("B", b), ("C", c)]:
        cv2.circle(out, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, label, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(out, "Generated triangle (GT)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 220), 2, cv2.LINE_AA)
    angle_text = f"A:{row['angle_a_deg']:.1f}  B:{row['angle_b_deg']:.1f}  C:{row['angle_c_deg']:.1f}"
    cv2.putText(out, angle_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def maybe_crop_like_annotation(frame: np.ndarray, crop_like_annotation: bool) -> np.ndarray:
    if not crop_like_annotation:
        return frame
    h, w = frame.shape[:2]
    y1, y2 = 0, min(600, h)
    x1, x2 = min(600, w), min(1200, w)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def build_video(
    test_set_root: Path,
    clip_id: str,
    output_path: Path,
    seconds: float,
    crop_like_annotation: bool,
) -> None:
    video_path = test_set_root / "songs" / f"{clip_id}.mp4"
    gt_csv = test_set_root / "gt" / f"gt_{clip_id}.csv"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    rows = load_gt_rows(gt_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_total = len(rows)
    max_frames = min(src_total, gt_total)

    if seconds > 0:
        max_frames = min(max_frames, int(round(seconds * fps)))
    if max_frames <= 0:
        cap.release()
        raise RuntimeError("No frames available to render.")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame.")

    first_frame = maybe_crop_like_annotation(first_frame, crop_like_annotation)
    h, w = first_frame.shape[:2]
    out_w = w * 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, h),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = maybe_crop_like_annotation(frame, crop_like_annotation)
        row = rows[i]
        left = draw_left_raw_with_points(frame, row, i, max_frames)
        right = draw_right_triangle(frame, row)
        canvas = np.hstack([left, right])
        writer.write(canvas)

    writer.release()
    cap.release()
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video for test_set_2.")
    parser.add_argument("--test_set_root", default="data/test_set_2", help="Path to test_set_2 root.")
    parser.add_argument("--clip_id", default="s01", help="Clip ID, e.g. s01.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Duration to render in seconds (<=0 means full clip).")
    parser.add_argument(
        "--no_annotation_crop",
        action="store_true",
        help="Disable anno_us-style crop (default uses frame[0:600,600:1200]).",
    )
    parser.add_argument(
        "--output",
        default="data/test_set_2/side_by_side_s01_points_vs_triangle.mp4",
        help="Output mp4 path.",
    )
    args = parser.parse_args()

    build_video(
        test_set_root=Path(args.test_set_root),
        clip_id=args.clip_id,
        output_path=Path(args.output),
        seconds=args.seconds,
        crop_like_annotation=not args.no_annotation_crop,
    )


if __name__ == "__main__":
    main()
