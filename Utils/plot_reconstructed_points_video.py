"""Render reconstructed 5-point markers + triangle as video overlay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows


def crop_like_anno(frame, enabled: bool):
    if not enabled:
        return frame
    h, w = frame.shape[:2]
    y1, y2 = 0, min(600, h)
    x1, x2 = min(600, w), min(1200, w)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def draw_overlay(frame, row: dict[str, float], idx: int, total: int):
    out = frame.copy()

    a = (int(round(row["tri_a_x"])), int(round(row["tri_a_y"])))
    b = (int(round(row["tri_b_x"])), int(round(row["tri_b_y"])))
    c = (int(round(row["tri_c_x"])), int(round(row["tri_c_y"])))

    cv2.line(out, a, b, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.line(out, b, c, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.line(out, c, a, (255, 200, 60), 2, cv2.LINE_AA)

    for i in range(1, 6):
        p = (int(round(row[f"p{i}_x"])), int(round(row[f"p{i}_y"])))
        cv2.circle(out, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    for label, p in [("A", a), ("B", b), ("C", c)]:
        cv2.putText(out, label, (p[0] + 6, p[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    txt = f"A:{row.get('angle_a', 0.0):.1f} B:{row.get('angle_b', 0.0):.1f} C:{row.get('angle_c', 0.0):.1f}"
    cv2.putText(out, "Reconstructed 5 points from angles", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(out, txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(out, f"Frame {idx + 1}/{total}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def build_video(input_video: Path, points_csv: Path, output: Path, seconds: float, crop: bool):
    rows = load_rows(points_csv)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_rows = len(rows)
    n = min(n_video, n_rows)
    if seconds > 0:
        n = min(n, int(round(seconds * fps)))
    if n <= 0:
        cap.release()
        raise RuntimeError("No frames to render")

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame")
    first = crop_like_anno(first, crop)
    h, w = first.shape[:2]

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = crop_like_anno(frame, crop)
        writer.write(draw_overlay(frame, rows[i], i, n))

    writer.release()
    cap.release()
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot reconstructed 5 points as overlay video.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--points_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=0.0, help="<=0 for full clip")
    parser.add_argument("--no_annotation_crop", action="store_true")
    args = parser.parse_args()

    build_video(
        input_video=Path(args.input_video),
        points_csv=Path(args.points_csv),
        output=Path(args.output),
        seconds=args.seconds,
        crop=not args.no_annotation_crop,
    )


if __name__ == "__main__":
    main()
