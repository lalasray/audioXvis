"""Render side-by-side video showing raw ultrasound frames and extracted geometry overlays."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


REQUIRED_COLUMNS = [
    "frame_idx",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
    "p5_x", "p5_y",
    "tri_a_x", "tri_a_y",
    "tri_b_x", "tri_b_y",
    "tri_c_x", "tri_c_y",
    "side_a_b",
    "side_b_c",
    "side_c_a",
    "angle_a_deg",
    "angle_b_deg",
    "angle_c_deg",
    "angle_sum_deg",
]


def load_geometry_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Geometry CSV has no header.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Geometry CSV missing required columns: {missing}")

        for row in reader:
            parsed = {}
            for key, val in row.items():
                if key == "frame_idx":
                    parsed[key] = int(float(val))
                else:
                    parsed[key] = float(val)
            rows.append(parsed)

    if not rows:
        raise ValueError("Geometry CSV contains no data rows.")

    return rows


def get_video_props(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height.")

    cap.release()
    return fps, width, height, frame_count


def get_output_path(output_video: str | None, input_video: str, geometry_csv: str) -> str:
    if output_video:
        return output_video

    in_stem = Path(input_video).stem
    geo_stem = Path(geometry_csv).stem
    return str(Path(f"{geo_stem}_{in_stem}_geometry_viz.mp4"))


def _draw_points_and_triangle(panel: np.ndarray, geometry_row: dict):
    h, w = panel.shape[:2]
    pad = 40

    points = np.array(
        [
            [geometry_row["p1_x"], geometry_row["p1_y"]],
            [geometry_row["p2_x"], geometry_row["p2_y"]],
            [geometry_row["p3_x"], geometry_row["p3_y"]],
            [geometry_row["p4_x"], geometry_row["p4_y"]],
            [geometry_row["p5_x"], geometry_row["p5_y"]],
        ],
        dtype=np.float32,
    )

    valid = ~np.all(np.isclose(points, 0.0), axis=1)
    if not np.any(valid):
        cv2.putText(
            panel,
            "No valid geometry points",
            (pad, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    pts = points[valid]
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    span = np.maximum(max_xy - min_xy, 1.0)
    target_w = max(1, w - 2 * pad)
    target_h = max(1, int((h * 0.62) - pad))
    scale = min(target_w / span[0], target_h / span[1])

    offset_x = pad + (target_w - span[0] * scale) * 0.5
    offset_y = pad + (target_h - span[1] * scale) * 0.5

    mapped = (points - min_xy) * scale
    mapped[:, 0] += offset_x
    mapped[:, 1] += offset_y

    p1, p2, p3, p4, p5 = mapped
    tri_a = np.array([geometry_row["tri_a_x"], geometry_row["tri_a_y"]], dtype=np.float32)
    tri_b = np.array([geometry_row["tri_b_x"], geometry_row["tri_b_y"]], dtype=np.float32)
    tri_c = np.array([geometry_row["tri_c_x"], geometry_row["tri_c_y"]], dtype=np.float32)

    tri = np.array([tri_a, tri_b, tri_c], dtype=np.float32)
    tri = (tri - min_xy) * scale
    tri[:, 0] += offset_x
    tri[:, 1] += offset_y

    cv2.line(panel, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), (150, 150, 150), 2, cv2.LINE_AA)
    cv2.line(panel, tuple(p3.astype(np.int32)), tuple(p4.astype(np.int32)), (150, 150, 150), 2, cv2.LINE_AA)

    cv2.polylines(panel, [tri.astype(np.int32)], True, (60, 80, 220), 2, cv2.LINE_AA)

    colors = [
        (255, 180, 60),
        (255, 180, 60),
        (60, 200, 255),
        (60, 200, 255),
        (80, 220, 80),
    ]

    for i, point in enumerate(mapped):
        if np.allclose(points[i], 0.0):
            continue

        cv2.circle(panel, tuple(point.astype(np.int32)), 5, colors[i], -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"P{i + 1}",
            (int(point[0]) + 6, int(point[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def draw_geometry_panel(frame: np.ndarray, geometry_row: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    h, w = frame.shape[:2]
    panel = np.full((h, w, 3), 245, dtype=np.uint8)

    cv2.putText(
        panel,
        f"Geometry (frame {frame_idx + 1}/{total_frames})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    _draw_points_and_triangle(panel, geometry_row)

    text_y = int(h * 0.70)
    line_h = 30

    side_a_b = geometry_row["side_a_b"]
    side_b_c = geometry_row["side_b_c"]
    side_c_a = geometry_row["side_c_a"]

    angle_a = geometry_row["angle_a_deg"]
    angle_b = geometry_row["angle_b_deg"]
    angle_c = geometry_row["angle_c_deg"]
    angle_sum = geometry_row["angle_sum_deg"]

    stats = [
        f"side_a_b: {side_a_b:.2f}",
        f"side_b_c: {side_b_c:.2f}",
        f"side_c_a: {side_c_a:.2f}",
        f"angle_a: {angle_a:.2f} deg",
        f"angle_b: {angle_b:.2f} deg",
        f"angle_c: {angle_c:.2f} deg",
        f"angle_sum: {angle_sum:.2f} deg",
    ]

    for i, line in enumerate(stats):
        cv2.putText(
            panel,
            line,
            (20, text_y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return panel


def draw_raw_frame(frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        f"Raw video (frame {frame_idx + 1}/{total_frames})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize extracted geometry side-by-side with the raw video. "
            "Left: raw frame, Right: geometry panel."
        )
    )
    parser.add_argument("-i", "--input_video", required=True, help="Path to input video.")
    parser.add_argument("-g", "--geometry_csv", required=True, help="Path to extracted geometry CSV.")
    parser.add_argument(
        "-o",
        "--output_video",
        default=None,
        help="Path to output side-by-side visualization video. Defaults to <geometry>_<video>_geometry_viz.mp4",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization while writing output video (press q to stop early).",
    )

    args = parser.parse_args()

    geometry_rows = load_geometry_csv(args.geometry_csv)
    fps, width, height, video_frames = get_video_props(args.input_video)

    total_frames = min(video_frames, len(geometry_rows))
    if total_frames <= 0:
        raise RuntimeError("No frames available to render.")

    if len(geometry_rows) != video_frames:
        print(
            f"Warning: video has {video_frames} frames but geometry has {len(geometry_rows)} rows. "
            f"Rendering {total_frames} synchronized frames."
        )

    output_path = get_output_path(args.output_video, args.input_video, args.geometry_csv)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input_video}")

    out_w = width * 2
    out_h = height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    window_name = "geometry_side_by_side"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        raw_vis = draw_raw_frame(frame, idx, total_frames)
        geom_vis = draw_geometry_panel(frame, geometry_rows[idx], idx, total_frames)
        side_by_side = np.hstack([raw_vis, geom_vis])
        writer.write(side_by_side)

        if args.show:
            cv2.imshow(window_name, side_by_side)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Stopped early by user (q).")
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved visualization video: {output_path}")


if __name__ == "__main__":
    main()
