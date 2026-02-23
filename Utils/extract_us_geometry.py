import argparse
import csv
import cv2
import numpy as np


def load_points(csv_path: str) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",")
    points = np.atleast_2d(points)
    if points.shape[1] != 10:
        raise ValueError(
            f"Annotation CSV must have 10 columns (5 x,y points). Got {points.shape[1]} columns."
        )
    return points.reshape(-1, 5, 2).astype(np.float32)


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        raise RuntimeError("Could not read frame count from video.")
    return frame_count


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


def safe_norm(vec: np.ndarray, eps: float = 1e-8) -> float:
    val = float(np.linalg.norm(vec))
    return val if val > eps else eps


def angle_degrees(vertex: np.ndarray, arm_a: np.ndarray, arm_b: np.ndarray) -> float:
    v1 = arm_a - vertex
    v2 = arm_b - vertex
    n1 = safe_norm(v1)
    n2 = safe_norm(v2)
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def extract_geometry(points: np.ndarray):
    rows = []
    for idx in range(points.shape[0]):
        p1, p2, p3, p4, p5 = points[idx]

        mid_12 = (p1 + p2) * 0.5
        mid_34 = (p3 + p4) * 0.5

        side_12_34 = float(np.linalg.norm(mid_12 - mid_34))
        side_34_5 = float(np.linalg.norm(mid_34 - p5))
        side_5_12 = float(np.linalg.norm(p5 - mid_12))

        angle_at_mid12 = angle_degrees(mid_12, mid_34, p5)
        angle_at_mid34 = angle_degrees(mid_34, p5, mid_12)
        angle_at_p5 = angle_degrees(p5, mid_12, mid_34)

        rows.append(
            [
                idx,
                p1[0], p1[1],
                p2[0], p2[1],
                p3[0], p3[1],
                p4[0], p4[1],
                p5[0], p5[1],
                mid_12[0], mid_12[1],
                mid_34[0], mid_34[1],
                p5[0], p5[1],
                side_12_34,
                side_34_5,
                side_5_12,
                angle_at_mid12,
                angle_at_mid34,
                angle_at_p5,
                angle_at_mid12 + angle_at_mid34 + angle_at_p5,
            ]
        )

    header = [
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

    return header, rows


def write_csv(output_path: str, header, rows):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-frame geometry from ultrasound 5-point annotations: "
            "landmarks, triangle points, side lengths, and angles."
        )
    )
    parser.add_argument("-a", "--annotation_csv", required=True, help="Path to input annotation CSV.")
    parser.add_argument("-o", "--output_csv", required=True, help="Path to output feature CSV.")
    parser.add_argument(
        "-i",
        "--input_video",
        required=False,
        help="Optional video path. If provided, annotations are interpolated to full video frame count.",
    )

    args = parser.parse_args()

    points = load_points(args.annotation_csv)

    if args.input_video:
        target_frames = get_video_frame_count(args.input_video)
        points = interpolate_points_to_frame_count(points, target_frames=target_frames)

    header, rows = extract_geometry(points)
    write_csv(args.output_csv, header, rows)

    print(f"Saved geometry features: {args.output_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
