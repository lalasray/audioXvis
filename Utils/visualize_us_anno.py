"""Visualize ultrasound landmark annotations and derived larynx-style shape overlays over video."""

import argparse
import cv2
import numpy as np

from annotation_utils import interpolate_points_to_frame_count, load_points_csv


def load_video_frames(video_path: str, crop: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read source FPS from video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while len(frames) < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if crop:
            frame = frame[0:600, 600:1200]
        frames.append(frame)

    cap.release()
    return frames, original_fps


def _norm(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mag = np.linalg.norm(vec)
    if mag < eps:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vec / mag


def _bezier_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


def draw_dotted_line(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color=(70, 70, 70),
    thickness: int = 2,
    gap: int = 9,
):
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1.0:
        return

    direction = vec / length
    n_dots = int(length // gap) + 1
    for i in range(n_dots + 1):
        p = start + direction * (i * gap)
        cv2.circle(image, tuple(p.astype(np.int32)), thickness, color, -1, cv2.LINE_AA)


def draw_angle_marker(
    image: np.ndarray,
    vertex: np.ndarray,
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    color=(150, 40, 180),
    radius: int = 18,
    thickness: int = 1,
):
    vec_a = arm_a - vertex
    vec_b = arm_b - vertex
    len_a = float(np.linalg.norm(vec_a))
    len_b = float(np.linalg.norm(vec_b))
    if len_a < 1.0 or len_b < 1.0:
        return

    unit_a = vec_a / len_a
    unit_b = vec_b / len_b
    dot_val = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot_val)))

    theta_a = float(np.arctan2(unit_a[1], unit_a[0]))
    theta_b = float(np.arctan2(unit_b[1], unit_b[0]))
    delta = (theta_b - theta_a + np.pi * 3.0) % (2.0 * np.pi) - np.pi

    n_samples = 40
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    thetas = theta_a + delta * t
    arc = np.stack(
        [
            vertex[0] + radius * np.cos(thetas),
            vertex[1] + radius * np.sin(thetas),
        ],
        axis=1,
    ).astype(np.int32)
    cv2.polylines(image, [arc], False, color, thickness, cv2.LINE_AA)

    bisector = unit_a + unit_b
    bisector_len = float(np.linalg.norm(bisector))
    if bisector_len < 1e-6:
        bisector = np.array([1.0, 0.0], dtype=np.float32)
    else:
        bisector = bisector / bisector_len

    text_pos = vertex + bisector * (radius + 14)
    cv2.putText(
        image,
        f"{angle_deg:.1f}",
        (int(text_pos[0]), int(text_pos[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_raw_with_points(
    frame: np.ndarray,
    points: np.ndarray,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    out = frame.copy()

    for i, pt in enumerate(points):
        if not np.allclose(pt, 0):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)
            cv2.putText(
                out,
                str(i + 1),
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        f"Frame {frame_idx + 1}/{total_frames}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "Raw + 5 landmarks",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_deformable_larynx(
    canvas: np.ndarray,
    points: np.ndarray,
    frame_idx: int,
    total_frames: int,
    show_points: bool,
) -> np.ndarray:
    out = canvas.copy()

    valid = [not np.allclose(pt, 0) for pt in points]
    if not all(valid):
        return out

    p1, p2, p3, p4, p5 = [pt.astype(np.float32) for pt in points]
    mid_bottom = (p2 + p3) * 0.5

    lr = _norm(p4 - p1)
    up = _norm(p5 - mid_bottom)
    width = float(np.linalg.norm(p4 - p1))
    if width < 2.0:
        width = 2.0

    arch_outer_ctrl = p5 + up * (0.34 * width)
    arch_inner_ctrl = p5 - up * (0.08 * width)
    outer_curve = _bezier_quad(p1, arch_outer_ctrl, p4, n=70)
    inner_curve = _bezier_quad(p1, arch_inner_ctrl, p4, n=70)

    overlay = out.copy()

    blue_ring = np.vstack([outer_curve, inner_curve[::-1]]).astype(np.int32)
    cv2.fillPoly(overlay, [blue_ring], color=(245, 210, 130))

    top_left = p5 - lr * (0.14 * width)
    top_right = p5 + lr * (0.14 * width)

    left_yellow = np.array([p1, top_left, p2], dtype=np.int32)
    right_yellow = np.array([p3, top_right, p4], dtype=np.int32)
    cv2.fillPoly(overlay, [left_yellow, right_yellow], color=(70, 190, 250))

    left_red = np.array([p2, p5, mid_bottom], dtype=np.int32)
    right_red = np.array([mid_bottom, p5, p3], dtype=np.int32)
    cv2.fillPoly(overlay, [left_red, right_red], color=(80, 80, 220))

    left_green = np.array(
        [
            p1,
            p2,
            mid_bottom - lr * (0.06 * width) - up * (0.08 * width),
            p1 + up * (0.10 * width),
        ],
        dtype=np.int32,
    )
    right_green = np.array(
        [
            p4,
            p3,
            mid_bottom + lr * (0.06 * width) - up * (0.08 * width),
            p4 + up * (0.10 * width),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [left_green, right_green], color=(90, 195, 90))

    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0)

    cv2.polylines(out, [outer_curve.astype(np.int32)], False, (220, 150, 40), 2, cv2.LINE_AA)
    cv2.polylines(out, [inner_curve.astype(np.int32)], False, (220, 150, 40), 2, cv2.LINE_AA)
    cv2.line(out, tuple(p5.astype(np.int32)), tuple(mid_bottom.astype(np.int32)), (10, 10, 10), 2, cv2.LINE_AA)

    mid_12 = (p1 + p2) * 0.5
    mid_34 = (p3 + p4) * 0.5
    draw_dotted_line(out, mid_12, mid_34, color=(80, 80, 80), thickness=2, gap=10)
    draw_dotted_line(out, mid_34, p5, color=(80, 80, 80), thickness=2, gap=10)
    draw_dotted_line(out, p5, mid_12, color=(80, 80, 80), thickness=2, gap=10)
    draw_angle_marker(out, mid_12, mid_34, p5)
    draw_angle_marker(out, mid_34, p5, mid_12)
    draw_angle_marker(out, p5, mid_12, mid_34)

    if show_points:
        for i, pt in enumerate(points):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)
            cv2.putText(
                out,
                str(i + 1),
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        "2D render",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize ultrasound point annotations on video frames.")
    parser.add_argument("-i", "--input_video", required=True, help="Path to input video.")
    parser.add_argument("-a", "--annotation_csv", required=True, help="Path to annotation CSV file.")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable ultrasound crop. Default uses frame[0:600, 600:1200] to match anno_us.py.",
    )
    parser.add_argument(
        "--hide_points",
        action="store_true",
        help="Hide numeric landmark points and show only deformable colored regions.",
    )

    args = parser.parse_args()

    frames, original_fps = load_video_frames(args.input_video, crop=not args.no_crop)
    points = load_points_csv(args.annotation_csv)

    if len(frames) == 0:
        raise RuntimeError("No frames loaded from video after sampling.")

    dense_frames = frames
    dense_points = interpolate_points_to_frame_count(points, target_frames=len(frames))
    total = len(dense_frames)

    print(f"Original FPS: {original_fps:.2f}")
    print(f"Video frames: {len(frames)}")
    print(f"Annotation rows: {len(points)}")
    print(f"Rendered frames: {total}")
    print("Controls: d=next, a=previous, space=play/pause, q=quit")

    idx = 0
    is_playing = False
    cv2.namedWindow("annotation_visualization", cv2.WINDOW_NORMAL)

    while True:
        left = draw_raw_with_points(
            dense_frames[idx],
            dense_points[idx],
            idx,
            total,
        )

        right_base = np.full_like(dense_frames[idx], 255)
        right = draw_deformable_larynx(
            right_base,
            dense_points[idx],
            idx,
            total,
            show_points=not args.hide_points,
        )

        canvas = np.hstack([left, right])
        canvas = cv2.resize(canvas, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("annotation_visualization", canvas)

        delay = max(1, int(round(1000.0 / max(1.0, original_fps)))) if is_playing else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("d"):
            idx = min(total - 1, idx + 1)
        elif key == ord("a"):
            idx = max(0, idx - 1)
        elif key == ord(" "):
            is_playing = not is_playing

        if is_playing and key == 255:
            idx += 1
            if idx >= total:
                idx = total - 1
                is_playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
