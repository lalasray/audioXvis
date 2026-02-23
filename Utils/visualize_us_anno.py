import argparse
import cv2
import numpy as np


def load_sampled_frames(video_path: str, target_fps: int, crop: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read source FPS from video.")

    frame_interval = original_fps / float(target_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_index = 0
    next_keep = 0.0

    while frame_index < frame_count:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index + 1e-9 >= next_keep:
            if crop:
                frame = frame[0:600, 600:1200]
            frames.append(frame)
            next_keep += frame_interval

        frame_index += 1

    cap.release()
    return frames, original_fps


def load_points(csv_path: str) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",")
    points = np.atleast_2d(points)

    if points.shape[1] != 10:
        raise ValueError(
            f"Annotation CSV must have 10 columns (5 x,y points). Got {points.shape[1]} columns."
        )

    return points.reshape(-1, 5, 2)


def _norm(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mag = np.linalg.norm(vec)
    if mag < eps:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vec / mag


def _bezier_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


def interpolate_sparse_points(points: np.ndarray, upsample: int, smooth_window: int) -> np.ndarray:
    """
    Interpolates sparse landmark tracks.
    - Missing points (0,0) are treated as gaps and linearly filled over time.
    - Sequence is upsampled for continuous playback.
    """
    n_frames, n_pts, _ = points.shape
    t_src = np.arange(n_frames, dtype=np.float32)
    t_dst = np.linspace(0.0, n_frames - 1, (n_frames - 1) * upsample + 1, dtype=np.float32)

    dense = np.zeros((len(t_dst), n_pts, 2), dtype=np.float32)

    for p in range(n_pts):
        for axis in range(2):
            signal = points[:, p, axis].astype(np.float32)
            valid = ~np.isclose(points[:, p, 0], 0.0) | ~np.isclose(points[:, p, 1], 0.0)

            if not np.any(valid):
                continue

            src_valid_t = t_src[valid]
            src_valid_v = signal[valid]

            filled = np.interp(t_src, src_valid_t, src_valid_v)
            interp = np.interp(t_dst, t_src, filled)

            if smooth_window > 1:
                kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
                interp = np.convolve(interp, kernel, mode="same")

            dense[:, p, axis] = interp

    return dense


def interpolate_frames(frames, upsample: int):
    """Frame interpolation by alpha blending between sampled frames."""
    if upsample <= 1:
        return frames

    dense_frames = []
    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]
        for k in range(upsample):
            alpha = k / float(upsample)
            blended = cv2.addWeighted(f0, 1.0 - alpha, f1, alpha, 0.0)
            dense_frames.append(blended)
    dense_frames.append(frames[-1])
    return dense_frames


def draw_deformable_larynx(
    frame: np.ndarray,
    points: np.ndarray,
    frame_idx: int,
    total_frames: int,
    show_points: bool,
) -> np.ndarray:
    out = frame.copy()

    valid = [not np.allclose(pt, 0) for pt in points]
    if not all(valid):
        cv2.putText(
            out,
            f"Frame {frame_idx + 1}/{total_frames} (missing landmark)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
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
    cv2.fillPoly(overlay, [blue_ring], color=(220, 150, 20))

    top_left = p5 - lr * (0.14 * width)
    top_right = p5 + lr * (0.14 * width)

    left_yellow = np.array([p1, top_left, p2], dtype=np.int32)
    right_yellow = np.array([p3, top_right, p4], dtype=np.int32)
    cv2.fillPoly(overlay, [left_yellow, right_yellow], color=(0, 190, 255))

    left_red = np.array([p2, top_left, p5, mid_bottom], dtype=np.int32)
    right_red = np.array([mid_bottom, p5, top_right, p3], dtype=np.int32)
    cv2.fillPoly(overlay, [left_red, right_red], color=(35, 35, 230))

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
    cv2.fillPoly(overlay, [left_green, right_green], color=(70, 210, 70))

    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0.0)

    cv2.polylines(out, [outer_curve.astype(np.int32)], False, (255, 210, 40), 2, cv2.LINE_AA)
    cv2.polylines(out, [inner_curve.astype(np.int32)], False, (255, 210, 40), 2, cv2.LINE_AA)
    cv2.line(out, tuple(p5.astype(np.int32)), tuple(mid_bottom.astype(np.int32)), (10, 10, 10), 2, cv2.LINE_AA)

    if show_points:
        for i, pt in enumerate(points):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (255, 255, 210), -1)
            cv2.putText(
                out,
                str(i + 1),
                (int(pt[0]) + 6, int(pt[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        f"Frame {frame_idx + 1}/{total_frames}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize ultrasound point annotations on video frames.")
    parser.add_argument("-i", "--input_video", required=True, help="Path to input video.")
    parser.add_argument("-a", "--annotation_csv", required=True, help="Path to annotation CSV file.")
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        required=True,
        help="Annotation FPS used during labeling (same value used in anno_us.py).",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable ultrasound crop. Default uses frame[0:600, 600:1200] to match anno_us.py.",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=4,
        help="Interpolation factor for continuous visualization (default: 4).",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Temporal smoothing window on interpolated points (default: 5).",
    )
    parser.add_argument(
        "--hide_points",
        action="store_true",
        help="Hide numeric landmark points and show only deformable colored regions.",
    )

    args = parser.parse_args()

    frames, original_fps = load_sampled_frames(args.input_video, args.fps, crop=not args.no_crop)
    points = load_points(args.annotation_csv)

    if len(frames) == 0:
        raise RuntimeError("No frames loaded from video after sampling.")

    total_sampled = min(len(frames), len(points))
    if len(frames) != len(points):
        print(
            f"Warning: sampled frames ({len(frames)}) and annotation rows ({len(points)}) differ. "
            f"Visualizing first {total_sampled} frames."
        )
    frames = frames[:total_sampled]
    points = points[:total_sampled]

    dense_points = interpolate_sparse_points(
        points,
        upsample=max(1, args.upsample),
        smooth_window=max(1, args.smooth_window),
    )
    dense_frames = interpolate_frames(frames, upsample=max(1, args.upsample))
    total = min(len(dense_frames), len(dense_points))
    dense_frames = dense_frames[:total]
    dense_points = dense_points[:total]

    print(f"Original FPS: {original_fps:.2f}")
    print(f"Sampled frames: {len(frames)}")
    print(f"Annotation rows: {len(points)}")
    print(f"Continuous frames: {total}")
    print("Controls: d=next, a=previous, space=play/pause, q=quit")

    idx = 0
    is_playing = False
    cv2.namedWindow("annotation_visualization", cv2.WINDOW_NORMAL)

    while True:
        canvas = draw_deformable_larynx(
            dense_frames[idx],
            dense_points[idx],
            idx,
            total,
            show_points=not args.hide_points,
        )
        cv2.imshow("annotation_visualization", canvas)

        delay = int(1000 / (args.fps * max(1, args.upsample))) if is_playing else 0
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
