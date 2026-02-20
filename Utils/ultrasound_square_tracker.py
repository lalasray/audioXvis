#!/usr/bin/env python3
"""
Interactive square ROI tracking for ultrasound videos.

Features:
- Draw multiple square ROIs on first frame with mouse.
- Hybrid tracking: CSRT + Lucas-Kanade optical flow + template fallback.
- Frame preprocessing tuned for ultrasound (median blur + CLAHE).
- Optional output video and CSV trajectory log.

Usage:
    python Utils/ultrasound_square_tracker.py --video input.mp4 --num-trackers 3 --show
    python Utils/ultrasound_square_tracker.py --video input.mp4 --out-video tracked.mp4 --out-csv track.csv --show
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackState:
    x: int
    y: int
    size: int
    confidence: float
    method: str


class SquareSelector:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.start = None
        self.end = None
        self.dragging = False
        self.final_square = None

    def _to_square(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x0, y0 = start
        x1, y1 = end

        dx = x1 - x0
        dy = y1 - y0
        side = int(max(abs(dx), abs(dy)))

        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1

        x2 = x0 + sx * side
        y2 = y0 + sy * side

        left = min(x0, x2)
        top = min(y0, y2)
        right = max(x0, x2)
        bottom = max(y0, y2)
        return left, top, right, bottom

    def mouse_callback(self, event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True
            self.final_square = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x, y)
            self.dragging = False
            self.final_square = self._to_square(self.start, self.end)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        if self.start is not None and self.end is not None:
            left, top, right, bottom = self._to_square(self.start, self.end)
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            display,
            "Drag to draw square ROI, ENTER to confirm, ESC to cancel",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return display


@dataclass
class ROITrack:
    track_id: int
    color: Tuple[int, int, int]
    tracker: any
    x: int
    y: int
    size: int
    method: str
    confidence: float
    base_template: Optional[np.ndarray]
    adaptive_template: Optional[np.ndarray]
    flow_points: Optional[np.ndarray]
    ema_cx: float
    ema_cy: float


def create_csrt_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError(
        "CSRT tracker is unavailable. Install opencv-contrib-python to use this script."
    )


def preprocess_for_ultrasound(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def clamp_square(x: int, y: int, size: int, width: int, height: int) -> Tuple[int, int, int]:
    size = max(8, min(size, width, height))
    x = max(0, min(x, width - size))
    y = max(0, min(y, height - size))
    return x, y, size


def extract_patch(gray: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    if x < 0 or y < 0 or x + size > w or y + size > h:
        return None
    patch = gray[y : y + size, x : x + size]
    if patch.size == 0:
        return None
    return patch


def init_flow_points(gray: np.ndarray, x: int, y: int, size: int, max_points: int = 80):
    roi = extract_patch(gray, x, y, size)
    if roi is None:
        return None
    pts = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
        useHarrisDetector=False,
    )
    if pts is None:
        return None
    pts[:, 0, 0] += x
    pts[:, 0, 1] += y
    return pts


def flow_update(prev_gray: np.ndarray, gray: np.ndarray, prev_points: np.ndarray):
    if prev_points is None or len(prev_points) < 4:
        return None, 0.0

    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None or status is None:
        return None, 0.0

    ok = status.reshape(-1) == 1
    if ok.sum() < 4:
        return None, 0.0

    p0 = prev_points[ok].reshape(-1, 2)
    p1 = next_pts[ok].reshape(-1, 2)
    disp = p1 - p0
    median_disp = np.median(disp, axis=0)
    confidence = float(ok.sum()) / float(len(status))

    new_points = p1.reshape(-1, 1, 2)
    return (float(median_disp[0]), float(median_disp[1]), new_points), confidence


def template_redetect(gray: np.ndarray, template: np.ndarray, x: int, y: int, size: int, search_scale: float = 2.2):
    if template is None:
        return None, 0.0

    h, w = gray.shape[:2]
    radius = int(size * search_scale)
    cx = x + size // 2
    cy = y + size // 2

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius)
    y1 = min(h, cy + radius)

    search = gray[y0:y1, x0:x1]
    th, tw = template.shape[:2]
    if search.shape[0] < th or search.shape[1] < tw:
        return None, 0.0

    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    nx = x0 + max_loc[0]
    ny = y0 + max_loc[1]

    return (nx, ny), float(max_val)


def select_square_roi(frame: np.ndarray, window_name: str = "Select ROI"):
    selector = SquareSelector(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    while True:
        vis = selector.draw_overlay(frame)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(10) & 0xFF

        if key == 13:  # ENTER
            if selector.final_square is not None:
                left, top, right, bottom = selector.final_square
                size = min(right - left, bottom - top)
                if size >= 8:
                    cv2.destroyWindow(window_name)
                    return left, top, size
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None


def select_multiple_square_rois(frame: np.ndarray, num_trackers: int):
    rois = []
    for i in range(num_trackers):
        roi = select_square_roi(frame, window_name=f"Select ROI #{i + 1}")
        if roi is None:
            return None
        rois.append(roi)
    return rois


def run_tracking(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video has no readable frames.")

    rois = select_multiple_square_rois(first_frame, args.num_trackers)
    if rois is None:
        cap.release()
        print("ROI selection cancelled.")
        return

    gray0 = preprocess_for_ultrasound(first_frame)
    track_img0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)

    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0)]
    tracks = []
    for idx, roi in enumerate(rois):
        x, y, size = roi
        x, y, size = clamp_square(x, y, size, width, height)

        tracker = create_csrt_tracker()
        tracker.init(track_img0, (x, y, size, size))

        base_template = extract_patch(gray0, x, y, size)
        adaptive_template = base_template.copy() if base_template is not None else None
        flow_points = init_flow_points(gray0, x, y, size, max_points=args.flow_points)

        tracks.append(
            ROITrack(
                track_id=idx + 1,
                color=palette[idx % len(palette)],
                tracker=tracker,
                x=x,
                y=y,
                size=size,
                method="csrt",
                confidence=1.0,
                base_template=base_template,
                adaptive_template=adaptive_template,
                flow_points=flow_points,
                ema_cx=x + size / 2.0,
                ema_cy=y + size / 2.0,
            )
        )

    prev_gray = gray0

    out_video = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    csv_file = None
    csv_writer = None
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        csv_file = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "tracker_id", "x", "y", "size", "confidence", "method"])

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = preprocess_for_ultrasound(frame)
        track_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for track in tracks:
            # 1) Primary tracker update
            t_ok, t_box = track.tracker.update(track_img)
            method = "csrt"
            conf = 0.65 if t_ok else 0.0

            tx, ty, size = track.x, track.y, track.size
            if t_ok:
                bx, by, bw, bh = t_box
                tx = int(round(bx + bw / 2.0 - size / 2.0))
                ty = int(round(by + bh / 2.0 - size / 2.0))

            # 2) Optical flow correction
            flow = None
            flow_conf = 0.0
            if track.flow_points is not None and prev_gray is not None:
                flow, flow_conf = flow_update(prev_gray, gray, track.flow_points)

            if flow is not None:
                dx, dy, new_flow_points = flow
                fx = int(round(track.x + dx))
                fy = int(round(track.y + dy))
                fx, fy, _ = clamp_square(fx, fy, size, width, height)

                if t_ok:
                    alpha = min(0.8, max(0.25, flow_conf))
                    tx = int(round((1 - alpha) * tx + alpha * fx))
                    ty = int(round((1 - alpha) * ty + alpha * fy))
                    conf = max(conf, 0.55 + 0.4 * flow_conf)
                    method = "csrt+flow"
                else:
                    tx, ty = fx, fy
                    conf = 0.45 + 0.45 * flow_conf
                    method = "flow"

                track.flow_points = new_flow_points
            else:
                track.flow_points = None

            # 3) Template fallback if confidence is weak
            if conf < args.redetect_threshold:
                best_xy = None
                best_score = -1.0

                if track.adaptive_template is not None:
                    cand, score = template_redetect(gray, track.adaptive_template, tx, ty, size)
                    if score > best_score:
                        best_xy, best_score = cand, score

                if track.base_template is not None:
                    cand, score = template_redetect(gray, track.base_template, tx, ty, size)
                    if score > best_score:
                        best_xy, best_score = cand, score

                if best_xy is not None and best_score >= args.redetect_threshold:
                    tx, ty = best_xy
                    conf = max(conf, best_score)
                    method = "template"

                    track.tracker = create_csrt_tracker()
                    track.tracker.init(track_img, (tx, ty, size, size))

            # 4) Clamp + smoothing
            tx, ty, size = clamp_square(tx, ty, size, width, height)

            cx = tx + size / 2.0
            cy = ty + size / 2.0
            track.ema_cx = args.smooth_alpha * cx + (1.0 - args.smooth_alpha) * track.ema_cx
            track.ema_cy = args.smooth_alpha * cy + (1.0 - args.smooth_alpha) * track.ema_cy

            track.x = int(round(track.ema_cx - size / 2.0))
            track.y = int(round(track.ema_cy - size / 2.0))
            track.x, track.y, track.size = clamp_square(track.x, track.y, size, width, height)

            # Re-seed flow features periodically or when missing
            if (frame_idx % args.reseed_interval == 0) or (track.flow_points is None) or (len(track.flow_points) < 10):
                track.flow_points = init_flow_points(gray, track.x, track.y, track.size, max_points=args.flow_points)

            # Adapt template when confident
            patch = extract_patch(gray, track.x, track.y, track.size)
            if patch is not None and conf >= max(args.redetect_threshold, 0.62):
                if track.adaptive_template is None:
                    track.adaptive_template = patch.copy()
                else:
                    track.adaptive_template = cv2.addWeighted(
                        track.adaptive_template,
                        1.0 - args.template_lr,
                        patch,
                        args.template_lr,
                        0.0,
                    )

            track.method = method
            track.confidence = conf

        vis = frame.copy()
        for i, track in enumerate(tracks):
            cv2.rectangle(
                vis,
                (track.x, track.y),
                (track.x + track.size, track.y + track.size),
                track.color,
                2,
            )
            cv2.putText(
                vis,
                f"T{track.track_id} {track.method} {track.confidence:.2f}",
                (10, 30 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                track.color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            vis,
            "q:quit  p:pause",
            (10, 30 + len(tracks) * 24 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if out_video is not None:
            out_video.write(vis)

        if csv_writer is not None:
            for track in tracks:
                csv_writer.writerow(
                    [
                        frame_idx,
                        track.track_id,
                        track.x,
                        track.y,
                        track.size,
                        f"{track.confidence:.4f}",
                        track.method,
                    ]
                )

        if args.show:
            cv2.imshow("Ultrasound Square Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 == ord("p"):
                        break
                    if k2 == ord("q"):
                        key = ord("q")
                        break
                if key == ord("q"):
                    break

        prev_gray = gray
        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()

    print("Tracking complete.")
    if args.out_video:
        print(f"Saved tracked video: {args.out_video}")
    if args.out_csv:
        print(f"Saved track CSV: {args.out_csv}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Robust square tracking for ultrasound video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--out-video", type=str, default="", help="Optional output video path")
    parser.add_argument("--out-csv", type=str, default="", help="Optional CSV path for tracked ROI")
    parser.add_argument("--num-trackers", type=int, default=3, help="Number of square ROIs to select and track")
    parser.add_argument("--show", action="store_true", help="Display tracking window")

    parser.add_argument("--flow-points", type=int, default=80, help="Max optical-flow points in ROI")
    parser.add_argument("--reseed-interval", type=int, default=8, help="How often to re-seed flow points")
    parser.add_argument(
        "--redetect-threshold",
        type=float,
        default=0.58,
        help="Template matching threshold for re-detection (0-1)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.55,
        help="EMA smoothing factor for center position (0-1)",
    )
    parser.add_argument(
        "--template-lr",
        type=float,
        default=0.08,
        help="Template adaptation learning rate (0-1)",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.num_trackers < 1:
        raise ValueError("--num-trackers must be >= 1")
    run_tracking(args)


if __name__ == "__main__":
    main()
