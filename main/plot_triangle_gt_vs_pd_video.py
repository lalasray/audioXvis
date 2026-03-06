"""
Plot GT vs Predicted triangle geometry as a video for a whole clip.
- Draw both GT and PD triangles per frame.
- Save as video using OpenCV.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def triangle_points_from_angles(center, angles, radius=100):
    """
    Given center and angles (a, b, c), return triangle points.
    Angles are in degrees, sum to 180.
    """
    a, b, c = angles
    # Place first point at center + radius along x
    p1 = np.array([center[0] + radius, center[1]], dtype=np.float32)
    # Second point: rotate by angle b
    theta_b = np.radians(180 - b)
    p2 = np.array([
        center[0] + radius * np.cos(theta_b),
        center[1] + radius * np.sin(theta_b)
    ], dtype=np.float32)
    # Third point: rotate by angle c
    theta_c = np.radians(180 - b - c)
    p3 = np.array([
        center[0] + radius * np.cos(theta_c),
        center[1] + radius * np.sin(theta_c)
    ], dtype=np.float32)
    return np.stack([p1, p2, p3], axis=0)


def draw_triangle(image, points, color, thickness=2):
    pts = points.astype(np.int32)
    cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(image, tuple(pt), 6, color, -1, cv2.LINE_AA)
        cv2.putText(image, f"P{i+1}", (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Plot GT vs PD triangle video.")
    parser.add_argument("--gt_csv", type=str, required=True, help="Path to GT angles CSV (frame,time,a,b,c)")
    parser.add_argument("--pd_csv", type=str, required=True, help="Path to PD angles CSV (frame,time,a,b,c)")
    parser.add_argument("--output", type=str, required=True, help="Output video path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=int, default=100)
    args = parser.parse_args()

    gt = np.loadtxt(args.gt_csv, delimiter=",", skiprows=1)
    pd = np.loadtxt(args.pd_csv, delimiter=",", skiprows=1)
    n_frames = min(len(gt), len(pd))

    center = np.array([200, 200], dtype=np.float32)
    size = 400
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (size, size))

    for i in range(n_frames):
        frame = np.full((size, size, 3), 245, dtype=np.uint8)
        gt_angles = gt[i, 2:5]
        pd_angles = pd[i, 2:5]
        gt_pts = triangle_points_from_angles(center, gt_angles, radius=args.radius)
        pd_pts = triangle_points_from_angles(center, pd_angles, radius=args.radius)
        draw_triangle(frame, gt_pts, (60, 80, 220), thickness=3)  # GT: blue
        draw_triangle(frame, pd_pts, (255, 180, 60), thickness=2)  # PD: orange
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
