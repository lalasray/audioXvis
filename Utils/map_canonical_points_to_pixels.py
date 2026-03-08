"""Map canonical 5-point reconstruction back to image pixel coordinates.

Canonical convention expected in input CSV:
  tri_a=(0,0), tri_b=(1,0), tri_c=(cx,cy), and p1..p5 in same canonical frame.

Pixel mapping uses one reference frame from a GT CSV:
  P_pixel = A_ref + (x * u_ref + y * v_ref) * |AB_ref|
where:
  u_ref = unit vector A_ref->B_ref
  v_ref = 90deg clockwise normal in image coordinates
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in CSV: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return list(reader.fieldnames), rows


def to_float(value: str) -> float:
    return float(value)


def get_reference_transform(gt_rows: List[Dict[str, str]], ref_index: int) -> Tuple[float, float, float, float, float]:
    idx = max(0, min(ref_index, len(gt_rows) - 1))
    r = gt_rows[idx]
    ax, ay = to_float(r["tri_a_x"]), to_float(r["tri_a_y"])
    bx, by = to_float(r["tri_b_x"]), to_float(r["tri_b_y"])
    dx, dy = bx - ax, by - ay
    scale = (dx * dx + dy * dy) ** 0.5
    if scale < 1e-8:
        raise ValueError("Reference frame has near-zero AB length.")
    ux, uy = dx / scale, dy / scale
    # clockwise normal for image y-down coordinates
    vx, vy = -uy, ux
    return ax, ay, ux, uy, scale, vx, vy


def map_point(xc: float, yc: float, ax: float, ay: float, ux: float, uy: float, scale: float, vx: float, vy: float):
    xp = ax + (xc * ux + yc * vx) * scale
    yp = ay + (xc * uy + yc * vy) * scale
    return xp, yp


def main() -> None:
    parser = argparse.ArgumentParser(description="Map canonical reconstructed points to pixel coordinates.")
    parser.add_argument("--canonical_csv", required=True, help="Input canonical reconstruction CSV.")
    parser.add_argument("--reference_gt_csv", required=True, help="Reference GT CSV with tri_a/tri_b in pixels.")
    parser.add_argument("--output_csv", required=True, help="Output CSV in pixel coordinates.")
    parser.add_argument(
        "--ref_frame_index",
        type=int,
        default=0,
        help="Reference frame index in GT CSV for transform (default: 0).",
    )
    args = parser.parse_args()

    canon_header, canon_rows = read_csv_rows(Path(args.canonical_csv))
    _, gt_rows = read_csv_rows(Path(args.reference_gt_csv))

    ax, ay, ux, uy, scale, vx, vy = get_reference_transform(gt_rows, args.ref_frame_index)

    out_rows: List[Dict[str, str]] = []
    point_names = ["p1", "p2", "p3", "p4", "p5", "tri_a", "tri_b", "tri_c"]

    for row in canon_rows:
        out = dict(row)
        for p in point_names:
            xc = to_float(row[f"{p}_x"])
            yc = to_float(row[f"{p}_y"])
            xp, yp = map_point(xc, yc, ax, ay, ux, uy, scale, vx, vy)
            out[f"{p}_x"] = f"{xp:.6f}"
            out[f"{p}_y"] = f"{yp:.6f}"
        out_rows.append(out)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=canon_header)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved pixel-space reconstruction: {out_path}")
    print(f"Reference transform from frame {args.ref_frame_index}: A=({ax:.2f},{ay:.2f}), scale={scale:.2f}")


if __name__ == "__main__":
    main()
