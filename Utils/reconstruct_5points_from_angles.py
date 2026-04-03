"""Reconstruct 5 marker points from triangle angles using dataset-specific priors.

This is a best-effort reconstruction in canonical coordinates:
  tri_a = A = (0, 0)
  tri_b = B = (1, 0)
  tri_c = C solved from angles

Then p1..p4 are reconstructed from learned per-user half-vectors:
  p1 = A - v12, p2 = A + v12
  p3 = B - v34, p4 = B + v34

User mapping:
  0 -> dataset_a
  1 -> dataset_b
  2 -> dataset_c
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

USER_TO_DATASET = {0: "dataset_a", 1: "dataset_b", 2: "dataset_c"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_points_rows(csv_path: Path) -> Iterable[List[Tuple[float, float]]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            vals = [parse_float(x) for x in row]
            if len(vals) != 10 or any(math.isnan(v) for v in vals):
                continue
            pts = [(vals[i], vals[i + 1]) for i in range(0, 10, 2)]
            if any(abs(x) < 1e-9 and abs(y) < 1e-9 for x, y in pts):
                continue
            yield pts


def canonicalize_point(
    p: Tuple[float, float],
    a: Tuple[float, float],
    ux: float,
    uy: float,
    scale: float,
) -> Tuple[float, float]:
    dx = p[0] - a[0]
    dy = p[1] - a[1]
    x = (dx * ux + dy * uy) / scale
    y = (-dx * uy + dy * ux) / scale
    return (x, y)


def discover_annotation_files(data_root: Path, user_id: int) -> List[Path]:
    if user_id == 0:
        candidates = [
            data_root / "test_dataset" / "annotation",
            data_root / "test_dataset" / "one_experienced_singer_anno",
        ]
        for candidate in candidates:
            files = sorted(candidate.glob("us_*.csv"))
            if files:
                return files
        return []
    if user_id == 1:
        return sorted((data_root / "test_set_2" / "annotation").glob("us_*.csv"))
    if user_id == 2:
        return sorted((data_root / "test_set_3" / "annotation").glob("us_*.csv"))
    raise ValueError(f"Unsupported user_id: {user_id}")


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def learn_user_template(data_root: Path, user_id: int) -> Dict[str, float]:
    files = discover_annotation_files(data_root, user_id)
    if not files:
        raise FileNotFoundError(f"No annotation files found for user_id={user_id} under {data_root}")

    v12x_list: List[float] = []
    v12y_list: List[float] = []
    v34x_list: List[float] = []
    v34y_list: List[float] = []
    c_sign_list: List[float] = []
    p5x_list: List[float] = []
    p5y_list: List[float] = []
    reg_x: List[List[float]] = []
    reg_y: List[List[float]] = []

    n_rows = 0
    for csv_path in files:
        for pts in load_points_rows(csv_path):
            p1, p2, p3, p4, p5 = pts
            a = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            b = ((p3[0] + p4[0]) * 0.5, (p3[1] + p4[1]) * 0.5)
            abx = b[0] - a[0]
            aby = b[1] - a[1]
            ab_len = math.hypot(abx, aby)
            if ab_len < 1e-8:
                continue
            ux, uy = abx / ab_len, aby / ab_len

            p1c = canonicalize_point(p1, a, ux, uy, ab_len)
            p2c = canonicalize_point(p2, a, ux, uy, ab_len)
            p3c = canonicalize_point(p3, a, ux, uy, ab_len)
            p4c = canonicalize_point(p4, a, ux, uy, ab_len)
            p5c = canonicalize_point(p5, a, ux, uy, ab_len)

            # Angles in canonical space (A=(0,0), B=(1,0), C=p5c)
            cx, cy = p5c
            ac = math.hypot(cx, cy)
            bc = math.hypot(cx - 1.0, cy)
            if ac < 1e-8 or bc < 1e-8:
                continue
            cos_a = max(-1.0, min(1.0, cx / ac))
            cos_b = max(-1.0, min(1.0, (1.0 - cx) / bc))
            angle_a = math.degrees(math.acos(cos_a))
            angle_b = math.degrees(math.acos(cos_b))
            angle_c = 180.0 - angle_a - angle_b

            v12x_list.append((p2c[0] - p1c[0]) * 0.5)
            v12y_list.append((p2c[1] - p1c[1]) * 0.5)
            v34x_list.append((p4c[0] - p3c[0]) * 0.5)
            v34y_list.append((p4c[1] - p3c[1]) * 0.5)
            c_sign_list.append(p5c[1])
            p5x_list.append(p5c[0])
            p5y_list.append(p5c[1])

            feats = [1.0, angle_a, angle_b, angle_c]
            reg_x.append(feats)
            reg_y.append([p1c[0], p2c[0], p3c[0], p4c[0], p5c[0]])
            reg_y.append([p1c[1], p2c[1], p3c[1], p4c[1], p5c[1]])
            n_rows += 1

    if n_rows == 0:
        raise RuntimeError(f"No valid annotation rows found for user_id={user_id}")

    # Angle-conditioned linear model for moving points
    X = np.asarray(reg_x, dtype=np.float64)
    # Rebuild target matrices from collected canonical points
    # Separate collections to avoid shape mistakes
    # y_x: [p1_x,p2_x,p3_x,p4_x,p5_x], y_y: [p1_y,p2_y,p3_y,p4_y,p5_y]
    y_x_rows: List[List[float]] = []
    y_y_rows: List[List[float]] = []
    for csv_path in files:
        for pts in load_points_rows(csv_path):
            p1, p2, p3, p4, p5 = pts
            a = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            b = ((p3[0] + p4[0]) * 0.5, (p3[1] + p4[1]) * 0.5)
            abx = b[0] - a[0]
            aby = b[1] - a[1]
            ab_len = math.hypot(abx, aby)
            if ab_len < 1e-8:
                continue
            ux, uy = abx / ab_len, aby / ab_len
            p1c = canonicalize_point(p1, a, ux, uy, ab_len)
            p2c = canonicalize_point(p2, a, ux, uy, ab_len)
            p3c = canonicalize_point(p3, a, ux, uy, ab_len)
            p4c = canonicalize_point(p4, a, ux, uy, ab_len)
            p5c = canonicalize_point(p5, a, ux, uy, ab_len)
            cx, cy = p5c
            ac = math.hypot(cx, cy)
            bc = math.hypot(cx - 1.0, cy)
            if ac < 1e-8 or bc < 1e-8:
                continue
            y_x_rows.append([p1c[0], p2c[0], p3c[0], p4c[0], p5c[0]])
            y_y_rows.append([p1c[1], p2c[1], p3c[1], p4c[1], p5c[1]])
    Yx = np.asarray(y_x_rows, dtype=np.float64)
    Yy = np.asarray(y_y_rows, dtype=np.float64)
    Wx, *_ = np.linalg.lstsq(X, Yx, rcond=None)  # (4,5)
    Wy, *_ = np.linalg.lstsq(X, Yy, rcond=None)  # (4,5)

    return {
        "user_id": user_id,
        "dataset_name": USER_TO_DATASET[user_id],
        "num_clips": len(files),
        "num_rows": n_rows,
        "v12_x_mean": mean(v12x_list),
        "v12_y_mean": mean(v12y_list),
        "v34_x_mean": mean(v34x_list),
        "v34_y_mean": mean(v34y_list),
        "v12_x_std": std(v12x_list),
        "v12_y_std": std(v12y_list),
        "v34_x_std": std(v34x_list),
        "v34_y_std": std(v34y_list),
        "c_y_sign_mean": mean(c_sign_list),  # usually negative in image coordinates
        "p5_x_mean": mean(p5x_list),
        "p5_y_mean": mean(p5y_list),
        "linreg_wx": Wx.tolist(),
        "linreg_wy": Wy.tolist(),
    }


def read_angles_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        for r in reader:
            # Supports both naming styles:
            # 1) angle_a, angle_b, angle_c
            # 2) angle_a_deg, angle_b_deg, angle_c_deg
            angle_a = parse_float(r.get("angle_a", r.get("angle_a_deg", "")))
            angle_b = parse_float(r.get("angle_b", r.get("angle_b_deg", "")))
            angle_c = parse_float(r.get("angle_c", r.get("angle_c_deg", "")))
            if math.isnan(angle_a) or math.isnan(angle_b) or math.isnan(angle_c):
                continue
            frame_val = r.get("frame", r.get("frame_idx", ""))
            time_val = r.get("time", "")
            frame = int(float(frame_val)) if frame_val not in ("", None) else len(rows)
            time_sec = float(time_val) if time_val not in ("", None) else 0.0
            rows.append(
                {
                    "frame": frame,
                    "time": time_sec,
                    "angle_a": angle_a,
                    "angle_b": angle_b,
                    "angle_c": angle_c,
                }
            )
    if not rows:
        raise ValueError(f"No angle rows found in {path}")
    return rows


def solve_c_from_angles(angle_a_deg: float, angle_b_deg: float, c_y_sign_mean: float) -> Tuple[float, float]:
    # Canonical baseline: A=(0,0), B=(1,0)
    # With law of sines:
    # AC = sin(B) / sin(C), BC = sin(A) / sin(C), AB = 1
    a = math.radians(angle_a_deg)
    b = math.radians(angle_b_deg)
    c = math.pi - a - b

    if c <= 1e-6:
        # Degenerate, fallback to a near-line triangle
        return (0.5, -1e-3 if c_y_sign_mean < 0 else 1e-3)

    sin_c = math.sin(c)
    if abs(sin_c) < 1e-8:
        return (0.5, -1e-3 if c_y_sign_mean < 0 else 1e-3)

    ac = math.sin(b) / sin_c
    bc = math.sin(a) / sin_c

    cx = (ac * ac + 1.0 - bc * bc) * 0.5
    y_sq = max(ac * ac - cx * cx, 0.0)
    cy = math.sqrt(y_sq)
    if c_y_sign_mean < 0:
        cy = -cy
    return (cx, cy)


def reconstruct_rows(
    angle_rows: List[Dict[str, float]],
    template: Dict[str, float],
    p5_constant: bool = False,
) -> List[Dict[str, float]]:
    v12 = (template["v12_x_mean"], template["v12_y_mean"])
    v34 = (template["v34_x_mean"], template["v34_y_mean"])

    out: List[Dict[str, float]] = []
    wx = np.asarray(template.get("linreg_wx", []), dtype=np.float64)
    wy = np.asarray(template.get("linreg_wy", []), dtype=np.float64)
    for row in angle_rows:
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        c_moving = solve_c_from_angles(row["angle_a"], row["angle_b"], template["c_y_sign_mean"])
        c = (template["p5_x_mean"], template["p5_y_mean"]) if p5_constant else c_moving

        # Predict p1..p4 from angles if linear model is available; fallback to template means.
        if wx.size and wy.size:
            feat = np.asarray([1.0, row["angle_a"], row["angle_b"], row["angle_c"]], dtype=np.float64)
            pred_x = feat @ wx  # (5,)
            pred_y = feat @ wy  # (5,)
            p1 = (float(pred_x[0]), float(pred_y[0]))
            p2 = (float(pred_x[1]), float(pred_y[1]))
            p3 = (float(pred_x[2]), float(pred_y[2]))
            p4 = (float(pred_x[3]), float(pred_y[3]))
            p5 = c if p5_constant else (float(pred_x[4]), float(pred_y[4]))
        else:
            p1 = (a[0] - v12[0], a[1] - v12[1])
            p2 = (a[0] + v12[0], a[1] + v12[1])
            p3 = (b[0] - v34[0], b[1] - v34[1])
            p4 = (b[0] + v34[0], b[1] + v34[1])
            p5 = c

        out.append(
            {
                "frame": row["frame"],
                "time": row["time"],
                "p1_x": p1[0],
                "p1_y": p1[1],
                "p2_x": p2[0],
                "p2_y": p2[1],
                "p3_x": p3[0],
                "p3_y": p3[1],
                "p4_x": p4[0],
                "p4_y": p4[1],
                "p5_x": p5[0],
                "p5_y": p5[1],
                "tri_a_x": a[0],
                "tri_a_y": a[1],
                "tri_b_x": b[0],
                "tri_b_y": b[1],
                "tri_c_x": c[0],
                "tri_c_y": c[1],
                "angle_a": row["angle_a"],
                "angle_b": row["angle_b"],
                "angle_c": row["angle_c"],
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct canonical 5-point markers from 3 angles using user-specific priors."
    )
    parser.add_argument("--angles_csv", required=True, help="Input angles CSV.")
    parser.add_argument("--output_csv", required=True, help="Output reconstructed points CSV.")
    parser.add_argument(
        "--user_id",
        required=True,
        type=int,
        choices=[0, 1, 2],
        help="Singer ID: 0=test_dataset, 1=test_set_2, 2=test_set_3.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Data root containing test_dataset/test_set_2/test_set_3.",
    )
    parser.add_argument(
        "--save_template_json",
        default=None,
        help="Optional path to save learned user template JSON.",
    )
    parser.add_argument(
        "--p5_constant",
        action="store_true",
        help="Force point 5 to stay constant (user mean), while other points remain angle-conditioned.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    template = learn_user_template(data_root, args.user_id)
    angle_rows = read_angles_csv(Path(args.angles_csv))
    out_rows = reconstruct_rows(angle_rows, template, p5_constant=args.p5_constant)
    write_csv(Path(args.output_csv), out_rows)

    if args.save_template_json:
        out_json = Path(args.save_template_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)

    print(
        f"Saved reconstructed points: {args.output_csv}\n"
        f"user_id={args.user_id} ({template['dataset_name']}) | "
        f"clips={template['num_clips']} rows={template['num_rows']}"
    )


if __name__ == "__main__":
    main()
