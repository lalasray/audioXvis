"""Prepare standardized songs/audio/annotation/gt folders for test sets."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import shutil
from pathlib import Path

def normalize_stem(name: str) -> str:
    stem = name.strip()
    if stem.startswith("s") and "_" not in stem and len(stem) >= 3 and stem[1:].isdigit():
        return f"s{int(stem[1:]):02d}"
    return stem


def copy_songs(src_dir: Path, dst_dir: Path) -> dict[str, Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    songs: dict[str, Path] = {}
    for src in sorted(src_dir.glob("*")):
        if not src.is_file() or src.suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
            continue
        stem = normalize_stem(src.stem)
        dst = dst_dir / f"{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        songs[stem] = dst
    return songs


def copy_annotations(src_dir: Path, dst_dir: Path) -> dict[str, Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    annos: dict[str, Path] = {}
    for src in sorted(src_dir.glob("*.csv")):
        if src.name.lower() == "annotation.csv":
            continue
        stem = normalize_stem(src.stem)
        dst = dst_dir / f"us_{stem}.csv"
        shutil.copy2(src, dst)
        annos[stem] = dst
    return annos


def find_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found in PATH.")
    return ffmpeg_path


def find_ffprobe() -> str:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise FileNotFoundError("ffprobe not found in PATH.")
    return ffprobe_path


def batch_extract_audio(video_dir: Path, audio_dir: Path) -> int:
    ffmpeg_bin = find_ffmpeg()
    audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for video_path in sorted(video_dir.glob("*")):
        if not video_path.is_file():
            continue
        out_path = audio_dir / f"{video_path.stem}.mp3"
        cmd = [
            ffmpeg_bin,
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-y",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed audio extraction: {video_path.name}")
            continue
        count += 1
    return count


def get_video_frame_count(video_path: Path) -> int | None:
    ffprobe_bin = find_ffprobe()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            value = int(line)
            if value > 0:
                return value
    return None


def load_points_csv(csv_path: Path) -> list[list[list[float]]]:
    points: list[list[list[float]]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            vals = [float(v) for v in row]
            if len(vals) != 10:
                raise ValueError(f"{csv_path}: expected 10 columns, got {len(vals)}")
            pts = [[vals[i], vals[i + 1]] for i in range(0, 10, 2)]
            points.append(pts)
    return points


def linear_interp_1d(x: list[float], y: list[float], x_new: list[float]) -> list[float]:
    out: list[float] = []
    j = 0
    for xn in x_new:
        while j + 1 < len(x) and x[j + 1] < xn:
            j += 1
        if xn <= x[0]:
            out.append(y[0])
        elif xn >= x[-1]:
            out.append(y[-1])
        else:
            x0, x1 = x[j], x[j + 1]
            y0, y1 = y[j], y[j + 1]
            t = 0.0 if x1 == x0 else (xn - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out


def interpolate_points_to_frame_count(points: list[list[list[float]]], target_frames: int) -> list[list[list[float]]]:
    n_frames = len(points)
    if n_frames == 0:
        return points
    if target_frames <= 1:
        return [points[0]]

    src_t = [float(i) for i in range(n_frames)]
    dst_t = [i * (n_frames - 1) / (target_frames - 1) for i in range(target_frames)]

    dense = [[[0.0, 0.0] for _ in range(5)] for _ in range(target_frames)]

    for p in range(5):
        valid_idx = [
            i
            for i in range(n_frames)
            if (abs(points[i][p][0]) > 1e-9 or abs(points[i][p][1]) > 1e-9)
        ]
        if not valid_idx:
            continue

        valid_t = [src_t[i] for i in valid_idx]
        for axis in [0, 1]:
            valid_v = [points[i][p][axis] for i in valid_idx]
            filled = linear_interp_1d(valid_t, valid_v, src_t)
            interp = linear_interp_1d(src_t, filled, dst_t)
            for i in range(target_frames):
                dense[i][p][axis] = interp[i]

    return dense


def euclid(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(vertex: list[float], a: list[float], b: list[float]) -> float:
    v1 = [a[0] - vertex[0], a[1] - vertex[1]]
    v2 = [b[0] - vertex[0], b[1] - vertex[1]]
    n1 = max(math.hypot(v1[0], v1[1]), 1e-8)
    n2 = max(math.hypot(v2[0], v2[1]), 1e-8)
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def extract_geometry_rows(points: list[list[list[float]]]) -> tuple[list[str], list[list[float]]]:
    rows: list[list[float]] = []
    for idx, frame in enumerate(points):
        p1, p2, p3, p4, p5 = frame
        mid_12 = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5]
        mid_34 = [(p3[0] + p4[0]) * 0.5, (p3[1] + p4[1]) * 0.5]

        side_a_b = euclid(mid_12, mid_34)
        side_b_c = euclid(mid_34, p5)
        side_c_a = euclid(p5, mid_12)

        angle_a = angle_deg(mid_12, mid_34, p5)
        angle_b = angle_deg(mid_34, p5, mid_12)
        angle_c = angle_deg(p5, mid_12, mid_34)

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
                side_a_b, side_b_c, side_c_a,
                angle_a, angle_b, angle_c,
                angle_a + angle_b + angle_c,
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
        "side_a_b", "side_b_c", "side_c_a",
        "angle_a_deg", "angle_b_deg", "angle_c_deg", "angle_sum_deg",
    ]
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_gt(annos: dict[str, Path], songs: dict[str, Path], gt_dir: Path) -> int:
    gt_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for stem, anno_path in annos.items():
        points = load_points_csv(anno_path)
        video_path = songs.get(stem)
        if video_path is not None and video_path.exists():
            target_frames = get_video_frame_count(video_path)
            if target_frames:
                points = interpolate_points_to_frame_count(points, target_frames=target_frames)
        header, rows = extract_geometry_rows(points)
        out_csv = gt_dir / f"gt_{stem}.csv"
        write_csv(out_csv, header, rows)
        count += 1
    return count


def prepare_one_set(set_root: Path, songs_src_name: str, anno_src_name: str, force: bool = False) -> None:
    songs_src = set_root / songs_src_name
    anno_src = set_root / anno_src_name
    if not songs_src.is_dir():
        candidates = [d for d in sorted(set_root.iterdir()) if d.is_dir() and "song" in d.name.lower()]
        if candidates:
            songs_src = candidates[0]
    if not anno_src.is_dir():
        candidates = [d for d in sorted(set_root.iterdir()) if d.is_dir() and "anno" in d.name.lower()]
        if candidates:
            anno_src = candidates[0]
    if not songs_src.is_dir():
        raise FileNotFoundError(f"Songs source folder not found: {songs_src}")
    if not anno_src.is_dir():
        raise FileNotFoundError(f"Annotation source folder not found: {anno_src}")

    songs_dst = set_root / "songs"
    audio_dst = set_root / "audio"
    anno_dst = set_root / "annotation"
    gt_dst = set_root / "gt"

    for out_dir in [songs_dst, audio_dst, anno_dst, gt_dst]:
        if out_dir.exists() and force:
            shutil.rmtree(out_dir)

    songs = copy_songs(songs_src, songs_dst)
    annos = copy_annotations(anno_src, anno_dst)
    extracted_audio_count = batch_extract_audio(songs_dst, audio_dst)
    gt_count = generate_gt(annos=annos, songs=songs, gt_dir=gt_dst)

    print(f"\nPrepared: {set_root}")
    print(f"  songs: {len(songs)}")
    print(f"  audio: {extracted_audio_count}")
    print(f"  annotation: {len(annos)}")
    print(f"  gt: {gt_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create songs/audio/annotation/gt folders for test_set_2 and test_set_3."
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root data directory containing test_set_2 and test_set_3.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output folders (songs/audio/annotation/gt) before regenerating.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    prepare_one_set(data_root / "test_set_2", "__auto_songs__", "__auto_anno__", force=args.force)
    prepare_one_set(data_root / "test_set_3", "__auto_songs__", "__auto_anno__", force=args.force)


if __name__ == "__main__":
    main()
