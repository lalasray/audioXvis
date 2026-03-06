"""Generate sliding-window video clips and aligned annotation CSV clips for model datasets."""

import argparse
import csv
import math
import subprocess
from pathlib import Path

import numpy as np
from media_utils import VIDEO_EXTS, get_video_capture_props, resolve_ffmpeg_path


def load_annotation_csv(annotation_csv: str):
    with open(annotation_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Annotation CSV has no header.")
        rows = list(reader)
        header = list(reader.fieldnames)

    if len(rows) == 0:
        raise ValueError("Annotation CSV has no rows.")

    return header, rows


def build_annotation_timestamps(rows: list[dict], video_fps: float, video_duration_sec: float) -> np.ndarray:
    n_rows = len(rows)

    if "frame_idx" in rows[0]:
        try:
            frame_idxs = np.array([float(r["frame_idx"]) for r in rows], dtype=np.float64)
            if np.all(np.diff(frame_idxs) >= 0):
                return frame_idxs / video_fps
        except Exception:
            pass

    return np.linspace(0.0, video_duration_sec, n_rows, endpoint=False, dtype=np.float64)


def write_rows_csv(path: Path, header: list[str], rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def cut_video_clip_with_audio(
    ffmpeg_path: str,
    input_video: str,
    output_video: str,
    start_sec: float,
    duration_sec: float,
):
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        input_video,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_video,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for clip: {output_video}")


def find_matching_video(video_dir: Path, base_stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{base_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def derive_video_stem_from_annotation(annotation_stem: str) -> str:
    if annotation_stem.startswith("gt_"):
        return annotation_stem[3:]
    if annotation_stem.startswith("us_"):
        return annotation_stem[3:]
    return annotation_stem


def collect_pairs(
    input_video: str | None,
    annotation_csv: str | None,
    video_dir: str | None,
    annotation_dir: str | None,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    single_mode = bool(input_video or annotation_csv)
    batch_mode = bool(video_dir or annotation_dir)

    if single_mode and batch_mode:
        raise ValueError("Use either single mode (-v and -a) or batch mode (--video_dir and --annotation_dir), not both.")
    if single_mode and not (input_video and annotation_csv):
        raise ValueError("Single mode requires both --input_video and --annotation_csv.")
    if batch_mode and not (video_dir and annotation_dir):
        raise ValueError("Batch mode requires both --video_dir and --annotation_dir.")
    if not single_mode and not batch_mode:
        raise ValueError("Provide either (-v and -a) for one pair, or (--video_dir and --annotation_dir) for batch mode.")

    if single_mode:
        video_path = Path(input_video)
        anno_path = Path(annotation_csv)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        return [(video_path, anno_path)], []

    videos_dir = Path(video_dir)
    annos_dir = Path(annotation_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if not annos_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annos_dir}")

    annotations = sorted(annos_dir.glob("*.csv"))
    if not annotations:
        raise RuntimeError(f"No CSV files found in annotation directory: {annos_dir}")

    pair_list: list[tuple[Path, Path]] = []
    skipped = []

    for annotation_path in annotations:
        video_stem = derive_video_stem_from_annotation(annotation_path.stem)
        video_path = find_matching_video(videos_dir, video_stem)
        if video_path is None:
            skipped.append(annotation_path.name)
            continue
        pair_list.append((video_path, annotation_path))

    if not pair_list:
        raise RuntimeError("No matching video/annotation pairs found.")

    return pair_list, skipped


def build_clip_specs(
    video_stem: str,
    duration_sec: float,
    window_sec: float,
    hop_sec: float,
    max_clips: int | None,
    full_clip: bool,
) -> list[dict]:
    if full_clip:
        return [
            {
                "clip_index": 0,
                "clip_name": f"full_{video_stem}",
                "start_sec": 0.0,
                "end_sec": duration_sec,
                "duration_sec": duration_sec,
            }
        ]

    max_start = duration_sec - window_sec
    n_steps = int(math.floor(max_start / hop_sec)) + 1
    specs = []

    for i in range(n_steps):
        if max_clips is not None and i >= max_clips:
            break

        start_sec = i * hop_sec
        end_sec = start_sec + window_sec
        if end_sec > duration_sec + 1e-9:
            break

        specs.append(
            {
                "clip_index": i,
                "clip_name": f"clip_{i:06d}_start{int(round(start_sec * 1000)):07d}ms_end{int(round(end_sec * 1000)):07d}ms",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": window_sec,
            }
        )

    return specs


def process_one_pair(
    ffmpeg_path: str,
    video_path: Path,
    annotation_path: Path,
    out_root: Path,
    window_sec: float,
    hop_sec: float,
    max_clips: int | None,
    full_clip: bool,
) -> dict:
    fps, _, _, frame_count, duration_sec = get_video_capture_props(str(video_path))
    header, rows = load_annotation_csv(str(annotation_path))

    anno_t = build_annotation_timestamps(rows, video_fps=fps, video_duration_sec=duration_sec)

    if (not full_clip) and duration_sec < window_sec:
        raise RuntimeError(
            f"Video duration ({duration_sec:.3f}s) is shorter than window ({window_sec:.3f}s)."
        )

    pair_name = f"{video_path.stem}__{annotation_path.stem}"
    pair_dir = out_root / pair_name
    video_dir = pair_dir / "video"
    annotation_dir = pair_dir / "annotation"

    video_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    clip_specs = build_clip_specs(
        video_stem=video_path.stem,
        duration_sec=duration_sec,
        window_sec=window_sec,
        hop_sec=hop_sec,
        max_clips=max_clips,
        full_clip=full_clip,
    )

    metadata_path = pair_dir / "clips_metadata.csv"
    metadata_header = [
        "clip_index",
        "clip_name",
        "video_clip_path",
        "annotation_clip_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "annotation_rows",
    ]

    meta_rows = []
    skipped_existing = 0

    print(f"Video: {video_path.name}")
    print(f"Annotation: {annotation_path.name}")
    print(f"FPS: {fps:.3f}, video frames: {frame_count}, duration: {duration_sec:.3f}s")
    print(f"Annotation rows: {len(rows)}")
    if full_clip:
        print("Mode: full_clip (single full-length clip)")
    else:
        print(f"Window: {window_sec:.3f}s, hop: {hop_sec:.3f}s, clips to generate: {len(clip_specs)}")

    for clip in clip_specs:
        clip_name = clip["clip_name"]
        start_sec = float(clip["start_sec"])
        end_sec = float(clip["end_sec"])
        clip_duration = float(clip["duration_sec"])

        video_out = video_dir / f"{clip_name}.mp4"
        anno_out = annotation_dir / f"{clip_name}.csv"

        if video_out.exists() and anno_out.exists():
            skipped_existing += 1
            continue

        cut_video_clip_with_audio(
            ffmpeg_path=ffmpeg_path,
            input_video=str(video_path),
            output_video=str(video_out),
            start_sec=start_sec,
            duration_sec=clip_duration,
        )

        if full_clip:
            clip_rows = rows
        else:
            in_window = (anno_t >= start_sec) & (anno_t < end_sec)
            clip_rows = [rows[idx] for idx in np.flatnonzero(in_window)]
        write_rows_csv(anno_out, header, clip_rows)

        meta_rows.append(
            {
                "clip_index": int(clip["clip_index"]),
                "clip_name": clip_name,
                "video_clip_path": str(video_out),
                "annotation_clip_path": str(anno_out),
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "duration_sec": f"{clip_duration:.3f}",
                "annotation_rows": len(clip_rows),
            }
        )

    write_rows_csv(metadata_path, metadata_header, meta_rows)

    print(f"Saved clips under: {pair_dir}")
    print(f"Total clips written: {len(meta_rows)}")
    print(f"Existing clips skipped: {skipped_existing}")
    print(f"Metadata: {metadata_path}")

    return {
        "pair_name": pair_name,
        "clips": len(meta_rows),
        "metadata": str(metadata_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate sliding-window 3-second video clips (with audio) and matching annotation CSV clips "
            "from one pair or all auto-matched pairs in directories."
        )
    )
    parser.add_argument("-v", "--input_video", required=False, help="Path to one input video file.")
    parser.add_argument("-a", "--annotation_csv", required=False, help="Path to one annotation CSV file.")
    parser.add_argument("--video_dir", required=False, help="Directory of videos for batch auto-pair mode.")
    parser.add_argument("--annotation_dir", required=False, help="Directory of annotation CSVs for batch auto-pair mode.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save generated clips.")
    parser.add_argument("--window_sec", type=float, default=3.0, help="Window length in seconds (default: 3.0).")
    parser.add_argument("--hop_sec", type=float, default=0.3, help="Sliding hop in seconds (default: 0.3).")
    parser.add_argument(
        "--full_clip",
        action="store_true",
        help="Generate one full-length clip per matched pair (ignores --window_sec and --hop_sec).",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Optional maximum number of clips to generate (default: all).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default=None,
        help="Path to ffmpeg executable (optional if ffmpeg is in PATH).",
    )

    args = parser.parse_args()

    if not args.full_clip:
        if args.window_sec <= 0:
            raise ValueError("--window_sec must be > 0")
        if args.hop_sec <= 0:
            raise ValueError("--hop_sec must be > 0")

    pair_list, skipped = collect_pairs(
        input_video=args.input_video,
        annotation_csv=args.annotation_csv,
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
    )

    ffmpeg_path = resolve_ffmpeg_path(args.ffmpeg_path)
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pass --ffmpeg_path.")

    out_root = Path(args.output_dir)

    results = []

    if args.video_dir and args.annotation_dir:
        print(f"Matched pairs: {len(pair_list)}")
        if skipped:
            print(f"Skipped (no video match): {len(skipped)}")

    for pair_idx, (video_path, annotation_path) in enumerate(pair_list, start=1):
        print("=" * 80)
        print(f"Processing pair {pair_idx}/{len(pair_list)}")
        result = process_one_pair(
            ffmpeg_path=ffmpeg_path,
            video_path=video_path,
            annotation_path=annotation_path,
            out_root=out_root,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            max_clips=args.max_clips,
            full_clip=args.full_clip,
        )
        results.append(result)

    print("=" * 80)
    print(f"Completed pairs: {len(results)}")
    print(f"Total clips: {sum(r['clips'] for r in results)}")


if __name__ == "__main__":
    main()
