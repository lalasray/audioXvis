"""
Generate rolling clips from media files in a directory.

Behavior:
- Compute durations of all media files in the input directory
- Compute the Nth percentile (default 70) of durations (unless explicit windows provided)
- For window sizes 1..floor(percentile) seconds (or user-specified list), generate rolling clips
    for any file with duration >= window_size
- Audio input -> output WAV clips
- Video input -> output video clips with audio (requires ffmpeg)
Clips are saved under output_dir/<window_s>s/<orig_stem>/
Default hop fraction is 0.5 (50% overlap)

Usage examples:
python generate_rolling_clips.py --audio-dir /path/to/audio/ --output-dir ./clips/ --min-percentile 70 --hop-frac 0.1 --dry-run
# explicit windows only:
#+ python generate_rolling_clips.py --audio-dir /path/to/audio/ --windows 1,2,5 --dry-run

# split videos with audio:
#+ python generate_rolling_clips.py --audio-dir ./data/test_dataset --media-type video --windows 1,2,5

"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import math
import subprocess

from media_utils import AUDIO_EXTS, VIDEO_EXTS, get_video_duration_ffprobe, resolve_ffmpeg_path, resolve_ffprobe_path


def find_media_files(input_dir, media_type='auto'):
    if media_type == 'audio':
        exts = AUDIO_EXTS
    elif media_type == 'video':
        exts = VIDEO_EXTS
    else:
        exts = AUDIO_EXTS + VIDEO_EXTS

    p = Path(input_dir)
    files = []
    for ext in exts:
        files.extend(p.glob(f'*{ext}'))
        files.extend(p.glob(f'*{ext.upper()}'))

    files = sorted(set(files))
    media_map = {}
    for f in files:
        suffix = f.suffix.lower()
        if suffix in VIDEO_EXTS:
            media_map[f] = 'video'
        else:
            media_map[f] = 'audio'
    return files, media_map


def compute_durations(files, media_map, sr=None, ffprobe_path='ffprobe'):
    durations = []
    for f in files:
        kind = media_map.get(f, 'audio')
        if kind == 'video':
            dur = get_video_duration_ffprobe(f, ffprobe_path=ffprobe_path)
        else:
            try:
                # librosa.get_duration can read file without loading full audio
                dur = librosa.get_duration(filename=str(f))
            except Exception:
                # fallback to load
                y, _ = librosa.load(str(f), sr=sr)
                dur = librosa.get_duration(y=y, sr=_)
        durations.append(dur)
    return np.array(durations)


def generate_clips_for_file(y, sr, out_dir, stem, win_sec, hop_frac, dry_run=False):
    samples_per_win = int(win_sec * sr)
    if samples_per_win <= 0:
        return 0
    hop_samples = max(1, int(math.floor(hop_frac * samples_per_win)))
    total_samples = len(y)
    count = 0
    if total_samples < samples_per_win:
        return 0
    # Create subdirectory
    subdir = out_dir / f"{win_sec}s" / stem
    if not dry_run:
        subdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, total_samples - samples_per_win + 1, hop_samples):
        clip = y[start:start + samples_per_win]
        start_sec = start / sr
        end_sec = (start + samples_per_win) / sr
        out_name = f"{stem}_{win_sec}s_start{int(start_sec*1000)}ms_end{int(end_sec*1000)}ms.wav"
        out_path = subdir / out_name
        if not dry_run:
            sf.write(str(out_path), clip, sr)
        count += 1
    return count


def generate_video_clips_for_file(video_path, duration_sec, ffmpeg_path, out_dir, stem, win_sec, hop_frac, dry_run=False):
    if duration_sec < win_sec:
        return 0

    hop_sec = max(0.001, hop_frac * float(win_sec))
    subdir = out_dir / f"{win_sec}s" / stem
    if not dry_run:
        subdir.mkdir(parents=True, exist_ok=True)

    count = 0
    start = 0.0
    while start + win_sec <= duration_sec + 1e-9:
        end = start + win_sec
        out_name = f"{stem}_{win_sec}s_start{int(start*1000)}ms_end{int(end*1000)}ms.mp4"
        out_path = subdir / out_name
        if not dry_run:
            cmd = [
                ffmpeg_path,
                '-y',
                '-ss', f'{start:.3f}',
                '-i', str(video_path),
                '-t', f'{win_sec:.3f}',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                str(out_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'ffmpeg failed for {video_path.name}: {result.stderr.strip()}')
        count += 1
        start += hop_sec
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', '--input-dir', dest='audio_dir', required=True,
                        help='Directory with media files (audio/video)')
    parser.add_argument('--output-dir', default='./rolling_clips', help='Directory to save clips')
    parser.add_argument('--min-percentile', type=float, default=70.0, help='Percentile to use for max window (default 70)')
    parser.add_argument('--hop-frac', type=float, default=0.5, help='Hop fraction relative to window length (default 0.5)')
    parser.add_argument('--dry-run', action='store_true', help='Only print planned actions, do not write files')
    parser.add_argument('--sr', type=int, default=None, help='Resample target sample rate (None = keep original)')
    parser.add_argument('--min-window', type=int, default=1, help='Minimum window size in seconds to generate (default 1)')
    parser.add_argument('--max-window', type=int, default=1, help='Maximum window size in seconds (default 1; overrides percentile when set)')
    parser.add_argument('--windows', type=str, default=None,
                        help='Comma-separated list of explicit window sizes in seconds (overrides min/max/percentile)')
    parser.add_argument('--media-type', choices=['auto', 'audio', 'video'], default='auto',
                        help='Input media type: auto detect, audio only, or video only (default auto)')
    parser.add_argument('--ffmpeg-path', type=str, default=None,
                        help='Path to ffmpeg executable (required for video splitting if ffmpeg is not in PATH)')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    files, media_map = find_media_files(audio_dir, media_type=args.media_type)
    if not files:
        print(f'No media files found in {audio_dir} for media-type={args.media_type}')
        return

    ffmpeg_path = None
    ffprobe_path = 'ffprobe'
    if any(media_map[f] == 'video' for f in files):
        ffmpeg_path = resolve_ffmpeg_path(args.ffmpeg_path)
        if ffmpeg_path is None:
            print('Video files detected but ffmpeg was not found. Install ffmpeg or pass --ffmpeg-path.')
            return
        ffprobe_path = resolve_ffprobe_path(ffmpeg_path)

    print(f'Found {len(files)} media files. Computing durations...')
    durations = compute_durations(files, media_map, sr=args.sr, ffprobe_path=ffprobe_path)
    for f, d in zip(files, durations):
        print(f'{f.name} [{media_map[f]}]: {d:.2f}s')

    # determine which window sizes to generate
    if args.windows:
        try:
            windows = sorted({int(w) for w in args.windows.split(',') if w.strip()})
        except ValueError:
            print(f'Invalid value for --windows: {args.windows}')
            return
        if not windows:
            print('No valid window sizes parsed from --windows')
            return
        print(f'Using explicit windows list: {windows}')
    else:
        if args.max_window is not None:
            max_window = int(args.max_window)
            print(f'Using explicit --max-window {max_window}s. Generating windows {args.min_window}s..{max_window}s')
        else:
            perc = np.percentile(durations, args.min_percentile)
            max_window = int(math.floor(perc))
            if max_window < args.min_window:
                print(f'{args.min_percentile}th percentile = {perc:.2f}s less than min-window {args.min_window}s. Using min-window.')
                max_window = args.min_window
            print(f'Using percentile {args.min_percentile} -> {perc:.2f}s. Generating windows {args.min_window}s..{max_window}s')
        windows = list(range(args.min_window, max_window + 1))

    total_clips = 0
    plan = {}
    for f, dur in zip(files, durations):
        stem = f.stem
        per_file_counts = {}
        if media_map[f] == 'audio':
            if args.sr is None:
                # load at native sr
                y, sr = librosa.load(str(f), sr=None)
            else:
                y, sr = librosa.load(str(f), sr=args.sr)
            for win in windows:
                # Only generate clips for files strictly longer than the window
                if dur > win:
                    cnt = generate_clips_for_file(y, sr, out_dir, stem, win, args.hop_frac, dry_run=args.dry_run)
                    per_file_counts[f'{win}s'] = cnt
                    total_clips += cnt
            del y
        else:
            for win in windows:
                # Only generate clips for files strictly longer than the window
                if dur > win:
                    cnt = generate_video_clips_for_file(
                        video_path=f,
                        duration_sec=float(dur),
                        ffmpeg_path=ffmpeg_path,
                        out_dir=out_dir,
                        stem=stem,
                        win_sec=win,
                        hop_frac=args.hop_frac,
                        dry_run=args.dry_run
                    )
                    per_file_counts[f'{win}s'] = cnt
                    total_clips += cnt
        plan[f.name] = per_file_counts

    print('\nGeneration plan summary:')
    for fname, counts in plan.items():
        print(f'  {fname}:')
        for w, c in counts.items():
            print(f'    {w}: {c} clips')

    print(f'\nTotal clips (estimated/written): {total_clips}')
    if args.dry_run:
        print('Dry-run mode: no files were written')
    else:
        print(f'Clips saved under: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
