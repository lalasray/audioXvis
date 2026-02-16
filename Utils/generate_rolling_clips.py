"""
Generate rolling audio clips from all audio files in a directory.

Behavior:
- Compute durations of all audio files in the input directory
- Compute the Nth percentile (default 70) of durations
- For window sizes 1..floor(percentile) seconds, generate rolling clips
  for any file with duration >= window_size
- Clips are saved under output_dir/<window_s>s/<orig_stem>/
- Default hop fraction is 0.5 (50% overlap)

Usage examples:
python generate_rolling_clips.py --audio-dir /path/to/audio/ --output-dir ./clips/ --min-percentile 70 --hop-frac 0.1 --dry-run

"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import math


def find_audio_files(audio_dir, exts=None):
    if exts is None:
        exts = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    p = Path(audio_dir)
    files = []
    for ext in exts:
        files.extend(p.glob(f'*{ext}'))
        files.extend(p.glob(f'*{ext.upper()}'))
    return sorted(files)


def compute_durations(files, sr=None):
    durations = []
    for f in files:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', required=True, help='Directory with audio files')
    parser.add_argument('--output-dir', default='./rolling_clips', help='Directory to save clips')
    parser.add_argument('--min-percentile', type=float, default=70.0, help='Percentile to use for max window (default 70)')
    parser.add_argument('--hop-frac', type=float, default=0.5, help='Hop fraction relative to window length (default 0.5)')
    parser.add_argument('--dry-run', action='store_true', help='Only print planned actions, do not write files')
    parser.add_argument('--sr', type=int, default=None, help='Resample target sample rate (None = keep original)')
    parser.add_argument('--min-window', type=int, default=1, help='Minimum window size in seconds to generate (default 1)')
    parser.add_argument('--max-window', type=int, default=10, help='Maximum window size in seconds (overrides percentile when set)')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    files = find_audio_files(audio_dir)
    if not files:
        print(f'No audio files found in {audio_dir}')
        return

    print(f'Found {len(files)} audio files. Computing durations...')
    durations = compute_durations(files, sr=args.sr)
    for f, d in zip(files, durations):
        print(f'{f.name}: {d:.2f}s')

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

    total_clips = 0
    plan = {}
    for f, dur in zip(files, durations):
        stem = f.stem
        if args.sr is None:
            # load at native sr
            y, sr = librosa.load(str(f), sr=None)
        else:
            y, sr = librosa.load(str(f), sr=args.sr)
        per_file_counts = {}
        for win in range(args.min_window, max_window + 1):
            # Only generate clips for files strictly longer than the window
            if dur > win:
                cnt = generate_clips_for_file(y, sr, out_dir, stem, win, args.hop_frac, dry_run=args.dry_run)
                per_file_counts[f'{win}s'] = cnt
                total_clips += cnt
        plan[f.name] = per_file_counts
        # free memory
        del y

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
