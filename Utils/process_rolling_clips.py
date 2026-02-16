#!/usr/bin/env python3
"""
Process rolling clips: extract features and save plots.

Saves per-clip features as compressed NPZ files and per-clip plots (PNG).
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from audio_feature_extraction import AudioFeatureExtractor


def process_clips(clips_dir, out_dir, sr=None, plot_all=True):
    clips_dir = Path(clips_dir)
    out_dir = Path(out_dir)
    feats_dir = out_dir / 'features'
    plots_dir = out_dir / 'plots'
    feats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # gather wav files recursively
    files = list(clips_dir.rglob('*.wav'))
    if not files:
        print(f'No .wav files found in {clips_dir}')
        return

    print(f'Found {len(files)} clips. Processing...')

    for f in tqdm(files, desc='clips'):
        try:
            rel = f.relative_to(clips_dir)
            stem = rel.with_suffix('')

            # create per-file output subdirs mirroring clips structure
            file_feats_dir = feats_dir / stem.parent
            file_plots_dir = plots_dir / stem.parent
            file_feats_dir.mkdir(parents=True, exist_ok=True)
            file_plots_dir.mkdir(parents=True, exist_ok=True)

            extractor = AudioFeatureExtractor(str(f), sr=sr or extractor_default_sr())

            # extract features and save compressed
            features = extractor.extract_all_features()
            out_npz = file_feats_dir / f"{f.stem}.npz"
            # convert any numpy-unfriendly objects to arrays where possible
            serializable = {}
            for k, v in features.items():
                try:
                    serializable[k] = np.asarray(v)
                except Exception:
                    # fallback: skip non-serializable entries
                    pass
            np.savez_compressed(str(out_npz), **serializable)

            # save plots
            extractor.plot_all_features(output_dir=str(file_plots_dir))

        except Exception as e:
            print(f'Error processing {f}: {e}')
            continue


def extractor_default_sr():
    return 22050


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips-dir', required=True, help='Directory with rolling clips')
    parser.add_argument('--out-dir', default='./rolling_clips_features', help='Directory to save features and plots')
    parser.add_argument('--sr', type=int, default=None, help='Target sample rate (None = keep original)')
    parser.add_argument('--no-plots', dest='plot_all', action='store_false', help='Do not save plots, only features')
    args = parser.parse_args()

    process_clips(args.clips_dir, args.out_dir, sr=args.sr, plot_all=args.plot_all)


if __name__ == '__main__':
    main()
