"""Extract all audio features from every full clip and save as plots."""

import sys
from pathlib import Path

# Ensure Utils/ is on import path so audio_feature_extraction can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_feature_extraction import AudioFeatureExtractor

FULL_CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset" / "full_clips"


def main():
    clip_dirs = sorted(
        d for d in FULL_CLIPS_DIR.iterdir()
        if d.is_dir() and (d / "video").is_dir()
    )
    if not clip_dirs:
        print(f"No clip directories found under {FULL_CLIPS_DIR}")
        return

    plots_root = FULL_CLIPS_DIR / "audio_feature_plots"
    plots_root.mkdir(exist_ok=True)

    for i, clip_dir in enumerate(clip_dirs, 1):
        clip_name = clip_dir.name
        video_dir = clip_dir / "video"
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            print(f"[{i}/{len(clip_dirs)}] No mp4 in {video_dir}, skipping.")
            continue

        video_path = videos[0]
        out_dir = plots_root / clip_name
        out_dir.mkdir(exist_ok=True)

        print(f"\n[{i}/{len(clip_dirs)}] {clip_name}")
        print(f"  Video: {video_path.name}")

        try:
            extractor = AudioFeatureExtractor(str(video_path), sr=22050)
            extractor.plot_all_features(output_dir=str(out_dir))
            print(f"  -> Saved {len(list(out_dir.glob('*.png')))} plots to {out_dir.name}/")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll audio feature plots saved under: {plots_root}")


if __name__ == "__main__":
    main()
