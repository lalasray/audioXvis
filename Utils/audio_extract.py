"""Audio extraction utility for extracting audio tracks from video files."""

import argparse
import os
import subprocess
from pathlib import Path

from media_utils import VIDEO_EXTS, resolve_ffmpeg_path


AUDIO_CODEC_MAP = {
    "mp3": "libmp3lame",
    "wav": "pcm_s16le",
    "aac": "aac",
    "flac": "flac",
}


def extract_audio_from_video(video_path, output_path=None, audio_format="mp3", ffmpeg_path=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_format = audio_format.lower()
    if audio_format not in AUDIO_CODEC_MAP:
        raise ValueError(f"Unsupported audio format. Supported: {list(AUDIO_CODEC_MAP.keys())}")

    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)
    if ffmpeg_bin is None:
        raise FileNotFoundError("FFmpeg not found. Please install ffmpeg or provide ffmpeg_path parameter")

    if output_path is None:
        video_name = Path(video_path).stem
        video_dir = Path(video_path).parent
        output_path = str(video_dir / f"{video_name}.{audio_format}")
    elif not output_path.endswith(f".{audio_format}"):
        output_path = f"{output_path}.{audio_format}"

    print(f"Extracting audio from: {video_path}")

    cmd = [
        ffmpeg_bin,
        "-i",
        video_path,
        "-vn",
        "-acodec",
        AUDIO_CODEC_MAP[audio_format],
        "-ab",
        "192k",
        "-y",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    print(f"Audio successfully saved to: {output_path}")
    return output_path


def batch_extract_audio(video_directory, output_directory=None, audio_format="mp3", ffmpeg_path=None):
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(Path(video_directory).glob(f"*{ext}"))
        video_files.extend(Path(video_directory).glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {video_directory}")
        return []

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    extracted_audio_files = []
    for video_file in video_files:
        try:
            output_path = None
            if output_directory:
                audio_name = Path(video_file).stem
                output_path = os.path.join(output_directory, f"{audio_name}.{audio_format}")

            audio_path = extract_audio_from_video(str(video_file), output_path, audio_format, ffmpeg_path)
            extracted_audio_files.append(audio_path)
        except Exception as exc:
            print(f"Failed to extract audio from {video_file}: {exc}")

    return extracted_audio_files


def main():
    parser = argparse.ArgumentParser(description="Extract audio from one video file.")
    parser.add_argument("video_file", help="Path to input video")
    parser.add_argument("output_file", nargs="?", default=None, help="Optional output audio path")
    parser.add_argument("format_type", nargs="?", default="mp3", help="Audio format: mp3|wav|aac|flac")
    args = parser.parse_args()

    extract_audio_from_video(args.video_file, args.output_file, args.format_type)


if __name__ == "__main__":
    main()
