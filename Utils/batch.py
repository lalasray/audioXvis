"""CLI helper to batch-extract audio tracks from test-dataset video files."""

import argparse
from pathlib import Path

from audio_extract import batch_extract_audio


def resolve_default_input_dir() -> str:
	candidates = [
		Path("data/test_dataset/videos"),
		Path("data/test_dataset/one_experienced_singer_dataset_songs"),
	]
	for candidate in candidates:
		if candidate.exists():
			return str(candidate)
	return str(candidates[0])


def resolve_default_output_dir() -> str:
	candidates = [
		Path("data/test_dataset/audio"),
		Path("data/test_dataset/one_experienced_singer_dataset_audio"),
	]
	for candidate in candidates:
		if candidate.exists():
			return str(candidate)
	return str(candidates[0])


def main() -> None:
	parser = argparse.ArgumentParser(description="Batch extract audio from test-dataset videos.")
	parser.add_argument(
		"--input_dir",
		default=resolve_default_input_dir(),
		help="Directory containing input videos.",
	)
	parser.add_argument(
		"--output_dir",
		default=resolve_default_output_dir(),
		help="Directory to save extracted audio.",
	)
	parser.add_argument("--audio_format", default="mp3", help="Output audio format: mp3|wav|aac|flac")
	parser.add_argument("--ffmpeg_path", default=None, help="Optional ffmpeg path.")
	args = parser.parse_args()

	outputs = batch_extract_audio(
		args.input_dir,
		args.output_dir,
		args.audio_format,
		args.ffmpeg_path,
	)
	print(f"Extracted {len(outputs)} audio files.")


if __name__ == "__main__":
	main()
