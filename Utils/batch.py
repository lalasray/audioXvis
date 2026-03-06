"""CLI helper to batch-extract audio tracks from test-dataset video files."""

import argparse

from audio_extract import batch_extract_audio


def main() -> None:
	parser = argparse.ArgumentParser(description="Batch extract audio from test-dataset videos.")
	parser.add_argument(
		"--input_dir",
		default="data/test_dataset/one_experienced_singer_dataset_songs",
		help="Directory containing input videos.",
	)
	parser.add_argument(
		"--output_dir",
		default="data/test_dataset/one_experienced_singer_dataset_audio",
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