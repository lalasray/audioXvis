"""Backward-compatible wrapper for the renamed audio_extract module."""

from audio_extract import batch_extract_audio, extract_audio_from_video, main


if __name__ == "__main__":
    main()
