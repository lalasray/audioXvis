"""Shared media/video utility helpers used by dataset preparation scripts."""

import os
import shutil
import subprocess
from pathlib import Path

import cv2


AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
VIDEO_EXTS = [".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv"]


def resolve_ffmpeg_path(ffmpeg_path: str | None = None) -> str | None:
    if ffmpeg_path:
        return ffmpeg_path

    path_env = os.environ.get("PATH", "")
    possible_paths = [
        "ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for path_entry in path_env.split(os.pathsep):
        if path_entry:
            possible_paths.append(str(Path(path_entry) / "ffmpeg"))

    for path in possible_paths:
        if path == "ffmpeg" and shutil.which("ffmpeg"):
            return "ffmpeg"
        if os.path.exists(path):
            return path
    return None


def resolve_ffprobe_path(ffmpeg_path: str) -> str:
    if ffmpeg_path == "ffmpeg":
        return "ffprobe"
    return str(Path(ffmpeg_path).with_name("ffprobe"))


def get_video_capture_props(video_path: str) -> tuple[float, int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise RuntimeError("Could not read FPS from video.")
    if frame_count <= 0:
        raise RuntimeError("Could not read frame count from video.")

    duration_sec = frame_count / fps
    return fps, width, height, frame_count, duration_sec


def get_video_duration_ffprobe(video_path: str | Path, ffprobe_path: str = "ffprobe") -> float:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Unable to read duration for {video_path}")
    return float(result.stdout.strip())
