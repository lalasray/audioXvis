from __future__ import annotations

import importlib
import json
import mimetypes
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import argparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
UPLOAD_DIR = GENERATED_DIR / "uploads"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/audio2vis-mpl")
MESH_GLOB = str(REPO_ROOT / "main" / "_fbx_baked" / "fbx_frame_*.obj")
DEFAULT_MTL_OBJ = str(REPO_ROOT / "main" / "_fbx_baked" / "fbx_frame_0000.obj")
HOST = "127.0.0.1"
PORT = 8765
SR_MIC = 44100
PITCH_SR = 11025
WINDOW_SEC = 0.5
HOP_SEC = 0.25
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
MESH_TRIANGLE_STRIDE = max(1, int(os.environ.get("AUDIO2VIS_MESH_STRIDE", "1")))

DEPS_IMPORT_ERROR: Exception | None = None
SD_IMPORT_ERROR: Exception | None = None
try:
    np = importlib.import_module("numpy")
except Exception as exc:  # pragma: no cover - import-time environment guard
    DEPS_IMPORT_ERROR = exc
    np = None

try:
    sd = importlib.import_module("sounddevice")
except Exception as exc:  # pragma: no cover - import-time environment guard
    SD_IMPORT_ERROR = exc
    sd = None


def natural_sort_key(path_str: str) -> list[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path_str)]


def parse_mtl_kd(mtl_path: Path) -> dict[str, np.ndarray]:
    mats: dict[str, np.ndarray] = {}
    if not mtl_path.exists():
        return mats
    current = None
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1].strip()
                continue
            if current and line.startswith("Kd "):
                parts = line.split()
                if len(parts) >= 4:
                    kd = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    mats[current] = np.clip(kd * 255.0, 0.0, 255.0).astype(np.uint8)
    return mats


def parse_obj(obj_path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]], list[str], Path | None]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_materials: list[str] = []
    current_material = "default"
    mtllib = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                mtllib = (obj_path.parent / line.split(maxsplit=1)[1].strip()).resolve()
                continue
            if line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1].strip()
                continue
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
                continue
            if not line.startswith("f "):
                continue
            refs = [part.split("/")[0] for part in line.split()[1:]]
            idx = [int(ref) - 1 for ref in refs if ref]
            if len(idx) < 3:
                continue
            for i in range(1, len(idx) - 1):
                faces.append((idx[0], idx[i], idx[i + 1]))
                face_materials.append(current_material)
    return np.asarray(vertices, dtype=np.float32), faces, face_materials, mtllib


def build_flat_frame(vertices: np.ndarray, faces: list[tuple[int, int, int]]) -> np.ndarray:
    flat = np.empty((len(faces) * 3, 3), dtype=np.float32)
    cursor = 0
    for a, b, c in faces:
        flat[cursor] = vertices[a]
        flat[cursor + 1] = vertices[b]
        flat[cursor + 2] = vertices[c]
        cursor += 3
    return flat


def compute_normals(flat_positions: np.ndarray) -> np.ndarray:
    tri = flat_positions.reshape(-1, 3, 3)
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(edge_1, edge_2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    normals = normals / lengths
    return np.repeat(normals, 3, axis=0).astype(np.float32)


class MeshSequenceCache:
    def __init__(self, mesh_glob: str, mtl_source_obj: str):
        self.mesh_glob = mesh_glob
        self.mtl_source_obj = Path(mtl_source_obj)
        self.triangle_stride = MESH_TRIANGLE_STRIDE
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path = GENERATED_DIR / "larynx_sequence.meta.json"
        self.positions_path = GENERATED_DIR / "larynx_sequence.positions.bin"
        self.colors_path = GENERATED_DIR / "larynx_sequence.colors.bin"
        self.normals_path = GENERATED_DIR / "larynx_sequence.normals.bin"
        self.meta = self._ensure()

    def _ensure(self) -> dict[str, Any]:
        if self.meta_path.exists() and self.positions_path.exists() and self.colors_path.exists() and self.normals_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if int(meta.get("triangleStride", 1)) == self.triangle_stride:
                return meta

        obj_files = sorted((Path(p) for p in __import__("glob").glob(self.mesh_glob)), key=lambda p: natural_sort_key(str(p)))
        if len(obj_files) < 2:
            raise FileNotFoundError(f"Need at least 2 OBJ frames for sequence rendering: {self.mesh_glob}")

        base_vertices, faces, face_materials, _ = parse_obj(obj_files[0])
        if self.triangle_stride > 1:
            faces = faces[:: self.triangle_stride]
            face_materials = face_materials[:: self.triangle_stride]
            if not faces:
                raise RuntimeError(f"Triangle stride {self.triangle_stride} removed every mesh face.")
        flat_base = build_flat_frame(base_vertices, faces)
        normals = compute_normals(flat_base)

        _, _, _, mtllib = parse_obj(self.mtl_source_obj)
        kd_map = parse_mtl_kd(mtllib) if mtllib else {}
        default_rgb = np.array([214, 157, 130], dtype=np.uint8)
        colors = np.empty((len(faces) * 3, 3), dtype=np.uint8)
        for idx, material in enumerate(face_materials):
            rgb = kd_map.get(material, default_rgb)
            colors[idx * 3 : idx * 3 + 3] = rgb

        with self.positions_path.open("wb") as handle:
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
            maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
            for frame_path in obj_files:
                vertices, _, _, _ = parse_obj(frame_path)
                flat = build_flat_frame(vertices, faces)
                mins = np.minimum(mins, flat.min(axis=0))
                maxs = np.maximum(maxs, flat.max(axis=0))
                handle.write(flat.astype(np.float32).tobytes())

        self.colors_path.write_bytes(colors.tobytes())
        self.normals_path.write_bytes(normals.tobytes())

        center = ((mins + maxs) * 0.5).astype(np.float32)
        extent = (maxs - mins).astype(np.float32)
        scale = float(max(extent.max(), 1e-6))
        meta = {
            "frameCount": len(obj_files),
            "vertexCount": int(flat_base.shape[0]),
            "triangleStride": self.triangle_stride,
            "bounds": {
                "min": mins.tolist(),
                "max": maxs.tolist(),
                "center": center.tolist(),
                "extent": extent.tolist(),
                "scale": scale,
            },
            "positionsPath": "/mesh/positions.bin",
            "colorsPath": "/mesh/colors.bin",
            "normalsPath": "/mesh/normals.bin",
            "sequence": {
                "frameMin": 0.0,
                "frameMax": float(len(obj_files) - 1),
                "neutralFrame": float((len(obj_files) - 1) / 2.0),
                "blendMinDeg": 0.0,
                "blendMaxDeg": 90.0,
                "neutralDeg": 60.0,
            },
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta


@dataclass
class RuntimeState:
    mode: str = "idle"
    status: str = "Idle"
    source_label: str = "No input"
    frame_index: int = 25
    angles: tuple[float, float, float] = (60.0, 60.0, 60.0)
    waveform: list[float] | None = None
    pitch_hz: float = 0.0
    pitch_contour: list[float] | None = None
    progress: float = 0.0
    elapsed_sec: float = 0.0
    duration_sec: float = 0.0
    error: str | None = None
    updated_at: float = 0.0


@dataclass
class TrackSession:
    key: str
    state: RuntimeState
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    inf_queue: queue.Queue[np.ndarray] = field(default_factory=lambda: queue.Queue(maxsize=8))
    stop_event: threading.Event = field(default_factory=threading.Event)
    audio_stream: sd.InputStream | None = None
    stream_thread: threading.Thread | None = None
    audio_buffer: np.ndarray = field(default_factory=lambda: np.zeros(int(WINDOW_SEC * SR_MIC), dtype=np.float32))
    run_token: int = 0


class Audio2VisService:
    def __init__(self, mesh_cache: MeshSequenceCache):
        self.mesh_cache = mesh_cache
        self.window_samples = int(WINDOW_SEC * SR_MIC)
        self.hop_samples = int(HOP_SEC * SR_MIC)
        self.state_condition = threading.Condition()
        self.tracks = {
            "primary": TrackSession(
                key="primary",
                state=RuntimeState(
                    source_label="Primary source",
                    waveform=[0.0] * 180,
                    pitch_contour=[0.0] * 180,
                    updated_at=time.time(),
                ),
            ),
            "compare": TrackSession(
                key="compare",
                state=RuntimeState(
                    source_label="Comparison source",
                    waveform=[0.0] * 180,
                    pitch_contour=[0.0] * 180,
                    updated_at=time.time(),
                ),
            ),
        }
        self.sample_library = self._discover_samples()
        for track_key in self.tracks:
            threading.Thread(target=self._inference_loop, args=(track_key,), daemon=True).start()

    def _discover_samples(self) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        candidate_patterns = ["data/test.*"]
        data_root = REPO_ROOT / "data"
        if data_root.exists():
            for dataset_dir in sorted(data_root.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                for subdir in sorted(dataset_dir.iterdir()):
                    if subdir.is_dir() and "audio" in subdir.name.lower():
                        candidate_patterns.append(str(subdir.relative_to(REPO_ROOT) / "*"))
        seen: set[Path] = set()
        for pattern in candidate_patterns:
            for path in sorted(REPO_ROOT.glob(pattern))[:4]:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                samples.append({"label": path.name, "path": str(resolved)})
        return samples

    def _get_track(self, track_key: str) -> TrackSession:
        if track_key not in self.tracks:
            raise ValueError(f"Unknown track: {track_key}")
        return self.tracks[track_key]

    def get_audio_devices(self) -> list[dict[str, Any]]:
        if sd is None:
            return []
        devices = []
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                devices.append({"id": index, "name": info["name"], "channels": int(info["max_input_channels"])})
        return devices

    def get_config(self) -> dict[str, Any]:
        return {
            "mesh": self.mesh_cache.meta,
            "devices": self.get_audio_devices(),
            "samples": self.sample_library,
            "windowSec": WINDOW_SEC,
            "hopSec": HOP_SEC,
        }

    def _track_state_dict(self, track_key: str) -> dict[str, Any]:
        track = self._get_track(track_key)
        with track.state_lock:
            return {
                "mode": track.state.mode,
                "status": track.state.status,
                "sourceLabel": track.state.source_label,
                "frameIndex": track.state.frame_index,
                "angles": {"a": track.state.angles[0], "b": track.state.angles[1], "c": track.state.angles[2]},
                "waveform": track.state.waveform,
                "pitchHz": track.state.pitch_hz,
                "pitchContour": track.state.pitch_contour,
                "progress": track.state.progress,
                "elapsedSec": track.state.elapsed_sec,
                "durationSec": track.state.duration_sec,
                "error": track.state.error,
                "updatedAt": track.state.updated_at,
            }

    def _comparison_summary(self, delta: dict[str, float], abs_mean: float, active: bool) -> str:
        if not active:
            return "Load a second voice to compare motion and angle deltas."
        lead = max(delta.items(), key=lambda item: abs(item[1]))
        direction = "wider" if lead[1] > 0 else "narrower"
        return f"Biggest difference is angle {lead[0].upper()} at {abs(lead[1]):.1f} degrees; comparison track is {direction} there. Mean gap {abs_mean:.1f} degrees."

    def get_state(self) -> dict[str, Any]:
        primary = self._track_state_dict("primary")
        compare = self._track_state_dict("compare")
        delta = {
            "a": compare["angles"]["a"] - primary["angles"]["a"],
            "b": compare["angles"]["b"] - primary["angles"]["b"],
            "c": compare["angles"]["c"] - primary["angles"]["c"],
        }
        abs_mean = float(np.mean(np.abs(np.array(list(delta.values()), dtype=np.float32))))
        active = compare["mode"] != "idle"
        return {
            "tracks": {
                "primary": primary,
                "compare": compare,
            },
            "comparison": {
                "active": active,
                "frameDelta": compare["frameIndex"] - primary["frameIndex"],
                "angleDelta": delta,
                "meanAbsAngleDelta": abs_mean,
                "summary": self._comparison_summary(delta, abs_mean, active),
            },
        }

    def _set_state(self, track_key: str, **updates: Any) -> None:
        track = self._get_track(track_key)
        with track.state_lock:
            for key, value in updates.items():
                setattr(track.state, key, value)
            track.state.updated_at = time.time()
        with self.state_condition:
            self.state_condition.notify_all()

    def _downsample_waveform(self, samples: np.ndarray, points: int = 180) -> list[float]:
        if samples.size == 0:
            return [0.0] * points
        idx = np.linspace(0, samples.size - 1, points).astype(np.int32)
        view = np.clip(samples[idx], -1.0, 1.0)
        return view.astype(np.float32).tolist()

    def _append_pitch(self, track: TrackSession, pitch_hz: float, points: int = 180) -> list[float]:
        contour = list(track.state.pitch_contour or [0.0] * points)
        contour.append(float(pitch_hz))
        return contour[-points:]

    def _resample_audio(self, samples: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        if source_sr == target_sr or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        duration = samples.size / max(source_sr, 1)
        target_count = max(1, int(round(duration * target_sr)))
        source_x = np.linspace(0.0, duration, samples.size, endpoint=False)
        target_x = np.linspace(0.0, duration, target_count, endpoint=False)
        return np.interp(target_x, source_x, samples).astype(np.float32)

    def _decode_wav(self, audio_path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(audio_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sr = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
            values = np.where(values & 0x800000, values - 0x1000000, values)
            audio = values.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return np.clip(audio, -1.0, 1.0).astype(np.float32), sr

    def _map_angle_to_frame(self, angle_val: float) -> float:
        seq = self.mesh_cache.meta["sequence"]
        min_deg = seq["blendMinDeg"]
        max_deg = seq["blendMaxDeg"]
        neutral_deg = seq["neutralDeg"]
        min_frame = seq["frameMin"]
        max_frame = seq["frameMax"]
        neutral_frame = seq["neutralFrame"]
        if angle_val <= neutral_deg:
            t = (angle_val - min_deg) / max(neutral_deg - min_deg, 1e-6)
            t = float(np.clip(t, 0.0, 1.0))
            return min_frame + t * (neutral_frame - min_frame)
        t = (angle_val - neutral_deg) / max(max_deg - neutral_deg, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        return neutral_frame + t * (max_frame - neutral_frame)

    def _load_audio_with_fallback(self, audio_path: Path) -> tuple[np.ndarray, int]:
        try:
            return self._decode_wav(audio_path)
        except Exception as exc:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(SR_MIC),
                "-",
            ]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"Audio decode failed for {audio_path.name}: {stderr}") from exc
            audio_np = np.frombuffer(proc.stdout, dtype=np.float32)
            if audio_np.size == 0:
                raise RuntimeError(f"Audio decode produced no samples for {audio_path.name}") from exc
            return audio_np, SR_MIC

    def _make_audio_callback(self, track_key: str):
        def audio_callback(indata, frames, _time_info, status):
            track = self._get_track(track_key)
            if status:
                self._set_state(track_key, status=f"Mic warning: {status}")
            mono = indata[:, 0].astype(np.float32)
            track.audio_buffer = np.roll(track.audio_buffer, -frames)
            track.audio_buffer[-frames:] = mono
            self._set_state(track_key, waveform=self._downsample_waveform(track.audio_buffer))
            try:
                track.inf_queue.put(track.audio_buffer.copy(), block=False)
            except queue.Full:
                pass

        return audio_callback

    def start_microphone(self, track_key: str, device_id: int | None) -> None:
        if sd is None:
            raise RuntimeError("Microphone capture needs sounddevice. Upload audio or video, or install sounddevice for live input.")
        track = self._get_track(track_key)
        self.stop(track_key)
        track.run_token += 1
        track.stop_event = threading.Event()
        track.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=SR_MIC,
            blocksize=self.hop_samples,
            callback=self._make_audio_callback(track_key),
        )
        stream.start()
        track.audio_stream = stream
        device_name = next((d["name"] for d in self.get_audio_devices() if d["id"] == device_id), "Microphone")
        self._set_state(
            track_key,
            mode="mic",
            status="Listening to live microphone",
            source_label=device_name,
            progress=0.0,
            elapsed_sec=0.0,
            duration_sec=0.0,
            error=None,
        )

    def start_audio_file(self, track_key: str, audio_path: str) -> None:
        track = self._get_track(track_key)
        self.stop(track_key)
        track.run_token += 1
        track.stop_event = threading.Event()
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        waveform_np, source_sr = self._load_audio_with_fallback(path)
        duration_sec = float(waveform_np.shape[-1] / max(source_sr, 1))
        self._set_state(
            track_key,
            mode="file",
            status="Playing audio through pitch/loudness analysis",
            source_label=path.name,
            progress=0.0,
            elapsed_sec=0.0,
            duration_sec=duration_sec,
            error=None,
        )
        token = track.run_token
        track.stream_thread = threading.Thread(
            target=self._stream_audio_file,
            args=(track_key, waveform_np, source_sr, duration_sec, token),
            daemon=True,
        )
        track.stream_thread.start()

    def start_url_file(self, track_key: str, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported.")
        name = Path(parsed.path).name or f"{track_key}_remote_audio.bin"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = (UPLOAD_DIR / safe_name).resolve()
        with urllib.request.urlopen(url, timeout=20) as response:
            target.write_bytes(response.read())
        self.start_audio_file(track_key, str(target))
        return str(target)

    def _stream_audio_file(
        self,
        track_key: str,
        waveform_np: np.ndarray,
        source_sr: int,
        duration_sec: float,
        token: int,
    ) -> None:
        track = self._get_track(track_key)
        if source_sr != SR_MIC:
            audio_np = self._resample_audio(waveform_np, source_sr, SR_MIC)
        else:
            audio_np = waveform_np
        idx = 0
        started = time.monotonic()
        while idx + self.window_samples <= len(audio_np) and not track.stop_event.is_set() and token == track.run_token:
            chunk = audio_np[idx : idx + self.window_samples]
            self._set_state(
                track_key,
                waveform=self._downsample_waveform(chunk),
                elapsed_sec=min(idx / SR_MIC, duration_sec),
                progress=float(np.clip((idx / SR_MIC) / max(duration_sec, 1e-6), 0.0, 1.0)),
                status="Driving larynx sequence from pitch and loudness",
            )
            try:
                track.inf_queue.put(chunk.copy(), block=False)
            except queue.Full:
                pass
            idx += self.hop_samples
            target_elapsed = idx / SR_MIC
            wait = target_elapsed - (time.monotonic() - started)
            if wait > 0:
                time.sleep(wait)

        if not track.stop_event.is_set() and token == track.run_token:
            self._set_state(track_key, progress=1.0, elapsed_sec=duration_sec, status="Playback finished")

    def stop(self, track_key: str | None = None) -> None:
        if track_key is None:
            for key in self.tracks:
                self.stop(key)
            return
        track = self._get_track(track_key)
        track.stop_event.set()
        track.run_token += 1
        if track.audio_stream is not None:
            try:
                track.audio_stream.stop()
                track.audio_stream.close()
            except Exception:
                pass
            track.audio_stream = None
        track.stream_thread = None
        track.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        neutral = float(self.mesh_cache.meta["sequence"]["neutralFrame"])
        self._set_state(
            track_key,
            mode="idle",
            status="Idle",
            source_label="No input",
            progress=0.0,
            elapsed_sec=0.0,
            duration_sec=0.0,
            waveform=[0.0] * 180,
            pitch_hz=0.0,
            pitch_contour=[0.0] * 180,
            frame_index=int(round(neutral)),
            angles=(60.0, 60.0, 60.0),
            error=None,
        )

    def _estimate_pitch_hz(self, audio_window: np.ndarray, sr: int = SR_MIC) -> tuple[float, float]:
        centered = audio_window.astype(np.float32) - float(np.mean(audio_window))
        rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
        if rms < 0.006:
            return 0.0, 0.0

        if sr > PITCH_SR:
            step = max(1, int(round(sr / PITCH_SR)))
            centered = centered[::step]
            sr = int(round(sr / step))

        windowed = centered * np.hanning(centered.size).astype(np.float32)
        min_lag = max(1, int(sr / 500.0))
        max_lag = min(int(sr / 55.0), windowed.size - 1)
        if max_lag <= min_lag:
            return 0.0, 0.0

        corr = np.correlate(windowed, windowed, mode="full")[windowed.size - 1 :]
        corr0 = max(float(corr[0]), 1e-9)
        search = corr[min_lag:max_lag]
        if search.size == 0:
            return 0.0, 0.0

        best_value = float(np.max(search))
        rel_idx = int(np.argmax(search))
        if search.size >= 3:
            peaks = np.flatnonzero((search[1:-1] > search[:-2]) & (search[1:-1] >= search[2:])) + 1
            strong_peaks = peaks[search[peaks] >= best_value * 0.72]
            if strong_peaks.size:
                rel_idx = int(strong_peaks[0])
        lag = float(min_lag + rel_idx)
        if 1 <= min_lag + rel_idx < corr.size - 1:
            left = float(corr[min_lag + rel_idx - 1])
            mid = float(corr[min_lag + rel_idx])
            right = float(corr[min_lag + rel_idx + 1])
            denom = left - 2.0 * mid + right
            if abs(denom) > 1e-9:
                lag += 0.5 * (left - right) / denom

        confidence = float(np.clip(corr[int(round(lag))] / corr0, 0.0, 1.0))
        if confidence < 0.2:
            return 0.0, confidence
        return float(sr / max(lag, 1e-6)), confidence

    def _predict_angles(self, audio_window: np.ndarray) -> tuple[float, float, float, float]:
        rms = float(np.sqrt(np.mean(audio_window * audio_window) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-6)
        loudness = float(np.clip((db + 55.0) / 45.0, 0.0, 1.0))

        pitch_hz, confidence = self._estimate_pitch_hz(audio_window)
        if pitch_hz > 0.0:
            pitch = float(np.clip((np.log2(pitch_hz) - np.log2(80.0)) / (np.log2(420.0) - np.log2(80.0)), 0.0, 1.0))
        else:
            pitch = 0.5

        voiced = float(np.clip(confidence * 1.7, 0.0, 1.0))
        a = 38.0 + 47.0 * loudness + 7.0 * (pitch - 0.5)
        c = 38.0 + 35.0 * pitch * voiced + 12.0 * loudness
        if voiced < 0.25:
            c = 50.0 + 10.0 * loudness

        a = float(np.clip(a, 28.0, 88.0))
        c = float(np.clip(c, 28.0, 88.0))
        if a + c > 155.0:
            scale = 155.0 / (a + c)
            a *= scale
            c *= scale
        b = float(180.0 - a - c)
        return a, b, c, float(pitch_hz)

    def _inference_loop(self, track_key: str) -> None:
        track = self._get_track(track_key)
        while True:
            try:
                window = track.inf_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                a, b, c, pitch_hz = self._predict_angles(window)
                driver = float((a + b) * 0.5)
                frame_float = self._map_angle_to_frame(driver)
                frame_index = int(round(np.clip(frame_float, 0.0, self.mesh_cache.meta["frameCount"] - 1)))
                self._set_state(
                    track_key,
                    frame_index=frame_index,
                    angles=(a, b, c),
                    pitch_hz=pitch_hz,
                    pitch_contour=self._append_pitch(track, pitch_hz),
                    error=None,
                )
            except Exception as exc:
                self._set_state(track_key, error=str(exc), status="Inference error")
            finally:
                track.inf_queue.task_done()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "Audio2VisHTTP/0.2"

    @property
    def service(self) -> Audio2VisService:
        return self.server.service  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._serve_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            return self._serve_file((STATIC_DIR / rel).resolve())
        if path == "/api/config":
            return self._send_json(self.service.get_config())
        if path == "/api/state":
            return self._send_json(self.service.get_state())
        if path == "/api/events":
            return self._send_events()
        if path == "/mesh/meta":
            return self._send_json(self.service.mesh_cache.meta)
        if path == "/mesh/positions.bin":
            return self._serve_file(self.service.mesh_cache.positions_path)
        if path == "/mesh/colors.bin":
            return self._serve_file(self.service.mesh_cache.colors_path)
        if path == "/mesh/normals.bin":
            return self._serve_file(self.service.mesh_cache.normals_path)
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/start-mic":
                payload = self._read_json()
                self.service.start_microphone(payload.get("track", "primary"), payload.get("device"))
                return self._send_json({"ok": True, "state": self.service.get_state()})
            if parsed.path == "/api/start-file":
                payload = self._read_json()
                self.service.start_audio_file(payload.get("track", "primary"), payload["path"])
                return self._send_json({"ok": True, "state": self.service.get_state()})
            if parsed.path == "/api/start-url":
                payload = self._read_json()
                path = self.service.start_url_file(payload.get("track", "compare"), payload["url"])
                return self._send_json({"ok": True, "path": path, "state": self.service.get_state()})
            if parsed.path == "/api/stop":
                payload = self._read_json()
                self.service.stop(payload.get("track"))
                return self._send_json({"ok": True, "state": self.service.get_state()})
            if parsed.path == "/api/upload-audio":
                path = self._save_upload()
                return self._send_json({"ok": True, "path": str(path)})
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _save_upload(self) -> Path:
        length = int(self.headers.get("Content-Length", "0"))
        filename = self.headers.get("X-Filename", "upload.wav")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = (UPLOAD_DIR / safe_name).resolve()
        path.write_bytes(self.rfile.read(length))
        return path

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_updated = -1.0
        while True:
            state = self.service.get_state()
            updated = max(track["updatedAt"] for track in state["tracks"].values())
            if updated != last_updated:
                last_updated = updated
                raw = f"data: {json.dumps(state)}\n\n".encode("utf-8")
                try:
                    self.wfile.write(raw)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            with self.service.state_condition:
                notified = self.service.state_condition.wait(timeout=15.0)
            if not notified:
                try:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

    def _serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        allowed_roots = [STATIC_DIR.resolve(), GENERATED_DIR.resolve()]
        if not resolved.exists() or not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        raw = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Audio2Vis local web app.")
    parser.add_argument("--host", default=HOST, help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind.")
    args = parser.parse_args()

    if np is None:
        raise RuntimeError(
            "Missing runtime dependencies for the web app. "
            "Install requirements.txt before launching the server."
        ) from DEPS_IMPORT_ERROR
    if SD_IMPORT_ERROR is not None:
        print("Microphone capture disabled: sounddevice is not installed.")

    print("Preparing mesh cache for the web viewer...")
    mesh_cache = MeshSequenceCache(mesh_glob=MESH_GLOB, mtl_source_obj=DEFAULT_MTL_OBJ)
    service = Audio2VisService(mesh_cache)
    httpd = ThreadingHTTPServer((args.host, args.port), AppHandler)
    httpd.service = service  # type: ignore[attr-defined]
    print(f"Audio2Vis web app running at http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
