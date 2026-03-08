"""
Realtime sequence-mesh driver with hybrid control:
  - 50% ML predicted angle signal
  - 50% audio loudness signal

Silence maps to frame 1, loud speech/singing pushes toward frame 50.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__)))
from infer_fullclip import extract_features, load_model, normalize_features

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyvista is required. Install with: pip install pyvista") from exc

try:
    import trimesh
except Exception:
    trimesh = None


SR_MIC = 44100
SR_MODEL = 22050
WINDOW_SEC = 0.5
HOP_SEC = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class StreamState:
    audio_buffer: np.ndarray
    inf_queue: queue.Queue
    pose_queue: queue.Queue
    stop_event: threading.Event


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime sequence mesh with ML+loudness hybrid control")
    p.add_argument("--ckpt", type=str, default="main/checkpoints/diffusion_v2/best.pt", help="Model checkpoint")
    p.add_argument("--mesh_seq_glob", type=str, required=True, help="Glob pattern for frame sequence, e.g. data/model/.../o*.obj")
    p.add_argument("--mtl_source_obj", type=str, default=None, help="OBJ path to source MTL colors from")
    p.add_argument("--audio", type=str, default=None, help="Optional audio file path instead of microphone")
    p.add_argument("--sample_steps", type=int, default=None, help="Override diffusion sample steps")
    p.add_argument("--user_id", type=int, default=0, help="User id embedding index for inference")
    p.add_argument("--smooth_alpha", type=float, default=0.25, help="EMA smoothing for angles")

    p.add_argument(
        "--blend_angle_source",
        type=str,
        default="mean_ab",
        choices=["a", "b", "c", "mean_ab", "mean_ac", "mean_abc"],
        help="ML angle signal used for frame mapping",
    )
    p.add_argument("--blend_min_deg", type=float, default=0.0, help="ML angle min mapped to seq_min_frame")
    p.add_argument("--blend_max_deg", type=float, default=90.0, help="ML angle max mapped to seq_max_frame")
    p.add_argument("--seq_min_frame", type=float, default=1.0, help="Lowest frame in sequence mapping")
    p.add_argument("--seq_max_frame", type=float, default=50.0, help="Highest frame in sequence mapping")
    p.add_argument("--seq_neutral_frame", type=float, default=25.0, help="Neutral frame for ML mapping")
    p.add_argument("--seq_neutral_deg", type=float, default=60.0, help="Neutral ML angle value")

    p.add_argument("--hybrid_loudness_weight", type=float, default=0.5, help="Weight of loudness in [0,1], default 0.5")
    p.add_argument("--loud_floor_db", type=float, default=-55.0, help="Loudness floor in dBFS -> frame min")
    p.add_argument("--loud_ceil_db", type=float, default=-15.0, help="Loudness ceiling in dBFS -> frame max")
    return p.parse_args()


def natural_sort_key(path_str: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path_str)]


def load_supported_mesh(mesh_path: str) -> pv.DataSet:
    try:
        mesh = pv.read(mesh_path)
        if getattr(mesh, "n_points", 0) == 0 or getattr(mesh, "n_cells", 0) == 0:
            raise ValueError("PyVista reader returned an empty mesh.")
        return mesh
    except Exception:
        if trimesh is None:
            raise
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                raise ValueError(f"No mesh geometry found in: {mesh_path}")
            loaded = trimesh.util.concatenate(geoms)
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int64)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).reshape(-1)
        return pv.PolyData(vertices, faces_pv)


def prepare_mesh(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    return mesh.triangulate()


def parse_mtl_kd(mtl_path: Path) -> dict[str, np.ndarray]:
    mats: dict[str, np.ndarray] = {}
    if not mtl_path.exists():
        return mats
    current = None
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1].strip()
            elif current and line.startswith("Kd "):
                p = line.split()
                if len(p) >= 4:
                    kd = np.array([float(p[1]), float(p[2]), float(p[3])], dtype=np.float32)
                    mats[current] = np.clip(kd * 255.0, 0.0, 255.0).astype(np.uint8)
    return mats


def parse_obj_face_materials(obj_path: Path) -> tuple[list[str], Path | None]:
    face_mats: list[str] = []
    current = "NO_MATERIAL"
    mtllib = None
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                mtllib = (obj_path.parent / line.split(maxsplit=1)[1].strip()).resolve()
            elif line.startswith("usemtl "):
                current = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                face_mats.append(current)
    return face_mats, mtllib


def build_mesh_cell_colors_from_obj_mtl(mesh: pv.PolyData, obj_path: str) -> np.ndarray | None:
    obj_p = Path(obj_path).resolve()
    if not obj_p.exists() or mesh.n_cells == 0:
        return None
    face_mats, mtllib = parse_obj_face_materials(obj_p)
    if not face_mats or mtllib is None:
        return None
    kd_map = parse_mtl_kd(mtllib)
    if not kd_map:
        return None
    if len(face_mats) != mesh.n_cells:
        return np.tile(kd_map.get(face_mats[0], np.array([220, 150, 120], dtype=np.uint8)), (mesh.n_cells, 1))
    out = np.zeros((mesh.n_cells, 3), dtype=np.uint8)
    default = np.array([220, 150, 120], dtype=np.uint8)
    for i, m in enumerate(face_mats):
        out[i] = kd_map.get(m, default)
    return out


def blend_driver_value(a: float, b: float, c: float, source: str) -> float:
    if source == "a":
        return a
    if source == "b":
        return b
    if source == "c":
        return c
    if source == "mean_ab":
        return (a + b) / 2.0
    if source == "mean_ac":
        return (a + c) / 2.0
    return (a + b + c) / 3.0


def map_angle_to_frame(angle_val: float, min_deg: float, neutral_deg: float, max_deg: float,
                       min_frame: float, neutral_frame: float, max_frame: float) -> float:
    if angle_val <= neutral_deg:
        t = (angle_val - min_deg) / max(neutral_deg - min_deg, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        return min_frame + t * (neutral_frame - min_frame)
    t = (angle_val - neutral_deg) / max(max_deg - neutral_deg, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    return neutral_frame + t * (max_frame - neutral_frame)


def loudness_norm_db(window: np.ndarray, floor_db: float, ceil_db: float) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(window)) + 1e-12))
    db = 20.0 * np.log10(max(rms, 1e-12))
    v = (db - floor_db) / max(ceil_db - floor_db, 1e-6)
    return float(np.clip(v, 0.0, 1.0)), db


def make_audio_callback(state: StreamState):
    def audio_callback(indata, frames, _time_info, status):
        if status:
            print(status)
        state.audio_buffer = np.roll(state.audio_buffer, -frames)
        state.audio_buffer[-frames:] = indata[:, 0]
        try:
            state.inf_queue.put(state.audio_buffer.copy(), block=False)
        except queue.Full:
            pass
    return audio_callback


def inference_worker(state: StreamState, model, y_mean, y_std, sample_steps: int, feature_stats,
                     user_id: int, loud_floor_db: float, loud_ceil_db: float):
    while not state.stop_event.is_set():
        try:
            y_window = state.inf_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            loud_norm, loud_db = loudness_norm_db(y_window, loud_floor_db, loud_ceil_db)
            y_window_resampled = torchaudio.functional.resample(torch.from_numpy(y_window), SR_MIC, SR_MODEL).numpy()
            feats = extract_features(y_window_resampled, sr=SR_MODEL)
            if feature_stats is not None:
                feats = normalize_features(feats, feature_stats)

            x_dict = {k: torch.from_numpy(v).unsqueeze(0).to(DEVICE) for k, v in feats.items()}
            uid = torch.full((1,), user_id, dtype=torch.long, device=DEVICE)
            pred_norm = model.sample_ddim(x_dict, sample_steps=sample_steps, user_id=uid)
            pred = pred_norm * y_std + y_mean
            a = float(pred[0, 0].item())
            c = float(pred[0, 1].item())
            b = 180.0 - a - c

            print(f"Pred: a={a:.2f}, b={b:.2f}, c={c:.2f} | loud={loud_db:.1f} dB ({loud_norm:.2f})")
            try:
                state.pose_queue.put((a, b, c, loud_norm), block=False)
            except queue.Full:
                pass
        except Exception as exc:
            print(f"Inference error: {exc}")
        finally:
            state.inf_queue.task_done()


def stream_audio_file(state: StreamState, audio_path: str, window_samples: int, hop_samples: int):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SR_MIC:
            waveform = torchaudio.functional.resample(waveform, sr, SR_MIC)
        audio_np = waveform[0].numpy()
    except Exception as exc:
        print(f"torchaudio decode failed ({exc}); falling back to ffmpeg decode.")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            audio_path,
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
            err = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg decode failed: {err}") from exc
        audio_np = np.frombuffer(proc.stdout, dtype=np.float32)
        if audio_np.size == 0:
            raise RuntimeError("ffmpeg decode produced no audio samples") from exc

    idx = 0
    while idx + window_samples <= len(audio_np) and not state.stop_event.is_set():
        try:
            state.inf_queue.put(audio_np[idx:idx + window_samples].copy(), block=False)
        except queue.Full:
            pass
        idx += hop_samples
        time.sleep(HOP_SEC)


def main():
    args = parse_args()
    window_samples = int(WINDOW_SEC * SR_MIC)
    hop_samples = int(HOP_SEC * SR_MIC)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    seq_files = sorted(glob.glob(args.mesh_seq_glob), key=natural_sort_key)
    if len(seq_files) < 2:
        raise FileNotFoundError(f"Need >=2 sequence files, got: {args.mesh_seq_glob}")
    if args.audio and not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model, y_mean, y_std, default_steps, _cfg, feature_stats = load_model(args.ckpt, DEVICE)
    sample_steps = args.sample_steps if args.sample_steps is not None else default_steps
    print(f"Model loaded. sample_steps={sample_steps}")

    base_mesh = prepare_mesh(load_supported_mesh(seq_files[0]))
    base_pts = np.array(base_mesh.points, dtype=np.float32)
    seq_pts = [base_pts]
    for f in seq_files[1:]:
        m = prepare_mesh(load_supported_mesh(f))
        pts = np.array(m.points, dtype=np.float32)
        if pts.shape[0] != base_pts.shape[0]:
            raise ValueError(
                f"All sequence meshes must match vertex count/order. Base={base_pts.shape[0]}, {f}={pts.shape[0]}"
            )
        seq_pts.append(pts)
    seq_pts = np.stack(seq_pts, axis=0)
    seq_axis = np.linspace(args.seq_min_frame, args.seq_max_frame, num=seq_pts.shape[0], dtype=np.float32)
    print(f"Loaded sequence: {len(seq_files)} files")

    mtl_obj = args.mtl_source_obj or seq_files[0]
    rgb = build_mesh_cell_colors_from_obj_mtl(base_mesh, mtl_obj)
    if rgb is not None and rgb.shape[0] == base_mesh.n_cells:
        base_mesh.cell_data["mtl_rgb"] = rgb
        print(f"Applied MTL colors from: {mtl_obj}")
    else:
        print(f"WARNING: Could not apply MTL colors from: {mtl_obj}; using fallback color.")

    state = StreamState(
        audio_buffer=np.zeros(window_samples, dtype=np.float32),
        inf_queue=queue.Queue(maxsize=8),
        pose_queue=queue.Queue(maxsize=8),
        stop_event=threading.Event(),
    )

    threading.Thread(
        target=inference_worker,
        args=(
            state, model, y_mean, y_std, sample_steps, feature_stats, args.user_id,
            args.loud_floor_db, args.loud_ceil_db,
        ),
        daemon=True,
    ).start()

    if args.audio:
        print(f"Streaming from audio file: {args.audio}")
        threading.Thread(
            target=stream_audio_file,
            args=(state, args.audio, window_samples, hop_samples),
            daemon=True,
        ).start()
        stream = None
    else:
        print("Using microphone input. Press Ctrl+C or close viewer to stop.")
        stream = sd.InputStream(channels=1, samplerate=SR_MIC, callback=make_audio_callback(state), blocksize=hop_samples)
        stream.start()

    plotter = pv.Plotter(window_size=(1200, 900))
    if "mtl_rgb" in base_mesh.cell_data:
        plotter.add_mesh(base_mesh, scalars="mtl_rgb", rgb=True, smooth_shading=True)
    else:
        plotter.add_mesh(base_mesh, color="lightcoral", smooth_shading=True)
    plotter.add_axes()
    plotter.set_background("black")
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.2)
    plotter.show(auto_close=False, interactive_update=True)

    alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
    loud_w = float(np.clip(args.hybrid_loudness_weight, 0.0, 1.0))
    smoothed = np.array([args.seq_neutral_deg, args.seq_neutral_deg, args.seq_neutral_deg], dtype=np.float32)

    try:
        while not state.stop_event.is_set():
            try:
                plotter.update()
            except Exception:
                break

            latest = None
            while True:
                try:
                    latest = state.pose_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                a, b, c, loud_norm = latest
                target = np.array([a, b, c], dtype=np.float32)
                smoothed = alpha * target + (1.0 - alpha) * smoothed

                ml_drive = blend_driver_value(float(smoothed[0]), float(smoothed[1]), float(smoothed[2]), args.blend_angle_source)
                ml_frame = map_angle_to_frame(
                    angle_val=ml_drive,
                    min_deg=args.blend_min_deg,
                    neutral_deg=args.seq_neutral_deg,
                    max_deg=args.blend_max_deg,
                    min_frame=args.seq_min_frame,
                    neutral_frame=args.seq_neutral_frame,
                    max_frame=args.seq_max_frame,
                )
                loud_frame = args.seq_min_frame + loud_norm * (args.seq_max_frame - args.seq_min_frame)
                frame_f = (1.0 - loud_w) * ml_frame + loud_w * loud_frame
                frame_f = float(np.clip(frame_f, args.seq_min_frame, args.seq_max_frame))

                hi = int(np.searchsorted(seq_axis, frame_f, side="left"))
                if hi <= 0:
                    base_mesh.points = seq_pts[0]
                elif hi >= len(seq_axis):
                    base_mesh.points = seq_pts[-1]
                else:
                    lo = hi - 1
                    f_lo = float(seq_axis[lo])
                    f_hi = float(seq_axis[hi])
                    w = 0.0 if (f_hi - f_lo) <= 1e-6 else (frame_f - f_lo) / (f_hi - f_lo)
                    base_mesh.points = (1.0 - w) * seq_pts[lo] + w * seq_pts[hi]
                plotter.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        if stream is not None:
            stream.stop()
            stream.close()
        plotter.close()


if __name__ == "__main__":
    main()
