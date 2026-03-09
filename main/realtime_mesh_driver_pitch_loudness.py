"""
Realtime sequence-mesh driver using audio pitch+loudness only.

Control rule:
  - silence -> pitch control = 0
  - high pitch -> -1
  - low pitch -> +1
  - loudness acts as multiplier of pitch control
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyvista is required. Install with: pip install pyvista") from exc

try:
    import trimesh
except Exception:
    trimesh = None


SR_MIC = 44100
WINDOW_SEC = 0.5
HOP_SEC = 0.25


@dataclass
class StreamState:
    audio_buffer: np.ndarray
    audio_queue: queue.Queue
    drive_queue: queue.Queue
    stop_event: threading.Event


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime sequence mesh with pitch*loudness control")
    p.add_argument("--mesh_seq_glob", type=str, default=None, help="Glob pattern for frame sequence, e.g. data/model/.../*.obj")
    p.add_argument("--fbx", type=str, default=None, help="Optional FBX animation file to bake into OBJ sequence with Blender")
    p.add_argument("--fbx_object_name", type=str, default=None, help="Optional mesh object name inside FBX to export")
    p.add_argument("--fbx_start_frame", type=int, default=0, help="FBX start frame for baking")
    p.add_argument("--fbx_end_frame", type=int, default=50, help="FBX end frame for baking (inclusive)")
    p.add_argument("--fbx_export_dir", type=str, default="main/_fbx_baked", help="Directory to write baked OBJ sequence")
    p.add_argument("--blender_bin", type=str, default="blender", help="Blender executable path/name used for FBX baking")
    p.add_argument("--mtl_source_obj", type=str, default=None, help="OBJ path to source MTL colors from")
    p.add_argument("--audio", type=str, default=None, help="Optional audio file path instead of microphone")

    p.add_argument("--seq_min_frame", type=float, default=1.0, help="Lowest frame in sequence mapping")
    p.add_argument("--seq_max_frame", type=float, default=50.0, help="Highest frame in sequence mapping")
    p.add_argument("--seq_neutral_frame", type=float, default=25.0, help="Neutral frame (silence/no pitch)")
    p.add_argument("--frame_transition_sec", type=float, default=0.01, help="Seconds to transition to each new target frame")

    p.add_argument("--pitch_min_hz", type=float, default=80.0, help="Pitch at +1 (low)")
    p.add_argument("--pitch_max_hz", type=float, default=350.0, help="Pitch at -1 (high)")
    p.add_argument("--silence_db", type=float, default=-55.0, help="At or below this dBFS, pitch control is forced to 0")
    p.add_argument("--loudness_floor_db", type=float, default=-55.0, help="Loudness multiplier 0.0")
    p.add_argument("--loudness_ceil_db", type=float, default=-15.0, help="Loudness multiplier 1.0")
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


def bake_fbx_to_obj_sequence(
    fbx_path: str,
    out_dir: str,
    start_frame: int,
    end_frame: int,
    blender_bin: str,
    object_name: str | None,
) -> list[str]:
    if start_frame > end_frame:
        raise ValueError(f"Invalid FBX frame range: start={start_frame}, end={end_frame}")
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")
    blender_resolved = shutil.which(blender_bin) if os.path.sep not in blender_bin else blender_bin
    if blender_resolved is None or not os.path.exists(blender_resolved):
        raise FileNotFoundError(
            f"Blender executable not found: {blender_bin}. Install Blender or pass --blender_bin /full/path/to/blender"
        )

    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    script_path = out_path / "_bake_fbx_to_obj.py"
    script_path.write_text(
        "import bpy\n"
        "import os\n"
        "import sys\n"
        "\n"
        "argv = sys.argv[sys.argv.index('--') + 1:]\n"
        "fbx_path = argv[0]\n"
        "out_dir = argv[1]\n"
        "start_frame = int(argv[2])\n"
        "end_frame = int(argv[3])\n"
        "object_name = argv[4] if len(argv) > 4 and argv[4] else None\n"
        "\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "bpy.ops.import_scene.fbx(filepath=fbx_path)\n"
        "\n"
        "scene = bpy.context.scene\n"
        "mesh_objs = [o for o in scene.objects if o.type == 'MESH']\n"
        "if object_name:\n"
        "    selected_objs = [o for o in mesh_objs if o.name == object_name]\n"
        "    if not selected_objs:\n"
        "        raise RuntimeError(f'FBX object not found: {object_name}')\n"
        "else:\n"
        "    selected_objs = mesh_objs\n"
        "if not selected_objs:\n"
        "    raise RuntimeError('No mesh objects found to export from FBX')\n"
        "\n"
        "for frame in range(start_frame, end_frame + 1):\n"
        "    scene.frame_set(frame)\n"
        "    bpy.ops.object.select_all(action='DESELECT')\n"
        "    for o in selected_objs:\n"
        "        o.select_set(True)\n"
        "    bpy.context.view_layer.objects.active = selected_objs[0]\n"
        "    out_file = os.path.join(out_dir, f'fbx_frame_{frame:04d}.obj')\n"
        "    try:\n"
        "        bpy.ops.wm.obj_export(filepath=out_file, export_selected_objects=True, export_materials=True)\n"
        "    except Exception:\n"
        "        bpy.ops.export_scene.obj(filepath=out_file, use_selection=True, use_materials=True)\n",
        encoding="utf-8",
    )

    cmd = [
        blender_resolved,
        "-b",
        "-noaudio",
        "--python",
        str(script_path),
        "--",
        str(Path(fbx_path).resolve()),
        str(out_path),
        str(start_frame),
        str(end_frame),
        object_name or "",
    ]
    print(f"Baking FBX -> OBJ with Blender: frames {start_frame}..{end_frame}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        stdout_tail = (proc.stdout or "").strip()[-2000:]
        raise RuntimeError(
            "Blender FBX bake failed.\n"
            f"STDERR tail:\n{stderr_tail}\n"
            f"STDOUT tail:\n{stdout_tail}"
        )

    seq_files = sorted(glob.glob(str(out_path / "fbx_frame_*.obj")), key=natural_sort_key)
    if len(seq_files) < 2:
        raise RuntimeError(f"FBX bake produced insufficient OBJ frames in: {out_path}")
    return seq_files


def loudness_norm_db(window: np.ndarray, floor_db: float, ceil_db: float) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(window)) + 1e-12))
    db = 20.0 * np.log10(max(rms, 1e-12))
    v = (db - floor_db) / max(ceil_db - floor_db, 1e-6)
    return float(np.clip(v, 0.0, 1.0)), db


def estimate_pitch_hz(window: np.ndarray, sr: int) -> float:
    wav = torch.from_numpy(window).float().unsqueeze(0)
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(wav, sample_rate=sr)
        f0_np = f0.squeeze(0).cpu().numpy().astype(np.float32)
        valid = f0_np[np.isfinite(f0_np) & (f0_np > 1e-3)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))
    except Exception:
        return 0.0


def pitch_signed_control(
    pitch_hz: float,
    loud_db: float,
    silence_db: float,
    pitch_min_hz: float,
    pitch_max_hz: float,
) -> float:
    if loud_db <= silence_db or pitch_hz <= 0.0:
        return 0.0
    t = (pitch_hz - pitch_min_hz) / max(pitch_max_hz - pitch_min_hz, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - 2.0 * t


def drive_to_frame(
    drive: float,
    min_frame: float,
    neutral_frame: float,
    max_frame: float,
) -> float:
    d = float(np.clip(drive, -1.0, 1.0))
    if d >= 0.0:
        return neutral_frame + d * (max_frame - neutral_frame)
    return neutral_frame + d * (neutral_frame - min_frame)


def make_audio_callback(state: StreamState):
    def audio_callback(indata, frames, _time_info, status):
        if status:
            print(status)
        state.audio_buffer = np.roll(state.audio_buffer, -frames)
        state.audio_buffer[-frames:] = indata[:, 0]
        try:
            state.audio_queue.put(state.audio_buffer.copy(), block=False)
        except queue.Full:
            pass
    return audio_callback


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
            state.audio_queue.put(audio_np[idx:idx + window_samples].copy(), block=False)
        except queue.Full:
            pass
        idx += hop_samples
        time.sleep(HOP_SEC)


def audio_drive_worker(
    state: StreamState,
    pitch_min_hz: float,
    pitch_max_hz: float,
    silence_db: float,
    loudness_floor_db: float,
    loudness_ceil_db: float,
):
    while not state.stop_event.is_set():
        try:
            y_window = state.audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            loud_norm, loud_db = loudness_norm_db(y_window, loudness_floor_db, loudness_ceil_db)
            pitch_hz = estimate_pitch_hz(y_window, SR_MIC)
            pitch_signed = pitch_signed_control(
                pitch_hz=pitch_hz,
                loud_db=loud_db,
                silence_db=silence_db,
                pitch_min_hz=pitch_min_hz,
                pitch_max_hz=pitch_max_hz,
            )
            drive = float(np.clip(pitch_signed * loud_norm, -1.0, 1.0))
            print(
                f"Audio: pitch={pitch_hz:7.2f}Hz, pitch_signed={pitch_signed:+.2f}, "
                f"loud={loud_db:+6.1f}dB ({loud_norm:.2f}), drive={drive:+.2f}"
            )
            try:
                state.drive_queue.put((drive,), block=False)
            except queue.Full:
                pass
        except Exception as exc:
            print(f"Audio drive error: {exc}")
        finally:
            state.audio_queue.task_done()


def main():
    args = parse_args()
    window_samples = int(WINDOW_SEC * SR_MIC)
    hop_samples = int(HOP_SEC * SR_MIC)

    if not args.mesh_seq_glob and not args.fbx:
        raise ValueError("Provide either --mesh_seq_glob or --fbx")
    if args.fbx:
        seq_files = bake_fbx_to_obj_sequence(
            fbx_path=args.fbx,
            out_dir=args.fbx_export_dir,
            start_frame=args.fbx_start_frame,
            end_frame=args.fbx_end_frame,
            blender_bin=args.blender_bin,
            object_name=args.fbx_object_name,
        )
    else:
        seq_files = sorted(glob.glob(args.mesh_seq_glob), key=natural_sort_key)
    if len(seq_files) < 2:
        src = args.fbx or args.mesh_seq_glob
        raise FileNotFoundError(f"Need >=2 sequence files, got: {src}")
    if args.audio and not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    seq_meshes: list[pv.PolyData] = [prepare_mesh(load_supported_mesh(f)) for f in seq_files]
    for mesh in seq_meshes:
        mesh.rotate_x(90.0, inplace=True)
    seq_point_counts = [int(m.n_points) for m in seq_meshes]
    topology_consistent = len(set(seq_point_counts)) == 1
    seq_axis = np.linspace(args.seq_min_frame, args.seq_max_frame, num=len(seq_meshes), dtype=np.float32)
    print(f"Loaded sequence: {len(seq_files)} files")

    if topology_consistent:
        seq_pts = np.stack([np.array(m.points, dtype=np.float32) for m in seq_meshes], axis=0)
        print(f"Topology mode: morph interpolation (n_points={seq_point_counts[0]})")
    else:
        seq_pts = None
        uniq_counts = ", ".join(str(v) for v in sorted(set(seq_point_counts)))
        print(f"Topology mode: discrete mesh swap (point counts found: {uniq_counts})")

    mtl_sources = [args.mtl_source_obj] * len(seq_files) if args.mtl_source_obj else seq_files
    default_rgb = np.array([220, 150, 120], dtype=np.uint8)
    colored_count = 0
    for mesh, obj_path in zip(seq_meshes, mtl_sources):
        rgb = build_mesh_cell_colors_from_obj_mtl(mesh, obj_path)
        if rgb is None or rgb.shape[0] != mesh.n_cells:
            rgb = np.tile(default_rgb, (mesh.n_cells, 1))
        else:
            colored_count += 1
        mesh.cell_data["mtl_rgb"] = rgb
    if colored_count > 0:
        if args.mtl_source_obj:
            print(f"Applied MTL colors from override source on {colored_count}/{len(seq_meshes)} frames: {args.mtl_source_obj}")
        else:
            print(f"Applied per-frame MTL colors on {colored_count}/{len(seq_meshes)} frames")
    else:
        src_msg = args.mtl_source_obj or "each frame OBJ"
        print(f"WARNING: Could not apply MTL colors from {src_msg}; using fallback color.")

    state = StreamState(
        audio_buffer=np.zeros(window_samples, dtype=np.float32),
        audio_queue=queue.Queue(maxsize=8),
        drive_queue=queue.Queue(maxsize=8),
        stop_event=threading.Event(),
    )

    threading.Thread(
        target=audio_drive_worker,
        args=(
            state,
            args.pitch_min_hz,
            args.pitch_max_hz,
            args.silence_db,
            args.loudness_floor_db,
            args.loudness_ceil_db,
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

    def add_mesh_for_frame(frame_idx: int):
        mesh = seq_meshes[frame_idx]
        return plotter.add_mesh(mesh, scalars="mtl_rgb", rgb=True, smooth_shading=True, name="seq_mesh", reset_camera=False)

    mesh_actor = add_mesh_for_frame(0)
    current_frame_idx = 0
    base_mesh = seq_meshes[0]
    plotter.add_axes()
    plotter.set_background("black")
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.2)
    plotter.show(auto_close=False, interactive_update=True)

    transition_sec = float(max(args.frame_transition_sec, 1e-3))
    current_frame_f = float(np.clip(args.seq_neutral_frame, args.seq_min_frame, args.seq_max_frame))
    transition_from_f = current_frame_f
    transition_to_f = current_frame_f
    transition_start_t = time.monotonic()

    try:
        while not state.stop_event.is_set():
            try:
                plotter.update()
            except Exception:
                break

            latest = None
            while True:
                try:
                    latest = state.drive_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                (drive,) = latest
                target_frame = drive_to_frame(
                    drive=drive,
                    min_frame=args.seq_min_frame,
                    neutral_frame=args.seq_neutral_frame,
                    max_frame=args.seq_max_frame,
                )
                transition_from_f = current_frame_f
                transition_to_f = float(np.clip(target_frame, args.seq_min_frame, args.seq_max_frame))
                transition_start_t = time.monotonic()

            elapsed = time.monotonic() - transition_start_t
            t = float(np.clip(elapsed / transition_sec, 0.0, 1.0))
            current_frame_f = (1.0 - t) * transition_from_f + t * transition_to_f
            frame_f = current_frame_f

            hi = int(np.searchsorted(seq_axis, frame_f, side="left"))
            if topology_consistent:
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
            else:
                if hi <= 0:
                    target_frame_idx = 0
                elif hi >= len(seq_axis):
                    target_frame_idx = len(seq_axis) - 1
                else:
                    lo = hi - 1
                    if abs(frame_f - float(seq_axis[lo])) <= abs(float(seq_axis[hi]) - frame_f):
                        target_frame_idx = lo
                    else:
                        target_frame_idx = hi
                if target_frame_idx != current_frame_idx:
                    try:
                        mesh_actor.mapper.SetInputDataObject(seq_meshes[target_frame_idx])
                    except Exception:
                        mesh_actor = add_mesh_for_frame(target_frame_idx)
                    current_frame_idx = target_frame_idx
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

