"""
Drive a realtime 3D triangle from predicted (a, b, c) angles.

Supports:
  - Live microphone input
  - Streaming from an audio file

Rendering:
  - PyVista-based interactive viewer
  - Deforming triangular prism driven by inferred angles
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__)))
from infer_fullclip import extract_features, load_model, normalize_features

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pyvista is required for mesh rendering. Install with: pip install pyvista"
    ) from exc


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
    p = argparse.ArgumentParser(description="Realtime deforming triangle from predicted angles")
    p.add_argument("--mesh", type=str, default=None, help="Optional mesh path (.obj/.ply/.stl/.glb/.gltf)")
    p.add_argument("--ckpt", type=str, default="main/checkpoints/diffusion_v2/best.pt", help="Model checkpoint")
    p.add_argument("--audio", type=str, default=None, help="Optional audio file path instead of mic input")
    p.add_argument("--sample_steps", type=int, default=None, help="Override diffusion sample steps")
    p.add_argument("--user_id", type=int, default=0, help="User id embedding index for inference")
    p.add_argument("--smooth_alpha", type=float, default=0.25, help="EMA smoothing factor for triangle angles")

    # Angle controls
    p.add_argument("--neutral_a", type=float, default=65.0, help="Neutral baseline for angle a")
    p.add_argument("--neutral_b", type=float, default=65.0, help="Neutral baseline for angle b")
    p.add_argument("--neutral_c", type=float, default=50.0, help="Neutral baseline for angle c")
    p.add_argument("--triangle_base", type=float, default=1.0, help="Base side length for generated triangle")
    p.add_argument("--triangle_thickness", type=float, default=0.2, help="Prism thickness (z dimension)")
    p.add_argument("--angle_min", type=float, default=20.0, help="Lower angle bound for color mapping")
    p.add_argument("--angle_max", type=float, default=100.0, help="Upper angle bound for color mapping")
    p.add_argument(
        "--mesh_deform_mode",
        type=str,
        default="virtual_rig",
        choices=["rigid", "virtual_rig"],
        help="Mesh animation mode when --mesh is provided",
    )
    p.add_argument("--mesh_gain_a", type=float, default=0.9, help="Rotation gain for angle a in mesh mode")
    p.add_argument("--mesh_gain_b", type=float, default=0.9, help="Rotation gain for angle b in mesh mode")
    p.add_argument("--mesh_gain_c", type=float, default=0.9, help="Rotation gain for angle c in mesh mode")
    return p.parse_args()


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


def inference_worker(
    state: StreamState,
    model,
    y_mean,
    y_std,
    sample_steps: int,
    feature_stats,
    user_id: int,
):
    while not state.stop_event.is_set():
        try:
            y_window = state.inf_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            y_window_resampled = torchaudio.functional.resample(
                torch.from_numpy(y_window), SR_MIC, SR_MODEL
            ).numpy()
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

            print(f"Predicted angles: a={a:.2f}, b={b:.2f}, c={c:.2f}")
            try:
                state.pose_queue.put((a, b, c), block=False)
            except queue.Full:
                pass
        except Exception as exc:
            print(f"Inference error: {exc}")
        finally:
            state.inf_queue.task_done()


def stream_audio_file(state: StreamState, audio_path: str, window_samples: int, hop_samples: int):
    waveform, sr = torchaudio.load(audio_path)
    if sr != SR_MIC:
        waveform = torchaudio.functional.resample(waveform, sr, SR_MIC)
    audio_np = waveform[0].numpy()

    idx = 0
    while idx + window_samples <= len(audio_np) and not state.stop_event.is_set():
        chunk = audio_np[idx : idx + window_samples]
        try:
            state.inf_queue.put(chunk.copy(), block=False)
        except queue.Full:
            pass
        idx += hop_samples
        time.sleep(HOP_SEC)


def triangle_vertices_from_angles(a_deg: float, b_deg: float, c_deg: float, base_len: float) -> np.ndarray:
    """Return 3D vertices for triangle A(0,0), B(x,y), C(base,0) from interior angles."""
    eps = 1e-6
    a_deg = float(max(a_deg, eps))
    b_deg = float(max(b_deg, eps))
    c_deg = float(max(c_deg, eps))
    total = a_deg + b_deg + c_deg
    if total < eps:
        a_deg, b_deg, c_deg = 60.0, 60.0, 60.0
    else:
        scale = 180.0 / total
        a_deg *= scale
        b_deg *= scale
        c_deg *= scale

    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)
    c_rad = np.deg2rad(c_deg)
    sin_b = max(float(np.sin(b_rad)), eps)

    # Side lengths by law of sines with AC fixed to base_len (opposite angle B).
    side_a = base_len * np.sin(a_rad) / sin_b  # opposite A, between B and C
    side_c = base_len * np.sin(c_rad) / sin_b  # opposite C, between A and B

    x_b = (side_c ** 2 + base_len ** 2 - side_a ** 2) / (2.0 * base_len)
    y_sq = max(side_c ** 2 - x_b ** 2, 0.0)
    y_b = np.sqrt(y_sq)

    return np.array(
        [[0.0, 0.0, 0.0], [x_b, y_b, 0.0], [base_len, 0.0, 0.0]],
        dtype=np.float32,
    )


def prism_from_triangle(tri_pts: np.ndarray, thickness: float) -> np.ndarray:
    """Build 6 prism vertices from 3 triangle vertices centered on z=0."""
    half = float(thickness) / 2.0
    bottom = tri_pts.copy()
    top = tri_pts.copy()
    bottom[:, 2] = -half
    top[:, 2] = half
    return np.vstack([bottom, top]).astype(np.float32)


def make_prism_polydata(points6: np.ndarray) -> pv.PolyData:
    # 6 points: 0:A 1:B 2:C 3:A' 4:B' 5:C'
    faces = np.hstack(
        [
            [3, 0, 1, 2],  # bottom triangle
            [3, 3, 5, 4],  # top triangle
            [4, 0, 1, 4, 3],  # AB side
            [4, 1, 2, 5, 4],  # BC side
            [4, 2, 0, 3, 5],  # CA side
        ]
    ).astype(np.int64)
    return pv.PolyData(points6, faces)


def load_supported_mesh(mesh_path: str) -> pv.DataSet:
    ext = os.path.splitext(mesh_path.lower())[1]
    if ext == ".fbx":
        raise RuntimeError(
            "FBX is not supported by the current PyVista/VTK reader stack in this environment. "
            "Convert to .obj or .glb first, then pass --mesh with the converted file."
        )
    return pv.read(mesh_path)


def rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def apply_rotation(points: np.ndarray, pivot: np.ndarray, rmat: np.ndarray) -> np.ndarray:
    return (points - pivot) @ rmat.T + pivot


def top_color_surface_points(tri_pts: np.ndarray, z_top: float) -> np.ndarray:
    a, b, c = tri_pts
    m_ab = 0.5 * (a + b)
    m_bc = 0.5 * (b + c)
    m_ca = 0.5 * (c + a)
    center = (a + b + c) / 3.0

    pts = np.array([a, b, c, m_ab, m_bc, m_ca, center], dtype=np.float32)
    pts[:, 2] = z_top
    return pts


def top_color_surface_faces() -> np.ndarray:
    # points: 0:A 1:B 2:C 3:Mab 4:Mbc 5:Mca 6:center
    return np.hstack(
        [
            [3, 0, 3, 6],
            [3, 3, 1, 6],
            [3, 1, 4, 6],
            [3, 4, 2, 6],
            [3, 2, 5, 6],
            [3, 5, 0, 6],
        ]
    ).astype(np.int64)


def angle_colors_from_angles(
    a: float, b: float, c: float, angle_min: float, angle_max: float
) -> tuple[np.ndarray, np.ndarray]:
    denom = max(angle_max - angle_min, 1e-6)
    vals = np.array([(a - angle_min) / denom, (b - angle_min) / denom, (c - angle_min) / denom], dtype=np.float32)
    vals = np.clip(vals, 0.0, 1.0)

    # Three angle anchors with distinct base hues.
    base = np.array(
        [
            [255.0, 60.0, 60.0],   # angle a anchor
            [70.0, 255.0, 120.0],  # angle b anchor
            [80.0, 120.0, 255.0],  # angle c anchor
        ],
        dtype=np.float32,
    )
    # Keep some baseline visibility, then brighten with angle value.
    strength = (0.25 + 0.75 * vals).reshape(3, 1)
    angle_rgb = np.clip(base * strength, 0.0, 255.0)
    return angle_rgb, vals


def triangle_point_colors(angle_rgb: np.ndarray, vals: np.ndarray) -> np.ndarray:
    # angle colors: 0=a, 1=b, 2=c
    c_a, c_b, c_c = angle_rgb
    v_a, v_b, v_c = vals

    # Vertex anchor colors (one per predicted angle).
    col_a = c_a
    col_b = c_b
    col_c = c_c

    # Midpoints blend adjacent angle-anchor colors.
    col_mab = 0.5 * (c_a + c_b)
    col_mbc = 0.5 * (c_b + c_c)
    col_mca = 0.5 * (c_c + c_a)

    # Center is weighted mix of all three angle-anchor colors.
    w = np.array([v_a, v_b, v_c], dtype=np.float32)
    w_sum = float(w.sum())
    if w_sum < 1e-6:
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        w_sum = 3.0
    w = w / w_sum
    col_center = w[0] * c_a + w[1] * c_b + w[2] * c_c

    cols = np.vstack([col_a, col_b, col_c, col_mab, col_mbc, col_mca, col_center])
    return np.clip(cols, 0.0, 255.0).astype(np.uint8)


def main():
    args = parse_args()
    window_samples = int(WINDOW_SEC * SR_MIC)
    hop_samples = int(HOP_SEC * SR_MIC)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if args.mesh is not None and not os.path.exists(args.mesh):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh}")

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model, y_mean, y_std, default_steps, _cfg, feature_stats = load_model(args.ckpt, DEVICE)
    sample_steps = args.sample_steps if args.sample_steps is not None else default_steps
    print(f"Model loaded. sample_steps={sample_steps}")

    state = StreamState(
        audio_buffer=np.zeros(window_samples, dtype=np.float32),
        inf_queue=queue.Queue(maxsize=8),
        pose_queue=queue.Queue(maxsize=8),
        stop_event=threading.Event(),
    )

    inf_thread = threading.Thread(
        target=inference_worker,
        args=(state, model, y_mean, y_std, sample_steps, feature_stats, args.user_id),
        daemon=True,
    )
    inf_thread.start()

    feeder_thread = None
    if args.audio:
        print(f"Streaming from audio file: {args.audio}")
        feeder_thread = threading.Thread(
            target=stream_audio_file,
            args=(state, args.audio, window_samples, hop_samples),
            daemon=True,
        )
        feeder_thread.start()
    else:
        print("Using microphone input. Press Ctrl+C in terminal or close viewer to stop.")

    init_tri = triangle_vertices_from_angles(
        args.neutral_a, args.neutral_b, args.neutral_c, args.triangle_base
    )
    prism_mesh = make_prism_polydata(prism_from_triangle(init_tri, args.triangle_thickness))
    z_top = float(args.triangle_thickness) / 2.0 + 1e-3
    color_pts = top_color_surface_points(init_tri, z_top)
    color_mesh = pv.PolyData(color_pts, top_color_surface_faces())
    init_angle_rgb, init_vals = angle_colors_from_angles(
        args.neutral_a, args.neutral_b, args.neutral_c, args.angle_min, args.angle_max
    )
    color_mesh["rgb"] = triangle_point_colors(init_angle_rgb, init_vals)
    use_mesh_mode = args.mesh is not None
    if use_mesh_mode:
        larynx_mesh = load_supported_mesh(args.mesh)
        if isinstance(larynx_mesh, pv.MultiBlock):
            larynx_mesh = larynx_mesh.combine()
        larynx_mesh = larynx_mesh.extract_surface().triangulate()
        rest_points = np.array(larynx_mesh.points, dtype=np.float32)
        y_vals = rest_points[:, 1]
        y_lo = float(np.quantile(y_vals, 0.33))
        y_hi = float(np.quantile(y_vals, 0.66))
        cx = float(rest_points[:, 0].mean())
        cz = float(rest_points[:, 2].mean())
        piv0 = np.array([cx, float(y_vals.min() + 0.10 * (y_vals.max() - y_vals.min())), cz], dtype=np.float32)
        piv1 = np.array([cx, y_lo, cz], dtype=np.float32)
        piv2 = np.array([cx, y_hi, cz], dtype=np.float32)
        print(f"Using mesh mode with: {args.mesh}")
    else:
        larynx_mesh = None
        rest_points = None
        y_lo = y_hi = 0.0
        piv0 = piv1 = piv2 = np.zeros(3, dtype=np.float32)
        print("Using deforming 3D triangle prism driven by predicted angles.")

    plotter = pv.Plotter(window_size=(1200, 900))
    if use_mesh_mode:
        actor = plotter.add_mesh(larynx_mesh, color="lightcoral", smooth_shading=True)
    else:
        actor = None
        plotter.add_mesh(prism_mesh, color="white", opacity=0.16, smooth_shading=True, show_edges=True)
        plotter.add_mesh(color_mesh, scalars="rgb", rgb=True, smooth_shading=True, show_edges=False)

    plotter.add_axes()
    plotter.set_background("black")
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.2)
    plotter.show(auto_close=False, interactive_update=True)

    if not args.audio:
        stream = sd.InputStream(
            channels=1,
            samplerate=SR_MIC,
            callback=make_audio_callback(state),
            blocksize=hop_samples,
        )
        stream.start()
    else:
        stream = None

    smoothed = np.array([args.neutral_a, args.neutral_b, args.neutral_c], dtype=np.float32)
    alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))

    try:
        while not state.stop_event.is_set():
            # Keep the PyVista interactor alive; this raises when window is closed.
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
                a, b, c = latest
                target = np.array([a, b, c], dtype=np.float32)
                smoothed = alpha * target + (1.0 - alpha) * smoothed
                if use_mesh_mode:
                    da = float((smoothed[0] - args.neutral_a) * args.mesh_gain_a)
                    db = float((smoothed[1] - args.neutral_b) * args.mesh_gain_b)
                    dc = float((smoothed[2] - args.neutral_c) * args.mesh_gain_c)
                    if args.mesh_deform_mode == "rigid":
                        actor.SetOrientation(da, db, dc)
                    else:
                        # Virtual armature: 3 chained joint rotations along the y-axis bands.
                        pts = rest_points.copy()
                        idx0 = pts[:, 1] < y_lo
                        idx1 = (pts[:, 1] >= y_lo) & (pts[:, 1] < y_hi)
                        idx2 = pts[:, 1] >= y_hi

                        r0 = rot_x(da)
                        r1 = rot_z(db)
                        r2 = rot_y(dc)

                        if np.any(idx0):
                            pts[idx0] = apply_rotation(pts[idx0], piv0, r0)
                        if np.any(idx1):
                            p = apply_rotation(pts[idx1], piv0, r0)
                            pts[idx1] = apply_rotation(p, piv1, r1)
                        if np.any(idx2):
                            p = apply_rotation(pts[idx2], piv0, r0)
                            p = apply_rotation(p, piv1, r1)
                            pts[idx2] = apply_rotation(p, piv2, r2)

                        larynx_mesh.points = pts
                else:
                    tri = triangle_vertices_from_angles(
                        float(smoothed[0]),
                        float(smoothed[1]),
                        float(smoothed[2]),
                        args.triangle_base,
                    )
                    prism_mesh.points = prism_from_triangle(tri, args.triangle_thickness)

                    pts = top_color_surface_points(tri, z_top)
                    color_mesh.points = pts
                    angle_rgb, vals = angle_colors_from_angles(
                        float(smoothed[0]),
                        float(smoothed[1]),
                        float(smoothed[2]),
                        args.angle_min,
                        args.angle_max,
                    )
                    color_mesh["rgb"] = triangle_point_colors(angle_rgb, vals)

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
