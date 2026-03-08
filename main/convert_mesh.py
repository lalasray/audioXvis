"""
Convert mesh files for realtime visualization.

Primary use:
  FBX -> OBJ/GLB via Blender in background mode.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BLENDER_SCRIPT = r"""
import bpy
import os
import sys

argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Missing '--' args for blender script")
args = argv[argv.index("--") + 1 :]
if len(args) != 2:
    raise RuntimeError("Expected: <input_path> <output_path>")

in_path = args[0]
out_path = args[1]
out_ext = os.path.splitext(out_path.lower())[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=in_path)
bpy.ops.object.select_all(action="SELECT")

if out_ext == ".obj":
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=out_path, export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=out_path, use_selection=True)
elif out_ext == ".glb":
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLB",
        use_selection=True,
    )
else:
    raise RuntimeError(f"Unsupported output extension: {out_ext}")
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert mesh format for realtime driver")
    p.add_argument("--input", required=True, help="Input mesh path (e.g. .fbx)")
    p.add_argument(
        "--output",
        default=None,
        help="Output mesh path (.obj or .glb). Defaults next to input with .obj",
    )
    p.add_argument(
        "--prefer",
        default="blender",
        choices=["blender", "assimp"],
        help="Preferred conversion backend",
    )
    return p.parse_args()


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def convert_with_blender(input_path: Path, output_path: Path) -> tuple[bool, str]:
    blender = shutil.which("blender")
    if blender is None:
        return False, "Blender not found in PATH."

    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / "blender_convert.py"
        script_path.write_text(BLENDER_SCRIPT, encoding="utf-8")
        cmd = [
            blender,
            "-b",
            "--python",
            str(script_path),
            "--",
            str(input_path),
            str(output_path),
        ]
        proc = run_cmd(cmd)
        if proc.returncode == 0 and output_path.exists():
            return True, f"Blender conversion successful: {output_path}"

        msg = [
            "Blender conversion failed.",
            f"Command: {' '.join(cmd)}",
            "stdout:",
            proc.stdout.strip(),
            "stderr:",
            proc.stderr.strip(),
        ]
        return False, "\n".join(msg)


def convert_with_assimp(input_path: Path, output_path: Path) -> tuple[bool, str]:
    assimp = shutil.which("assimp")
    if assimp is None:
        return False, "assimp not found in PATH."

    cmd = [assimp, "export", str(input_path), str(output_path)]
    proc = run_cmd(cmd)
    if proc.returncode == 0 and output_path.exists():
        return True, f"assimp conversion successful: {output_path}"
    msg = [
        "assimp conversion failed.",
        f"Command: {' '.join(cmd)}",
        "stdout:",
        proc.stdout.strip(),
        "stderr:",
        proc.stderr.strip(),
    ]
    return False, "\n".join(msg)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 1

    if args.output is None:
        output_path = input_path.with_suffix(".obj")
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_ext = output_path.suffix.lower()
    if output_ext not in {".obj", ".glb"}:
        print(f"ERROR: output extension must be .obj or .glb, got: {output_ext}")
        return 1

    backends = [args.prefer, "assimp" if args.prefer == "blender" else "blender"]
    converters = {
        "blender": convert_with_blender,
        "assimp": convert_with_assimp,
    }

    for backend in backends:
        ok, msg = converters[backend](input_path, output_path)
        print(msg)
        if ok:
            print("\nUse this in realtime driver:")
            print(
                f"python -u main/realtime_mesh_driver.py --ckpt main/checkpoints/diffusion_v2/best.pt --mesh {output_path}"
            )
            return 0

    print(
        "\nAll conversion backends failed.\n"
        "If Blender is from snap and fails with confinement/AppArmor, install a non-snap Blender build and retry."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

