# audioXvis

Repository accompanying the anonymous UIST 2026 submission _“Seeing Vocals: Real-Time Voice-to-Larynx dynamics for Anatomically Grounded Interactive Physiological Feedback.”_

The project combines ultrasound-derived supervision, audio-driven inference, and anatomy-grounded visualization. It includes model training/inference code, geometry extraction utilities, realtime mesh drivers, and a local browser-based demo.

## Repository Layout

- `main/`: training, offline inference, realtime inference, and mesh drivers
- `webapp/`: local web UI for live microphone or uploaded audio
- `Utils/`: data preparation, annotation, extraction, plotting, and conversion scripts
- `tracked/`: tracking artifacts and example outputs
- `data/`: local datasets, meshes, and checkpoints when present in your local setup

## Data

The repository does not include the project datasets in Git.

Expected local assets may include:

- model checkpoints such as `main/checkpoints/diffusion_v2/best.pt`
- local media, annotations, and derived geometry under `data/`
- optional sample audio used by the web app

If you clone the repository on a new machine, you will need to place the required local data and checkpoints yourself.

## Setup

Create the conda environment from `environment.yml`:

```bash
/home/usr/miniconda3/bin/conda env create -f environment.yml
/home/usr/miniconda3/bin/conda activate audio2vis
```

If `conda` is already initialized in your shell, the shorter form also works:

```bash
conda env create -f environment.yml
conda activate audio2vis
```

For microphone input on Linux, you may also need:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev ffmpeg
```

## Run The Web App

From the repository root:

```bash
python webapp/server.py
```

Then open:

```text
http://127.0.0.1:8765
```

Optional arguments:

```bash
python webapp/server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --ckpt main/checkpoints/diffusion_v2/best.pt
```

What the web app expects:

- a trained checkpoint at `main/checkpoints/diffusion_v2/best.pt`, or a custom one passed with `--ckpt`
- baked OBJ mesh frames in `main/_fbx_baked/`
- dependencies from `environment.yml`
- any optional local sample audio under `data/` if you want the sample picker populated

What happens on first launch:

- the server prepares cached mesh buffers in `webapp/generated/`
- the UI auto-discovers a few local sample audio files when available
- the app supports live microphone capture and uploaded audio files

Additional web-app notes are in `webapp/README.md`.

## Core Commands

Realtime inference from microphone:

```bash
python -u main/infer_realtime.py
```

Realtime-style streaming from an audio file:

```bash
python -u main/infer_realtime.py --audio /path/to/input.wav
```

Offline full-clip inference:

```bash
python -u main/infer_fullclip.py \
  --ckpt main/checkpoints/diffusion_v2/best.pt \
  --clips_root /path/to/full_clips \
  --output_dir main/checkpoints/diffusion_v2/inference_plots
```

## Geometry Extraction Workflow

Extract triangle geometry from a 5-point annotation CSV:

```bash
python Utils/extract_us_geometry.py \
  -a /path/to/annotation/us_example.csv \
  -i /path/to/video/example.mp4 \
  -o gt_example.csv
```

Visualize extracted geometry next to the source video:

```bash
python Utils/visualize_us_geometry.py \
  -i /path/to/video/example.mp4 \
  -g gt_example.csv \
  -o example_geometry_side_by_side.mp4
```

Each extracted frame includes:

- original landmarks `p1..p5`
- triangle vertices `tri_a`, `tri_b`, `tri_c`
- side lengths `side_a_b`, `side_b_c`, `side_c_a`
- angles `angle_a_deg`, `angle_b_deg`, `angle_c_deg`
- `angle_sum_deg` as a geometry sanity check

## Reproducibility Notes

- default checkpoint path: `main/checkpoints/diffusion_v2/best.pt`
- default web app bind address: `127.0.0.1:8765`
- the code auto-detects CPU vs CUDA
- generated browser cache files under `webapp/generated/` do not need to be committed

## License

See `LICENSE`.
