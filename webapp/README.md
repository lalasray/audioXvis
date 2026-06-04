# Web App

This app serves the local Audio2Vis demo UI. It runs inference on live microphone input or uploaded audio, then drives the baked larynx mesh sequence in the browser.

## Setup

Create and activate the conda environment from the repository root:

```bash
/home/lala/miniconda3/bin/conda env create -f environment.yml
/home/lala/miniconda3/bin/conda activate audio2vis
```

If `conda` is already initialized in your shell, the shorter form also works:

```bash
conda env create -f environment.yml
conda activate audio2vis
```

## Run

From the repository root:

```bash
python webapp/server.py
```

Then open `http://127.0.0.1:8765`.

## Optional Arguments

```bash
python webapp/server.py --host 127.0.0.1 --port 8765 --ckpt main/checkpoints/diffusion_v2/best.pt
```

## Requirements

- Python dependencies from `environment.yml`
- A model checkpoint at `main/checkpoints/diffusion_v2/best.pt` or a custom path passed with `--ckpt`
- `ffmpeg` in `PATH` if you want broader audio format fallback support

## Notes

- The first launch creates cached mesh assets in `webapp/generated/`.
- Microphone sessions are saved as local WAV files in `webapp/generated/recordings/`.
- The UI auto-discovers a few bundled sample audio files when available.
- If microphone capture fails on Linux, install PortAudio development/runtime packages.
