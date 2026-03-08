# audioXvis

Tools for ultrasound-based vocal tract/larynx analysis using manually annotated landmark points, geometric feature extraction, and visualization.

## Project Overview

Current runnable defaults in this repo are scoped to `data/test_dataset`.
The larger `data/dataset` tree is kept intact for archival/reference and is not used by default scripts.

This repository includes:

- Ultrasound/video datasets and annotation CSV files.
- Utility scripts for processing annotations and extracting frame-wise geometry.
- Visualization scripts for inspecting annotations and derived geometry.

## Repository Layout (high level)

- `data/` - datasets, annotations, and test data.
- `Utils/` - utility scripts for extraction, visualization, and preprocessing.
- `tracked/` - generated tracking outputs.

## Geometry Extraction Workflow

### 1 Extract geometry from annotation CSV

Script: `Utils/extract_us_geometry.py`

Input:
- Annotation CSV with 5 points per frame (`10` columns: `x,y` for each point).
- Optional source video for frame-count synchronization by interpolation.

Output:
- Geometry CSV with per-frame landmarks + triangle features.

Example:

```bash
python Utils/extract_us_geometry.py \
	-a data/test_dataset/one_experienced_singer_anno/us_s6_05.csv \
	-i data/test_dataset/one_experienced_singer_dataset_songs/s6_05.mkv \
	-o gt.csv
```

Notes:
- If `-i/--input_video` is provided, output filename automatically includes the video stem (for example `gt_s6_05.csv`).
- Annotation rows are interpolated to match the video frame count when needed.

### 2 Visualize extracted geometry side-by-side with raw video

Script: `Utils/visualize_us_geometry.py`

Input:
- Raw video (`-i`)
- Extracted geometry CSV (`-g`)

Output:
- Side-by-side MP4 (`left = raw frame`, `right = geometry panel`)

Example:

```bash
python Utils/visualize_us_geometry.py \
	-i data/test_dataset/one_experienced_singer_dataset_songs/s6_05.mkv \
	-g gt_s6_05.csv \
	-o s6_05_geometry_side_by_side.mp4
```

Optional preview while rendering:

```bash
python Utils/visualize_us_geometry.py \
	-i data/test_dataset/one_experienced_singer_dataset_songs/s6_05.mkv \
	-g gt_s6_05.csv \
	--show
```

## Extracted Geometry Columns

Per frame, extraction includes:

- Original landmark positions: `p1..p5` (`x,y` each)
- Triangle vertices:
	- `tri_a = midpoint(p1,p2)`
	- `tri_b = midpoint(p3,p4)`
	- `tri_c = p5`
- Side lengths:
	- `side_a_b`, `side_b_c`, `side_c_a`
- Angles:
	- `angle_a_deg`, `angle_b_deg`, `angle_c_deg`
- Sanity metric:
	- `angle_sum_deg` (typically near `180`)

## Suggested NN Feature Sets

This section focuses on feature choices that are robust across different speakers (different anatomy size, probe placement, and zoom).

### A General triangle (no isosceles assumption)

#### Recommended minimal (shape-only, scale-invariant)

- `2` side ratios + `1` angle
	- `r1 = side_b_c / side_a_b`
	- `r2 = side_c_a / side_a_b`
	- `theta = angle_a_deg` (or any one angle)

Why this works:

- Translation and rotation are removed automatically (uses only lengths/angles).
- Scale is removed by ratios.
- Triangle shape can be represented with `3` independent values.

#### Practical start (minimal code changes)

- Use all `3` sides + all `3` angles first.
- Normalize sides by one reference side (for example divide each side by `side_a_b`).
- Keep `2` angles for training if you want less redundancy (`angle_sum_deg` is near `180`).

### B Isosceles-shape assumption

If you assume the triangle is approximately isosceles in most frames (`b ≈ c`):

#### Isosceles shape descriptor (robust)

- `r_iso = equal_side / base_side`
- `theta_apex = apex_angle`

In this case, a very compact and stable per-frame feature set is:

- `2` values: `r_iso`, `theta_apex`

#### Strict mathematical minimum (ideal isosceles)

- `1` value can define shape (for example `theta_apex`), because ratio and angles are constrained.

In real annotated data, prefer `2` values (`r_iso + theta_apex`) to absorb noise and small deviations from perfect isosceles geometry.

### C Optional quality-control features

- `angle_sum_deg` as a QC field (should stay close to `180`).
- Isosceles consistency metric (for analysis):
	- `iso_error = abs(side_b_c - side_c_a) / max(side_b_c, side_c_a)`

You can use `iso_error` for filtering or weighting frames when training an isosceles-assumption model.

## Environment

Requirements:

- Python 3.11+
- `numpy`
- `opencv-python`

Install example:

```bash
pip install numpy opencv-python
```

## Inference Setup (Model + Realtime)

This section covers everything needed to run model inference, including realtime mode from microphone or audio-file streaming.

### Python packages required for inference

- `torch`
- `torchaudio`
- `sounddevice` (for mic/audio device input in realtime mode)
- `numpy`
- `matplotlib` (used by full-clip inference/plotting)

Install example:

```bash
pip install torch torchaudio sounddevice numpy matplotlib
```

### System package needed for `sounddevice` (Linux)

If realtime mic mode fails with PortAudio errors, install:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev
```

### Recommended conda environment (`audio2vis`)

```bash
conda create -n audio2vis python=3.12 -y
conda activate audio2vis
pip install torch torchaudio sounddevice numpy matplotlib opencv-python
```

### Checkpoint requirements

- Default realtime checkpoint path is:
  - `main/checkpoints/diffusion_v2/best.pt`
- The checkpoint must include model config and weights (`ema_state` or `model_state`).
- If you trained with multiple users/datasets, inference now auto-detects `num_users` from checkpoint weights.

### Realtime inference from microphone

Run from repo root:

```bash
source /home/lala/miniconda3/etc/profile.d/conda.sh
conda activate audio2vis
python -u main/infer_realtime.py
```

Behavior:

- Uses mic input at 44.1 kHz (`SR_MIC=44100`)
- Internally resamples to model rate 22.05 kHz
- Prints predictions continuously:
  - `Predicted angles: a=..., b=..., c=...`
- Stop with `Ctrl+C`

### Realtime-style streaming from an audio file

Use `--audio` to simulate realtime inference from a file:

```bash
source /home/lala/miniconda3/etc/profile.d/conda.sh
conda activate audio2vis
python -u main/infer_realtime.py --audio /absolute/path/to/audio.wav
```

Supported file types depend on your `torchaudio` backend (commonly `.wav`, `.flac`, and many `.mp3/.mp4` cases).

### Full-clip inference (offline, with plots)

Script:

- `main/infer_fullclip.py`

This is for non-realtime full-sequence inference and visualization, while `main/infer_realtime.py` is for live/streaming-style prediction.

## License

See `LICENSE`.
