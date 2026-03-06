"""
Generate GT vs PD triangle videos for all test clips.
- Extract GT and PD angles from annotation and inference.
- Save CSVs for each clip.
- Run triangle video script for each.
"""

import os
import glob
import numpy as np
import pandas as pd
import subprocess

INFER_PLOTS = "main/checkpoints/diffusion_v2/inference_plots"
CLIPS_ROOT = "data/test_dataset/full_clips"
TRIANGLE_SCRIPT = "main/plot_triangle_gt_vs_pd_video.py"

for clip_dir in sorted(os.listdir(CLIPS_ROOT)):
    clip_path = os.path.join(CLIPS_ROOT, clip_dir)
    anno_csv = os.path.join(clip_path, "annotation", f"full_{clip_dir.split('__')[0]}.csv")
    pred_png = os.path.join(INFER_PLOTS, f"{clip_dir}_gt_vs_pred.png")
    gt_csv = os.path.join(INFER_PLOTS, f"{clip_dir}_gt_angles.csv")
    pd_csv = os.path.join(INFER_PLOTS, f"{clip_dir}_pred_angles.csv")
    out_video = os.path.join(INFER_PLOTS, f"{clip_dir}_triangle_gt_vs_pd.mp4")

    if not os.path.exists(anno_csv) or not os.path.exists(pred_png):
        print(f"Skipping {clip_dir}: missing annotation or prediction plot.")
        continue

    # Extract GT angles
    df_gt = pd.read_csv(anno_csv)
    gt_angles = df_gt[["frame_idx", "angle_a_deg", "angle_b_deg", "angle_c_deg"]].copy()
    gt_angles["time"] = gt_angles["frame_idx"] / 60.0  # Assuming 60fps
    gt_angles = gt_angles[["frame_idx", "time", "angle_a_deg", "angle_b_deg", "angle_c_deg"]]
    gt_angles.columns = ["frame", "time", "angle_a", "angle_b", "angle_c"]
    gt_angles.to_csv(gt_csv, index=False)

    # Placeholder: Extract PD angles from inference (not implemented, needs code)
    # pd_angles = ...
    # pd_angles.to_csv(pd_csv, index=False)

    # Run triangle video script
    if os.path.exists(gt_csv) and os.path.exists(pd_csv):
        subprocess.run([
            "python", TRIANGLE_SCRIPT,
            "--gt_csv", gt_csv,
            "--pd_csv", pd_csv,
            "--output", out_video,
            "--fps", "30"
        ])
    else:
        print(f"Missing GT or PD CSV for {clip_dir}, skipping video generation.")
