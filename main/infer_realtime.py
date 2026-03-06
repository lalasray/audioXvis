"""
Real-time larynx angle prediction from microphone audio.
- Captures audio from mic in sliding windows
- Runs model inference for each window
- Prints predicted angles in real time
"""

import torch
import torchaudio
import sounddevice as sd
import numpy as np
import sys
import os
import threading
import queue
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__)))
from infer_fullclip import extract_features, normalize_features, load_model

BATCH_SIZE = 1
SR_MIC = 44100  # Mic input sample rate
SR_MODEL = 22050  # Model training sample rate
WINDOW_SEC = 0.5  # Faster window
HOP_SEC = 0.25   # Faster hop
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = "main/checkpoints/diffusion_v2/best.pt"

print("Loading model...")
model, y_mean, y_std, sample_steps, cfg, feature_stats = load_model(CKPT, DEVICE)
print("Model loaded.")

window_samples = int(WINDOW_SEC * SR_MIC)
hop_samples = int(HOP_SEC * SR_MIC)


parser = argparse.ArgumentParser(description="Realtime larynx angle prediction")
parser.add_argument('--audio', type=str, default=None, help='Path to audio file to stream instead of mic')
args = parser.parse_args()

audio_buffer = np.zeros(window_samples, dtype=np.float32)
inference_queue = queue.Queue(maxsize=2)
pred_history = []

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]
    try:
        inference_queue.put(audio_buffer.copy(), block=False)
    except queue.Full:
        pass

def inference_worker():
    while True:
        try:
            y_window = inference_queue.get()
            y_window_resampled = torchaudio.functional.resample(torch.from_numpy(y_window), SR_MIC, SR_MODEL).numpy()
            feats = extract_features(y_window_resampled, sr=SR_MODEL)
            if feature_stats is not None:
                feats = normalize_features(feats, feature_stats)
            x_dict = {k: torch.from_numpy(feats[k]).unsqueeze(0).to(DEVICE) for k in feats}
            user_id = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            pred_norm = model.sample_ddim(x_dict, sample_steps=sample_steps, user_id=user_id)
            pred = pred_norm * y_std + y_mean
            pred_a = float(pred[0, 0].cpu().numpy())
            pred_c = float(pred[0, 1].cpu().numpy())
            pred_b = 180.0 - pred_a - pred_c
            print(f"Predicted angles: a={pred_a:.2f}, b={pred_b:.2f}, c={pred_c:.2f}")
            if len(pred_history) > 1000:
                pred_history.clear()
            pred_history.append((pred_a, pred_b, pred_c))
            inference_queue.task_done()
        except Exception as e:
            print(f"Inference error: {e}")

threading.Thread(target=inference_worker, daemon=True).start()

if args.audio:
    print(f"Streaming audio file: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)
    if sr != SR_MIC:
        waveform = torchaudio.functional.resample(waveform, sr, SR_MIC)
    audio_np = waveform[0].numpy()
    n_samples = len(audio_np)
    idx = 0
    try:
        while idx + window_samples <= n_samples:
            audio_buffer = audio_np[idx:idx+window_samples]
            if inference_queue.qsize() < 10:
                inference_queue.put(audio_buffer.copy())
            idx += hop_samples
            sd.sleep(10)  # Minimal sleep for stability
    except KeyboardInterrupt:
        print("Stopped.")
else:
    print("Speak into the mic. Press Ctrl+C to stop.")
    with sd.InputStream(channels=1, samplerate=SR_MIC, callback=audio_callback, blocksize=hop_samples):
        try:
            while True:
                sd.sleep(int(HOP_SEC * 1000))
        except KeyboardInterrupt:
            print("Stopped.")
