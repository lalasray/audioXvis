from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import create_audio_angles_dataloader


def main() -> None:
    root = Path("/home/lala/Documents/GitHub/audioXvis/data/test_dataset/sliding_1s_hop_0p1_all")

    dataset, loader = create_audio_angles_dataloader(
        root_dir=root,
        batch_size=16,
        shuffle=True,
        selected_features=["mel_spectrogram", "mfcc", "pitch", "rms_energy"],
        feature_mode="stats",
        target_mode="mean",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch = next(iter(loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    input_dim = x.shape[1]
    output_dim = y.shape[1]

    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    print(f"Samples in dataset: {len(dataset)}")
    print(f"Batch x: {tuple(x.shape)}")
    print(f"Batch y: {tuple(y.shape)}")
    print(f"Pred y: {tuple(pred.shape)}")
    print(f"Smoke-train loss: {loss.item():.6f}")
    print("Smoke train step completed successfully.")


if __name__ == "__main__":
    main()
