"""
Train and evaluate a real-time conditional diffusion regressor.

Pipeline:
  full clip videos + annotations
    -> 1s rolling windows (hop 0.5s)
    -> 5 audio features (mel, mfcc, rms, zcr, spectral_contrast)
    -> predict angle_a, angle_c  (derive angle_b = 180 - a - c)
"""

from __future__ import annotations

import argparse
import json
import math
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloader_fullclip import (
    FullClipRollingDataset,
    rolling_collate,
    PREDICT_COLUMNS,
)


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    clips_root: str = "data/test_dataset/full_clips"
    batch_size: int = 64
    epochs: int = 80
    lr: float = 5e-4
    weight_decay: float = 1e-5
    num_workers: int = 0      # 0 because audio is pre-loaded into RAM
    seed: int = 42
    val_ratio: float = 0.1
    diffusion_steps: int = 100
    sample_steps: int = 20
    hidden_dim: int = 512
    t_embed_dim: int = 128
    feature_embed_dim: int = 64
    window_sec: float = 1.0
    hop_sec: float = 0.5
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    output_dir: str = "main/checkpoints/diffusion_fullclip"


# ── model components (same architecture as existing, but y_dim=2) ────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, t_embed_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.Linear(t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )
        self.t_embed_dim = t_embed_dim

    def forward(self, x_cond: torch.Tensor, y_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.t_embed_dim)
        t_feat = self.t_proj(t_emb)
        h = torch.cat([x_cond, y_noisy, t_feat], dim=1)
        return self.net(h)


class FeatureCNNEncoder(nn.Module):
    def __init__(self, feature_shapes: Dict[str, Tuple[int, ...]], embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.feature_order = sorted(feature_shapes.keys())
        self.branches = nn.ModuleDict()
        self.branch_projs = nn.ModuleDict()

        for key in self.feature_order:
            shape = feature_shapes[key]
            if len(shape) == 1:
                in_channels = 1
            elif len(shape) == 2:
                in_channels = int(shape[0])
            else:
                raise ValueError(f"Unsupported feature shape for {key}: {shape}")

            self.branches[key] = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.branch_projs[key] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, embed_dim),
                nn.SiLU(),
            )

        fusion_dim = embed_dim * len(self.feature_order)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        for key in self.feature_order:
            x = x_dict[key]
            if x.ndim == 2:
                x = x.unsqueeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Feature {key} must have [B,T] or [B,C,T], got {tuple(x.shape)}")
            feat = self.branches[key](x)
            feat = self.branch_projs[key](feat)
            encoded.append(feat)
        fused = torch.cat(encoded, dim=1)
        return self.fusion(fused)


class DiffusionRegressor(nn.Module):
    def __init__(
        self,
        feature_shapes: Dict[str, Tuple[int, ...]],
        y_dim: int,
        steps: int,
        t_embed_dim: int,
        hidden_dim: int,
        feature_embed_dim: int,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.y_dim = y_dim
        self.encoder = FeatureCNNEncoder(
            feature_shapes=feature_shapes,
            embed_dim=feature_embed_dim,
            out_dim=hidden_dim,
        )
        self.denoiser = ConditionalDenoiser(
            x_dim=hidden_dim,
            y_dim=y_dim,
            t_embed_dim=t_embed_dim,
            hidden_dim=hidden_dim,
        )

        betas = torch.linspace(1e-4, 2e-2, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(a_bar) * y0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_noise(self, x_dict: Dict[str, torch.Tensor], y_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_cond = self.encoder(x_dict)
        return self.denoiser(x_cond, y_noisy, t)

    @torch.no_grad()
    def sample_ddim(self, x_dict: Dict[str, torch.Tensor], sample_steps: int) -> torch.Tensor:
        first_tensor = next(iter(x_dict.values()))
        device = first_tensor.device
        batch_size = first_tensor.shape[0]
        y = torch.randn(batch_size, self.y_dim, device=device)
        x_cond = self.encoder(x_dict)

        ts = torch.linspace(self.steps - 1, 0, sample_steps, device=device).long()

        for i in range(len(ts)):
            t = ts[i]
            t_batch = torch.full((batch_size,), int(t.item()), device=device, dtype=torch.long)
            eps = self.denoiser(x_cond, y, t_batch)
            a_bar = self.alpha_bars[t]
            sqrt_ab = torch.sqrt(a_bar)
            sqrt_1mab = torch.sqrt(1.0 - a_bar)
            y0_hat = (y - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

            if i == len(ts) - 1:
                y = y0_hat
            else:
                t_prev = ts[i + 1]
                a_bar_prev = self.alpha_bars[t_prev]
                y = torch.sqrt(a_bar_prev) * y0_hat + torch.sqrt(1.0 - a_bar_prev) * eps

        return y


# ── helpers ──────────────────────────────────────────────────────────────────

def move_dict(x_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in x_dict.items()}


def derive_angle_b(pred_a: torch.Tensor, pred_c: torch.Tensor) -> torch.Tensor:
    """Derive angle_b from the triangle constraint: a + b + c = 180."""
    return 180.0 - pred_a - pred_c


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: DiffusionRegressor,
    loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mae_ac_total = 0.0
    mse_ac_total = 0.0
    mae_b_total = 0.0
    n_ac = 0
    n_b = 0

    for batch in loader:
        x = move_dict(batch["x"], device)
        y = batch["y"].to(device, non_blocking=True)  # (B, 2) = [a, c]

        y_pred_norm = model.sample_ddim(x, sample_steps=sample_steps)
        y_pred = y_pred_norm * y_std + y_mean  # un-normalize

        # angle_a, angle_c error
        mae_ac_total += torch.abs(y_pred - y).sum().item()
        mse_ac_total += torch.square(y_pred - y).sum().item()
        n_ac += y.numel()

        # Derive angle_b and compute its error
        pred_b = derive_angle_b(y_pred[:, 0], y_pred[:, 1])
        gt_b = 180.0 - y[:, 0] - y[:, 1]
        mae_b_total += torch.abs(pred_b - gt_b).sum().item()
        n_b += pred_b.numel()

    mae_ac = mae_ac_total / max(n_ac, 1)
    rmse_ac = math.sqrt(mse_ac_total / max(n_ac, 1))
    mae_b = mae_b_total / max(n_b, 1)

    return {"mae_ac": mae_ac, "rmse_ac": rmse_ac, "mae_b": mae_b}


# ── training ─────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── data ──
    dataset = FullClipRollingDataset(
        clips_root=cfg.clips_root,
        window_sec=cfg.window_sec,
        hop_sec=cfg.hop_sec,
    )

    total = len(dataset)
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(total, generator=generator)

    val_size = max(1, int(total * cfg.val_ratio))
    train_size = total - val_size
    train_indices = perm[:train_size].tolist()
    val_indices = perm[train_size:].tolist()

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=rolling_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=rolling_collate,
        pin_memory=True,
    )

    sample = dataset[0]["x"]
    feature_shapes = {k: tuple(v.shape) for k, v in sample.items()}
    y_dim = len(PREDICT_COLUMNS)  # 2

    print(f"\nFeature shapes:")
    for k, v in feature_shapes.items():
        print(f"  {k}: {v}")
    print(f"Target dim: {y_dim} ({', '.join(PREDICT_COLUMNS)})")
    print(f"Train: {train_size}, Val: {val_size}")

    # ── model ──
    model = DiffusionRegressor(
        feature_shapes=feature_shapes,
        y_dim=y_dim,
        steps=cfg.diffusion_steps,
        t_embed_dim=cfg.t_embed_dim,
        hidden_dim=cfg.hidden_dim,
        feature_embed_dim=cfg.feature_embed_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── target normalization ──
    print("Computing target normalization stats...")
    y_all = []
    for batch in train_loader:
        y_all.append(batch["y"])
    y_cat = torch.cat(y_all, dim=0).to(device)
    y_mean = y_cat.mean(dim=0, keepdim=True)
    y_std = y_cat.std(dim=0, keepdim=True).clamp_min(1e-6)
    print(f"Target mean: {y_mean.cpu().numpy().flatten()}")
    print(f"Target std:  {y_std.cpu().numpy().flatten()}")

    # ── training loop ──
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print(f"\nStarting training for {cfg.epochs} epochs...\n")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            x = move_dict(batch["x"], device)
            y = batch["y"].to(device, non_blocking=True)
            y_norm = (y - y_mean) / y_std

            bs = y.shape[0]
            t = torch.randint(0, cfg.diffusion_steps, (bs,), device=device)
            noise = torch.randn_like(y_norm)
            y_noisy = model.q_sample(y_norm, t=t, noise=noise)

            pred_noise = model.predict_noise(x, y_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)
        metrics = evaluate(model, val_loader, y_mean, y_std, cfg.sample_steps, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6f} | "
            f"val_MAE(a,c)={metrics['mae_ac']:.4f}° | "
            f"val_RMSE(a,c)={metrics['rmse_ac']:.4f}° | "
            f"val_MAE(b=180-a-c)={metrics['mae_b']:.4f}°"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "y_mean": y_mean.detach().cpu(),
            "y_std": y_std.detach().cpu(),
            "feature_shapes": feature_shapes,
            "y_dim": y_dim,
            "predict_columns": list(PREDICT_COLUMNS),
            "epoch": epoch,
            "metrics": metrics,
        }

        torch.save(ckpt, out_dir / "last.pt")

        if (best_val_mae - metrics["mae_ac"]) > cfg.early_stopping_min_delta:
            best_val_mae = metrics["mae_ac"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ** New best: val_MAE(a,c)={best_val_mae:.4f}°")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch}: no improvement for "
                f"{cfg.early_stopping_patience} epochs. "
                f"Best epoch={best_epoch}, best val_MAE(a,c)={best_val_mae:.4f}°"
            )
            break

    # ── save history ──
    hist_path = out_dir / "training_history.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with (out_dir / "train_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nTraining complete. Best epoch={best_epoch}, val_MAE(a,c)={best_val_mae:.4f}°")
    print(f"Checkpoints: {out_dir}")

    # ── final evaluation on full val set with best model ──
    print("\n" + "=" * 70)
    print("Final evaluation with best checkpoint")
    print("=" * 70)

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    final_metrics = evaluate(model, val_loader, y_mean, y_std, cfg.sample_steps, device)
    print(f"  val MAE  (angle_a, angle_c): {final_metrics['mae_ac']:.4f}°")
    print(f"  val RMSE (angle_a, angle_c): {final_metrics['rmse_ac']:.4f}°")
    print(f"  val MAE  (angle_b = 180-a-c): {final_metrics['mae_b']:.4f}°")
    print(f"\n  angle_b is derived via:  angle_b = 180 - angle_a - angle_c")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train diffusion regressor on full-clip rolling windows"
    )
    parser.add_argument("--clips_root", type=str, default="data/test_dataset/full_clips")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--t_embed_dim", type=int, default=128)
    parser.add_argument("--feature_embed_dim", type=int, default=64)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="main/checkpoints/diffusion_fullclip")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
