"""
Train a conditional diffusion regressor — V2 (improved).

Improvements over V1:
  - Per-channel feature normalization (computed from training data)
  - Deeper encoder: ResConv1d blocks + GroupNorm + attention pooling
  - Improved denoiser: AdaLN residual MLP blocks
  - Cosine noise schedule
  - EMA of model weights
  - Cosine annealing LR with warm restarts
  - Gradient clipping
  - Mixup augmentation
  - Denser windows (hop 0.25s)
  - Clip-based train/val split (no leakage)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloader_fullclip import (
    FullClipRollingDataset,
    compute_feature_stats,
    rolling_collate,
    PREDICT_COLUMNS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    clips_root: str = "data/test_dataset/full_clips"
    batch_size: int = 64
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    val_ratio: float = 0.15   # ~4-5 clips for val
    # diffusion
    diffusion_steps: int = 200
    sample_steps: int = 50
    # model
    hidden_dim: int = 512
    t_embed_dim: int = 128
    feature_embed_dim: int = 128
    branch_channels: int = 64
    n_denoiser_blocks: int = 4
    dropout: float = 0.1
    # windows
    window_sec: float = 1.0
    hop_sec: float = 0.25
    # training
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    mixup_alpha: float = 0.3
    holdout_clips: str = ""  # comma-separated clip names to exclude from train+val
    # output
    output_dir: str = "main/checkpoints/diffusion_v2"


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


def move_dict(x_dict: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in x_dict.items()}


def derive_angle_b(pred_a: torch.Tensor, pred_c: torch.Tensor) -> torch.Tensor:
    return 180.0 - pred_a - pred_c


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Components
# ═══════════════════════════════════════════════════════════════════════════════


class ResConv1dBlock(nn.Module):
    """Conv1d residual block: GroupNorm → GELU → Conv → GroupNorm → GELU → Dropout → Conv."""

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        g = min(groups, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentionPool1d(nn.Module):
    """Learned attention-weighted temporal pooling."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_t = x.permute(0, 2, 1)                     # (B, T, C)
        w = self.attn(x_t).squeeze(-1)                # (B, T)
        w = torch.softmax(w, dim=-1).unsqueeze(1)     # (B, 1, T)
        pooled = (x * w).sum(dim=-1)                  # (B, C)
        return pooled


class FeatureCNNEncoder(nn.Module):
    """Per-feature CNN branches → attention pool → fusion MLP."""

    def __init__(
        self,
        feature_shapes: Dict[str, Tuple[int, ...]],
        branch_channels: int = 64,
        embed_dim: int = 128,
        out_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_order = sorted(feature_shapes.keys())
        self.branches = nn.ModuleDict()

        for key in self.feature_order:
            shape = feature_shapes[key]
            in_ch = 1 if len(shape) == 1 else int(shape[0])
            ch = branch_channels

            self.branches[key] = nn.ModuleDict(
                {
                    "stem": nn.Sequential(
                        nn.Conv1d(in_ch, ch, kernel_size=7, padding=3),
                        nn.GroupNorm(min(8, ch), ch),
                        nn.GELU(),
                    ),
                    "res1": ResConv1dBlock(ch, kernel_size=5, dropout=dropout),
                    "res2": ResConv1dBlock(ch, kernel_size=3, dropout=dropout),
                    "pool": AttentionPool1d(ch),
                    "proj": nn.Sequential(
                        nn.Linear(ch, embed_dim),
                        nn.GELU(),
                    ),
                }
            )

        fusion_in = embed_dim * len(self.feature_order)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        for key in self.feature_order:
            x = x_dict[key]
            if x.ndim == 2:
                x = x.unsqueeze(1)
            branch = self.branches[key]
            h = branch["stem"](x)
            h = branch["res1"](h)
            h = branch["res2"](h)
            h = branch["pool"](h)
            h = branch["proj"](h)
            encoded.append(h)
        fused = torch.cat(encoded, dim=1)
        return self.fusion(fused)


# ── Denoiser with Adaptive Layer Norm ────────────────────────────────────────


class AdaLN(nn.Module):
    """Adaptive LayerNorm: shift and scale predicted from conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ResMLPBlock(nn.Module):
    """Residual MLP with AdaLN conditioning."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.adaln = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.adaln(x, cond))


class ConditionalDenoiser(nn.Module):
    """
    Noise predictor conditioned on audio features and timestep.
    Uses AdaLN residual blocks for better conditioning.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        t_embed_dim: int = 128,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_embed_dim = t_embed_dim

        # Timestep → hidden
        self.t_proj = nn.Sequential(
            nn.Linear(t_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Audio condition → hidden
        self.x_proj = nn.Linear(x_dim, hidden_dim)
        # Noisy target → hidden
        self.y_proj = nn.Linear(y_dim, hidden_dim)

        # Residual blocks conditioned via AdaLN on (t + x_cond)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, y_dim)

    def forward(
        self,
        x_cond: torch.Tensor,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.t_embed_dim)
        cond = self.t_proj(t_emb) + self.x_proj(x_cond)  # fuse time + audio

        h = self.y_proj(y_noisy)
        for block in self.blocks:
            h = block(h, cond)

        return self.out_proj(self.out_norm(h))


# ═══════════════════════════════════════════════════════════════════════════════
#  Diffusion Regressor
# ═══════════════════════════════════════════════════════════════════════════════


class DiffusionRegressor(nn.Module):
    def __init__(
        self,
        feature_shapes: Dict[str, Tuple[int, ...]],
        y_dim: int,
        steps: int = 200,
        t_embed_dim: int = 128,
        hidden_dim: int = 512,
        feature_embed_dim: int = 128,
        branch_channels: int = 64,
        n_denoiser_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.steps = steps
        self.y_dim = y_dim

        self.encoder = FeatureCNNEncoder(
            feature_shapes=feature_shapes,
            branch_channels=branch_channels,
            embed_dim=feature_embed_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.denoiser = ConditionalDenoiser(
            x_dim=hidden_dim,
            y_dim=y_dim,
            t_embed_dim=t_embed_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_denoiser_blocks,
            dropout=dropout,
        )

        # Cosine noise schedule (clamp alpha_bars away from 0 for stable DDIM)
        betas = cosine_beta_schedule(steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).clamp(min=1e-4)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, y0, t, noise):
        a_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(a_bar) * y0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_noise(self, x_dict, y_noisy, t):
        x_cond = self.encoder(x_dict)
        return self.denoiser(x_cond, y_noisy, t)

    @torch.no_grad()
    def sample_ddim(self, x_dict, sample_steps: int) -> torch.Tensor:
        first_t = next(iter(x_dict.values()))
        device = first_t.device
        B = first_t.shape[0]
        y = torch.randn(B, self.y_dim, device=device)
        x_cond = self.encoder(x_dict)

        # Start from step T-2 to avoid near-zero alpha_bar at the very end
        max_t = min(self.steps - 1, self.steps - 2)
        ts = torch.linspace(max_t, 0, sample_steps, device=device).long()

        for i in range(len(ts)):
            t = ts[i]
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            eps = self.denoiser(x_cond, y, t_batch)

            a_bar = self.alpha_bars[t].clamp(min=1e-4)
            sqrt_ab = torch.sqrt(a_bar)
            sqrt_1mab = torch.sqrt((1.0 - a_bar).clamp(min=0))
            y0_hat = (y - sqrt_1mab * eps) / sqrt_ab
            # Clamp y0_hat to prevent extreme predictions
            y0_hat = y0_hat.clamp(-10, 10)

            if i == len(ts) - 1:
                y = y0_hat
            else:
                a_bar_prev = self.alpha_bars[ts[i + 1]].clamp(min=1e-4)
                y = torch.sqrt(a_bar_prev) * y0_hat + torch.sqrt((1.0 - a_bar_prev).clamp(min=0)) * eps

        return y


# ═══════════════════════════════════════════════════════════════════════════════
#  EMA
# ═══════════════════════════════════════════════════════════════════════════════


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup: Dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_shadow(self, model: nn.Module):
        """Load shadow weights into model (for evaluation). Backs up current weights."""
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original (non-EMA) weights after evaluation."""
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Mixup
# ═══════════════════════════════════════════════════════════════════════════════


def mixup_batch(x_dict, y, alpha: float = 0.3):
    """Apply mixup augmentation to a batch."""
    if alpha <= 0:
        return x_dict, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 to stay close to original

    B = y.shape[0]
    perm = torch.randperm(B, device=y.device)

    mixed_x = {k: lam * v + (1 - lam) * v[perm] for k, v in x_dict.items()}
    mixed_y = lam * y + (1 - lam) * y[perm]
    return mixed_x, mixed_y


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(
    model: DiffusionRegressor,
    loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    sample_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mae_ac_sum = 0.0
    mse_ac_sum = 0.0
    mae_b_sum = 0.0
    n_ac = 0
    n_b = 0

    for batch in loader:
        x = move_dict(batch["x"], device)
        y = batch["y"].to(device)  # raw angles from dataset

        pred_norm = model.sample_ddim(x, sample_steps=sample_steps)
        pred = pred_norm * y_std + y_mean  # un-normalize prediction

        # y is already raw angles; pred is now un-normalized → compare directly
        mae_ac_sum += torch.abs(pred - y).sum().item()
        mse_ac_sum += torch.square(pred - y).sum().item()
        n_ac += y.numel()

        pred_b = derive_angle_b(pred[:, 0], pred[:, 1])
        gt_b = derive_angle_b(y[:, 0], y[:, 1])
        mae_b_sum += torch.abs(pred_b - gt_b).sum().item()
        n_b += pred_b.numel()

    return {
        "mae_ac": mae_ac_sum / max(n_ac, 1),
        "rmse_ac": math.sqrt(mse_ac_sum / max(n_ac, 1)),
        "mae_b": mae_b_sum / max(n_b, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── data (precomputed features, no normalization yet — stats computed below) ──
    dataset = FullClipRollingDataset(
        clips_root=cfg.clips_root,
        window_sec=cfg.window_sec,
        hop_sec=cfg.hop_sec,
        augment=False,  # augmentation set per-subset below
    )

    # Clip-based train/val split
    clip_idx = dataset.get_clip_indices()
    clip_names = sorted(clip_idx.keys())

    # Remove holdout clips (used for fair held-out evaluation)
    if cfg.holdout_clips:
        holdout = set(c.strip() for c in cfg.holdout_clips.split(",") if c.strip())
        clip_names = [c for c in clip_names if c not in holdout]
        print(f"  Holdout clips removed from pool: {sorted(holdout)}")

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(clip_names)
    n_val_clips = max(1, int(len(clip_names) * cfg.val_ratio))
    val_clips = set(clip_names[:n_val_clips])
    train_clips = set(clip_names[n_val_clips:])

    train_indices = [i for cn in train_clips for i in clip_idx[cn]]
    val_indices = [i for cn in val_clips for i in clip_idx[cn]]

    print(f"\nClip split: {len(train_clips)} train clips, {len(val_clips)} val clips")
    print(f"  Train clips: {sorted(train_clips)}")
    print(f"  Val clips:   {sorted(val_clips)}")
    print(f"  Train windows: {len(train_indices)}, Val windows: {len(val_indices)}")

    # Compute feature stats from TRAINING data only
    train_features = [dataset._features[i] for i in train_indices]
    feature_stats = compute_feature_stats(train_features)
    dataset.feature_stats = feature_stats  # apply to full dataset

    # Create augmented wrapper for training subset
    # Training subset gets augmentation via dataset's augment flag
    # We'll toggle it in the training loop

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

    # ── feature shapes ──
    sample = dataset[0]["x"]
    feature_shapes = {k: tuple(v.shape) for k, v in sample.items()}
    y_dim = len(PREDICT_COLUMNS)

    print(f"\nFeature shapes:")
    for k, v in feature_shapes.items():
        print(f"  {k}: {v}")
    print(f"Target dim: {y_dim} ({', '.join(PREDICT_COLUMNS)})")

    # ── model ──
    model = DiffusionRegressor(
        feature_shapes=feature_shapes,
        y_dim=y_dim,
        steps=cfg.diffusion_steps,
        t_embed_dim=cfg.t_embed_dim,
        hidden_dim=cfg.hidden_dim,
        feature_embed_dim=cfg.feature_embed_dim,
        branch_channels=cfg.branch_channels,
        n_denoiser_blocks=cfg.n_denoiser_blocks,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    ema = EMA(model, decay=cfg.ema_decay)

    # ── target normalization ──
    print("Computing target normalization stats (from training data)...")
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
    history: List[Dict] = []

    print(f"\nStarting training for up to {cfg.epochs} epochs...\n")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        dataset.augment = True   # enable augmentation for training
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            x = move_dict(batch["x"], device)
            y = batch["y"].to(device)
            y_norm = (y - y_mean) / y_std

            # Mixup augmentation (at batch level)
            if cfg.mixup_alpha > 0 and np.random.random() < 0.5:
                x, y_norm = mixup_batch(x, y_norm, alpha=cfg.mixup_alpha)

            bs = y_norm.shape[0]
            t = torch.randint(0, cfg.diffusion_steps, (bs,), device=device)
            noise = torch.randn_like(y_norm)
            y_noisy = model.q_sample(y_norm, t=t, noise=noise)

            pred_noise = model.predict_noise(x, y_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            batches += 1

        scheduler.step()
        train_loss = running_loss / max(batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        # Evaluate with EMA weights
        dataset.augment = False
        ema.apply_shadow(model)
        metrics = evaluate(model, val_loader, y_mean, y_std, cfg.sample_steps, device)
        ema.restore(model)

        row = {"epoch": epoch, "train_loss": train_loss, "lr": lr_now, **metrics}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.5f} lr={lr_now:.2e} | "
            f"val_MAE(a,c)={metrics['mae_ac']:.4f}° "
            f"RMSE={metrics['rmse_ac']:.4f}° "
            f"MAE(b)={metrics['mae_b']:.4f}°"
        )

        # Save checkpoint
        ckpt = {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "y_mean": y_mean.detach().cpu(),
            "y_std": y_std.detach().cpu(),
            "feature_shapes": feature_shapes,
            "feature_stats": {
                k: (m.tolist(), s.tolist()) for k, (m, s) in feature_stats.items()
            },
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

    # ── final evaluation with EMA best ──
    print("\n" + "=" * 70)
    print("Final evaluation (EMA best checkpoint)")
    print("=" * 70)

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["ema_state"])

    final_m = evaluate(model, val_loader, y_mean, y_std, cfg.sample_steps, device)
    print(f"  val MAE  (angle_a, angle_c): {final_m['mae_ac']:.4f}°")
    print(f"  val RMSE (angle_a, angle_c): {final_m['rmse_ac']:.4f}°")
    print(f"  val MAE  (angle_b = 180-a-c): {final_m['mae_b']:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train diffusion regressor V2")
    for field_name, field_obj in TrainConfig.__dataclass_fields__.items():
        p.add_argument(f"--{field_name}", type=type(field_obj.default), default=field_obj.default)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
