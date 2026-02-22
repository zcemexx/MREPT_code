#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "checkpoint_final.pth",
        path / "checkpoint_latest.pth",
        path / "checkpoint_best.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No checkpoint found under {path}. Expected one of: "
        "checkpoint_final.pth, checkpoint_latest.pth, checkpoint_best.pth"
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    if window > len(values):
        window = len(values)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _to_1d_float(values: List) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _save_multi_format(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def _get_series(log: Dict, key: str) -> np.ndarray:
    if key not in log or log[key] is None:
        return np.asarray([], dtype=np.float64)
    return _to_1d_float(log[key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export paper-style nnUNet training curves from checkpoint logging."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to fold directory or checkpoint_*.pth file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: checkpoint directory).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window for loss curves (default: 1 = no smoothing).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="nnUNet Training Curves",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300).",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.input)
    output_dir = args.output_dir if args.output_dir is not None else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "logging" not in checkpoint:
        raise KeyError(f"'logging' not found in checkpoint: {checkpoint_path}")
    log = checkpoint["logging"]

    train_losses = _get_series(log, "train_losses")
    val_losses = _get_series(log, "val_losses")
    mean_fg_dice = _get_series(log, "mean_fg_dice")
    ema_fg_dice = _get_series(log, "ema_fg_dice")

    n_epochs = max(len(train_losses), len(val_losses))
    if n_epochs == 0:
        raise ValueError("No train/val loss values found in checkpoint logging.")

    x_loss = np.arange(n_epochs)
    train_losses = train_losses[:n_epochs]
    val_losses = val_losses[:n_epochs]
    train_smooth = _moving_average(train_losses, args.smooth_window)
    val_smooth = _moving_average(val_losses, args.smooth_window)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig_loss, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_loss, train_smooth, color="#1f77b4", linewidth=2.0, label="Train loss")
    ax.plot(x_loss, val_smooth, color="#d62728", linewidth=2.0, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(args.title)
    ax.legend(frameon=True)
    _save_multi_format(fig_loss, output_dir / "paper_loss_curve", dpi=args.dpi)
    plt.close(fig_loss)

    n_dice = min(len(mean_fg_dice), len(ema_fg_dice))
    if n_dice > 0:
        x_dice = np.arange(n_dice)
        fig_dice, ax2 = plt.subplots(figsize=(7.2, 4.4))
        ax2.plot(x_dice, mean_fg_dice[:n_dice], color="#2ca02c", linewidth=1.8, label="Mean FG Dice")
        ax2.plot(x_dice, ema_fg_dice[:n_dice], color="#17becf", linewidth=2.0, label="EMA FG Dice")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title("Validation Dice")
        ax2.legend(frameon=True)
        _save_multi_format(fig_dice, output_dir / "paper_dice_curve", dpi=args.dpi)
        plt.close(fig_dice)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print("Saved: paper_loss_curve.(png|pdf)")
    if n_dice > 0:
        print("Saved: paper_dice_curve.(png|pdf)")
    else:
        print("No dice series found. Skipped dice figure.")


if __name__ == "__main__":
    main()
