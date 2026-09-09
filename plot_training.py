#!/usr/bin/env python
"""Plot training and validation curves from persisted training history.

Reads outputs/training/training_history.json and generates visual PNG plots:
- training_loss.png (Train vs Validation Loss)
- validation_loss.png (Validation Loss & Perplexity)
- learning_rate.png (Warmup Learning Rate schedule)
- training_curves.png (Combined dashboard)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Use non-interactive Agg backend for headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OdiaTransformer training metrics.")
    parser.add_argument(
        "--history-path",
        "--metrics",
        dest="history_path",
        type=str,
        default="outputs/training/training_history.json",
        help="Path to training history JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Directory to save generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_file = Path(args.history_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_file.exists():
        print(f"ERROR: Training history file not found: {history_file}")
        sys.exit(1)

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print("ERROR: Training history is empty.")
        sys.exit(1)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_ppls = [h["val_ppl"] for h in history]
    lrs = [h["learning_rate"] for h in history]

    # 1. Train vs Validation Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", color="#ff7f0e", linewidth=2)
    plt.title("OdiaTransformer — Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = out_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_path}")

    # 2. Validation Loss & Perplexity Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, val_losses, color="#ff7f0e", marker="s", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Loss", color="#ff7f0e", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ppls, color="#2ca02c", marker="^", linestyle="--", linewidth=2, label="Val Perplexity")
    ax2.set_ylabel("Validation Perplexity", color="#2ca02c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    plt.title("OdiaTransformer — Validation Loss & Perplexity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    val_path = out_dir / "validation_loss.png"
    plt.savefig(val_path, dpi=300)
    plt.close()
    print(f"Saved: {val_path}")

    # 3. Dedicated Perplexity Plot
    train_ppls = [h.get("train_ppl", math.exp(min(h.get("train_loss", 0), 100.0))) for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_ppls, label="Train PPL", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_ppls, label="Val PPL", marker="^", color="#2ca02c", linewidth=2)
    plt.title("OdiaTransformer — Training & Validation Perplexity", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    ppl_path = out_dir / "perplexity.png"
    plt.savefig(ppl_path, dpi=300)
    plt.close()
    print(f"Saved: {ppl_path}")

    # 4. Learning Rate Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lrs, label="Learning Rate", marker="d", color="#d62728", linewidth=2)
    plt.title("OdiaTransformer — Learning Rate Schedule", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    lr_path = out_dir / "learning_rate.png"
    plt.savefig(lr_path, dpi=300)
    plt.close()
    print(f"Saved: {lr_path}")

    # 4. Combined Dashboard
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Loss
    axs[0].plot(epochs, train_losses, label="Train", marker="o", color="#1f77b4")
    axs[0].plot(epochs, val_losses, label="Val", marker="s", color="#ff7f0e")
    axs[0].set_title("Loss", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # Perplexity
    axs[1].plot(epochs, val_ppls, label="Val PPL", marker="^", color="#2ca02c")
    axs[1].set_title("Validation Perplexity", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PPL")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # LR
    axs[2].plot(epochs, lrs, label="Learning Rate", marker="d", color="#d62728")
    axs[2].set_title("Learning Rate", fontsize=12, fontweight="bold")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("LR")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend()

    plt.suptitle("OdiaTransformer Training Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    dashboard_path = out_dir / "training_curves.png"
    plt.savefig(dashboard_path, dpi=300)
    plt.close()
    print(f"Saved: {dashboard_path}")


if __name__ == "__main__":
    main()
