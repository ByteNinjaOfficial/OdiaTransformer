#!/usr/bin/env python
"""Production training CLI entry point for OdiaTransformer.

Orchestrates Phase 3 DataLoaders, Phase 4 TranslationTransformer, and Phase 5 Trainer.
Supports full epoch training, step-limited smoke training, checkpoint resumption,
automatic mixed precision, and early stopping.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from dataset import create_dataloaders
from experiment import ExperimentManager, capture_environment_metadata
from tokenizer import PAD_ID
from transformer import TranslationTransformer
from training import Trainer, TrainingConfig, load_checkpoint


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the English -> Odia Translation Transformer from scratch."
    )
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name (routes outputs into experiments/<name>/)")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow reusing existing experiment directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of full training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="Warmup schedule learning rate scale (default: 1.0)")
    parser.add_argument("--warmup-steps", type=int, default=4000, help="Warmup steps for LR scheduler (default: 4000)")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience in epochs (0 to disable, default: 3)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--log-dir", type=str, default="outputs/training", help="Directory for saving logs and metrics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (for smoke tests / quick runs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--data-dir", type=str, default="outputs", help="Directory containing preprocessed data parquets")
    parser.add_argument("--en-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_en.model", help="Path to English tokenizer model")
    parser.add_argument("--or-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_or.model", help="Path to Odia tokenizer model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("==================================================")
    print("      OdiaTransformer Training Execution          ")
    print("==================================================")

    # 1. Experiment Management Integration (if specified)
    exp_mgr = None
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir

    if args.experiment:
        exp_mgr = ExperimentManager(
            experiment_name=args.experiment,
            allow_overwrite=args.allow_overwrite or bool(args.resume),
        )
        checkpoint_dir = str(exp_mgr.checkpoints_dir)
        log_dir = str(exp_mgr.metrics_dir)
        print(f"Experiment Name      : {exp_mgr.experiment_name}")
        print(f"Experiment Directory : {exp_mgr.root_dir}")

    # 2. Training Configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        seed=args.seed,
        use_amp=not args.no_amp,
        device=args.device,
        max_train_steps=args.max_steps,
    )
    device = config.resolved_device()

    print(f"Device               : {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"AMP Autocast         : {config.use_amp}")
    print(f"Batch Size           : {config.batch_size}")
    print(f"Target Epochs        : {config.num_epochs}")
    print(f"Max Steps (if set)   : {config.max_train_steps}")
    print(f"Warmup Steps         : {config.warmup_steps}")
    print(f"Early Stopping       : {config.early_stopping_patience} epochs")

    # 2. Instantiate Model
    model = TranslationTransformer(
        src_vocab_size=16000,
        tgt_vocab_size=32000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1,
        pad_id=PAD_ID,
    )
    total_params, trainable_params = model.count_parameters()
    print(f"Model Parameters     : {trainable_params:,} ({trainable_params / 1e6:.2f}M trainable)")

    # 3. Create Trainer
    trainer = Trainer(model=model, config=config)

    # 4. Handle Checkpoint Resumption
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"\nResuming training from checkpoint: {resume_path}")
        state = trainer.load_from_checkpoint(resume_path)
        print(f"Resumed at Epoch {trainer.current_epoch}, Global Step {trainer.global_step}, Best Val Loss {trainer.best_val_loss:.4f}")

    # 5. Load DataLoaders
    print("\nLoading dataset splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        en_tokenizer_path=args.en_tokenizer,
        or_tokenizer_path=args.or_tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(f"Training dataset     : {len(train_loader.dataset):,} pairs ({len(train_loader):,} batches)")
    print(f"Validation dataset   : {len(val_loader.dataset):,} pairs ({len(val_loader):,} batches)")

    # Save Experiment Config and Metadata if managed experiment
    if exp_mgr is not None:
        dataset_stats = {
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset) if test_loader else 0,
        }
        metadata = capture_environment_metadata(
            seed=args.seed,
            dataset_stats=dataset_stats,
            model_config={
                "src_vocab_size": 16000,
                "tgt_vocab_size": 32000,
                "d_model": 128,
                "num_heads": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "d_ff": 512,
                "dropout": 0.1,
                "pad_id": PAD_ID,
            },
            training_config=config,
        )
        exp_mgr.save_config(config)
        exp_mgr.save_metadata(metadata)

    # 6. Run Training Loop
    print("\nStarting training loop...")
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print("\n==================================================")
    print("           Training Run Summary                   ")
    print("==================================================")
    print(f"Total Epochs Completed : {summary['total_epochs']}")
    print(f"Total Steps Completed  : {summary['global_step']}")
    print(f"Best Validation Loss   : {summary['best_val_loss']:.4f}")
    if summary['best_val_loss'] < float('inf'):
        print(f"Best Validation PPL    : {math.exp(min(summary['best_val_loss'], 100.0)):.2f}")
    print(f"Checkpoints Saved to   : {config.checkpoint_dir}")
    print(f"Logs Saved to          : {config.log_dir}")
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
