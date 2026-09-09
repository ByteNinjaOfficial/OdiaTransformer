#!/usr/bin/env python
"""Phase 7A: Training Readiness, Experiment Management & Benchmarking CLI.

Executes controlled benchmarks of the from-scratch English -> Odia Transformer:
- Initializes isolated experiment directory
- Records environment, hardware, git, and configuration metadata
- Profiles training throughput with warmup isolation and GPU memory telemetry
- Estimates single-epoch and multi-epoch training durations mathematically
- Persists all results into experiments/<experiment_name>/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from benchmark import BenchmarkResult, TrainingBenchmark
from dataset import create_dataloaders
from experiment import ExperimentManager, capture_environment_metadata
from tokenizer import PAD_ID
from training import Trainer, TrainingConfig
from transformer import TranslationTransformer


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7A: Benchmark training throughput, GPU memory, and estimate epoch duration."
    )
    parser.add_argument("--experiment", type=str, default="phase7a_benchmark", help="Experiment name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps (default: 5)")
    parser.add_argument("--benchmark-steps", type=int, default=20, help="Benchmark steps (default: 20)")
    parser.add_argument("--val-steps", type=int, default=10, help="Validation steps to benchmark (default: 10)")
    parser.add_argument("--epochs-estimate", type=int, default=10, help="Number of epochs to estimate duration for (default: 10)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow reusing existing experiment directory")
    parser.add_argument("--data-dir", type=str, default="outputs", help="Directory containing preprocessed data parquets")
    parser.add_argument("--en-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_en.model", help="Path to English tokenizer model")
    parser.add_argument("--or-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_or.model", help="Path to Odia tokenizer model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("==================================================")
    print("   OdiaTransformer Phase 7A Training Benchmark    ")
    print("==================================================")

    # 1. Initialize Experiment Manager
    exp_mgr = ExperimentManager(
        experiment_name=args.experiment,
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Experiment Name      : {exp_mgr.experiment_name}")
    print(f"Experiment Directory : {exp_mgr.root_dir}")

    # 2. Training Configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs_estimate,
        learning_rate=1.0,
        warmup_steps=4000,
        early_stopping_patience=3,
        num_workers=0,
        checkpoint_dir=str(exp_mgr.checkpoints_dir),
        log_dir=str(exp_mgr.metrics_dir),
        seed=args.seed,
        use_amp=not args.no_amp,
        device=args.device,
    )
    device = config.resolved_device()

    print(f"Target Device        : {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"AMP Autocast         : {config.use_amp}")
    print(f"Batch Size           : {config.batch_size}")
    print(f"Warmup Steps         : {args.warmup_steps} (excluded from stats)")
    print(f"Benchmark Steps      : {args.benchmark_steps}")
    print(f"Validation Steps     : {args.val_steps}")

    # 3. Model Architecture
    model_config = {
        "src_vocab_size": 16000,
        "tgt_vocab_size": 32000,
        "d_model": 128,
        "num_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "pad_id": PAD_ID,
    }
    model = TranslationTransformer(**model_config)
    total_params, trainable_params = model.count_parameters()
    print(f"Model Parameters     : {trainable_params:,} ({trainable_params / 1e6:.2f}M trainable)")

    # 4. Load DataLoaders
    print("\nLoading dataset splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        en_tokenizer_path=args.en_tokenizer,
        or_tokenizer_path=args.or_tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    dataset_stats = {
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset) if test_loader else 0,
        "train_batches": len(train_loader),
        "validation_batches": len(val_loader),
    }
    print(f"Training dataset     : {dataset_stats['train_samples']:,} pairs ({dataset_stats['train_batches']:,} batches)")
    print(f"Validation dataset   : {dataset_stats['validation_samples']:,} pairs ({dataset_stats['validation_batches']:,} batches)")

    # 5. Capture & Persist Metadata and Config
    metadata = capture_environment_metadata(
        seed=args.seed,
        dataset_stats=dataset_stats,
        model_config=model_config,
        training_config=config,
    )
    exp_mgr.save_metadata(metadata)
    exp_mgr.save_config(config)

    # 6. Execute Benchmark
    print("\nExecuting warmup and benchmark steps...")
    trainer = Trainer(model=model, config=config)
    benchmark_engine = TrainingBenchmark(
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_id=PAD_ID,
    )

    result = benchmark_engine.run_benchmark(
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        val_steps=args.val_steps,
        epochs_estimate=args.epochs_estimate,
    )

    # 7. Persist Benchmark Results
    exp_mgr.save_benchmark(result.to_dict())

    # 8. Output Formatted Summary Report
    print("\n==================================================")
    print("            Benchmark Summary Report              ")
    print("==================================================")
    print(f"Device                     : {result.device}")
    if result.gpu_name:
        print(f"GPU                        : {result.gpu_name}")
    print(f"Batch Size                 : {result.batch_size}")
    print(f"Warmup Steps               : {result.warmup_steps}")
    print(f"Benchmark Steps            : {result.benchmark_steps}")
    print(f"Average Step Time          : {result.average_step_time_ms:.2f} ms")
    print(f"Median Step Time           : {result.median_step_time_ms:.2f} ms")
    print(f"Step Time Range (Min/Max)  : {result.min_step_time_ms:.2f} ms / {result.max_step_time_ms:.2f} ms")
    print(f"Training Throughput        : {result.training_steps_per_second:.2f} steps/sec")
    print(f"Training Sample Rate       : {result.training_samples_per_second:.2f} samples/sec")
    print(f"Training Token Rate (All)  : {result.training_tokens_per_second:.2f} tokens/sec")
    print(f"Training Token Rate (Src)  : {result.training_src_tokens_per_second:.2f} tokens/sec")
    print(f"Training Token Rate (Tgt)  : {result.training_tgt_tokens_per_second:.2f} tokens/sec")

    if result.validation_steps > 0:
        print(f"\nValidation Steps           : {result.validation_steps}")
        print(f"Validation Sample Rate     : {result.validation_samples_per_second:.2f} samples/sec")
        print(f"Validation Token Rate      : {result.validation_tokens_per_second:.2f} tokens/sec")
        print(f"Validation Avg Step Time   : {result.average_val_step_time_ms:.2f} ms")

    if result.peak_allocated_mb is not None:
        print(f"\nPeak Allocated VRAM        : {result.peak_allocated_mb:.2f} MB")
        print(f"Peak Reserved VRAM         : {result.peak_reserved_mb:.2f} MB")
        print(f"Total GPU VRAM             : {result.total_gpu_memory_mb:.2f} MB")
        print(f"VRAM Utilization           : {result.memory_utilization_pct:.2f}%")

    print("\n--- Mathematically Derived Epoch Duration Estimates ---")
    print(f"Estimated 1 Train Epoch    : {result.estimated_training_epoch_minutes:.2f} min ({result.estimated_training_epoch_hours:.2f} hours)")
    print(f"Estimated 1 Val Epoch      : {result.estimated_validation_epoch_minutes:.2f} min ({result.estimated_validation_epoch_hours:.2f} hours)")
    print(f"Estimated 1 Full Epoch     : {result.estimated_total_epoch_hours:.2f} hours")
    print(f"Estimated {result.epochs_estimate} Epochs Total     : {result.estimated_total_training_hours:.2f} hours")

    print("\nSaved Artifacts:")
    print(f"  - Config   : {exp_mgr.config_path}")
    print(f"  - Metadata : {exp_mgr.metadata_path}")
    print(f"  - Benchmark: {exp_mgr.benchmark_path}")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()

