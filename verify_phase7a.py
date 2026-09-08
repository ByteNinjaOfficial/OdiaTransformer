#!/usr/bin/env python
"""Phase 7A End-to-End Integration Verifier.

Tests:
1. Isolated experiment directory and metadata capture
2. Real data and model benchmark execution with warmup isolation
3. GPU memory telemetry (allocated, reserved, utilization)
4. Mathematically derived epoch duration estimation
5. JSON artifact persistence (config.json, metadata.json, benchmark.json)
6. Controlled checkpoint resumption, parameter restoration, and step continuation
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from benchmark import BenchmarkResult, TrainingBenchmark, validate_checkpoint_resumption
from dataset import create_dataloaders
from experiment import ExperimentManager, capture_environment_metadata
from tokenizer import PAD_ID
from training import Trainer, TrainingConfig
from transformer import TranslationTransformer


def main() -> None:
    print("=== OdiaTransformer Phase 7A End-to-End Verification ===\n")

    exp_name = "phase7a_verify_smoke"
    exp_mgr = ExperimentManager(experiment_name=exp_name, allow_overwrite=True)
    print(f"1. Experiment directory created: {exp_mgr.root_dir}")

    # Configuration
    config = TrainingConfig(
        batch_size=8,
        num_epochs=10,
        learning_rate=1.0,
        warmup_steps=4000,
        early_stopping_patience=3,
        num_workers=0,
        checkpoint_dir=str(exp_mgr.checkpoints_dir),
        log_dir=str(exp_mgr.metrics_dir),
        seed=42,
        use_amp=True,
    )
    device = config.resolved_device()
    print(f"2. Target device resolved: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Model definition
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
    print(f"3. TranslationTransformer created ({trainable_params:,} trainable parameters)")

    # Load DataLoaders
    print("4. Loading DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="outputs",
        en_tokenizer_path="outputs/tokenizer_sep_en16000_or32000_en.model",
        or_tokenizer_path="outputs/tokenizer_sep_en16000_or32000_or.model",
        batch_size=config.batch_size,
        num_workers=0,
    )
    dataset_stats = {
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset) if test_loader else 0,
    }
    print(f"   Train samples: {dataset_stats['train_samples']:,}, Val samples: {dataset_stats['validation_samples']:,}")

    # Metadata & Config Persistence
    metadata = capture_environment_metadata(
        seed=42,
        dataset_stats=dataset_stats,
        model_config=model_config,
        training_config=config,
    )
    exp_mgr.save_metadata(metadata)
    exp_mgr.save_config(config)
    print("5. Metadata and configuration persisted to JSON.")

    # Benchmark Execution
    print("\n6. Running controlled training & validation benchmark (2 warmup, 5 benchmark, 3 val steps)...")
    trainer = Trainer(model=model, config=config)
    benchmark_engine = TrainingBenchmark(
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_id=PAD_ID,
    )

    result = benchmark_engine.run_benchmark(
        warmup_steps=2,
        benchmark_steps=5,
        val_steps=3,
        epochs_estimate=10,
    )
    exp_mgr.save_benchmark(result.to_dict())

    # Assertions
    assert result.training_steps == 5, f"Expected 5 training steps, got {result.training_steps}"
    assert result.training_steps_per_second > 0, "Throughput must be positive"
    assert result.average_step_time_ms > 0, "Step time must be positive"
    assert result.estimated_training_epoch_hours > 0, "Epoch estimate must be positive"
    assert exp_mgr.benchmark_path.exists(), "benchmark.json must exist"

    print(f"   - Average Step Time      : {result.average_step_time_ms:.2f} ms")
    print(f"   - Training Throughput    : {result.training_steps_per_second:.2f} steps/sec ({result.training_samples_per_second:.2f} samples/sec)")
    print(f"   - Token Throughput       : {result.training_tokens_per_second:.2f} tokens/sec")
    if result.peak_allocated_mb is not None:
        print(f"   - Peak Allocated VRAM    : {result.peak_allocated_mb:.2f} MB")
        print(f"   - Peak Reserved VRAM     : {result.peak_reserved_mb:.2f} MB")
        print(f"   - VRAM Utilization       : {result.memory_utilization_pct:.2f}%")
    print(f"   - Estimated 1 Epoch      : {result.estimated_training_epoch_hours:.2f} hours (train) + {result.estimated_validation_epoch_hours:.2f} hours (val)")
    print(f"   - Estimated 10 Epochs    : {result.estimated_total_training_hours:.2f} hours")

    # Checkpoint Resume Validation
    print("\n7. Validating checkpoint save, restore, parameter integrity, and step continuation...")
    ckpt_path = exp_mgr.get_checkpoint_path("resume_verify.pt")

    def model_factory():
        return TranslationTransformer(**model_config)

    resume_stats = validate_checkpoint_resumption(
        model_fn=model_factory,
        config=config,
        train_loader=train_loader,
        checkpoint_path=ckpt_path,
        steps_initial=3,
        steps_resume=1,
    )
    print(f"   - Initial Trained Steps  : {resume_stats['saved_step']}")
    print(f"   - Restored Step          : {resume_stats['restored_step']}")
    print(f"   - Continued Step         : {resume_stats['continued_step']}")
    print(f"   - Parameters Updated     : {resume_stats['parameter_update_verified']}")
    print(f"   - Loss After Resume      : {resume_stats['loss_after_resume']:.4f}")

    print("\n=== ALL PHASE 7A VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()

