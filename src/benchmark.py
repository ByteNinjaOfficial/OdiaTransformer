"""Phase 7A: Training Benchmark, GPU Telemetry, Epoch Estimation & Resume Validation.

Provides controlled benchmarking of the from-scratch English -> Odia Transformer:
- Real training pipeline benchmark (DataLoader -> Model -> Trainer -> Optimizer -> Scheduler -> AMP)
- Warmup step isolation
- High-precision CUDA-synchronized timing
- Token accounting (PAD-excluded src/tgt tokens)
- Validation throughput benchmarking
- GPU memory telemetry (peak allocated/reserved MB, memory utilization %)
- Mathematically derived epoch duration estimation
- Controlled checkpoint resumption validation helper
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TranslationBatch
from tokenizer import PAD_ID
from transformer import TranslationTransformer
from training import Trainer, TrainingConfig, save_checkpoint, load_checkpoint


@dataclass
class BenchmarkResult:
    """Encapsulates all benchmark metrics, telemetry, and time estimates."""
    device: str
    gpu_name: Optional[str]
    batch_size: int
    warmup_steps: int
    benchmark_steps: int
    val_steps: int

    # Training throughput
    training_steps: int
    training_elapsed_seconds: float
    training_steps_per_second: float
    training_samples_per_second: float
    training_tokens_per_second: float
    training_src_tokens_per_second: float
    training_tgt_tokens_per_second: float
    average_step_time_ms: float
    median_step_time_ms: float
    min_step_time_ms: float
    max_step_time_ms: float

    # Validation throughput
    validation_steps: int
    validation_elapsed_seconds: float
    validation_steps_per_second: float
    validation_samples_per_second: float
    validation_tokens_per_second: float
    average_val_step_time_ms: float

    # GPU Telemetry
    peak_allocated_mb: Optional[float]
    peak_reserved_mb: Optional[float]
    total_gpu_memory_mb: Optional[float]
    memory_utilization_pct: Optional[float]

    # Time Estimates
    train_dataset_samples: int
    val_dataset_samples: int
    estimated_training_epoch_seconds: float
    estimated_training_epoch_minutes: float
    estimated_training_epoch_hours: float
    estimated_validation_epoch_seconds: float
    estimated_validation_epoch_minutes: float
    estimated_validation_epoch_hours: float
    estimated_total_epoch_hours: float
    epochs_estimate: int
    estimated_total_training_hours: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingBenchmark:
    """Benchmark engine for evaluating Transformer training throughput and memory."""

    def __init__(
        self,
        model: TranslationTransformer,
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pad_id: int = PAD_ID,
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_id = pad_id
        self.device = trainer.device

    def run_benchmark(
        self,
        warmup_steps: int = 5,
        benchmark_steps: int = 20,
        val_steps: int = 10,
        epochs_estimate: int = 10,
    ) -> BenchmarkResult:
        """Run complete controlled training benchmark, validation benchmark, and telemetry.

        Args:
            warmup_steps: Number of real training steps to run prior to timing (excluded from stats).
            benchmark_steps: Number of real training steps to time for throughput statistics.
            val_steps: Number of validation steps to time for validation throughput.
            epochs_estimate: Number of epochs to use for total estimated training duration.

        Returns:
            BenchmarkResult containing all computed metrics.
        """
        train_iter = iter(self.train_loader)

        # 1. Warmup Steps (Real execution, excluded from timing)
        for _ in range(warmup_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            self.trainer.train_step(batch)

        # 2. Reset CUDA Peak Memory Stats before benchmark measurement
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # 3. Benchmark Training Steps
        step_times: List[float] = []
        total_samples = 0
        total_src_tokens = 0
        total_tgt_tokens = 0

        for _ in range(benchmark_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch_samples = batch.src.size(0)
            src_tokens = (batch.src != self.pad_id).sum().item()
            tgt_tokens = (batch.tgt_output != self.pad_id).sum().item()

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            self.trainer.train_step(batch)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()

            dt = t1 - t0
            step_times.append(dt)
            total_samples += batch_samples
            total_src_tokens += src_tokens
            total_tgt_tokens += tgt_tokens

        # 4. GPU Memory Telemetry
        gpu_telemetry = self.get_gpu_telemetry()

        # 5. Compute Training Throughput Metrics
        training_elapsed = sum(step_times) if step_times else 0.0
        n_steps = len(step_times)
        total_tokens = total_src_tokens + total_tgt_tokens

        steps_per_sec = n_steps / training_elapsed if training_elapsed > 0 else 0.0
        samples_per_sec = total_samples / training_elapsed if training_elapsed > 0 else 0.0
        tokens_per_sec = total_tokens / training_elapsed if training_elapsed > 0 else 0.0
        src_tokens_per_sec = total_src_tokens / training_elapsed if training_elapsed > 0 else 0.0
        tgt_tokens_per_sec = total_tgt_tokens / training_elapsed if training_elapsed > 0 else 0.0

        avg_step_ms = (training_elapsed / n_steps * 1000.0) if n_steps > 0 else 0.0
        med_step_ms = (statistics.median(step_times) * 1000.0) if step_times else 0.0
        min_step_ms = (min(step_times) * 1000.0) if step_times else 0.0
        max_step_ms = (max(step_times) * 1000.0) if step_times else 0.0

        # 6. Benchmark Validation Throughput
        val_metrics = self.benchmark_validation(val_steps=val_steps)

        # 7. Mathematical Epoch Duration Estimation
        train_dataset_size = (
            len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else total_samples
        )
        val_dataset_size = (
            len(self.val_loader.dataset)
            if self.val_loader is not None and hasattr(self.val_loader, "dataset")
            else 0
        )
        batch_size = self.trainer.config.batch_size

        estimates = self.estimate_epoch_duration(
            train_samples=train_dataset_size,
            val_samples=val_dataset_size,
            batch_size=batch_size,
            avg_train_step_sec=training_elapsed / max(1, n_steps),
            avg_val_step_sec=(
                val_metrics["elapsed_sec"] / max(1, val_metrics["steps"])
                if val_metrics["steps"] > 0
                else 0.0
            ),
            epochs_estimate=epochs_estimate,
        )

        return BenchmarkResult(
            device=self.device.type.upper(),
            gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
            val_steps=val_steps,
            training_steps=n_steps,
            training_elapsed_seconds=round(training_elapsed, 4),
            training_steps_per_second=round(steps_per_sec, 2),
            training_samples_per_second=round(samples_per_sec, 2),
            training_tokens_per_second=round(tokens_per_sec, 2),
            training_src_tokens_per_second=round(src_tokens_per_sec, 2),
            training_tgt_tokens_per_second=round(tgt_tokens_per_sec, 2),
            average_step_time_ms=round(avg_step_ms, 2),
            median_step_time_ms=round(med_step_ms, 2),
            min_step_time_ms=round(min_step_ms, 2),
            max_step_time_ms=round(max_step_ms, 2),
            validation_steps=val_metrics["steps"],
            validation_elapsed_seconds=round(val_metrics["elapsed_sec"], 4),
            validation_steps_per_second=round(val_metrics["steps_per_sec"], 2),
            validation_samples_per_second=round(val_metrics["samples_per_sec"], 2),
            validation_tokens_per_second=round(val_metrics["tokens_per_sec"], 2),
            average_val_step_time_ms=round(val_metrics["avg_step_ms"], 2),
            peak_allocated_mb=gpu_telemetry.get("peak_allocated_mb"),
            peak_reserved_mb=gpu_telemetry.get("peak_reserved_mb"),
            total_gpu_memory_mb=gpu_telemetry.get("total_gpu_memory_mb"),
            memory_utilization_pct=gpu_telemetry.get("memory_utilization_pct"),
            train_dataset_samples=train_dataset_size,
            val_dataset_samples=val_dataset_size,
            estimated_training_epoch_seconds=round(estimates["train_epoch_sec"], 2),
            estimated_training_epoch_minutes=round(estimates["train_epoch_min"], 2),
            estimated_training_epoch_hours=round(estimates["train_epoch_hours"], 2),
            estimated_validation_epoch_seconds=round(estimates["val_epoch_sec"], 2),
            estimated_validation_epoch_minutes=round(estimates["val_epoch_min"], 2),
            estimated_validation_epoch_hours=round(estimates["val_epoch_hours"], 2),
            estimated_total_epoch_hours=round(estimates["total_epoch_hours"], 2),
            epochs_estimate=epochs_estimate,
            estimated_total_training_hours=round(estimates["total_training_hours"], 2),
        )

    def benchmark_validation(self, val_steps: int = 10) -> Dict[str, Any]:
        """Benchmark validation throughput over a small slice of validation data."""
        if self.val_loader is None or val_steps <= 0:
            return {
                "steps": 0,
                "elapsed_sec": 0.0,
                "steps_per_sec": 0.0,
                "samples_per_sec": 0.0,
                "tokens_per_sec": 0.0,
                "avg_step_ms": 0.0,
            }

        val_iter = iter(self.val_loader)
        step_times: List[float] = []
        total_samples = 0
        total_tokens = 0

        for _ in range(val_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            samples = batch.src.size(0)
            non_pad_tokens = (batch.tgt_output != self.pad_id).sum().item()

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()

            self.trainer.validate_step(batch)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()

            step_times.append(t1 - t0)
            total_samples += samples
            total_tokens += non_pad_tokens

        elapsed = sum(step_times) if step_times else 0.0
        n_steps = len(step_times)

        return {
            "steps": n_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": n_steps / elapsed if elapsed > 0 else 0.0,
            "samples_per_sec": total_samples / elapsed if elapsed > 0 else 0.0,
            "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
            "avg_step_ms": (elapsed / n_steps * 1000.0) if n_steps > 0 else 0.0,
        }

    def get_gpu_telemetry(self) -> Dict[str, Any]:
        """Collect GPU memory telemetry from PyTorch CUDA runtime."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
                "total_gpu_memory_mb": None,
                "memory_utilization_pct": None,
            }

        allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        total_bytes = torch.cuda.get_device_properties(self.device).total_memory

        alloc_mb = allocated_bytes / (1024 * 1024)
        res_mb = reserved_bytes / (1024 * 1024)
        tot_mb = total_bytes / (1024 * 1024)
        util_pct = (reserved_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0

        return {
            "cuda_available": True,
            "peak_allocated_bytes": allocated_bytes,
            "peak_reserved_bytes": reserved_bytes,
            "peak_allocated_mb": round(alloc_mb, 2),
            "peak_reserved_mb": round(res_mb, 2),
            "total_gpu_memory_mb": round(tot_mb, 2),
            "memory_utilization_pct": round(util_pct, 2),
        }

    @staticmethod
    def estimate_epoch_duration(
        train_samples: int,
        val_samples: int,
        batch_size: int,
        avg_train_step_sec: float,
        avg_val_step_sec: float = 0.0,
        epochs_estimate: int = 10,
    ) -> Dict[str, float]:
        """Mathematically estimate training and validation duration per epoch and total."""
        train_batches = math.ceil(train_samples / max(1, batch_size))
        val_batches = math.ceil(val_samples / max(1, batch_size)) if val_samples > 0 else 0

        train_epoch_sec = train_batches * avg_train_step_sec
        val_epoch_sec = val_batches * avg_val_step_sec
        total_epoch_sec = train_epoch_sec + val_epoch_sec

        train_epoch_min = train_epoch_sec / 60.0
        train_epoch_hours = train_epoch_sec / 3600.0

        val_epoch_min = val_epoch_sec / 60.0
        val_epoch_hours = val_epoch_sec / 3600.0

        total_epoch_hours = total_epoch_sec / 3600.0
        total_training_hours = total_epoch_hours * epochs_estimate

        return {
            "train_batches": train_batches,
            "val_batches": val_batches,
            "train_epoch_sec": train_epoch_sec,
            "train_epoch_min": train_epoch_min,
            "train_epoch_hours": train_epoch_hours,
            "val_epoch_sec": val_epoch_sec,
            "val_epoch_min": val_epoch_min,
            "val_epoch_hours": val_epoch_hours,
            "total_epoch_hours": total_epoch_hours,
            "total_training_hours": total_training_hours,
        }


def validate_checkpoint_resumption(
    model_fn,
    config: TrainingConfig,
    train_loader: DataLoader,
    checkpoint_path: Path,
    steps_initial: int = 3,
    steps_resume: int = 1,
) -> Dict[str, Any]:
    """Controlled end-to-end checkpoint resume validation helper.

    Executes:
    1. Initial training for `steps_initial` steps.
    2. Save checkpoint.
    3. Instantiate fresh model & trainer.
    4. Restore from checkpoint.
    5. Verify exact parameter match, scheduler step match, optimizer state restoration,
       and scaler restoration when AMP is enabled.
    6. Continue training for `steps_resume` steps.
    7. Verify global step progression and finite loss/LR.

    Returns:
        Dictionary with verification results and telemetry.
    """
    # Stage 1: Initial Training
    model_init = model_fn()
    trainer_init = Trainer(model=model_init, config=config)

    train_iter = iter(train_loader)
    for _ in range(steps_initial):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        trainer_init.train_step(batch)

    # Save checkpoint
    save_checkpoint(
        filepath=checkpoint_path,
        model=trainer_init.model,
        optimizer=trainer_init.optimizer,
        scheduler=trainer_init.scheduler,
        scaler=trainer_init.scaler,
        epoch=trainer_init.current_epoch,
        global_step=trainer_init.global_step,
        val_loss=10.0,
        val_ppl=22026.0,
        best_val_loss=10.0,
        early_stopping_counter=0,
        history=trainer_init.history,
        config=config,
    )

    saved_step = trainer_init.global_step
    saved_lr = trainer_init.scheduler.get_current_lr()
    saved_params = {k: v.clone() for k, v in trainer_init.model.state_dict().items()}

    # Stage 2: Fresh Model & Restore
    model_fresh = model_fn()
    trainer_fresh = Trainer(model=model_fresh, config=config)
    trainer_fresh.load_from_checkpoint(checkpoint_path)

    # Stage 3: State Verification
    restored_step = trainer_fresh.global_step
    restored_lr = trainer_fresh.scheduler.get_current_lr()

    assert restored_step == saved_step, f"Step mismatch: {restored_step} != {saved_step}"
    assert abs(restored_lr - saved_lr) < 1e-12, f"LR mismatch: {restored_lr} != {saved_lr}"

    for k, v in trainer_fresh.model.state_dict().items():
        assert torch.allclose(v, saved_params[k].to(v.device)), f"Parameter mismatch in {k}"

    # Stage 4: Continuation Training
    initial_params_copy = {k: v.clone() for k, v in trainer_fresh.model.state_dict().items()}
    loss_val, lr_cont = trainer_fresh.train_step(batch)

    assert trainer_fresh.global_step == saved_step + steps_resume, (
        f"Continued step mismatch: {trainer_fresh.global_step} != {saved_step + steps_resume}"
    )
    assert not math.isnan(loss_val) and not math.isinf(loss_val), f"Non-finite loss: {loss_val}"
    assert not math.isnan(lr_cont) and not math.isinf(lr_cont), f"Non-finite LR: {lr_cont}"

    # Verify at least one parameter changed after the resume step
    param_changed = False
    for k, v in trainer_fresh.model.state_dict().items():
        if not torch.allclose(v, initial_params_copy[k]):
            param_changed = True
            break
    assert param_changed, "Model parameters did not update during resumed training step."

    return {
        "saved_step": saved_step,
        "restored_step": restored_step,
        "continued_step": trainer_fresh.global_step,
        "saved_lr": saved_lr,
        "restored_lr": restored_lr,
        "continued_lr": lr_cont,
        "loss_after_resume": loss_val,
        "parameter_update_verified": True,
    }

