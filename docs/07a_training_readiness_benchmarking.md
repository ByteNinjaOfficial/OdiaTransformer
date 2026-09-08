# Phase 7A Documentation: Training Readiness, Experiment Management & Benchmarking

This document outlines the architecture, implementation, benchmarking methodology, telemetry metrics, and verification protocols for **Phase 7A** of the from-scratch English → Odia Neural Machine Translation system.

---

## 1. Objective and System Overview

Phase 7A prepares the repository for full-scale training on the **NVIDIA GeForce RTX 2050 (4GB VRAM)** laptop GPU without executing unmonitored full epochs. It introduces:

1. **Structured Experiment Management**: Standardized directory hierarchy, overwrite prevention, and configuration persistence.
2. **Reproducibility Metadata**: Comprehensive capture of Git commits, Python/PyTorch/CUDA runtime environments, hardware attributes, and dataset statistics.
3. **Training & Validation Benchmark Engine**: Isolated warmup execution, high-precision CUDA-synchronized throughput profiling, and non-pad token rate calculation.
4. **GPU Memory Telemetry**: Runtime monitoring of peak allocated VRAM, peak reserved VRAM, and memory utilization percentage.
5. **Mathematical Epoch Duration Estimation**: Derived projection of training and validation epoch durations.
6. **Controlled Checkpoint Resume Validation**: Verifiable state-restoration protocol confirming weights, optimizer momentum, scheduler steps, and AMP scaler continuity.

---

## 2. Experiment Management Architecture

All experiment runs are isolated under `experiments/<experiment_name>/` using `ExperimentManager` ([`src/experiment.py`](file:///d:/GitHub/FWC-TrainingProj/Test-7/src/experiment.py)):

```
experiments/
└── <experiment_name>/
    ├── config.json          # Resolved TrainingConfig
    ├── metadata.json        # Hardware, environment, and reproducibility metadata
    ├── benchmark.json       # Benchmark results, throughput, and time estimates
    ├── checkpoints/         # Model weights (latest.pt, best.pt)
    ├── metrics/             # Training history JSON and summary metrics
    └── logs/                # Stdout / training logs
```

### Overwrite Safety
- When initializing an existing experiment directory with `allow_overwrite=False` (default), `ExperimentManager` raises a `FileExistsError` to prevent accidental overwrites.
- Setting `allow_overwrite=True` permits reuse of the existing directory structure for resumption workflows.

---

## 3. Reproducibility Metadata Schema

The `capture_environment_metadata` function extracts execution context into `metadata.json`:

```json
{
  "timestamp": "2026-09-08T16:45:00.000000+00:00",
  "git": {
    "commit_hash": "27d34fe4...",
    "branch": "main"
  },
  "system": {
    "os": "nt",
    "platform": "Windows-11-...",
    "python_version": "3.12.10",
    "pytorch_version": "2.14.0+cu126"
  },
  "hardware": {
    "cuda_available": true,
    "cuda_version": "12.6",
    "device_count": 1,
    "devices": [
      {
        "index": 0,
        "name": "NVIDIA GeForce RTX 2050",
        "total_memory_mb": 4096.0
      }
    ]
  },
  "reproducibility": {
    "random_seed": 42
  },
  "dataset": {
    "train_samples": 975020,
    "validation_samples": 9949,
    "test_samples": 9949
  },
  "model_config": {
    "src_vocab_size": 16000,
    "tgt_vocab_size": 32000,
    "d_model": 128,
    "num_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "d_ff": 512,
    "dropout": 0.1,
    "pad_id": 0
  },
  "training_config": {
    "batch_size": 8,
    "learning_rate": 1.0,
    "warmup_steps": 4000,
    "use_amp": true
  }
}
```

*Note: If Git is unavailable, `commit_hash` and `branch` gracefully fall back to `"unknown"` without throwing errors.*

---

## 4. Benchmark Methodology & Timing Precision

### 4.1 Warmup Isolation
To eliminate CUDA kernel compilation and memory allocator warm-up transients from throughput statistics:
- **Warmup Phase** executes `warmup_steps` (default: 5) full training steps (forward, loss, backward, optimizer step, scheduler step).
- Timing starts strictly **after** warmup finishes.

### 4.2 High-Precision CUDA Synchronization
Asynchronous CUDA kernel launches can skew timing measurements. The benchmark executes:
```python
if device.type == "cuda":
    torch.cuda.synchronize(device)
t0 = time.perf_counter()

trainer.train_step(batch)

if device.type == "cuda":
    torch.cuda.synchronize(device)
t1 = time.perf_counter()
```

### 4.3 Token Throughput Accounting
Token throughput reflects actual model compute by excluding padding tokens:
$$\text{Tokens}_{\text{src}} = \sum (x_{\text{src}} \neq \text{PAD\_ID})$$
$$\text{Tokens}_{\text{tgt}} = \sum (x_{\text{tgt\_out}} \neq \text{PAD\_ID})$$
$$\text{Total Tokens} = \text{Tokens}_{\text{src}} + \text{Tokens}_{\text{tgt}}$$
$$\text{Token Rate} = \frac{\text{Total Tokens}}{\sum \Delta t_{\text{steps}}}$$

---

## 5. GPU Memory Telemetry

Before timing begins, peak statistics are reset via `torch.cuda.reset_peak_memory_stats(device)`.

Telemetry metrics recorded:
- **Peak Allocated VRAM**: `torch.cuda.max_memory_allocated(device) / (1024 * 1024)` MB
- **Peak Reserved VRAM**: `torch.cuda.max_memory_reserved(device) / (1024 * 1024)` MB
- **Total GPU VRAM**: `props.total_memory / (1024 * 1024)` MB
- **VRAM Utilization**: $\frac{\text{Peak Reserved VRAM}}{\text{Total GPU VRAM}} \times 100\%$

---

## 6. Epoch Duration Estimation Formulas

Given measured average training step time $\bar{t}_{\text{train}}$ and validation step time $\bar{t}_{\text{val}}$:

1. **Training Batches**:
   $$B_{\text{train}} = \left\lceil \frac{N_{\text{train}}}{\text{batch\_size}} \right\rceil$$
2. **Training Epoch Duration**:
   $$T_{\text{train\_epoch}} = B_{\text{train}} \times \bar{t}_{\text{train}} \quad (\text{seconds})$$
3. **Validation Batches**:
   $$B_{\text{val}} = \left\lceil \frac{N_{\text{val}}}{\text{batch\_size}} \right\rceil$$
4. **Validation Epoch Duration**:
   $$T_{\text{val\_epoch}} = B_{\text{val}} \times \bar{t}_{\text{val}} \quad (\text{seconds})$$
5. **Total Single Epoch Duration**:
   $$T_{\text{epoch\_hours}} = \frac{T_{\text{train\_epoch}} + T_{\text{val\_epoch}}}{3600}$$
6. **Total Estimated Multi-Epoch Duration**:
   $$T_{\text{total\_hours}} = T_{\text{epoch\_hours}} \times \text{Epochs}$$

---

## 7. Checkpoint Resumption Validation Protocol

A 5-stage verification protocol validates state restoration:
1. **Stage A**: Train an initial instance for $N$ steps and save `checkpoint.pt`.
2. **Stage B**: Extract parameter tensors, optimizer state, scheduler step count, and scaler state.
3. **Stage C**: Instantiate a fresh model and trainer; invoke `load_from_checkpoint(checkpoint.pt)`.
4. **Stage D**: Assert that restored parameters are identical (`torch.allclose`), scheduler step equals $N$, and optimizer state is populated.
5. **Stage E**: Execute step $N + 1$. Assert that `global_step` is $N+1$, learning rate advances according to the Noam formula, loss is finite, and parameters differ from restored state.

---

## 8. CLI Usage Guide

### 8.1 Training Readiness Benchmark CLI
```bash
# Standard benchmark on RTX 2050
python benchmark_training.py \
    --experiment phase7a_benchmark \
    --batch-size 8 \
    --warmup-steps 5 \
    --benchmark-steps 20 \
    --epochs-estimate 10
```

### 8.2 Experiment Training Integration
```bash
# Run training managed under experiments/baseline_v1/
python train.py \
    --experiment baseline_v1 \
    --batch-size 8 \
    --epochs 10 \
    --warmup-steps 4000
```

