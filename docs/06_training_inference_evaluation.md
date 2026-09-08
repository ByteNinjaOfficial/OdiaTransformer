# Phase 6 Documentation: Training Execution, Checkpoint Resumption, Greedy Inference & Evaluation

This document details the design, architecture, implementation, hardware profiling, and verification of **Phase 6** in the English → Odia Neural Machine Translation system.

---

## 1. System Architecture & Component Design

Phase 6 implements the complete operational layer on top of Phases 1–5:
1. **Production Training CLI (`train.py`)**: End-to-end training pipeline supporting mixed precision (`torch.cuda.amp`), early stopping, learning rate warmup, and dynamic step limiting.
2. **Stateful Checkpointing & Resumption (`src/training.py`)**: Dual-checkpoint persistence (`latest.pt`, `best.pt`) preserving model parameters, optimizer state, scheduler state, epoch/step counters, best validation loss, and configuration metadata.
3. **Autoregressive Greedy Decoder (`src/inference.py`, `translate.py`)**: Optimized generation executing the Transformer encoder exactly once per sequence and decoding step-by-step until `<EOS>` or `max_length`.
4. **Corpus Evaluation Engine (`evaluate.py`)**: Model evaluation on test/validation sets computing loss, perplexity, and corpus BLEU via `sacrebleu`, with qualitative translation inspection.
5. **Visualization Utility (`plot_training.py`)**: Generates high-resolution training curves for loss, perplexity, and learning rate.

```
+-----------------------------------------------------------------------------------+
|                                 PHASE 6 PIPELINE                                  |
|                                                                                   |
|  [Parquet Data] -> [TranslationDataset & Dynamic Collation (Phase 3)]             |
|                                  |                                                |
|                                  v                                                |
|                [TranslationTransformer (§5.6 hand-written Phase 4)]               |
|                                  |                                                |
|         +------------------------+------------------------+                       |
|         |                                                 |                       |
|         v                                                 v                       |
|  [Trainer (Phase 5/6)]                           [Greedy Decoder]                 |
|   - AdamW + Warmup Schedule                       - Single Encoder Pass           |
|   - AMP FP16 Autocast                             - Autoregressive Loop           |
|   - Label Smoothed Cross-Entropy                  - EOS Stopping / Max Length     |
|   - Early Stopping (Patience=3)                           |                       |
|   - Checkpoints (latest.pt, best.pt)                      v                       |
|         |                                       [SacreBLEU Metric]                |
|         v                                       [Qualitative Samples]             |
|  [plot_training.py / Metrics]                                                     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Modules & Implementation Details

### 2.1 Training Pipeline (`train.py` & `src/training.py`)
- **Optimizer & Scheduler**: AdamW with $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$, coupled with the Vaswani et al. Noam warmup schedule:
  $$\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$
- **Loss Function**: Custom `CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)` applied strictly to flattened predicted logits and target shift tokens (`tgt_y`).
- **Mixed Precision**: Automatic FP16 mixed precision via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` with FP16-safe attention masking ($\text{mask\_value} = -10000.0$).
- **Early Stopping**: Configurable patience counter tracking validation loss improvements; stops training when validation loss fails to decrease for $P$ consecutive epochs.
- **Checkpoint Resumption**:
  - `save_checkpoint`: Atomically persists model state, optimizer state, scheduler state, epoch, step, best val loss, and `config`.
  - `load_from_checkpoint`: Reconstitutes model weights, optimizer/scheduler internal state, and training metadata to resume cleanly.

### 2.2 Greedy Inference Engine (`src/inference.py` & `translate.py`)
- **Encoder Efficiency**: The encoder stack runs **once** to obtain `memory` tensor of shape `(1, S, d_model)`.
- **Autoregressive Decoding Loop**:
  1. Initialize target sequence with `[<SOS>]` (`id=1`).
  2. In step $t$, construct causal subsequent mask and pass current target tokens `(1, t)` + `memory` through the decoder.
  3. Obtain logits for step $t$, pick token $\arg\max(\text{logits}_{t, :})$, and append to target sequence.
  4. Terminate immediately if `<EOS>` (`id=2`) is emitted or when $t \ge \text{max\_length}$.
- **Detokenization**: Detokenizes target IDs with Odia SentencePiece tokenizer (`src_vocab=16000`, `tgt_vocab=32000`).

### 2.3 Evaluation Pipeline (`evaluate.py`)
- **Loss & Perplexity**: Evaluates cross-entropy loss over validation/test splits ignoring padding, computing perplexity as $\exp(\text{loss})$.
- **Corpus BLEU**: Decodes candidate translations across the split and computes standard SacreBLEU score against references.
- **Sample Inspection**: Prints comparative tables of Source English, Target Reference Odia, and Model Hypothesis Odia.

### 2.4 Plotting (`plot_training.py`)
- Reads `training_metrics.json` output by the trainer.
- Uses matplotlib (`Agg` headless backend) to generate 3-panel publication-ready curves:
  1. Training & Validation Loss vs Epoch.
  2. Training & Validation Perplexity vs Epoch.
  3. Learning Rate vs Step.

---

## 3. Hardware Profiling & Resource Safety (RTX 2050 4GB)

The implementation was validated on the target NVIDIA GeForce RTX 2050 Laptop GPU (4096 MB VRAM):

| Parameter | Configuration |
| :--- | :--- |
| **Model Size** | 2 Encoder layers, 2 Decoder layers, $d_{\text{model}}=128$, $d_{\text{ff}}=512$, $h=4$ |
| **Total Parameters** | 8,981,856 parameters (~34.26 MB fp32) |
| **Batch Size** | 8 (or 4/16 configurable) with dynamic length bucketing |
| **Precision** | FP16 AMP Autocast + FP32 Master Weights |
| **Allocated VRAM** | ~383 MB |
| **Reserved VRAM** | ~532 MB |
| **Available Headroom** | > 3.4 GB |

---

## 4. CLI Usage Guide

### 4.1 Training
```bash
# Smoke test (3 steps)
python train.py --max-steps 3 --batch-size 4

# Full production training run
python train.py \
    --train-path outputs/train.parquet \
    --val-path outputs/val.parquet \
    --src-tok tokenizer/bpe_en_16k.model \
    --tgt-tok tokenizer/bpe_or_32k.model \
    --epochs 10 \
    --batch-size 8 \
    --warmup-steps 4000 \
    --early-stopping-patience 3 \
    --checkpoint-dir checkpoints
```

### 4.2 Resuming Training
```bash
python train.py \
    --resume checkpoints/latest.pt \
    --epochs 15
```

### 4.3 Translating Text
```bash
python translate.py \
    --checkpoint checkpoints/best.pt \
    --text "Welcome to our state." \
    --max-length 64
```

### 4.4 Evaluating on Test Set
```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --data-path outputs/test.parquet \
    --max-samples 100
```

### 4.5 Plotting Metrics
```bash
python plot_training.py \
    --metrics checkpoints/training_metrics.json \
    --output-dir checkpoints/
```

---

## 5. Verification & Test Suite

All unit tests and end-to-end integration tests pass cleanly:
- `tests/test_inference.py`: Verified greedy decoding termination on `<EOS>`, length bounding, single encoder call invariance, and full translation wrapper.
- `tests/test_train_cli.py`: Verified command line argument parsing with defaults and custom overrides.
- `verify_phase6.py`: Verified end-to-end multi-step training, dual checkpoint serialization (`latest.pt`/`best.pt`), checkpoint re-loading, and greedy inference generation on validation samples.
- Complete test suite: 104+ unit tests across Phases 1–6 passing.

