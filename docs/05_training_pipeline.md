# Phase 5 — Training Pipeline, Optimization, Checkpointing & Validation

**Project:** Test-7 / OdiaTransformer — English → Odia Translation  
**Phase:** Phase 5 — Training Pipeline, Warmup Scheduling, Checkpointing & Hardware Validation  
**Status:** Complete & Verified.

---

## 1. Phase 5 Objective

Phase 5 implements the complete training and validation engine for the from-scratch English → Odia Transformer. It integrates teacher-forcing training, learning rate warmup scheduling, Cross-Entropy Loss with `<PAD>` exclusion, checkpoint persistence, Automatic Mixed Precision (AMP) for RTX 2050 4GB GPU safety, and validation telemetry.

---

## 2. Optimization & Learning Rate Schedule

### 2.1 Optimizer
- **Algorithm:** AdamW (`torch.optim.AdamW`)
- **Hyperparameters:**
  - $\beta_1 = 0.9$
  - $\beta_2 = 0.98$
  - $\epsilon = 10^{-9}$
  - $\text{weight\_decay} = 10^{-4}$

### 2.2 Transformer Warmup Scheduler
The learning rate schedule follows the formula from Vaswani et al. §5.3:
$$\text{lr}(step) = \text{learning\_rate} \times d_{\text{model}}^{-0.5} \times \min\left(step^{-0.5}, step \times \text{warmup\_steps}^{-1.5}\right)$$

- **Warmup Phase ($step \le 4000$):** Learning rate scales up linearly from $0.0$ to peak learning rate ($3.494 \times 10^{-4}$).
- **Decay Phase ($step > 4000$):** Learning rate decays proportionally to $step^{-0.5}$.
- **Checkpoint Resumption:** The step counter `current_step` is saved in `state_dict`, ensuring continuous learning rate progression upon recovery.

---

## 3. Loss Function & Sequence Alignment

- **Criterion:** `torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)`
- **Input:** Raw unnormalized logits reshaped to $[B \times L_{\text{tgt}}, 32000]$.
- **Target:** `batch.tgt_output` reshaped to $[B \times L_{\text{tgt}}]$.
- **Padding Invariant:** `<PAD>` tokens (ID = 0) are strictly excluded from loss computation and gradient backpropagation.
- **Perplexity Metric:** Computed as $\text{PPL} = \exp(\min(\text{Loss}, 100.0))$.

---

## 4. Hardware Safety & Telemetry (RTX 2050 4GB)

### 4.1 Memory Management
- **Default Batch Size:** `8`
- **Automatic Mixed Precision (AMP):** Enabled with `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.
- **Gradient Clipping:** Gradients are unscaled before `torch.nn.utils.clip_grad_norm_(max_norm=1.0)`.
- **Dynamic Device Resolution:** `TrainingConfig.device` defaults to `None`, dynamically resolving to CUDA if available or falling back to CPU.

### 4.2 Hardware Telemetry
Measured on **NVIDIA GeForce RTX 2050 (4GB VRAM)**:
- **Peak Allocated Memory:** ~307.86 MB
- **Peak Reserved Memory:** ~458.00 MB
- **Headroom:** $> 3.5\text{ GB}$ free VRAM available on the GPU.

---

## 5. Checkpointing Specification

Checkpoints are saved to `checkpoints/` and include complete training state:
```python
{
    "epoch": int,
    "global_step": int,
    "val_loss": float,
    "val_ppl": float,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict,
    "config": dict,
    "timestamp": float,
}
```
- **`latest_checkpoint.pt`:** Saved at the end of every epoch or validation check.
- **`best_checkpoint.pt`:** Updated whenever validation loss reaches a new global minimum.

---

## 6. Verification Results

- **Unit Tests:** `tests/test_training.py` (7/7 tests passing).
- **Full Test Suite:** 98/98 tests passing across all 5 phases (`tests/`).
- **Hardware Smoke Test:** Executed [`verify_training.py`](file:///d:/GitHub/FWC-TrainingProj/Test-7/verify_training.py) on real validation/train slices with full parameter gradient update verification, validation loss/perplexity calculation, checkpoint save/load roundtrip, and VRAM telemetry.

