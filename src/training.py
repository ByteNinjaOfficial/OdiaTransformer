"""Phase 5: Training Pipeline, Optimization, Checkpointing & Validation for OdiaTransformer.

Consumes Phase 3 DataLoaders and Phase 4 TranslationTransformer, implements teacher-forcing
training, custom Transformer warmup learning rate scheduling, Automatic Mixed Precision (AMP),
gradient clipping, CrossEntropyLoss ignoring padding, checkpoint persistence, and validation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from dataset import TranslationBatch
from tokenizer import PAD_ID
from transformer import TranslationTransformer


# ---------------------------------------------------------------------------
# 1. Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for model training, optimization, checkpointing, and validation."""
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1.0       # Scale factor for Transformer warmup schedule
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "outputs/training"
    seed: int = 42
    use_amp: bool = True
    device: Optional[str] = None     # Resolved dynamically at runtime (cuda if available, else cpu)
    val_check_interval: Optional[int] = None  # None = validate at end of each epoch
    max_train_steps: Optional[int] = None
    early_stopping_patience: int = 3  # 0 to disable early stopping

    def resolved_device(self) -> torch.device:
        """Resolve device dynamically to prevent device-binding issues when loading configs."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        return cls(**data)


# ---------------------------------------------------------------------------
# 2. Transformer Warmup Learning Rate Scheduler
# ---------------------------------------------------------------------------

class TransformerWarmupScheduler:
    """Transformer learning rate scheduler as defined in Vaswani et al. §5.3.

    lr(step) = learning_rate * (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    Maintains step count persistently in state_dict for checkpoint resumption.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 128,
        warmup_steps: int = 4000,
        learning_rate: float = 1.0,
        last_step: int = 0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.current_step = last_step
        self._step_internal()

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute the effective learning rate for a given step."""
        s = max(1, step if step is not None else self.current_step)
        arg1 = s ** -0.5
        arg2 = s * (self.warmup_steps ** -1.5)
        return self.learning_rate * (self.d_model ** -0.5) * min(arg1, arg2)

    def _step_internal(self) -> None:
        """Update optimizer param_groups with the current computed learning rate."""
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> float:
        """Advance the step counter by 1 and update optimizer learning rate."""
        self.current_step += 1
        self._step_internal()
        return self.get_current_lr()

    def get_current_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_step = state_dict.get("current_step", 0)
        self.d_model = state_dict.get("d_model", self.d_model)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.learning_rate = state_dict.get("learning_rate", self.learning_rate)
        self._step_internal()


# ---------------------------------------------------------------------------
# 3. Checkpointing Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[TransformerWarmupScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    val_loss: float = float("inf"),
    val_ppl: float = float("inf"),
    best_val_loss: float = float("inf"),
    early_stopping_counter: int = 0,
    history: Optional[List[dict]] = None,
    config: Optional[TrainingConfig] = None,
) -> Path:
    """Save model and training state into a checkpoint file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "best_val_loss": best_val_loss,
        "early_stopping_counter": early_stopping_counter,
        "history": history or [],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict() if config is not None else None,
        "timestamp": time.time(),
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[TransformerWarmupScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: Optional[str | torch.device] = None,
) -> dict:
    """Load model and training state from a checkpoint file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    return state


# ---------------------------------------------------------------------------
# 4. Trainer Class
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates model training, validation, optimizer updates, AMP, and metrics tracking."""

    def __init__(
        self,
        model: TranslationTransformer,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[TransformerWarmupScheduler] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = self.config.resolved_device()

        self.model = model.to(self.device)

        self.optimizer = optimizer or AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = scheduler or TransformerWarmupScheduler(
            self.optimizer,
            d_model=self.model.d_model,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
        )

        # Loss function with PAD token exclusion and label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.model.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        # Automatic Mixed Precision (AMP) setup
        self.use_amp = (self.device.type == "cuda") and self.config.use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Tracking state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.history: List[Dict[str, Any]] = []

        # Directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """Restore model, optimizer, scheduler, scaler, and training counters from checkpoint."""
        state = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.current_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", state.get("val_loss", float("inf")))
        self.early_stopping_counter = state.get("early_stopping_counter", 0)
        self.history = state.get("history", [])
        return state

    def train_step(self, batch: TranslationBatch) -> Tuple[float, float]:
        """Perform a single training step on a batch.

        Returns:
            Tuple of (loss_value, learning_rate_used).
        """
        self.model.train()
        batch.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(
                src=batch.src,
                tgt_input=batch.tgt_input,
                src_mask=batch.src_mask,
                tgt_mask=batch.tgt_mask,
            )
            # Flatten logits [B * L_t, V_tgt] and targets [B * L_t]
            loss = self.criterion(
                logits.view(-1, self.model.tgt_vocab_size),
                batch.tgt_output.view(-1),
            )

        # Backpropagation with AMP scaling
        self.scaler.scale(loss).backward()

        # Unscale before gradient clipping
        self.scaler.unscale_(self.optimizer)
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        # Optimizer & Scaler step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Step learning rate scheduler
        lr = self.scheduler.step()
        self.global_step += 1

        return loss.item(), lr

    def validate(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Evaluate model on validation loader without gradients.

        Args:
            val_loader: DataLoader containing validation data.
            max_batches: Optional ceiling on batches to evaluate.

        Returns:
            Tuple of (average_val_loss, val_perplexity).
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches_processed = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break

                batch.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(
                        src=batch.src,
                        tgt_input=batch.tgt_input,
                        src_mask=batch.src_mask,
                        tgt_mask=batch.tgt_mask,
                    )
                    loss = self.criterion(
                        logits.view(-1, self.model.tgt_vocab_size),
                        batch.tgt_output.view(-1),
                    )

                # Count non-PAD target tokens
                non_pad_tokens = (batch.tgt_output != self.model.pad_id).sum().item()
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
                batches_processed += 1

        if total_tokens == 0 or batches_processed == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / total_tokens
        # Safe perplexity computation avoiding overflow
        ppl = math.exp(min(avg_loss, 100.0))
        return avg_loss, ppl

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full training and validation loop over configured epochs.

        Returns:
            Summary dictionary containing history and best metrics.
        """
        start_time = time.time()

        for epoch in range(self.current_epoch + 1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_batches = 0

            for i, batch in enumerate(train_loader):
                if max_train_batches is not None and i >= max_train_batches:
                    break
                if self.config.max_train_steps is not None and self.global_step >= self.config.max_train_steps:
                    break

                loss_val, lr_val = self.train_step(batch)

                non_pad = (batch.tgt_output != self.model.pad_id).sum().item()
                epoch_loss += loss_val * non_pad
                epoch_tokens += non_pad
                epoch_batches += 1

                # In-epoch validation check interval if configured
                if (
                    self.config.val_check_interval is not None
                    and self.global_step % self.config.val_check_interval == 0
                ):
                    val_loss, val_ppl = self.validate(val_loader, max_batches=max_val_batches)
                    self._handle_checkpoint(val_loss, val_ppl)

            train_loss = epoch_loss / max(1, epoch_tokens)
            train_ppl = math.exp(min(train_loss, 100.0))

            # Epoch validation
            val_loss, val_ppl = self.validate(val_loader, max_batches=max_val_batches)
            is_best = self._handle_checkpoint(val_loss, val_ppl)

            if is_best:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            epoch_record = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "learning_rate": self.scheduler.get_current_lr(),
                "is_best": is_best,
                "elapsed_sec": time.time() - start_time,
            }
            self.history.append(epoch_record)
            self._save_history()

            # Early stopping check
            if (
                self.config.early_stopping_patience > 0
                and self.early_stopping_counter >= self.config.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"validation loss did not improve for {self.config.early_stopping_patience} epochs."
                )
                break

            if self.config.max_train_steps is not None and self.global_step >= self.config.max_train_steps:
                break

        return {
            "total_epochs": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

    def _handle_checkpoint(self, val_loss: float, val_ppl: float) -> bool:
        """Save latest and best checkpoints based on validation loss."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # Save latest checkpoints (both latest.pt and latest_checkpoint.pt)
        for name in ("latest.pt", "latest_checkpoint.pt"):
            save_checkpoint(
                filepath=self.checkpoint_dir / name,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=self.current_epoch,
                global_step=self.global_step,
                val_loss=val_loss,
                val_ppl=val_ppl,
                best_val_loss=self.best_val_loss,
                early_stopping_counter=self.early_stopping_counter,
                history=self.history,
                config=self.config,
            )

        # Save best checkpoints (both best.pt and best_checkpoint.pt)
        if is_best:
            for name in ("best.pt", "best_checkpoint.pt"):
                save_checkpoint(
                    filepath=self.checkpoint_dir / name,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    best_val_loss=self.best_val_loss,
                    early_stopping_counter=self.early_stopping_counter,
                    history=self.history,
                    config=self.config,
                )

        return is_best

    def _save_history(self) -> None:
        """Save training history and summary metrics to JSON."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        summary_path = self.log_dir / "metrics_summary.json"
        summary = {
            "total_epochs": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": math.exp(min(self.best_val_loss, 100.0)),
            "last_record": self.history[-1] if self.history else None,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

