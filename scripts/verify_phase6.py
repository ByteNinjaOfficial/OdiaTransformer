#!/usr/bin/env python
"""Independent verification script for Phase 6 Training Execution, Checkpointing, Greedy Inference & Evaluation.

Executes a 5-batch training step, saves a checkpoint, reloads the checkpoint, and performs
greedy autoregressive translations on 3 real validation examples.
"""

from __future__ import annotations

import itertools
import sys
import tempfile
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
from dataset import create_dataloaders
from inference import translate
from tokenizer import PAD_ID, load_tokenizer
from transformer import TranslationTransformer
from training import Trainer, TrainingConfig, load_checkpoint, save_checkpoint


def main() -> None:
    print("=== OdiaTransformer Phase 6 End-to-End Verification ===")

    data_dir = Path("outputs")
    en_tok_path = data_dir / "tokenizer_sep_en16000_or32000_en.model"
    or_tok_path = data_dir / "tokenizer_sep_en16000_or32000_or.model"
    val_parquet_path = data_dir / "val.parquet"

    assert en_tok_path.exists(), f"Missing English tokenizer model: {en_tok_path}"
    assert or_tok_path.exists(), f"Missing Odia tokenizer model: {or_tok_path}"
    assert val_parquet_path.exists(), f"Missing outputs/val.parquet"
    print("1. All input data and tokenizer files located successfully.")

    # 1. Load Tokenizers
    tok_en = load_tokenizer(en_tok_path)
    tok_or = load_tokenizer(or_tok_path)

    # 2. Instantiate Phase 4 Model
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
    print("2. TranslationTransformer initialized from scratch (11.20M parameters).")

    # 3. Create Trainer and Train on 5 Batches
    config = TrainingConfig(
        batch_size=8,
        learning_rate=1.0,
        warmup_steps=4000,
        use_amp=True,
    )
    device = config.resolved_device()
    trainer = Trainer(model, config)
    print(f"3. Trainer initialized on device: {device.type.upper()}")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        en_tokenizer_path=en_tok_path,
        or_tokenizer_path=or_tok_path,
        batch_size=config.batch_size,
    )

    print("\n4. Running 5-step training smoke test:")
    train_slice = list(itertools.islice(train_loader, 5))
    for step_idx, batch in enumerate(train_slice, start=1):
        loss_val, lr_val = trainer.train_step(batch)
        print(f"   - Step {step_idx}/5 | Loss: {loss_val:.4f} | LR: {lr_val:.6e}")

    # 4. Checkpoint Save and Resumption Verification
    print("\n5. Verifying Checkpoint Resumption:")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "phase6_checkpoint.pt"
        save_checkpoint(
            filepath=ckpt_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=1,
            global_step=trainer.global_step,
            val_loss=10.38,
            val_ppl=32000.0,
            config=config,
        )
        assert ckpt_path.exists(), "Checkpoint file was not created"

        # Create fresh inference model and load state
        eval_model = TranslationTransformer(
            src_vocab_size=16000,
            tgt_vocab_size=32000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            pad_id=PAD_ID,
        )
        loaded_state = load_checkpoint(ckpt_path, eval_model, map_location=device)
        eval_model.to(device)
        assert loaded_state["global_step"] == 5
        print(f"   - Checkpoint saved and reloaded successfully ({ckpt_path.stat().st_size / 1024**2:.2f} MB).")

    # 5. Greedy Translation on 3 Real Validation Examples
    print("\n6. Running Greedy Translation Inference on 3 Real Validation Samples:")
    val_df = pd.read_parquet(val_parquet_path).head(3)

    for idx, row in val_df.iterrows():
        src_text = str(row["src"])
        ref_text = str(row["tgt"])

        pred_text = translate(
            model=eval_model,
            text=src_text,
            src_tokenizer=tok_en,
            tgt_tokenizer=tok_or,
            device=device,
            max_length=50,
        )

        assert isinstance(pred_text, str), "Translation output must be a string"
        print(f"\n--- [Sample {idx+1}] ---")
        print(f"English Source      : {src_text}")
        print(f"Odia Reference      : {ref_text}")
        print(f"Odia Translation    : {pred_text}")

    print("\n=== ALL PHASE 6 VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
