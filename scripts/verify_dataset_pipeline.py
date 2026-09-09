#!/usr/bin/env python
"""Independent verification script for Phase 3 Dataset, DataLoader, Collation, and Mask pipeline.

Loads the real preprocessed splits and trained SentencePiece tokenizers, constructs
DataLoaders, inspects sample batches, and verifies tensor shapes, padding, special tokens,
teacher forcing alignment, and attention masks.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from dataset import create_dataloaders, TranslationBatch
from tokenizer import EOS_ID, PAD_ID, SOS_ID, UNK_ID


def main():
    print("=== OdiaTransformer Phase 3 Dataset & DataLoader Pipeline Verification ===")

    data_dir = Path("outputs")
    en_tok_path = data_dir / "tokenizer_sep_en16000_or32000_en.model"
    or_tok_path = data_dir / "tokenizer_sep_en16000_or32000_or.model"

    # Verify input files exist
    assert en_tok_path.exists(), f"Missing English tokenizer model: {en_tok_path}"
    assert or_tok_path.exists(), f"Missing Odia tokenizer model: {or_tok_path}"
    assert (data_dir / "train.parquet").exists(), "Missing outputs/train.parquet"
    assert (data_dir / "val.parquet").exists(), "Missing outputs/val.parquet"
    assert (data_dir / "test.parquet").exists(), "Missing outputs/test.parquet"
    print("1. All input data and tokenizer files located successfully.")

    # Create DataLoaders
    batch_size = 8
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        en_tokenizer_path=en_tok_path,
        or_tokenizer_path=or_tok_path,
        batch_size=batch_size,
        num_workers=0,
        add_eos_to_source=True,
    )

    print(f"2. DataLoaders created:")
    print(f"   - Train samples : {len(train_loader.dataset):,} (batches: {len(train_loader):,})")
    print(f"   - Val samples   : {len(val_loader.dataset):,} (batches: {len(val_loader):,})")
    print(f"   - Test samples  : {len(test_loader.dataset):,} (batches: {len(test_loader):,})")

    # Fetch one batch from validation loader for detailed inspection
    batch: TranslationBatch = next(iter(val_loader))

    print("\n3. Batch Tensor Shapes & Types:")
    print(f"   - src shape          : {list(batch.src.shape)} (dtype: {batch.src.dtype})")
    print(f"   - tgt_input shape    : {list(batch.tgt_input.shape)} (dtype: {batch.tgt_input.dtype})")
    print(f"   - tgt_output shape   : {list(batch.tgt_output.shape)} (dtype: {batch.tgt_output.dtype})")
    print(f"   - src_mask shape     : {list(batch.src_mask.shape)} (dtype: {batch.src_mask.dtype})")
    print(f"   - tgt_mask shape     : {list(batch.tgt_mask.shape)} (dtype: {batch.tgt_mask.dtype})")
    print(f"   - causal_mask shape  : {list(batch.causal_mask.shape)} (dtype: {batch.causal_mask.dtype})")

    # Verify invariants
    B, L_s = batch.src.shape
    _, L_t = batch.tgt_input.shape

    assert batch.tgt_output.shape == (B, L_t), "tgt_output shape must match tgt_input"
    assert batch.src_mask.shape == (B, 1, 1, L_s), "src_mask shape mismatch"
    assert batch.tgt_mask.shape == (B, 1, L_t, L_t), "tgt_mask shape mismatch"
    assert batch.causal_mask.shape == (1, 1, L_t, L_t), "causal_mask shape mismatch"
    print("\n4. Shape assertions passed.")

    # Verify Teacher Forcing & Shift Alignment
    print("\n5. Teacher-Forcing Alignment Verification:")
    for i in range(min(4, B)):
        tgt_len = batch.tgt_lengths[i]
        sos_token = batch.tgt_input[i, 0].item()
        eos_token = batch.tgt_output[i, tgt_len - 1].item()
        assert sos_token == SOS_ID, f"Sample {i} tgt_input does not start with SOS (got {sos_token})"
        assert eos_token == EOS_ID, f"Sample {i} tgt_output does not end with EOS (got {eos_token})"

        # Verify shift
        inp_shifted = batch.tgt_input[i, 1:tgt_len].tolist()
        out_shifted = batch.tgt_output[i, 0:tgt_len - 1].tolist()
        assert inp_shifted == out_shifted, f"Sample {i} shift mismatch: {inp_shifted} != {out_shifted}"

        print(f"   [Sample {i}]")
        print(f"     src_text : {batch.src_texts[i]}")
        print(f"     tgt_text : {batch.tgt_texts[i]}")
        print(f"     src_len={batch.src_lengths[i]}, tgt_len={tgt_len}")
        print(f"     tgt_input  : {batch.tgt_input[i, :min(tgt_len+2, L_t)].tolist()}")
        print(f"     tgt_output : {batch.tgt_output[i, :min(tgt_len+2, L_t)].tolist()}")

    print("\n6. Mask Behavior Verification:")
    # Causal mask checks
    causal_mat = batch.causal_mask[0, 0]
    for r in range(L_t):
        for c in range(L_t):
            if c <= r:
                assert causal_mat[r, c].item() is True, f"Causal mask failed at ({r}, {c}): expected True"
            else:
                assert causal_mat[r, c].item() is False, f"Causal mask failed at ({r}, {c}): expected False"
    print("   - Causal mask strictly blocks future tokens (j > i is False, j <= i is True).")

    # Source padding mask checks
    for i in range(B):
        s_len = batch.src_lengths[i]
        for pos in range(L_s):
            val = batch.src_mask[i, 0, 0, pos].item()
            if pos < s_len:
                assert val is True, f"src_mask sample {i} pos {pos} should be True"
            else:
                assert val is False, f"src_mask sample {i} pos {pos} should be False (PAD)"
    print("   - Source padding mask correctly identifies valid tokens and PAD tokens.")

    # Device transfer test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch.to(device)
    assert batch.src.device.type == device
    assert batch.tgt_mask.device.type == device
    print(f"   - TranslationBatch.to('{device}') device transfer verified.")

    print("\n=== ALL PHASE 3 VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
