#!/usr/bin/env python
"""Independent verification script for Phase 4 from-scratch Transformer architecture.

Validates the complete TranslationTransformer model on CPU and CUDA (if available),
using real tokenizers and preprocessed Parquet validation data from Phase 3 DataLoaders.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn as nn
from dataset import TranslationBatch, create_dataloaders
from tokenizer import EOS_ID, PAD_ID, SOS_ID, UNK_ID
from transformer import TranslationTransformer


def main():
    print("=== OdiaTransformer Phase 4 Architecture Verification ===")

    # 1. Paths to real tokenizer models and processed data
    data_dir = Path("outputs")
    en_tok_path = data_dir / "tokenizer_sep_en16000_or32000_en.model"
    or_tok_path = data_dir / "tokenizer_sep_en16000_or32000_or.model"

    assert en_tok_path.exists(), f"Missing English tokenizer model: {en_tok_path}"
    assert or_tok_path.exists(), f"Missing Odia tokenizer model: {or_tok_path}"
    assert (data_dir / "val.parquet").exists(), "Missing outputs/val.parquet"
    print("1. All input data and tokenizer files located successfully.")

    # 2. Instantiate the Phase 4 TranslationTransformer model
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
    print("2. TranslationTransformer initialized from scratch:")
    print("   - d_model            : 128")
    print("   - num_heads          : 4 (head_dim = 32)")
    print("   - num_encoder_layers : 2")
    print("   - num_decoder_layers : 2")
    print("   - d_ff               : 512")
    print("   - src_vocab_size     : 16,000 (English)")
    print("   - tgt_vocab_size     : 32,000 (Odia)")

    # 3. Parameter count analysis
    total_params, trainable_params = model.count_parameters()
    breakdown = model.parameter_breakdown()
    print("\n3. Parameter Count Analysis:")
    print(f"   - Source Embedding   : {breakdown['source_embedding']:,}")
    print(f"   - Target Embedding   : {breakdown['target_embedding']:,}")
    print(f"   - Encoder (2 layers) : {breakdown['encoder_layers']:,}")
    print(f"   - Decoder (2 layers) : {breakdown['decoder_layers']:,}")
    print(f"   - Output Projection  : {breakdown['output_projection']:,}")
    print(f"   - Total Trainable    : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # 4. Load Phase 3 DataLoaders and retrieve real validation batch
    batch_size = 4
    _, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        en_tokenizer_path=en_tok_path,
        or_tokenizer_path=or_tok_path,
        batch_size=batch_size,
    )

    batch: TranslationBatch = next(iter(val_loader))
    print(f"\n4. Retrieved Real Validation Batch (B={batch_size}):")
    print(f"   - src shape          : {list(batch.src.shape)}")
    print(f"   - tgt_input shape    : {list(batch.tgt_input.shape)}")
    print(f"   - tgt_output shape   : {list(batch.tgt_output.shape)}")
    print(f"   - src_mask shape     : {list(batch.src_mask.shape)}")
    print(f"   - tgt_mask shape     : {list(batch.tgt_mask.shape)}")

    # 5. CPU Forward Pass Verification
    print("\n5. Running CPU Forward Pass:")
    model.eval()
    with torch.no_grad():
        logits_cpu = model(
            src=batch.src,
            tgt_input=batch.tgt_input,
            src_mask=batch.src_mask,
            tgt_mask=batch.tgt_mask,
        )

    B, L_t = batch.tgt_input.shape
    assert logits_cpu.shape == (B, L_t, 32000), f"Shape mismatch: {logits_cpu.shape}"
    assert not torch.isnan(logits_cpu).any(), "NaN found in CPU logits"
    assert not torch.isinf(logits_cpu).any(), "Inf found in CPU logits"
    print(f"   - CPU Output Logits Shape : {list(logits_cpu.shape)} [OK]")
    print(f"   - CPU Logits Finite       : min={logits_cpu.min():.4f}, max={logits_cpu.max():.4f}, mean={logits_cpu.mean():.4f} [OK]")

    # 6. Gradient & Backpropagation Check on CPU
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    logits = model(
        src=batch.src,
        tgt_input=batch.tgt_input,
        src_mask=batch.src_mask,
        tgt_mask=batch.tgt_mask,
    )
    loss = criterion(logits.view(-1, 32000), batch.tgt_output.view(-1))
    model.zero_grad()
    loss.backward()
    assert model.encoder.src_embed.weight.grad is not None
    assert model.decoder.output_projection.weight.grad is not None
    print(f"   - Backward Gradient Flow  : Loss={loss.item():.4f} [OK]")

    # 7. CUDA Verification (if available)
    if torch.cuda.is_available():
        print("\n6. Running CUDA Forward & Backward Pass:")
        device = torch.device("cuda")
        model_cuda = model.to(device)
        batch_cuda = batch.to(device)

        model_cuda.eval()
        with torch.no_grad():
            logits_cuda = model_cuda(
                src=batch_cuda.src,
                tgt_input=batch_cuda.tgt_input,
                src_mask=batch_cuda.src_mask,
                tgt_mask=batch_cuda.tgt_mask,
            )

        assert logits_cuda.shape == (B, L_t, 32000)
        assert not torch.isnan(logits_cuda).any(), "NaN in CUDA logits"
        assert not torch.isinf(logits_cuda).any(), "Inf in CUDA logits"
        print(f"   - CUDA Device             : {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Output Logits      : {list(logits_cuda.shape)} [OK]")

        model_cuda.train()
        logits_c = model_cuda(
            src=batch_cuda.src,
            tgt_input=batch_cuda.tgt_input,
            src_mask=batch_cuda.src_mask,
            tgt_mask=batch_cuda.tgt_mask,
        )
        loss_c = criterion(logits_c.view(-1, 32000), batch_cuda.tgt_output.view(-1))
        model_cuda.zero_grad()
        loss_c.backward()
        print(f"   - CUDA Gradient Flow      : Loss={loss_c.item():.4f} [OK]")
    else:
        print("\n6. CUDA not available; skipped GPU test.")

    print("\n=== ALL PHASE 4 ARCHITECTURE VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()

