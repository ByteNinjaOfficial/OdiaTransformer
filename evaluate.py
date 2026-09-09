#!/usr/bin/env python
"""Evaluation entry point for English -> Odia Translation Transformer.

Loads test Parquet split and model checkpoint, computes Loss, Perplexity, and Corpus BLEU
(via sacrebleu) over decoded translations, and displays sample prediction pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
from dataset import TranslationDataset, create_dataloader
from inference import translate
from tokenizer import PAD_ID, load_tokenizer
from transformer import TranslationTransformer
from training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate English -> Odia Translation Transformer on test split."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--max-samples", type=int, default=100, help="Max test samples to evaluate (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for loss evaluation (default: 8)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--data-dir", type=str, default="outputs", help="Directory containing test.parquet")
    parser.add_argument("--en-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_en.model", help="English tokenizer path")
    parser.add_argument("--or-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_or.model", help="Odia tokenizer path")
    parser.add_argument("--num-display-samples", type=int, default=5, help="Number of sample translations to display (default: 5)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save evaluation metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Resolve Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Verify File Paths
    ckpt_path = Path(args.checkpoint)
    test_parquet_path = Path(args.data_dir) / "test.parquet"
    en_tok_path = Path(args.en_tokenizer)
    or_tok_path = Path(args.or_tokenizer)

    if not ckpt_path.exists():
        alt_path = ckpt_path.parent / "best_checkpoint.pt"
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            sys.exit(1)

    if not test_parquet_path.exists():
        print(f"ERROR: Test parquet file not found: {test_parquet_path}")
        sys.exit(1)

    # 3. Load Tokenizers & Test Dataset
    tok_en = load_tokenizer(en_tok_path)
    tok_or = load_tokenizer(or_tok_path)

    test_df = pd.read_parquet(test_parquet_path)
    if args.max_samples and len(test_df) > args.max_samples:
        test_df = test_df.iloc[: args.max_samples].reset_index(drop=True)

    print(f"Evaluating {len(test_df)} samples on device: {device.type.upper()}")

    # 4. Instantiate and Load Model
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
    load_checkpoint(filepath=ckpt_path, model=model, map_location=device)
    model.to(device)
    model.eval()

    # 5. Compute Test Loss & Perplexity on Batch Tensors
    test_ds = TranslationDataset(test_df, tok_en, tok_or)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            logits = model(
                src=batch.src,
                tgt_input=batch.tgt_input,
                src_mask=batch.src_mask,
                tgt_mask=batch.tgt_mask,
            )
            loss = criterion(
                logits.reshape(-1, 32000),
                batch.tgt_output.reshape(-1),
            )
            non_pad = (batch.tgt_output != PAD_ID).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    test_loss = total_loss / max(1, total_tokens)
    test_ppl = math.exp(min(test_loss, 100.0))

    # 6. Run Greedy Decoding & Compute BLEU
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []

    print("\nRunning greedy translation generation...")
    for idx, row in test_df.iterrows():
        src_text = str(row["src"])
        ref_text = str(row["tgt"])

        pred_text = translate(
            model=model,
            text=src_text,
            src_tokenizer=tok_en,
            tgt_tokenizer=tok_or,
            device=device,
            max_length=100,
        )

        sources.append(src_text)
        references.append(ref_text)
        predictions.append(pred_text)

    # Compute Corpus BLEU score
    bleu_score = sacrebleu.corpus_bleu(predictions, [references])

    # 7. Print Evaluation Summary
    print("\n==================================================")
    print("            Test Evaluation Summary               ")
    print("==================================================")
    print(f"Samples Evaluated : {len(predictions):,}")
    print(f"Test Cross-Entropy: {test_loss:.4f}")
    print(f"Test Perplexity   : {test_ppl:.2f}")
    print(f"Corpus BLEU Score : {bleu_score.score:.2f}")
    print(f"BLEU Breakdown    : {bleu_score.format()}")

    # 8. Display Sample Translation Pairs
    num_display = min(args.num_display_samples, len(predictions))
    sample_list = []
    print(f"\nDisplaying {num_display} Sample Translations:")
    for i in range(num_display):
        print(f"\n--- [Sample {i+1}] ---")
        print(f"English (Source)    : {sources[i]}")
        print(f"Odia (Reference)    : {references[i]}")
        print(f"Odia (Predicted)    : {predictions[i]}")

    for i in range(len(predictions)):
        sample_list.append({
            "index": i + 1,
            "english_source": sources[i],
            "odia_reference": references[i],
            "odia_predicted": predictions[i],
        })

    # 9. Save JSON Evaluation Metrics if requested
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        eval_payload = {
            "checkpoint_used": str(ckpt_path),
            "samples_evaluated": len(predictions),
            "test_cross_entropy": round(test_loss, 4),
            "test_perplexity": round(test_ppl, 2),
            "corpus_bleu_score": round(bleu_score.score, 2),
            "bleu_breakdown": bleu_score.format(),
            "sample_translations": sample_list[:args.num_display_samples],
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation metrics to: {out_json_path}")


if __name__ == "__main__":
    main()
