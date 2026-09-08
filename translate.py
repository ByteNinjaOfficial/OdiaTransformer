#!/usr/bin/env python
"""Command-line translation tool for English -> Odia translation.

Loads trained model checkpoint and tokenizers, accepts custom English text, and
outputs greedy autoregressive Odia translation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Configure UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from inference import translate
from tokenizer import PAD_ID, load_tokenizer
from transformer import TranslationTransformer
from training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate English text to Odia using trained Transformer checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="English sentence to translate")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generated tokens (default: 100)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto)")
    parser.add_argument("--en-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_en.model", help="English tokenizer path")
    parser.add_argument("--or-tokenizer", type=str, default="outputs/tokenizer_sep_en16000_or32000_or.model", help="Odia tokenizer path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Resolve Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Check Paths
    ckpt_path = Path(args.checkpoint)
    en_tok_path = Path(args.en_tokenizer)
    or_tok_path = Path(args.or_tokenizer)

    if not ckpt_path.exists():
        # Check alternative alias
        alt_path = ckpt_path.parent / "best_checkpoint.pt"
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            sys.exit(1)

    if not en_tok_path.exists() or not or_tok_path.exists():
        print(f"ERROR: Tokenizer model not found in {en_tok_path.parent}")
        sys.exit(1)

    # 3. Load Tokenizers
    tok_en = load_tokenizer(en_tok_path)
    tok_or = load_tokenizer(or_tok_path)

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

    # 5. Translate Text
    translated_text = translate(
        model=model,
        text=args.text,
        src_tokenizer=tok_en,
        tgt_tokenizer=tok_or,
        device=device,
        max_length=args.max_length,
    )

    # 6. Display Translation
    print("\nEnglish:")
    print(args.text)
    print("\nOdia:")
    print(translated_text)


if __name__ == "__main__":
    main()
