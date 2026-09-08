"""Phase 6: Translation Inference & Greedy Autoregressive Decoding for OdiaTransformer.

Provides reusable inference interfaces for English -> Odia translation, executing
source encoding once and autoregressively decoding tokens one-by-one until <EOS> or max_length.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn

from dataset import create_causal_mask, create_padding_mask, create_target_mask
from tokenizer import EOS_ID, PAD_ID, SOS_ID, Tokenizer
from transformer import TranslationTransformer


def greedy_decode(
    model: TranslationTransformer,
    src_tokens: List[int],
    max_length: int = 100,
    device: Optional[Union[str, torch.device]] = None,
) -> List[int]:
    """Perform autoregressive greedy decoding for a single tokenized source sequence.

    The encoder is executed exactly once to produce memory representations, and the
    decoder step iterates autoregressively until <EOS> is predicted or max_length is reached.

    Args:
        model: TranslationTransformer instance.
        src_tokens: List of source token IDs (typically ending with EOS_ID).
        max_length: Maximum number of tokens to generate.
        device: Device on which model and tensors reside.

    Returns:
        List of generated target token IDs (excluding SOS_ID and EOS_ID).
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    # Prepare source tensor [1, L_src] and mask [1, 1, 1, L_src]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = create_padding_mask(src_tensor, pad_id=model.pad_id)

    with torch.no_grad():
        # Encode source sequence ONCE
        memory = model.encode(src=src_tensor, src_mask=src_mask)

        # Initialize target sequence with <SOS>
        generated = [SOS_ID]

        for _ in range(max_length):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_mask = create_target_mask(tgt_tensor, pad_id=model.pad_id)

            # Decode target sequence against cached encoder memory
            logits = model.decode(
                tgt_input=tgt_tensor,
                memory=memory,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
            )

            # Greedily select the token with maximum probability at the last position
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

            if next_token == EOS_ID:
                break

            generated.append(next_token)

    # Strip <SOS> (and any trailing <EOS>)
    output_tokens = [t for t in generated if t not in (SOS_ID, EOS_ID, PAD_ID)]
    return output_tokens


def translate(
    model: TranslationTransformer,
    text: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 100,
    add_eos_to_source: bool = True,
) -> str:
    """Translate an English text string to Odia text using greedy decoding.

    Args:
        model: Trained TranslationTransformer model.
        text: Input English text string.
        src_tokenizer: English SentencePiece tokenizer.
        tgt_tokenizer: Odia SentencePiece tokenizer.
        device: PyTorch device.
        max_length: Maximum generation length.
        add_eos_to_source: If True, appends EOS_ID to source tokens.

    Returns:
        Translated Odia text string.
    """
    if not text.strip():
        return ""

    # Tokenize English source
    src_tokens = src_tokenizer.encode(text)
    if add_eos_to_source:
        src_tokens = src_tokens + [EOS_ID]

    # Perform greedy decoding
    gen_token_ids = greedy_decode(
        model=model,
        src_tokens=src_tokens,
        max_length=max_length,
        device=device,
    )

    # Detokenize to Odia text
    return tgt_tokenizer.decode(gen_token_ids)


def batch_translate(
    model: TranslationTransformer,
    texts: List[str],
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 100,
) -> List[str]:
    """Translate a list of English text strings to Odia text."""
    return [
        translate(
            model=model,
            text=t,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_length=max_length,
        )
        for t in texts
    ]
