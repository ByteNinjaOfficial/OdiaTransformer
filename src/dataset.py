"""Phase 3 Dataset, DataLoader, Collation, and Mask Pipeline for OdiaTransformer.

Bridges the preprocessed Parquet datasets and trained SentencePiece tokenizers into
PyTorch tensors with teacher forcing alignment, dynamic per-batch padding, and
attention masks ready for the from-scratch Transformer.

Contracts & Invariants:
----------------------
1. Special Tokens:
   PAD_ID = 0 (<PAD>)
   SOS_ID = 1 (<SOS>)
   EOS_ID = 2 (<EOS>)
   UNK_ID = 3 (<UNK>)

2. Source Sequence:
   [token_1, token_2, ..., token_n, EOS_ID]
   (Appends EOS_ID, no SOS_ID; standard bidirectional encoder input).

3. Target Teacher-Forcing Sequences:
   Given raw target tokens [t_1, t_2, ..., t_m]:
   - tgt_input  = [SOS_ID, t_1, t_2, ..., t_m]         (length m + 1)
   - tgt_output = [t_1, t_2, ..., t_m, EOS_ID]         (length m + 1)
   Exact 1-step shift alignment; identical lengths.

4. Dynamic Per-Batch Padding:
   Sequences within each batch are padded with PAD_ID (0) up to the maximum
   length in that specific batch (no fixed global padding).

5. Mask Convention:
   - Boolean value `True`  = VALID / ATTENDABLE token.
   - Boolean value `False` = MASKED / IGNORED token (filled with -inf in attention).
   - Causal mask blocks positions where key index j > query index i (j <= i is True).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import (
    EOS_ID,
    PAD_ID,
    SOS_ID,
    UNK_ID,
    Tokenizer,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslationSample:
    """Individual translation sample containing tokenized sequences and raw strings."""
    src_ids: list[int]
    tgt_input_ids: list[int]
    tgt_output_ids: list[int]
    src_text: str = ""
    tgt_text: str = ""

    def __post_init__(self):
        if len(self.tgt_input_ids) != len(self.tgt_output_ids):
            raise ValueError(
                f"Target input and output lengths must match for teacher forcing: "
                f"got {len(self.tgt_input_ids)} vs {len(self.tgt_output_ids)}"
            )


@dataclass
class TranslationBatch:
    """Collated batch of translation pairs with dynamic padding and attention masks."""
    src: torch.Tensor           # [batch_size, src_seq_len], dtype=torch.long
    tgt_input: torch.Tensor     # [batch_size, tgt_seq_len], dtype=torch.long
    tgt_output: torch.Tensor    # [batch_size, tgt_seq_len], dtype=torch.long
    src_mask: torch.Tensor      # [batch_size, 1, 1, src_seq_len], dtype=torch.bool, True=valid
    tgt_mask: torch.Tensor      # [batch_size, 1, tgt_seq_len, tgt_seq_len], dtype=torch.bool, True=valid
    src_pad_mask: torch.Tensor  # [batch_size, 1, 1, src_seq_len], dtype=torch.bool, True=valid
    tgt_pad_mask: torch.Tensor  # [batch_size, 1, 1, tgt_seq_len], dtype=torch.bool, True=valid
    causal_mask: torch.Tensor   # [1, 1, tgt_seq_len, tgt_seq_len], dtype=torch.bool, True=valid
    src_lengths: list[int]
    tgt_lengths: list[int]
    src_texts: list[str] = field(default_factory=list)
    tgt_texts: list[str] = field(default_factory=list)

    def to(self, device: torch.device | str) -> TranslationBatch:
        """Move all tensor attributes to target device in-place and return self."""
        self.src = self.src.to(device)
        self.tgt_input = self.tgt_input.to(device)
        self.tgt_output = self.tgt_output.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.src_pad_mask = self.src_pad_mask.to(device)
        self.tgt_pad_mask = self.tgt_pad_mask.to(device)
        self.causal_mask = self.causal_mask.to(device)
        return self


# ---------------------------------------------------------------------------
# Mask Generation Helpers
# ---------------------------------------------------------------------------

def create_padding_mask(seq: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Generate a padding mask where True indicates a valid (non-PAD) token.

    Args:
        seq: Tensor of token IDs, shape [batch_size, seq_len].
        pad_id: Special token ID for padding (default 0).

    Returns:
        Boolean mask tensor of shape [batch_size, 1, 1, seq_len], where:
        - True  = Valid (attendable) token.
        - False = PAD (masked out) token.
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Generate a lower-triangular causal / look-ahead mask for autoregressive decoding.

    Args:
        seq_len: Length of the target sequence.
        device: PyTorch device on which to construct the tensor.

    Returns:
        Boolean mask tensor of shape [1, 1, seq_len, seq_len], where:
        - True  = Position (i, j) where j <= i (valid past/current position).
        - False = Position (i, j) where j > i (future position, masked out).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt_input: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Generate the combined target mask (padding mask AND causal mask).

    Args:
        tgt_input: Tensor of decoder input IDs, shape [batch_size, tgt_seq_len].
        pad_id: Special token ID for padding (default 0).

    Returns:
        Boolean mask tensor of shape [batch_size, 1, tgt_seq_len, tgt_seq_len], where:
        - True  = Position (i, j) is valid (j <= i AND token j is non-PAD).
        - False = Masked out (future token or PAD token).
    """
    pad_mask = create_padding_mask(tgt_input, pad_id=pad_id)  # [B, 1, 1, L_t]
    causal = create_causal_mask(tgt_input.size(1), device=tgt_input.device)  # [1, 1, L_t, L_t]
    return pad_mask & causal  # Broadcasts to [B, 1, L_t, L_t]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """PyTorch Dataset for English -> Odia translation pairs.

    Loads sentence pairs from a preprocessed Parquet file (or DataFrame) and uses
    separate SentencePiece tokenizers to encode source and target sequences.
    """

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        tokenizer_en: Tokenizer,
        tokenizer_or: Tokenizer,
        add_eos_to_source: bool = True,
    ):
        """Initialize the TranslationDataset.

        Args:
            data: Path to preprocessed Parquet file or an existing pandas DataFrame.
            tokenizer_en: Tokenizer instance for English source text.
            tokenizer_or: Tokenizer instance for Odia target text.
            add_eos_to_source: If True, appends EOS_ID to source token sequence (default True).
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            self.df = pd.read_parquet(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.reset_index(drop=True)
        else:
            raise TypeError(f"data must be str, Path, or pd.DataFrame, got {type(data)}")

        if "src" not in self.df.columns or "tgt" not in self.df.columns:
            raise ValueError(f"DataFrame must contain 'src' and 'tgt' columns; got {list(self.df.columns)}")

        self.tokenizer_en = tokenizer_en
        self.tokenizer_or = tokenizer_or
        self.add_eos_to_source = add_eos_to_source

        # Extract underlying Series as lists/arrays for fast indexing without DataFrame overhead
        self._src_texts = self.df["src"].astype(str).tolist()
        self._tgt_texts = self.df["tgt"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self._src_texts)

    def __getitem__(self, idx: int) -> TranslationSample:
        src_text = self._src_texts[idx]
        tgt_text = self._tgt_texts[idx]

        # Encode source (English)
        src_tokens = self.tokenizer_en.encode(src_text)
        src_ids = src_tokens + [EOS_ID] if self.add_eos_to_source else src_tokens

        # Encode target (Odia)
        tgt_tokens = self.tokenizer_or.encode(tgt_text)
        tgt_input_ids = [SOS_ID] + tgt_tokens
        tgt_output_ids = tgt_tokens + [EOS_ID]

        return TranslationSample(
            src_ids=src_ids,
            tgt_input_ids=tgt_input_ids,
            tgt_output_ids=tgt_output_ids,
            src_text=src_text,
            tgt_text=tgt_text,
        )


# ---------------------------------------------------------------------------
# Dynamic Batch Collation
# ---------------------------------------------------------------------------

def collate_translation_samples(
    samples: Sequence[TranslationSample],
    pad_id: int = PAD_ID,
) -> TranslationBatch:
    """Collate variable-length TranslationSamples into a padded TranslationBatch.

    Args:
        samples: Sequence of TranslationSample instances.
        pad_id: Special token ID for padding (default 0).

    Returns:
        TranslationBatch containing padded batch tensors and corresponding masks.
    """
    if not samples:
        raise ValueError("Cannot collate an empty list of samples")

    batch_size = len(samples)
    src_lengths = [len(s.src_ids) for s in samples]
    tgt_lengths = [len(s.tgt_input_ids) for s in samples]

    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    # Initialize padded tensors with PAD_ID (0)
    src_tensor = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_input_tensor = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_output_tensor = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)

    src_texts = []
    tgt_texts = []

    for i, sample in enumerate(samples):
        s_len = len(sample.src_ids)
        t_len = len(sample.tgt_input_ids)

        if s_len > 0:
            src_tensor[i, :s_len] = torch.tensor(sample.src_ids, dtype=torch.long)
        if t_len > 0:
            tgt_input_tensor[i, :t_len] = torch.tensor(sample.tgt_input_ids, dtype=torch.long)
            tgt_output_tensor[i, :t_len] = torch.tensor(sample.tgt_output_ids, dtype=torch.long)

        src_texts.append(sample.src_text)
        tgt_texts.append(sample.tgt_text)

    # Generate masks
    src_pad_mask = create_padding_mask(src_tensor, pad_id=pad_id)        # [B, 1, 1, L_s]
    tgt_pad_mask = create_padding_mask(tgt_input_tensor, pad_id=pad_id)  # [B, 1, 1, L_t]
    causal_mask = create_causal_mask(max_tgt_len)                        # [1, 1, L_t, L_t]
    tgt_mask = tgt_pad_mask & causal_mask                                # [B, 1, L_t, L_t]

    return TranslationBatch(
        src=src_tensor,
        tgt_input=tgt_input_tensor,
        tgt_output=tgt_output_tensor,
        src_mask=src_pad_mask,
        tgt_mask=tgt_mask,
        src_pad_mask=src_pad_mask,
        tgt_pad_mask=tgt_pad_mask,
        causal_mask=causal_mask,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        src_texts=src_texts,
        tgt_texts=tgt_texts,
    )


# ---------------------------------------------------------------------------
# DataLoader Factories
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader wrapping a TranslationDataset with dynamic collation.

    Args:
        dataset: TranslationDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data at every epoch.
        num_workers: Subprocesses to use for data loading (0 = main process).
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning.
        drop_last: If True, drops the last incomplete batch.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_translation_samples,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloaders(
    data_dir: str | Path = "outputs",
    en_tokenizer_path: str | Path = "outputs/tokenizer_sep_en16000_or32000_en.model",
    or_tokenizer_path: str | Path = "outputs/tokenizer_sep_en16000_or32000_or.model",
    batch_size: int = 32,
    num_workers: int = 0,
    add_eos_to_source: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for the English -> Odia translation pipeline.

    Args:
        data_dir: Directory containing preprocessed train.parquet, val.parquet, test.parquet.
        en_tokenizer_path: Path to trained SentencePiece English tokenizer model.
        or_tokenizer_path: Path to trained SentencePiece Odia tokenizer model.
        batch_size: Batch size for all DataLoaders.
        num_workers: DataLoader worker processes.
        add_eos_to_source: If True, appends EOS_ID to source sequences.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_path = Path(data_dir)
    tok_en = load_tokenizer(Path(en_tokenizer_path))
    tok_or = load_tokenizer(Path(or_tokenizer_path))

    train_ds = TranslationDataset(data_path / "train.parquet", tok_en, tok_or, add_eos_to_source=add_eos_to_source)
    val_ds = TranslationDataset(data_path / "val.parquet", tok_en, tok_or, add_eos_to_source=add_eos_to_source)
    test_ds = TranslationDataset(data_path / "test.parquet", tok_en, tok_or, add_eos_to_source=add_eos_to_source)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

