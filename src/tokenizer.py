"""Phase 2B tokenizer module for OdiaTransformer.

Implements SentencePiece BPE tokenization with explicit special-token ID contract,
identity normalization, and deterministic train-only vocabulary learning.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Special-token contract (hard requirements from Phase 2A)
# ---------------------------------------------------------------------------

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_PIECE = "<PAD>"
SOS_PIECE = "<SOS>"
EOS_PIECE = "<EOS>"
UNK_PIECE = "<UNK>"

SPECIAL_PIECES = [PAD_PIECE, SOS_PIECE, EOS_PIECE, UNK_PIECE]
SPECIAL_IDS = {PAD_PIECE: PAD_ID, SOS_PIECE: SOS_ID, EOS_PIECE: EOS_ID, UNK_PIECE: UNK_ID}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenizerConfig:
    """Immutable configuration for a tokenizer experiment."""
    mode: Literal["separate", "shared"]
    vocab_size_en: int | None = None
    vocab_size_or: int | None = None
    vocab_size_shared: int | None = None
    seed: int = 42
    model_type: str = "bpe"
    normalization_rule_name: str = "identity"
    remove_extra_whitespaces: bool = False
    character_coverage: float = 1.0
    input_sentence_size: int = 0  # 0 = use all
    shuffle_input_sentence: bool = False

    def __post_init__(self):
        if self.mode == "separate":
            if self.vocab_size_en is None or self.vocab_size_or is None:
                raise ValueError("separate mode requires vocab_size_en and vocab_size_or")
            if self.vocab_size_shared is not None:
                raise ValueError("separate mode must not set vocab_size_shared")
        elif self.mode == "shared":
            if self.vocab_size_shared is None:
                raise ValueError("shared mode requires vocab_size_shared")
            if self.vocab_size_en is not None or self.vocab_size_or is not None:
                raise ValueError("shared mode must not set vocab_size_en/vocab_size_or")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def config_id(self) -> str:
        """Short identifier for this configuration, used in filenames."""
        if self.mode == "separate":
            return f"sep_en{self.vocab_size_en}_or{self.vocab_size_or}"
        return f"shared{self.vocab_size_shared}"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Corpus builders (train-only, explicit leakage prevention)
# ---------------------------------------------------------------------------

def load_train_corpus(path: Path) -> pd.DataFrame:
    """Load the preprocessed training split only."""
    return pd.read_parquet(path)


def write_corpus_file(texts: Iterable[str], out_path: Path) -> int:
    """Write an iterator of sentences to a text file, one per line.
    Returns the number of lines written.
    """
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            if text:
                f.write(text + "\n")
                count += 1
    return count


def build_separate_corpus_files(
    train_df: pd.DataFrame,
    en_path: Path,
    or_path: Path,
) -> tuple[int, int]:
    """Build separate English and Odia corpus files from train split only.
    Returns (en_count, or_count).
    """
    en_count = write_corpus_file(train_df["src"].astype(str), en_path)
    or_count = write_corpus_file(train_df["tgt"].astype(str), or_path)
    return en_count, or_count


def build_shared_corpus_file(
    train_df: pd.DataFrame,
    out_path: Path,
) -> int:
    """Build a single interleaved corpus file from train split only.
    Returns total line count.
    """
    all_texts = []
    all_texts.extend(train_df["src"].astype(str))
    all_texts.extend(train_df["tgt"].astype(str))
    return write_corpus_file(all_texts, out_path)


# ---------------------------------------------------------------------------
# SentencePiece training
# ---------------------------------------------------------------------------

def _sp_train_args(
    config: TokenizerConfig,
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
) -> dict:
    """Build SentencePieceTrainer arguments respecting the special-token contract."""
    return {
        "input": str(input_path),
        "model_prefix": str(model_prefix),
        "vocab_size": vocab_size,
        "model_type": config.model_type,
        "normalization_rule_name": config.normalization_rule_name,
        "remove_extra_whitespaces": str(config.remove_extra_whitespaces).lower(),
        "character_coverage": str(config.character_coverage),
        "input_sentence_size": str(config.input_sentence_size),
        "shuffle_input_sentence": str(config.shuffle_input_sentence).lower(),
        # Special token ID contract — explicit, not relying on defaults
        "pad_id": str(PAD_ID),
        "bos_id": str(SOS_ID),
        "eos_id": str(EOS_ID),
        "unk_id": str(UNK_ID),
        "pad_piece": PAD_PIECE,
        "bos_piece": SOS_PIECE,
        "eos_piece": EOS_PIECE,
        "unk_piece": UNK_PIECE,
    }


def train_bpe(config: TokenizerConfig, input_path: Path, model_prefix: Path, vocab_size: int) -> None:
    """Train a single SentencePiece BPE model."""
    args = _sp_train_args(config, input_path, model_prefix, vocab_size)
    spm.SentencePieceTrainer.train(**args)


def train_separate(config: TokenizerConfig, en_path: Path, or_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Train separate English and Odia tokenizers.
    Returns (en_model_path, or_model_path).
    """
    en_model = out_dir / f"tokenizer_{config.config_id()}_en"
    or_model = out_dir / f"tokenizer_{config.config_id()}_or"
    train_bpe(config, en_path, en_model, config.vocab_size_en)
    train_bpe(config, or_path, or_model, config.vocab_size_or)
    return en_model.with_suffix(".model"), or_model.with_suffix(".model")


def train_shared(config: TokenizerConfig, shared_path: Path, out_dir: Path) -> Path:
    """Train a shared tokenizer.
    Returns model path.
    """
    model_path = out_dir / f"tokenizer_{config.config_id()}"
    train_bpe(config, shared_path, model_path, config.vocab_size_shared)
    return model_path.with_suffix(".model")


# ---------------------------------------------------------------------------
# Load / wrapper
# ---------------------------------------------------------------------------

class Tokenizer:
    """Wrapper around a trained SentencePiece model with explicit encode/decode."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))
        self._verify_special_ids()

    def _verify_special_ids(self) -> None:
        """Verify the hard special-token ID contract."""
        for piece, expected_id in SPECIAL_IDS.items():
            actual = self.sp.piece_to_id(piece)
            if actual != expected_id:
                raise ValueError(
                    f"Special token ID mismatch: {piece} expected {expected_id}, got {actual}"
                )

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Encode text to subword token IDs. Does NOT add SOS/EOS."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)

    def encode_with_sos_eos(self, text: str) -> list[int]:
        """Encode with SOS prepended and EOS appended (for decoder input/target construction)."""
        return [SOS_ID] + self.encode(text) + [EOS_ID]

    def get_vocab(self) -> dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def get_id_to_piece(self) -> dict[int, str]:
        return {i: self.sp.id_to_piece(i) for i in range(self.vocab_size)}

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)


def load_tokenizer(model_path: Path) -> Tokenizer:
    """Load a tokenizer from a .model file."""
    return Tokenizer(model_path)


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def _whitespace_words(text: str) -> list[str]:
    return text.split()


def _subword_tokens(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def compute_vocab_utilization(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    """Fraction of vocabulary pieces that appear at least once."""
    seen = set()
    for text in texts:
        seen.update(tokenizer.encode(text))
    return len(seen) / tokenizer.vocab_size if tokenizer.vocab_size > 0 else 0.0


def compute_unk_rate(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    """Percentage of tokens that are UNK_ID."""
    total = 0
    unk = 0
    for text in texts:
        ids = tokenizer.encode(text)
        total += len(ids)
        unk += sum(1 for i in ids if i == UNK_ID)
    return (unk / total * 100) if total > 0 else 0.0


def compute_fragmentation(tokenizer: Tokenizer, texts: Iterable[str]) -> tuple[float, float]:
    """Returns (avg_subwords_per_whitespace_word, avg_subwords_per_sentence)."""
    total_subwords = 0
    total_words = 0
    total_sentences = 0
    for text in texts:
        ids = tokenizer.encode(text)
        total_subwords += len(ids)
        total_words += len(_whitespace_words(text))
        total_sentences += 1
    sw_per_word = (total_subwords / total_words) if total_words > 0 else 0.0
    sw_per_sent = (total_subwords / total_sentences) if total_sentences > 0 else 0.0
    return sw_per_word, sw_per_sent


def compute_sequence_lengths(tokenizer: Tokenizer, texts: Iterable[str]) -> dict:
    """Returns dict with p50, p95, p99, max of tokenized sequence lengths."""
    lengths = [len(tokenizer.encode(text)) for text in texts]
    if not lengths:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0}
    import numpy as np
    arr = np.array(lengths, dtype=np.int64)
    return {
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def compute_piece_lengths(tokenizer: Tokenizer) -> dict:
    """Character-length distribution of individual subword pieces (p50, p95, p99)."""
    import numpy as np
    lengths = [len(tokenizer.id_to_piece(i)) for i in range(tokenizer.vocab_size)]
    if not lengths:
        return {"p50": 0, "p95": 0, "p99": 0}
    arr = np.array(lengths, dtype=np.int64)
    return {
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }


def compute_roundtrip_fidelity(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    """Fraction of texts that roundtrip exactly: decode(encode(text)) == text."""
    total = 0
    exact = 0
    for text in texts:
        total += 1
        if tokenizer.decode(tokenizer.encode(text)) == text:
            exact += 1
    return (exact / total * 100) if total > 0 else 0.0


def compute_parameter_cost(config: TokenizerConfig) -> int:
    """Vocabulary-related Transformer parameter count (d_model=128)."""
    d_model = 128
    if config.mode == "separate":
        return (config.vocab_size_en + 2 * config.vocab_size_or) * d_model
    else:
        return 2 * config.vocab_size_shared * d_model


def evaluate_split(
    tokenizer_en: Tokenizer | None,
    tokenizer_or: Tokenizer | None,
    df: pd.DataFrame,
    mode: str,
) -> dict:
    """Evaluate a single split (train/val/test) and return metrics dict."""
    src_texts = df["src"].astype(str).tolist()
    tgt_texts = df["tgt"].astype(str).tolist()

    if mode == "separate":
        # English metrics
        en_util = compute_vocab_utilization(tokenizer_en, src_texts)
        en_unk = compute_unk_rate(tokenizer_en, src_texts)
        en_frag_word, en_frag_sent = compute_fragmentation(tokenizer_en, src_texts)
        en_seq = compute_sequence_lengths(tokenizer_en, src_texts)
        # Odia metrics
        or_util = compute_vocab_utilization(tokenizer_or, tgt_texts)
        or_unk = compute_unk_rate(tokenizer_or, tgt_texts)
        or_frag_word, or_frag_sent = compute_fragmentation(tokenizer_or, tgt_texts)
        or_seq = compute_sequence_lengths(tokenizer_or, tgt_texts)
        # Roundtrip
        en_rt = compute_roundtrip_fidelity(tokenizer_en, src_texts)
        or_rt = compute_roundtrip_fidelity(tokenizer_or, tgt_texts)
        return {
            "en": {
                "vocab_utilization": en_util,
                "unk_rate": en_unk,
                "fragmentation_per_word": en_frag_word,
                "fragmentation_per_sentence": en_frag_sent,
                "sequence_lengths": en_seq,
                "roundtrip_fidelity": en_rt,
            },
            "or": {
                "vocab_utilization": or_util,
                "unk_rate": or_unk,
                "fragmentation_per_word": or_frag_word,
                "fragmentation_per_sentence": or_frag_sent,
                "sequence_lengths": or_seq,
                "roundtrip_fidelity": or_rt,
            },
        }
    else:  # shared
        # Single tokenizer for both
        shared = tokenizer_en  # tokenizer_en is the shared one
        src_util = compute_vocab_utilization(shared, src_texts)
        src_unk = compute_unk_rate(shared, src_texts)
        src_frag_word, src_frag_sent = compute_fragmentation(shared, src_texts)
        src_seq = compute_sequence_lengths(shared, src_texts)
        tgt_util = compute_vocab_utilization(shared, tgt_texts)
        tgt_unk = compute_unk_rate(shared, tgt_texts)
        tgt_frag_word, tgt_frag_sent = compute_fragmentation(shared, tgt_texts)
        tgt_seq = compute_sequence_lengths(shared, tgt_texts)
        src_rt = compute_roundtrip_fidelity(shared, src_texts)
        tgt_rt = compute_roundtrip_fidelity(shared, tgt_texts)
        return {
            "shared": {
                "src": {
                    "vocab_utilization": src_util,
                    "unk_rate": src_unk,
                    "fragmentation_per_word": src_frag_word,
                    "fragmentation_per_sentence": src_frag_sent,
                    "sequence_lengths": src_seq,
                    "roundtrip_fidelity": src_rt,
                },
                "tgt": {
                    "vocab_utilization": tgt_util,
                    "unk_rate": tgt_unk,
                    "fragmentation_per_word": tgt_frag_word,
                    "fragmentation_per_sentence": tgt_frag_sent,
                    "sequence_lengths": tgt_seq,
                    "roundtrip_fidelity": tgt_rt,
                },
            }
        }


def collect_piece_lengths(tokenizer: Tokenizer) -> dict:
    return compute_piece_lengths(tokenizer)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    config: TokenizerConfig
    model_paths: list[Path]
    metrics_train: dict
    metrics_val: dict
    metrics_test: dict
    parameter_cost: int
    piece_lengths: dict  # for en/or or shared

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "model_paths": [str(p) for p in self.model_paths],
            "metrics_train": self.metrics_train,
            "metrics_val": self.metrics_val,
            "metrics_test": self.metrics_test,
            "parameter_cost": self.parameter_cost,
            "piece_lengths": self.piece_lengths,
        }


def run_single_experiment(
    config: TokenizerConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> ExperimentResult:
    """Run one tokenizer experiment configuration end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if config.mode == "separate":
            # Build corpora
            en_corpus = tmp / "en_corpus.txt"
            or_corpus = tmp / "or_corpus.txt"
            build_separate_corpus_files(train_df, en_corpus, or_corpus)
            # Train
            en_model, or_model = train_separate(config, en_corpus, or_corpus, out_dir)
            # Load
            tok_en = load_tokenizer(en_model)
            tok_or = load_tokenizer(or_model)
            model_paths = [en_model, or_model]
            piece_lengths = {
                "en": collect_piece_lengths(tok_en),
                "or": collect_piece_lengths(tok_or),
            }
        else:  # shared
            shared_corpus = tmp / "shared_corpus.txt"
            build_shared_corpus_file(train_df, shared_corpus)
            model_path = train_shared(config, shared_corpus, out_dir)
            tok_shared = load_tokenizer(model_path)
            model_paths = [model_path]
            piece_lengths = {"shared": collect_piece_lengths(tok_shared)}

        # Evaluate
        if config.mode == "separate":
            metrics_train = evaluate_split(tok_en, tok_or, train_df, "separate")
            metrics_val = evaluate_split(tok_en, tok_or, val_df, "separate")
            metrics_test = evaluate_split(tok_en, tok_or, test_df, "separate")
        else:
            metrics_train = evaluate_split(tok_shared, None, train_df, "shared")
            metrics_val = evaluate_split(tok_shared, None, val_df, "shared")
            metrics_test = evaluate_split(tok_shared, None, test_df, "shared")

    param_cost = compute_parameter_cost(config)

    return ExperimentResult(
        config=config,
        model_paths=model_paths,
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        parameter_cost=param_cost,
        piece_lengths=piece_lengths,
    )


def run_all_experiments(
    configs: list[TokenizerConfig],
    train_path: Path,
    val_path: Path,
    test_path: Path,
    out_dir: Path,
) -> list[ExperimentResult]:
    """Run the full experiment matrix and return all results."""
    train_df = load_train_corpus(train_path)
    val_df = load_train_corpus(val_path)
    test_df = load_train_corpus(test_path)

    results = []
    for config in configs:
        print(f"Running: {config.config_id()}")
        result = run_single_experiment(config, train_df, val_df, test_df, out_dir)
        results.append(result)
        print(f"  Completed: {config.config_id()}")
    return results


def save_experiment_results(results: list[ExperimentResult], out_path: Path) -> None:
    """Save all experiment results to JSON."""
    data = [r.to_dict() for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_experiment_configs() -> list[TokenizerConfig]:
    """Generate the full 20-configuration experiment matrix per Phase 2A."""
    configs = []

    # Separate: 4 EN × 4 OR = 16
    en_sizes = [4000, 8000, 12000, 16000]
    or_sizes = [8000, 16000, 24000, 32000]
    for en_v in en_sizes:
        for or_v in or_sizes:
            configs.append(TokenizerConfig(
                mode="separate",
                vocab_size_en=en_v,
                vocab_size_or=or_v,
            ))

    # Shared: 4 sizes = 4
    shared_sizes = [16000, 24000, 32000, 40000]
    for v in shared_sizes:
        configs.append(TokenizerConfig(
            mode="shared",
            vocab_size_shared=v,
        ))

    return configs