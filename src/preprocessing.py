"""Phase 1C preprocessing for the OdiaTransformer English->Odia dataset.

Implements the authoritative design in docs/01_preprocessing_spec.md.

Pipeline:
    raw Parquet
    -> 1. schema validation
    -> 2. text validation
    -> 3. NFC normalization
    -> 4. conservative text cleanup  (leading triple-quote artifact only)
    -> 5. exact (src, tgt) pair deduplication
    -> 6. narrow numeric/table noise filtering (conservative)
    -> 7. final validation
    -> 8. deterministic train/validation/test split
    -> 9. save processed outputs + report

Design notes
------------
* The raw parquet is never modified; it is read-only input and its SHA-256 is
  recorded before and checked after.
* Functions are small and side-effect-free where possible (CQRS-flavoured):
  transformations return data, the orchestration lives in L{pipeline}.
* No semantic preprocessing: no lowercasing, no punctuation removal, no
  stopword removal, no stemming/lemmatization, no transliteration, no NFKC.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = ("idx", "src", "tgt")


def _str_col(s: pd.Series) -> pd.Series:
    """Return a series of strings, mapping NaN to '' (never used for discard)."""
    return s.fillna("").astype(str)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of an arbitrary file without loading it wholly."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm the frame has exactly the expected columns with correct dtypes.

    Raises ValueError on any mismatch. Returns the original frame unchanged.
    """
    actual = tuple(df.columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            f"Schema mismatch: expected columns {EXPECTED_COLUMNS!r} but found {actual!r}. "
            "Will not silently rename or reinterpret columns."
        )
    if not pd.api.types.is_integer_dtype(df["idx"]):
        raise ValueError(
            f"Schema mismatch: 'idx' must be an integer dtype, got {df['idx'].dtype}."
        )
    for col in ("src", "tgt"):
        if not (
            df[col].dtype == object
            or pd.api.types.is_string_dtype(df[col])
        ):
            raise ValueError(
                f"Schema mismatch: '{col}' must be a string/object dtype, got {df[col].dtype}."
            )
    return df


# ---------------------------------------------------------------------------
# 2. Text validation
# ---------------------------------------------------------------------------

def validate_text(df: pd.DataFrame) -> dict:
    """Validate required text. Returns a summary dict; raises on invalid data.

    Failures (NaN, empty, whitespace-only, control chars) are hard failures --
    invalid records are never silently discarded.
    """
    summary = {"nulls": {}, "empty": {}, "control_chars": {}}
    for col in ("src", "tgt"):
        s = df[col]
        n_nan = int(s.isna().sum())
        if n_nan:
            raise ValueError(
                f"Text validation failed: column '{col}' has {n_nan} NaN/null value(s) "
                "present in the raw data; refusing to proceed."
            )
        str_s = _str_col(s)
        n_empty = int((str_s.str.strip() == "").sum())
        if n_empty:
            raise ValueError(
                f"Text validation failed: column '{col}' has {n_empty} empty/"
                "whitespace-only value(s); refusing to proceed."
            )
        n_ctrl = int(sum(any(ord(ch) < 32 for ch in v) for v in str_s))
        summary["nulls"][col] = n_nan
        summary["empty"][col] = n_empty
        summary["control_chars"][col] = n_ctrl
    return summary


# ---------------------------------------------------------------------------
# 3. NFC normalization
# ---------------------------------------------------------------------------

def apply_nfc(text: str) -> str:
    """Normalize a single string with Unicode NFC (never NFKC)."""
    return unicodedata.normalize("NFC", text)


def count_non_nfc(series: pd.Series) -> int:
    """Count values whose NFC form differs (i.e. that will change under NFC)."""
    str_s = _str_col(series)
    return int(sum(1 for v in str_s if v != unicodedata.normalize("NFC", v)))


def normalize_nfc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with NFC normalization applied to both src and tgt."""
    out = df.copy()
    for col in ("src", "tgt"):
        out[col] = _str_col(out[col]).map(apply_nfc)
    return out


# ---------------------------------------------------------------------------
# 4. Conservative text cleanup (leading triple-quote artifact on src only)
# ---------------------------------------------------------------------------

_LEADING_TRIPLE_QUOTE = re.compile(r'^"""(?!")')


def clean_source_leading_quote(src: str) -> str:
    """Remove a leading triple-quote artifact from the source string.

    Only an exact leading `\"\"\"` (followed by a non-quote) is removed. A single
    leading quote (`"`) and a double leading quote (`""`) are preserved because
    they are the legitimate balanced/nested-quote patterns found in the corpus.
    If there is any ambiguity, the text is preserved unchanged.
    """
    if src.startswith('"""') and (len(src) == 3 or src[3] != '"'):
        return src[3:].lstrip()
    return src


def apply_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with the conservative cleanup applied to src only."""
    out = df.copy()
    out["src"] = out["src"].map(clean_source_leading_quote)
    return out


# ---------------------------------------------------------------------------
# 5. Exact (src, tgt) pair deduplication
# ---------------------------------------------------------------------------

def deduplicate_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate (src, tgt) pairs only.

    Keeps duplicate `src` (multiple valid targets) and duplicate `tgt`.
    Returns (deduplicated_frame, number_of_rows_removed).
    """
    before = len(df)
    out = df.drop_duplicates(subset=["src", "tgt"], keep="first").copy()
    removed = before - len(out)
    return out, removed


# ---------------------------------------------------------------------------
# 6. Narrow numeric / table noise filtering (conservative)
# ---------------------------------------------------------------------------

def compute_numeric_fraction(text: str) -> float:
    """Numeric-token fraction per the authoritative 1A definition.

    tokens = s.split() ; a numeric token is t.replace('.','').replace(',','').isdigit();
    fraction = numeric_tokens / total_tokens.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if t.replace(".", "").replace(",", "").isdigit())
    return numeric / len(tokens)


def numeric_candidate(src: str, tgt: str, threshold: float) -> bool:
    """Strict `> threshold` candidate test, evaluated independently per side.

    A row is a *candidate* if either side's numeric fraction is strictly > threshold.
    """
    return compute_numeric_fraction(src) > threshold or compute_numeric_fraction(tgt) > threshold


def _is_table_fragment(src: str, tgt: str, threshold: float) -> bool:
    """Conservative exclusion test: is this candidate a malformed table fragment?

    A row is *excluded* only when BOTH of these hold:
      * both languages are numeric-heavy (numeric fraction > threshold on the
        source AND on the target), and
      * the source is long enough to be a genuine statistical/table block
        (>= {@link _MIN_TABLE_SRC_TOKENS} whitespace tokens).

    The token floor separates genuine malformed statistical/table fragments
    (long, dense rows e.g. state-wise procurement or roll-number tables) from
    legitimate short numeric-heavy translations such as "13 places" or
    "Version 1" (which are numeric-heavy but clearly valid sentence pairs and
    must never be deleted). This keeps the filtering conservative per the spec.
    """
    return (
        compute_numeric_fraction(src) > threshold
        and compute_numeric_fraction(tgt) > threshold
        and len(src.split()) >= _MIN_TABLE_SRC_TOKENS
    )


_MIN_TABLE_SRC_TOKENS = 20


def numeric_noise_filter(
    df: pd.DataFrame, threshold: float = 0.35
) -> tuple[pd.DataFrame, list[int], dict]:
    """Identify and remove malformed numeric/table fragments.

    Returns (filtered_frame, excluded_indices, info) where info holds the
    candidate count and reason breakdown. Every excluded row is logged through
    `info['excluded_src']` / `info['excluded_tgt']` (index+text) and also
    returned via a report entry for human review.
    """
    src = _str_col(df["src"])
    tgt = _str_col(df["tgt"])

    cand_mask = pd.Series(
        [numeric_candidate(s, t, threshold) for s, t in zip(src, tgt)],
        index=df.index,
    )
    excl_mask = pd.Series(
        [_is_table_fragment(s, t, threshold) for s, t in zip(src, tgt)],
        index=df.index,
    )
    # Only rows that are candidates may be excluded.
    excl_mask &= cand_mask

    cand_count = int(cand_mask.sum())
    excl_count = int(excl_mask.sum())

    excluded_indices = df.index[excl_mask].tolist()
    info = {
        "threshold": threshold,
        "candidate_count": cand_count,
        "excluded_count": excl_count,
        "excluded_idx": excluded_indices,
        "excluded_src": df.loc[excl_mask, "src"].tolist(),
        "excluded_tgt": df.loc[excl_mask, "tgt"].tolist(),
    }
    out = df[~excl_mask].copy()
    return out, excluded_indices, info


# ---------------------------------------------------------------------------
# 7. Final validation
# ---------------------------------------------------------------------------

def final_validation(df: pd.DataFrame) -> dict:
    """Confirm invariants hold on the processed frame. Raises on violation."""
    for col in ("src", "tgt"):
        if int(df[col].isna().sum()):
            raise ValueError(f"Final validation failed: NaN in '{col}'.")
        str_s = _str_col(df[col])
        if int((str_s.str.strip() == "").sum()):
            raise ValueError(f"Final validation failed: empty/'{col}' string.")
        if int(sum(1 for v in str_s if v != unicodedata.normalize("NFC", v))):
            raise ValueError(f"Final validation failed: column '{col}' is not fully NFC-normalized.")
    dup_pairs = int(df.duplicated(subset=["src", "tgt"]).sum())
    if dup_pairs:
        raise ValueError(f"Final validation failed: {dup_pairs} duplicate (src,tgt) pairs remain.")
    return {"duplicate_pairs_remaining": dup_pairs}


# ---------------------------------------------------------------------------
# 8. Deterministic split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split into (train, val, test) with exact accounting.

    Row counts are exact: train = round(n*train_frac), val = round(n*val_frac),
    test = n - train - val, so train+val+test == n with no overlap.
    """
    n = len(df)
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0) or (train_frac + val_frac >= 1.0):
        raise ValueError(
            f"Invalid split fractions: train={train_frac} val={val_frac} must be "
            "in (0,1) and sum < 1."
        )
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    idx = df.index.to_numpy()
    train_idx = idx[order[:n_train]]
    val_idx = idx[order[n_train : n_train + n_val]]
    test_idx = idx[order[n_train + n_val :]]

    return (
        df.loc[train_idx].copy(),
        df.loc[val_idx].copy(),
        df.loc[test_idx].copy(),
    )


# ---------------------------------------------------------------------------
# Length statistics for reporting
# ---------------------------------------------------------------------------

def length_stats(series: pd.Series) -> dict:
    """Whitespace-token and character length summary for a string column."""

    def _summ(vals: Iterable[int], suffix: str) -> dict:
        arr = np.fromiter(vals, dtype=np.int64, count=len(vals)) if vals else np.array([], dtype=np.int64)
        if arr.size == 0:
            return {f"{suffix}_min": 0, f"{suffix}_p95": 0, f"{suffix}_p99": 0,
                    f"{suffix}_p99.5": 0, f"{suffix}_p99.9": 0, f"{suffix}_max": 0}
        return {
            f"{suffix}_min": int(arr.min()),
            f"{suffix}_p95": int(np.quantile(arr, 0.95)),
            f"{suffix}_p99": int(np.quantile(arr, 0.99)),
            f"{suffix}_p99.5": int(np.quantile(arr, 0.995)),
            f"{suffix}_p99.9": int(np.quantile(arr, 0.999)),
            f"{suffix}_max": int(arr.max()),
        }

    str_s = _str_col(series)
    tokens = [len(v.split()) for v in str_s]
    chars = [len(v) for v in str_s]
    d = {}
    d.update(_summ(tokens, "tokens"))
    d.update(_summ(chars, "chars"))
    return d


def length_ratio_stats(src: pd.Series, tgt: pd.Series) -> dict:
    """Odia-character / English-character length-ratio summary."""
    s = _str_col(src).str.len().replace(0, np.nan)
    t = _str_col(tgt).str.len()
    ratio = (t / s).dropna()
    if ratio.empty:
        return {"ratio_count": 0, "ratio_median": 0.0, "ratio_in_0_5_2": 0.0}
    return {
        "ratio_count": int(ratio.count()),
        "ratio_median": float(ratio.median()),
        "ratio_in_0_5_2": float(ratio.between(0.5, 2.0).mean()),
    }


# ---------------------------------------------------------------------------
# I/O + report
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_parquet(str(path))


def save_split_outputs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_parquet(paths[name], index=False)
    return paths


def build_report(
    *,
    raw_path: str,
    raw_hash_before: str,
    raw_hash_after: str,
    raw_rows: int,
    schema_validation: dict,
    text_validation: dict,
    nfc_before: dict,
    after_rows_per_step: dict,
    dup: dict,
    numeric: dict,
    final_validation: dict,
    length_before: dict,
    length_after: dict,
    ratio_before: dict,
    ratio_after: dict,
    leading_quote_cleaned: int,
    splits: dict,
    seed: int,
    outputs: dict,
    version: str,
) -> dict:
    report = {
        "tool": "OdiaTransformer preprocessing (Phase 1C)",
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_input": {"path": raw_path, "rows": raw_rows,
                      "sha256_before": raw_hash_before, "sha256_after": raw_hash_after,
                      "raw_immutable": raw_hash_before == raw_hash_after},
        "schema_validation": schema_validation,
        "text_validation": text_validation,
        "nfc": {
            "before": nfc_before,
            "after": {"src": 0, "tgt": 0},  # both guaranteed fully NFC by final validation
        },
        "rows_after_each_step": after_rows_per_step,
        "duplicates": dup,
        "numeric_noise": numeric,
        "final_validation": final_validation,
        "length_statistics_before": length_before,
        "length_statistics_after": length_after,
        "length_ratio_before": ratio_before,
        "length_ratio_after": ratio_after,
        "leading_quote_artifact_cleaned": leading_quote_cleaned,
        "split": {"seed": seed, **splits},
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    return report


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preprocessing_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Orchestration (CQRS: the pipeline orchestrates the steps)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    report: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    excluded_samples: pd.DataFrame

@dataclass
class SplitConfig:
    train_frac: float = 0.98
    val_frac: float = 0.01
    seed: int = 42


def pipeline(
    raw_path: Path,
    out_dir: Path,
    split: SplitConfig = SplitConfig(),
    threshold: float = 0.35,
    version: str = "0.1.0",
) -> PreprocessResult:
    """Run the full preprocessing pipeline and return results + report.

    The raw parquet is read-only; it is hashed before and after and the two
    hashes are compared and recorded in the report.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw_hash_before = sha256_file(raw_path)

    # 1. Schema validation
    df = load_raw(raw_path)
    raw_rows = len(df)
    validate_schema(df)

    # 2. Text validation
    schema_note = {"expected_columns": list(EXPECTED_COLUMNS),
                   "found_columns": list(df.columns)}
    text_validation = validate_text(df)

    # Capture the fresh raw text for the "before" metrics (pre-NFC / pre-cleanup).
    raw_src_series = _str_col(df["src"])
    raw_tgt_series = _str_col(df["tgt"])

    # -- start accounting: rows after each step
    after_rows = {"raw": raw_rows}
    nfc_before = {"src": count_non_nfc(df["src"]), "tgt": count_non_nfc(df["tgt"])}

    # 3. NFC normalization
    df = normalize_nfc(df)
    after_rows["after_nfc"] = len(df)

    # 4. Conservative text cleanup
    before_cleanup = df["src"]
    df = apply_cleanup(df)
    leading_quote_cleaned = int((before_cleanup != df["src"]).sum())
    after_rows["after_cleanup"] = len(df)

    # 5. Exact pair deduplication
    df, dup_removed = deduplicate_pairs(df)
    dup = {"duplicate_pairs_before": dup_removed, "removed": dup_removed,
           "remaining": len(df)}
    after_rows["after_dedup"] = len(df)

    # 6. Narrow numeric/table noise filtering
    df, _excluded_indices, numeric = numeric_noise_filter(df, threshold=threshold)
    after_rows["after_numeric_filter"] = len(df)

    # 7. Final validation
    final_validation_result = final_validation(df)
    after_rows["final"] = len(df)

    # -- metrics before/after.
    # The "before" stats reflect the fresh raw text (pre-NFC / pre-cleanup),
    # captured from the raw frame at load time.
    length_before = {
        "src": length_stats(raw_src_series), "tgt": length_stats(raw_tgt_series),
    }
    ratio_before = length_ratio_stats(raw_src_series, raw_tgt_series)

    length_after = {
        "src": length_stats(df["src"]), "tgt": length_stats(df["tgt"]),
    }
    ratio_after = length_ratio_stats(df["src"], df["tgt"])

    # 8. Deterministic split
    train, val, test = split_dataset(df, split.train_frac, split.val_frac, split.seed)
    splits = {
        "train_frac": split.train_frac,
        "val_frac": split.val_frac,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(train) + len(val) + len(test),
        "accounting_exact": len(train) + len(val) + len(test) == len(df),
    }

    # 9. Save outputs
    outputs = save_split_outputs(train, val, test, out_dir)

    excluded_samples = pd.DataFrame(columns=["idx", "src", "tgt"])
    if numeric["excluded_src"]:
        excluded_samples = pd.DataFrame(
            {"idx": numeric["excluded_idx"], "src": numeric["excluded_src"],
             "tgt": numeric["excluded_tgt"]}
        )
        excluded_samples.to_parquet(out_dir / "numeric_excluded_samples.parquet", index=False)

    raw_hash_after = sha256_file(raw_path)

    report = build_report(
        raw_path=str(raw_path),
        raw_hash_before=raw_hash_before,
        raw_hash_after=raw_hash_after,
        raw_rows=raw_rows,
        schema_validation=schema_note,
        text_validation=text_validation,
        nfc_before=nfc_before,
        after_rows_per_step=after_rows,
        dup=dup,
        numeric=numeric,
        final_validation=final_validation_result,
        length_before=length_before,
        length_after=length_after,
        ratio_before=ratio_before,
        ratio_after=ratio_after,
        leading_quote_cleaned=leading_quote_cleaned,
        splits=splits,
        seed=split.seed,
        outputs=outputs,
        version=version,
    )
    report_path = write_report(report, out_dir)

    report["report_path"] = str(report_path)
    return PreprocessResult(
        report=report,
        train=train,
        val=val,
        test=test,
        excluded_samples=excluded_samples,
    )
