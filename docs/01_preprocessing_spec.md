# Phase 1B — Preprocessing Design Specification

**Project:** Test-7 / OdiaTransformer — English → Odia translation from the Samanantar `or` subset.
**Status:** Design / specification only. **No preprocessing code is implemented yet.**
**Scope:** This document defines *what* preprocessing will do and *why*. It does NOT implement preprocessing, does not choose tokenizer/vocabulary settings, does not choose train/validation/test percentages, and does not create any Transformer code.

---

## 0. Provenance legend

Every decision below is labeled with its source of truth. Use these labels to audit the pipeline:

- **[EDA]** — established directly by `notebooks/01_dataset_eda.ipynb` (the committed Phase-1A notebook).
- **[1A-inv]** — established by the **separate Phase 1A targeted investigation**, which ran in *temporary analysis scripts that were deleted and are not part of any committed file*. This is important: rules labeled 1A-inv have **no anchor in the EDA notebook** and must be re-established verbatim from this spec before being implemented.
- **[decision]** — a design decision now adopted for Phase 1B preprocessing.

---

## 1. Scope & constraints

1. The **raw parquet must never be modified.** Preprocessing reads `dataset/train-00000-of-00001.parquet` (columns `idx`, `src`, `tgt`; no `data_source`) and writes a *separate* processed dataset.
2. No pre-trained-model libraries are used (see `AGENTS.md`). Preprocessing is limited to dataset handling, subword/token-level statistics, and PyTorch tensor ops that are already allowed.
3. This phase produces a **processed, split dataset** only. Tokenization, vocabulary construction, and the Transformer are **out of scope for this document's decisions** (noted as later pipeline steps).
4. Train/validation/test **percentages and split mechanics are a later decision** — the pipeline includes a split step, but the exact split is **not** fixed here.

---

## 2. Authoritative preprocessing decisions

### 2.1 Unicode normalization — NFC **[EDA] + [1A-inv]**

- Apply **Unicode NFC** normalization to **both `src` and `tgt`**.
- **Do NOT use NFKC.** NFKC differs on 129,377 Odia rows; it performs compatibility decomposition (heavier) and is rejected as too aggressive.
- EDA evidence: `src` has **0** non-NFC rows; `tgt` has **116,794** non-NFC rows (11.7%).
- 1A-inv validation: every NFC change is **canonical composition** (e.g. `ୋ` → `ୋ`, `ୋ` → `ୌ`), shifting string length by only 0–4 codepoints, with **no semantic/textual change**. NFC is therefore safe.
- The raw dataset is read-only; normalization is applied to the processed copy only.

### 2.2 Duplicate handling — exact `(src, tgt)` pairs only **[EDA]**

- **Remove exact duplicate `(src, tgt)` pairs** if any are encountered.
- **Keep duplicate `src` values** — the same English source may have multiple valid Odia targets.
- **Keep duplicate `tgt` values.**
- **Do NOT deduplicate on `src` alone.**
- Current dataset state (EDA): **0** duplicate `(src, tgt)` pairs; 372,334 repeated `src`; 6,087 repeated `tgt`. Deduplication on the pair is therefore a *guardrail* that should effectively change nothing today, but must be correct if new data is added.

### 2.3 Broad suspicious mask — analysis-only, NOT a deletion filter **[EDA] + [1A-inv]**

- The existing suspicious/path-like mask (`has_ctrl | low_odia | pathish`) is **analysis-only**.
- It must **NOT** be used as a deletion filter.
- The 1A investigation showed **~94% of the 20,908 `pathish` flags were false positives** (valid natural-language rows flagged by ordinary quotes / ellipsis). Using this mask to drop rows would delete high-quality translation data.

### 2.4 Numeric / table noise — narrow filter **[1A-inv — from a TEMPORARY analysis script, NOT the EDA notebook]**

> **Provenance warning:** This rule came from the Phase 1A targeted-investigation *temporary script* (deleted). It is **NOT** in `01_dataset_eda.ipynb`. It is codified here verbatim so implementation matches the exact investigated definition. **Do not invent a different threshold.**

Exact definition:

- Whitespace-tokenize a string with `str.split()` (pandas default — splits on arbitrary whitespace runs).
- A **numeric token** is one where:
  ```python
  t.replace('.', '').replace(',', '').isdigit()
  ```
  i.e. a token that is entirely digits after stripping `.` and `,`.
- **numeric fraction** = (numeric-token count) / (total whitespace-token count).
- Evaluate **`src` and `tgt` independently**.
- Candidate threshold is **strictly `> 0.35`** (greater-than, not greater-than-or-equal).

Reference (from the 1A script):

```python
def numeric_token_frac(s):
    toks = s.split()
    if not toks:
        return 0.0
    num = sum(1 for t in toks if t.replace('.', '').replace(',', '').isdigit())
    return num / len(toks)

src_numeric_candidate = df['src'].fillna('').apply(numeric_token_frac) > 0.35   # ~214 rows
tgt_numeric_candidate = df['tgt'].fillna('').apply(numeric_token_frac) > 0.35   # ~176 rows
```

> **Conservative implementation guardrail:** Do **not** blindly remove every row above the 35% threshold. The intent is to remove only **clearly malformed numeric / statistical-table fragments** (e.g. misaligned `STATE NAE BASIN 1 417.44 Rajasthan …` spreadsheet rows, `Nos. 2015-16 2016-17 …` numeric tables). Before dropping, **validate the intended behavior against the investigated examples** and confirm that legitimate sentences containing numbers are **not** caught. Report and log the exact rows that would be excluded for human review. If any legitimate sentence is flagged, refine/confirm the rule rather than deleting.

### 2.5 Leading quote artifact — conservative cleanup, NOT a filter **[1A-inv]**

- The 1A investigation found **many English sources begin with a leading triple-quote artifact** (e.g. sources starting with `"""`), e.g. \`"""He has worked for the people."\`.
- **Do NOT globally strip quotation marks.**
- **Do NOT delete these rows.**
- If implemented, remove **only the specific leading quote artifact** identified during the investigation — i.e. an odd leading `"""` / `""` run **at the beginning of the source text / token boundary** — and only when doing so does not touch balanced/nested quotation marks.
- **Preserve legitimate balanced/nested quotes** (the corpus genuinely contains valid quoted sentences and nested quotes).
- **This is a normalization / cleanup operation, NOT a quality filter.**

### 2.6 Sentence length — NO dataset-level hard cap **[EDA] + [1A-inv]**

- **Do NOT impose a dataset-level hard sentence-length cap.**
- **Retain legitimate long sentences** (long rows are overwhelmingly valid news / legal / scripture / speech content).
- The later tokenizer / training pipeline **may** impose model sequence limits if technically necessary, but that is **NOT** a Phase-1 dataset filtering rule.

### 2.7 URL / UI / format / localization pairs — retain **[1A-inv] + [decision]**

- **Retain legitimate URL-containing, UI-format, and localization pairs** (e.g. `%s`, `%d` template strings; `Manipuri: https://…` ↔ `ମଣିପୁରୀ ଭାଷା : https://…`).
- They must **NOT** be removed merely because they triggered the broad suspicious detector. They are valid English→Odia text representing a localization/UI domain.

### 2.8 No semantic preprocessing **[decision]**

The translation task needs the **original linguistic information**. Do **not** perform any of the following, unless a later project decision explicitly introduces the transformation:

- lowercase the corpus
- remove punctuation
- remove stopwords
- stem / lemmatize
- transliterate Odia (or transliterate to Latin)
- use NFKC

---

## 3. Preprocessing pipeline (in order)

```
raw Parquet
→ 1. schema validation
→ 2. text validation
→ 3. NFC normalization
→ 4. conservative text cleanup
→ 5. exact (src, tgt) pair deduplication
→ 6. narrow numeric / table noise filtering
→ 7. final validation
→ 8. train/validation/test split   (percentages = later decision)
→ 9. save processed dataset
```

High-level step responsibilities:

1. **Schema validation** — confirm columns are exactly `idx`, `src`, `tgt`; `idx` is integer, `src`/`tgt` are strings. Hard-fail on unexpected schema.
2. **Text validation** — detect/record nulls, empty/whitespace-only strings, and control characters; log counts. Do not silently proceed on unexpected quality.
3. **NFC normalization** — apply on `src` and `tgt` (§2.1).
4. **Conservative text cleanup** — optional removal of the leading quote artifact only (§2.5). No other textual rewriting.
5. **Exact pair deduplication** — drop duplicate `(src, tgt)` rows only (§2.2).
6. **Narrow numeric/table noise filtering** — conservative `> 0.35` numeric fraction evaluation, independently per side, validated against investigated examples before dropping (§2.4).
7. **Final validation** — confirm invariants (§5).
8. **Train/validation/test split** — present as a step; **exact percentages and mechanics are a later decision** (§1.4).
9. **Save processed dataset** — write split files + a preprocessing metrics/report artifact under `outputs/` (generated data; not committed — see `AGENTS.md`).

---

## 4. Expected inputs & outputs

**Expected inputs**

- `dataset/train-00000-of-00001.parquet` (the raw corpus; ~998,228 rows; columns `idx: int64`, `src: str`, `tgt: str`).
- No `data_source` column is expected (confirmed absent by EDA).

**Expected outputs** (written under `outputs/`, generated data):

- Processed, split dataset files (e.g. train/validation/test parquet files — exact names/counts TBD at implementation).
- A **preprocessing metrics / report** artifact summarizing before/after statistics (§6).
- An optional record (log) of any rows excluded by the numeric/table filter, for human review (§2.4).

---

## 5. Invariants preprocessing must preserve

1. **Identity preservation:** `idx` values are preserved; the final split row counts sum exactly to the post-dedup/post-filter total, with **no dropped-unless-recorded** rows and **no overlap** between splits.
2. **NFC idempotency:** applying NFC a second time must change nothing (i.e. output is fully NFC-normalized, including `src`).
3. **Raw immutability:** the raw parquet bytes are unchanged (validate by size/hash before and after).
4. **No semantic loss:** no lowercase / punctuation-removal / stopword removal / stemming / lemmatization / transliteration / NFKC (§2.8).
5. **Deduplication scope:** only exact `(src, tgt)` pairs may be dropped; `src`-only or `tgt`-only duplicates are preserved (§2.2).
6. **Length integrity:** no dataset-level length cap is applied; long legitimate rows are retained (§2.6).

---

## 6. Metrics reported before/after preprocessing

- Row counts (raw → after each step → final per split).
- Duplicate `(src, tgt)` pair count removed (expected 0 today).
- Non-NFC row counts for `src` and `tgt` (expected 0 on output).
- Numeric-fraction `> 0.35` candidate counts for `src` and `tgt` (≈214 / ≈176 on raw) and how many were actually removed, **with a sample of the excluded rows logged for review**.
- Length distribution stats (tokens/chars, p95/p99/p99.5/p99.9/max) and length-ratio distribution, before vs after.
- Leading-quote-artifact row count affected by cleanup (if the cleanup is implemented).

---

## 7. Reproducibility requirements

- **Fixed RNG seed** for every shuffle and for the split step.
- The **seed** is recorded in the preprocessing report.
- A **hash (and size) of the raw parquet** is recorded in the report.
- Deterministic, ordered processing so re-running on the same raw input yields byte-identical outputs (or identical reported stats/metrics).

---

## 8. Error handling expectations

- **Schema mismatch** → hard fail with a clear message.
- **Unexpected null/empty/control-character text** after validation → warn and log; fail if it violates a defined invariant.
- **NFC** → never errors; always succeeds (pure canonical transformation).
- **Numeric/table filter** → conservative: log the full set of rows that would be dropped and do **not** proceed to save if the excluded set contains evident legitimate sentences (i.e. validate intended behavior first, §2.4).
- Any unexpected failure must abort rather than silently produce a partial/corrupt processed dataset.

---

## 9. Provenance appendix

| Decision | Label | Source |
|---|---|---|
| NFC on `src`+`tgt`; no NFKC | [EDA] + [1A-inv] | EDA notebook cells 41–42; 1A Unicode validation |
| Dedup exact `(src,tgt)` only; keep dup `src`/`tgt`; 0 dup pairs | [EDA] | EDA notebook cells 21–24 |
| Broad suspicious mask not a deletion filter; ~94% FPs | [EDA] + [1A-inv] | EDA cells 53–56; 1A categorization |
| Numeric/table noise `>0.35` (exact defn.) | [1A-inv] | **temporary 1A script (deleted) — NOT the notebook** |
| Leading quote artifact cleanup (only) | [1A-inv] | temporary 1A script sampling |
| No dataset-level length cap | [EDA] + [1A-inv] | EDA cells 28–29, 38, 51; 1A tail inspection |
| Retain URL / UI / format / localization pairs | [1A-inv] + [decision] | 1A categorization; adopted decision |
| No semantic preprocessing | [decision] | Adopted for translation task |

---

*End of Phase 1B preprocessing specification. No code, tokenizer, vocabulary, Transformer, split-percentage, or Graphify changes are made by this document.*
