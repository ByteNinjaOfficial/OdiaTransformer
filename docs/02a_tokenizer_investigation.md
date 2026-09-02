# Phase 2A — Tokenizer Investigation Design

**Project:** Test-7 / OdiaTransformer — English → Odia translation, Transformer implemented from scratch.
**Phase:** 2A — Tokenizer investigation / design specification.
**Status:** Design/documentation only. **No tokenizer is installed, trained, or selected yet.**

---

## 0. Provenance legend

Every decision below is labeled with its source of truth:

- **[AGENTS]** — a hard constraint or requirement from `AGENTS.md`.
- **[Phase 1]** — a decision established by the Phase 1C pipeline (`docs/01c_preprocessing_report.md`) or its authoritative spec (`docs/01_preprocessing_spec.md`).
- **[approved]** — a decision approved for Phase 2A by the user.
- **[Phase 2B]** — planned implementation work, not done by this document.

---

## 1. Scope & constraints

1. This document **formally captures the tokenizer investigation plan and the approved Phase 2A decisions** only. It does **not** implement, train, or select a tokenizer.
2. **No package is installed** by this phase. SentencePiece is the chosen library **[approved]** but is not installed here.
3. `requirements.txt` is **not modified** by this phase.
4. `src/tokenizer.py`, `tests/test_tokenization.py`, and all tokenizer model/vocab artifacts are **not created** in this phase.
5. `outputs/`, `src/preprocessing.py`, and the Phase 1 preprocessing tests are **not modified**.
6. The **raw dataset remains immutable** and the **processed splits are not modified** **[Phase 1]**.

---

## 2. Phase 1 context & data available to Phase 2

Phase 1 completed: dataset EDA, preprocessing specification, preprocessing implementation, verification, and Graphify regeneration.

The preprocessing pipeline produced the processed, split corpus **[Phase 1]**:

| File | Rows | Columns |
|------|------|---------|
| `outputs/train.parquet` | 975,020 | `idx`, `src`, `tgt` |
| `outputs/val.parquet` | 9,949 | `idx`, `src`, `tgt` |
| `outputs/test.parquet` | 9,949 | `idx`, `src`, `tgt` |

Relevant Phase 1 preprocessing decisions carried into Phase 2 **[Phase 1]**:

- NFC normalization applied (both `src` and `tgt`); **no NFKC**.
- **No lowercasing**, no punctuation removal, no stopword removal, no stemming/lemmatization, no transliteration.
- No dataset-level hard sentence-length cap; long sentences are retained.
- Deterministic train/validation/test split, **seed 42**.

Consequence for Phase 2: the text fed to tokenization is already NFC-normalized and otherwise untouched by aggressive transformation.

---

## 3. AGENTS.md constraints (preserved)

- The Transformer must be implemented **from scratch**. Libraries are permitted only for: **dataset handling, subword tokenization (BPE), BLEU scoring, and PyTorch tensor ops** **[AGENTS]**.
- **Odia is morphologically rich** and requires **subword tokenization** **[AGENTS]**.
- Architecture constraints to be supported by the tokenizer choice **[AGENTS]**:
  - `d_model = 128`, `heads = 4`, `N = 2` encoder blocks, `N = 2` decoder blocks.
  - Embedding layer + sinusoidal positional encoding.
  - Cross-entropy loss **ignoring padding tokens**.
  - Teacher forcing.
  - Adam + warmup.
  - Padding masks.
  - Causal/look-ahead masks.
  - Greedy decoding; beam search is an optional bonus.
- No architecture requirements are invented beyond what AGENTS.md supports.

The tokenizer choice must produce token **IDs**, vocab sizes, and special-token IDs that plug into this architecture (embeddings of shape `vocab × d_model`, positional encoding, output projection `d_model → vocab`) **[AGENTS]**.

---

## 4. Approved decisions

### 4.1 Tokenization library **[approved]**

**SentencePiece only.** HuggingFace `tokenizers` is **not** evaluated.

SentencePiece is selected because it:
- supports BPE,
- trains **directly on raw text**,
- provides **self-contained tokenizer model files**,
- supports vocabulary persistence and `token ↔ ID` lookup,
- avoids a dependency on the HuggingFace ecosystem (consistent with AGENTS.md's no-pretrained-model stance).

### 4.2 Tokenization algorithm **[approved]**

**BPE only.** Unigram is **not** evaluated. The investigation evaluates SentencePiece BPE configurations only.

### 4.3 Input text handling **[approved]**

Feed the **already-preprocessed NFC-normalized text directly into SentencePiece**. Do **not**:
- apply additional Unicode normalization,
- pre-tokenize by whitespace,
- apply Unicode-aware word-boundary preprocessing,
- lowercase, remove punctuation, remove stopwords, stem, lemmatize, or transliterate.

SentencePiece receives the processed raw sentence strings directly.

### 4.4 No character fallback **[approved]**

There is **no character-level fallback** for unknown tokens. Unknown-token behavior is measured directly. If UNK behavior is poor, vocabulary/configuration quality is reconsidered rather than silently falling back to characters.

### 4.5 Sequence length policy **[approved]**

**No max sequence length and no truncation** are imposed during the tokenizer investigation. The purpose of the investigation is to measure the **full real distribution** (see §7 metrics). Any future `max_seq_len` or truncation decision belongs to the later dataloader/training design phase — this document does **not** recommend a sequence cap as an approved decision.

---

## 5. Vocabulary learning rules (data-leakage prevention) **[approved]**

Vocabulary learning must use **only `outputs/train.parquet`**.

- Validation and test text must **never** contribute to tokenizer training.
- Validation and test splits are **evaluation-only** and are used later to measure:
  - UNK rate,
  - fragmentation,
  - sequence-length distributions,
  - coverage / domain shift.

This is an explicit **data-leakage prevention** requirement. Training tokenizer vocabularies on validation or test text would inflate quality estimates for the target splits.

---

## 6. Special token ID contract **[approved]**

The following IDs are **mandatory** and treated as an explicit contract:

| Token | ID |
|-------|----|
| `<PAD>` | 0 |
| `<SOS>` | 1 |
| `<EOS>` | 2 |
| `<UNK>` | 3 |

Requirements:

- Special tokens are **reserved** rather than learned through BPE merges.
- The implementation must **NOT assume SentencePiece automatically assigns the desired IDs**.
- Phase 2B must **explicitly configure and verify** the IDs after tokenizer training.
- The tokenizer model must **persist** the vocabulary and the IDs.
- **Token → ID** and **ID → token** mappings must be **loadable for inference**.
- Phase 2B tests must **verify all four IDs**.

---

## 7. Tokenizer architecture investigation

Both architectures are evaluated; neither has automatic preference **[approved]**.

### 7.1 Separate tokenizers

One SentencePiece BPE tokenizer per language:

- **English BPE tokenizer** → encoder input.
- **Odia BPE tokenizer** → decoder input/output.

Candidate vocabulary sizes:

| Language | Candidate vocab sizes |
|----------|-----------------------|
| English | 4K, 8K, 12K, 16K |
| Odia | 8K, 16K, 24K, 32K |

This produces **4 × 4 = 16 separate-tokenizer configurations**.

### 7.2 Shared tokenizer

One SentencePiece BPE tokenizer trained on **both** English and Odia training text.

Candidate vocabulary sizes: **16K, 24K, 32K, 40K**.

This produces **4 shared-tokenizer configurations**.

### 7.3 Total experiment matrix

| Mode | Configurations |
|------|----------------|
| Separate | 16 |
| Shared | 4 |
| **Total** | **20** |

The Phase 2B matrix must contain all 20 configurations; this matrix is **not reduced** during Phase 2A documentation.

---

## 8. Metrics to collect in Phase 2B

Measured on the **train split** for every configuration, and additionally evaluated on **validation/test** (UNK rate, fragmentation, sequence-length distributions, coverage) without training on them.

### 8.1 Vocabulary metrics
- Vocabulary size.
- Vocabulary utilization (fraction of vocabulary tokens seen ≥ 1 time in the corpus). **Diagnostic only — no hard utilization threshold is defined.**
- **UNK rate** (% of tokens mapped to `<UNK>`).

### 8.2 Fragmentation metrics
- Average subwords per whitespace word.
- Average subwords per sentence.
- English fragmentation (avg subwords per English whitespace word).
- Odia fragmentation (avg subwords per Odia whitespace word).

### 8.3 Sequence metrics (no truncation)
- Tokenized sequence length **p50, p95, p99, maximum**.

### 8.4 Subword piece metrics
- Character-length distribution of individual subword pieces: **p50, p95, p99**.

### 8.5 Fidelity
- **Encode → decode roundtrip.** The decoded text should match the input text according to the tokenizer's supported normalization behavior.
- Because Phase 1 text is already NFC-normalized, roundtrip comparisons are performed against that **preprocessed text**.

### 8.6 Qualitative inspection
Inspect **approximately 5–10 representative Odia examples**. These examples are intended to inspect **whether segmentation appears linguistically reasonable for morphologically rich Odia**, rather than relying only on numeric metrics.

---

## 9. Vocabulary-related Transformer parameter cost (d_model = 128)

Report **approximate vocabulary-related parameter cost** for each candidate configuration, using `d_model = 128` **[AGENTS]**. This is **vocabulary-related parameter cost only**, not the total Transformer parameter count.

### 9.1 Separate tokenizers
| Component | Formula |
|-----------|---------|
| Encoder embedding | `vocab_en × d_model` |
| Decoder embedding | `vocab_or × d_model` |
| Decoder output projection | `vocab_or × d_model` |
| **Total** | `(vocab_en + 2 × vocab_or) × 128` |

### 9.2 Shared tokenizer
| Component | Formula |
|-----------|---------|
| Shared embedding | `vocab_shared × d_model` |
| Decoder output projection | `vocab_shared × d_model` |
| **Total** | `2 × vocab_shared × 128` |

---

## 10. Recommendation criteria (no final selection in Phase 2A)

Phase 2A does **not** choose a final tokenizer. The following criteria are what Phase 2B will use to compare configurations, in priority order:

1. **Correct reproducibility.**
2. **Correct special-token ID contract** (`<PAD>`=0, `<SOS>`=1, `<EOS>`=2, `<UNK>`=3).
3. **Train-only vocabulary learning** (no validation/test leakage into vocab).
4. **Good validation/test coverage and low UNK behavior.**
5. **Practical fragmentation.**
6. **Strong Odia segmentation quality.**
7. **Full sequence-length distributions measured without truncation.**
8. **Reasonable vocabulary-related parameter cost for class compute.**

There is **no automatic preference for shared tokenization** — separate and shared tokenizers are evaluated on actual results. There is **no hard vocabulary-utilization threshold** (utilization is diagnostic only).

---

## 11. Expected Phase 2B artifacts

These are **planned** Phase 2B outputs; they are **not created** in Phase 2A.

### 11.1 `src/tokenizer.py`
Responsibilities:
- SentencePiece BPE training.
- Tokenizer loading.
- Tokenizer saving.
- Encoding.
- Decoding.
- Metrics collection.
- Special-token ID verification.
- Deterministic/reproducible configuration handling.

### 11.2 `tests/test_tokenization.py`
Expected tests:
- Encode/decode roundtrip.
- Special token IDs.
- `<PAD>` handling.
- `<SOS>` handling.
- `<EOS>` handling.
- UNK behavior.
- Explicit ID contract verification.

### 11.3 Experiment report
- `docs/02b_tokenizer_experiments.md` — full results across the 20-configuration matrix.

### 11.4 Generated tokenizer artifacts (must remain gitignored)
- `outputs/tokenizer_<config>.model`
- `outputs/tokenizer_<config>.vocab`

These generated artifacts are **not created** during Phase 2A and must remain gitignored.

---

## 12. Reproducibility requirements

Every experiment must be identifiable and reproducible. Record:

- **Train split only** for tokenizer vocabulary learning.
- Deterministic training configuration where supported.
- Fixed seed recorded.
- Exact configuration recorded.
- Vocabulary size recorded.
- Algorithm recorded (**BPE**).
- Architecture mode recorded (**separate** or **shared**).
- Source dataset provenance recorded (which parquet file(s) and split).
- Evaluation metrics recorded consistently.

---

## 13. Phase boundary

### Phase 2A Status
**Completed** by creating this investigation/design specification (`docs/02a_tokenizer_investigation.md`).

### Phase 2B Status
**Not started** by this task. Phase 2B will:
1. Install SentencePiece.
2. Implement `src/tokenizer.py`.
3. Implement `tests/test_tokenization.py`.
4. Train all 20 configurations.
5. Evaluate train/validation/test metrics.
6. Inspect representative Odia segmentation.
7. Produce `docs/02b_tokenizer_experiments.md`.
8. Recommend the best tokenizer configuration.
9. Wait for approval before moving into the next Transformer stage.

---

*End of Phase 2A tokenizer investigation design. No package installed, no tokenizer trained, no configuration selected, no code or data modified by this document.*
