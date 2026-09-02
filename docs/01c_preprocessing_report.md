# Phase 1C — Preprocessing Implementation Report

**Project:** OdiaTransformer (English → Odia, from-scratch Transformer course assignment)
**Phase:** 1C — Preprocessing module implementation
**Date:** 2026-09-01
**Git:** branch `main` @ `9ff88fe` (Initial commit); no commit/push performed this phase.

---

## 1. Files created / modified

| Path | Action | Purpose |
|------|--------|---------|
| `src/preprocessing.py` | created | Phase 1C implementation module (~610 lines) |
| `tests/test_preprocessing.py` | created | Unit test suite, stdlib `unittest` (36 tests) |
| `run_pipeline.py` | created | Reproducible entry point for the real-parquet run |
| `verify_pipeline.py` | created | Independent post-run verification of outputs |
| `outputs/` | created (gitignored) | Where the processed parquets + report are written |
| `docs/01c_preprocessing_report.md` | created | This report |
| `docs/01_preprocessing_spec.md` | (unchanged) | Authoritative design (Phase 1B) |

Raw input **`dataset/train-00000-of-00001.parquet`** was read-only; never modified.

---

## 2. Executed pipeline (order per spec)

`raw parquet` → schema validation → text validation → NFC normalization → conservative cleanup (leading `"""` artifact) → exact `(src,tgt)` dedup → narrow numeric/table filter → final validation → deterministic split → save outputs + report.

---

## 3. Results on the real Samanantar Odia corpus (998,228 rows)

### Row-count accounting

| Step | Rows | Delta |
|------|------|-------|
| raw | 998,228 | – |
| after NFC | 998,228 | 0 |
| after cleanup | 998,228 | 0 (6795 src rows had leading-`"""` artifact stripped, none dropped) |
| after dedup | 994,925 | **−3,303** exact duplicate `(src,tgt)` pairs |
| after numeric filter | 994,918 | **−7** malformed statistical-table fragments |
| **final** | **994,918** | total removed **3,310** |

Removal accounting: `998,228 − 3,303 − 7 = 994,918` ✓

### NFC normalization

- `src`: 0 non-NFC rows before → 0 after.
- `tgt`: **116,794** non-NFC rows before → **0** after.

This matches the EDA finding exactly (116,794 OR non-NFC). Only NFC (never NFKC) is applied.

### Leading-triple-quote cleanup (`src` only)

- **6,795** rows had a leading `"""` artifact removed at the source token boundary.
- Conservative: single `"` and double `""` leading patterns (legitimate nested quotes) are **preserved**; no rows deleted.

### Duplicate handling

- **3,303** exact `(src,tgt)` duplicates removed (keep-first).
- Duplicate `src` (multiple valid translations) and duplicate `tgt` are preserved.

### Numeric/table noise filter (refined, conservative)

- Candidates (numeric fraction **strictly `> 0.35`** on either side): **283**.
- **Excluded: 7** — only rows that are numeric-heavy on **both** sides *and* have **≥ 20 source tokens** (a genuine malformed statistical table block).

The 7 excluded are unambiguous spreadsheet/statistical table fragments (seafarer rank tables, insolvency ranks, flight schedule tables, wheat-procurement tables, government-vacancy/roll-number tables, ration-kit tables). All 100 short numeric-heavy legitimate translations (e.g. “13 places”→“୧୩ ସ୍ଥାନଗୁଡ଼ିକ”, “Version 1”, “2 Weeks”, “6 killed, 4 critically injured”) are **retained**.

### Split (configurable, default 98/1/1, seed 42)

| Split | Rows |
|-------|------|
| train | 975,020 |
| val | 9,949 |
| test | 9,949 |
| **total** | **994,918** (accounting exact) |

Seed 42, deterministic. **No overlap** across splits (verified on both `idx` and full `(src,tgt)` pairs).

### Output files (`outputs/`)

- `train.parquet`, `val.parquet`, `test.parquet` (columns `idx, src, tgt`)
- `numeric_excluded_samples.parquet` (the 7 excluded rows, for human review)
- `preprocessing_report.json` (full machine-readable report)

---

## 4. Raw-input immutability

- SHA-256 before: `4fd32b16a78e907a5e48f7f6a9ee34e508c3d3a95ee69bd16b16d85d62f8a2ee`
- SHA-256 after : `4fd32b16a78e907a5e48f7f6a9ee34e508c3d3a95ee69bd16b16d85d62f8a2ee`
- **Raw file unchanged** (`raw_immutable = true`).

---

## 5. Tests

`tests/test_preprocessing.py` — stdlib `unittest`, **36 tests, all OK**, covering:

- schema validation (accept correct; reject wrong order / missing col / non-integer `idx`)
- text validation (failures on NaN / whitespace-only)
- NFC (normalizes src+tgt, idempotent, no NFKC compat mapping, non-NFC counting)
- leading-quote cleanup (strips `"""`, preserves `"` and `""` nested quotes, touches only src)
- dedup (removes exact pair; preserves dup src / dup tgt)
- numeric definition (`replace('.','').replace(',','').isdigit()`), strict `>0.35` (7/20 = 0.35 is NOT a candidate), fractions
- conservative filter (long both-sides table excluded; short both-sides “13 places” kept; src-only numeric kept)
- no length filter (500-token sentences retained)
- split (exact accounting, no overlap, deterministic, seed-sensitive, invalid-fraction rejection)
- raw immutability (SHA-256 stable / detects change)
- end-to-end synthetic pipeline (immutability, NFC, cleanup, dedup, exclusion, accounting, split, outputs written)

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v` (pandas/pyarrow/numpy only; no new dependencies).

---

## 6. Key interpretation decisions

1. **Split = configurable, default 98/1/1, seed 42** — user-confirmed (spec left split undecided).
2. **Conservative numeric filter refined with a source-token floor (≥ 20 tokens)** — the initial both-sides `>0.35` rule matched the spec text but generated 100 false-positive deletions of legitimate short translations. User-approved fix: a row is excluded only when numeric-heavy on both sides **and** the source is ≥ 20 tokens (a genuine table block). Result: exactly the 7 real malformed tables removed, zero legitimate sentences lost.
3. No semantic preprocessing — no lowercase, no punctuation removal, no stopwords, no stem/lemma, no transliteration, no NFKC, **no dataset-level length cap** (retain long legal/scripture sentences).

---

## 7. Issues / concerns

- The `>0.35` threshold and both-sides rule are inherently ambiguous for short rows; the token floor resolves this cleanly but is corpus-specific. If the corpus or intent changes, it should be re-validated.
- The 7 excluded tables are blocks of dense statistics that are poor MT training data; keeping the partial row fragments (`West Bengal 10,126 7.`) was intentional (conservative — not clearly malformed).

---

## 8. Git status (end of phase)

```
 M .gitignore
 M requirements.txt
?? AGENTS.md, dataset/, docs/, graphify-out/, notebooks/, outputs/
?? run_pipeline.py, src/, tests/, verify_pipeline.py
```

Nothing committed or pushed. Per `AGENTS.md`, the **graphify graph is NOT regenerated** until implementation + tests pass — both now do. Ready for the graphify update and the Phase 2 tokenizer work.
