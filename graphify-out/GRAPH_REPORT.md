# Graph Report - Test-7  (2026-09-02)

## Corpus Check
- Corpus is ~16,902 words - fits in a single context window. You may not need a graph.

## Summary
- 284 nodes · 485 edges · 9 communities (6 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Tokenizer Implementation & Orchestration
- Tokenizer Unit Tests
- Preprocessing Pipeline
- Preprocessing Unit Tests
- Tokenizer & Project Documentation
- Dataset & Corpus (Samanantar)
- Tokenizer Config Validation Tests
- Token Encoding & SOS/EOS
- Project License

## God Nodes (most connected - your core abstractions)
1. `make_df()` - 23 edges
2. `pipeline()` - 22 edges
3. `Tokenizer` - 22 edges
4. `Phase 2A Tokenizer Investigation Design` - 19 edges
5. `Phase 2B Tokenizer Experiments Report` - 18 edges
6. `run_single_experiment()` - 15 edges
7. `_train_shared_synthetic()` - 14 edges
8. `TokenizerConfig` - 13 edges
9. `_str_col()` - 11 edges
10. `evaluate_split()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Train-Only Vocabulary Learning (Leakage Prevention)` --references--> `build_separate_corpus_files()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `Train-Only Vocabulary Learning (Leakage Prevention)` --references--> `build_shared_corpus_file()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `BPE Tokenization` --references--> `train_bpe()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `Separate Tokenizer Architecture` --references--> `train_separate()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `From-Scratch Transformer (d_model=128, heads=4, N=2)` --conceptually_related_to--> `Vocabulary-Related Parameter Cost (d_model=128)`  [INFERRED]
  AGENTS.md → docs/02a_tokenizer_investigation.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Samanantar Parallel Corpus and Multilingual NMT Training** — dataset_readme_samanantar, dataset_readme_indic_languages, dataset_readme_english, dataset_readme_nmt_models, dataset_readme_flores [INFERRED 0.85]
- **Tokenizer Configuration Selection Decision** — concept_recommended_tokenizer_config, concept_separate_tokenizer, concept_shared_tokenizer, concept_parameter_cost, concept_fragmentation_metrics [INFERRED 0.85]
- **Preprocessing Pipeline Execution** — concept_preprocessing_pipeline, concept_nfc_normalization, concept_phase1_deduplication, concept_phase1_numeric_filter, concept_raw_immutability [EXTRACTED 1.00]
- **Vocabulary Training Framework** — rationale_train_only_vocab, concept_special_token_contract, concept_bpe_tokenization, concept_sentencepiece [INFERRED 0.85]

## Communities (9 total, 3 thin omitted)

### Community 0 - "Tokenizer Implementation & Orchestration"
Cohesion: 0.05
Nodes (55): build_separate_corpus_files(), build_shared_corpus_file(), collect_piece_lengths(), compute_fragmentation(), compute_parameter_cost(), compute_piece_lengths(), compute_roundtrip_fidelity(), compute_sequence_lengths() (+47 more)

### Community 1 - "Tokenizer Unit Tests"
Cohesion: 0.06
Nodes (24): _make_synthetic_train_parquet(), Path, Unit tests for src/tokenizer.py (Phase 2B). Run from the repository root:…, Encode → decode roundtrip fidelity against preprocessed text., SOS/EOS IDs are correctly reserved and available., UNK token behavior for out-of-vocabulary content., Explicit special-ID verification function / constructor behavior., Tokenizer.__init__ should raise if special IDs don't match. (+16 more)

### Community 2 - "Preprocessing Pipeline"
Cohesion: 0.07
Nodes (50): Run the Phase 1C preprocessing pipeline against the real Samanantar Odia…, Series, apply_cleanup(), apply_nfc(), build_report(), clean_source_leading_quote(), compute_numeric_fraction(), count_non_nfc() (+42 more)

### Community 3 - "Preprocessing Unit Tests"
Cohesion: 0.06
Nodes (14): make_df(), Unit tests for src/preprocessing.py (Phase 1C). Run from the repository root:…, Build a DataFrame with the expected schema from a list of (idx, src, tgt)., TestConservativeNumericFilter, TestDedup, TestLeadingQuoteCleanup, TestNFC, TestNoLengthFilter (+6 more)

### Community 4 - "Tokenizer & Project Documentation"
Cohesion: 0.12
Nodes (33): BPE Tokenization, Fragmentation Metrics (Subwords per Word), From-Scratch Transformer (d_model=128, heads=4, N=2), NFC Normalization, Odia Morphological Richness / Subword Tokenization, Vocabulary-Related Parameter Cost (d_model=128), Phase 1 Exact Pair Deduplication, Phase 1 Numeric/Table Noise Filter (>0.35 threshold) (+25 more)

### Community 5 - "Dataset & Corpus (Samanantar)"
Cohesion: 0.25
Nodes (9): ai4bharat/samanantar (Hugging Face Dataset), AI4Bharat, CC BY-NC 4.0 License, English Language, FLORES Benchmark, 11 Indic Languages, Multilingual NMT Models, Samanantar Dataset (+1 more)

## Knowledge Gaps
- **9 isolated node(s):** `Apache License 2.0`, `BYTENINJA (Copyright Holder)`, `Samanantar Paper (TACL 2022)`, `CC BY-NC 4.0 License`, `11 Indic Languages` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tokenizer` connect `Tokenizer Implementation & Orchestration` to `Tokenizer & Project Documentation`, `Token Encoding & SOS/EOS`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `Train-Only Vocabulary Learning (Leakage Prevention)` connect `Tokenizer & Project Documentation` to `Tokenizer Implementation & Orchestration`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `Apache License 2.0`, `BYTENINJA (Copyright Holder)`, `Samanantar Paper (TACL 2022)` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Tokenizer Implementation & Orchestration` be split into smaller, more focused modules?**
  _Cohesion score 0.05487269534679543 - nodes in this community are weakly interconnected._
- **Should `Tokenizer Unit Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05519480519480519 - nodes in this community are weakly interconnected._
- **Should `Preprocessing Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07256894049346879 - nodes in this community are weakly interconnected._
- **Should `Preprocessing Unit Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05568627450980392 - nodes in this community are weakly interconnected._