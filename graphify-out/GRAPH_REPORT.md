# Graph Report - Test-7  (2026-09-01)

## Corpus Check
- Corpus is ~11,732 words - fits in a single context window. You may not need a graph.

## Summary
- 134 nodes · 190 edges · 24 communities (9 shown, 15 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Preprocessing Core Module
- Conservative Numeric Filter Tests
- Core Test Classes (NFC, Schema, etc.)
- Preprocessing Specification Concepts
- Statistics & Reporting Functions
- Numeric Filter Implementation
- Leading Quote Cleanup Tests
- Schema Validation Tests
- Requirements Dependencies
- Dataset Info & HuggingFace URL
- Run Pipeline Script
- Verify Pipeline Script
- AGENTS.md Workflow
- CQRS SOLID DRY Principles
- Graphify Dependency
- Mandatory Workflow Order
- OdiaTransformer README Reference
- Phase 1C Report Results
- Preprocessing Report Document
- Apache License Concept
- License Document
- OdiaTransformer README Concept

## God Nodes (most connected - your core abstractions)
1. `make_df()` - 23 edges
2. `pipeline()` - 22 edges
3. `_str_col()` - 11 edges
4. `Preprocessing Specification` - 9 edges
5. `numeric_noise_filter()` - 7 edges
6. `normalize_nfc()` - 6 edges
7. `TestLeadingQuoteCleanup` - 6 edges
8. `TestConservativeNumericFilter` - 6 edges
9. `TestSplit` - 6 edges
10. `validate_text()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Numeric Noise Filter` --semantically_similar_to--> `7 Malformed Tables Excluded`  [INFERRED] [semantically similar]
  docs/01_preprocessing_spec.md → docs/01c_preprocessing_report.md
- `Conservative Exclusion Rule` --semantically_similar_to--> `100 False Positives Fixed`  [INFERRED] [semantically similar]
  docs/01_preprocessing_spec.md → docs/01c_preprocessing_report.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Preprocessing Pipeline Steps** — docs_01_preprocessing_spec_nfc_normalization, docs_01_preprocessing_spec_leading_quote_cleanup, docs_01_preprocessing_spec_exact_pair_dedup, docs_01_preprocessing_spec_numeric_noise_filter, docs_01_preprocessing_spec_conservative_exclusion, docs_01_preprocessing_spec_source_token_floor, docs_01_preprocessing_spec_split_config [INFERRED 0.75]

## Communities (24 total, 15 thin omitted)

### Community 0 - "Preprocessing Core Module"
Cohesion: 0.12
Nodes (31): DataFrame, Path, apply_cleanup(), apply_nfc(), build_report(), clean_source_leading_quote(), deduplicate_pairs(), final_validation() (+23 more)

### Community 1 - "Conservative Numeric Filter Tests"
Cohesion: 0.14
Nodes (6): make_df(), Build a DataFrame with the expected schema from a list of (idx, src, tgt)., TestConservativeNumericFilter, TestDedup, TestSplit, TestTextValidation

### Community 2 - "Core Test Classes (NFC, Schema, etc.)"
Cohesion: 0.11
Nodes (6): Unit tests for src/preprocessing.py (Phase 1C). Run from the repository root:…, TestNFC, TestNoLengthFilter, TestNumericDefinition, TestPipelineEndToEnd, TestRawImmutability

### Community 3 - "Preprocessing Specification Concepts"
Cohesion: 0.17
Nodes (12): Conservative Exclusion Rule, Exact Pair Deduplication, Leading Triple-Quote Cleanup, NFC Normalization, No Semantic Preprocessing, Numeric Noise Filter, Preprocessing Specification, Raw Input Immutability (+4 more)

### Community 4 - "Statistics & Reporting Functions"
Cohesion: 0.31
Nodes (9): Series, count_non_nfc(), length_ratio_stats(), length_stats(), Count values whose NFC form differs (i.e. that will change under NFC)., Whitespace-token and character length summary for a string column., Odia-character / English-character length-ratio summary., Return a series of strings, mapping NaN to '' (never used for discard). (+1 more)

### Community 5 - "Numeric Filter Implementation"
Cohesion: 0.29
Nodes (8): compute_numeric_fraction(), _is_table_fragment(), numeric_candidate(), numeric_noise_filter(), Numeric-token fraction per the authoritative 1A definition. tokens = s.split()…, Strict `> threshold` candidate test, evaluated independently per side. A row is…, Conservative exclusion test: is this candidate a malformed table fragment? A…, Identify and remove malformed numeric/table fragments. Returns (filtered_frame,…

### Community 8 - "Requirements Dependencies"
Cohesion: 0.50
Nodes (4): numpy, pandas, pyarrow, Requirements

## Knowledge Gaps
- **24 isolated node(s):** `AGENTS.md Workflow`, `CQRS SOLID DRY Principles`, `Mandatory Workflow Order`, `Graphify Graph Regeneration`, `Apache License 2.0` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_df()` connect `Conservative Numeric Filter Tests` to `Core Test Classes (NFC, Schema, etc.)`, `Leading Quote Cleanup Tests`, `Schema Validation Tests`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `TestLeadingQuoteCleanup` connect `Leading Quote Cleanup Tests` to `Core Test Classes (NFC, Schema, etc.)`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `pipeline()` connect `Preprocessing Core Module` to `Statistics & Reporting Functions`, `Numeric Filter Implementation`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `Preprocessing Specification` (e.g. with `Conservative Exclusion Rule` and `Exact Pair Deduplication`) actually correct?**
  _`Preprocessing Specification` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `AGENTS.md Workflow`, `CQRS SOLID DRY Principles`, `Mandatory Workflow Order` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Preprocessing Core Module` be split into smaller, more focused modules?**
  _Cohesion score 0.11895161290322581 - nodes in this community are weakly interconnected._
- **Should `Conservative Numeric Filter Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.13852813852813853 - nodes in this community are weakly interconnected._