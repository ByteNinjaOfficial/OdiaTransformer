# Graph Report - Test-7  (2026-09-09)

## Corpus Check
- 59 files · ~67,314 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 712 nodes · 1275 edges · 47 communities (41 shown, 6 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 189 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Inference & Evaluation CLI
- Preprocessing Pipeline
- Preprocessing Unit Tests
- Transformer & Data Concepts
- Training Optimizer & Checkpointing
- ExperimentManager (Phase 7A)
- Project & Tokenizer Docs
- Tokenizer Compute & Metrics
- Benchmark Engine & Telemetry
- Tokenizer Training & Config
- Tokenizer Experiment Orchestration
- Transformer Sub-Layers
- Tokenizer Corpus & Loading
- Dataset & Mask Generation
- Collation & TranslationSample
- Trainer Orchestration
- Scaled Dot-Product Attention
- Transformer Encoder/Decoder Layers
- TranslationDataset & Integrations
- Dataset Unit Tests
- TranslationBatch & Validation
- Reproducibility Metadata (Phase 7A)
- Train CLI & Tests
- Training Benchmark CLI
- Positional Encoding
- Token-only Corpus Tests
- Samanantar Dataset Card
- DataLoader Factory & Verify
- TrainingConfig
- Multi-Head Attention
- Benchmark Unit Tests
- Special Token Contract Tests
- SOS/EOS Availability Tests
- Checkpoint Resume Tests
- Parameter Cost Tests
- Encode/Decode Roundtrip Tests
- UNK Behavior Tests
- Special ID Verification Tests
- TranslationDataset Init
- Transformer Encoder
- Transformer Decoder
- Tokenizer Config Validation Tests
- Padding Handling Tests
- Training Plot Script
- Experiment Results IO
- Project License
- Dynamic Padding Pipeline

## God Nodes (most connected - your core abstractions)
1. `TranslationTransformer` - 41 edges
2. `Trainer` - 25 edges
3. `make_df()` - 23 edges
4. `TrainingConfig` - 23 edges
5. `Tokenizer` - 22 edges
6. `pipeline()` - 22 edges
7. `ExperimentManager` - 22 edges
8. `Phase 2A Tokenizer Investigation Design` - 19 edges
9. `Phase 2B Tokenizer Experiments Report` - 18 edges
10. `TranslationDataset` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Train-Only Vocabulary Learning (Leakage Prevention)` --references--> `build_separate_corpus_files()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `Train-Only Vocabulary Learning (Leakage Prevention)` --references--> `build_shared_corpus_file()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `BPE Tokenization` --references--> `train_bpe()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `Separate Tokenizer Architecture` --references--> `train_separate()`  [EXTRACTED]
  docs/02a_tokenizer_investigation.md → src/tokenizer.py
- `main()` --uses--> `TrainingBenchmark`  [INFERRED]
  benchmark_training.py → src/benchmark.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Transformer Encoder Stack Components** — docs_04_transformer_architecture_sinusoidal_pe, docs_04_transformer_architecture_multi_head_attention, docs_04_transformer_architecture_position_wise_ffn, docs_04_transformer_architecture_residual_layernorm [INFERRED 0.85]
- **Training Optimization Pipeline Components** — docs_05_training_pipeline_adamw_optimizer, docs_05_training_pipeline_noam_warmup, docs_05_training_pipeline_cross_entropy_loss, docs_05_training_pipeline_amp [INFERRED 0.85]
- **End-to-End Evaluation Pipeline** — docs_06_training_inference_evaluation_greedy_decoder, docs_06_training_inference_evaluation_corpus_evaluation, docs_07b_evaluation_results_test_metrics, docs_07b_evaluation_results_qualitative_samples [INFERRED 0.75]
- **Preprocessing Pipeline Execution** — concept_preprocessing_pipeline, concept_nfc_normalization, concept_phase1_deduplication, concept_phase1_numeric_filter, concept_raw_immutability [EXTRACTED 1.00]
- **Tokenizer Configuration Selection Decision** — concept_recommended_tokenizer_config, concept_separate_tokenizer, concept_shared_tokenizer, concept_parameter_cost, concept_fragmentation_metrics [INFERRED 0.85]
- **Vocabulary Training Framework** — rationale_train_only_vocab, concept_special_token_contract, concept_bpe_tokenization, concept_sentencepiece [INFERRED 0.85]
- **Samanantar Parallel Corpus and Multilingual NMT Training** — dataset_readme_samanantar, dataset_readme_indic_languages, dataset_readme_english, dataset_readme_nmt_models, dataset_readme_flores [INFERRED 0.85]

## Communities (47 total, 6 thin omitted)

### Community 0 - "Inference & Evaluation CLI"
Cohesion: 0.06
Nodes (33): main(), parse_args(), Namespace, batch_translate(), greedy_decode(), device, Tokenizer, Phase 6: Translation Inference & Greedy Autoregressive Decoding for… (+25 more)

### Community 1 - "Preprocessing Pipeline"
Cohesion: 0.07
Nodes (50): Run the Phase 1C preprocessing pipeline against the real Samanantar Odia…, Series, apply_cleanup(), apply_nfc(), build_report(), clean_source_leading_quote(), compute_numeric_fraction(), count_non_nfc() (+42 more)

### Community 2 - "Preprocessing Unit Tests"
Cohesion: 0.06
Nodes (14): make_df(), Unit tests for src/preprocessing.py (Phase 1C). Run from the repository root:…, Build a DataFrame with the expected schema from a list of (idx, src, tgt)., TestConservativeNumericFilter, TestDedup, TestLeadingQuoteCleanup, TestNFC, TestNoLengthFilter (+6 more)

### Community 3 - "Transformer & Data Concepts"
Cohesion: 0.06
Nodes (43): Autoregressive Decoding, BLEU Evaluation, Causal / Look-ahead Masking, Dataset, DataLoader, Collation & Mask Generation, Attention Mask Generation, TranslationBatch, TranslationDataset, From-Scratch Transformer Architecture (§5.6) (+35 more)

### Community 4 - "Training Optimizer & Checkpointing"
Cohesion: 0.08
Nodes (23): GradScaler, Module, Optimizer, load_checkpoint(), device, Path, Update optimizer param_groups with the current computed learning rate., Advance the step counter by 1 and update optimizer learning rate. (+15 more)

### Community 5 - "ExperimentManager (Phase 7A)"
Cohesion: 0.08
Nodes (17): ExperimentManager, Any, Path, Manages an isolated experiment directory, metadata, configs, and artifacts., Initialize and create experiment directory tree. Args: experiment_name: Name of…, Create directory hierarchy safely., Save configuration dictionary or dataclass to config.json., Load configuration from config.json. (+9 more)

### Community 6 - "Project & Tokenizer Docs"
Cohesion: 0.14
Nodes (31): BPE Tokenization, Fragmentation Metrics (Subwords per Word), From-Scratch Transformer (d_model=128, heads=4, N=2), NFC Normalization, Odia Morphological Richness / Subword Tokenization, Vocabulary-Related Parameter Cost (d_model=128), Phase 1 Exact Pair Deduplication, Phase 1 Numeric/Table Noise Filter (>0.35 threshold) (+23 more)

### Community 7 - "Tokenizer Compute & Metrics"
Cohesion: 0.12
Nodes (10): collect_piece_lengths(), compute_piece_lengths(), Wrapper around a trained SentencePiece model with explicit encode/decode., Verify the hard special-token ID contract., Encode text to subword token IDs. Does NOT add SOS/EOS., Decode token IDs to text., Encode with SOS prepended and EOS appended (for decoder input/target…, Character-length distribution of individual subword pieces (p50, p95, p99). (+2 more)

### Community 8 - "Benchmark Engine & Telemetry"
Cohesion: 0.15
Nodes (11): BenchmarkResult, Any, Run complete controlled training benchmark, validation benchmark, and…, Benchmark validation throughput over a small slice of validation data., Collect GPU memory telemetry from PyTorch CUDA runtime., Mathematically estimate training and validation duration per epoch and total., Encapsulates all benchmark metrics, telemetry, and time estimates., Benchmark engine for evaluating Transformer training throughput and memory. (+3 more)

### Community 9 - "Tokenizer Training & Config"
Cohesion: 0.13
Nodes (15): compute_parameter_cost(), generate_experiment_configs(), Build SentencePieceTrainer arguments respecting the special-token contract., Train a single SentencePiece BPE model., Train separate English and Odia tokenizers. Returns (en_model_path,…, Train a shared tokenizer. Returns model path., Vocabulary-related Transformer parameter count (d_model=128)., Immutable configuration for a tokenizer experiment. (+7 more)

### Community 10 - "Tokenizer Experiment Orchestration"
Cohesion: 0.16
Nodes (14): compute_fragmentation(), compute_roundtrip_fidelity(), compute_sequence_lengths(), compute_unk_rate(), compute_vocab_utilization(), evaluate_split(), Phase 2B tokenizer module for OdiaTransformer. Implements SentencePiece BPE…, Fraction of vocabulary pieces that appear at least once. (+6 more)

### Community 11 - "Transformer Sub-Layers"
Cohesion: 0.13
Nodes (9): Tensor, Compute scaled dot-product attention. Args: query: Tensor of shape [batch_size,…, Perform Multi-Head Attention. Args: query: Tensor of shape [batch_size, q_len,…, Forward pass for position-wise feed-forward. Args: x: Tensor of shape…, Forward pass for an encoder layer. Args: x: Input embeddings/states, shape…, Forward pass for a decoder layer. Args: x: Target representations [batch_size,…, Encode source sequence into continuous representations. Args: src: Source token…, Decode target sequence given encoder memory. Args: tgt_input: Target input… (+1 more)

### Community 12 - "Tokenizer Corpus & Loading"
Cohesion: 0.21
Nodes (16): build_separate_corpus_files(), build_shared_corpus_file(), load_tokenizer(), load_train_corpus(), DataFrame, Path, Build separate English and Odia corpus files from train split only. Returns…, Build a single interleaved corpus file from train split only. Returns total… (+8 more)

### Community 13 - "Dataset & Mask Generation"
Cohesion: 0.20
Nodes (10): create_causal_mask(), create_padding_mask(), create_target_mask(), Tensor, Phase 3 Dataset, DataLoader, Collation, and Mask Pipeline for OdiaTransformer.…, Generate a padding mask where True indicates a valid (non-PAD) token. Args:…, Generate a lower-triangular causal / look-ahead mask for autoregressive…, Generate the combined target mask (padding mask AND causal mask). Args:… (+2 more)

### Community 14 - "Collation & TranslationSample"
Cohesion: 0.20
Nodes (8): collate_translation_samples(), Collate variable-length TranslationSamples into a padded TranslationBatch.…, Individual translation sample containing tokenized sequences and raw strings., TranslationSample, Tests for collate_translation_samples and TranslationBatch generation., Tests for TranslationSample dataclass invariants., TestDynamicCollation, TestTranslationSample

### Community 15 - "Trainer Orchestration"
Cohesion: 0.20
Nodes (9): Any, DataLoader, Orchestrates model training, validation, optimizer updates, AMP, and metrics…, Perform a single training step on a batch. Returns: Tuple of (loss_value,…, Evaluate model on validation loader without gradients. Args: val_loader:…, Run full training and validation loop over configured epochs. Returns: Summary…, Save latest and best checkpoints based on validation loss., Save training history and summary metrics to JSON. (+1 more)

### Community 16 - "Scaled Dot-Product Attention"
Cohesion: 0.16
Nodes (5): Initialize parameters with Xavier uniform for linear projections, ones for…, Scaled Dot-Product Attention: Attention(Q, K, V) = softmax((Q @ K^T) /…, ScaledDotProductAttention, Tests for ScaledDotProductAttention module., TestScaledDotProductAttention

### Community 17 - "Transformer Encoder/Decoder Layers"
Cohesion: 0.20
Nodes (8): PositionwiseFeedForward, Position-wise Feed-Forward Network: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2, Single Transformer Encoder Layer: Input -> Self-MHA -> Dropout -> Add &…, Single Transformer Decoder Layer: 1. Masked Self-Attention (tgt_mask) ->…, TransformerDecoderLayer, TransformerEncoderLayer, Tests for PositionwiseFeedForward, EncoderLayer, and DecoderLayer., TestFeedForwardAndLayers

### Community 18 - "TranslationDataset & Integrations"
Cohesion: 0.22
Nodes (7): Dataset, PyTorch Dataset for English -> Odia translation pairs. Loads sentence pairs…, TranslationDataset, Tests for TranslationDataset creation, indexing, and token flows., Integration test using actual trained tokenizer models and outputs/ splits., TestRealDataSmokeTest, TestTranslationDataset

### Community 19 - "Dataset Unit Tests"
Cohesion: 0.17
Nodes (5): DummyTokenizer, Unit tests for Phase 3 Dataset, DataLoader, Collation, and Mask generation…, Tests for create_dataloader and create_dataloaders factory functions., Lightweight mock tokenizer for deterministic unit testing., TestDataLoaderFactories

### Community 20 - "TranslationBatch & Validation"
Cohesion: 0.21
Nodes (7): device, Collated batch of translation pairs with dynamic padding and attention masks., Move all tensor attributes to target device in-place and return self., TranslationBatch, Evaluate a single batch without gradient computation. Args: batch:…, Tests for Trainer train_step, validate, and parameter updates., TestTrainerExecution

### Community 21 - "Reproducibility Metadata (Phase 7A)"
Cohesion: 0.21
Nodes (8): capture_environment_metadata(), get_git_info(), Phase 7A: Experiment Management and Reproducibility for OdiaTransformer.…, Safely query current Git commit hash and branch name. Returns: Dictionary with…, Capture comprehensive environment, hardware, and reproducibility metadata.…, Test suite for reproducibility metadata extraction and graceful fallbacks., TestReproducibilityMetadata, main()

### Community 22 - "Train CLI & Tests"
Cohesion: 0.24
Nodes (8): Unit tests for train.py CLI parsing and configuration construction., Tests for train.py command-line argument parsing., TestTrainCLI, main(), parse_args(), Namespace, Set deterministic seeds across random, numpy, and torch., set_seed()

### Community 23 - "Training Benchmark CLI"
Cohesion: 0.27
Nodes (7): main(), parse_args(), Namespace, Set deterministic seeds across random, numpy, and torch., set_seed(), Phase 7A: Training Benchmark, GPU Telemetry, Epoch Estimation & Resume…, Phase 5: Training Pipeline, Optimization, Checkpointing & Validation for…

### Community 24 - "Positional Encoding"
Cohesion: 0.29
Nodes (5): Sinusoidal positional encoding as described in Vaswani et al. (2017) §3.5.…, Add scaled positional encodings and apply dropout. Args: x: Input embeddings of…, SinusoidalPositionalEncoding, Tests for SinusoidalPositionalEncoding module., TestSinusoidalPositionalEncoding

### Community 25 - "Token-only Corpus Tests"
Cohesion: 0.27
Nodes (5): _make_synthetic_train_parquet(), Path, Train-only corpus construction — leakage guard., Create a tiny train.parquet with synthetic EN/OR pairs., TestTrainOnlyCorpusConstruction

### Community 26 - "Samanantar Dataset Card"
Cohesion: 0.25
Nodes (9): ai4bharat/samanantar (Hugging Face Dataset), AI4Bharat, CC BY-NC 4.0 License, English Language, FLORES Benchmark, 11 Indic Languages, Multilingual NMT Models, Samanantar Dataset (+1 more)

### Community 27 - "DataLoader Factory & Verify"
Cohesion: 0.25
Nodes (7): create_dataloader(), create_dataloaders(), DataLoader, Create a PyTorch DataLoader wrapping a TranslationDataset with dynamic…, Create train, validation, and test DataLoaders for the English -> Odia…, main(), main()

### Community 28 - "TrainingConfig"
Cohesion: 0.31
Nodes (4): Configuration for model training, optimization, checkpointing, and validation., TrainingConfig, Tests for TrainingConfig dataclass and dynamic device resolution., TestTrainingConfig

### Community 29 - "Multi-Head Attention"
Cohesion: 0.36
Nodes (4): MultiHeadAttention, Multi-Head Attention using separate linear projections for Q, K, V, and O.…, Tests for MultiHeadAttention module., TestMultiHeadAttention

### Community 30 - "Benchmark Unit Tests"
Cohesion: 0.28
Nodes (4): Dataset, Generates synthetic TranslationBatch objects for fast unit testing., synthetic_collate_fn(), SyntheticTranslationDataset

### Community 31 - "Special Token Contract Tests"
Cohesion: 0.25
Nodes (4): Train separate tokenizers on synthetic data., Verify the hard special-token ID contract: PAD=0, SOS=1, EOS=2, UNK=3., TestSpecialTokenContract, _train_separate_synthetic()

### Community 32 - "SOS/EOS Availability Tests"
Cohesion: 0.32
Nodes (4): SOS/EOS IDs are correctly reserved and available., Train shared tokenizer on synthetic data., TestSosEosAvailability, _train_shared_synthetic()

### Community 33 - "Checkpoint Resume Tests"
Cohesion: 0.33
Nodes (4): DataLoader, Path, Controlled end-to-end checkpoint resume validation helper. Executes: 1. Initial…, validate_checkpoint_resumption()

### Community 34 - "Parameter Cost Tests"
Cohesion: 0.33
Nodes (3): Unit tests for src/tokenizer.py (Phase 2B). Run from the repository root:…, Vocabulary-related parameter cost with d_model=128., TestParameterCostCalculation

### Community 37 - "Special ID Verification Tests"
Cohesion: 0.33
Nodes (3): Explicit special-ID verification function / constructor behavior., Tokenizer.__init__ should raise if special IDs don't match., TestSpecialIdVerification

### Community 38 - "TranslationDataset Init"
Cohesion: 0.40
Nodes (4): DataFrame, Path, Tokenizer, Initialize the TranslationDataset. Args: data: Path to preprocessed Parquet…

### Community 39 - "Transformer Encoder"
Cohesion: 0.40
Nodes (3): Stack of N Transformer Encoder Layers with source embedding and positional…, Forward pass for the complete encoder. Args: src: Source token IDs [batch_size,…, TransformerEncoder

### Community 40 - "Transformer Decoder"
Cohesion: 0.40
Nodes (3): Stack of N Transformer Decoder Layers with target embedding, positional…, Forward pass for the complete decoder. Args: tgt_input: Target input token IDs…, TransformerDecoder

### Community 43 - "Training Plot Script"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Namespace

### Community 44 - "Experiment Results IO"
Cohesion: 0.50
Nodes (3): ExperimentResult, Save all experiment results to JSON., save_experiment_results()

## Knowledge Gaps
- **18 isolated node(s):** `Samanantar Paper (TACL 2022)`, `SentencePiece`, `English Language`, `FLORES Benchmark`, `11 Indic Languages` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TranslationTransformer` connect `Inference & Evaluation CLI` to `Checkpoint Resume Tests`, `Training Optimizer & Checkpointing`, `Benchmark Engine & Telemetry`, `Transformer Sub-Layers`, `Trainer Orchestration`, `Scaled Dot-Product Attention`, `TranslationBatch & Validation`, `Reproducibility Metadata (Phase 7A)`, `Train CLI & Tests`, `Training Benchmark CLI`, `DataLoader Factory & Verify`, `TrainingConfig`, `Benchmark Unit Tests`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `ExperimentManager` connect `ExperimentManager (Phase 7A)` to `Reproducibility Metadata (Phase 7A)`, `Train CLI & Tests`, `Training Benchmark CLI`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `Trainer` connect `Trainer Orchestration` to `Inference & Evaluation CLI`, `Checkpoint Resume Tests`, `Training Optimizer & Checkpointing`, `Benchmark Engine & Telemetry`, `TranslationBatch & Validation`, `Reproducibility Metadata (Phase 7A)`, `Train CLI & Tests`, `Training Benchmark CLI`, `TrainingConfig`, `Benchmark Unit Tests`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `TranslationTransformer` (e.g. with `main()` and `main()`) actually correct?**
  _`TranslationTransformer` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `Trainer` (e.g. with `main()` and `TrainingBenchmark`) actually correct?**
  _`Trainer` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `TrainingConfig` (e.g. with `main()` and `validate_checkpoint_resumption()`) actually correct?**
  _`TrainingConfig` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Samanantar Paper (TACL 2022)`, `SentencePiece`, `English Language` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._