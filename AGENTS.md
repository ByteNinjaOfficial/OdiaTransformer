# AGENTS.md

Project: **Test-7** — a complete from-scratch Transformer machine-translation system (English → Odia), built as a course assignment. The exact problem statement + a genAI-authored breakdown live in `md files/explanation.md` (gitignored). Read it for reference.

Status: **Phase 7 complete.** Phases 1 (preprocessing) → 7 (training, inference, evaluation, benchmarking) are all implemented, tested (117 unit tests), and documented. See `docs/01…07b_*.md` (report sections map 1:1 to module phases) and README.md for results (BLEU 3.70 on 500 test pairs, 4-epoch baseline).

## Hard constraint: no pre-trained models
The assignment's whole point is implementing §5.6 Transformer yourself. Do **NOT** load `transformers` models (T5, mT5, MarianMT, NLLB, BART, etc.) and do **NOT** use high-level black-boxes like `torch.nn.Transformer` or `torch.nn.MultiheadAttention`. Libraries are fine only for: dataset handling, subword tokenization (BPE via SentencePiece), BLEU scoring (SacreBLEU), and PyTorch tensor ops. The Transformer architecture itself must be hand-written.

## Required architecture & hyperparams (as implemented)
- Hand-written Embedding → sinusoidal positional encoding → 2 encoder blocks (self-attn → FFN, residual+LN) → 2 decoder blocks (masked self-attn → cross-attn → FFN) → linear → softmax. See `src/transformer.py`.
- `d_model=128`, `heads=4`, `N=2`, `d_ff=512`, `dropout=0.1` → ~11.2M params (~11,198,208). Head dim = 128/4 = 32.
- Tokenizer: SentencePiece BPE, **separate vocabularies** — EN 16,000 / OR 32,000. Special-token ID contract: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`. Models at `outputs/tokenizer_sep_en16000_or32000_{en,or}.model`.
- Train: teacher forcing (1-step target shift), cross-entropy **ignoring pad tokens** (`ignore_index=0`) + label smoothing 0.1, AdamW + Noam warmup, AMP on GPU. Watch the causal/look-ahead mask — a suspiciously perfect val loss means the decoder is peeking.
- Infer: greedy auto-regressive decode loop (feed outputs back until `<EOS>` or max len). See `src/inference.py`.
- Evaluate: SacreBLEU corpus BLEU + 5 sample translations including one long sentence (see `docs/07b_evaluation_results.md`).
- Odia: Unicode-normalize (NFC), use subword tokenization (morphologically rich/agglutinative language).

## Mandatory workflow (follow in this order)
1. **Code** following **CQRS, SOLID, DRY** principles.
2. **Test** the code (a) that it adheres to CQRS/SOLID/DRY and (b) that the application actually runs/works. Treat any red flag as a failure to fix.
3. Only if everything is green, **update the graphify graph** (see below).

## Data — where it lives
- `dataset/` holds only the **Odia (`or`)** config of the Samanantar corpus.
  - `dataset/README.md` — the creator's full dataset card (schema, 11 Indic languages, citation, license).
  - `dataset/dataset_info.md` — just the source URL: `https://huggingface.co/datasets/ai4bharat/samanantar/tree/main/or`.
  - `dataset/train-00000-of-00001.parquet` — the ~136 MB actual data (columns `idx`, `src`, `tgt`, `data_source`). Read via `pandas.read_parquet` / `pyarrow`.
- The full Samanantar corpus is large/multi-config; only the Odia `or` subset was downloaded here — don't expect other languages to be present locally.

## Repo quirks / gotchas
- `README.md` (academic report: architecture, results, CLI guide) and `requirements.txt` (per-phase deps) are **filled in** — don't treat them as TODOs.
- `.venv/` is a local Python 3.12 virtualenv and is gitignored. Prefer it (or the system Python) for running code.
- `.gitignore` ignores `*.csv`, `*.tsv`, `outputs/`, `checkpoints/`, `experiments/**/checkpoints/*`, `*.pt/*.pth/*.ckpt`, `runs/`, `wandb/`, `*.model`, `*.vocab`, the parquet, `md files/explanation.md`, and all `graphify-out/` internals (`.graphify_*`, `*.graphify-*.py`). `experiments/**/*.json` (configs, metadata, metrics) **are** tracked for academic review — don't ignore them.
- `experiments/` holds reproducible artifacts (`phase7a_benchmark/`, `phase7b_baseline/`) but checkpoints are gitignored (`*.pt`) — the README links to `experiments/phase7b_baseline/checkpoints/best.pt` exist only locally.
- `graphify-out/*.py` are one-off Graphify helper scripts — gitignored, don't commit them.
- Don't hand-edit `graphify-out/` — it's regenerated.

## graphify (knowledge graph)
`graphify-out/graph.json` is the persisted codebase graph. Run the `/graphify ` pipeline to update it (use the skill's instructions; it builds graph.html, GRAPH_REPORT.md, graph.json). Do this **only** after coding + testing pass. The graph currently reflects ALL phases: 712 nodes, 1275 edges, 47 labeled communities (Transformer internals, training, inference/CLI, benchmarking, tokenizer, docs, etc.). Use `--update` for incremental re-extraction of changed files; full rebuild treats `graphify-out/graph.json` as the baseline.
