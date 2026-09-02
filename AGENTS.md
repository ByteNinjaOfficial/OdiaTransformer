# AGENTS.md

Project: **Test-7** — build a Transformer machine-translation system **from scratch** (English → Odia), as a course assignment. The exact problem statement + a genAI-authored breakdown live in `md files/explanation.md` (gitignored). Read it for reference.

## Hard constraint: no pre-trained models
The assignment's whole point is implementing §5.6 Transformer yourself. Do **NOT** load `transformers` models (T5, mT5, MarianMT, NLLB, BART, etc.). Libraries are fine only for: dataset handling, subword tokenization (BPE), BLEU scoring, and PyTorch tensor ops. The Transformer architecture itself must be hand-written.

## Required architecture & hyperparams
- Embedding + sinusoidal positional encoding → 2 encoder blocks (self-attn → FFN, residual+LN) → 2 decoder blocks (masked self-attn → cross-attn → FFN) → linear → softmax.
- `d_model=128`, `heads=4`, `N=2`.
- Train: teacher forcing, cross-entropy **ignoring pad tokens**, Adam + warmup. Watch the causal/look-ahead mask — a suspiciously perfect val loss means the decoder is peeking.
- Infer: greedy decode loop (feed outputs back until `<EOS>`); beam search is optional bonus.
- Evaluate: BLEU + 5 sample translations including one long sentence.
- Odia: Unicode-normalize (NFC), use subword tokenization (morphologically rich language).

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
- `README.md` and `requirements.txt` at repo root are currently **empty** — they will likely need to be filled in as part of the deliverable.
- `.venv/` is a local Python 3.12 virtualenv and is gitignored. Prefer it (or the system Python) for running code.
- `.gitignore` ignores `*.csv`, `*.tsv`, `checkpoints/`, `* (weights)`, the parquet, and `md files/explanation.md`. Generated data/tables won't be committed.
- Don't hand-edit `graphify-out/` — it's regenerated.

## graphify (knowledge graph)
`graphify-out/graph.json` is the persisted codebase graph. Run the `/graphify ` pipeline to update it (use the skill's instructions; it builds graph.html, GRAPH_REPORT.md, graph.json). Do this **only** after coding + testing pass. The graph currently reflects the docs/dataset only; it has no code nodes until source files exist.
