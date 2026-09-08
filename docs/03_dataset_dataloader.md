# Phase 3 — Dataset, DataLoader, Collation, and Mask Generation

**Project:** Test-7 / OdiaTransformer — English → Odia Translation (Transformer From Scratch)  
**Phase:** Phase 3 — PyTorch Dataset, Dynamic Collation, Teacher-Forcing Sequences & Attention Mask Pipeline  
**Status:** Complete.

---

## 1. Phase Objective

The goal of Phase 3 is to construct the data loading and tensor formulation bridge between the Phase 1 preprocessed Parquet datasets (`outputs/train.parquet`, `outputs/val.parquet`, `outputs/test.parquet`) and the Phase 2 trained SentencePiece BPE tokenizers (`outputs/tokenizer_sep_en16000_or32000_*.model`), preparing batched PyTorch tensors and attention masks for the hand-written Transformer model.

---

## 2. Dataset Architecture & Token Flow

The complete data pipeline follows a clean separation of concerns:

```text
Preprocessed Parquet (src: EN, tgt: OR)
                │
                ▼
      TranslationDataset
   ├── English Tokenizer (16k BPE) -> src_ids = [tok_1, ..., tok_n, EOS]
   └── Odia Tokenizer (32k BPE)    -> tgt_input_ids  = [SOS, t_1, ..., t_m]
                                   -> tgt_output_ids = [t_1, ..., t_m, EOS]
                │
                ▼
       collate_translation_samples
   ├── Dynamic batch padding to max(L_s), max(L_t) with PAD_ID (0)
   ├── Source padding mask: [B, 1, 1, L_s] (True = valid)
   ├── Target padding mask: [B, 1, 1, L_t] (True = valid)
   ├── Causal / look-ahead mask: [1, 1, L_t, L_t] (j <= i is True)
   └── Combined target mask: [B, 1, L_t, L_t] (True = valid)
                │
                ▼
        TranslationBatch
   (Ready for Transformer Encoder & Decoder forward pass)
```

---

## 3. Source & Target Sequence Contracts

### 3.1 Special-Token Contract (Locked from Phase 2)
| Special Token | Fixed ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Dynamic batch sequence padding |
| `<SOS>` | 1 | Start of sequence prompt for decoder |
| `<EOS>` | 2 | End of sequence delimiter & stopping condition |
| `<UNK>` | 3 | Out of vocabulary token fallback |

### 3.2 Source Sequence Format (English Encoder Input)
- **Format:** `[token_1, token_2, ..., token_n, EOS_ID]`
- **Decision & Rationale:** The Transformer encoder processes the entire sequence bidirectionally in parallel, so `<SOS>` is unnecessary. Appending `<EOS>` provides an explicit sentence boundary signal distinct from trailing `<PAD>` tokens.

### 3.3 Target Teacher-Forcing Sequence Alignment (Odia Decoder Input & Labels)
Given raw tokenized Odia target tokens `[t_1, t_2, ..., t_m]`:
- **`tgt_input` (Decoder Input):** `[SOS_ID, t_1, t_2, ..., t_m]` (Length: $m + 1$)
- **`tgt_output` (Target Labels):** `[t_1, t_2, ..., t_m, EOS_ID]` (Length: $m + 1$)

Both sequences maintain identical lengths $m+1$. At decoding step $t$, the decoder observes $\text{tokens}_{0 \dots t}$ from `tgt_input` and predicts $\text{token}_{t}$ in `tgt_output` (which corresponds to target token $t+1$).

---

## 4. Dynamic Padding & Tensor Shapes

Sequences are padded dynamically per batch up to the maximum sequence length in that batch ($L_{s, \max}$ and $L_{t, \max}$), avoiding global fixed padding overhead.

| Tensor Field | Shape | Data Type | Description |
|---|---|---|---|
| `src` | `[B, L_{s, \max}]` | `torch.long` | Padded source token IDs |
| `tgt_input` | `[B, L_{t, \max}]` | `torch.long` | Padded decoder input token IDs |
| `tgt_output` | `[B, L_{t, \max}]` | `torch.long` | Padded target label token IDs for loss computation |
| `src_mask` | `[B, 1, 1, L_{s, \max}]` | `torch.bool` | Source validity mask (`True` = valid, `False` = PAD) |
| `tgt_mask` | `[B, 1, L_{t, \max}, L_{t, \max}]` | `torch.bool` | Combined target mask (`True` = attendable) |
| `src_pad_mask` | `[B, 1, 1, L_{s, \max}]` | `torch.bool` | Source padding mask |
| `tgt_pad_mask` | `[B, 1, 1, L_{t, \max}]` | `torch.bool` | Target padding mask |
| `causal_mask` | `[1, 1, L_{t, \max}, L_{t, \max}]` | `torch.bool` | Lower-triangular causal mask |

---

## 5. Mask Conventions & Semantic Rules

### 5.1 Boolean Convention
Across all masks in this repository:
- **`True`** = **VALID / ATTENDABLE** position.
- **`False`** = **MASKED / IGNORED** position (filled with $-\infty$ during multi-head attention softmax).

### 5.2 Source Padding Mask (`src_mask`)
- Computed as `(src != PAD_ID).unsqueeze(1).unsqueeze(2)`.
- Broadcastable against encoder attention weights `[B, heads, L_q, L_k]`.

### 5.3 Causal Mask (`causal_mask`)
- Lower-triangular matrix where position $(i, j)$ is `True` if $j \le i$ and `False` if $j > i$.
- Prevents the decoder from attending to future target tokens during self-attention.

### 5.4 Combined Target Mask (`tgt_mask`)
- Computed as `tgt_pad_mask & causal_mask`.
- Position $(i, j)$ is attendable (`True`) iff $j \le i$ **AND** target input token $j$ is non-PAD.

---

## 6. DataLoader Configuration & Memory Invariants

- **`create_dataloader` / `create_dataloaders`**:
  - `train_loader`: `shuffle = True`
  - `val_loader`: `shuffle = False`
  - `test_loader`: `shuffle = False`
- **Memory & Invariants**:
  - Parquet data is read into memory CPU arrays with lazy string access on `__getitem__`.
  - No GPU memory allocation occurs inside the Dataset or DataLoader; tensors are transferred to GPU via `batch.to(device)` during the training/evaluation step.
  - Zero data leakage: splits (`outputs/*.parquet`) and tokenizers (`train`-only learned vocabs) remain completely separated.

---

## 7. Verification & Tests

- **Unit Tests:** `tests/test_dataset.py` (15 tests passing).
- **Full Test Suite:** 71 tests passing across `test_preprocessing.py`, `test_tokenization.py`, and `test_dataset.py`.
- **Verification Runner:** `verify_dataset_pipeline.py` executes end-to-end smoke checks on real data and outputs tensor shapes, alignment, and device transfer.

