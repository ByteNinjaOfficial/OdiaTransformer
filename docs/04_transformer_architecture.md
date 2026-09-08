# Phase 4 — From-Scratch Transformer Architecture

**Project:** Test-7 / OdiaTransformer — English → Odia Translation  
**Phase:** Phase 4 — Hand-Written PyTorch Transformer Model Architecture  
**Status:** Complete & Verified.

---

## 1. Phase 4 Objective & Hard Constraints

Phase 4 implements the complete Transformer architecture (§5.6) strictly from scratch using PyTorch primitives without using `torch.nn.Transformer`, `torch.nn.MultiheadAttention`, or third-party libraries. The model is designed to consume Phase 3 `TranslationBatch` tensors and produce raw, unnormalized logits across the Odia target vocabulary (`32,000`).

---

## 2. Locked Hyperparameters & Configuration

| Parameter | Value | Description |
|---|---|---|
| **Translation Direction** | English (`src`) $\to$ Odia (`tgt`) | Parallel sequence-to-sequence translation |
| **Source Vocabulary ($V_{\text{src}}$)** | 16,000 | SentencePiece BPE English vocabulary |
| **Target Vocabulary ($V_{\text{tgt}}$)** | 32,000 | SentencePiece BPE Odia vocabulary |
| **Special Token IDs** | `PAD=0, SOS=1, EOS=2, UNK=3` | Preserved from Phase 2 contract |
| **Model Dimension ($d_{\text{model}}$)** | 128 | Hidden embedding and channel dimension |
| **Attention Heads ($h$)** | 4 | Multi-head attention heads |
| **Head Dimension ($d_k = d_v$)** | 32 | $d_{\text{model}} // h = 128 // 4$ |
| **Encoder Layers ($N_{\text{enc}}$)** | 2 | Stack of independent encoder layers |
| **Decoder Layers ($N_{\text{dec}}$)** | 2 | Stack of independent decoder layers |
| **Feed-Forward Hidden ($d_{\text{ff}}$)** | 512 | Intermediate dimension ($4 \times d_{\text{model}}$) |
| **Dropout** | 0.1 | Applied to attention weights, sub-layers, and embeddings |
| **Max Positional Length** | 5,000 | Maximum supported sequence length |

---

## 3. High-Level Architecture Diagram

```text
               English Source Token IDs [B, L_src]
                                │
                                ▼
         nn.Embedding(16000, 128, padding_idx=0)
                                │
                                ▼
                      * sqrt(d_model) = 11.3137
                                │
                                ▼
         + Sinusoidal Positional Encoding + Dropout(0.1)
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │     Encoder Layer 1 (d=128, h=4)       │
           │  • Self-MHA (src_mask) + Res + Norm    │
           │  • FFN (d_ff=512) + Res + Norm         │
           └───────────────────┬────────────────────┘
                               │
                               ▼
           ┌────────────────────────────────────────┐
           │     Encoder Layer 2 (d=128, h=4)       │
           │  • Self-MHA (src_mask) + Res + Norm    │
           │  • FFN (d_ff=512) + Res + Norm         │
           └───────────────────┬────────────────────┘
                               │
                               ▼
                        Final LayerNorm
                               │
                               ▼
                   Encoder Memory [B, L_src, 128]
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             │
Odia Target Input IDs [B, L_tgt]                      │
        │                                             │
        ▼                                             │
nn.Embedding(32000, 128, padding_idx=0)               │
        │                                             │
        ▼                                             │
 * sqrt(d_model) = 11.3137                            │
        │                                             │
        ▼                                             │
+ Sinusoidal Positional Encoding + Dropout(0.1)       │
        │                                             │
        ▼                                             │
┌────────────────────────────────────────┐            │
│      Decoder Layer 1 (d=128, h=4)      │            │
│  • Masked Self-MHA (tgt_mask)          │            │
│  • Cross-MHA (src_mask) ◄──────────────┼────────────┘
│  • FFN (d_ff=512)                      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│      Decoder Layer 2 (d=128, h=4)      │
│  • Masked Self-MHA (tgt_mask)          │
│  • Cross-MHA (src_mask)                │
│  • FFN (d_ff=512)                      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
            Final LayerNorm
                   │
                   ▼
   nn.Linear(128 -> 32000) (No Softmax)
                   │
                   ▼
    Raw Logits [B, L_tgt, 32000]
```

---

## 4. Mathematical Components & Implementation

### 4.1 Sinusoidal Positional Encoding
Deterministic sinusoidal waves are computed across the channel dimensions:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

- **Registered Buffer:** Registered with `register_buffer('pe', pe, persistent=False)` to move automatically between CPU and CUDA devices without being treated as trainable parameters.
- **Embedding Scaling:** Input token embeddings are scaled by $\sqrt{d_{\text{model}}}$ before adding $PE$, preserving the relative magnitude of token representations against positional signals.

### 4.2 Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- **Phase 3 Mask Compatibility:** Handles boolean masks where `True` indicates a valid token and `False` indicates masked.
- **Numerical Stability:** Masked positions are substituted with $-10^9$ (`masked_fill(~mask, -1e9)`) before softmax, ensuring zero attention probability ($0.0$) on masked positions and preventing `NaN`s on fully-masked sequences.

### 4.3 Multi-Head Attention
Linearly projects $Q, K, V$ with matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:
- Reshaped to $[B, h, L, d_k]$ where $d_k = 32$.
- Evaluated across attention heads in parallel.
- Recombined and projected through $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.
- Validates that $d_{\text{model}} \pmod h == 0$.

### 4.4 Position-wise Feed-Forward Network
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
- $W_1 \in \mathbb{R}^{128 \times 512}$, $b_1 \in \mathbb{R}^{512}$
- $W_2 \in \mathbb{R}^{512 \times 128}$, $b_2 \in \mathbb{R}^{128}$
- Includes intermediate dropout ($p=0.1$).

### 4.5 Residual Connections & Layer Normalization
Post-LN architecture (Vaswani et al. 2017):
$$\text{SubLayerOutput} = \text{LayerNorm}(x + \text{Dropout}(\text{SubLayer}(x)))$$

---

## 5. Mask Broadcasting & Interaction

| Submodule | Query ($Q$) | Key ($K$) / Value ($V$) | Mask Applied | Broadcast Dimension |
|---|---|---|---|---|
| **Encoder Self-Attention** | Source ($x$) | Source ($x$) | `src_mask` | $[B, 1, 1, L_{\text{src}}] \to [B, h, L_{\text{src}}, L_{\text{src}}]$ |
| **Decoder Masked Self-Attention** | Target ($x$) | Target ($x$) | `tgt_mask` | $[B, 1, L_{\text{tgt}}, L_{\text{tgt}}] \to [B, h, L_{\text{tgt}}, L_{\text{tgt}}]$ |
| **Decoder Cross-Attention** | Target ($x$) | Encoder Memory | `src_mask` | $[B, 1, 1, L_{\text{src}}] \to [B, h, L_{\text{tgt}}, L_{\text{src}}]$ |

- **`tgt_mask`:** Combines lower-triangular causal mask ($j \le i$) with target padding mask, ensuring the autoregressive decoder never peeks into future tokens or attends to padding.
- **`src_mask`:** Prevents cross-attention from attending to source `<PAD>` tokens.

---

## 6. Output Contract: Raw Logits

The model returns **raw unnormalized logits** of shape $[B, L_{\text{tgt}}, 32000]$.
- **No Softmax in Model:** Applying softmax internally is redundant and numerically unstable when training with PyTorch's `nn.CrossEntropyLoss(ignore_index=0)`, which internally applies `log_softmax` over logits.

---

## 7. Parameter Allocation Breakdown

| Component | Calculation | Trainable Parameters | % of Total |
|---|---|---|---|
| **Source Embedding** | $16,000 \times 128$ | $2,048,000$ | 18.29% |
| **Target Embedding** | $32,000 \times 128$ | $4,096,000$ | 36.58% |
| **Encoder (2 layers + LN)** | $2 \times (4 \times 128^2 + 4 \times 128 + 2 \times 128 \times 512 + 128 + 512 + 4 \times 128) + 2 \times 128$ | $396,800$ | 3.54% |
| **Decoder (2 layers + LN)** | $2 \times (2 \times (4 \times 128^2 + 4 \times 128) + 2 \times 128 \times 512 + 128 + 512 + 6 \times 128) + 2 \times 128$ | $529,408$ | 4.73% |
| **Output Linear Projection** | $128 \times 32,000 + 32,000$ | $4,128,000$ | 36.86% |
| **Total Trainable Parameters** | — | **11,198,208 (~11.20M)** | **100.0%** |

*Note: The core Transformer computation layers (Encoder + Decoder) account for only ~926K parameters; the remaining ~10.27M parameters are vocabulary embeddings and output projection.*

---

## 8. Verification Results

- **Unit Tests:** `tests/test_transformer.py` (20/20 passing).
- **Full Test Suite:** `tests/` (91/91 passing across all 4 phases).
- **Hardware Smoke Test:** Verified on CPU and CUDA (`NVIDIA GeForce RTX 2050`). No `NaN`, no `Inf`, loss backpropagation verified.

