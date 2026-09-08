"""Phase 4: Hand-written Transformer architecture for English -> Odia translation.

Implemented strictly from scratch using PyTorch primitives without torch.nn.Transformer
or torch.nn.MultiheadAttention. Consumes Phase 3 TranslationBatch tensors and produces
raw unnormalized logits over the Odia vocabulary.

Architecture Specifications:
---------------------------
- Translation Direction: English (src) -> Odia (tgt)
- Vocabulary Sizes: Source = 16,000, Target = 32,000
- Special Token IDs: PAD_ID=0, SOS_ID=1, EOS_ID=2, UNK_ID=3
- Hyperparameters:
    d_model = 128
    num_heads = 4
    head_dim = 32 (d_model // num_heads)
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512 (4 * d_model)
    dropout = 0.1
- Output: Raw unnormalized logits [B, L_tgt, tgt_vocab_size] (no softmax inside model)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import PAD_ID


# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in Vaswani et al. (2017) §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The positional encodings are registered as a persistent buffer (non-trainable)
    and automatically move between CPU and CUDA devices with the module.
    Embeddings are scaled by sqrt(d_model) prior to adding positional encoding.
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings in log space for numerical precision
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Buffer shape: [1, max_len, d_model] for batch broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add scaled positional encodings and apply dropout.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max positional encoding length {self.pe.size(1)}"
            )

        # Scale embeddings by sqrt(d_model) before adding positional encoding
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + M) @ V

    Mask convention (Phase 3 compatible):
    - True  = VALID / ATTENDABLE token
    - False = MASKED / BLOCKED token (filled with -1e9 before softmax)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            query: Tensor of shape [batch_size, num_heads, q_len, d_k].
            key:   Tensor of shape [batch_size, num_heads, k_len, d_k].
            value: Tensor of shape [batch_size, num_heads, k_len, d_v].
            mask:  Broadcastable boolean mask [batch_size, 1, q_len, k_len] or [B, 1, 1, k_len].
                   True indicates valid token; False indicates masked.

        Returns:
            Tuple of (output [batch_size, num_heads, q_len, d_v],
                      attention_weights [batch_size, num_heads, q_len, k_len]).
        """
        d_k = query.size(-1)
        # scores: [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask is True for valid tokens, False for masked tokens.
            # Replace False positions with a large negative value before softmax.
            # Use -1e4 for float16/AMP compatibility (min float16 is ~ -65504) and -1e9 for float32.
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask, fill_value)

        attn_weights = F.softmax(scores, dim=-1)

        # If a row was entirely masked out, softmax produces uniform 1/N.
        # Zero out fully-masked rows for clean mathematical purity:
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)

        attn_weights_dropped = self.dropout(attn_weights)
        output = torch.matmul(attn_weights_dropped, value)
        return output, attn_weights


# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention using separate linear projections for Q, K, V, and O.

    Supports:
    - Self-attention (query = key = value)
    - Cross-attention (query from decoder, key & value from encoder memory)
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform Multi-Head Attention.

        Args:
            query: Tensor of shape [batch_size, q_len, d_model].
            key:   Tensor of shape [batch_size, k_len, d_model].
            value: Tensor of shape [batch_size, k_len, d_model].
            mask:  Optional broadcastable boolean mask (True=valid, False=masked).

        Returns:
            Tensor of shape [batch_size, q_len, d_model].
        """
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)

        # 1. Project inputs: [batch_size, seq_len, d_model]
        # 2. Reshape and transpose to: [batch_size, num_heads, seq_len, head_dim]
        q = self.w_q(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        attn_out, _ = self.attention(q, k, v, mask=mask)

        # 4. Concatenate heads back to [batch_size, q_len, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)

        # 5. Final output projection
        return self.w_o(attn_out)


# ---------------------------------------------------------------------------
# 4. Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for position-wise feed-forward.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor of shape [batch_size, seq_len, d_model].
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 5. Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer:
    Input -> Self-MHA -> Dropout -> Add & LayerNorm -> FFN -> Dropout -> Add & LayerNorm
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for an encoder layer.

        Args:
            x: Input embeddings/states, shape [batch_size, src_len, d_model].
            src_mask: Optional source padding mask, shape [batch_size, 1, 1, src_len].

        Returns:
            Tensor of shape [batch_size, src_len, d_model].
        """
        # Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(query=x, key=x, value=x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: Feed-Forward Network + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 6. Transformer Decoder Layer
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer:
    1. Masked Self-Attention (tgt_mask) -> Dropout -> Add & LayerNorm
    2. Cross-Attention with Encoder Memory (src_mask) -> Dropout -> Add & LayerNorm
    3. Position-wise Feed-Forward -> Dropout -> Add & LayerNorm
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a decoder layer.

        Args:
            x: Target representations [batch_size, tgt_len, d_model].
            memory: Encoder representations [batch_size, src_len, d_model].
            tgt_mask: Target combined mask [batch_size, 1, tgt_len, tgt_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].

        Returns:
            Tensor of shape [batch_size, tgt_len, d_model].
        """
        # Sub-layer 1: Masked Self-Attention (future tokens and pad tokens masked)
        self_attn_out = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Sub-layer 2: Cross-Attention over encoder output (source pad tokens masked)
        cross_attn_out = self.cross_attn(query=x, key=memory, value=memory, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: Position-wise Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 7. Transformer Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of N Transformer Encoder Layers with source embedding and positional encoding."""

    def __init__(
        self,
        src_vocab_size: int = 16000,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        max_len: int = 5000,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the complete encoder.

        Args:
            src: Source token IDs [batch_size, src_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].

        Returns:
            Encoder memory tensor of shape [batch_size, src_len, d_model].
        """
        x = self.src_embed(src)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 8. Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Stack of N Transformer Decoder Layers with target embedding, positional encoding,
    and final linear projection to vocabulary logits (unnormalized).
    """

    def __init__(
        self,
        tgt_vocab_size: int = 32000,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        max_len: int = 5000,
    ):
        super().__init__()
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the complete decoder.

        Args:
            tgt_input: Target input token IDs [batch_size, tgt_len].
            memory: Encoder representations [batch_size, src_len, d_model].
            tgt_mask: Target combined mask [batch_size, 1, tgt_len, tgt_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].

        Returns:
            Raw unnormalized logits of shape [batch_size, tgt_len, tgt_vocab_size].
        """
        x = self.tgt_embed(tgt_input)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)
        x = self.norm(x)
        # Produce raw logits (no softmax)
        return self.output_projection(x)


# ---------------------------------------------------------------------------
# 9. Full Translation Transformer Model
# ---------------------------------------------------------------------------

class TranslationTransformer(nn.Module):
    """Full Encoder-Decoder Transformer for English -> Odia Neural Machine Translation.

    Integrates hand-written TransformerEncoder and TransformerDecoder with Xavier
    parameter initialization and clean encode/decode/forward interfaces.
    """

    def __init__(
        self,
        src_vocab_size: int = 16000,
        tgt_vocab_size: int = 32000,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        max_len: int = 5000,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout_rate = dropout
        self.pad_id = pad_id

        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=self.d_ff,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=self.d_ff,
            dropout=dropout,
            pad_id=pad_id,
            max_len=max_len,
        )

        # Apply standard Xavier uniform initialization
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters with Xavier uniform for linear projections,
        ones for LayerNorm weights, and zeros for biases.
        Preserves special zero-padding embedding behavior for padding_idx.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                if "norm" in name and name.endswith("weight"):
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

        # Ensure padding embeddings are initialized to 0
        with torch.no_grad():
            self.encoder.src_embed.weight[self.pad_id].fill_(0)
            self.decoder.tgt_embed.weight[self.pad_id].fill_(0)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence into continuous representations.

        Args:
            src: Source token IDs [batch_size, src_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].

        Returns:
            Encoder memory tensor [batch_size, src_len, d_model].
        """
        return self.encoder(src=src, src_mask=src_mask)

    def decode(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory.

        Args:
            tgt_input: Target input token IDs [batch_size, tgt_len].
            memory: Encoder memory tensor [batch_size, src_len, d_model].
            tgt_mask: Target combined mask [batch_size, 1, tgt_len, tgt_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].

        Returns:
            Logits of shape [batch_size, tgt_len, tgt_vocab_size].
        """
        return self.decoder(
            tgt_input=tgt_input,
            memory=memory,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass for training with teacher forcing.

        Args:
            src: Source token IDs [batch_size, src_len].
            tgt_input: Target input token IDs [batch_size, tgt_len].
            src_mask: Optional source padding mask [batch_size, 1, 1, src_len].
            tgt_mask: Optional target combined mask [batch_size, 1, tgt_len, tgt_len].

        Returns:
            Raw unnormalized logits of shape [batch_size, tgt_len, tgt_vocab_size].
        """
        memory = self.encode(src=src, src_mask=src_mask)
        logits = self.decode(
            tgt_input=tgt_input,
            memory=memory,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
        )
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def parameter_breakdown(self) -> dict[str, int]:
        """Return parameter count breakdown by submodule."""
        return {
            "source_embedding": sum(p.numel() for p in self.encoder.src_embed.parameters()),
            "target_embedding": sum(p.numel() for p in self.decoder.tgt_embed.parameters()),
            "encoder_layers": sum(p.numel() for p in self.encoder.layers.parameters())
            + sum(p.numel() for p in self.encoder.norm.parameters()),
            "decoder_layers": sum(p.numel() for p in self.decoder.layers.parameters())
            + sum(p.numel() for p in self.decoder.norm.parameters()),
            "output_projection": sum(p.numel() for p in self.decoder.output_projection.parameters()),
            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

