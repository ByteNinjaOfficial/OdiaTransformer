"""Unit tests for Phase 4 from-scratch Transformer architecture."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    TranslationBatch,
    TranslationDataset,
    create_dataloader,
    create_dataloaders,
    create_padding_mask,
    create_causal_mask,
    create_target_mask,
)
from tokenizer import (
    EOS_ID,
    PAD_ID,
    SOS_ID,
    UNK_ID,
    Tokenizer,
    load_tokenizer,
)
from transformer import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    ScaledDotProductAttention,
    SinusoidalPositionalEncoding,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TranslationTransformer,
)


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """Tests for SinusoidalPositionalEncoding module."""

    def test_shape_preservation(self):
        d_model = 128
        pe_module = SinusoidalPositionalEncoding(d_model=d_model, dropout=0.0, max_len=100)
        x = torch.randn(2, 25, d_model)
        out = pe_module(x)
        self.assertEqual(out.shape, (2, 25, d_model))

    def test_registered_buffer_not_parameter(self):
        pe_module = SinusoidalPositionalEncoding(d_model=64, max_len=50)
        param_names = [name for name, _ in pe_module.named_parameters()]
        buffer_names = [name for name, _ in pe_module.named_buffers()]
        self.assertNotIn("pe", param_names)
        self.assertIn("pe", buffer_names)

    def test_positions_differ(self):
        pe_module = SinusoidalPositionalEncoding(d_model=32, dropout=0.0, max_len=10)
        pe_module.eval()
        x = torch.zeros(1, 5, 32)
        out = pe_module(x)
        # Verify position 0 != position 1
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1]))

    def test_exceeding_max_len_raises(self):
        pe_module = SinusoidalPositionalEncoding(d_model=32, max_len=10)
        x = torch.randn(1, 15, 32)
        with self.assertRaises(ValueError):
            pe_module(x)


class TestScaledDotProductAttention(unittest.TestCase):
    """Tests for ScaledDotProductAttention module."""

    def setUp(self):
        self.attn = ScaledDotProductAttention(dropout=0.0)

    def test_output_shapes_and_probability_sum(self):
        B, H, L_q, L_k, d_k = 2, 4, 6, 8, 32
        q = torch.randn(B, H, L_q, d_k)
        k = torch.randn(B, H, L_k, d_k)
        v = torch.randn(B, H, L_k, d_k)

        out, weights = self.attn(q, k, v)
        self.assertEqual(out.shape, (B, H, L_q, d_k))
        self.assertEqual(weights.shape, (B, H, L_q, L_k))

        # Weights along key dimension should sum to ~1.0
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_padding_mask_zero_weight(self):
        # B=1, H=1, L_q=2, L_k=4, d_k=16
        q = torch.randn(1, 1, 2, 16)
        k = torch.randn(1, 1, 4, 16)
        v = torch.randn(1, 1, 4, 16)

        # Mask where position 3 is PAD (False)
        mask = torch.tensor([[[[True, True, True, False]]]])  # [1, 1, 1, 4]

        _, weights = self.attn(q, k, v, mask=mask)
        # Position 3 should receive 0.0 weight
        for q_idx in range(2):
            self.assertAlmostEqual(weights[0, 0, q_idx, 3].item(), 0.0, places=5)
            self.assertGreater(weights[0, 0, q_idx, 0].item(), 0.0)

    def test_causal_mask_future_blocked(self):
        L = 4
        q = torch.randn(1, 1, L, 16)
        k = torch.randn(1, 1, L, 16)
        v = torch.randn(1, 1, L, 16)

        # Causal mask: [1, 1, 4, 4]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

        _, weights = self.attn(q, k, v, mask=causal)

        for i in range(L):
            for j in range(L):
                if j > i:
                    self.assertEqual(weights[0, 0, i, j].item(), 0.0)
                else:
                    self.assertGreater(weights[0, 0, i, j].item(), 0.0)


class TestMultiHeadAttention(unittest.TestCase):
    """Tests for MultiHeadAttention module."""

    def test_d_model_divisibility_validation(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=128, num_heads=5)

    def test_self_attention_shape(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 128)
        out = mha(query=x, key=x, value=x)
        self.assertEqual(out.shape, (2, 10, 128))

    def test_cross_attention_shape(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4, dropout=0.0)
        q = torch.randn(2, 7, 128)   # Target length 7
        kv = torch.randn(2, 12, 128) # Source length 12
        out = mha(query=q, key=kv, value=kv)
        self.assertEqual(out.shape, (2, 7, 128))

    def test_mask_broadcasting(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 64)
        # Phase 3 target mask: [2, 1, 5, 5]
        tgt_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(2, 1, 5, 5)
        out = mha(query=x, key=x, value=x, mask=tgt_mask)
        self.assertEqual(out.shape, (2, 5, 64))

    def test_gradient_flow(self):
        mha = MultiHeadAttention(d_model=32, num_heads=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = mha(x, x, x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(mha.w_q.weight.grad)
        self.assertIsNotNone(mha.w_o.weight.grad)


class TestFeedForwardAndLayers(unittest.TestCase):
    """Tests for PositionwiseFeedForward, EncoderLayer, and DecoderLayer."""

    def test_feed_forward_shape_and_grad(self):
        ffn = PositionwiseFeedForward(d_model=64, d_ff=256)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 8, 64))
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_encoder_layer_shape_and_residual(self):
        enc_layer = TransformerEncoderLayer(d_model=64, num_heads=4, d_ff=256)
        x = torch.randn(2, 10, 64)
        src_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
        out = enc_layer(x, src_mask=src_mask)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_decoder_layer_shape_and_masks(self):
        dec_layer = TransformerDecoderLayer(d_model=64, num_heads=4, d_ff=256)
        x = torch.randn(2, 6, 64)       # Target seq len 6
        memory = torch.randn(2, 10, 64) # Source seq len 10
        tgt_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(2, 1, 6, 6)
        src_mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)

        out = dec_layer(x, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)
        self.assertEqual(out.shape, (2, 6, 64))


class TestTranslationTransformer(unittest.TestCase):
    """Tests for the complete TranslationTransformer model."""

    def setUp(self):
        # Small lightweight model for fast unit testing
        self.model = TranslationTransformer(
            src_vocab_size=50,
            tgt_vocab_size=60,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128,
            dropout=0.1,
            pad_id=PAD_ID,
        )

    def test_forward_output_shape_and_raw_logits(self):
        B, L_s, L_t = 3, 7, 5
        src = torch.randint(1, 50, (B, L_s))
        tgt_input = torch.randint(1, 60, (B, L_t))

        src_mask = create_padding_mask(src, pad_id=PAD_ID)
        tgt_mask = create_target_mask(tgt_input, pad_id=PAD_ID)

        logits = self.model(src=src, tgt_input=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        # Expected shape: [B, L_t, tgt_vocab_size]
        self.assertEqual(logits.shape, (B, L_t, 60))
        # Logits should be finite numbers
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
        # Verify raw logits (not normalized probabilities summing to 1)
        sums = logits.sum(dim=-1)
        self.assertFalse(torch.allclose(sums, torch.ones_like(sums)))

    def test_encode_decode_interface(self):
        src = torch.randint(1, 50, (2, 8))
        tgt_input = torch.randint(1, 60, (2, 4))

        memory = self.model.encode(src)
        self.assertEqual(memory.shape, (2, 8, 32))

        logits = self.model.decode(tgt_input, memory)
        self.assertEqual(logits.shape, (2, 4, 60))

    def test_backward_gradient_propagation(self):
        src = torch.randint(1, 50, (2, 6))
        tgt_input = torch.randint(1, 60, (2, 5))
        tgt_output = torch.randint(1, 60, (2, 5))

        logits = self.model(src=src, tgt_input=tgt_input)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loss = criterion(logits.view(-1, 60), tgt_output.view(-1))

        self.model.zero_grad()
        loss.backward()

        # Check gradient on embedding and projections
        self.assertIsNotNone(self.model.encoder.src_embed.weight.grad)
        self.assertIsNotNone(self.model.decoder.tgt_embed.weight.grad)
        self.assertIsNotNone(self.model.decoder.output_projection.weight.grad)

    def test_parameter_counting(self):
        total, trainable = self.model.count_parameters()
        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)
        breakdown = self.model.parameter_breakdown()
        self.assertIn("source_embedding", breakdown)
        self.assertIn("output_projection", breakdown)
        self.assertEqual(breakdown["total_trainable"], total)


class TestRealDataIntegration(unittest.TestCase):
    """Integration test with real tokenizers and real Phase 3 DataLoaders."""

    def test_real_batch_forward_pass(self):
        en_model_path = Path("outputs/tokenizer_sep_en16000_or32000_en.model")
        or_model_path = Path("outputs/tokenizer_sep_en16000_or32000_or.model")
        val_parquet_path = Path("outputs/val.parquet")

        if not (en_model_path.exists() and or_model_path.exists() and val_parquet_path.exists()):
            self.skipTest("Real tokenizer models or outputs/val.parquet not found; skipping integration test.")

        # Instantiate full Phase 4 configuration
        model = TranslationTransformer(
            src_vocab_size=16000,
            tgt_vocab_size=32000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=512,
            dropout=0.1,
            pad_id=PAD_ID,
        )

        # Load real DataLoaders
        _, val_loader, _ = create_dataloaders(
            data_dir="outputs",
            en_tokenizer_path=en_model_path,
            or_tokenizer_path=or_model_path,
            batch_size=4,
        )

        batch: TranslationBatch = next(iter(val_loader))

        model.eval()
        with torch.no_grad():
            logits = model(
                src=batch.src,
                tgt_input=batch.tgt_input,
                src_mask=batch.src_mask,
                tgt_mask=batch.tgt_mask,
            )

        B, L_t = batch.tgt_input.shape
        self.assertEqual(logits.shape, (B, L_t, 32000))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())


if __name__ == "__main__":
    unittest.main()
