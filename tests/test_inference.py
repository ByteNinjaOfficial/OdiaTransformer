"""Unit tests for Phase 6 Translation Inference and Greedy Autoregressive Decoding."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference import batch_translate, greedy_decode, translate
from tokenizer import EOS_ID, PAD_ID, SOS_ID, UNK_ID
from transformer import TranslationTransformer


class DummyTokenizer:
    """Mock Tokenizer for deterministic unit testing of inference."""

    def __init__(self, vocab_map: dict[str, list[int]] | None = None, id_map: dict[int, str] | None = None):
        self.vocab_map = vocab_map or {}
        self.id_map = id_map or {}

    def encode(self, text: str) -> list[int]:
        if text in self.vocab_map:
            return list(self.vocab_map[text])
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        words = [self.id_map.get(i, f"tok{i}") for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return " ".join(words)


class TestGreedyDecoding(unittest.TestCase):
    """Tests for greedy autoregressive decoding mechanics."""

    def setUp(self):
        # Create a small Transformer model
        self.model = TranslationTransformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            pad_id=PAD_ID,
        )
        self.model.eval()

    def test_greedy_decode_stops_at_eos(self):
        src_tokens = [5, 6, EOS_ID]

        # Patch decode to return a fixed sequence: tok 10 -> tok 20 -> EOS_ID
        step_counter = 0

        def mock_decode(tgt_input, memory, tgt_mask=None, src_mask=None):
            nonlocal step_counter
            step_counter += 1
            B, L_t = tgt_input.shape
            logits = torch.zeros(B, L_t, 50)
            if L_t == 1:
                logits[:, -1, 10] = 10.0  # First token: 10
            elif L_t == 2:
                logits[:, -1, 20] = 10.0  # Second token: 20
            else:
                logits[:, -1, EOS_ID] = 10.0  # Third token: EOS
            return logits

        with patch.object(self.model, "decode", side_effect=mock_decode):
            out_ids = greedy_decode(self.model, src_tokens, max_length=20, device="cpu")

        self.assertEqual(out_ids, [10, 20])
        self.assertEqual(step_counter, 3)

    def test_greedy_decode_respects_max_length(self):
        src_tokens = [5, 6, EOS_ID]

        # Patch decode to never predict EOS
        def mock_decode(tgt_input, memory, tgt_mask=None, src_mask=None):
            B, L_t = tgt_input.shape
            logits = torch.zeros(B, L_t, 50)
            logits[:, -1, 15] = 10.0  # Always predict 15
            return logits

        with patch.object(self.model, "decode", side_effect=mock_decode):
            out_ids = greedy_decode(self.model, src_tokens, max_length=5, device="cpu")

        # Should generate exactly max_length tokens
        self.assertEqual(len(out_ids), 5)
        self.assertEqual(out_ids, [15, 15, 15, 15, 15])

    def test_encoder_called_once(self):
        src_tokens = [5, 6, EOS_ID]
        encode_call_count = 0

        original_encode = self.model.encode

        def counted_encode(src, src_mask=None):
            nonlocal encode_call_count
            encode_call_count += 1
            return original_encode(src, src_mask)

        with patch.object(self.model, "encode", side_effect=counted_encode):
            greedy_decode(self.model, src_tokens, max_length=4, device="cpu")

        # Encoder must be called exactly once
        self.assertEqual(encode_call_count, 1)


class TestTranslateFunction(unittest.TestCase):
    """Tests for end-to-end translate and batch_translate functions."""

    def setUp(self):
        self.model = TranslationTransformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        self.tok_en = DummyTokenizer(vocab_map={"Hello": [11, 12]})
        self.tok_or = DummyTokenizer(id_map={21: "ନମସ୍କାର", 22: "ଓଡ଼ିଶା"})

    def test_translate_empty_text(self):
        res = translate(self.model, "   ", self.tok_en, self.tok_or)
        self.assertEqual(res, "")

    def test_translate_with_mock_generation(self):
        # Patch greedy_decode to return [21, 22]
        with patch("inference.greedy_decode", return_value=[21, 22]):
            result = translate(self.model, "Hello", self.tok_en, self.tok_or)
            self.assertEqual(result, "ନମସ୍କାର ଓଡ଼ିଶା")

    def test_batch_translate(self):
        with patch("inference.greedy_decode", side_effect=[[21], [22]]):
            results = batch_translate(self.model, ["text1", "text2"], self.tok_en, self.tok_or)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], "ନମସ୍କାର")
            self.assertEqual(results[1], "ଓଡ଼ିଶା")


if __name__ == "__main__":
    unittest.main()
