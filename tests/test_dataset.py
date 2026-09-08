"""Unit tests for Phase 3 Dataset, DataLoader, Collation, and Mask generation pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    TranslationBatch,
    TranslationDataset,
    TranslationSample,
    collate_translation_samples,
    create_causal_mask,
    create_dataloader,
    create_dataloaders,
    create_padding_mask,
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


class DummyTokenizer:
    """Lightweight mock tokenizer for deterministic unit testing."""

    def __init__(self, vocab_map: dict[str, list[int]] | None = None):
        self.vocab_map = vocab_map or {}

    def encode(self, text: str) -> list[int]:
        if text in self.vocab_map:
            return list(self.vocab_map[text])
        # Default fallback: encode word hashes offset by 10
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID))


class TestTranslationSample(unittest.TestCase):
    """Tests for TranslationSample dataclass invariants."""

    def test_valid_sample(self):
        sample = TranslationSample(
            src_ids=[10, 20, EOS_ID],
            tgt_input_ids=[SOS_ID, 30, 40],
            tgt_output_ids=[30, 40, EOS_ID],
            src_text="hello world",
            tgt_text="namaskar",
        )
        self.assertEqual(sample.src_ids, [10, 20, EOS_ID])
        self.assertEqual(len(sample.tgt_input_ids), len(sample.tgt_output_ids))

    def test_mismatched_target_lengths_raises(self):
        with self.assertRaises(ValueError):
            TranslationSample(
                src_ids=[10, EOS_ID],
                tgt_input_ids=[SOS_ID, 30],
                tgt_output_ids=[30, 40, EOS_ID],  # Length 3 vs 2
            )


class TestMaskGenerators(unittest.TestCase):
    """Tests for padding, causal, and combined target mask generation."""

    def test_create_padding_mask_shape_and_values(self):
        # Batch of 2, seq_len 4
        # Sample 0 has 3 tokens + 1 PAD
        # Sample 1 has 2 tokens + 2 PAD
        seq = torch.tensor([
            [10, 20, 30, PAD_ID],
            [15, 25, PAD_ID, PAD_ID],
        ], dtype=torch.long)

        mask = create_padding_mask(seq, pad_id=PAD_ID)

        self.assertEqual(mask.shape, (2, 1, 1, 4))
        self.assertEqual(mask.dtype, torch.bool)

        # Check sample 0: True, True, True, False
        self.assertTrue(mask[0, 0, 0, 0].item())
        self.assertTrue(mask[0, 0, 0, 1].item())
        self.assertTrue(mask[0, 0, 0, 2].item())
        self.assertFalse(mask[0, 0, 0, 3].item())

        # Check sample 1: True, True, False, False
        self.assertTrue(mask[1, 0, 0, 0].item())
        self.assertTrue(mask[1, 0, 0, 1].item())
        self.assertFalse(mask[1, 0, 0, 2].item())
        self.assertFalse(mask[1, 0, 0, 3].item())

    def test_create_causal_mask_properties(self):
        seq_len = 4
        causal = create_causal_mask(seq_len)

        self.assertEqual(causal.shape, (1, 1, 4, 4))
        self.assertEqual(causal.dtype, torch.bool)

        # Expected 4x4 lower triangular matrix
        # [[T, F, F, F],
        #  [T, T, F, F],
        #  [T, T, T, F],
        #  [T, T, T, T]]
        matrix = causal.squeeze(0).squeeze(0)
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    self.assertTrue(matrix[i, j].item(), f"Position ({i}, {j}) should be True (past/current)")
                else:
                    self.assertFalse(matrix[i, j].item(), f"Position ({i}, {j}) should be False (future)")

    def test_create_target_mask_combined(self):
        # tgt_input with PAD at the end
        # Sample: [SOS, 50, 60, PAD] (L_t = 4)
        tgt_input = torch.tensor([
            [SOS_ID, 50, 60, PAD_ID],
        ], dtype=torch.long)

        tgt_mask = create_target_mask(tgt_input, pad_id=PAD_ID)

        self.assertEqual(tgt_mask.shape, (1, 1, 4, 4))
        mat = tgt_mask[0, 0]

        # Step 0 (SOS): attend only to SOS (pos 0) -> True, F, F, F
        self.assertEqual(mat[0].tolist(), [True, False, False, False])
        # Step 1 (50): attend to SOS, 50 -> True, True, False, False
        self.assertEqual(mat[1].tolist(), [True, True, False, False])
        # Step 2 (60): attend to SOS, 50, 60 -> True, True, True, False
        self.assertEqual(mat[2].tolist(), [True, True, True, False])
        # Step 3 (PAD): even if j <= 3, position 3 is PAD so masked -> True, True, True, False
        self.assertEqual(mat[3].tolist(), [True, True, True, False])


class TestTranslationDataset(unittest.TestCase):
    """Tests for TranslationDataset creation, indexing, and token flows."""

    def setUp(self):
        self.tok_en = DummyTokenizer({"Good morning": [101, 102], "Hello": [201]})
        self.tok_or = DummyTokenizer({"ଶୁଭ ପ୍ରଭାତ": [501, 502], "ନମସ୍କାର": [601]})
        self.df = pd.DataFrame({
            "idx": [0, 1],
            "src": ["Good morning", "Hello"],
            "tgt": ["ଶୁଭ ପ୍ରଭାତ", "ନମସ୍କାର"],
        })

    def test_dataset_len_and_indexing(self):
        ds = TranslationDataset(self.df, self.tok_en, self.tok_or, add_eos_to_source=True)
        self.assertEqual(len(ds), 2)

        sample0 = ds[0]
        # Source should have EOS appended: [101, 102, EOS_ID]
        self.assertEqual(sample0.src_ids, [101, 102, EOS_ID])
        # Target input should have SOS prepended: [SOS_ID, 501, 502]
        self.assertEqual(sample0.tgt_input_ids, [SOS_ID, 501, 502])
        # Target output should have EOS appended: [501, 502, EOS_ID]
        self.assertEqual(sample0.tgt_output_ids, [501, 502, EOS_ID])
        self.assertEqual(sample0.src_text, "Good morning")
        self.assertEqual(sample0.tgt_text, "ଶୁଭ ପ୍ରଭାତ")

    def test_dataset_source_without_eos(self):
        ds = TranslationDataset(self.df, self.tok_en, self.tok_or, add_eos_to_source=False)
        sample0 = ds[0]
        self.assertEqual(sample0.src_ids, [101, 102])

    def test_dataset_parquet_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.df.to_parquet(parquet_path)

            ds = TranslationDataset(parquet_path, self.tok_en, self.tok_or)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[1].src_ids, [201, EOS_ID])
            self.assertEqual(ds[1].tgt_input_ids, [SOS_ID, 601])
            self.assertEqual(ds[1].tgt_output_ids, [601, EOS_ID])

    def test_dataset_invalid_columns_raises(self):
        bad_df = pd.DataFrame({"text_a": ["a"], "text_b": ["b"]})
        with self.assertRaises(ValueError):
            TranslationDataset(bad_df, self.tok_en, self.tok_or)


class TestDynamicCollation(unittest.TestCase):
    """Tests for collate_translation_samples and TranslationBatch generation."""

    def test_dynamic_padding_and_tensor_shapes(self):
        # 3 samples with different sequence lengths
        s1 = TranslationSample(src_ids=[10, 20, EOS_ID], tgt_input_ids=[SOS_ID, 30], tgt_output_ids=[30, EOS_ID], src_text="s1", tgt_text="t1")
        s2 = TranslationSample(src_ids=[15, EOS_ID], tgt_input_ids=[SOS_ID, 35, 45, 55], tgt_output_ids=[35, 45, 55, EOS_ID], src_text="s2", tgt_text="t2")
        s3 = TranslationSample(src_ids=[11, 21, 31, 41, EOS_ID], tgt_input_ids=[SOS_ID, 60], tgt_output_ids=[60, EOS_ID], src_text="s3", tgt_text="t3")

        batch = collate_translation_samples([s1, s2, s3], pad_id=PAD_ID)

        # Max src_len = 5 (s3), Max tgt_len = 4 (s2)
        self.assertEqual(batch.src.shape, (3, 5))
        self.assertEqual(batch.tgt_input.shape, (3, 4))
        self.assertEqual(batch.tgt_output.shape, (3, 4))
        self.assertEqual(batch.src_mask.shape, (3, 1, 1, 5))
        self.assertEqual(batch.tgt_mask.shape, (3, 1, 4, 4))

        # Check padding on s1: src_ids=[10, 20, EOS_ID, PAD, PAD]
        self.assertEqual(batch.src[0].tolist(), [10, 20, EOS_ID, PAD_ID, PAD_ID])
        # Check padding on s1: tgt_input=[SOS, 30, PAD, PAD]
        self.assertEqual(batch.tgt_input[0].tolist(), [SOS_ID, 30, PAD_ID, PAD_ID])
        # Check padding on s1: tgt_output=[30, EOS, PAD, PAD]
        self.assertEqual(batch.tgt_output[0].tolist(), [30, EOS_ID, PAD_ID, PAD_ID])

        # Check lengths tracking
        self.assertEqual(batch.src_lengths, [3, 2, 5])
        self.assertEqual(batch.tgt_lengths, [2, 4, 2])

    def test_batch_device_transfer(self):
        s1 = TranslationSample(src_ids=[10, EOS_ID], tgt_input_ids=[SOS_ID, 30], tgt_output_ids=[30, EOS_ID])
        batch = collate_translation_samples([s1])
        batch_cpu = batch.to("cpu")
        self.assertEqual(batch_cpu.src.device.type, "cpu")
        self.assertEqual(batch_cpu.tgt_mask.device.type, "cpu")

    def test_collate_empty_list_raises(self):
        with self.assertRaises(ValueError):
            collate_translation_samples([])


class TestDataLoaderFactories(unittest.TestCase):
    """Tests for create_dataloader and create_dataloaders factory functions."""

    def setUp(self):
        self.tok_en = DummyTokenizer()
        self.tok_or = DummyTokenizer()
        self.df = pd.DataFrame({
            "idx": list(range(10)),
            "src": [f"Source sentence {i}" for i in range(10)],
            "tgt": [f"Target sentence {i}" for i in range(10)],
        })

    def test_create_dataloader_iteration(self):
        ds = TranslationDataset(self.df, self.tok_en, self.tok_or)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)

        batches = list(loader)
        self.assertEqual(len(batches), 3)  # 4 + 4 + 2 = 10 items
        self.assertEqual(batches[0].src.size(0), 4)
        self.assertEqual(batches[1].src.size(0), 4)
        self.assertEqual(batches[2].src.size(0), 2)
        self.assertIsInstance(batches[0], TranslationBatch)

    def test_create_dataloaders_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create synthetic train, val, test parquets
            self.df.to_parquet(tmp_path / "train.parquet")
            self.df.iloc[:2].to_parquet(tmp_path / "val.parquet")
            self.df.iloc[:2].to_parquet(tmp_path / "test.parquet")

            # Mock tokenizer model loading by patching load_tokenizer
            mock_en = self.tok_en
            mock_or = self.tok_or

            from unittest.mock import patch
            with patch("dataset.load_tokenizer", side_effect=[mock_en, mock_or]):
                train_loader, val_loader, test_loader = create_dataloaders(
                    data_dir=tmp_path,
                    en_tokenizer_path="dummy_en.model",
                    or_tokenizer_path="dummy_or.model",
                    batch_size=4,
                )

                self.assertTrue(isinstance(train_loader.sampler, torch.utils.data.RandomSampler))
                self.assertTrue(isinstance(val_loader.sampler, torch.utils.data.SequentialSampler))
                self.assertTrue(isinstance(test_loader.sampler, torch.utils.data.SequentialSampler))


class TestRealDataSmokeTest(unittest.TestCase):
    """Integration test using actual trained tokenizer models and outputs/ splits."""

    def test_real_tokenizer_and_parquet_integration(self):
        en_model_path = Path("outputs/tokenizer_sep_en16000_or32000_en.model")
        or_model_path = Path("outputs/tokenizer_sep_en16000_or32000_or.model")
        val_parquet_path = Path("outputs/val.parquet")

        if not (en_model_path.exists() and or_model_path.exists() and val_parquet_path.exists()):
            self.skipTest("Real tokenizer models or outputs/val.parquet not found; skipping real-data test.")

        tok_en = load_tokenizer(en_model_path)
        tok_or = load_tokenizer(or_model_path)

        # Verify special token contract on real tokenizers
        self.assertEqual(tok_en.piece_to_id("<PAD>"), PAD_ID)
        self.assertEqual(tok_en.piece_to_id("<SOS>"), SOS_ID)
        self.assertEqual(tok_en.piece_to_id("<EOS>"), EOS_ID)
        self.assertEqual(tok_en.piece_to_id("<UNK>"), UNK_ID)

        self.assertEqual(tok_or.piece_to_id("<PAD>"), PAD_ID)
        self.assertEqual(tok_or.piece_to_id("<SOS>"), SOS_ID)
        self.assertEqual(tok_or.piece_to_id("<EOS>"), EOS_ID)
        self.assertEqual(tok_or.piece_to_id("<UNK>"), UNK_ID)

        # Load small slice from validation split
        val_df = pd.read_parquet(val_parquet_path).head(16)
        ds = TranslationDataset(val_df, tok_en, tok_or, add_eos_to_source=True)

        loader = create_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))

        self.assertIsInstance(batch, TranslationBatch)
        self.assertEqual(batch.src.size(0), 8)
        self.assertEqual(batch.tgt_input.size(0), 8)
        self.assertEqual(batch.tgt_output.size(0), 8)

        # Verify teacher forcing alignment on real batch
        for i in range(8):
            self.assertEqual(batch.tgt_input[i, 0].item(), SOS_ID)
            # Find last non-PAD index in tgt_output
            tgt_len = batch.tgt_lengths[i]
            self.assertEqual(batch.tgt_output[i, tgt_len - 1].item(), EOS_ID)
            # Verify shifted tokens match
            input_tokens = batch.tgt_input[i, 1:tgt_len].tolist()
            output_tokens = batch.tgt_output[i, 0:tgt_len-1].tolist()
            self.assertEqual(input_tokens, output_tokens)


if __name__ == "__main__":
    unittest.main()
