import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from benchmark import BenchmarkResult, TrainingBenchmark, validate_checkpoint_resumption
from dataset import TranslationBatch
from tokenizer import PAD_ID
from training import Trainer, TrainingConfig
from transformer import TranslationTransformer


class SyntheticTranslationDataset(Dataset):
    """Generates synthetic TranslationBatch objects for fast unit testing."""

    def __init__(self, size: int = 16, seq_len: int = 10, src_vocab: int = 100, tgt_vocab: int = 100):
        self.size = size
        self.seq_len = seq_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        return {
            "src": torch.randint(4, self.src_vocab, (self.seq_len,)),
            "tgt": torch.randint(4, self.tgt_vocab, (self.seq_len,)),
        }


def synthetic_collate_fn(batch_items: list[dict]) -> TranslationBatch:
    batch_size = len(batch_items)
    seq_len = batch_items[0]["src"].size(0)

    src = torch.stack([b["src"] for b in batch_items])
    tgt = torch.stack([b["tgt"] for b in batch_items])

    tgt_input = tgt[:, :-1].contiguous()
    tgt_output = tgt[:, 1:].contiguous()
    tgt_seq_len = tgt_input.size(1)

    src_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    tgt_mask = torch.ones(batch_size, 1, tgt_seq_len, tgt_seq_len, dtype=torch.bool)
    src_pad_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    tgt_pad_mask = torch.ones(batch_size, 1, tgt_seq_len, dtype=torch.bool)
    causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    return TranslationBatch(
        src=src,
        tgt_input=tgt_input,
        tgt_output=tgt_output,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        src_pad_mask=src_pad_mask,
        tgt_pad_mask=tgt_pad_mask,
        causal_mask=causal_mask,
        src_lengths=[seq_len] * batch_size,
        tgt_lengths=[tgt_seq_len] * batch_size,
    )


class TestBenchmarkEngine(unittest.TestCase):
    """Test suite for TrainingBenchmark throughput calculation and epoch estimation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.src_vocab = 50
        self.tgt_vocab = 50
        self.model = TranslationTransformer(
            src_vocab_size=self.src_vocab,
            tgt_vocab_size=self.tgt_vocab,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            pad_id=PAD_ID,
        )
        self.config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            learning_rate=1.0,
            warmup_steps=10,
            checkpoint_dir=str(Path(self.temp_dir) / "checkpoints"),
            log_dir=str(Path(self.temp_dir) / "logs"),
            use_amp=False,
            device="cpu",
        )
        self.trainer = Trainer(model=self.model, config=self.config)

        dataset = SyntheticTranslationDataset(size=16, seq_len=8, src_vocab=self.src_vocab, tgt_vocab=self.tgt_vocab)
        self.train_loader = DataLoader(dataset, batch_size=4, collate_fn=synthetic_collate_fn)
        self.val_loader = DataLoader(dataset, batch_size=4, collate_fn=synthetic_collate_fn)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_epoch_duration_estimation_math(self) -> None:
        estimates = TrainingBenchmark.estimate_epoch_duration(
            train_samples=1000,
            val_samples=200,
            batch_size=8,
            avg_train_step_sec=0.05,
            avg_val_step_sec=0.02,
            epochs_estimate=10,
        )
        # 1000 / 8 = 125 batches -> 125 * 0.05 = 6.25 sec
        self.assertEqual(estimates["train_batches"], 125)
        self.assertAlmostEqual(estimates["train_epoch_sec"], 6.25)
        self.assertAlmostEqual(estimates["train_epoch_min"], 6.25 / 60.0)
        self.assertAlmostEqual(estimates["train_epoch_hours"], 6.25 / 3600.0)

        # 200 / 8 = 25 batches -> 25 * 0.02 = 0.5 sec
        self.assertEqual(estimates["val_batches"], 25)
        self.assertAlmostEqual(estimates["val_epoch_sec"], 0.5)

        # Total epoch sec = 6.25 + 0.5 = 6.75 sec
        self.assertAlmostEqual(estimates["total_epoch_hours"], 6.75 / 3600.0)
        self.assertAlmostEqual(estimates["total_training_hours"], (6.75 / 3600.0) * 10)

    def test_benchmark_execution_and_result_dataclass(self) -> None:
        engine = TrainingBenchmark(
            model=self.model,
            trainer=self.trainer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            pad_id=PAD_ID,
        )

        res = engine.run_benchmark(
            warmup_steps=2,
            benchmark_steps=3,
            val_steps=2,
            epochs_estimate=5,
        )

        self.assertIsInstance(res, BenchmarkResult)
        self.assertEqual(res.training_steps, 3)
        self.assertGreater(res.training_elapsed_seconds, 0.0)
        self.assertGreater(res.training_steps_per_second, 0.0)
        self.assertGreater(res.training_samples_per_second, 0.0)
        self.assertGreater(res.training_tokens_per_second, 0.0)
        self.assertEqual(res.validation_steps, 2)
        self.assertGreater(res.validation_samples_per_second, 0.0)
        self.assertGreater(res.estimated_training_epoch_seconds, 0.0)

        # Ensure dictionary conversion is valid JSON serializable
        dict_res = res.to_dict()
        self.assertIn("average_step_time_ms", dict_res)
        self.assertIn("median_step_time_ms", dict_res)

    def test_checkpoint_resume_validation_helper(self) -> None:
        ckpt_path = Path(self.temp_dir) / "checkpoints" / "test_resume.pt"

        def make_model():
            return TranslationTransformer(
                src_vocab_size=self.src_vocab,
                tgt_vocab_size=self.tgt_vocab,
                d_model=32,
                num_heads=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                d_ff=64,
                dropout=0.0,
                pad_id=PAD_ID,
            )

        resume_result = validate_checkpoint_resumption(
            model_fn=make_model,
            config=self.config,
            train_loader=self.train_loader,
            checkpoint_path=ckpt_path,
            steps_initial=3,
            steps_resume=1,
        )

        self.assertEqual(resume_result["saved_step"], 3)
        self.assertEqual(resume_result["restored_step"], 3)
        self.assertEqual(resume_result["continued_step"], 4)
        self.assertTrue(resume_result["parameter_update_verified"])


if __name__ == "__main__":
    unittest.main()
