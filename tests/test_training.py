"""Unit tests for Phase 5 Training Pipeline, Warmup Scheduler, Checkpointing & Validation."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    TranslationBatch,
    collate_translation_samples,
    TranslationSample,
)
from tokenizer import PAD_ID, SOS_ID, EOS_ID
from transformer import TranslationTransformer
from training import (
    TrainingConfig,
    TransformerWarmupScheduler,
    Trainer,
    load_checkpoint,
    save_checkpoint,
)


class TestTrainingConfig(unittest.TestCase):
    """Tests for TrainingConfig dataclass and dynamic device resolution."""

    def test_default_config(self):
        config = TrainingConfig()
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.warmup_steps, 4000)
        self.assertIsNone(config.device)
        # Resolved device should be torch.device
        resolved = config.resolved_device()
        self.assertIsInstance(resolved, torch.device)

    def test_dict_serialization_roundtrip(self):
        config = TrainingConfig(batch_size=16, learning_rate=0.5, device="cpu")
        d = config.to_dict()
        self.assertEqual(d["batch_size"], 16)
        self.assertEqual(d["learning_rate"], 0.5)
        self.assertEqual(d["device"], "cpu")

        reconstructed = TrainingConfig.from_dict(d)
        self.assertEqual(reconstructed.batch_size, 16)
        self.assertEqual(reconstructed.learning_rate, 0.5)
        self.assertEqual(reconstructed.resolved_device().type, "cpu")


class TestTransformerWarmupScheduler(unittest.TestCase):
    """Tests for TransformerWarmupScheduler formula and step progression."""

    def setUp(self):
        self.dummy_param = nn.Parameter(torch.zeros(10))
        self.optimizer = AdamW([self.dummy_param], lr=1.0)
        self.d_model = 128
        self.warmup_steps = 4000
        self.learning_rate = 1.0

    def test_warmup_formula_progression(self):
        scheduler = TransformerWarmupScheduler(
            self.optimizer,
            d_model=self.d_model,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
        )

        # Step 1: linear warmup phase
        # lr(1) = 1.0 * (128**-0.5) * min(1**-0.5, 1 * 4000**-1.5)
        #       = (128**-0.5) * (4000**-1.5)
        expected_lr_1 = (self.d_model ** -0.5) * (1 * (self.warmup_steps ** -1.5))
        self.assertAlmostEqual(scheduler.get_lr(1), expected_lr_1, places=7)

        # Step 4000 (peak): 4000**-0.5 == 4000 * 4000**-1.5
        expected_peak_lr = (self.d_model ** -0.5) * (self.warmup_steps ** -0.5)
        self.assertAlmostEqual(scheduler.get_lr(4000), expected_peak_lr, places=7)

        # Step 8000 (decay): 8000**-0.5
        expected_decay_lr = (self.d_model ** -0.5) * (8000 ** -0.5)
        self.assertAlmostEqual(scheduler.get_lr(8000), expected_decay_lr, places=7)

        # Confirm peak is greater than step 1 and greater than step 8000
        self.assertGreater(expected_peak_lr, expected_lr_1)
        self.assertGreater(expected_peak_lr, expected_decay_lr)

    def test_stepping_and_state_dict(self):
        scheduler = TransformerWarmupScheduler(
            self.optimizer,
            d_model=self.d_model,
            warmup_steps=100,
        )
        self.assertEqual(scheduler.current_step, 0)
        scheduler.step()
        self.assertEqual(scheduler.current_step, 1)
        scheduler.step()
        self.assertEqual(scheduler.current_step, 2)

        state = scheduler.state_dict()
        self.assertEqual(state["current_step"], 2)

        new_optimizer = AdamW([self.dummy_param], lr=1.0)
        new_scheduler = TransformerWarmupScheduler(new_optimizer, d_model=self.d_model, warmup_steps=100)
        new_scheduler.load_state_dict(state)
        self.assertEqual(new_scheduler.current_step, 2)
        self.assertEqual(new_scheduler.get_current_lr(), scheduler.get_current_lr())


class TestCheckpointing(unittest.TestCase):
    """Tests for checkpoint saving, loading, and parameter fidelity."""

    def test_save_and_load_roundtrip(self):
        model = TranslationTransformer(
            src_vocab_size=30,
            tgt_vocab_size=40,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = TransformerWarmupScheduler(optimizer, d_model=32, warmup_steps=100)
        scheduler.current_step = 50
        config = TrainingConfig(batch_size=4, num_epochs=5, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"

            save_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=2,
                global_step=50,
                val_loss=3.25,
                val_ppl=math.exp(3.25),
                config=config,
            )

            # Create a fresh model with different initial weights
            new_model = TranslationTransformer(
                src_vocab_size=30,
                tgt_vocab_size=40,
                d_model=32,
                num_heads=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
            )
            new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = TransformerWarmupScheduler(new_optimizer, d_model=32, warmup_steps=100)

            loaded_state = load_checkpoint(
                filepath=ckpt_path,
                model=new_model,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
            )

            self.assertEqual(loaded_state["epoch"], 2)
            self.assertEqual(loaded_state["global_step"], 50)
            self.assertAlmostEqual(loaded_state["val_loss"], 3.25)
            self.assertEqual(new_scheduler.current_step, 50)

            # Assert parameter tensors are exactly equal
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
                self.assertTrue(torch.equal(p1, p2), f"Parameter {n1} mismatch after checkpoint load")


class TestTrainerExecution(unittest.TestCase):
    """Tests for Trainer train_step, validate, and parameter updates."""

    def setUp(self):
        self.model = TranslationTransformer(
            src_vocab_size=30,
            tgt_vocab_size=40,
            d_model=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            pad_id=PAD_ID,
        )
        self.config = TrainingConfig(
            batch_size=2,
            learning_rate=1.0,
            warmup_steps=50,
            device="cpu",
            use_amp=False,
        )
        self.trainer = Trainer(self.model, self.config)

    def _create_synthetic_batch(self) -> TranslationBatch:
        s1 = TranslationSample(src_ids=[5, 10, EOS_ID], tgt_input_ids=[SOS_ID, 7, 8], tgt_output_ids=[7, 8, EOS_ID])
        s2 = TranslationSample(src_ids=[12, EOS_ID], tgt_input_ids=[SOS_ID, 15], tgt_output_ids=[15, EOS_ID])
        return collate_translation_samples([s1, s2], pad_id=PAD_ID)

    def test_train_step_updates_parameters(self):
        batch = self._create_synthetic_batch()

        # Clone an initial parameter tensor
        initial_param = self.model.encoder.src_embed.weight.clone()

        loss, lr = self.trainer.train_step(batch)

        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss))
        self.assertFalse(math.isinf(loss))
        self.assertGreater(lr, 0.0)
        self.assertEqual(self.trainer.global_step, 1)

        # Check that parameter changed after optimizer step
        updated_param = self.model.encoder.src_embed.weight
        self.assertFalse(torch.equal(initial_param, updated_param))

    def test_validate_step_finite(self):
        batch = self._create_synthetic_batch()
        loader = [batch]

        val_loss, val_ppl = self.trainer.validate(loader)

        self.assertIsInstance(val_loss, float)
        self.assertFalse(math.isnan(val_loss))
        self.assertFalse(math.isinf(val_loss))
        self.assertGreater(val_ppl, 1.0)


if __name__ == "__main__":
    unittest.main()
