"""Unit tests for train.py CLI parsing and configuration construction."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import parse_args


class TestTrainCLI(unittest.TestCase):
    """Tests for train.py command-line argument parsing."""

    def test_default_arguments(self):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
            self.assertEqual(args.epochs, 10)
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.learning_rate, 1.0)
            self.assertEqual(args.warmup_steps, 4000)
            self.assertEqual(args.early_stopping_patience, 3)
            self.assertEqual(args.seed, 42)
            self.assertIsNone(args.resume)
            self.assertIsNone(args.max_steps)
            self.assertFalse(args.no_amp)

    def test_custom_arguments(self):
        cmd = [
            "train.py",
            "--epochs", "20",
            "--batch-size", "16",
            "--learning-rate", "0.5",
            "--warmup-steps", "2000",
            "--early-stopping-patience", "5",
            "--max-steps", "100",
            "--resume", "checkpoints/latest.pt",
            "--no-amp",
            "--device", "cpu",
        ]
        with patch("sys.argv", cmd):
            args = parse_args()
            self.assertEqual(args.epochs, 20)
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.learning_rate, 0.5)
            self.assertEqual(args.warmup_steps, 2000)
            self.assertEqual(args.early_stopping_patience, 5)
            self.assertEqual(args.max_steps, 100)
            self.assertEqual(args.resume, "checkpoints/latest.pt")
            self.assertTrue(args.no_amp)
            self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
