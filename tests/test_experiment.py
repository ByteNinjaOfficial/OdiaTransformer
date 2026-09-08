import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import (
    ExperimentManager,
    capture_environment_metadata,
    get_git_info,
)


@dataclass
class DummyConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    model_name: str = "toy_transformer"

    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "model_name": self.model_name,
        }


class TestExperimentManager(unittest.TestCase):
    """Test suite for ExperimentManager directory structure, safety, and persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_creation(self) -> None:
        mgr = ExperimentManager(
            experiment_name="exp_test_01",
            base_dir=self.temp_dir,
            allow_overwrite=False,
        )
        self.assertTrue(mgr.root_dir.exists())
        self.assertTrue(mgr.checkpoints_dir.exists())
        self.assertTrue(mgr.metrics_dir.exists())
        self.assertTrue(mgr.logs_dir.exists())
        self.assertEqual(mgr.get_checkpoint_path("latest.pt"), mgr.checkpoints_dir / "latest.pt")
        self.assertEqual(mgr.get_metrics_path("history.json"), mgr.metrics_dir / "history.json")

    def test_overwrite_protection(self) -> None:
        # Create first time
        mgr1 = ExperimentManager(
            experiment_name="exp_protected",
            base_dir=self.temp_dir,
            allow_overwrite=False,
        )
        # Put a file inside so it's not considered an empty dir
        (mgr1.root_dir / "dummy.txt").write_text("data")

        # Second creation should raise FileExistsError
        with self.assertRaises(FileExistsError):
            ExperimentManager(
                experiment_name="exp_protected",
                base_dir=self.temp_dir,
                allow_overwrite=False,
            )

        # But with allow_overwrite=True, it should succeed
        mgr2 = ExperimentManager(
            experiment_name="exp_protected",
            base_dir=self.temp_dir,
            allow_overwrite=True,
        )
        self.assertEqual(mgr1.root_dir, mgr2.root_dir)

    def test_empty_experiment_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentManager(experiment_name="", base_dir=self.temp_dir)
        with self.assertRaises(ValueError):
            ExperimentManager(experiment_name="   ", base_dir=self.temp_dir)

    def test_config_serialization(self) -> None:
        mgr = ExperimentManager(
            experiment_name="exp_config",
            base_dir=self.temp_dir,
        )
        cfg = DummyConfig(batch_size=32, learning_rate=0.0005)
        saved_path = mgr.save_config(cfg)
        self.assertTrue(saved_path.exists())

        loaded = mgr.load_config()
        self.assertEqual(loaded["batch_size"], 32)
        self.assertEqual(loaded["learning_rate"], 0.0005)
        self.assertEqual(loaded["model_name"], "toy_transformer")

    def test_metadata_serialization(self) -> None:
        mgr = ExperimentManager(
            experiment_name="exp_meta",
            base_dir=self.temp_dir,
        )
        meta = {"version": "1.0", "author": "tester", "seed": 42}
        saved_path = mgr.save_metadata(meta)
        self.assertTrue(saved_path.exists())

        loaded = mgr.load_metadata()
        self.assertEqual(loaded, meta)

    def test_benchmark_serialization(self) -> None:
        mgr = ExperimentManager(
            experiment_name="exp_bench",
            base_dir=self.temp_dir,
        )
        bench_data = {"steps_per_sec": 45.2, "peak_vram_mb": 380.5}
        saved_path = mgr.save_benchmark(bench_data)
        self.assertTrue(saved_path.exists())

        loaded = mgr.load_benchmark()
        self.assertEqual(loaded["steps_per_sec"], 45.2)
        self.assertEqual(loaded["peak_vram_mb"], 380.5)


class TestReproducibilityMetadata(unittest.TestCase):
    """Test suite for reproducibility metadata extraction and graceful fallbacks."""

    def test_metadata_structure_and_types(self) -> None:
        dataset_stats = {"train_samples": 975000, "val_samples": 9949}
        model_cfg = {"d_model": 128, "heads": 4}
        train_cfg = {"batch_size": 8, "lr": 1.0}

        meta = capture_environment_metadata(
            seed=42,
            dataset_stats=dataset_stats,
            model_config=model_cfg,
            training_config=train_cfg,
        )

        self.assertIn("timestamp", meta)
        self.assertIn("git", meta)
        self.assertIn("system", meta)
        self.assertIn("hardware", meta)
        self.assertIn("reproducibility", meta)
        self.assertIn("dataset", meta)
        self.assertIn("model_config", meta)
        self.assertIn("training_config", meta)

        self.assertEqual(meta["reproducibility"]["random_seed"], 42)
        self.assertEqual(meta["dataset"]["train_samples"], 975000)
        self.assertEqual(meta["model_config"]["d_model"], 128)
        self.assertEqual(meta["training_config"]["batch_size"], 8)

        # Verify JSON serializability
        json_str = json.dumps(meta)
        self.assertIsInstance(json_str, str)

    def test_git_info_fallback(self) -> None:
        with patch("subprocess.check_output", side_effect=Exception("Git not found")):
            info = get_git_info()
            self.assertEqual(info["commit_hash"], "unknown")
            self.assertEqual(info["branch"], "unknown")


if __name__ == "__main__":
    unittest.main()
