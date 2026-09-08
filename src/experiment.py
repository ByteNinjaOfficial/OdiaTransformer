"""Phase 7A: Experiment Management and Reproducibility for OdiaTransformer.

Provides structured experiment lifecycle management:
- Directory tree management (experiments/<experiment_name>/)
- Overwrite protection
- Reproducibility metadata extraction (hardware, environment, git, model/training configs)
- JSON serialization of configurations, metadata, and benchmark results
- Standardized Path accessors
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def get_git_info() -> Dict[str, Optional[str]]:
    """Safely query current Git commit hash and branch name.

    Returns:
        Dictionary with 'commit_hash' and 'branch', or 'unknown' if Git is unavailable.
    """
    info = {"commit_hash": "unknown", "branch": "unknown"}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        info["commit_hash"] = commit
    except Exception:
        pass

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        info["branch"] = branch
    except Exception:
        pass

    return info


def capture_environment_metadata(
    seed: Optional[int] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Union[Dict[str, Any], Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture comprehensive environment, hardware, and reproducibility metadata.

    Args:
        seed: Random seed used for reproducibility.
        dataset_stats: Dataset statistics (e.g. sample counts).
        model_config: Model architectural parameters.
        training_config: Training configuration dictionary or dataclass.
        extra_metadata: Optional additional metadata fields.

    Returns:
        JSON-serializable dictionary containing all environment metadata.
    """
    git_info = get_git_info()

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None

    gpu_info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }

    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    else:
        gpu_info["fallback"] = "CPU execution (CUDA unavailable)"

    # Serialize training_config if it is a dataclass or custom object
    serializable_train_cfg = None
    if training_config is not None:
        if hasattr(training_config, "to_dict"):
            serializable_train_cfg = training_config.to_dict()
        elif is_dataclass(training_config):
            serializable_train_cfg = asdict(training_config)
        elif isinstance(training_config, dict):
            serializable_train_cfg = training_config
        else:
            serializable_train_cfg = str(training_config)

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "system": {
            "os": os.name,
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "pytorch_version": torch.__version__,
        },
        "hardware": gpu_info,
        "reproducibility": {
            "random_seed": seed,
        },
        "dataset": dataset_stats or {},
        "model_config": model_config or {},
        "training_config": serializable_train_cfg or {},
    }

    if extra_metadata:
        metadata["extra"] = extra_metadata

    return metadata


class ExperimentManager:
    """Manages an isolated experiment directory, metadata, configs, and artifacts."""

    def __init__(
        self,
        experiment_name: str,
        base_dir: Union[str, Path] = "experiments",
        allow_overwrite: bool = False,
    ) -> None:
        """Initialize and create experiment directory tree.

        Args:
            experiment_name: Name of the experiment.
            base_dir: Base directory under which experiment subfolders are placed.
            allow_overwrite: If False, raises FileExistsError when experiment dir exists.
        """
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")

        self.experiment_name = experiment_name.strip()
        self.base_dir = Path(base_dir).resolve()
        self.root_dir = self.base_dir / self.experiment_name
        self.allow_overwrite = allow_overwrite

        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.metrics_dir = self.root_dir / "metrics"
        self.logs_dir = self.root_dir / "logs"

        self.config_path = self.root_dir / "config.json"
        self.metadata_path = self.root_dir / "metadata.json"
        self.benchmark_path = self.root_dir / "benchmark.json"
        self.training_log_path = self.logs_dir / "training.log"

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Create directory hierarchy safely."""
        if self.root_dir.exists() and not self.allow_overwrite:
            # Check if directory already has files
            if any(self.root_dir.iterdir()):
                raise FileExistsError(
                    f"Experiment directory already exists: '{self.root_dir}'. "
                    "Set allow_overwrite=True to reuse or choose a new experiment_name."
                )

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Union[Dict[str, Any], Any]) -> Path:
        """Save configuration dictionary or dataclass to config.json."""
        if hasattr(config, "to_dict"):
            data = config.to_dict()
        elif is_dataclass(config):
            data = asdict(config)
        elif isinstance(config, dict):
            data = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return self.config_path

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save metadata dictionary to metadata.json."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return self.metadata_path

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata from metadata.json."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_benchmark(self, benchmark_data: Dict[str, Any]) -> Path:
        """Save benchmark metrics to benchmark.json."""
        with open(self.benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)
        return self.benchmark_path

    def load_benchmark(self) -> Dict[str, Any]:
        """Load benchmark results from benchmark.json."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_checkpoint_path(self, filename: str = "latest.pt") -> Path:
        """Return the absolute path to a checkpoint within this experiment."""
        return self.checkpoints_dir / filename

    def get_metrics_path(self, filename: str = "training_history.json") -> Path:
        """Return the absolute path to a metrics file within this experiment."""
        return self.metrics_dir / filename

