#!/usr/bin/env python
"""Run the full Phase 2B tokenizer experiment matrix.

Executes all 20 configurations (16 separate + 4 shared), evaluates on
train/validation/test splits, and saves results to outputs/.
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tokenizer as tk


def main():
    # Paths
    train_path = Path("outputs/train.parquet")
    val_path = Path("outputs/val.parquet")
    test_path = Path("outputs/test.parquet")
    out_dir = Path("outputs")

    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: {val_path} not found")
        sys.exit(1)
    if not test_path.exists():
        print(f"ERROR: {test_path} not found")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment configs (20 total)
    configs = tk.generate_experiment_configs()
    print(f"Total configurations: {len(configs)}")
    for c in configs:
        print(f"  {c.config_id()}")

    # Run all experiments
    results = tk.run_all_experiments(configs, train_path, val_path, test_path, out_dir)

    # Save machine-readable results
    results_path = out_dir / "tokenizer_experiments.json"
    tk.save_experiment_results(results, results_path)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"{'Config':<25} {'Mode':<10} {'Params':>12} {'EN UNK%':>8} {'OR UNK%':>8} {'EN Seq P99':>10} {'OR Seq P99':>10}")
    print("-" * 95)
    for r in results:
        c = r.config
        if c.mode == "separate":
            en_unk = r.metrics_train["en"]["unk_rate"]
            or_unk = r.metrics_train["or"]["unk_rate"]
            en_p99 = r.metrics_train["en"]["sequence_lengths"]["p99"]
            or_p99 = r.metrics_train["or"]["sequence_lengths"]["p99"]
            params = f"{r.parameter_cost/1e6:.2f}M"
        else:
            en_unk = r.metrics_train["shared"]["src"]["unk_rate"]
            or_unk = r.metrics_train["shared"]["tgt"]["unk_rate"]
            en_p99 = r.metrics_train["shared"]["src"]["sequence_lengths"]["p99"]
            or_p99 = r.metrics_train["shared"]["tgt"]["sequence_lengths"]["p99"]
            params = f"{r.parameter_cost/1e6:.2f}M"
        print(f"{c.config_id():<25} {c.mode:<10} {params:>12} {en_unk:>7.2f}% {or_unk:>7.2f}% {en_p99:>10} {or_p99:>10}")

    print("\nAll 20 experiments completed successfully.")


if __name__ == "__main__":
    main()