"""
Aggregate results from multiple experimental runs (different seeds).

This script computes mean ± SD for JS divergence across multiple runs
to address reviewer concerns about sampling variability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ss_utils import read_json, write_json


def load_metrics_from_run(metrics_path: str) -> Dict[str, float]:
    """
    Load JS divergence values from a metrics.json file.

    Returns:
        Dict mapping variable name to JS divergence (ln)
    """
    metrics = read_json(metrics_path)
    js_values = {}
    for var_name, var_data in metrics.items():
        if "js_divergence_ln" in var_data:
            js_values[var_name] = var_data["js_divergence_ln"]
    return js_values


def aggregate_runs(run_dirs: List[str], model_name: str) -> pd.DataFrame:
    """
    Aggregate JS divergence across multiple runs.

    Args:
        run_dirs: List of result directories (e.g., ["results/gpt52_experiment_seed42", ...])
        model_name: Model identifier for reporting

    Returns:
        DataFrame with columns: variable, run1, run2, ..., mean, std, cv
    """
    all_runs = []

    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            print(f"⚠️  Skipping {run_dir}: metrics.json not found")
            continue

        js_values = load_metrics_from_run(str(metrics_path))
        all_runs.append(js_values)
        print(f"✅ Loaded {run_dir}: {len(js_values)} variables")

    if len(all_runs) < 2:
        raise ValueError(f"Need at least 2 runs, got {len(all_runs)}")

    # Get all variables (should be consistent across runs)
    variables = sorted(all_runs[0].keys())

    # Build DataFrame
    data = []
    for var in variables:
        row = {"variable": var, "model": model_name}

        # Collect JS divergence from each run
        js_values = [run[var] for run in all_runs if var in run]

        # Add individual run values
        for i, js_val in enumerate(js_values, 1):
            row[f"run{i}"] = js_val

        # Compute statistics
        row["mean"] = np.mean(js_values)
        row["std"] = np.std(js_values, ddof=1)  # Sample std
        row["cv"] = (row["std"] / row["mean"]) * 100 if row["mean"] > 0 else 0  # Coefficient of variation (%)
        row["n_runs"] = len(js_values)

        data.append(row)

    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple experimental runs")
    parser.add_argument("--gpt52-runs", nargs="+", required=True,
                        help="List of GPT-5.2 result directories")
    parser.add_argument("--clova-runs", nargs="+", required=False,
                        help="List of CLOVA result directories")
    parser.add_argument("--out-dir", default="results/aggregated",
                        help="Output directory for aggregated results")

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Aggregate GPT-5.2 runs
    print("\n=== Aggregating GPT-5.2 Runs ===")
    gpt_df = aggregate_runs(args.gpt52_runs, "GPT-5.2")

    # Save GPT-5.2 results
    gpt_csv = Path(args.out_dir) / "gpt52_aggregated.csv"
    gpt_df.to_csv(gpt_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved GPT-5.2 aggregated results: {gpt_csv}")

    # Display summary
    print("\n📊 GPT-5.2 Summary (JS Divergence)")
    print(gpt_df[["variable", "mean", "std", "cv", "n_runs"]].to_string(index=False))

    # Aggregate CLOVA runs if provided
    if args.clova_runs:
        print("\n=== Aggregating CLOVA Runs ===")
        clova_df = aggregate_runs(args.clova_runs, "CLOVA HCX-007")

        clova_csv = Path(args.out_dir) / "clova_aggregated.csv"
        clova_df.to_csv(clova_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved CLOVA aggregated results: {clova_csv}")

        print("\n📊 CLOVA Summary (JS Divergence)")
        print(clova_df[["variable", "mean", "std", "cv", "n_runs"]].to_string(index=False))

        # Create comparison table
        print("\n=== Model Comparison (Mean JS Divergence) ===")
        comparison = gpt_df[["variable", "mean", "std"]].copy()
        comparison.columns = ["variable", "gpt_mean", "gpt_std"]
        comparison["clova_mean"] = clova_df["mean"].values
        comparison["clova_std"] = clova_df["std"].values
        comparison["improvement"] = ((comparison["gpt_mean"] - comparison["clova_mean"]) / comparison["gpt_mean"] * 100).round(1)
        comparison["winner"] = comparison["improvement"].apply(lambda x: "CLOVA" if x > 0 else "GPT" if x < 0 else "Tie")

        comparison_csv = Path(args.out_dir) / "model_comparison.csv"
        comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved comparison: {comparison_csv}")
        print(comparison.to_string(index=False))

        # Count wins
        clova_wins = (comparison["winner"] == "CLOVA").sum()
        gpt_wins = (comparison["winner"] == "GPT").sum()
        ties = (comparison["winner"] == "Tie").sum()
        print(f"\n🏆 Winner Count: CLOVA {clova_wins}/{len(comparison)}, GPT {gpt_wins}/{len(comparison)}, Ties {ties}/{len(comparison)}")


if __name__ == "__main__":
    main()
