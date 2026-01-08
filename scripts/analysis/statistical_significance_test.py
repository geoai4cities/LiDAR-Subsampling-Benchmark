#!/usr/bin/env python3
"""
Statistical Significance Test for Multi-Seed PTv3 Experiments

This script:
1. Reads pre-extracted metrics from docs/tables/inference_on_original
2. Computes mean and standard deviation for each method
3. Performs Welch's t-test for pairwise comparisons
4. Saves results to analysis_results directory

Usage:
    python statistical_significance_test.py [--loss LOSS]
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Configuration
PROJECT_ROOT = Path("/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark")
EXTRACTED_DATA_DIR = PROJECT_ROOT / "docs/tables/inference_on_original"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results"


def parse_metrics_file(file_path: Path) -> dict:
    """
    Parse individual metrics file to extract mIoU, mAcc, allAcc.

    Format:
        mIoU:    0.5104
        mAcc:    0.5956
        allAcc:  0.8666
    """
    metrics = {"mIoU": None, "mAcc": None, "allAcc": None}

    if not file_path.exists():
        print(f"  Warning: File not found: {file_path}")
        return metrics

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse mIoU
        miou_match = re.search(r'mIoU:\s+([\d.]+)', content)
        if miou_match:
            metrics["mIoU"] = float(miou_match.group(1))

        # Parse mAcc
        macc_match = re.search(r'mAcc:\s+([\d.]+)', content)
        if macc_match:
            metrics["mAcc"] = float(macc_match.group(1))

        # Parse allAcc
        allacc_match = re.search(r'allAcc:\s+([\d.]+)', content)
        if allacc_match:
            metrics["allAcc"] = float(allacc_match.group(1))

    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

    return metrics


def find_multi_seed_experiments(data_dir: Path, loss_level: int = 90) -> dict:
    """
    Find all multi-seed experiments for a given loss level from extracted metrics files.

    Returns dict: {method_name: {seed: metrics_dict}}
    """
    experiments = {}

    # Pattern for multi-seed files: METHOD_lossXX_seedN_metrics.txt
    seed_pattern = re.compile(rf'(\w+)_loss{loss_level}_seed(\d+)_metrics\.txt$')

    # Pattern for single-run files: METHOD_lossXX_metrics.txt
    single_pattern = re.compile(rf'(\w+)_loss{loss_level}_metrics\.txt$')

    for file_path in data_dir.iterdir():
        if not file_path.is_file() or not file_path.name.endswith('_metrics.txt'):
            continue

        # Try multi-seed pattern first
        match = seed_pattern.match(file_path.name)
        if match:
            method = match.group(1)
            seed = int(match.group(2))
        else:
            # Try single-run pattern
            match = single_pattern.match(file_path.name)
            if match:
                method = match.group(1)
                seed = 1  # Default seed for single runs
            else:
                continue

        # Map method names for consistency
        if method == "Poisson":
            method = "SB"
        elif method == "Voxel":
            method = "VB"

        if method not in experiments:
            experiments[method] = {}

        metrics = parse_metrics_file(file_path)

        if metrics["mIoU"] is not None:
            experiments[method][seed] = metrics
            print(f"  Found: {method} seed{seed} -> mIoU={metrics['mIoU']:.4f}")
        else:
            print(f"  Warning: No metrics found in {file_path.name}")

    return experiments


def compute_statistics(experiments: dict) -> dict:
    """
    Compute mean and std for each method.
    """
    stats_results = {}

    for method, seeds in experiments.items():
        if len(seeds) < 2:
            print(f"  Warning: {method} has only {len(seeds)} seed(s), skipping statistics")
            continue

        miou_values = [seeds[s]["mIoU"] for s in sorted(seeds.keys())]

        stats_results[method] = {
            "seeds": sorted(seeds.keys()),
            "mIoU_values": miou_values,
            "mIoU_mean": np.mean(miou_values),
            "mIoU_std": np.std(miou_values, ddof=1),  # Sample std
            "n_seeds": len(miou_values)
        }

    return stats_results


def perform_significance_tests(stats_results: dict, alpha: float = 0.05) -> list:
    """
    Perform Welch's t-test for all pairwise comparisons.
    """
    tests = []

    methods = list(stats_results.keys())

    for m1, m2 in combinations(methods, 2):
        s1 = stats_results[m1]
        s2 = stats_results[m2]

        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(
            s1["mIoU_values"],
            s2["mIoU_values"],
            equal_var=False
        )

        tests.append({
            "comparison": f"{m1} vs {m2}",
            "method1": m1,
            "method2": m2,
            "method1_mean": s1["mIoU_mean"],
            "method2_mean": s2["mIoU_mean"],
            "t_statistic": abs(t_stat),
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha
        })

    return tests


def save_results(experiments: dict, stats_results: dict, tests: list,
                 loss_level: int, output_dir: Path):
    """
    Save results to JSON and generate markdown table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Prepare JSON output
    results = {
        "metadata": {
            "loss_level": loss_level,
            "test_type": "Welch's t-test",
            "alpha": 0.05,
            "timestamp": timestamp,
            "data_source": str(EXTRACTED_DATA_DIR)
        },
        "multi_seed_results": {},
        "statistics": {},
        "significance_tests": tests
    }

    # Add per-method results
    for method, seeds in experiments.items():
        results["multi_seed_results"][method] = {
            f"seed{s}": {"mIoU": seeds[s]["mIoU"]} for s in sorted(seeds.keys())
        }

    for method, st in stats_results.items():
        results["statistics"][method] = {
            "mIoU_mean": st["mIoU_mean"],
            "mIoU_std": st["mIoU_std"],
            "n_seeds": st["n_seeds"]
        }

    # Save JSON
    json_path = output_dir / f"significance_test_loss{loss_level}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved JSON: {json_path}")

    # Generate Markdown table
    md_lines = [
        f"# Statistical Significance Test Results (Loss {loss_level}%)",
        f"\nGenerated: {timestamp}\n",
        "## Multi-Seed Results (Test → Original Data)\n",
        "| Method | Seed 1 | Seed 2 | Seed 3 | Mean ± Std |",
        "|--------|--------|--------|--------|------------|"
    ]

    for method in sorted(stats_results.keys()):
        st = stats_results[method]
        seeds = experiments[method]

        seed_vals = [f"{seeds.get(i, {}).get('mIoU', 'N/A'):.4f}" if i in seeds else "N/A"
                     for i in [1, 2, 3]]

        md_lines.append(
            f"| {method} | {seed_vals[0]} | {seed_vals[1]} | {seed_vals[2]} | "
            f"**{st['mIoU_mean']:.4f} ± {st['mIoU_std']:.4f}** |"
        )

    md_lines.extend([
        "",
        "## Statistical Significance Testing (Welch's t-test)\n",
        "| Comparison | t-statistic | p-value | Significant? |",
        "|------------|-------------|---------|--------------|"
    ])

    for test in tests:
        sig_mark = "✓ Yes" if test["significant"] else "✗ No"
        p_str = f"< 0.001" if test["p_value"] < 0.001 else f"{test['p_value']:.4f}"
        sig_str = f"{sig_mark} (p < 0.01)" if test["p_value"] < 0.01 else f"{sig_mark}"

        md_lines.append(
            f"| {test['comparison']} | {test['t_statistic']:.2f} | {p_str} | {sig_str} |"
        )

    md_lines.extend([
        "",
        "## Interpretation\n",
        "All pairwise comparisons with p < 0.05 indicate statistically significant "
        "differences between methods. Standard deviations below 0.01 mIoU indicate "
        "excellent reproducibility across random seeds."
    ])

    md_path = output_dir / f"significance_test_loss{loss_level}.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved Markdown: {md_path}")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance test for multi-seed PTv3 experiments"
    )
    parser.add_argument(
        "--loss", type=int, default=90,
        help="Loss level to analyze (default: 90)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Statistical Significance Test - Loss {args.loss}%")
    print("=" * 70)
    print(f"\nSource: {EXTRACTED_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Find experiments
    print("Finding multi-seed experiments...")
    experiments = find_multi_seed_experiments(EXTRACTED_DATA_DIR, args.loss)

    if not experiments:
        print("No multi-seed experiments found!")
        return

    print(f"\nFound {len(experiments)} methods with multi-seed data")

    # Compute statistics
    print("\nComputing statistics...")
    stats_results = compute_statistics(experiments)

    # Perform significance tests
    print("\nPerforming Welch's t-tests...")
    tests = perform_significance_tests(stats_results)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nMulti-Seed Results:")
    for method, st in sorted(stats_results.items()):
        print(f"  {method}: {st['mIoU_mean']:.4f} ± {st['mIoU_std']:.4f} (n={st['n_seeds']})")

    print("\nSignificance Tests:")
    for test in tests:
        sig = "SIGNIFICANT" if test["significant"] else "not significant"
        print(f"  {test['comparison']}: t={test['t_statistic']:.2f}, p={test['p_value']:.4f} ({sig})")

    # Save results
    save_results(experiments, stats_results, tests, args.loss, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
