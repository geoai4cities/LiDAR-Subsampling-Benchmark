#!/usr/bin/env python3
"""
PTv3 Training Script for SemanticKITTI Dataset

Manages training experiments for SemanticKITTI with all subsampling methods.

Methods: RS, IDIS, FPS, DBSCAN, Voxel, Poisson, DEPOCO
Loss levels: 0, 10, 30, 50, 70, 90
Seeds: 1, 2, 3 (deterministic methods use seed 1 only)

Usage:
    # Train all experiments
    python train_experiments.py --all

    # Train specific tier
    python train_experiments.py --tier tier1  # RS, IDIS
    python train_experiments.py --tier tier2  # FPS, DBSCAN, Voxel, Poisson, DEPOCO

    # Train specific method
    python train_experiments.py --method RS --loss 50 --seed 1

    # List all experiments
    python train_experiments.py --list
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent
PTv3_ROOT = DATASET_DIR.parent
BENCHMARK_ROOT = PTv3_ROOT.parent

DATASET_NAME = "SemanticKITTI"
CONFIG_DIR = DATASET_DIR / "configs" / "semantickitti" / "generated"
POINTCEPT_DIR = PTv3_ROOT / "pointcept"

# All methods available
ALL_METHODS = ["RS", "IDIS", "FPS", "DBSCAN", "Voxel", "Poisson", "DEPOCO"]
STOCHASTIC_METHODS = ["RS", "IDIS", "FPS", "DBSCAN", "Poisson"]
DETERMINISTIC_METHODS = ["Voxel", "DEPOCO"]

# All loss levels
ALL_LOSS_LEVELS = [0, 10, 30, 50, 70, 90]

# Seeds
ALL_SEEDS = [1, 2, 3]

# Tier definitions
TIER1_METHODS = ["RS", "IDIS"]
TIER2_METHODS = ["FPS", "DBSCAN", "Voxel", "Poisson", "DEPOCO"]


# =============================================================================
# Experiment Management
# =============================================================================

def get_config_path(method: str, loss: int, seed: int) -> Path:
    """Get config file path for an experiment."""
    config_name = f"ptv3_semantickitti_{method}_loss{loss}_seed{seed}.py"
    return CONFIG_DIR / config_name


def config_exists(method: str, loss: int, seed: int) -> bool:
    """Check if config file exists."""
    return get_config_path(method, loss, seed).exists()


def get_all_experiments(methods=None, loss_levels=None, seeds=None, include_baseline=True):
    """Generate list of all experiments.

    Note: 0% loss (baseline) is trained ONCE per dataset, not per method.
    The 0% baseline uses original data without any subsampling, so the
    training result is identical regardless of which method is claimed.
    """
    methods = methods or ALL_METHODS
    loss_levels = loss_levels or ALL_LOSS_LEVELS
    seeds = seeds or ALL_SEEDS

    experiments = []
    baseline_added = False

    for method in methods:
        for loss in loss_levels:
            # 0% loss: Train once per dataset (use first method's config as baseline)
            if loss == 0:
                if not baseline_added and include_baseline:
                    # Only add ONE 0% baseline experiment per dataset
                    if config_exists(method, loss, 1):
                        experiments.append({
                            'method': 'baseline',  # Mark as baseline
                            'loss': 0,
                            'seed': 1,
                            'config': get_config_path(method, loss, 1),
                            'note': '0% baseline (applies to all methods)'
                        })
                        baseline_added = True
                continue  # Skip other 0% configs

            # Non-zero loss levels: Per method, per seed
            method_seeds = [1] if method in DETERMINISTIC_METHODS else seeds
            for seed in method_seeds:
                if config_exists(method, loss, seed):
                    experiments.append({
                        'method': method,
                        'loss': loss,
                        'seed': seed,
                        'config': get_config_path(method, loss, seed)
                    })

    return experiments


def list_experiments(experiments):
    """Print list of experiments."""
    print(f"\n{'='*70}")
    print(f"SemanticKITTI Experiments ({len(experiments)} total)")
    print(f"{'='*70}")

    # First show baseline if present
    baseline_exps = [e for e in experiments if e['method'] == 'baseline']
    if baseline_exps:
        print(f"\n0% Baseline (1 experiment - applies to ALL methods):")
        for exp in baseline_exps:
            status = "EXISTS" if exp['config'].exists() else "MISSING"
            print(f"  - loss0_seed1 [{status}] (original data, no subsampling)")

    # Group by method
    by_method = {}
    for exp in experiments:
        if exp['method'] == 'baseline':
            continue
        method = exp['method']
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(exp)

    for method in ALL_METHODS:
        if method in by_method:
            exps = by_method[method]
            print(f"\n{method} ({len(exps)} experiments):")
            for exp in exps:
                status = "EXISTS" if exp['config'].exists() else "MISSING"
                print(f"  - loss{exp['loss']}_seed{exp['seed']} [{status}]")

    print(f"\n{'='*70}\n")


def run_experiment(method: str, loss: int, seed: int, gpu_id: int = 0, dry_run: bool = False):
    """Run a single experiment."""
    config_path = get_config_path(method, loss, seed)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return False

    exp_name = f"ptv3_semantickitti_{method}_loss{loss}_seed{seed}"

    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"Config: {config_path}")
    print(f"GPU: {gpu_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    if dry_run:
        print("DRY RUN - Skipping actual training")
        return True

    # Build command
    cmd = [
        "python",
        str(POINTCEPT_DIR / "tools" / "train.py"),
        "--config-file", str(config_path),
        "--num-gpus", "1",
        "--options",
        f"save_path={BENCHMARK_ROOT}/experiments/checkpoints/ptv3/SemanticKITTI/{exp_name}"
    ]

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(POINTCEPT_DIR)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_experiments(experiments, gpu_id: int = 0, dry_run: bool = False):
    """Run multiple experiments sequentially."""
    total = len(experiments)
    completed = 0
    failed = 0

    print(f"\nStarting {total} experiments on GPU {gpu_id}")
    print(f"{'='*70}")

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {exp['method']}_loss{exp['loss']}_seed{exp['seed']}")

        success = run_experiment(
            method=exp['method'],
            loss=exp['loss'],
            seed=exp['seed'],
            gpu_id=gpu_id,
            dry_run=dry_run
        )

        if success:
            completed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY: {completed} completed, {failed} failed out of {total}")
    print(f"{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"PTv3 Training for {DATASET_NAME}"
    )

    # Actions
    parser.add_argument("--list", action="store_true",
                       help="List all available experiments")
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--tier", choices=["tier1", "tier2"],
                       help="Run experiments for specific tier")

    # Specific experiment
    parser.add_argument("--method", choices=ALL_METHODS,
                       help="Specific method to train")
    parser.add_argument("--loss", type=int, choices=ALL_LOSS_LEVELS,
                       help="Specific loss level")
    parser.add_argument("--seed", type=int, choices=ALL_SEEDS,
                       help="Specific seed")

    # Options
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")

    args = parser.parse_args()

    # Determine which experiments to run
    if args.list:
        experiments = get_all_experiments()
        list_experiments(experiments)
        return

    if args.method and args.loss is not None and args.seed is not None:
        # Single experiment
        run_experiment(
            method=args.method,
            loss=args.loss,
            seed=args.seed,
            gpu_id=args.gpu,
            dry_run=args.dry_run
        )
        return

    if args.tier:
        methods = TIER1_METHODS if args.tier == "tier1" else TIER2_METHODS
        experiments = get_all_experiments(methods=methods)
    elif args.all:
        experiments = get_all_experiments()
    else:
        parser.print_help()
        return

    if not experiments:
        print("No experiments found. Generate configs first:")
        print(f"  python {SCRIPT_DIR}/generate_configs.py --tier all")
        return

    # Confirm before running
    print(f"\nReady to run {len(experiments)} experiments")
    print(f"GPU: {args.gpu}")

    if not args.dry_run:
        response = input("\nStart training? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    run_experiments(experiments, gpu_id=args.gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
