#!/usr/bin/env python3
"""
Generate PTv3 experiment configs for SemanticKITTI dataset.

This script generates all experiment configurations from templates by replacing
placeholders with actual experiment parameters.

Usage:
    python generate_configs.py [--tier1] [--tier2] [--all]

Options:
    --tier1: Generate Tier 1 configs only (3 seeds, 3 loss levels) [default]
    --tier2: Generate Tier 2 configs (5 seeds, 6 loss levels)
    --all: Generate all configs
"""

import os
import argparse
from pathlib import Path


def generate_config(
    template_path,
    output_path,
    method,
    loss,
    seed,
    data_root,
    gpu_type="40gb"
):
    """
    Generate a single config from template.

    Args:
        template_path: Path to template config file
        output_path: Path to output config file
        method: Subsampling method (RS, IDIS, FPS, DBSCAN)
        loss: Loss level (0, 10, 30, 50, 70, 90)
        seed: Random seed (1, 2, 3, 4, 5)
        data_root: Path to data directory
        gpu_type: GPU configuration (40gb or 140gb)
    """
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Replace placeholders
    config_content = template_content.replace(
        "DATA_ROOT_PLACEHOLDER", data_root
    ).replace(
        "{METHOD}", method
    ).replace(
        "{LOSS}", str(loss)
    ).replace(
        "{SEED}", str(seed)
    ).replace(
        "seed=42,  # WILL BE REPLACED", f"seed={seed},"
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Generated: {output_path.name}")


def generate_all_configs(
    base_dir,
    tier="tier1",
    gpu_types=["40gb", "140gb"]
):
    """
    Generate all experiment configs.

    Args:
        base_dir: Base directory (SemanticKITTI or DALES)
        tier: "tier1" or "tier2" or "all"
        gpu_types: List of GPU types to generate configs for
    """
    base_dir = Path(base_dir)
    dataset_name = base_dir.name  # "SemanticKITTI" or "DALES"

    # Define experiment parameters
    # All 7 methods: RS, IDIS, FPS, DBSCAN, Voxel, Poisson, DEPOCO
    ALL_METHODS = ["RS", "IDIS", "FPS", "DBSCAN", "Voxel", "Poisson", "DEPOCO"]

    # All loss levels available
    # Subsampled: 10, 30, 50, 70, 90 (with different seeds)
    # Original: 0 (no subsampling, uses original data)
    ALL_LOSS_LEVELS = [0, 10, 30, 50, 70, 90]
    SUBSAMPLED_LOSS_LEVELS = [10, 30, 50, 70, 90]

    # Stochastic methods need multiple seeds, deterministic (Voxel, DEPOCO) needs only 1
    STOCHASTIC_METHODS = ["RS", "IDIS", "FPS", "DBSCAN", "Poisson"]
    DETERMINISTIC_METHODS = ["Voxel", "DEPOCO"]

    if tier == "tier1":
        # Tier 1: RS and IDIS only (core comparison) - all loss levels
        methods = ["RS", "IDIS"]
        loss_levels = ALL_LOSS_LEVELS  # 0, 10, 30, 50, 70, 90
        seeds = [1, 2, 3]
        print(f"Generating Tier 1 configs for {dataset_name}...")
        print(f"Methods: {methods}")
        print(f"Loss levels: {loss_levels}")
        print(f"Seeds: {seeds}")
    elif tier == "tier2":
        # Tier 2: Baseline methods (FPS, DBSCAN, Voxel, Poisson, DEPOCO) - all loss levels
        methods = ["FPS", "DBSCAN", "Voxel", "Poisson", "DEPOCO"]
        loss_levels = ALL_LOSS_LEVELS  # 0, 10, 30, 50, 70, 90
        seeds = [1, 2, 3]
        print(f"Generating Tier 2 configs for {dataset_name}...")
        print(f"Methods: {methods}")
        print(f"Loss levels: {loss_levels}")
        print(f"Seeds: {seeds} (Voxel/DEPOCO use seed 1 only)")
    elif tier == "extended":
        # Extended: All methods with all loss levels
        methods = ALL_METHODS
        loss_levels = ALL_LOSS_LEVELS
        seeds = [1, 2, 3]
        print(f"Generating extended configs for {dataset_name}...")
        print(f"Methods: {methods}")
        print(f"Loss levels: {loss_levels}")
        print(f"Seeds: {seeds} (deterministic methods use seed 1 only)")
    else:  # all - same as extended
        methods = ALL_METHODS
        loss_levels = ALL_LOSS_LEVELS
        seeds = [1, 2, 3]
        print(f"Generating all configs for {dataset_name}...")
        print(f"Methods: {methods}")
        print(f"Loss levels: {loss_levels}")
        print(f"Seeds: {seeds} (deterministic methods use seed 1 only)")

    # Categorize methods
    stochastic_methods = [m for m in methods if m in STOCHASTIC_METHODS]
    deterministic_methods = [m for m in methods if m in DETERMINISTIC_METHODS]

    # Data root base path (relative to config file location)
    # Config location: PTv3/{dataset}/configs/{dataset}/generated/
    # Data location: data/{dataset}/subsampled/{method}_loss{loss}/
    data_root_base = f"../../../../../data/{dataset_name}/subsampled"

    total_configs = 0

    for gpu_type in gpu_types:
        # Template path
        template_path = base_dir / f"configs/{dataset_name.lower()}/ptv3_{gpu_type}_official_template.py"

        if not template_path.exists():
            print(f"⚠ Warning: Template not found: {template_path}")
            continue

        print(f"\n{'='*60}")
        print(f"GPU Type: {gpu_type.upper()}")
        print(f"{'='*60}")

        # Generate configs for stochastic methods (RS, IDIS, FPS, DBSCAN, Poisson)
        for method in stochastic_methods:
            for loss in loss_levels:
                for seed in seeds:
                    # Output path (without gpu_type suffix for simplicity)
                    output_path = base_dir / f"configs/{dataset_name.lower()}/generated/ptv3_{dataset_name.lower()}_{method}_loss{loss}_seed{seed}.py"

                    # Data root for this experiment
                    # Handle special case for 0% loss (uses original data)
                    if loss == 0:
                        data_root = f"../../../../../data/{dataset_name}/original"
                    else:
                        data_root = f"{data_root_base}/{method}_loss{loss}_seed{seed}"

                    # Generate config
                    generate_config(
                        template_path=template_path,
                        output_path=output_path,
                        method=method,
                        loss=loss,
                        seed=seed,
                        data_root=data_root,
                        gpu_type=gpu_type
                    )
                    total_configs += 1

        # Generate configs for deterministic methods (Voxel, DEPOCO) - single seed only
        for method in deterministic_methods:
            for loss in loss_levels:
                seed = 1  # Deterministic methods only need seed 1
                output_path = base_dir / f"configs/{dataset_name.lower()}/generated/ptv3_{dataset_name.lower()}_{method}_loss{loss}_seed{seed}.py"

                # Handle special case for 0% loss (uses original data)
                if loss == 0:
                    data_root = f"../../../../../data/{dataset_name}/original"
                else:
                    data_root = f"{data_root_base}/{method}_loss{loss}_seed{seed}"

                generate_config(
                    template_path=template_path,
                    output_path=output_path,
                    method=method,
                    loss=loss,
                    seed=seed,
                    data_root=data_root,
                    gpu_type=gpu_type
                )
                total_configs += 1

    print(f"\n{'='*60}")
    print(f"✅ Total configs generated: {total_configs}")
    print(f"Output directory: {base_dir}/configs/{dataset_name.lower()}/generated/")
    print(f"{'='*60}\n")

    # Create summary file
    summary_path = base_dir / f"configs/{dataset_name.lower()}/generated/CONFIG_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(f"# Generated Config Summary - {dataset_name}\n\n")
        f.write(f"**Generated:** {total_configs} configs\n")
        f.write(f"**Tier:** {tier}\n")
        f.write(f"**GPU Types:** {', '.join(gpu_types)}\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"- **Methods:** {', '.join(methods)}\n")
        f.write(f"- **Loss levels:** {', '.join(map(str, loss_levels))}\n")
        f.write(f"- **Seeds:** {', '.join(map(str, seeds))}\n\n")
        f.write(f"## Config Naming Convention\n\n")
        f.write(f"```\n")
        f.write(f"ptv3_{dataset_name.lower()}_{{METHOD}}_loss{{LOSS}}_seed{{SEED}}_{{GPU}}.py\n")
        f.write(f"```\n\n")
        f.write(f"**Examples:**\n")
        f.write(f"- `ptv3_{dataset_name.lower()}_RS_loss0_seed1_40gb.py`\n")
        f.write(f"- `ptv3_{dataset_name.lower()}_IDIS_loss50_seed2_140gb.py`\n")

    print(f"📄 Summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTv3 experiment configs for SemanticKITTI"
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "extended", "all"],
        default="tier1",
        help="Which tier of configs to generate: tier1 (RS,IDIS), tier2 (baselines), extended/all (all methods, all loss levels)"
    )
    parser.add_argument(
        "--gpu",
        choices=["40gb", "140gb", "both"],
        default="both",
        help="GPU configuration (default: both)"
    )

    args = parser.parse_args()

    # Determine base directory (where this script is located)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # PTv3/SemanticKITTI/

    # Determine GPU types
    if args.gpu == "both":
        gpu_types = ["40gb", "140gb"]
    else:
        gpu_types = [args.gpu]

    # Generate configs
    generate_all_configs(
        base_dir=base_dir,
        tier=args.tier,
        gpu_types=gpu_types
    )


if __name__ == "__main__":
    main()
