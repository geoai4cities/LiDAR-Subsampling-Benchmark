#!/usr/bin/env python3
"""
Generate PTv3 Experiment Configs for SemanticKITTI Subsampling Benchmark

This script generates all experiment configurations from templates by replacing
placeholders with actual experiment parameters.

================================================================================
METHOD PROPERTIES
================================================================================

Deterministic methods (no seed needed - same input always produces same output):
  - IDIS, IDIS_R5, IDIS_R15, IDIS_R20: Inverse Distance Importance Sampling
  - DBSCAN: Density-based clustering with centroid selection
  - Voxel: Voxel grid downsampling

Non-deterministic methods (seed required for reproducibility):
  - RS: Random Sampling
  - FPS: Farthest Point Sampling
  - Poisson: Poisson Disk Sampling

Special:
  - DEPOCO: External model (treated as deterministic)

================================================================================
CONFIGURATION STRUCTURE
================================================================================

Loss Levels (6 total):
  - 0%: Original data (no subsampling) - baseline
  - 10%, 30%, 50%, 70%, 90%: Subsampled data

Seeds (for non-deterministic methods only):
  - Default: 1
  - Extended: 1, 2, 3 (for statistical significance)

GPU Types (2):
  - 40gb: For smaller GPUs (batch_size=12)
  - 140gb: For H200/H100 (batch_size=20)

================================================================================
OUTPUT NAMING CONVENTION
================================================================================

Deterministic methods:
  - Config: ptv3_semantickitti_{METHOD}_loss{LOSS}_140gb.py
  - Data:   data/SemanticKITTI/subsampled/{METHOD}_loss{LOSS}/

Non-deterministic methods:
  - Config: ptv3_semantickitti_{METHOD}_loss{LOSS}_seed{SEED}_140gb.py
  - Data:   data/SemanticKITTI/subsampled/{METHOD}_loss{LOSS}_seed{SEED}/

================================================================================
USAGE
================================================================================

  # Generate default configs (seed 1 for non-deterministic, both GPU types)
  python generate_configs.py

  # Generate with specific seeds (for non-deterministic methods)
  python generate_configs.py --seeds 1 2 3

  # Generate for specific GPU type
  python generate_configs.py --gpu 140gb

  # Generate specific tier
  python generate_configs.py --tier tier1 --seeds 1 2 3

  # Generate configs and profile them for GFLOPs/memory
  python generate_configs.py --profile --gpu 140gb

  # Profile specific config only
  python generate_configs.py --profile-only --config path/to/config.py

================================================================================
TIERS
================================================================================

  priority: All methods, seed 1 only (default)
  tier1:    RS, IDIS only (core comparison)
  tier2:    FPS, DBSCAN, Voxel, Poisson, DEPOCO (baselines)
  all:      All methods

================================================================================
PROFILING
================================================================================

  The --profile flag runs model profiling to measure:
  - GFLOPs (Floating point operations)
  - GPU Memory consumption (peak training memory)
  - Model parameters count
  - Inference/training time estimates

  Results are saved to: configs/semantickitti/generated/PROFILING_RESULTS.md
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


# Method classification
DETERMINISTIC_METHODS = ["IDIS", "IDIS_R5", "IDIS_R15", "IDIS_R20", "DBSCAN", "Voxel", "DEPOCO"]
NON_DETERMINISTIC_METHODS = ["RS", "FPS", "Poisson"]
ALL_METHODS = DETERMINISTIC_METHODS + NON_DETERMINISTIC_METHODS

# All loss levels
ALL_LOSS_LEVELS = [0, 10, 30, 50, 70, 90]

# IDIS method - only generated for loss 50, 70, 90
IDIS_LOSS_LEVELS = [50, 70, 90]

# IDIS R ablation variants - only generated for high loss levels
IDIS_R_VARIANTS = ["IDIS_R5", "IDIS_R15", "IDIS_R20"]
IDIS_R_LOSS_LEVELS = [90]  # Only generate R variants for loss 90


def is_deterministic(method):
    """Check if a method is deterministic (doesn't need seed)."""
    return method in DETERMINISTIC_METHODS


def generate_config(
    template_path,
    output_path,
    method,
    loss,
    seed,
    data_root,
    is_deterministic_method,
):
    """
    Generate a single config from template.

    Args:
        template_path: Path to template config file
        output_path: Path to output config file
        method: Subsampling method
        loss: Loss level (0, 10, 30, 50, 70, 90)
        seed: Random seed (1, 2, 3) - only used for non-deterministic methods
        data_root: Path to data directory (relative to pointcept/)
        is_deterministic_method: Whether the method is deterministic
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
    )

    # Handle seed placeholder based on method type
    if is_deterministic_method:
        # Deterministic: remove seed from experiment name
        config_content = config_content.replace(
            "_seed{SEED}", ""
        ).replace(
            "{SEED}", ""
        ).replace(
            "seed=42,  # WILL BE REPLACED", "seed=42,"
        )
    else:
        # Non-deterministic: include seed
        config_content = config_content.replace(
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
    tier="priority",
    gpu_types=["40gb", "140gb"],
    seeds=[1]
):
    """
    Generate all experiment configs.

    Args:
        base_dir: Base directory (SemanticKITTI)
        tier: "priority", "tier1", "tier2", or "all"
        gpu_types: List of GPU types to generate configs for
        seeds: List of seeds for non-deterministic methods (default: [1])
    """
    base_dir = Path(base_dir)
    dataset_name = base_dir.name  # "SemanticKITTI"

    # =========================================================================
    # Select methods based on tier
    # =========================================================================

    if tier == "priority":
        methods = ALL_METHODS
        loss_levels = ALL_LOSS_LEVELS
        print(f"\n{'='*60}")
        print(f"PRIORITY TIER: All methods")
        print(f"{'='*60}")
    elif tier == "tier1":
        # Core comparison: RS vs IDIS (including R variants)
        methods = ["RS", "IDIS", "IDIS_R5", "IDIS_R15", "IDIS_R20"]
        loss_levels = ALL_LOSS_LEVELS
        print(f"\n{'='*60}")
        print(f"TIER 1: Core comparison (RS vs IDIS + R ablation)")
        print(f"{'='*60}")
    elif tier == "tier2":
        # Baseline methods
        methods = ["FPS", "DBSCAN", "Voxel", "Poisson", "DEPOCO"]
        loss_levels = ALL_LOSS_LEVELS
        print(f"\n{'='*60}")
        print(f"TIER 2: Baseline methods")
        print(f"{'='*60}")
    else:  # all
        methods = ALL_METHODS
        loss_levels = ALL_LOSS_LEVELS
        print(f"\n{'='*60}")
        print(f"ALL: Complete configuration set")
        print(f"{'='*60}")

    # Categorize selected methods
    deterministic_methods = [m for m in methods if is_deterministic(m)]
    non_deterministic_methods = [m for m in methods if not is_deterministic(m)]

    print(f"\nDataset:     {dataset_name}")
    print(f"Methods:     {', '.join(methods)} ({len(methods)} total)")
    print(f"  Deterministic (no seed):     {', '.join(deterministic_methods) or 'None'}")
    print(f"  Non-deterministic (seed):    {', '.join(non_deterministic_methods) or 'None'}")
    print(f"Loss levels: {', '.join(map(str, loss_levels))} ({len(loss_levels)} total)")
    print(f"Seeds:       {', '.join(map(str, seeds))} (for non-deterministic only)")
    print(f"GPU types:   {', '.join(gpu_types)} ({len(gpu_types)} total)")

    # Calculate expected config count
    # Deterministic: method × loss_levels × 1 (no seed variations)
    # Non-deterministic: method × loss_levels × seeds
    # Special: IDIS only for loss 50, 70, 90; IDIS R variants only for loss 70, 90
    regular_deterministic = [m for m in deterministic_methods if m not in IDIS_R_VARIANTS and m != "IDIS"]
    idis_method = "IDIS" in deterministic_methods
    idis_r_methods = [m for m in deterministic_methods if m in IDIS_R_VARIANTS]

    regular_det_configs = len(regular_deterministic) * len(loss_levels)
    idis_configs = len([l for l in loss_levels if l in IDIS_LOSS_LEVELS]) if idis_method else 0
    idis_r_configs = len(idis_r_methods) * len([l for l in loss_levels if l in IDIS_R_LOSS_LEVELS])
    deterministic_configs = regular_det_configs + idis_configs + idis_r_configs
    non_deterministic_configs = len(non_deterministic_methods) * len(loss_levels) * len(seeds)
    configs_per_gpu = deterministic_configs + non_deterministic_configs
    total_expected = configs_per_gpu * len(gpu_types)

    print(f"\nExpected configs:")
    print(f"  Deterministic (regular): {len(regular_deterministic)} × {len(loss_levels)} = {regular_det_configs}")
    if idis_method:
        print(f"  IDIS:                    1 × {len(IDIS_LOSS_LEVELS)} = {idis_configs} (loss 50, 70, 90 only)")
    if idis_r_methods:
        print(f"  IDIS R variants:         {len(idis_r_methods)} × {len(IDIS_R_LOSS_LEVELS)} = {idis_r_configs} (loss 70, 90 only)")
    print(f"  Non-deterministic:       {len(non_deterministic_methods)} × {len(loss_levels)} × {len(seeds)} = {non_deterministic_configs}")
    print(f"  Per GPU type:            {configs_per_gpu}")
    print(f"  Total:                   {configs_per_gpu} × {len(gpu_types)} = {total_expected}")

    # =========================================================================
    # Generate configs
    # =========================================================================

    data_root_base = f"../../data/{dataset_name}/subsampled"
    total_configs = 0

    for gpu_type in gpu_types:
        # Template path
        template_path = base_dir / f"configs/{dataset_name.lower()}/ptv3_{gpu_type}_official_template.py"

        if not template_path.exists():
            print(f"\n⚠ Warning: Template not found: {template_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"GPU Type: {gpu_type.upper()}")
        print(f"Template: {template_path.name}")
        print(f"{'─'*60}")

        # Generate configs for deterministic methods (no seed in path)
        for method in deterministic_methods:
            for loss in loss_levels:
                # Skip IDIS for loss levels other than 50, 70, 90
                if method == "IDIS" and loss not in IDIS_LOSS_LEVELS:
                    continue

                # Skip IDIS R variants (R5, R15) for loss levels other than 70 and 90
                if method in IDIS_R_VARIANTS and loss not in IDIS_R_LOSS_LEVELS:
                    continue

                # Output path - no seed suffix
                output_path = base_dir / f"configs/{dataset_name.lower()}/generated/ptv3_{dataset_name.lower()}_{method}_loss{loss}_{gpu_type}.py"

                # Data root: loss=0 uses original, others use subsampled (no seed)
                if loss == 0:
                    data_root = f"../../data/{dataset_name}/original"
                else:
                    data_root = f"{data_root_base}/{method}_loss{loss}"

                generate_config(
                    template_path=template_path,
                    output_path=output_path,
                    method=method,
                    loss=loss,
                    seed=None,
                    data_root=data_root,
                    is_deterministic_method=True,
                )
                total_configs += 1

        # Generate configs for non-deterministic methods (seed in path)
        for method in non_deterministic_methods:
            for loss in loss_levels:
                for seed in seeds:
                    # Output path - with seed suffix
                    output_path = base_dir / f"configs/{dataset_name.lower()}/generated/ptv3_{dataset_name.lower()}_{method}_loss{loss}_seed{seed}_{gpu_type}.py"

                    # Data root: loss=0 uses original, others use subsampled with seed
                    if loss == 0:
                        data_root = f"../../data/{dataset_name}/original"
                    else:
                        data_root = f"{data_root_base}/{method}_loss{loss}_seed{seed}"

                    generate_config(
                        template_path=template_path,
                        output_path=output_path,
                        method=method,
                        loss=loss,
                        seed=seed,
                        data_root=data_root,
                        is_deterministic_method=False,
                    )
                    total_configs += 1

    # =========================================================================
    # Summary
    # =========================================================================

    output_dir = base_dir / f"configs/{dataset_name.lower()}/generated/"

    print(f"\n{'='*60}")
    print(f"✅ Total configs generated: {total_configs}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Create summary file
    summary_path = output_dir / "CONFIG_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(f"# Generated Config Summary - {dataset_name}\n\n")
        f.write(f"**Generated:** {total_configs} configs\n")
        f.write(f"**Tier:** {tier}\n")
        f.write(f"**GPU Types:** {', '.join(gpu_types)}\n")
        f.write(f"**Seeds:** {', '.join(map(str, seeds))} (for non-deterministic methods)\n\n")

        f.write(f"## Method Properties\n\n")
        f.write(f"| Method | Deterministic | Seed Required | Output Directory |\n")
        f.write(f"|--------|---------------|---------------|------------------|\n")
        for m in methods:
            det = "Yes" if is_deterministic(m) else "No"
            seed_req = "No" if is_deterministic(m) else "Yes"
            if is_deterministic(m):
                out_dir = f"`{m}_loss{{XX}}/`"
            else:
                out_dir = f"`{m}_loss{{XX}}_seed{{N}}/`"
            f.write(f"| {m} | {det} | {seed_req} | {out_dir} |\n")

        f.write(f"\n## Configuration Breakdown\n\n")
        f.write(f"| Category | Methods | Loss Levels | Seeds | Configs/GPU |\n")
        f.write(f"|----------|---------|-------------|-------|-------------|\n")
        f.write(f"| Deterministic | {len(deterministic_methods)} | {len(loss_levels)} | N/A | {deterministic_configs} |\n")
        f.write(f"| Non-deterministic | {len(non_deterministic_methods)} | {len(loss_levels)} | {len(seeds)} | {non_deterministic_configs} |\n")
        f.write(f"| **Total** | {len(methods)} | {len(loss_levels)} | - | **{configs_per_gpu}** |\n\n")

        f.write(f"## Parameters\n\n")
        f.write(f"### Deterministic Methods ({len(deterministic_methods)} total)\n")
        f.write(f"- {', '.join(deterministic_methods) or 'None'}\n")
        f.write(f"- No seed needed - same input always produces same output\n\n")

        f.write(f"### Non-deterministic Methods ({len(non_deterministic_methods)} total)\n")
        f.write(f"- {', '.join(non_deterministic_methods) or 'None'}\n")
        f.write(f"- Seeds: {', '.join(map(str, seeds))}\n\n")

        f.write(f"### Loss Levels ({len(loss_levels)} total)\n")
        f.write(f"- **Baseline:** 0% (original data)\n")
        f.write(f"- **Subsampled:** {', '.join(map(str, [l for l in loss_levels if l != 0]))}%\n\n")

        f.write(f"## Config Naming Convention\n\n")
        f.write(f"### Deterministic Methods\n")
        f.write(f"```\n")
        f.write(f"ptv3_{dataset_name.lower()}_{{METHOD}}_loss{{LOSS}}_{{GPU}}.py\n")
        f.write(f"```\n\n")

        f.write(f"### Non-deterministic Methods\n")
        f.write(f"```\n")
        f.write(f"ptv3_{dataset_name.lower()}_{{METHOD}}_loss{{LOSS}}_seed{{SEED}}_{{GPU}}.py\n")
        f.write(f"```\n\n")

        f.write(f"**Examples:**\n")
        f.write(f"- `ptv3_{dataset_name.lower()}_IDIS_loss50_140gb.py` - IDIS (deterministic)\n")
        f.write(f"- `ptv3_{dataset_name.lower()}_RS_loss50_seed1_140gb.py` - RS (non-deterministic)\n")
        f.write(f"- `ptv3_{dataset_name.lower()}_FPS_loss50_seed2_140gb.py` - FPS (non-deterministic)\n\n")

        f.write(f"## Output Directory\n\n")
        f.write(f"```\n")
        f.write(f"{output_dir}\n")
        f.write(f"```\n")

    print(f"📄 Summary written to: {summary_path}")

    return list(output_dir.glob("*.py"))


def format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def profile_single_config(config_path, gpu_id=0, iterations=3, venv_path=None):
    """
    Profile a single config file for GFLOPs and memory.

    Args:
        config_path: Path to config file
        gpu_id: GPU ID to use
        iterations: Number of iterations for profiling
        venv_path: Path to virtual environment

    Returns:
        dict: Profiling results or None if failed
    """
    script_dir = Path(__file__).parent
    profile_script = script_dir / "profile_during_training.py"

    if not profile_script.exists():
        print(f"⚠ Profile script not found: {profile_script}")
        return None

    # Build command
    cmd = [
        sys.executable,
        str(profile_script),
        "--config", str(config_path),
        "--gpu", str(gpu_id),
        "--iterations", str(iterations),
    ]

    print(f"\n{'─'*60}")
    print(f"Profiling: {config_path.name}")
    print(f"{'─'*60}")

    try:
        # Run profiling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(script_dir),
        )

        if result.returncode != 0:
            print(f"⚠ Profiling failed for {config_path.name}")
            print(f"  Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")
            return None

        # Parse output for key metrics
        output = result.stdout
        metrics = {}

        # Extract metrics from output
        for line in output.split('\n'):
            if 'Trainable parameters:' in line:
                try:
                    params_str = line.split(':')[1].strip()
                    if '(' in params_str:
                        params_str = params_str.split('(')[0].strip()
                    metrics['parameters'] = params_str.replace(',', '')
                except (IndexError, ValueError):
                    pass
            elif 'GFLOPs (forward pass):' in line:
                try:
                    metrics['gflops'] = float(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
            elif 'Maximum peak memory:' in line or 'Peak training memory:' in line:
                try:
                    mem_str = line.split(':')[1].strip()
                    metrics['peak_memory'] = mem_str
                except (IndexError, ValueError):
                    pass
            elif 'Average iteration time:' in line:
                try:
                    time_str = line.split(':')[1].strip().replace('ms', '').strip()
                    metrics['iteration_time_ms'] = float(time_str)
                except (IndexError, ValueError):
                    pass
            elif 'Estimated training time:' in line:
                try:
                    hours_str = line.split(':')[1].strip().replace('hours', '').strip()
                    metrics['estimated_hours'] = float(hours_str)
                except (IndexError, ValueError):
                    pass

        if metrics:
            print(f"✓ Profiling complete")
            if 'gflops' in metrics:
                print(f"  GFLOPs: {metrics['gflops']:.2f}")
            if 'peak_memory' in metrics:
                print(f"  Peak Memory: {metrics['peak_memory']}")
            if 'iteration_time_ms' in metrics:
                print(f"  Iteration Time: {metrics['iteration_time_ms']:.2f} ms")
            return metrics
        else:
            print(f"⚠ Could not parse profiling output")
            return None

    except subprocess.TimeoutExpired:
        print(f"⚠ Profiling timed out for {config_path.name}")
        return None
    except Exception as e:
        print(f"⚠ Profiling error: {e}")
        return None


def profile_configs(config_paths, output_dir, gpu_id=0, sample_only=True):
    """
    Profile multiple config files and generate a summary report.

    Args:
        config_paths: List of config file paths
        output_dir: Directory to save results
        gpu_id: GPU ID to use
        sample_only: If True, only profile one config per method (faster)
    """
    print(f"\n{'='*60}")
    print("MODEL PROFILING")
    print(f"{'='*60}")

    if not config_paths:
        print("No config files to profile")
        return

    # Filter to sample configs if requested
    if sample_only:
        # Select one config per method at loss=50 (representative)
        seen_methods = set()
        filtered_configs = []
        for config_path in config_paths:
            name = config_path.stem
            # Extract method from name: ptv3_semantickitti_{METHOD}_loss{LOSS}...
            parts = name.split('_')
            if len(parts) >= 4:
                method = parts[2]  # Method is typically the 3rd part
                if method not in seen_methods and 'loss50' in name:
                    seen_methods.add(method)
                    filtered_configs.append(config_path)

        # If no loss50 found, take first config per method
        if not filtered_configs:
            for config_path in config_paths:
                name = config_path.stem
                parts = name.split('_')
                if len(parts) >= 4:
                    method = parts[2]
                    if method not in seen_methods:
                        seen_methods.add(method)
                        filtered_configs.append(config_path)

        config_paths = filtered_configs
        print(f"Profiling {len(config_paths)} representative configs (one per method)")
    else:
        print(f"Profiling all {len(config_paths)} configs")

    results = {}
    start_time = time.time()

    for config_path in config_paths:
        metrics = profile_single_config(config_path, gpu_id=gpu_id)
        if metrics:
            results[config_path.name] = metrics

    elapsed_time = time.time() - start_time

    # Generate report
    report_path = output_dir / "PROFILING_RESULTS.md"
    json_path = output_dir / "profiling_results.json"

    print(f"\n{'='*60}")
    print("PROFILING SUMMARY")
    print(f"{'='*60}")

    with open(report_path, 'w') as f:
        f.write("# PTv3 Model Profiling Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Configs Profiled:** {len(results)}\n")
        f.write(f"**Profiling Time:** {elapsed_time:.1f}s\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Config | Parameters | GFLOPs | Peak Memory | Iter Time (ms) | Est. Hours |\n")
        f.write("|--------|------------|--------|-------------|----------------|------------|\n")

        for config_name, metrics in sorted(results.items()):
            params = metrics.get('parameters', 'N/A')
            gflops = f"{metrics['gflops']:.2f}" if 'gflops' in metrics else 'N/A'
            memory = metrics.get('peak_memory', 'N/A')
            iter_time = f"{metrics['iteration_time_ms']:.2f}" if 'iteration_time_ms' in metrics else 'N/A'
            est_hours = f"{metrics['estimated_hours']:.1f}" if 'estimated_hours' in metrics else 'N/A'

            # Shorten config name for table
            short_name = config_name.replace('ptv3_semantickitti_', '').replace('.py', '')
            f.write(f"| {short_name} | {params} | {gflops} | {memory} | {iter_time} | {est_hours} |\n")

            print(f"  {short_name}: {gflops} GFLOPs, {memory}")

        f.write("\n## Notes\n\n")
        f.write("- **GFLOPs**: Floating point operations for forward pass (billions)\n")
        f.write("- **Peak Memory**: Maximum GPU memory during training (forward + backward + optimizer)\n")
        f.write("- **Iter Time**: Average time per training iteration\n")
        f.write("- **Est. Hours**: Estimated total training time for default epochs\n\n")
        f.write("## Hardware\n\n")
        f.write("Profiling was performed on a single GPU. Memory and time may vary based on:\n")
        f.write("- GPU model and memory capacity\n")
        f.write("- Input point cloud size (varies per sample)\n")
        f.write("- Batch size configuration\n")

    # Save JSON results
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to:")
    print(f"   Markdown: {report_path}")
    print(f"   JSON: {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTv3 experiment configs for SemanticKITTI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Method Properties:
  Deterministic (no seed):     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel, DEPOCO
  Non-deterministic (seed):    RS, FPS, Poisson

Examples:
  # Generate default configs (seed 1 for non-deterministic)
  python generate_configs.py

  # Generate with multiple seeds (for non-deterministic methods)
  python generate_configs.py --seeds 1 2 3

  # Generate for specific GPU type only
  python generate_configs.py --gpu 140gb

  # Generate tier 1 only (RS, IDIS) with multiple seeds
  python generate_configs.py --tier tier1 --seeds 1 2 3

  # Generate configs and profile them for GFLOPs/memory
  python generate_configs.py --profile --gpu 140gb

  # Profile a specific config file
  python generate_configs.py --profile-only --config path/to/config.py

Config naming:
  Deterministic:     ptv3_semantickitti_{METHOD}_loss{LOSS}_{GPU}.py
  Non-deterministic: ptv3_semantickitti_{METHOD}_loss{LOSS}_seed{SEED}_{GPU}.py
        """
    )
    parser.add_argument(
        "--tier",
        choices=["priority", "tier1", "tier2", "all"],
        default="priority",
        help="Config tier: priority/all (all methods), tier1 (RS,IDIS), tier2 (baselines)"
    )
    parser.add_argument(
        "--gpu",
        choices=["40gb", "140gb", "both"],
        default="both",
        help="GPU type: 40gb, 140gb, or both (default: both)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Seeds for non-deterministic methods (default: 1). Use '--seeds 1 2 3' for multiple."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile generated configs for GFLOPs and memory consumption"
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Only run profiling (skip config generation). Use with --config"
    )
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="Profile all configs (not just one per method). Slower but comprehensive."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config file to profile (use with --profile-only)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for profiling (default: 0)"
    )

    args = parser.parse_args()

    # Determine base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # PTv3/SemanticKITTI/
    output_dir = base_dir / "configs/semantickitti/generated"

    # Handle profile-only mode
    if args.profile_only:
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Config file not found: {config_path}")
                sys.exit(1)
            print(f"\n{'='*60}")
            print("PROFILE SINGLE CONFIG")
            print(f"{'='*60}")
            metrics = profile_single_config(config_path, gpu_id=args.gpu_id)
            if metrics:
                print(f"\n✅ Profiling complete!")
                print(f"   GFLOPs: {metrics.get('gflops', 'N/A')}")
                print(f"   Peak Memory: {metrics.get('peak_memory', 'N/A')}")
                print(f"   Parameters: {metrics.get('parameters', 'N/A')}")
        else:
            # Profile all existing configs in generated directory
            if not output_dir.exists():
                print(f"Error: No generated configs found at {output_dir}")
                sys.exit(1)
            config_paths = list(output_dir.glob("*.py"))
            if not config_paths:
                print(f"Error: No config files found in {output_dir}")
                sys.exit(1)
            profile_configs(
                config_paths,
                output_dir,
                gpu_id=args.gpu_id,
                sample_only=not args.profile_all
            )
        return

    # Determine GPU types
    if args.gpu == "both":
        gpu_types = ["40gb", "140gb"]
    else:
        gpu_types = [args.gpu]

    # Generate configs
    config_paths = generate_all_configs(
        base_dir=base_dir,
        tier=args.tier,
        gpu_types=gpu_types,
        seeds=args.seeds
    )

    # Profile if requested
    if args.profile:
        if not config_paths:
            print("\n⚠ No configs generated, skipping profiling")
            return

        profile_configs(
            config_paths,
            output_dir,
            gpu_id=args.gpu_id,
            sample_only=not args.profile_all
        )


if __name__ == "__main__":
    main()
