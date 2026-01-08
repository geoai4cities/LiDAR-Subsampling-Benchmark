#!/usr/bin/env python3
"""
GPU-Accelerated Subsampling for SemanticKITTI (IDIS and FPS)

Uses pointops CUDA kernels for ~40-200x speedup over CPU multiprocessing.
Processes scans sequentially on GPU (faster than CPU multiprocessing).

Supported methods:
    - IDIS (GPU): ~0.3s per scan (vs ~30-60s CPU)
    - FPS (GPU): ~1.6s per scan (vs ~60-120s CPU)

Output Structure (PTv3-compatible):
    data/SemanticKITTI/subsampled/
    ├── IDIS_loss50/              # Deterministic - no seed in path
    │   └── sequences/
    │       ├── 00/velodyne/*.bin
    │       └── 00/labels/*.label
    ├── IDIS_R5_loss50/           # IDIS with R=5m ablation
    └── FPS_loss50_seed1/         # Non-deterministic - seed in path
        └── ...

Usage:
    # IDIS with default R=10m
    python generate_subsampled_gpu.py --method IDIS --loss-levels 10 30 50 70 90

    # FPS
    python generate_subsampled_gpu.py --method FPS --loss-levels 10 30 50 70 90

    # IDIS with custom radius (for ablation)
    python generate_subsampled_gpu.py --method IDIS --radius 5 --loss-levels 50
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Check GPU availability
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Use generate_subsampled_semantickitti_v2.py for CPU processing.")
    sys.exit(1)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import GPU functions
try:
    from subsampling.idis_gpu import idis_subsample_with_loss_gpu
    from subsampling.fps_gpu import fps_subsample_with_loss_gpu
    print("GPU subsampling functions loaded successfully")
except ImportError as e:
    print(f"ERROR: Failed to import GPU functions: {e}")
    print("Make sure pointops is installed (from PTv3/Pointcept)")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQUENCES = ['08']
ALL_SEQUENCES = TRAIN_SEQUENCES + VAL_SEQUENCES

DEFAULT_DATA_ROOT = "data/SemanticKITTI/original"
DEFAULT_OUTPUT_DIR = "data/SemanticKITTI/subsampled"

# Deterministic methods don't need seed in output path
# IDIS: Same input always produces same output (deterministic importance sampling)
# FPS: Starting point affects results (non-deterministic)
DETERMINISTIC_METHODS = {'IDIS'}


# ============================================================================
# Data Loading/Saving
# ============================================================================

def load_semantickitti_scan(bin_file):
    """Load SemanticKITTI binary scan file."""
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    intensity = scan[:, 3]
    return points, intensity


def save_semantickitti_scan(points, intensity, output_file):
    """Save SemanticKITTI scan in binary format."""
    scan = np.column_stack([points, intensity]).astype(np.float32)
    scan.tofile(output_file)


def load_semantickitti_labels(label_file):
    """Load SemanticKITTI labels."""
    if not os.path.exists(label_file):
        return None
    return np.fromfile(label_file, dtype=np.uint32)


def save_semantickitti_labels(labels, output_file):
    """Save SemanticKITTI labels."""
    labels.astype(np.uint32).tofile(output_file)


# ============================================================================
# GPU Processing
# ============================================================================

def process_scan_gpu(
    scan_file: str,
    output_file: str,
    label_input: str,
    label_output: str,
    method: str,
    loss_level: float,
    seed: int,
    radius: float = 10.0
):
    """Process a single scan using GPU acceleration."""

    # Check if output already exists
    if os.path.exists(output_file):
        return {'status': 'skipped', 'scan': scan_file}

    # Load scan
    points, intensity = load_semantickitti_scan(scan_file)
    n_original = len(points)

    # Load labels if available
    labels = None
    if label_input and os.path.exists(label_input):
        labels = load_semantickitti_labels(label_input)

    # Apply GPU subsampling
    if method == 'IDIS':
        if labels is not None:
            sampled_points, sampled_intensity, sampled_labels = idis_subsample_with_loss_gpu(
                points,
                loss_percentage=loss_level,
                features=intensity[:, np.newaxis],
                labels=labels,
                radius=radius,
                seed=seed,
                verbose=False
            )
        else:
            sampled_points, sampled_intensity = idis_subsample_with_loss_gpu(
                points,
                loss_percentage=loss_level,
                features=intensity[:, np.newaxis],
                radius=radius,
                seed=seed,
                verbose=False
            )
            sampled_labels = None

    elif method == 'FPS':
        if labels is not None:
            sampled_points, sampled_intensity, sampled_labels = fps_subsample_with_loss_gpu(
                points,
                loss_percentage=loss_level,
                features=intensity[:, np.newaxis],
                labels=labels,
                seed=seed,
                verbose=False
            )
        else:
            sampled_points, sampled_intensity = fps_subsample_with_loss_gpu(
                points,
                loss_percentage=loss_level,
                features=intensity[:, np.newaxis],
                seed=seed,
                verbose=False
            )
            sampled_labels = None
    else:
        raise ValueError(f"Unknown method: {method}. Use 'IDIS' or 'FPS'")

    sampled_intensity = sampled_intensity.flatten()
    n_sampled = len(sampled_points)

    # Create output directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save subsampled scan
    save_semantickitti_scan(sampled_points, sampled_intensity, output_file)

    # Save labels if available
    if sampled_labels is not None and label_output:
        os.makedirs(os.path.dirname(label_output), exist_ok=True)
        save_semantickitti_labels(sampled_labels, label_output)

    return {
        'status': 'success',
        'scan': scan_file,
        'n_original': n_original,
        'n_sampled': n_sampled
    }


def generate_subsampled_gpu(
    data_root: str,
    output_dir: str,
    sequences: list,
    method: str,
    loss_levels: list,
    seed: int = 1,
    radius: float = 10.0,
    radius_suffix: str = None
):
    """Generate subsampled dataset using GPU acceleration."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)

    print(f"\n{'='*70}")
    print(f"GPU-Accelerated {method} Subsampling")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Sequences: {', '.join(sequences)}")
    print(f"Loss levels: {', '.join(map(str, loss_levels))}%")
    print(f"Seed: {seed}")
    if method == 'IDIS':
        print(f"Radius: {radius}m")
    print()

    total_start = time.time()

    for loss_level in loss_levels:
        print(f"\n{'─'*70}")
        print(f"Processing {method} loss={loss_level}%")
        print(f"{'─'*70}")

        # Create output directory name
        # Deterministic methods (IDIS): no seed in path
        # Non-deterministic methods (FPS): seed in path for reproducibility
        is_deterministic = method in DETERMINISTIC_METHODS

        if radius_suffix:
            if is_deterministic:
                output_subdir = f"{method}_{radius_suffix}_loss{loss_level}"
            else:
                output_subdir = f"{method}_{radius_suffix}_loss{loss_level}_seed{seed}"
        else:
            if is_deterministic:
                output_subdir = f"{method}_loss{loss_level}"
            else:
                output_subdir = f"{method}_loss{loss_level}_seed{seed}"

        success_count = 0
        skip_count = 0
        error_count = 0

        loss_start = time.time()

        for seq in sequences:
            seq_velodyne = data_root / "sequences" / seq / "velodyne"
            seq_labels = data_root / "sequences" / seq / "labels"

            if not seq_velodyne.exists():
                print(f"  Warning: Sequence {seq} not found, skipping")
                continue

            out_velodyne = output_dir / output_subdir / "sequences" / seq / "velodyne"
            out_labels = output_dir / output_subdir / "sequences" / seq / "labels"

            scan_files = sorted(seq_velodyne.glob("*.bin"))

            # Process scans with progress bar
            desc = f"Seq {seq}"
            for scan_file in tqdm(scan_files, desc=desc, leave=False):
                scan_name = scan_file.name
                label_input = seq_labels / scan_name.replace('.bin', '.label')

                output_file = out_velodyne / scan_name
                label_output = out_labels / scan_name.replace('.bin', '.label')

                result = process_scan_gpu(
                    str(scan_file),
                    str(output_file),
                    str(label_input) if label_input.exists() else None,
                    str(label_output),
                    method=method,
                    loss_level=loss_level,
                    seed=seed,
                    radius=radius
                )

                if result['status'] == 'success':
                    success_count += 1
                elif result['status'] == 'skipped':
                    skip_count += 1
                else:
                    error_count += 1

        loss_time = time.time() - loss_start
        print(f"  loss={loss_level}%: success={success_count}, skipped={skip_count}, "
              f"errors={error_count}, time={loss_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Completed in {total_time/60:.1f} minutes")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated subsampling for SemanticKITTI (IDIS/FPS)'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['IDIS', 'FPS'],
        help='Subsampling method'
    )
    parser.add_argument(
        '--loss-levels',
        type=int,
        nargs='+',
        default=[10, 30, 50, 70, 90],
        help='Loss percentages (default: 10 30 50 70 90)'
    )
    parser.add_argument(
        '--sequences',
        type=str,
        nargs='+',
        default=ALL_SEQUENCES,
        help='Sequences to process (default: all 00-10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=10.0,
        help='IDIS radius in meters (default: 10.0)'
    )
    parser.add_argument(
        '--radius-suffix',
        type=str,
        default=None,
        help='Custom suffix for radius (e.g., "R5" for IDIS_R5_loss50_seed1)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f'Data root directory (default: {DEFAULT_DATA_ROOT})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    args = parser.parse_args()

    # Validate sequences
    for seq in args.sequences:
        if seq not in ALL_SEQUENCES:
            print(f"Warning: Sequence {seq} is not in standard sequences")

    generate_subsampled_gpu(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sequences=args.sequences,
        method=args.method,
        loss_levels=args.loss_levels,
        seed=args.seed,
        radius=args.radius,
        radius_suffix=args.radius_suffix
    )


if __name__ == '__main__':
    main()
