#!/usr/bin/env python3
"""
Generate Subsampled SemanticKITTI Dataset (V2 - All Sequences)

This script generates subsampled versions of SemanticKITTI using all sequences
required for training (00-10 except 08) and validation (08).

Output Structure (PTv3-compatible):
    data/SemanticKITTI/subsampled/
    ├── RS_loss10_seed1/          # Non-deterministic - seed in path
    │   └── sequences/
    │       ├── 00/
    │       │   ├── velodyne/
    │       │   │   ├── 000000.bin
    │       │   │   └── ...
    │       │   └── labels/
    │       │       ├── 000000.label
    │       │       └── ...
    │       ├── 01/
    │       ├── ...
    │       └── 10/
    ├── DBSCAN_loss50/            # Deterministic - no seed in path
    ├── Voxel_loss50/             # Deterministic - no seed in path
    └── Poisson_loss50_seed1/     # Non-deterministic - seed in path

Usage:
    python generate_subsampled_semantickitti_v2.py --methods RS --workers 48
    python generate_subsampled_semantickitti_v2.py --methods RS IDIS --workers 4
    python generate_subsampled_semantickitti_v2.py --sequences 00 08 --methods RS --workers 48
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from subsampling import (
    get_sampler,
    AVAILABLE_METHODS,
    list_available_methods
)


# ============================================================================
# Configuration
# ============================================================================

# SemanticKITTI sequences
TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQUENCES = ['08']
ALL_SEQUENCES = TRAIN_SEQUENCES + VAL_SEQUENCES

DEFAULT_DATA_ROOT = "data/SemanticKITTI/original"
DEFAULT_OUTPUT_DIR = "data/SemanticKITTI/subsampled"

METHODS = ['RS', 'IDIS', 'FPS', 'DBSCAN', 'Voxel', 'Poisson']
LOSS_LEVELS = [10, 30, 50, 70, 90]
SEEDS = [1]  # Default: seed 1 only (use --seeds to specify more)

# Deterministic methods don't need seed in output path
# Voxel: Grid-based centroid computation (deterministic)
# DBSCAN: Cluster centroid selection (deterministic)
DETERMINISTIC_METHODS = {'Voxel', 'DBSCAN'}

# Method-specific MAX workers (memory optimization)
METHOD_MAX_WORKERS = {
    'RS': 64,
    'IDIS': 4,
    'FPS': 4,
    'DBSCAN': 32,
    'Voxel': 48,
    'Poisson': 32,
}


def get_max_workers_for_method(method, requested_workers):
    max_workers = METHOD_MAX_WORKERS.get(method, requested_workers)
    return min(requested_workers, max_workers)


def get_seeds_for_method(method, user_seeds):
    """
    Get seeds to use for a method.

    Args:
        method: Method name
        user_seeds: List of seeds provided by user

    Returns:
        List of seeds to use (deterministic methods use only first seed)
    """
    if method in DETERMINISTIC_METHODS:
        # Deterministic methods only need one seed
        return [user_seeds[0]]
    return user_seeds


# ============================================================================
# SemanticKITTI Data Loading/Saving
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
    labels = np.fromfile(label_file, dtype=np.uint32)
    return labels  # Keep full 32-bit for instance info


def save_semantickitti_labels(labels, output_file):
    """Save SemanticKITTI labels."""
    labels.astype(np.uint32).tofile(output_file)


# ============================================================================
# Data Generation
# ============================================================================

def process_single_scan(args):
    """Process a single scan with given method, loss level, and seed."""
    (scan_file, output_file, label_input, label_output,
     method, loss_level, seed, verbose) = args

    try:
        # Check if output already exists (resume capability)
        if os.path.exists(output_file):
            return {
                'status': 'skipped',
                'scan': scan_file,
                'method': method,
                'loss': loss_level
            }

        # Load scan
        points, intensity = load_semantickitti_scan(scan_file)
        n_original = len(points)

        # Load labels if available
        labels = None
        if label_input and os.path.exists(label_input):
            labels = load_semantickitti_labels(label_input)

        # Get sampler
        sampler = get_sampler(
            method,
            loss_percentage=loss_level,
            seed=seed,
            dataset='semantickitti'
        )

        # Apply subsampling
        if labels is not None:
            result = sampler(
                points,
                features=intensity[:, np.newaxis],
                labels=labels,
                return_indices=False,
                verbose=False
            )
            sampled_points, sampled_intensity, sampled_labels = result
            sampled_intensity = sampled_intensity.flatten()
        else:
            result = sampler(
                points,
                features=intensity[:, np.newaxis],
                return_indices=False,
                verbose=False
            )
            sampled_points, sampled_intensity = result
            sampled_intensity = sampled_intensity.flatten()
            sampled_labels = None

        n_sampled = len(sampled_points)

        # Create output directory
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
            'method': method,
            'loss': loss_level,
            'seed': seed,
            'n_original': n_original,
            'n_sampled': n_sampled,
            'actual_loss': (1 - n_sampled / n_original) * 100
        }

    except Exception as e:
        return {
            'status': 'error',
            'scan': scan_file,
            'method': method,
            'loss': loss_level,
            'seed': seed,
            'error': str(e)
        }


def generate_subsampled_dataset(
    data_root,
    output_dir,
    sequences,
    methods,
    loss_levels,
    seeds,
    workers=1,
    verbose=True
):
    """
    Generate subsampled SemanticKITTI datasets for all sequences.

    Output structure: output_dir/METHOD_lossXX_seedY/dataset/sequences/XX/velodyne/
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"SemanticKITTI Subsampled Dataset Generation (V2 - All Sequences)")
        print(f"{'='*80}")
        print(f"Data root: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Sequences: {', '.join(sequences)}")
        print(f"Methods: {', '.join(methods)}")
        print(f"Loss levels: {', '.join(map(str, loss_levels))}%")
        print(f"Workers: {workers}")
        print(f"{'='*80}\n")

    # Build task list for all sequences
    all_tasks = []

    for method in methods:
        method_seeds = get_seeds_for_method(method, seeds)
        is_deterministic = method in DETERMINISTIC_METHODS

        for loss in loss_levels:
            for seed in method_seeds:
                # Deterministic methods: no seed in path
                # Non-deterministic methods: seed in path for reproducibility
                if is_deterministic:
                    output_subdir = f"{method}_loss{loss}"
                else:
                    output_subdir = f"{method}_loss{loss}_seed{seed}"

                for seq in sequences:
                    # Input paths
                    seq_velodyne = Path(data_root) / "sequences" / seq / "velodyne"
                    seq_labels = Path(data_root) / "sequences" / seq / "labels"

                    if not seq_velodyne.exists():
                        if verbose:
                            print(f"WARNING: Sequence {seq} not found at {seq_velodyne}")
                        continue

                    # Output paths (PTv3-compatible structure)
                    out_velodyne = Path(output_dir) / output_subdir / "sequences" / seq / "velodyne"
                    out_labels = Path(output_dir) / output_subdir / "sequences" / seq / "labels"

                    # Get all scan files
                    scan_files = sorted(seq_velodyne.glob("*.bin"))

                    for scan_file in scan_files:
                        scan_name = scan_file.name
                        output_file = out_velodyne / scan_name

                        label_input = seq_labels / scan_name.replace('.bin', '.label')
                        label_output = out_labels / scan_name.replace('.bin', '.label')

                        all_tasks.append((
                            str(scan_file),
                            str(output_file),
                            str(label_input) if label_input.exists() else None,
                            str(label_output),
                            method,
                            loss,
                            seed,
                            False  # verbose
                        ))

    if verbose:
        print(f"Total tasks: {len(all_tasks)}")

    # Process by method to manage memory
    start_time = time.time()
    all_results = []

    for method in methods:
        method_tasks = [t for t in all_tasks if t[4] == method]
        if not method_tasks:
            continue

        method_workers = get_max_workers_for_method(method, workers)

        if verbose:
            print(f"\nProcessing {method}: {len(method_tasks)} tasks with {method_workers} workers")

        if method_workers > 1:
            with Pool(method_workers, maxtasksperchild=50) as pool:
                results = list(tqdm(
                    pool.imap(process_single_scan, method_tasks),
                    total=len(method_tasks),
                    desc=f"  {method}",
                    unit="scan"
                ))
        else:
            results = []
            for task in tqdm(method_tasks, desc=f"  {method}", unit="scan"):
                results.append(process_single_scan(task))

        all_results.extend(results)

        # Memory cleanup
        gc.collect()

        # Print stats
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        errors = sum(1 for r in results if r['status'] == 'error')
        if verbose:
            print(f"  ✓ {method}: success={success}, skipped={skipped}, errors={errors}")

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*80}")
        print(f"Complete! Total time: {elapsed/60:.1f} minutes")
        total_success = sum(1 for r in all_results if r['status'] == 'success')
        total_skipped = sum(1 for r in all_results if r['status'] == 'skipped')
        total_errors = sum(1 for r in all_results if r['status'] == 'error')
        print(f"Success: {total_success}, Skipped: {total_skipped}, Errors: {total_errors}")
        print(f"{'='*80}\n")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate subsampled SemanticKITTI dataset (all sequences)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help=f'Data root directory (default: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--sequences', type=str, nargs='+', default=ALL_SEQUENCES,
                        help=f'Sequences to process (default: {ALL_SEQUENCES})')
    parser.add_argument('--methods', type=str, nargs='+', default=METHODS,
                        choices=METHODS, help='Subsampling methods')
    parser.add_argument('--loss-levels', type=int, nargs='+', default=LOSS_LEVELS,
                        help=f'Loss percentages (default: {LOSS_LEVELS})')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help=f'Random seeds (default: {SEEDS})')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available methods and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if args.list_methods:
        list_available_methods()
        return 0

    if args.workers == -1:
        args.workers = cpu_count()

    # Validate data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root not found: {data_root}")
        return 1

    try:
        generate_subsampled_dataset(
            data_root=str(data_root),
            output_dir=args.output_dir,
            sequences=args.sequences,
            methods=args.methods,
            loss_levels=args.loss_levels,
            seeds=args.seeds,
            workers=args.workers,
            verbose=not args.quiet
        )
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
