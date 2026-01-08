"""
Generate Subsampled SemanticKITTI Dataset

This script generates subsampled versions of SemanticKITTI sequence 00 using
all 6 subsampling methods (RS, IDIS, FPS, DBSCAN, Voxel, Poisson) at multiple
loss levels and with different random seeds.

Output Structure:
    data/SemanticKITTI/subsampled/
    ├── RS_loss10_seed1/
    │   ├── 000000.bin
    │   ├── 000001.bin
    │   └── ...
    ├── IDIS_loss50_seed2/
    │   └── ...
    └── ...

    Note: 0% loss (baseline) uses original data directly from:
          data/SemanticKITTI/original/sequences/00/velodyne/

Usage:
    # Generate all subsampled datasets
    python scripts/generate_subsampled_semantickitti.py

    # Test with first 10 scans
    python scripts/generate_subsampled_semantickitti.py --test --n-scans 10

    # Generate specific methods only
    python scripts/generate_subsampled_semantickitti.py --methods RS IDIS

    # Generate specific loss levels
    python scripts/generate_subsampled_semantickitti.py --loss-levels 0 50 90

    # Use parallel processing (4 workers)
    python scripts/generate_subsampled_semantickitti.py --workers 4
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# Add src to path (scripts/preprocessing/ -> scripts/ -> project_root/ -> src/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from subsampling import (
    get_sampler,
    AVAILABLE_METHODS,
    list_available_methods
)


# ============================================================================
# Configuration
# ============================================================================

# SemanticKITTI sequences for train (0-10 except 8) and val (8)
# Test sequences (11-21) are not used for training
TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQUENCES = ['08']
ALL_SEQUENCES = TRAIN_SEQUENCES + VAL_SEQUENCES

DEFAULT_DATA_ROOT = "data/SemanticKITTI/original"
DEFAULT_OUTPUT_DIR = "data/SemanticKITTI/subsampled"

METHODS = ['RS', 'IDIS', 'FPS', 'DBSCAN', 'Voxel', 'Poisson']
LOSS_LEVELS = [10, 30, 50, 70, 90]  # 0% excluded - use original data for baseline
SEEDS = [1, 2, 3]

# Method-specific seed configuration
# TODO: Add seeds [1, 2, 3] later for stochastic methods (RS, IDIS, FPS, DBSCAN, Poisson)
# For now using single seed to speed up initial processing
METHOD_SEEDS = {
    'RS': [1],            # TODO: [1, 2, 3] later
    'IDIS': [1],          # TODO: [1, 2, 3] later
    'FPS': [1],           # TODO: [1, 2, 3] later
    'DBSCAN': [1],        # TODO: [1, 2, 3] later
    'Voxel': [1],         # Deterministic - single seed sufficient
    'Poisson': [1],       # TODO: [1, 2, 3] later
}

# Method-specific MAX workers (memory optimization)
# Based on OBSERVED memory usage per worker (measured on 377GB system):
#   RS: ~1GB, IDIS: ~45GB, FPS: ~45GB, DBSCAN: ~10GB, Voxel: ~2GB, Poisson: ~5GB
# For 377GB system with ~80GB reserved for system/other, ~300GB available for workers
METHOD_MAX_WORKERS = {
    'RS': 64,       # Lightweight ~1GB per worker
    'IDIS': 4,      # ~45GB per worker -> max 4 workers (~180GB) - MEMORY INTENSIVE!
    'FPS': 4,       # ~45GB per worker -> max 4 workers (~180GB) - MEMORY INTENSIVE!
    'DBSCAN': 32,   # ~2GB per worker (measured) -> increase from 8 to 32
    'Voxel': 48,    # ~1GB per worker -> increase from 32 to 48
    'Poisson': 32,  # ~2GB per worker (measured) -> increase from 16 to 32
}

def get_max_workers_for_method(method, requested_workers):
    """
    Get safe number of workers for a method based on memory requirements.

    Args:
        method: Method name
        requested_workers: User-requested number of workers

    Returns:
        Safe number of workers (min of requested and max for method)
    """
    max_workers = METHOD_MAX_WORKERS.get(method, requested_workers)
    return min(requested_workers, max_workers)

def get_seeds_for_method(method, default_seeds=SEEDS):
    """
    Get appropriate seeds for a method based on its determinism.

    Args:
        method: Method name
        default_seeds: Default seeds to use if method not in METHOD_SEEDS

    Returns:
        List of seeds to use for this method
    """
    return METHOD_SEEDS.get(method, default_seeds)


# ============================================================================
# SemanticKITTI Data Loading
# ============================================================================

def load_semantickitti_scan(bin_file):
    """
    Load SemanticKITTI binary scan file.

    Args:
        bin_file: Path to .bin file

    Returns:
        points: (N, 3) array of xyz coordinates
        intensity: (N,) array of intensity values
    """
    # SemanticKITTI format: x, y, z, intensity (4 float32 values per point)
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    points = scan[:, :3]  # xyz
    intensity = scan[:, 3]  # intensity

    return points, intensity


def save_semantickitti_scan(points, intensity, output_file):
    """
    Save SemanticKITTI scan in binary format.

    Args:
        points: (N, 3) array of xyz coordinates
        intensity: (N,) array of intensity values
        output_file: Path to output .bin file
    """
    # Combine xyz + intensity
    scan = np.column_stack([points, intensity]).astype(np.float32)

    # Save as binary
    scan.tofile(output_file)


def load_semantickitti_labels(label_file):
    """
    Load SemanticKITTI labels.

    Args:
        label_file: Path to .label file

    Returns:
        labels: (N,) array of semantic labels (uint32)
    """
    if not os.path.exists(label_file):
        return None

    # SemanticKITTI labels are stored as uint32
    labels = np.fromfile(label_file, dtype=np.uint32)
    labels = labels & 0xFFFF  # Lower 16 bits are semantic label

    return labels


def save_semantickitti_labels(labels, output_file):
    """
    Save SemanticKITTI labels.

    Args:
        labels: (N,) array of semantic labels
        output_file: Path to output .label file
    """
    # Convert to uint32 format (semantic label in lower 16 bits)
    labels_uint32 = labels.astype(np.uint32)
    labels_uint32.tofile(output_file)


# ============================================================================
# Data Generation
# ============================================================================

def process_single_scan(args):
    """
    Process a single scan with given method, loss level, and seed.

    Args:
        args: Tuple of (scan_file, output_file, label_input, label_output,
                        method, loss_level, seed, verbose)

    Returns:
        dict with processing statistics
    """
    (scan_file, output_file, label_input, label_output,
     method, loss_level, seed, verbose) = args

    try:
        # Check if output already exists (resume capability)
        if os.path.exists(output_file):
            if verbose:
                print(f"  Skipping {os.path.basename(scan_file)} (already exists)")
            return {
                'status': 'skipped',
                'scan': os.path.basename(scan_file),
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
            dataset='semantickitti'  # For Voxel and Poisson
        )

        # Apply subsampling
        if labels is not None:
            result = sampler(
                points,
                features=intensity[:, np.newaxis],  # (N, 1) for consistency
                labels=labels,
                return_indices=False,
                verbose=False
            )
            sampled_points, sampled_intensity, sampled_labels = result
            sampled_intensity = sampled_intensity.flatten()  # Back to (N,)
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
            'scan': os.path.basename(scan_file),
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
            'scan': os.path.basename(scan_file),
            'method': method,
            'loss': loss_level,
            'seed': seed,
            'error': str(e)
        }


def generate_subsampled_dataset(
    input_dir,
    output_dir,
    labels_dir=None,
    methods=METHODS,
    loss_levels=LOSS_LEVELS,
    seeds=SEEDS,
    n_scans=None,
    workers=1,
    verbose=True
):
    """
    Generate subsampled SemanticKITTI datasets.

    Methods are processed SEQUENTIALLY (one at a time) to prevent memory issues.
    Within each method, scans are processed in parallel using workers.

    Args:
        input_dir: Input directory containing .bin files
        output_dir: Output directory for subsampled data
        labels_dir: Input directory containing .label files (optional)
        methods: List of subsampling methods to use
        loss_levels: List of loss percentages
        seeds: List of random seeds
        n_scans: Number of scans to process (None = all)
        workers: Number of parallel workers
        verbose: Print progress information
    """
    import gc

    # Get list of scan files
    scan_files = sorted(Path(input_dir).glob('*.bin'))

    if len(scan_files) == 0:
        raise ValueError(f"No .bin files found in {input_dir}")

    if n_scans is not None:
        scan_files = scan_files[:n_scans]

    if verbose:
        print(f"\n{'='*80}")
        print(f"SemanticKITTI Subsampled Dataset Generation")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Labels directory: {labels_dir if labels_dir else 'None'}")
        print(f"Number of scans: {len(scan_files)}")
        print(f"Methods: {', '.join(methods)}")
        print(f"Loss levels: {', '.join(map(str, loss_levels))}%")
        print(f"Seeds (default): {', '.join(map(str, seeds))}")
        print(f"")
        print(f"Method-specific seed configuration (optimization):")
        for method in methods:
            method_seeds = get_seeds_for_method(method, seeds)
            seed_info = ', '.join(map(str, method_seeds))
            if len(method_seeds) == 1:
                print(f"  {method:8s}: [{seed_info}] (deterministic - single seed)")
            else:
                print(f"  {method:8s}: [{seed_info}]")
        print(f"")
        print(f"Workers: {workers}")
        print(f"Processing mode: SEQUENTIAL (one method at a time)")
        print(f"{'='*80}\n")

    # Calculate total tasks for progress info
    total_tasks = 0
    for method in methods:
        method_seeds = get_seeds_for_method(method, seeds)
        total_tasks += len(loss_levels) * len(method_seeds) * len(scan_files)

    if verbose:
        print(f"Total tasks: {total_tasks}")
        print(f"  Methods: {len(methods)}")
        print(f"  Loss levels: {len(loss_levels)}")
        print(f"  Seeds: {len(seeds)}")
        print(f"  Scans: {len(scan_files)}")
        print(f"\nStarting generation...\n")

    # Process tasks - METHODS SEQUENTIALLY to prevent memory issues
    start_time = time.time()
    all_results = []

    for method_idx, method in enumerate(methods):
        method_start_time = time.time()

        if verbose:
            print(f"\n{'─'*80}")
            print(f"[{method_idx+1}/{len(methods)}] Processing method: {method}")
            print(f"{'─'*80}")

        # Build task list for this method only
        method_tasks = []
        method_seeds = get_seeds_for_method(method, seeds)

        for loss in loss_levels:
            for seed in method_seeds:
                output_subdir = f"{method}_loss{loss}_seed{seed}"
                output_path = Path(output_dir) / output_subdir

                for scan_file in scan_files:
                    scan_name = scan_file.name
                    output_file = output_path / scan_name

                    label_input = None
                    label_output = None
                    if labels_dir:
                        label_name = scan_name.replace('.bin', '.label')
                        label_input = Path(labels_dir) / label_name
                        label_output = output_path / label_name

                    method_tasks.append((
                        str(scan_file),
                        str(output_file),
                        str(label_input) if label_input else None,
                        str(label_output) if label_output else None,
                        method,
                        loss,
                        seed,
                        verbose and workers == 1
                    ))

        # Get safe worker count for this method (memory optimization)
        method_workers = get_max_workers_for_method(method, workers)

        if verbose:
            print(f"Tasks for {method}: {len(method_tasks)}")
            if method_workers < workers:
                print(f"  (Using {method_workers} workers instead of {workers} for memory safety)")

        # Process this method's tasks
        if method_workers > 1:
            with Pool(method_workers, maxtasksperchild=50) as pool:
                try:
                    results = list(tqdm(
                        pool.imap(process_single_scan, method_tasks),
                        total=len(method_tasks),
                        desc=f"  {method}",
                        unit="scan"
                    ))
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user - terminating workers...")
                    pool.terminate()
                    pool.join()
                    raise
                except Exception as e:
                    print(f"\n\n✗ Error in parallel processing: {e}")
                    pool.terminate()
                    pool.join()
                    raise
        else:
            # Serial processing (method_workers == 1)
            results = []
            for task in tqdm(method_tasks, desc=f"  {method}", unit="scan"):
                result = process_single_scan(task)
                results.append(result)

        all_results.extend(results)

        method_elapsed = time.time() - method_start_time
        method_success = sum(1 for r in results if r['status'] == 'success')
        method_skipped = sum(1 for r in results if r['status'] == 'skipped')

        if verbose:
            print(f"  ✓ {method} completed in {method_elapsed/60:.1f} min "
                  f"(success: {method_success}, skipped: {method_skipped})")

        # Force garbage collection between methods to free memory
        gc.collect()

    results = all_results
    elapsed_time = time.time() - start_time

    # Summarize results
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    if verbose:
        print(f"\n{'='*80}")
        print(f"Generation Complete!")
        print(f"{'='*80}")
        print(f"Total tasks: {total_tasks}")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Average time per scan: {elapsed_time/total_tasks:.2f} seconds")
        print(f"{'='*80}\n")

        # Show loss statistics for successful scans
        if success_count > 0:
            print("Loss Statistics (Actual vs Target):")
            print("-" * 80)
            for method in methods:
                for loss in loss_levels:
                    method_loss_results = [
                        r for r in results
                        if r['status'] == 'success' and
                           r['method'] == method and
                           r['loss'] == loss
                    ]
                    if method_loss_results:
                        avg_actual_loss = np.mean([r['actual_loss'] for r in method_loss_results])
                        print(f"  {method:8s} Loss={loss:2d}%: Actual={avg_actual_loss:.1f}%")
            print("-" * 80)

        # Show errors if any
        if error_count > 0:
            print("\nErrors:")
            print("-" * 80)
            for r in results:
                if r['status'] == 'error':
                    print(f"  {r['scan']}: {r['error']}")
            print("-" * 80)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate subsampled SemanticKITTI dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all subsampled datasets
  python scripts/generate_subsampled_semantickitti.py

  # Test with first 10 scans
  python scripts/generate_subsampled_semantickitti.py --test --n-scans 10

  # Generate specific methods
  python scripts/generate_subsampled_semantickitti.py --methods RS IDIS FPS

  # Use 4 parallel workers
  python scripts/generate_subsampled_semantickitti.py --workers 4

  # List available methods
  python scripts/generate_subsampled_semantickitti.py --list-methods
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory with .bin files (default: {DEFAULT_INPUT_DIR})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--labels-dir',
        type=str,
        default=DEFAULT_LABELS_DIR,
        help=f'Labels directory with .label files (default: {DEFAULT_LABELS_DIR})'
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=METHODS,
        choices=METHODS,
        help=f'Subsampling methods to use (default: all)'
    )

    parser.add_argument(
        '--loss-levels',
        type=int,
        nargs='+',
        default=LOSS_LEVELS,
        help=f'Loss percentages (default: {LOSS_LEVELS})'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=SEEDS,
        help=f'Random seeds (default: {SEEDS})'
    )

    parser.add_argument(
        '--n-scans',
        type=int,
        default=None,
        help='Number of scans to process (default: all)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, use -1 for cpu_count)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process first 10 scans only'
    )

    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List available subsampling methods and exit'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # List methods and exit
    if args.list_methods:
        list_available_methods()
        return

    # Test mode
    if args.test:
        args.n_scans = 10
        print("TEST MODE: Processing first 10 scans")

    # Workers
    if args.workers == -1:
        args.workers = cpu_count()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return 1

    # Validate labels directory
    labels_dir = args.labels_dir if os.path.exists(args.labels_dir) else None
    if args.labels_dir and not labels_dir:
        print(f"WARNING: Labels directory not found: {args.labels_dir}")
        print("Proceeding without labels...")

    # Generate datasets
    try:
        generate_subsampled_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            labels_dir=labels_dir,
            methods=args.methods,
            loss_levels=args.loss_levels,
            seeds=args.seeds,
            n_scans=args.n_scans,
            workers=args.workers,
            verbose=not args.quiet
        )

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
