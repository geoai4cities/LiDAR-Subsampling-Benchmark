"""
Generate Subsampled DALES Dataset

This script generates subsampled versions of the DALES dataset (40 tiles) using
all 6 subsampling methods (RS, IDIS, FPS, DBSCAN, Voxel, Poisson) at multiple
loss levels and with different random seeds.

Output Structure:
    data/DALES/subsampled/
    ├── RS_loss10_seed1/
    │   ├── 5085_54320.txt
    │   ├── 5095_54440.txt
    │   └── ...
    ├── IDIS_loss50_seed2/
    │   └── ...
    └── ...

    Note: 0% loss (baseline) uses original data directly from:
          data/DALES/original/

Usage:
    # Generate all subsampled datasets
    python scripts/generate_subsampled_dales.py

    # Test with 3 tiles
    python scripts/generate_subsampled_dales.py --test --n-tiles 3

    # Generate specific methods only
    python scripts/generate_subsampled_dales.py --methods RS IDIS

    # Generate specific loss levels
    python scripts/generate_subsampled_dales.py --loss-levels 0 50 90

    # Use parallel processing (4 workers)
    python scripts/generate_subsampled_dales.py --workers 4
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

DEFAULT_INPUT_DIR = "data/DALES/original"
DEFAULT_OUTPUT_DIR = "data/DALES/subsampled"

METHODS = ['RS', 'IDIS', 'FPS', 'DBSCAN', 'Voxel', 'Poisson']
LOSS_LEVELS = [10, 30, 50, 70, 90]  # 0% excluded - use original data for baseline
SEEDS = [1]  # Default: seed 1 only (use --seeds to specify more)

# Deterministic methods don't need seed in output path
# Voxel: Grid-based centroid computation (deterministic)
# DBSCAN: Cluster centroid selection (deterministic)
DETERMINISTIC_METHODS = {'Voxel', 'DBSCAN'}

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

# DALES tile structure (from dales.py dataset loader)
TRAIN_TILES = [
    "5085_54320", "5095_54440", "5095_54455", "5105_54405", "5105_54460",
    "5110_54320", "5110_54460", "5110_54475", "5110_54495", "5115_54480",
    "5135_54495", "5140_54445", "5145_54340", "5145_54405", "5145_54460",
    "5145_54480", "5150_54340", "5160_54330", "5165_54390", "5180_54435",
    "5180_54485", "5185_54390", "5185_54485"
]

VAL_TILES = [
    "5080_54435", "5100_54495", "5130_54355", "5145_54470", "5165_54395", "5190_54400"
]

TEST_TILES = [
    "5080_54400", "5080_54470", "5100_54440", "5100_54490", "5120_54445",
    "5135_54430", "5135_54435", "5140_54390", "5150_54325", "5155_54335",
    "5175_54395"
]

ALL_TILES = TRAIN_TILES + VAL_TILES + TEST_TILES  # 40 tiles total


# ============================================================================
# DALES Data Loading
# ============================================================================

def load_dales_tile(txt_file):
    """
    Load DALES TXT tile file.

    Format: x, y, z, intensity, return_num, num_returns, class

    Args:
        txt_file: Path to .txt file

    Returns:
        points: (N, 3) array of xyz coordinates
        intensity: (N,) array of intensity values
        labels: (N,) array of class labels
    """
    # Load TXT file
    # Format: x, y, z, intensity, return_num, num_returns, class (7 columns)
    data = np.loadtxt(txt_file)

    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Invalid DALES file format: expected (N, 7), got {data.shape}")

    points = data[:, :3].astype(np.float32)  # x, y, z
    intensity = data[:, 3].astype(np.float32)  # intensity
    labels = data[:, 6].astype(np.int32)  # class (column 6, 0-indexed)

    return points, intensity, labels


def save_dales_tile(points, intensity, labels, output_file):
    """
    Save DALES tile in TXT format.

    Format: x, y, z, intensity, return_num, num_returns, class

    Args:
        points: (N, 3) array of xyz coordinates
        intensity: (N,) array of intensity values
        labels: (N,) array of class labels
        output_file: Path to output .txt file
    """
    # For simplified output, we'll set return_num=1 and num_returns=1
    # since these are not critical for semantic segmentation
    return_num = np.ones(len(points), dtype=np.int32)
    num_returns = np.ones(len(points), dtype=np.int32)

    # Stack all columns
    # Format: x, y, z, intensity, return_num, num_returns, class
    output = np.column_stack([
        points,  # x, y, z
        intensity,  # intensity
        return_num,  # return_num
        num_returns,  # num_returns
        labels  # class
    ])

    # Save with appropriate format
    # Float for x, y, z, intensity; int for return_num, num_returns, class
    np.savetxt(
        output_file,
        output,
        fmt='%.6f %.6f %.6f %.3f %d %d %d',
        delimiter=' '
    )


# ============================================================================
# Data Generation
# ============================================================================

def process_single_tile(args):
    """
    Process a single tile with given method, loss level, and seed.

    Args:
        args: Tuple of (tile_file, output_file, method, loss_level, seed, verbose)

    Returns:
        dict with processing statistics
    """
    tile_file, output_file, method, loss_level, seed, verbose = args

    try:
        # Check if output already exists (resume capability)
        if os.path.exists(output_file):
            if verbose:
                print(f"  Skipping {os.path.basename(tile_file)} (already exists)")
            return {
                'status': 'skipped',
                'tile': os.path.basename(tile_file),
                'method': method,
                'loss': loss_level
            }

        # Load tile
        points, intensity, labels = load_dales_tile(tile_file)
        n_original = len(points)

        # Get sampler
        sampler = get_sampler(
            method,
            loss_percentage=loss_level,
            seed=seed,
            dataset='dales'  # For Voxel and Poisson
        )

        # Apply subsampling
        result = sampler(
            points,
            features=intensity[:, np.newaxis],  # (N, 1) for consistency
            labels=labels,
            return_indices=False,
            verbose=False
        )

        sampled_points, sampled_intensity, sampled_labels = result
        sampled_intensity = sampled_intensity.flatten()  # Back to (N,)

        n_sampled = len(sampled_points)

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save subsampled tile
        save_dales_tile(sampled_points, sampled_intensity, sampled_labels, output_file)

        return {
            'status': 'success',
            'tile': os.path.basename(tile_file),
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
            'tile': os.path.basename(tile_file),
            'method': method,
            'loss': loss_level,
            'seed': seed,
            'error': str(e)
        }


def find_tile_files(input_dir, tiles=None):
    """
    Find DALES tile files.

    Args:
        input_dir: Root input directory
        tiles: List of tile names (e.g., ['5085_54320', ...]) or None for all

    Returns:
        List of tile file paths
    """
    # Look in train and test subdirectories
    tile_files = []

    for subdir in ['train', 'test']:
        subdir_path = Path(input_dir) / subdir
        if not subdir_path.exists():
            continue

        for txt_file in subdir_path.glob('*.txt'):
            tile_name = txt_file.stem  # filename without extension
            if tiles is None or tile_name in tiles:
                tile_files.append(txt_file)

    return sorted(tile_files)


def generate_subsampled_dataset(
    input_dir,
    output_dir,
    tiles=None,
    methods=METHODS,
    loss_levels=LOSS_LEVELS,
    seeds=SEEDS,
    n_tiles=None,
    workers=1,
    verbose=True
):
    """
    Generate subsampled DALES datasets.

    Methods are processed SEQUENTIALLY (one at a time) to prevent memory issues.
    Within each method, tiles are processed in parallel using workers.

    Args:
        input_dir: Input directory containing train/ and test/ subdirs with .txt files
        output_dir: Output directory for subsampled data
        tiles: List of tile names to process (None = all)
        methods: List of subsampling methods to use
        loss_levels: List of loss percentages
        seeds: List of random seeds
        n_tiles: Number of tiles to process (None = all)
        workers: Number of parallel workers
        verbose: Print progress information
    """
    import gc

    # Get list of tile files
    if tiles is None:
        tiles = ALL_TILES

    tile_files = find_tile_files(input_dir, tiles)

    if len(tile_files) == 0:
        raise ValueError(f"No .txt tile files found in {input_dir}")

    if n_tiles is not None:
        tile_files = tile_files[:n_tiles]

    if verbose:
        print(f"\n{'='*80}")
        print(f"DALES Subsampled Dataset Generation")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Number of tiles: {len(tile_files)}")
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
        total_tasks += len(loss_levels) * len(method_seeds) * len(tile_files)

    if verbose:
        print(f"Total tasks: {total_tasks}")
        print(f"  Methods: {len(methods)}")
        print(f"  Loss levels: {len(loss_levels)}")
        print(f"  Seeds: {len(seeds)}")
        print(f"  Tiles: {len(tile_files)}")
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
        is_deterministic = method in DETERMINISTIC_METHODS

        for loss in loss_levels:
            for seed in method_seeds:
                # Deterministic methods: no seed in path
                # Non-deterministic methods: seed in path for reproducibility
                if is_deterministic:
                    output_subdir = f"{method}_loss{loss}"
                else:
                    output_subdir = f"{method}_loss{loss}_seed{seed}"
                output_path = Path(output_dir) / output_subdir

                for tile_file in tile_files:
                    tile_name = tile_file.stem + '.txt'
                    output_file = output_path / tile_name

                    method_tasks.append((
                        str(tile_file),
                        str(output_file),
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
                        pool.imap(process_single_tile, method_tasks),
                        total=len(method_tasks),
                        desc=f"  {method}",
                        unit="tile"
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
            for task in tqdm(method_tasks, desc=f"  {method}", unit="tile"):
                result = process_single_tile(task)
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
        print(f"Average time per tile: {elapsed_time/len(tile_files):.2f} seconds")
        print(f"{'='*80}\n")

        # Show loss statistics for successful tiles
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
                    print(f"  {r['tile']}: {r['error']}")
            print("-" * 80)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate subsampled DALES dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all subsampled datasets
  python scripts/generate_subsampled_dales.py

  # Test with first 3 tiles
  python scripts/generate_subsampled_dales.py --test --n-tiles 3

  # Generate specific methods
  python scripts/generate_subsampled_dales.py --methods RS IDIS FPS

  # Use 4 parallel workers
  python scripts/generate_subsampled_dales.py --workers 4

  # List available methods
  python scripts/generate_subsampled_dales.py --list-methods
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory (should contain train/ and test/ subdirs) (default: {DEFAULT_INPUT_DIR})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
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
        '--tiles',
        type=str,
        nargs='+',
        default=None,
        help='Specific tiles to process (default: all 40 tiles)'
    )

    parser.add_argument(
        '--n-tiles',
        type=int,
        default=None,
        help='Number of tiles to process (default: all)'
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
        help='Test mode: process first 3 tiles only'
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
        args.n_tiles = 3
        print("TEST MODE: Processing first 3 tiles")

    # Workers
    if args.workers == -1:
        args.workers = cpu_count()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return 1

    # Check for train/test subdirectories
    train_dir = Path(args.input_dir) / 'train'
    test_dir = Path(args.input_dir) / 'test'

    if not train_dir.exists() and not test_dir.exists():
        print(f"ERROR: Neither train/ nor test/ subdirectories found in {args.input_dir}")
        print("Expected structure: {input_dir}/train/*.txt and {input_dir}/test/*.txt")
        return 1

    # Generate datasets
    try:
        generate_subsampled_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tiles=args.tiles,
            methods=args.methods,
            loss_levels=args.loss_levels,
            seeds=args.seeds,
            n_tiles=args.n_tiles,
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
