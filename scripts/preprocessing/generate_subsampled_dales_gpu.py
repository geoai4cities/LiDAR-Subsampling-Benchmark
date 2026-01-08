#!/usr/bin/env python3
"""
GPU-Accelerated Subsampling for DALES (IDIS and FPS)

Uses pointops CUDA kernels for ~40-200x speedup over CPU multiprocessing.
Processes tiles sequentially on GPU (faster than CPU multiprocessing for these methods).

Supported methods:
    - IDIS (GPU): ~1-3s per tile (vs ~300-600s CPU)
    - FPS (GPU): ~5-15s per tile (vs ~600-1200s CPU)

Output Structure:
    data/DALES/subsampled/
    ├── IDIS_loss50_seed1/
    │   ├── 5085_54320.txt
    │   ├── 5095_54440.txt
    │   └── ...
    └── FPS_loss50_seed1/
        └── ...

Usage:
    # IDIS with default R=10m
    python generate_subsampled_dales_gpu.py --method IDIS --loss-levels 10 30 50 70 90

    # FPS
    python generate_subsampled_dales_gpu.py --method FPS --loss-levels 10 30 50 70 90

    # IDIS with custom radius (for ablation)
    python generate_subsampled_dales_gpu.py --method IDIS --radius 5 --loss-levels 50
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
    print("ERROR: CUDA not available. Use generate_subsampled_dales.py for CPU processing.")
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

DEFAULT_INPUT_DIR = "data/DALES/original"
DEFAULT_OUTPUT_DIR = "data/DALES/subsampled"

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
# DALES Data Loading/Saving
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
    return_num = np.ones(len(points), dtype=np.int32)
    num_returns = np.ones(len(points), dtype=np.int32)

    # Stack all columns: x, y, z, intensity, return_num, num_returns, class
    output = np.column_stack([
        points,
        intensity,
        return_num,
        num_returns,
        labels
    ])

    # Save with appropriate format
    np.savetxt(
        output_file,
        output,
        fmt='%.6f %.6f %.6f %.3f %d %d %d',
        delimiter=' '
    )


# ============================================================================
# GPU Processing
# ============================================================================

def process_tile_gpu(
    tile_file: str,
    output_file: str,
    method: str,
    loss_level: float,
    seed: int,
    radius: float = 10.0
):
    """Process a single DALES tile using GPU acceleration."""

    # Check if output already exists
    if os.path.exists(output_file):
        return {'status': 'skipped', 'tile': tile_file}

    # Load tile
    points, intensity, labels = load_dales_tile(tile_file)
    n_original = len(points)

    # Apply GPU subsampling
    if method == 'IDIS':
        sampled_points, sampled_intensity, sampled_labels = idis_subsample_with_loss_gpu(
            points,
            loss_percentage=loss_level,
            features=intensity[:, np.newaxis],
            labels=labels,
            radius=radius,
            seed=seed,
            verbose=False
        )

    elif method == 'FPS':
        sampled_points, sampled_intensity, sampled_labels = fps_subsample_with_loss_gpu(
            points,
            loss_percentage=loss_level,
            features=intensity[:, np.newaxis],
            labels=labels,
            seed=seed,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'IDIS' or 'FPS'")

    sampled_intensity = sampled_intensity.flatten()
    n_sampled = len(sampled_points)

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save subsampled tile
    save_dales_tile(sampled_points, sampled_intensity, sampled_labels, output_file)

    return {
        'status': 'success',
        'tile': tile_file,
        'n_original': n_original,
        'n_sampled': n_sampled,
        'actual_loss': (1 - n_sampled / n_original) * 100
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
    tile_files = []

    for subdir in ['train', 'test']:
        subdir_path = Path(input_dir) / subdir
        if not subdir_path.exists():
            continue

        for txt_file in subdir_path.glob('*.txt'):
            tile_name = txt_file.stem
            if tiles is None or tile_name in tiles:
                tile_files.append(txt_file)

    return sorted(tile_files)


def generate_subsampled_gpu(
    input_dir: str,
    output_dir: str,
    tiles: list,
    method: str,
    loss_levels: list,
    seed: int = 1,
    radius: float = 10.0,
    radius_suffix: str = None
):
    """Generate subsampled DALES dataset using GPU acceleration."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find tile files
    tile_files = find_tile_files(input_dir, tiles)

    if len(tile_files) == 0:
        raise ValueError(f"No .txt tile files found in {input_dir}")

    print(f"\n{'='*70}")
    print(f"GPU-Accelerated {method} Subsampling (DALES)")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tiles: {len(tile_files)}")
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
        if radius_suffix:
            output_subdir = f"{method}_{radius_suffix}_loss{loss_level}_seed{seed}"
        else:
            output_subdir = f"{method}_loss{loss_level}_seed{seed}"

        success_count = 0
        skip_count = 0
        error_count = 0

        loss_start = time.time()

        # Process tiles with progress bar
        for tile_file in tqdm(tile_files, desc=f"  {method}", unit="tile"):
            tile_name = tile_file.stem + '.txt'
            output_file = output_dir / output_subdir / tile_name

            try:
                result = process_tile_gpu(
                    str(tile_file),
                    str(output_file),
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

            except Exception as e:
                print(f"\n  Error processing {tile_file.name}: {e}")
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
        description='GPU-accelerated subsampling for DALES (IDIS/FPS)'
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
        '--tiles',
        type=str,
        nargs='+',
        default=None,
        help='Specific tiles to process (default: all 40 tiles)'
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
        '--input-dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'Input directory (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return 1

    try:
        generate_subsampled_gpu(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tiles=args.tiles,
            method=args.method,
            loss_levels=args.loss_levels,
            seed=args.seed,
            radius=args.radius,
            radius_suffix=args.radius_suffix
        )
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
