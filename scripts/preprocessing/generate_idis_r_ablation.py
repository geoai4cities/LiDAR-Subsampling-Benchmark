#!/usr/bin/env python3
"""
Generate IDIS R-Value Ablation Data

Generates subsampled data with IDIS using different radius (R) values
for sensitivity analysis as requested by reviewers (Comment 1.2, 6.5).

R values tested: 5m, 10m, 15m, 20m
Loss level: 50% (fixed for ablation study)
Seed: 1 (single seed sufficient for parameter sensitivity)

Output Structure:
    data/{dataset}/subsampled/
    ├── IDIS_R5_loss50_seed1/   # R=5m (same as default IDIS)
    ├── IDIS_R10_loss50_seed1/  # R=10m
    ├── IDIS_R15_loss50_seed1/  # R=15m
    └── IDIS_R20_loss50_seed1/  # R=20m

Usage:
    # Generate for SemanticKITTI
    python generate_idis_r_ablation.py --dataset semantickitti --workers 4

    # Generate for DALES
    python generate_idis_r_ablation.py --dataset dales --workers 4

    # Generate for both
    python generate_idis_r_ablation.py --dataset all --workers 4
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from subsampling import idis_subsample_with_loss

# ============================================================================
# Configuration
# ============================================================================

# R values for ablation study (meters)
R_VALUES = [5, 10, 15, 20]

# Fixed parameters for ablation
LOSS_LEVEL = 50  # 50% loss (keep 50% of points)
SEED = 1         # Single seed for ablation
DISTANCE_EXPONENT = -2.0  # Default exponent

# Dataset configurations
DATASETS = {
    'semantickitti': {
        'input_dir': 'data/SemanticKITTI/original/sequences/00/velodyne',
        'labels_dir': 'data/SemanticKITTI/original/sequences/00/labels',
        'output_dir': 'data/SemanticKITTI/subsampled',
        'file_ext': '.bin',
        'loader': 'semantickitti'
    },
    'dales': {
        'input_dir': 'data/DALES/original',
        'labels_dir': None,  # Labels in same file
        'output_dir': 'data/DALES/subsampled',
        'file_ext': '.txt',
        'loader': 'dales'
    }
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_semantickitti_scan(bin_file, labels_file=None):
    """Load SemanticKITTI scan."""
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    intensity = scan[:, 3]

    labels = None
    if labels_file and os.path.exists(labels_file):
        labels = np.fromfile(labels_file, dtype=np.uint32)
        labels = labels & 0xFFFF  # Lower 16 bits are semantic labels

    return points, intensity, labels


def save_semantickitti_scan(points, intensity, labels, output_file, labels_output=None):
    """Save SemanticKITTI scan."""
    scan = np.hstack([points, intensity.reshape(-1, 1)]).astype(np.float32)
    scan.tofile(output_file)

    if labels is not None and labels_output:
        labels.astype(np.uint32).tofile(labels_output)


def load_dales_tile(txt_file):
    """Load DALES tile from TXT file."""
    data = np.loadtxt(txt_file)
    points = data[:, :3]

    # DALES format: x, y, z, intensity, return_num, num_returns, class
    intensity = data[:, 3] if data.shape[1] > 3 else np.zeros(len(points))
    labels = data[:, 6].astype(np.int32) if data.shape[1] > 6 else None

    return points, intensity, labels, data


def save_dales_tile(points, data_template, output_file):
    """Save DALES tile maintaining original format."""
    # Update coordinates in template
    output_data = data_template.copy()
    output_data[:len(points), :3] = points
    output_data = output_data[:len(points)]

    np.savetxt(output_file, output_data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f %.0f')


# ============================================================================
# Processing Functions
# ============================================================================

def process_semantickitti_scan(args):
    """Process a single SemanticKITTI scan with IDIS at specific R value."""
    scan_file, labels_file, output_file, labels_output, r_value = args

    try:
        # Skip if already exists
        if os.path.exists(output_file):
            return scan_file, 'skipped', r_value

        # Load scan
        points, intensity, labels = load_semantickitti_scan(scan_file, labels_file)

        if len(points) == 0:
            return scan_file, 'empty', r_value

        # Apply IDIS with specific R value
        if labels is not None:
            sampled_points, sampled_intensity, sampled_labels = idis_subsample_with_loss(
                points=points,
                loss_percentage=LOSS_LEVEL,
                features=intensity.reshape(-1, 1),
                labels=labels,
                radius=r_value,
                distance_exponent=DISTANCE_EXPONENT,
                seed=SEED
            )
        else:
            sampled_points, sampled_intensity = idis_subsample_with_loss(
                points=points,
                loss_percentage=LOSS_LEVEL,
                features=intensity.reshape(-1, 1),
                radius=r_value,
                distance_exponent=DISTANCE_EXPONENT,
                seed=SEED
            )
            sampled_labels = None

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_semantickitti_scan(
            sampled_points,
            sampled_intensity.flatten(),
            sampled_labels,
            output_file,
            labels_output
        )

        return scan_file, 'success', r_value

    except Exception as e:
        return scan_file, f'error: {e}', r_value


def process_dales_tile(args):
    """Process a single DALES tile with IDIS at specific R value."""
    tile_file, output_file, r_value = args

    try:
        # Skip if already exists
        if os.path.exists(output_file):
            return tile_file, 'skipped', r_value

        # Load tile
        points, intensity, labels, data_template = load_dales_tile(tile_file)

        if len(points) == 0:
            return tile_file, 'empty', r_value

        # Apply IDIS with specific R value
        if labels is not None:
            sampled_points, sampled_intensity, sampled_labels, indices = idis_subsample_with_loss(
                points=points,
                loss_percentage=LOSS_LEVEL,
                features=intensity.reshape(-1, 1),
                labels=labels,
                radius=r_value,
                distance_exponent=DISTANCE_EXPONENT,
                seed=SEED,
                return_indices=True
            )
        else:
            sampled_points, sampled_intensity, indices = idis_subsample_with_loss(
                points=points,
                loss_percentage=LOSS_LEVEL,
                features=intensity.reshape(-1, 1),
                radius=r_value,
                distance_exponent=DISTANCE_EXPONENT,
                seed=SEED,
                return_indices=True
            )

        # Save with original format
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sampled_data = data_template[indices]
        sampled_data[:, :3] = sampled_points
        np.savetxt(output_file, sampled_data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f %.0f')

        return tile_file, 'success', r_value

    except Exception as e:
        return tile_file, f'error: {e}', r_value


# ============================================================================
# Main Functions
# ============================================================================

def generate_semantickitti_r_ablation(workers=4):
    """Generate IDIS R-value variants for SemanticKITTI."""
    config = DATASETS['semantickitti']

    input_dir = Path(config['input_dir'])
    labels_dir = Path(config['labels_dir'])
    output_base = Path(config['output_dir'])

    # Get all scan files
    scan_files = sorted(input_dir.glob('*.bin'))

    print(f"\nSemanticKITTI IDIS R-Value Ablation")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_base}")
    print(f"Scans: {len(scan_files)}")
    print(f"R values: {R_VALUES}")
    print(f"Loss: {LOSS_LEVEL}%")
    print(f"Workers: {workers}")
    print()

    for r_value in R_VALUES:
        print(f"\n{'─'*60}")
        print(f"Processing R = {r_value}m")
        print(f"{'─'*60}")

        output_dir = output_base / f"IDIS_R{r_value}_loss{LOSS_LEVEL}_seed{SEED}"
        velodyne_out = output_dir / "velodyne"
        labels_out = output_dir / "labels"

        # Prepare tasks
        tasks = []
        for scan_file in scan_files:
            scan_name = scan_file.stem
            label_file = labels_dir / f"{scan_name}.label"

            out_scan = velodyne_out / f"{scan_name}.bin"
            out_label = labels_out / f"{scan_name}.label"

            tasks.append((
                str(scan_file),
                str(label_file) if label_file.exists() else None,
                str(out_scan),
                str(out_label),
                r_value
            ))

        # Process with progress bar
        success = 0
        skipped = 0
        errors = 0

        with Pool(workers) as pool:
            for result in tqdm(pool.imap_unordered(process_semantickitti_scan, tasks),
                             total=len(tasks), desc=f"R={r_value}m"):
                _, status, _ = result
                if status == 'success':
                    success += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    errors += 1

        print(f"  R={r_value}m: {success} success, {skipped} skipped, {errors} errors")

    print(f"\n{'='*60}")
    print("SemanticKITTI IDIS R-value ablation complete!")


def generate_dales_r_ablation(workers=4):
    """Generate IDIS R-value variants for DALES."""
    config = DATASETS['dales']

    input_base = Path(config['input_dir'])
    output_base = Path(config['output_dir'])

    # Find all tile files
    tile_files = []
    for subdir in ['train', 'test']:
        subdir_path = input_base / subdir
        if subdir_path.exists():
            tile_files.extend(sorted(subdir_path.glob('*.txt')))

    # Also check dales_ply directory structure
    ply_path = input_base / 'dales_ply'
    if ply_path.exists():
        for subdir in ['train', 'test']:
            subdir_path = ply_path / subdir
            if subdir_path.exists():
                tile_files.extend(sorted(subdir_path.glob('*.txt')))

    print(f"\nDALES IDIS R-Value Ablation")
    print(f"{'='*60}")
    print(f"Input: {input_base}")
    print(f"Output: {output_base}")
    print(f"Tiles: {len(tile_files)}")
    print(f"R values: {R_VALUES}")
    print(f"Loss: {LOSS_LEVEL}%")
    print(f"Workers: {workers}")
    print()

    for r_value in R_VALUES:
        print(f"\n{'─'*60}")
        print(f"Processing R = {r_value}m")
        print(f"{'─'*60}")

        output_dir = output_base / f"IDIS_R{r_value}_loss{LOSS_LEVEL}_seed{SEED}"

        # Prepare tasks
        tasks = []
        for tile_file in tile_files:
            tile_name = tile_file.stem
            out_file = output_dir / f"{tile_name}.txt"

            tasks.append((str(tile_file), str(out_file), r_value))

        # Process with progress bar
        success = 0
        skipped = 0
        errors = 0

        with Pool(workers) as pool:
            for result in tqdm(pool.imap_unordered(process_dales_tile, tasks),
                             total=len(tasks), desc=f"R={r_value}m"):
                _, status, _ = result
                if status == 'success':
                    success += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    errors += 1

        print(f"  R={r_value}m: {success} success, {skipped} skipped, {errors} errors")

    print(f"\n{'='*60}")
    print("DALES IDIS R-value ablation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate IDIS R-value ablation data for sensitivity analysis'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['semantickitti', 'dales', 'all'],
        help='Dataset to process (default: all)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, max recommended due to memory)'
    )
    parser.add_argument(
        '--r-values',
        type=int,
        nargs='+',
        default=R_VALUES,
        help=f'R values to test in meters (default: {R_VALUES})'
    )

    args = parser.parse_args()

    # Update R values if custom provided
    global R_VALUES
    R_VALUES = args.r_values

    print("="*70)
    print("IDIS R-Value Ablation Study - Data Generation")
    print("="*70)
    print(f"R values: {R_VALUES} meters")
    print(f"Loss level: {LOSS_LEVEL}%")
    print(f"Seed: {SEED}")
    print(f"Distance exponent: {DISTANCE_EXPONENT}")
    print(f"Workers: {args.workers}")
    print()

    start_time = time.time()

    if args.dataset in ['semantickitti', 'all']:
        generate_semantickitti_r_ablation(workers=args.workers)

    if args.dataset in ['dales', 'all']:
        generate_dales_r_ablation(workers=args.workers)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
