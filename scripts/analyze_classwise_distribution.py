#!/usr/bin/env python3
"""
Analyze Class-wise Point Distribution for Subsampled SemanticKITTI Data

This script analyzes how different subsampling methods affect the class-wise
distribution of points in the SemanticKITTI dataset, separately for train and val splits.

Usage:
    python analyze_classwise_distribution.py --method RS --loss 90 --seed 1
    python analyze_classwise_distribution.py --method IDIS --loss 90 --r 10
    python analyze_classwise_distribution.py --method DBSCAN --loss 90
    python analyze_classwise_distribution.py --all  # Analyze all available methods

Output: Tables showing class-wise point counts and retention rates for train and val.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict
from datetime import datetime
import json

# SemanticKITTI label mapping (learning map)
# Maps raw labels to training labels (0-19, where 0 is ignored/unlabeled)
LEARNING_MAP = {
    0: 0,      # "unlabeled"
    1: 0,      # "outlier"
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" -> other-vehicle
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" -> other-vehicle
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" -> unlabeled
    60: 9,     # "lane-marking" -> road
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" -> unlabeled
    252: 1,    # "moving-car" -> car
    253: 7,    # "moving-bicyclist" -> bicyclist
    254: 6,    # "moving-person" -> person
    255: 8,    # "moving-motorcyclist" -> motorcyclist
    256: 5,    # "moving-on-rails" -> other-vehicle
    257: 5,    # "moving-bus" -> other-vehicle
    258: 4,    # "moving-truck" -> truck
    259: 5,    # "moving-other-vehicle" -> other-vehicle
}

# Class names for SemanticKITTI (19 classes + unlabeled)
CLASS_NAMES = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign',
}

# Training and validation sequences
TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQUENCES = ['08']


def read_labels(label_path):
    """Read SemanticKITTI label file."""
    labels = np.fromfile(label_path, dtype=np.uint32)
    # Lower 16 bits are semantic labels, upper 16 bits are instance labels
    semantic_labels = labels & 0xFFFF
    return semantic_labels


def map_labels(labels):
    """Map raw labels to training labels using learning map."""
    mapped = np.zeros_like(labels)
    for raw_label, train_label in LEARNING_MAP.items():
        mapped[labels == raw_label] = train_label
    return mapped


def count_class_points(data_path, sequences, verbose=True):
    """
    Count points per class across specified sequences.

    Args:
        data_path: Path to data directory (original or subsampled)
        sequences: List of sequences to process
        verbose: Print progress for each sequence

    Returns:
        dict: {class_id: point_count}, total_points
    """
    class_counts = defaultdict(int)
    total_points = 0

    seq_path = Path(data_path) / 'sequences'

    for seq in sequences:
        labels_dir = seq_path / seq / 'labels'
        if not labels_dir.exists():
            if verbose:
                print(f"    Sequence {seq}: not found, skipping")
            continue

        label_files = sorted(glob(str(labels_dir / '*.label')))
        seq_points = 0

        for label_file in label_files:
            labels = read_labels(label_file)
            mapped_labels = map_labels(labels)

            # Count points per class
            unique, counts = np.unique(mapped_labels, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_counts[class_id] += count
                total_points += count
                seq_points += count

        if verbose:
            print(f"    Sequence {seq}: {len(label_files):,} frames, {seq_points:,} points")

    return dict(class_counts), total_points


def get_data_path(base_path, method, loss, seed=None, r_value=None):
    """
    Construct the data path based on method, loss, seed, and R value.
    """
    if loss == 0:
        return Path(base_path) / 'original'

    # Non-deterministic methods require seed
    non_det_methods = ['RS', 'FPS', 'Poisson']

    if method in non_det_methods:
        if seed is None:
            seed = 1
        folder_name = f"{method}_loss{loss}_seed{seed}"
    elif method == 'IDIS' and r_value and r_value != 10:
        folder_name = f"IDIS_R{r_value}_loss{loss}"
    else:
        folder_name = f"{method}_loss{loss}"

    return Path(base_path) / 'subsampled' / folder_name


def format_split_table(baseline_counts, baseline_total, subsampled_counts, subsampled_total,
                       split_name, method, loss, seed=None, r_value=None):
    """Format results for a single split (train or val) as a table section."""
    lines = []

    # Header for split
    lines.append(f"\n{'-'*100}")
    lines.append(f"{split_name.upper()} SPLIT")
    lines.append(f"{'-'*100}")

    # Summary
    retention = (subsampled_total / baseline_total * 100) if baseline_total > 0 else 0
    actual_loss = 100 - retention
    lines.append(f"Baseline Points: {baseline_total:,}")
    lines.append(f"Subsampled Points: {subsampled_total:,}")
    lines.append(f"Retention: {retention:.2f}% | Actual Loss: {actual_loss:.2f}%")
    lines.append("")

    # Class-wise table header
    header = f"{'Class':<15} | {'ID':^4} | {'Baseline':>12} | {'Subsampled':>12} | {'Retention %':>12} | {'% of Total':>10}"
    lines.append(header)
    lines.append("-" * 80)

    # Data rows (skip unlabeled class 0)
    for class_id in range(1, 20):
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        baseline = baseline_counts.get(class_id, 0)
        subsampled = subsampled_counts.get(class_id, 0)

        retention_pct = (subsampled / baseline * 100) if baseline > 0 else 0
        pct_of_total = (subsampled / subsampled_total * 100) if subsampled_total > 0 else 0

        row = f"{class_name:<15} | {class_id:^4} | {baseline:>12,} | {subsampled:>12,} | {retention_pct:>11.2f}% | {pct_of_total:>9.2f}%"
        lines.append(row)

    lines.append("-" * 80)

    # Unlabeled row
    unlabeled_baseline = baseline_counts.get(0, 0)
    unlabeled_subsampled = subsampled_counts.get(0, 0)
    unlabeled_retention = (unlabeled_subsampled / unlabeled_baseline * 100) if unlabeled_baseline > 0 else 0
    unlabeled_pct = (unlabeled_subsampled / subsampled_total * 100) if subsampled_total > 0 else 0
    lines.append(f"{'unlabeled':<15} | {0:^4} | {unlabeled_baseline:>12,} | {unlabeled_subsampled:>12,} | {unlabeled_retention:>11.2f}% | {unlabeled_pct:>9.2f}%")

    # Total row
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':<15} | {'-':^4} | {baseline_total:>12,} | {subsampled_total:>12,} | {retention:>11.2f}% | {'100.00':>9}%")

    return lines


def format_combined_table(train_data, val_data, method, loss, seed=None, r_value=None):
    """Format combined train+val table with class, train retention, val retention columns."""
    lines = []

    # Main header
    lines.append("=" * 120)
    title = f"CLASS-WISE POINT DISTRIBUTION: {method}"
    if seed:
        title += f" (Seed {seed})"
    if r_value:
        title += f" (R={r_value})"
    title += f" at {loss}% Loss"
    lines.append(title)
    lines.append("=" * 120)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overall summary
    train_retention = (train_data['sub_total'] / train_data['base_total'] * 100) if train_data['base_total'] > 0 else 0
    val_retention = (val_data['sub_total'] / val_data['base_total'] * 100) if val_data['base_total'] > 0 else 0

    lines.append(f"TRAIN: Baseline {train_data['base_total']:,} -> Subsampled {train_data['sub_total']:,} ({train_retention:.2f}% retained)")
    lines.append(f"VAL:   Baseline {val_data['base_total']:,} -> Subsampled {val_data['sub_total']:,} ({val_retention:.2f}% retained)")
    lines.append("")

    # Combined class-wise table
    lines.append("-" * 120)
    header = f"{'Class':<15} | {'ID':^4} | {'Train Base':>12} | {'Train Sub':>12} | {'Train %':>9} | {'Val Base':>12} | {'Val Sub':>12} | {'Val %':>9}"
    lines.append(header)
    lines.append("-" * 120)

    # Data rows
    for class_id in range(1, 20):
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')

        train_base = train_data['base_counts'].get(class_id, 0)
        train_sub = train_data['sub_counts'].get(class_id, 0)
        train_pct = (train_sub / train_base * 100) if train_base > 0 else 0

        val_base = val_data['base_counts'].get(class_id, 0)
        val_sub = val_data['sub_counts'].get(class_id, 0)
        val_pct = (val_sub / val_base * 100) if val_base > 0 else 0

        row = f"{class_name:<15} | {class_id:^4} | {train_base:>12,} | {train_sub:>12,} | {train_pct:>8.2f}% | {val_base:>12,} | {val_sub:>12,} | {val_pct:>8.2f}%"
        lines.append(row)

    lines.append("-" * 120)

    # Unlabeled
    train_base = train_data['base_counts'].get(0, 0)
    train_sub = train_data['sub_counts'].get(0, 0)
    train_pct = (train_sub / train_base * 100) if train_base > 0 else 0
    val_base = val_data['base_counts'].get(0, 0)
    val_sub = val_data['sub_counts'].get(0, 0)
    val_pct = (val_sub / val_base * 100) if val_base > 0 else 0
    lines.append(f"{'unlabeled':<15} | {0:^4} | {train_base:>12,} | {train_sub:>12,} | {train_pct:>8.2f}% | {val_base:>12,} | {val_sub:>12,} | {val_pct:>8.2f}%")

    # Totals
    lines.append("-" * 120)
    lines.append(f"{'TOTAL':<15} | {'-':^4} | {train_data['base_total']:>12,} | {train_data['sub_total']:>12,} | {train_retention:>8.2f}% | {val_data['base_total']:>12,} | {val_data['sub_total']:>12,} | {val_retention:>8.2f}%")
    lines.append("=" * 120)
    lines.append("")

    return "\n".join(lines)


def analyze_single(base_path, method, loss, seed=None, r_value=None, output_dir=None):
    """Analyze a single method/loss combination for both train and val."""
    # Build title
    title = f"{method}"
    if seed:
        title += f" (Seed {seed})"
    if r_value:
        title += f" (R={r_value})"
    title += f" at {loss}% Loss"

    print("")
    print("=" * 70)
    print(f"  Analyzing: {title}")
    print("=" * 70)
    print(f"  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started")
    print("")

    # Get baseline (original data)
    baseline_path = Path(base_path) / 'original'

    # Count baseline for train and val
    print("  [1/4] Reading BASELINE (Train: sequences 00-07, 09-10)")
    print("  " + "-" * 50)
    train_base_counts, train_base_total = count_class_points(baseline_path, TRAIN_SEQUENCES)
    print(f"  Train baseline total: {train_base_total:,} points")
    print("")

    print("  [2/4] Reading BASELINE (Val: sequence 08)")
    print("  " + "-" * 50)
    val_base_counts, val_base_total = count_class_points(baseline_path, VAL_SEQUENCES)
    print(f"  Val baseline total: {val_base_total:,} points")
    print("")

    if loss == 0:
        # Baseline only
        train_sub_counts, train_sub_total = train_base_counts, train_base_total
        val_sub_counts, val_sub_total = val_base_counts, val_base_total
        print("  [3/4] Skipped (baseline = subsampled for loss 0%)")
        print("  [4/4] Skipped (baseline = subsampled for loss 0%)")
    else:
        # Get subsampled data path
        subsampled_path = get_data_path(base_path, method, loss, seed, r_value)
        print(f"  Subsampled data: {subsampled_path}")
        print("")

        if not subsampled_path.exists():
            print(f"  ERROR: Path does not exist!")
            return None

        print("  [3/4] Reading SUBSAMPLED (Train: sequences 00-07, 09-10)")
        print("  " + "-" * 50)
        train_sub_counts, train_sub_total = count_class_points(subsampled_path, TRAIN_SEQUENCES)
        print(f"  Train subsampled total: {train_sub_total:,} points")
        print("")

        print("  [4/4] Reading SUBSAMPLED (Val: sequence 08)")
        print("  " + "-" * 50)
        val_sub_counts, val_sub_total = count_class_points(subsampled_path, VAL_SEQUENCES)
        print(f"  Val subsampled total: {val_sub_total:,} points")
        print("")

    print(f"  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing complete")
    print("=" * 70)
    print("")

    # Prepare data structures
    train_data = {
        'base_counts': train_base_counts,
        'base_total': train_base_total,
        'sub_counts': train_sub_counts,
        'sub_total': train_sub_total,
    }
    val_data = {
        'base_counts': val_base_counts,
        'base_total': val_base_total,
        'sub_counts': val_sub_counts,
        'sub_total': val_sub_total,
    }

    # Format combined table
    table = format_combined_table(train_data, val_data, method, loss, seed, r_value)

    # Print to console
    print(table)

    # Save to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"classwise_{method}_loss{loss}"
        if seed:
            filename += f"_seed{seed}"
        if r_value and r_value != 10:
            filename += f"_R{r_value}"
        filename += ".txt"

        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"  Saved to: {output_file}")

    # Return data for further processing
    return {
        'method': method,
        'loss': loss,
        'seed': seed,
        'r_value': r_value,
        'train': train_data,
        'val': val_data,
    }


def discover_available_methods(base_path):
    """Discover all available subsampled methods."""
    subsampled_dir = Path(base_path) / 'subsampled'
    if not subsampled_dir.exists():
        return []

    methods = []
    for folder in sorted(subsampled_dir.iterdir()):
        if folder.is_dir() and folder.name != 'reports':
            name = folder.name
            parts = name.split('_')
            method = parts[0]

            # Check for R value (IDIS_R5, IDIS_R15)
            r_value = None
            if method == 'IDIS' and len(parts) > 1 and parts[1].startswith('R'):
                r_value = int(parts[1][1:])
                parts = [parts[0]] + parts[2:]

            # Extract loss and seed
            loss = None
            seed = None
            for part in parts[1:]:
                if part.startswith('loss'):
                    loss = int(part.replace('loss', ''))
                elif part.startswith('seed'):
                    seed = int(part.replace('seed', ''))

            if loss is not None:
                methods.append({
                    'method': method,
                    'loss': loss,
                    'seed': seed,
                    'r_value': r_value,
                    'folder': name
                })

    return methods


def create_summary_table(all_results, output_dir=None):
    """Create a summary table comparing all methods for train and val."""
    lines = []
    lines.append("=" * 160)
    lines.append("SUMMARY: CLASS-WISE RETENTION RATES - TRAIN vs VAL")
    lines.append("=" * 160)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Group by loss level
    by_loss = defaultdict(list)
    for result in all_results:
        if result:
            by_loss[result['loss']].append(result)

    for loss in sorted(by_loss.keys()):
        results = by_loss[loss]

        lines.append("")
        lines.append(f"{'='*160}")
        lines.append(f"LOSS LEVEL: {loss}%")
        lines.append(f"{'='*160}")
        lines.append("")

        # Method headers
        method_headers = []
        for r in results:
            header = r['method']
            if r['seed']:
                header += f"_S{r['seed']}"
            if r['r_value'] and r['r_value'] != 10:
                header += f"_R{r['r_value']}"
            method_headers.append(header)

        # Header row
        header_row = f"{'Class':<15} |"
        for h in method_headers:
            header_row += f" {h:^15} |"
        lines.append("-" * len(header_row))

        # Sub-header for Train/Val
        sub_header = f"{'':<15} |"
        for _ in method_headers:
            sub_header += f" {'Train':>6} {'Val':>6} |"
        lines.append(header_row)
        lines.append(sub_header)
        lines.append("-" * len(header_row))

        # Data rows
        for class_id in range(1, 20):
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')[:15]
            row = f"{class_name:<15} |"

            for r in results:
                train_base = r['train']['base_counts'].get(class_id, 0)
                train_sub = r['train']['sub_counts'].get(class_id, 0)
                train_pct = (train_sub / train_base * 100) if train_base > 0 else 0

                val_base = r['val']['base_counts'].get(class_id, 0)
                val_sub = r['val']['sub_counts'].get(class_id, 0)
                val_pct = (val_sub / val_base * 100) if val_base > 0 else 0

                row += f" {train_pct:>5.1f}% {val_pct:>5.1f}% |"

            lines.append(row)

        lines.append("-" * len(header_row))

        # Overall retention
        row = f"{'OVERALL':<15} |"
        for r in results:
            train_pct = (r['train']['sub_total'] / r['train']['base_total'] * 100) if r['train']['base_total'] > 0 else 0
            val_pct = (r['val']['sub_total'] / r['val']['base_total'] * 100) if r['val']['base_total'] > 0 else 0
            row += f" {train_pct:>5.1f}% {val_pct:>5.1f}% |"
        lines.append(row)
        lines.append("=" * len(header_row))

    summary = "\n".join(lines)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "classwise_summary_all_methods.txt"
        with open(output_file, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze class-wise point distribution for subsampled SemanticKITTI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single method
    python analyze_classwise_distribution.py --method RS --loss 90 --seed 1
    python analyze_classwise_distribution.py --method IDIS --loss 90 --r 10
    python analyze_classwise_distribution.py --method DBSCAN --loss 90

    # Analyze baseline (original data)
    python analyze_classwise_distribution.py --method baseline --loss 0

    # Analyze all available methods
    python analyze_classwise_distribution.py --all

    # List available methods
    python analyze_classwise_distribution.py --list
        """
    )

    parser.add_argument('--method', '-m', type=str,
                       help='Subsampling method (RS, FPS, Poisson, IDIS, DBSCAN, Voxel, DEPOCO, baseline)')
    parser.add_argument('--loss', '-l', type=int,
                       help='Loss level (0, 10, 30, 50, 70, 90)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Seed for non-deterministic methods (RS, FPS, Poisson)')
    parser.add_argument('--r', type=int, default=None,
                       help='R value for IDIS method (5, 10, 15, 20)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Analyze all available methods')
    parser.add_argument('--list', action='store_true',
                       help='List all available methods')
    # Default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent of scripts/

    parser.add_argument('--data-path', '-d', type=str,
                       default=os.path.join(base_dir, 'data/SemanticKITTI'),
                       help='Path to SemanticKITTI data directory')
    parser.add_argument('--output', '-o', type=str,
                       default=os.path.join(base_dir, 'docs/tables/classwise'),
                       help='Output directory for tables')

    args = parser.parse_args()

    # List available methods
    if args.list:
        print("Discovering available methods...")
        methods = discover_available_methods(args.data_path)
        print(f"\nAvailable subsampled datasets ({len(methods)}):")
        print("-" * 60)
        for m in methods:
            print(f"  {m['folder']}")
        return

    # Analyze all methods
    if args.all:
        print("Analyzing all available methods...")
        methods = discover_available_methods(args.data_path)

        all_results = []
        for m in methods:
            result = analyze_single(args.data_path, m['method'], m['loss'],
                                   m['seed'], m['r_value'], args.output)
            all_results.append(result)

        # Create summary
        if all_results:
            summary = create_summary_table(all_results, args.output)
            print("\n" + summary)

        return

    # Single method analysis
    if args.method is None or args.loss is None:
        parser.print_help()
        print("\nError: --method and --loss are required for single analysis")
        print("       Use --all to analyze all methods, or --list to see available methods")
        sys.exit(1)

    # Handle baseline
    if args.method.lower() == 'baseline':
        args.method = 'baseline'
        args.loss = 0

    analyze_single(args.data_path, args.method, args.loss, args.seed, args.r, args.output)


if __name__ == '__main__':
    main()
