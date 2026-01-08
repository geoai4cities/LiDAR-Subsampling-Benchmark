#!/usr/bin/env python3
"""
Restructure Subsampled SemanticKITTI Data for PTv3 Dataset Loader

The PTv3 SemanticKITTI dataset loader expects:
    data_root/dataset/sequences/XX/velodyne/XXXXXX.bin
    data_root/dataset/sequences/XX/labels/XXXXXX.label

Current subsampled structure is:
    subsampled/RS_loss10_seed1/XXXXXX.bin
    subsampled/RS_loss10_seed1/XXXXXX.label

This script restructures to:
    subsampled/RS_loss10_seed1/dataset/sequences/00/velodyne/XXXXXX.bin
    subsampled/RS_loss10_seed1/dataset/sequences/00/labels/XXXXXX.label

Usage:
    python restructure_subsampled_semantickitti.py
    python restructure_subsampled_semantickitti.py --dry-run  # Preview changes
"""

import os
import shutil
from pathlib import Path
import argparse


def restructure_subsampled_dir(subsampled_root, dry_run=False):
    """
    Restructure all subsampled directories to match PTv3 expected format.

    Args:
        subsampled_root: Path to subsampled directory
        dry_run: If True, only print what would be done
    """
    subsampled_path = Path(subsampled_root)

    if not subsampled_path.exists():
        print(f"Error: {subsampled_path} does not exist")
        return

    # Get all method directories (RS_loss10_seed1, etc.)
    method_dirs = [d for d in subsampled_path.iterdir() if d.is_dir()]

    print(f"Found {len(method_dirs)} subsampled variants to restructure")

    for method_dir in sorted(method_dirs):
        # Check if already restructured
        expected_structure = method_dir / "dataset" / "sequences" / "00" / "velodyne"
        if expected_structure.exists():
            print(f"  {method_dir.name}: Already restructured, skipping")
            continue

        # Check if it has flat .bin files
        bin_files = list(method_dir.glob("*.bin"))
        label_files = list(method_dir.glob("*.label"))

        if not bin_files:
            print(f"  {method_dir.name}: No .bin files found, skipping")
            continue

        print(f"  {method_dir.name}: Restructuring {len(bin_files)} bin files, {len(label_files)} label files")

        if dry_run:
            print(f"    Would create: {method_dir}/dataset/sequences/00/velodyne/")
            print(f"    Would create: {method_dir}/dataset/sequences/00/labels/")
            print(f"    Would move {len(bin_files)} .bin files")
            print(f"    Would move {len(label_files)} .label files")
            continue

        # Create new directory structure
        velodyne_dir = method_dir / "dataset" / "sequences" / "00" / "velodyne"
        labels_dir = method_dir / "dataset" / "sequences" / "00" / "labels"

        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Move .bin files to velodyne/
        for bin_file in bin_files:
            dest = velodyne_dir / bin_file.name
            shutil.move(str(bin_file), str(dest))

        # Move .label files to labels/
        for label_file in label_files:
            dest = labels_dir / label_file.name
            shutil.move(str(label_file), str(dest))

        print(f"    Done! Moved {len(bin_files)} bin + {len(label_files)} label files")

    print("\nRestructuring complete!")


def main():
    parser = argparse.ArgumentParser(description="Restructure subsampled SemanticKITTI data")
    parser.add_argument("--subsampled-root", type=str,
                        default="data/SemanticKITTI/subsampled",
                        help="Path to subsampled directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without making them")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    subsampled_root = project_root / args.subsampled_root

    print(f"Restructuring subsampled data in: {subsampled_root}")
    print(f"Dry run: {args.dry_run}")
    print()

    restructure_subsampled_dir(subsampled_root, args.dry_run)


if __name__ == "__main__":
    main()
