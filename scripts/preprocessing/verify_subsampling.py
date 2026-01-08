#!/usr/bin/env python3
"""
Verification Script for Subsampled Datasets

This script verifies the actual point loss percentage in subsampled datasets
by comparing point counts with the original data.

Usage:
    # Verify a specific method/loss combination
    python verify_subsampling.py --method DBSCAN --loss 10 --seed 1

    # Verify all subsampled datasets
    python verify_subsampling.py --all

    # Verify specific folder
    python verify_subsampling.py --folder DBSCAN_loss10_seed1

    # Quick verification (sample only)
    python verify_subsampling.py --method IDIS --loss 50 --quick

Reports are automatically saved to:
    data/SemanticKITTI/subsampled/reports/
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# SemanticKITTI sequences
SEMANTICKITTI_TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
SEMANTICKITTI_VAL_SEQS = ["08"]
SEMANTICKITTI_TEST_SEQS = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]


def count_points_in_bin(bin_path: str) -> int:
    """Count number of points in a .bin file."""
    try:
        points = np.fromfile(bin_path, dtype=np.float32)
        # SemanticKITTI format: x, y, z, intensity (4 floats per point)
        return len(points) // 4
    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return 0


def get_sequence_stats(seq_path: str, sample_rate: float = 1.0) -> Dict:
    """
    Get statistics for a sequence.

    Args:
        seq_path: Path to sequence folder (e.g., sequences/00)
        sample_rate: Fraction of files to sample (1.0 = all, 0.1 = 10%)

    Returns:
        Dict with total_points, num_scans, points_per_scan stats
    """
    velodyne_path = os.path.join(seq_path, "velodyne")
    if not os.path.exists(velodyne_path):
        return {"total_points": 0, "num_scans": 0, "points_per_scan": []}

    bin_files = sorted([f for f in os.listdir(velodyne_path) if f.endswith(".bin")])

    # Sample files if requested
    if sample_rate < 1.0:
        np.random.seed(42)  # Reproducible sampling
        n_sample = max(1, int(len(bin_files) * sample_rate))
        indices = np.linspace(0, len(bin_files) - 1, n_sample, dtype=int)
        bin_files = [bin_files[i] for i in indices]

    total_points = 0
    points_per_scan = []

    for bin_file in bin_files:
        bin_path = os.path.join(velodyne_path, bin_file)
        n_points = count_points_in_bin(bin_path)
        total_points += n_points
        points_per_scan.append(n_points)

    return {
        "total_points": total_points,
        "num_scans": len(bin_files),
        "points_per_scan": points_per_scan,
        "avg_points_per_scan": np.mean(points_per_scan) if points_per_scan else 0,
        "min_points_per_scan": np.min(points_per_scan) if points_per_scan else 0,
        "max_points_per_scan": np.max(points_per_scan) if points_per_scan else 0,
    }


def verify_subsampling(
    original_path: str,
    subsampled_path: str,
    target_loss: float,
    sequences: List[str] = None,
    quick: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Verify subsampling by comparing original and subsampled point counts.

    Args:
        original_path: Path to original data (e.g., data/SemanticKITTI/original)
        subsampled_path: Path to subsampled data (e.g., data/SemanticKITTI/subsampled/DBSCAN_loss10_seed1)
        target_loss: Expected loss percentage (0-100)
        sequences: List of sequences to check (None = all available)
        quick: If True, sample only 10% of scans for faster verification
        verbose: Print detailed progress

    Returns:
        Dict with verification results
    """
    sample_rate = 0.1 if quick else 1.0

    original_seq_path = os.path.join(original_path, "sequences")
    subsampled_seq_path = os.path.join(subsampled_path, "sequences")

    if not os.path.exists(original_seq_path):
        raise FileNotFoundError(f"Original sequences not found: {original_seq_path}")

    if not os.path.exists(subsampled_seq_path):
        raise FileNotFoundError(f"Subsampled sequences not found: {subsampled_seq_path}")

    # Get available sequences
    available_seqs = sorted([d for d in os.listdir(subsampled_seq_path)
                            if os.path.isdir(os.path.join(subsampled_seq_path, d))])

    if sequences:
        available_seqs = [s for s in sequences if s in available_seqs]

    if not available_seqs:
        raise ValueError("No sequences found to verify")

    if verbose:
        print(f"\nVerifying: {os.path.basename(subsampled_path)}")
        print(f"Target loss: {target_loss}%")
        print(f"Sequences: {available_seqs}")
        print(f"Sample rate: {sample_rate * 100:.0f}%")
        print("-" * 70)

    results = {
        "subsampled_folder": os.path.basename(subsampled_path),
        "target_loss": target_loss,
        "sequences": {},
        "total_original_points": 0,
        "total_subsampled_points": 0,
        "quick_mode": quick,
    }

    for seq in available_seqs:
        orig_seq = os.path.join(original_seq_path, seq)
        sub_seq = os.path.join(subsampled_seq_path, seq)

        if not os.path.exists(orig_seq):
            if verbose:
                print(f"  Seq {seq}: Original not found, skipping")
            continue

        if verbose:
            print(f"  Processing sequence {seq}...", end=" ", flush=True)

        orig_stats = get_sequence_stats(orig_seq, sample_rate)
        sub_stats = get_sequence_stats(sub_seq, sample_rate)

        if orig_stats["total_points"] > 0:
            actual_loss = (1 - sub_stats["total_points"] / orig_stats["total_points"]) * 100
            retention = 100 - actual_loss
        else:
            actual_loss = 0
            retention = 100

        seq_result = {
            "original_points": orig_stats["total_points"],
            "subsampled_points": sub_stats["total_points"],
            "num_scans": sub_stats["num_scans"],
            "actual_loss": actual_loss,
            "retention": retention,
            "target_loss": target_loss,
            "deviation": actual_loss - target_loss,
            "avg_orig_points": orig_stats["avg_points_per_scan"],
            "avg_sub_points": sub_stats["avg_points_per_scan"],
        }

        results["sequences"][seq] = seq_result
        results["total_original_points"] += orig_stats["total_points"]
        results["total_subsampled_points"] += sub_stats["total_points"]

        if verbose:
            status = "OK" if abs(seq_result["deviation"]) < 5 else "WARN"
            print(f"Loss: {actual_loss:.1f}% (target: {target_loss}%) [{status}]")

    # Calculate overall stats
    if results["total_original_points"] > 0:
        results["overall_actual_loss"] = (
            (1 - results["total_subsampled_points"] / results["total_original_points"]) * 100
        )
        results["overall_retention"] = 100 - results["overall_actual_loss"]
        results["overall_deviation"] = results["overall_actual_loss"] - target_loss
    else:
        results["overall_actual_loss"] = 0
        results["overall_retention"] = 100
        results["overall_deviation"] = -target_loss

    return results


def generate_report_text(results: Dict) -> str:
    """Generate a text report from verification results."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"VERIFICATION REPORT: {results['subsampled_folder']}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    lines.append(f"\nTarget Loss:     {results['target_loss']:.1f}%")
    lines.append(f"Actual Loss:     {results['overall_actual_loss']:.1f}%")
    lines.append(f"Deviation:       {results['overall_deviation']:+.1f}%")
    lines.append(f"Retention:       {results['overall_retention']:.1f}%")

    lines.append(f"\nTotal Points:")
    lines.append(f"  Original:      {results['total_original_points']:,}")
    lines.append(f"  Subsampled:    {results['total_subsampled_points']:,}")

    # Per-sequence table
    lines.append(f"\nPer-Sequence Results:")
    lines.append("-" * 70)
    lines.append(f"{'Seq':<6} {'Original':>12} {'Subsampled':>12} {'Loss %':>10} {'Target':>10} {'Status':>10}")
    lines.append("-" * 70)

    for seq, data in sorted(results["sequences"].items()):
        status = "OK" if abs(data["deviation"]) < 5 else "WARN" if abs(data["deviation"]) < 10 else "BAD"
        lines.append(f"{seq:<6} {data['original_points']:>12,} {data['subsampled_points']:>12,} "
                    f"{data['actual_loss']:>9.1f}% {data['target_loss']:>9.1f}% {status:>10}")

    lines.append("-" * 70)

    # Overall assessment
    if abs(results["overall_deviation"]) < 3:
        assessment = "EXCELLENT - Within 3% of target"
    elif abs(results["overall_deviation"]) < 5:
        assessment = "GOOD - Within 5% of target"
    elif abs(results["overall_deviation"]) < 10:
        assessment = "ACCEPTABLE - Within 10% of target"
    else:
        assessment = "WARNING - Deviation > 10% from target"

    lines.append(f"\nAssessment: {assessment}")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_report(results: Dict, report_dir: str) -> str:
    """
    Save verification report to text file.

    Args:
        results: Verification results dict
        report_dir: Directory to save reports

    Returns:
        Path to saved report file
    """
    os.makedirs(report_dir, exist_ok=True)

    folder_name = results['subsampled_folder']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"verify_{folder_name}_{timestamp}.txt"
    report_path = os.path.join(report_dir, report_filename)

    report_text = generate_report_text(results)

    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_path


def print_summary(results: Dict):
    """Print a formatted summary of verification results."""
    print(generate_report_text(results))


def find_subsampled_folders(data_root: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Find all subsampled folders and parse their method/loss/seed.

    Returns:
        List of (folder_path, method, loss, seed) tuples
    """
    subsampled_root = os.path.join(data_root, "subsampled")
    if not os.path.exists(subsampled_root):
        return []

    folders = []
    for folder_name in os.listdir(subsampled_root):
        folder_path = os.path.join(subsampled_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Parse folder name: METHOD_lossXX_seedN or METHOD_lossXX
        parts = folder_name.split("_")

        method = None
        loss = None
        seed = None

        for i, part in enumerate(parts):
            if part.startswith("loss"):
                try:
                    loss = int(part[4:])
                    method = "_".join(parts[:i])
                except ValueError:
                    continue
            elif part.startswith("seed"):
                try:
                    seed = int(part[4:])
                except ValueError:
                    continue

        if method and loss is not None:
            folders.append((folder_path, method, loss, seed))

    return sorted(folders, key=lambda x: (x[1], x[2], x[3] or 0))


def main():
    parser = argparse.ArgumentParser(
        description="Verify subsampling point loss percentages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify specific method/loss
    python verify_subsampling.py --method DBSCAN --loss 10 --seed 1

    # Verify by folder name
    python verify_subsampling.py --folder DBSCAN_loss10_seed1

    # Verify all subsampled datasets
    python verify_subsampling.py --all

    # Quick verification (10% sample)
    python verify_subsampling.py --method IDIS --loss 50 --quick

    # Save results to JSON
    python verify_subsampling.py --all --output results.json
        """
    )

    parser.add_argument("--method", type=str, help="Subsampling method (RS, IDIS, FPS, DBSCAN, Voxel, Poisson)")
    parser.add_argument("--loss", type=int, help="Target loss percentage (10, 30, 50, 70, 90)")
    parser.add_argument("--seed", type=int, help="Random seed (1, 2, 3)")
    parser.add_argument("--folder", type=str, help="Folder name directly (e.g., DBSCAN_loss10_seed1)")
    parser.add_argument("--all", action="store_true", help="Verify all subsampled datasets")
    parser.add_argument("--quick", action="store_true", help="Quick mode: sample 10% of scans")
    parser.add_argument("--sequences", type=str, nargs="+", help="Specific sequences to verify")
    parser.add_argument("--data-root", type=str,
                       default=str(PROJECT_ROOT / "data" / "SemanticKITTI"),
                       help="Path to SemanticKITTI data root")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.folder and not (args.method and args.loss is not None):
        parser.print_help()
        print("\nError: Specify --method and --loss, --folder, or --all")
        sys.exit(1)

    data_root = args.data_root
    original_path = os.path.join(data_root, "original")

    if not os.path.exists(original_path):
        print(f"Error: Original data not found at {original_path}")
        sys.exit(1)

    # Report directory
    report_dir = os.path.join(data_root, "subsampled", "reports")

    all_results = []
    saved_reports = []

    if args.all:
        # Verify all subsampled folders
        folders = find_subsampled_folders(data_root)
        if not folders:
            print("No subsampled folders found")
            sys.exit(1)

        print(f"Found {len(folders)} subsampled datasets to verify")

        for folder_path, method, loss, seed in folders:
            try:
                results = verify_subsampling(
                    original_path=original_path,
                    subsampled_path=folder_path,
                    target_loss=loss,
                    sequences=args.sequences,
                    quick=args.quick,
                    verbose=not args.quiet
                )
                all_results.append(results)

                if not args.quiet:
                    print_summary(results)

                # Save report
                if not args.no_save:
                    report_path = save_report(results, report_dir)
                    saved_reports.append(report_path)
                    if not args.quiet:
                        print(f"Report saved: {report_path}\n")

            except Exception as e:
                print(f"Error verifying {folder_path}: {e}")

    else:
        # Verify single folder
        if args.folder:
            folder_name = args.folder
            # Try to parse loss from folder name
            for part in folder_name.split("_"):
                if part.startswith("loss"):
                    try:
                        target_loss = int(part[4:])
                        break
                    except ValueError:
                        pass
            else:
                target_loss = args.loss or 0
        else:
            # Build folder name from method/loss/seed
            if args.seed:
                folder_name = f"{args.method}_loss{args.loss}_seed{args.seed}"
            else:
                folder_name = f"{args.method}_loss{args.loss}"
            target_loss = args.loss

        subsampled_path = os.path.join(data_root, "subsampled", folder_name)

        if not os.path.exists(subsampled_path):
            # Try with _seed1 suffix for backward compatibility
            subsampled_path_alt = os.path.join(data_root, "subsampled", f"{folder_name}_seed1")
            if os.path.exists(subsampled_path_alt):
                subsampled_path = subsampled_path_alt
                folder_name = f"{folder_name}_seed1"
            else:
                print(f"Error: Subsampled folder not found: {subsampled_path}")
                sys.exit(1)

        results = verify_subsampling(
            original_path=original_path,
            subsampled_path=subsampled_path,
            target_loss=target_loss,
            sequences=args.sequences,
            quick=args.quick,
            verbose=not args.quiet
        )
        all_results.append(results)

        if not args.quiet:
            print_summary(results)

        # Save report
        if not args.no_save:
            report_path = save_report(results, report_dir)
            saved_reports.append(report_path)
            print(f"\nReport saved: {report_path}")

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "data_root": data_root,
            "quick_mode": args.quick,
            "results": all_results
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Print summary table for --all
    if args.all and len(all_results) > 1:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"{'Folder':<35} {'Target':>8} {'Actual':>8} {'Deviation':>10} {'Status':>10}")
        print("-" * 80)

        for r in all_results:
            status = "OK" if abs(r["overall_deviation"]) < 5 else "WARN"
            print(f"{r['subsampled_folder']:<35} {r['target_loss']:>7.0f}% "
                  f"{r['overall_actual_loss']:>7.1f}% {r['overall_deviation']:>+9.1f}% {status:>10}")

        print("=" * 80)

    # Print saved reports summary
    if saved_reports:
        print(f"\nReports saved to: {report_dir}")
        print(f"Total reports: {len(saved_reports)}")


if __name__ == "__main__":
    main()
