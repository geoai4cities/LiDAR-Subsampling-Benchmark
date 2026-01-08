#!/usr/bin/env python3
"""
Extract training metrics from PTv3 training logs and generate tables.

This script parses training logs to extract:
- Per-epoch training loss
- Per-epoch validation mIoU, mAcc, allAcc
- GPU memory usage
- Epoch training time
- Total training time
- Best mIoU achieved

Usage:
    # Process specific log files
    python extract_training_metrics.py /path/to/train.log /path/to/train2.log

    # Process all logs in a directory
    python extract_training_metrics.py --dir /path/to/outputs

    # Auto-discover all experiments (default behavior)
    python extract_training_metrics.py --auto

    # Specify output directory
    python extract_training_metrics.py --output /path/to/tables /path/to/train.log

Output: Text files with formatted tables for each experiment.
"""

import argparse
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from glob import glob


def parse_epoch_summary(line: str) -> Optional[Dict]:
    """Parse epoch summary line to extract time and GPU memory."""
    pattern = r'Epoch (\d+)/(\d+) Summary: Time: (\d+:\d+:\d+) \| GPU Peak: ([\d.]+) GB \(alloc\) / ([\d.]+) GB \(reserved\)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'total_epochs': int(match.group(2)),
            'time': match.group(3),
            'gpu_alloc': float(match.group(4)),
            'gpu_reserved': float(match.group(5))
        }
    return None


def parse_val_result(line: str) -> Optional[Dict]:
    """Parse validation result line."""
    # Pattern: Val result: mIoU/mAcc/allAcc 0.6187/0.7025/0.9058.
    pattern = r'Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)\.?'
    match = re.search(pattern, line)
    if match:
        # Remove trailing period if present
        miou = match.group(1).rstrip('.')
        macc = match.group(2).rstrip('.')
        allacc = match.group(3).rstrip('.')
        return {
            'mIoU': float(miou),
            'mAcc': float(macc),
            'allAcc': float(allacc)
        }
    return None


def parse_train_loss(line: str) -> Optional[Dict]:
    """Parse final training iteration to get loss."""
    # Pattern: Train: [1/10][9565/9565] ... loss: 0.7329 Lr: 0.00198
    pattern = r'Train: \[(\d+)/\d+\]\[\d+/\d+\].*loss: ([\d.]+) Lr: ([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'lr': float(match.group(3))
        }
    return None


def parse_total_training_time(line: str) -> Optional[str]:
    """Parse total training time."""
    pattern = r'Total Training Time:\s+([\d:]+)'
    match = re.search(pattern, line)
    if match:
        return match.group(1)
    return None


def parse_best_miou(line: str) -> Optional[float]:
    """Parse best mIoU update."""
    pattern = r'Best validation mIoU updated to: ([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return None


def parse_test_result(line: str) -> Optional[Dict]:
    """Parse test result (final evaluation)."""
    # Pattern from test.py: Val result: mIoU/mAcc/allAcc 0.6721/0.7308/0.9253
    if 'test.py' in line:
        pattern = r'Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)'
        match = re.search(pattern, line)
        if match:
            return {
                'mIoU': float(match.group(1)),
                'mAcc': float(match.group(2)),
                'allAcc': float(match.group(3))
            }
    return None


def parse_class_result(line: str, is_test: bool = False) -> Optional[Dict]:
    """Parse class-wise IoU/accuracy result.

    Two formats:
    - Validation: Class_0-car Result: iou/accuracy 0.9494/0.9814
    - Test: Class_0 - car Result: iou/accuracy 0.9660/0.9925
    """
    if is_test and 'test.py' in line:
        # Test format: Class_0 - car Result: iou/accuracy 0.9660/0.9925
        pattern = r'Class_(\d+)\s*-\s*(\S+)\s+Result:\s*iou/accuracy\s+([\d.]+)/([\d.]+)'
    else:
        # Validation format: Class_0-car Result: iou/accuracy 0.9494/0.9814
        pattern = r'Class_(\d+)-(\S+)\s+Result:\s*iou/accuracy\s+([\d.]+)/([\d.]+)'

    match = re.search(pattern, line)
    if match:
        return {
            'class_id': int(match.group(1)),
            'class_name': match.group(2),
            'iou': float(match.group(3)),
            'accuracy': float(match.group(4))
        }
    return None


# SemanticKITTI class names (19 classes)
SEMANTICKITTI_CLASSES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]


def extract_experiment_name(log_path: str) -> str:
    """Extract experiment name from log path."""
    # Path format: .../outputs/baseline_loss0_seed1_140gb/train.log
    parts = Path(log_path).parts
    for i, part in enumerate(parts):
        if part == 'outputs' and i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.dirname(log_path))


def parse_log_file(log_path: str) -> Dict:
    """Parse a training log file and extract all metrics."""
    metrics = {
        'experiment_name': extract_experiment_name(log_path),
        'log_path': log_path,
        'epochs': [],
        'total_training_time': None,
        'best_miou': None,
        'test_result': None,
        'best_class_results': {},  # Class-wise results at best epoch
        'test_class_results': {},  # Class-wise results from test.py
    }

    current_epoch_data = {}
    current_class_results = {}  # Temporary storage for class results
    in_test_section = False

    with open(log_path, 'r') as f:
        for line in f:
            # Check if we're in test section
            if 'test.py' in line:
                in_test_section = True

            # Check for epoch summary
            epoch_summary = parse_epoch_summary(line)
            if epoch_summary:
                current_epoch_data['epoch'] = epoch_summary['epoch']
                current_epoch_data['time'] = epoch_summary['time']
                current_epoch_data['gpu_alloc'] = epoch_summary['gpu_alloc']
                current_epoch_data['gpu_reserved'] = epoch_summary['gpu_reserved']

            # Check for validation result (follows epoch summary)
            val_result = parse_val_result(line)
            if val_result and 'test.py' not in line:
                current_epoch_data['val_mIoU'] = val_result['mIoU']
                current_epoch_data['val_mAcc'] = val_result['mAcc']
                current_epoch_data['val_allAcc'] = val_result['allAcc']

                # Save completed epoch data
                if 'epoch' in current_epoch_data:
                    metrics['epochs'].append(current_epoch_data.copy())
                    current_epoch_data = {}

            # Check for training loss at end of epoch
            train_loss = parse_train_loss(line)
            if train_loss:
                # Update the corresponding epoch
                epoch_num = train_loss['epoch']
                for ep in metrics['epochs']:
                    if ep.get('epoch') == epoch_num:
                        ep['train_loss'] = train_loss['loss']
                        ep['lr'] = train_loss['lr']
                        break
                else:
                    # Epoch not yet in list, store for later
                    current_epoch_data['train_loss'] = train_loss['loss']
                    current_epoch_data['lr'] = train_loss['lr']

            # Check for class-wise results
            class_result = parse_class_result(line, is_test=in_test_section)
            if class_result:
                class_id = class_result['class_id']
                if in_test_section:
                    metrics['test_class_results'][class_id] = class_result
                else:
                    current_class_results[class_id] = class_result

            # Check for best mIoU update - save the class results before this
            best_miou = parse_best_miou(line)
            if best_miou:
                metrics['best_miou'] = best_miou
                # Save the class results collected before this best update
                if current_class_results:
                    metrics['best_class_results'] = current_class_results.copy()
                current_class_results = {}

            # Check for total training time
            total_time = parse_total_training_time(line)
            if total_time:
                metrics['total_training_time'] = total_time

            # Check for test result
            test_result = parse_test_result(line)
            if test_result:
                metrics['test_result'] = test_result

    return metrics


def format_class_table(class_results: Dict, title: str = "Class-wise Performance") -> List[str]:
    """Format class-wise results as a table."""
    lines = []
    if not class_results:
        return lines

    lines.append("")
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)
    header = f"{'Class':<20} | {'IoU':^10} | {'Accuracy':^10}"
    lines.append(header)
    lines.append("-" * 80)

    for class_id in sorted(class_results.keys()):
        result = class_results[class_id]
        class_name = result.get('class_name', SEMANTICKITTI_CLASSES[class_id] if class_id < len(SEMANTICKITTI_CLASSES) else f'class_{class_id}')
        iou = result.get('iou', 0)
        acc = result.get('accuracy', 0)
        row = f"{class_name:<20} | {iou:^10.4f} | {acc:^10.4f}"
        lines.append(row)

    lines.append("-" * 80)
    return lines


def parse_experiment_info(exp_name: str) -> Dict:
    """Parse experiment name to extract method, loss level, seed, and radius.

    Naming conventions:
    - Deterministic: {METHOD}_loss{XX}_140gb (e.g., DBSCAN_loss90_140gb, Voxel_loss90_140gb)
    - Non-deterministic: {METHOD}_loss{XX}_seed{N}_140gb (e.g., RS_loss90_seed1_140gb, FPS_loss90_seed1_140gb)
    - IDIS variants: IDIS_loss{XX}_140gb (R=10 default), IDIS_R5_loss{XX}_140gb, IDIS_R15_loss{XX}_140gb
    - Baseline: baseline_loss0_seed1_140gb

    Method types:
    - Non-deterministic (requires seed): RS, FPS, Poisson
    - Deterministic (no seed): IDIS, DBSCAN, Voxel
    - IDIS has radius parameter (R=5, 10, 15)
    """
    info = {
        'method': 'Unknown',
        'loss_level': 'Unknown',
        'seed': None,  # None means deterministic or not applicable
        'radius': None,  # For IDIS: 5, 10, or 15 (None for other methods)
    }

    # Methods that require seeds (non-deterministic)
    NON_DETERMINISTIC_METHODS = ['RS', 'FPS', 'Poisson']

    parts = exp_name.split('_')

    # Extract method (first part)
    if parts:
        info['method'] = parts[0]

    # Extract loss level, seed, and radius
    for i, part in enumerate(parts):
        if part.startswith('loss'):
            loss_num = part.replace('loss', '')
            info['loss_level'] = f"{loss_num}%"
        elif part.startswith('seed'):
            seed_num = part.replace('seed', '')
            info['seed'] = seed_num
        elif part.startswith('R') and part[1:].isdigit():
            # IDIS radius: R5, R10, R15
            radius_num = part.replace('R', '')
            info['radius'] = radius_num

    # For IDIS without explicit radius, default is R=10
    if info['method'] == 'IDIS' and info['radius'] is None:
        info['radius'] = '10'

    # Mark seed as required but missing for non-deterministic methods without seed
    # (helps identify if seed info is missing from experiment name)
    if info['method'] in NON_DETERMINISTIC_METHODS and info['seed'] is None:
        # Check if this might be an old naming convention
        pass  # Leave as None, will show '-' in tables

    return info


def format_table(metrics: Dict) -> str:
    """Format metrics as a readable table."""
    lines = []
    exp_name = metrics['experiment_name']

    # Header
    lines.append("=" * 100)
    lines.append(f"EXPERIMENT: {exp_name}")
    lines.append("=" * 100)
    lines.append("")

    # Parse experiment name for details
    exp_info = parse_experiment_info(exp_name)
    method = exp_info['method']
    loss_level = exp_info['loss_level']
    seed = exp_info['seed']
    radius = exp_info['radius']

    lines.append(f"Method: {method}")
    lines.append(f"Loss Level: {loss_level}")
    if seed is not None:
        lines.append(f"Seed: {seed}")
    if radius is not None:
        lines.append(f"Radius: {radius}m")
    if metrics['total_training_time']:
        lines.append(f"Total Training Time: {metrics['total_training_time']}")
    if metrics['best_miou']:
        lines.append(f"Best Validation mIoU: {metrics['best_miou']:.4f}")
    if metrics['test_result']:
        lines.append(f"Test mIoU: {metrics['test_result']['mIoU']:.4f}")
        lines.append(f"Test mAcc: {metrics['test_result']['mAcc']:.4f}")
        lines.append(f"Test allAcc: {metrics['test_result']['allAcc']:.4f}")
    lines.append("")

    # Epoch-by-epoch table
    lines.append("-" * 100)
    header = f"{'Epoch':^6} | {'Train Loss':^12} | {'Val mIoU':^10} | {'Val mAcc':^10} | {'Val allAcc':^10} | {'GPU (GB)':^12} | {'Time':^10}"
    lines.append(header)
    lines.append("-" * 100)

    for ep in metrics['epochs']:
        epoch_num = ep.get('epoch', '-')
        train_loss = f"{ep.get('train_loss', 0):.4f}" if 'train_loss' in ep else '-'
        val_miou = f"{ep.get('val_mIoU', 0):.4f}" if 'val_mIoU' in ep else '-'
        val_macc = f"{ep.get('val_mAcc', 0):.4f}" if 'val_mAcc' in ep else '-'
        val_allacc = f"{ep.get('val_allAcc', 0):.4f}" if 'val_allAcc' in ep else '-'
        gpu_mem = f"{ep.get('gpu_alloc', 0):.2f}" if 'gpu_alloc' in ep else '-'
        time_str = ep.get('time', '-')

        row = f"{epoch_num:^6} | {train_loss:^12} | {val_miou:^10} | {val_macc:^10} | {val_allacc:^10} | {gpu_mem:^12} | {time_str:^10}"
        lines.append(row)

    lines.append("-" * 100)

    # Add Final Results Summary Table
    lines.append("")
    lines.append("=" * 80)
    lines.append("FINAL RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Summary metrics table
    lines.append("-" * 60)
    lines.append(f"{'Metric':<25} | {'Best Val':^15} | {'Test':^15}")
    lines.append("-" * 60)

    best_miou = metrics.get('best_miou', 0) or 0
    test_result = metrics.get('test_result', {})
    test_miou = test_result.get('mIoU', 0) if test_result else 0
    test_macc = test_result.get('mAcc', 0) if test_result else 0
    test_allacc = test_result.get('allAcc', 0) if test_result else 0

    # Find best epoch metrics
    best_epoch_data = None
    for ep in metrics['epochs']:
        if ep.get('val_mIoU') == best_miou:
            best_epoch_data = ep
            break

    best_macc = best_epoch_data.get('val_mAcc', 0) if best_epoch_data else 0
    best_allacc = best_epoch_data.get('val_allAcc', 0) if best_epoch_data else 0

    lines.append(f"{'mIoU':<25} | {best_miou:^15.4f} | {test_miou:^15.4f}" if test_miou else f"{'mIoU':<25} | {best_miou:^15.4f} | {'-':^15}")
    lines.append(f"{'mAcc':<25} | {best_macc:^15.4f} | {test_macc:^15.4f}" if test_macc else f"{'mAcc':<25} | {best_macc:^15.4f} | {'-':^15}")
    lines.append(f"{'allAcc':<25} | {best_allacc:^15.4f} | {test_allacc:^15.4f}" if test_allacc else f"{'allAcc':<25} | {best_allacc:^15.4f} | {'-':^15}")
    lines.append("-" * 60)

    # Add class-wise performance tables
    test_class_results = metrics.get('test_class_results', {})
    best_class_results = metrics.get('best_class_results', {})

    if test_class_results:
        lines.extend(format_class_table(test_class_results, "CLASS-WISE PERFORMANCE (Test - Best Model)"))
    elif best_class_results:
        lines.extend(format_class_table(best_class_results, "CLASS-WISE PERFORMANCE (Best Validation Epoch)"))

    lines.append("")

    return "\n".join(lines)


def process_experiment(log_path: str, output_dir: str) -> Tuple[Dict, str]:
    """Process a single experiment log file."""
    print(f"Processing: {log_path}")

    metrics = parse_log_file(log_path)
    table_str = format_table(metrics)

    # Save individual table
    exp_name = metrics['experiment_name']
    output_file = os.path.join(output_dir, f"{exp_name}_metrics.txt")
    with open(output_file, 'w') as f:
        f.write(table_str)

    print(f"  Saved to: {output_file}")
    return metrics, table_str


def get_seed_or_radius(exp_info: Dict) -> str:
    """Get the appropriate parameter value (Seed for RS/FPS/Poisson, R for IDIS, - for others)."""
    method = exp_info['method']

    # Non-deterministic methods show seed
    if method in ['RS', 'FPS', 'Poisson']:
        return f"S{exp_info['seed']}" if exp_info['seed'] else '-'
    # IDIS shows radius
    elif method == 'IDIS':
        return f"R{exp_info['radius']}" if exp_info['radius'] else 'R10'
    # Other deterministic methods (DBSCAN, Voxel)
    else:
        return '-'


def create_summary_table(all_metrics: List[Dict], output_dir: str):
    """Create a summary table comparing all experiments."""
    lines = []
    lines.append("=" * 150)
    lines.append("SUMMARY: All Experiments Comparison")
    lines.append("=" * 150)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Note: Seed/R column shows Seed (S1,S2,S3) for RS/FPS/Poisson, Radius (R5,R10,R15) for IDIS, - for deterministic methods")
    lines.append("")

    # Summary header with Seed/R column
    lines.append("-" * 150)
    header = f"{'Experiment':<35} | {'Method':^10} | {'Loss':^6} | {'Seed/R':^7} | {'Best mIoU':^10} | {'Test mIoU':^10} | {'GPU Peak':^10} | {'Total Time':^12}"
    lines.append(header)
    lines.append("-" * 150)

    for metrics in sorted(all_metrics, key=lambda x: x.get('best_miou', 0) or 0, reverse=True):
        exp_name = metrics['experiment_name'][:35]
        exp_info = parse_experiment_info(metrics['experiment_name'])

        method = exp_info['method'][:10]
        loss = exp_info['loss_level']
        seed_or_r = get_seed_or_radius(exp_info)

        best_miou = f"{metrics.get('best_miou', 0):.4f}" if metrics.get('best_miou') else '-'
        test_miou = f"{metrics['test_result']['mIoU']:.4f}" if metrics.get('test_result') else '-'

        # Get max GPU usage across epochs
        max_gpu = max([ep.get('gpu_alloc', 0) for ep in metrics['epochs']], default=0)
        gpu_str = f"{max_gpu:.1f} GB" if max_gpu > 0 else '-'

        total_time = metrics.get('total_training_time') or '-'

        row = f"{exp_name:<35} | {method:^10} | {loss:^6} | {seed_or_r:^7} | {best_miou:^10} | {test_miou:^10} | {gpu_str:^10} | {total_time:^12}"
        lines.append(row)

    lines.append("-" * 150)
    lines.append("")

    # Add per-method comparison at 90% loss
    lines.append("")
    lines.append("=" * 110)
    lines.append("METHOD COMPARISON (90% Loss Level)")
    lines.append("=" * 110)
    lines.append("")

    loss90_experiments = [m for m in all_metrics if 'loss90' in m['experiment_name']]
    baseline = [m for m in all_metrics if 'loss0' in m['experiment_name']]

    base_miou = None
    if baseline:
        base_miou = baseline[0].get('test_result', {}).get('mIoU', baseline[0].get('best_miou', 0))
        if base_miou:
            lines.append(f"Baseline (0% loss) mIoU: {base_miou:.4f}")
            lines.append("")

    lines.append("-" * 110)
    header2 = f"{'Method':<15} | {'Seed/R':^7} | {'Test mIoU':^12} | {'Drop vs Base':^18} | {'Training Time':^15}"
    lines.append(header2)
    lines.append("-" * 110)

    def get_test_miou(m):
        tr = m.get('test_result')
        if tr and isinstance(tr, dict):
            return tr.get('mIoU', 0) or 0
        return m.get('best_miou', 0) or 0

    for metrics in sorted(loss90_experiments, key=get_test_miou, reverse=True):
        exp_name = metrics['experiment_name']
        exp_info = parse_experiment_info(exp_name)

        method = exp_info['method']
        seed_or_r = get_seed_or_radius(exp_info)

        tr = metrics.get('test_result')
        if tr and isinstance(tr, dict):
            test_miou = tr.get('mIoU', metrics.get('best_miou', 0))
        else:
            test_miou = metrics.get('best_miou', 0)
        test_miou_str = f"{test_miou:.4f}" if test_miou else '-'

        if base_miou and test_miou:
            drop = base_miou - test_miou
            drop_pct = (drop / base_miou) * 100
            drop_str = f"-{drop:.4f} ({drop_pct:.1f}%)"
        else:
            drop_str = '-'

        total_time = metrics.get('total_training_time') or '-'

        row = f"{method:<15} | {seed_or_r:^7} | {test_miou_str:^12} | {drop_str:^18} | {total_time:^15}"
        lines.append(row)

    lines.append("-" * 110)

    # Save summary
    summary_file = os.path.join(output_dir, "summary_all_experiments.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved to: {summary_file}")
    return "\n".join(lines)


def find_log_files(directory: str) -> List[str]:
    """Find all train.log files in a directory."""
    log_files = []

    # Check for direct train.log files in subdirectories
    pattern = os.path.join(directory, "*", "train.log")
    log_files.extend(glob(pattern))

    # Also check for train.log directly in the directory
    direct_log = os.path.join(directory, "train.log")
    if os.path.exists(direct_log):
        log_files.append(direct_log)

    return sorted(log_files)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract training metrics from PTv3 training logs and generate tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific log files
    python extract_training_metrics.py /path/to/train.log /path/to/train2.log

    # Process all logs in a directory
    python extract_training_metrics.py --dir /path/to/outputs

    # Auto-discover all experiments (default behavior)
    python extract_training_metrics.py --auto

    # Specify output directory
    python extract_training_metrics.py --output /path/to/tables /path/to/train.log

    # Only generate summary (no individual tables)
    python extract_training_metrics.py --summary-only --dir /path/to/outputs
        """
    )

    parser.add_argument(
        'log_files',
        nargs='*',
        help='Path(s) to train.log file(s) to process'
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory containing experiment folders with train.log files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for generated tables (default: docs/tables)'
    )

    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Auto-discover experiments from default location'
    )

    parser.add_argument(
        '--summary-only', '-s',
        action='store_true',
        help='Only generate summary table, not individual experiment tables'
    )

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip generating the summary table'
    )

    parser.add_argument(
        '--print', '-p',
        action='store_true',
        help='Print tables to stdout in addition to saving files'
    )

    return parser.parse_args()


def main():
    """Main function to process all experiments."""
    args = parse_args()

    # Determine base directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent of scripts/

    # Determine output directory
    if args.output:
        tables_dir = args.output
    else:
        tables_dir = os.path.join(base_dir, "docs/tables")

    # Create output directory
    os.makedirs(tables_dir, exist_ok=True)

    # Collect log files to process
    log_files = []

    if args.log_files:
        # Use provided log files
        for log_file in args.log_files:
            if os.path.isfile(log_file):
                log_files.append(log_file)
            elif os.path.isdir(log_file):
                # If a directory is provided, look for train.log in it
                train_log = os.path.join(log_file, "train.log")
                if os.path.exists(train_log):
                    log_files.append(train_log)
                else:
                    print(f"Warning: No train.log found in {log_file}")
            else:
                print(f"Warning: File not found: {log_file}")

    if args.dir:
        # Find all log files in the directory
        found_logs = find_log_files(args.dir)
        log_files.extend(found_logs)
        if not found_logs:
            print(f"Warning: No train.log files found in {args.dir}")

    if args.auto or (not args.log_files and not args.dir):
        # Auto-discover from default location
        outputs_dir = os.path.join(base_dir, "PTv3/SemanticKITTI/outputs")
        if os.path.exists(outputs_dir):
            found_logs = find_log_files(outputs_dir)
            log_files.extend(found_logs)
        else:
            print(f"Warning: Default outputs directory not found: {outputs_dir}")

    # Remove duplicates while preserving order
    seen = set()
    unique_log_files = []
    for lf in log_files:
        abs_path = os.path.abspath(lf)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_log_files.append(lf)
    log_files = unique_log_files

    if not log_files:
        print("Error: No log files to process.")
        print("Use --help for usage information.")
        return

    all_metrics = []
    all_tables = []

    print("=" * 60)
    print("Extracting Training Metrics from PTv3 Logs")
    print("=" * 60)
    print(f"Output directory: {tables_dir}")
    print(f"Log files to process: {len(log_files)}")
    print("")

    for log_path in log_files:
        if os.path.exists(log_path):
            metrics = parse_log_file(log_path)
            table_str = format_table(metrics)

            if not args.summary_only:
                # Save individual table
                exp_name = metrics['experiment_name']
                output_file = os.path.join(tables_dir, f"{exp_name}_metrics.txt")
                with open(output_file, 'w') as f:
                    f.write(table_str)
                print(f"Processing: {log_path}")
                print(f"  Saved to: {output_file}")
            else:
                print(f"Processing: {log_path}")

            if args.print:
                print(table_str)

            all_metrics.append(metrics)
            all_tables.append(table_str)
        else:
            print(f"Warning: Log file not found: {log_path}")

    # Create summary
    if all_metrics and not args.no_summary:
        create_summary_table(all_metrics, tables_dir)

        # Also create combined file with all tables
        if not args.summary_only:
            combined_file = os.path.join(tables_dir, "all_experiments_detailed.txt")
            with open(combined_file, 'w') as f:
                f.write("=" * 100)
                f.write("\n")
                f.write("DETAILED METRICS FOR ALL EXPERIMENTS")
                f.write("\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                f.write("\n")
                f.write("=" * 100)
                f.write("\n\n")
                for table in all_tables:
                    f.write(table)
                    f.write("\n")

            print(f"\nCombined detailed metrics saved to: {combined_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
