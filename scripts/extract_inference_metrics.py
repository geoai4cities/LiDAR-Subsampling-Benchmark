#!/usr/bin/env python3
"""
Extract inference metrics from PTv3 inference logs and generate tables.

This script parses inference logs to extract:
- mIoU, mAcc, allAcc results
- Class-wise IoU and accuracy
- Inference timing (total time, time per scan)
- GPU memory usage

Two types of inference:
1. inference/: Baseline model evaluated on subsampled data
2. inference_on_original/: Trained models evaluated on original data

Usage:
    # Auto-discover all inference experiments (default behavior)
    python extract_inference_metrics.py --auto

    # Process specific type
    python extract_inference_metrics.py --type inference
    python extract_inference_metrics.py --type inference_on_original

    # Specify output directory
    python extract_inference_metrics.py --output /path/to/tables

Output: Text files with formatted tables for each inference experiment.
"""

import argparse
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from glob import glob


# SemanticKITTI class names (19 classes)
SEMANTICKITTI_CLASSES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]


def parse_val_result(line: str) -> Optional[Dict]:
    """Parse validation result line."""
    # Pattern: Val result: mIoU/mAcc/allAcc 0.6720/0.7308/0.9253
    pattern = r'Val result: mIoU/mAcc/allAcc ([\d.]+)/([\d.]+)/([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'mIoU': float(match.group(1)),
            'mAcc': float(match.group(2)),
            'allAcc': float(match.group(3))
        }
    return None


def parse_class_result(line: str) -> Optional[Dict]:
    """Parse class-wise IoU/accuracy result.

    Format: Class_0 - car Result: iou/accuracy 0.9660/0.9925
    """
    pattern = r'Class_(\d+)\s*-\s*(\S+)\s+Result:\s*iou/accuracy\s+([\d.]+)/([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'class_id': int(match.group(1)),
            'class_name': match.group(2),
            'iou': float(match.group(3)),
            'accuracy': float(match.group(4))
        }
    return None


def parse_inference_metrics_file(metrics_path: str) -> Dict:
    """Parse inference_metrics.txt file."""
    metrics = {
        'method': None,
        'loss_level': None,
        'seed': None,
        'data_path': None,
        'model_path': None,
        'start_time': None,
        'end_time': None,
        'total_time': None,
        'total_seconds': None,
        'gpu_id': None,
        'initial_memory': None,
        'peak_memory': None,
    }

    if not os.path.exists(metrics_path):
        return metrics

    with open(metrics_path, 'r') as f:
        content = f.read()

    # Parse method
    match = re.search(r'Method:\s*(\S+)', content)
    if match:
        metrics['method'] = match.group(1)

    # Parse loss level
    match = re.search(r'Loss Level:\s*(\d+)%', content)
    if match:
        metrics['loss_level'] = match.group(1)

    # Parse seed
    match = re.search(r'Seed:\s*(\d+|N/A)', content)
    if match:
        seed = match.group(1)
        metrics['seed'] = None if seed == 'N/A' or 'deterministic' in content else seed

    # Parse data path
    match = re.search(r'Data Path:\s*(.+)', content)
    if match:
        metrics['data_path'] = match.group(1).strip()

    # Parse model path
    match = re.search(r'Model Path:\s*(.+)', content)
    if match:
        metrics['model_path'] = match.group(1).strip()
    elif re.search(r'Baseline Model:\s*(.+)', content):
        match = re.search(r'Baseline Model:\s*(.+)', content)
        metrics['model_path'] = match.group(1).strip()

    # Parse timing
    match = re.search(r'Start Time:\s*(.+)', content)
    if match:
        metrics['start_time'] = match.group(1).strip()

    match = re.search(r'End Time:\s*(.+)', content)
    if match:
        metrics['end_time'] = match.group(1).strip()

    match = re.search(r'Total Time:\s*(\S+)\s*\((\d+)\s*seconds\)', content)
    if match:
        metrics['total_time'] = match.group(1)
        metrics['total_seconds'] = int(match.group(2))

    # Parse GPU memory
    match = re.search(r'GPU ID:\s*(\d+)', content)
    if match:
        metrics['gpu_id'] = match.group(1)

    match = re.search(r'Initial Memory:\s*(\d+)\s*MB', content)
    if match:
        metrics['initial_memory'] = int(match.group(1))

    match = re.search(r'Peak Memory:\s*(\d+)\s*MB', content)
    if match:
        metrics['peak_memory'] = int(match.group(1))

    return metrics


def parse_test_log(log_path: str) -> Dict:
    """Parse test.log file to extract mIoU, mAcc, allAcc and class-wise results."""
    results = {
        'mIoU': None,
        'mAcc': None,
        'allAcc': None,
        'class_results': {},
        'num_scans': None,
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, 'r') as f:
        for line in f:
            # Parse overall results
            val_result = parse_val_result(line)
            if val_result:
                results['mIoU'] = val_result['mIoU']
                results['mAcc'] = val_result['mAcc']
                results['allAcc'] = val_result['allAcc']

            # Parse class-wise results
            class_result = parse_class_result(line)
            if class_result:
                results['class_results'][class_result['class_id']] = class_result

            # Parse number of scans
            # Pattern: Test: 4071/4071-08_004070 [4071/4071]
            match = re.search(r'Test: (\d+)/(\d+)-', line)
            if match:
                results['num_scans'] = int(match.group(2))

    return results


def parse_inference_log_gpu_memory(log_path: str) -> Dict:
    """Parse inference.log file to extract GPU memory and timing information.

    The inference.log contains at the end:
    ========================================================================
    INFERENCE COMPLETE: 2025-12-24 05:15:57
    ========================================================================
    Total Time:     2240m 49s (134449 seconds)
    Initial GPU:    14888 MB
    Peak GPU:       5251 MB
    ========================================================================
    """
    results = {
        'initial_memory': None,
        'peak_memory': None,
        'total_time': None,
        'total_seconds': None,
    }

    if not os.path.exists(log_path):
        return results

    # Read last 50 lines of the file (where the summary is)
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # Check last 50 lines for the summary section
            for line in lines[-50:]:
                # Parse Initial GPU
                match = re.search(r'Initial GPU:\s+(\d+)\s*MB', line)
                if match:
                    results['initial_memory'] = int(match.group(1))

                # Parse Peak GPU
                match = re.search(r'Peak GPU:\s+(\d+)\s*MB', line)
                if match:
                    results['peak_memory'] = int(match.group(1))

                # Parse Total Time
                match = re.search(r'Total Time:\s+(\d+m\s*\d+s)\s*\((\d+)\s*seconds\)', line)
                if match:
                    results['total_time'] = match.group(1)
                    results['total_seconds'] = int(match.group(2))
    except Exception as e:
        print(f"Warning: Could not parse {log_path}: {e}")

    return results


def extract_experiment_name(exp_path: str) -> str:
    """Extract experiment name from path."""
    return os.path.basename(exp_path)


def parse_experiment_info(exp_name: str) -> Dict:
    """Parse experiment name to extract method, loss level, seed, and radius."""
    info = {
        'method': 'Unknown',
        'loss_level': 'Unknown',
        'seed': None,
        'radius': None,
    }

    parts = exp_name.split('_')

    # Extract method (first part)
    if parts:
        info['method'] = parts[0]

    # Handle IDIS_R variants
    if len(parts) > 1 and parts[1].startswith('R') and parts[1][1:].isdigit():
        info['method'] = f"{parts[0]}_{parts[1]}"
        info['radius'] = parts[1].replace('R', '')

    # Extract loss level and seed
    for part in parts:
        if part.startswith('loss'):
            loss_num = part.replace('loss', '')
            info['loss_level'] = f"{loss_num}%"
        elif part.startswith('seed'):
            seed_num = part.replace('seed', '')
            info['seed'] = seed_num
        elif part.startswith('R') and part[1:].isdigit() and info['radius'] is None:
            info['radius'] = part.replace('R', '')

    # For IDIS without explicit radius, default is R=10
    if info['method'] == 'IDIS' and info['radius'] is None:
        info['radius'] = '10'

    return info


def format_class_table(class_results: Dict, title: str = "Class-wise Performance") -> List[str]:
    """Format class-wise results as a table."""
    lines = []
    if not class_results:
        return lines

    lines.append("")
    lines.append("-" * 60)
    lines.append(title)
    lines.append("-" * 60)
    header = f"{'Class':<20} | {'IoU':^10} | {'Accuracy':^10}"
    lines.append(header)
    lines.append("-" * 60)

    for class_id in sorted(class_results.keys()):
        result = class_results[class_id]
        class_name = result.get('class_name', SEMANTICKITTI_CLASSES[class_id] if class_id < len(SEMANTICKITTI_CLASSES) else f'class_{class_id}')
        iou = result.get('iou', 0)
        acc = result.get('accuracy', 0)
        row = f"{class_name:<20} | {iou:^10.4f} | {acc:^10.4f}"
        lines.append(row)

    # Add mean IoU
    if class_results:
        mean_iou = sum(r['iou'] for r in class_results.values()) / len(class_results)
        mean_acc = sum(r['accuracy'] for r in class_results.values()) / len(class_results)
        lines.append("-" * 60)
        lines.append(f"{'Mean':<20} | {mean_iou:^10.4f} | {mean_acc:^10.4f}")

    lines.append("-" * 60)
    return lines


def process_inference_experiment(exp_dir: str, inference_type: str) -> Dict:
    """Process a single inference experiment directory."""
    exp_name = extract_experiment_name(exp_dir)

    # Parse inference_metrics.txt
    metrics_path = os.path.join(exp_dir, 'inference_metrics.txt')
    metrics = parse_inference_metrics_file(metrics_path)

    # Parse test.log
    test_log_path = os.path.join(exp_dir, 'test.log')
    test_results = parse_test_log(test_log_path)

    # Parse inference.log for GPU memory (more reliable source)
    inference_log_path = os.path.join(exp_dir, 'inference.log')
    inference_log_results = parse_inference_log_gpu_memory(inference_log_path)

    # Combine results
    result = {
        'experiment_name': exp_name,
        'inference_type': inference_type,
        'exp_dir': exp_dir,
        **metrics,
        **test_results,
    }

    # Override with inference.log values if available (more reliable)
    if inference_log_results.get('initial_memory') is not None:
        result['initial_memory'] = inference_log_results['initial_memory']
    if inference_log_results.get('peak_memory') is not None:
        result['peak_memory'] = inference_log_results['peak_memory']
    if inference_log_results.get('total_time') is not None:
        result['total_time'] = inference_log_results['total_time']
    if inference_log_results.get('total_seconds') is not None:
        result['total_seconds'] = inference_log_results['total_seconds']

    # Parse experiment info from name if not in metrics
    exp_info = parse_experiment_info(exp_name)
    if result['method'] is None:
        result['method'] = exp_info['method']
    if result['loss_level'] is None:
        result['loss_level'] = exp_info['loss_level']
    if result['seed'] is None and exp_info['seed']:
        result['seed'] = exp_info['seed']
    result['radius'] = exp_info.get('radius')

    return result


def format_inference_table(result: Dict) -> str:
    """Format inference result as a readable table."""
    lines = []
    exp_name = result['experiment_name']
    inference_type = result['inference_type']

    # Header
    lines.append("=" * 100)
    if inference_type == 'inference':
        lines.append(f"INFERENCE: Baseline Model on Subsampled Data - {exp_name}")
    else:
        lines.append(f"INFERENCE: Trained Model on Original Data - {exp_name}")
    lines.append("=" * 100)
    lines.append("")

    # Experiment info
    lines.append(f"Method: {result.get('method', 'Unknown')}")
    lines.append(f"Loss Level: {result.get('loss_level', 'Unknown')}")
    if result.get('seed'):
        lines.append(f"Seed: {result['seed']}")
    if result.get('radius'):
        lines.append(f"Radius: {result['radius']}m")
    lines.append("")

    # Data/Model info
    if inference_type == 'inference':
        lines.append("Evaluation Type: Baseline model on SUBSAMPLED data")
        lines.append(f"Data: {result.get('data_path', 'Unknown')}")
    else:
        lines.append("Evaluation Type: Trained model on ORIGINAL data")
        lines.append(f"Model: {result.get('model_path', 'Unknown')}")
    lines.append("")

    # Results
    lines.append("-" * 80)
    lines.append("RESULTS")
    lines.append("-" * 80)
    miou = result.get('mIoU')
    macc = result.get('mAcc')
    allacc = result.get('allAcc')

    if miou is not None:
        lines.append(f"mIoU:    {miou:.4f}")
        lines.append(f"mAcc:    {macc:.4f}")
        lines.append(f"allAcc:  {allacc:.4f}")
    else:
        lines.append("Results not available (inference may not have completed)")
    lines.append("")

    # Timing
    if result.get('total_time'):
        lines.append("-" * 80)
        lines.append("TIMING")
        lines.append("-" * 80)
        lines.append(f"Total Time: {result['total_time']}")
        if result.get('total_seconds') and result.get('num_scans'):
            time_per_scan = result['total_seconds'] / result['num_scans']
            lines.append(f"Scans Processed: {result['num_scans']}")
            lines.append(f"Time per Scan: {time_per_scan:.2f}s")
        lines.append("")

    # GPU Memory
    if result.get('peak_memory'):
        lines.append("-" * 80)
        lines.append("GPU MEMORY")
        lines.append("-" * 80)
        lines.append(f"GPU ID: {result.get('gpu_id', '0')}")
        lines.append(f"Initial: {result['initial_memory']} MB")
        lines.append(f"Peak: {result['peak_memory']} MB")
        lines.append("")

    # Class-wise results
    class_results = result.get('class_results', {})
    if class_results:
        lines.extend(format_class_table(class_results, "CLASS-WISE PERFORMANCE"))

    lines.append("")
    return "\n".join(lines)


def get_seed_or_radius(result: Dict) -> str:
    """Get the appropriate parameter value (Seed for RS/FPS/Poisson, R for IDIS, - for others)."""
    method = result.get('method', '')

    # Non-deterministic methods show seed
    if method in ['RS', 'FPS', 'Poisson']:
        return f"S{result['seed']}" if result.get('seed') else '-'
    # IDIS shows radius
    elif 'IDIS' in method:
        return f"R{result['radius']}" if result.get('radius') else 'R10'
    # Other deterministic methods (DBSCAN, Voxel)
    else:
        return '-'


def create_summary_table(all_results: List[Dict], output_dir: str, inference_type: str):
    """Create a summary table comparing all inference experiments."""
    lines = []

    if inference_type == 'inference':
        title = "SUMMARY: Baseline Model on Subsampled Data"
        description = "Evaluating baseline model (trained on original data) on various subsampled datasets"
    else:
        title = "SUMMARY: Trained Models on Original Data"
        description = "Evaluating models (trained on subsampled data) on original full-resolution data"

    lines.append("=" * 160)
    lines.append(title)
    lines.append("=" * 160)
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Filter results that have valid mIoU
    valid_results = [r for r in all_results if r.get('mIoU') is not None]

    if not valid_results:
        lines.append("No completed inference results found.")
        return "\n".join(lines)

    # Summary header with GPU memory columns
    lines.append("-" * 160)
    header = f"{'Experiment':<35} | {'Method':^10} | {'Loss':^6} | {'Param':^7} | {'mIoU':^10} | {'mAcc':^10} | {'allAcc':^10} | {'Peak GPU':^12} | {'Init GPU':^12}"
    lines.append(header)
    lines.append("-" * 160)

    # Sort by mIoU descending
    for result in sorted(valid_results, key=lambda x: x.get('mIoU', 0), reverse=True):
        exp_name = result['experiment_name'][:35]
        method = (result.get('method') or 'Unknown')[:10]
        loss = result.get('loss_level') or '-'
        param = get_seed_or_radius(result)
        miou = f"{result['mIoU']:.4f}"
        macc = f"{result['mAcc']:.4f}"
        allacc = f"{result['allAcc']:.4f}"

        # GPU memory
        peak_gpu = f"{result['peak_memory']} MB" if result.get('peak_memory') else '-'
        init_gpu = f"{result['initial_memory']} MB" if result.get('initial_memory') else '-'

        row = f"{exp_name:<35} | {method:^10} | {loss:^6} | {param:^7} | {miou:^10} | {macc:^10} | {allacc:^10} | {peak_gpu:^12} | {init_gpu:^12}"
        lines.append(row)

    lines.append("-" * 160)
    lines.append("")

    # Add per-loss-level comparison
    lines.append("")
    lines.append("=" * 120)
    lines.append("COMPARISON BY LOSS LEVEL")
    lines.append("=" * 120)
    lines.append("")

    # Group by loss level
    loss_levels = sorted(set(r.get('loss_level', 'Unknown') for r in valid_results))

    for loss_level in loss_levels:
        loss_results = [r for r in valid_results if r.get('loss_level') == loss_level]
        if not loss_results:
            continue

        lines.append(f"\n--- {loss_level} Loss ---")
        lines.append("-" * 120)
        header2 = f"{'Method':<20} | {'Param':^7} | {'mIoU':^10} | {'mAcc':^10} | {'allAcc':^10} | {'Peak GPU':^12} | {'Init GPU':^12}"
        lines.append(header2)
        lines.append("-" * 120)

        for result in sorted(loss_results, key=lambda x: x.get('mIoU', 0), reverse=True):
            method = result.get('method') or 'Unknown'
            param = get_seed_or_radius(result)
            miou = f"{result['mIoU']:.4f}"
            macc = f"{result['mAcc']:.4f}"
            allacc = f"{result['allAcc']:.4f}"

            # GPU memory
            peak_gpu = f"{result['peak_memory']} MB" if result.get('peak_memory') else '-'
            init_gpu = f"{result['initial_memory']} MB" if result.get('initial_memory') else '-'

            row = f"{method:<20} | {param:^7} | {miou:^10} | {macc:^10} | {allacc:^10} | {peak_gpu:^12} | {init_gpu:^12}"
            lines.append(row)

        lines.append("-" * 120)

    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{inference_type}.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved to: {summary_file}")
    return "\n".join(lines)


def create_combined_summary(inference_results: List[Dict], inference_on_original_results: List[Dict], output_dir: str):
    """Create a combined summary comparing both inference types."""
    lines = []

    lines.append("=" * 180)
    lines.append("COMBINED INFERENCE SUMMARY")
    lines.append("=" * 180)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Two types of inference:")
    lines.append("  1. inference/: Baseline model (trained on original) evaluated on SUBSAMPLED data")
    lines.append("  2. inference_on_original/: Models (trained on subsampled) evaluated on ORIGINAL data")
    lines.append("")

    # Get valid results
    valid_inf = [r for r in inference_results if r.get('mIoU') is not None]
    valid_orig = [r for r in inference_on_original_results if r.get('mIoU') is not None]

    # Create comparison table with GPU memory
    lines.append("=" * 180)
    lines.append("SIDE-BY-SIDE COMPARISON (Same Method/Loss)")
    lines.append("=" * 180)
    lines.append("")
    lines.append("-" * 180)
    header = f"{'Method':<15} | {'Loss':^6} | {'Param':^7} | {'Baseline→Sub mIoU':^18} | {'Train→Orig mIoU':^16} | {'Diff':^10} | {'Peak GPU (Orig)':^16} | {'Init GPU (Orig)':^16}"
    lines.append(header)
    lines.append("-" * 180)

    # Match experiments
    for inf_result in sorted(valid_inf, key=lambda x: (x.get('method') or '', x.get('loss_level') or '')):
        method = inf_result.get('method', 'Unknown')
        loss = inf_result.get('loss_level', '-')
        seed = inf_result.get('seed')
        radius = inf_result.get('radius')
        param = get_seed_or_radius(inf_result)

        # Find matching inference_on_original result
        matching_orig = None
        for orig_result in valid_orig:
            if (orig_result.get('method') == method and
                orig_result.get('loss_level') == loss and
                orig_result.get('seed') == seed):
                matching_orig = orig_result
                break

        inf_miou = f"{inf_result['mIoU']:.4f}"

        if matching_orig:
            orig_miou = f"{matching_orig['mIoU']:.4f}"
            diff = matching_orig['mIoU'] - inf_result['mIoU']
            diff_str = f"{diff:+.4f}"
            peak_gpu = f"{matching_orig['peak_memory']} MB" if matching_orig.get('peak_memory') else '-'
            init_gpu = f"{matching_orig['initial_memory']} MB" if matching_orig.get('initial_memory') else '-'
        else:
            orig_miou = "-"
            diff_str = "-"
            peak_gpu = "-"
            init_gpu = "-"

        row = f"{method:<15} | {loss:^6} | {param:^7} | {inf_miou:^18} | {orig_miou:^16} | {diff_str:^10} | {peak_gpu:^16} | {init_gpu:^16}"
        lines.append(row)

    lines.append("-" * 180)
    lines.append("")
    lines.append("Note: Positive difference means trained model on original data performs better than baseline on subsampled")
    lines.append("      Peak GPU and Init GPU columns show memory usage when evaluating on ORIGINAL (full-resolution) data")

    # Add GPU memory summary for deployment
    lines.append("")
    lines.append("")
    lines.append("=" * 120)
    lines.append("INFERENCE GPU MEMORY SUMMARY (For Deployment on Original Data)")
    lines.append("=" * 120)
    lines.append("")
    lines.append("Models trained on subsampled data, evaluated on original full-resolution data:")
    lines.append("")
    lines.append("-" * 120)
    header2 = f"{'Method':<20} | {'Loss':^8} | {'Param':^7} | {'mIoU':^10} | {'Peak GPU (MB)':^15} | {'Init GPU (MB)':^15}"
    lines.append(header2)
    lines.append("-" * 120)

    for result in sorted(valid_orig, key=lambda x: x.get('mIoU', 0), reverse=True):
        method = result.get('method') or 'Unknown'
        loss = result.get('loss_level', '-')
        param = get_seed_or_radius(result)
        miou = f"{result['mIoU']:.4f}"
        peak_gpu = f"{result['peak_memory']}" if result.get('peak_memory') else '-'
        init_gpu = f"{result['initial_memory']}" if result.get('initial_memory') else '-'

        row = f"{method:<20} | {loss:^8} | {param:^7} | {miou:^10} | {peak_gpu:^15} | {init_gpu:^15}"
        lines.append(row)

    lines.append("-" * 120)

    # Save combined summary
    summary_file = os.path.join(output_dir, "summary_inference_combined.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(lines))

    print(f"\nCombined summary saved to: {summary_file}")
    return "\n".join(lines)


def find_inference_experiments(base_dir: str, inference_type: str) -> List[str]:
    """Find all inference experiment directories."""
    if inference_type == 'inference':
        inference_dir = os.path.join(base_dir, "PTv3/SemanticKITTI/outputs/inference")
    else:
        inference_dir = os.path.join(base_dir, "PTv3/SemanticKITTI/outputs/inference_on_original")

    if not os.path.exists(inference_dir):
        return []

    exp_dirs = []
    for item in os.listdir(inference_dir):
        item_path = os.path.join(inference_dir, item)
        if os.path.isdir(item_path):
            # Check if it has test.log or inference_metrics.txt
            if os.path.exists(os.path.join(item_path, 'test.log')) or \
               os.path.exists(os.path.join(item_path, 'inference_metrics.txt')):
                exp_dirs.append(item_path)

    return sorted(exp_dirs)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract inference metrics from PTv3 inference logs and generate tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-discover all inference experiments (default behavior)
    python extract_inference_metrics.py --auto

    # Process specific inference type
    python extract_inference_metrics.py --type inference
    python extract_inference_metrics.py --type inference_on_original

    # Specify output directory
    python extract_inference_metrics.py --output /path/to/tables
        """
    )

    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['inference', 'inference_on_original', 'all'],
        default='all',
        help='Type of inference to process (default: all)'
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
        '--print', '-p',
        action='store_true',
        help='Print tables to stdout in addition to saving files'
    )

    return parser.parse_args()


def main():
    """Main function to process all inference experiments."""
    args = parse_args()

    # Determine base directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent of scripts/

    # Determine output directories
    if args.output:
        output_base = args.output
    else:
        output_base = os.path.join(base_dir, "docs/tables")

    # Create output directories
    inference_output_dir = os.path.join(output_base, "inference")
    inference_on_original_output_dir = os.path.join(output_base, "inference_on_original")

    os.makedirs(inference_output_dir, exist_ok=True)
    os.makedirs(inference_on_original_output_dir, exist_ok=True)

    print("=" * 60)
    print("Extracting Inference Metrics from PTv3 Logs")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print("")

    inference_results = []
    inference_on_original_results = []

    # Process inference/ experiments
    if args.type in ['inference', 'all']:
        print("\n" + "=" * 60)
        print("Processing: inference/ (Baseline model on subsampled data)")
        print("=" * 60)

        exp_dirs = find_inference_experiments(base_dir, 'inference')
        print(f"Found {len(exp_dirs)} experiments")

        for exp_dir in exp_dirs:
            print(f"\nProcessing: {os.path.basename(exp_dir)}")
            result = process_inference_experiment(exp_dir, 'inference')
            inference_results.append(result)

            # Format and save individual table
            table_str = format_inference_table(result)
            output_file = os.path.join(inference_output_dir, f"{result['experiment_name']}_metrics.txt")
            with open(output_file, 'w') as f:
                f.write(table_str)
            print(f"  Saved to: {output_file}")

            if args.print:
                print(table_str)

        # Create summary
        if inference_results:
            create_summary_table(inference_results, inference_output_dir, 'inference')

    # Process inference_on_original/ experiments
    if args.type in ['inference_on_original', 'all']:
        print("\n" + "=" * 60)
        print("Processing: inference_on_original/ (Trained models on original data)")
        print("=" * 60)

        exp_dirs = find_inference_experiments(base_dir, 'inference_on_original')
        print(f"Found {len(exp_dirs)} experiments")

        for exp_dir in exp_dirs:
            print(f"\nProcessing: {os.path.basename(exp_dir)}")
            result = process_inference_experiment(exp_dir, 'inference_on_original')
            inference_on_original_results.append(result)

            # Format and save individual table
            table_str = format_inference_table(result)
            output_file = os.path.join(inference_on_original_output_dir, f"{result['experiment_name']}_metrics.txt")
            with open(output_file, 'w') as f:
                f.write(table_str)
            print(f"  Saved to: {output_file}")

            if args.print:
                print(table_str)

        # Create summary
        if inference_on_original_results:
            create_summary_table(inference_on_original_results, inference_on_original_output_dir, 'inference_on_original')

    # Create combined summary if both types processed
    if args.type == 'all' and inference_results and inference_on_original_results:
        create_combined_summary(inference_results, inference_on_original_results, output_base)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nOutput directories:")
    print(f"  inference/: {inference_output_dir}")
    print(f"  inference_on_original/: {inference_on_original_output_dir}")


if __name__ == "__main__":
    main()
