#!/usr/bin/env python3
"""
Quick Model Profiler - Runs at training start, logs estimated metrics to file.

Usage: python quick_profile.py --config CONFIG_FILE --output OUTPUT_FILE [--gpu GPU_ID]
"""

import argparse
import re
import sys
from pathlib import Path

import torch


def format_bytes(size_bytes):
    """Format bytes to GB."""
    return f"{size_bytes / 1e9:.1f} GB"


def parse_config_value(config_text, key, default):
    """Extract a simple value from config text using regex."""
    # Match patterns like: epoch = 10 or batch_size = 20
    pattern = rf'^{key}\s*=\s*(\d+)'
    match = re.search(pattern, config_text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return default


def quick_profile(config_path, gpu_id=0):
    """Run quick profiling and return metrics dict."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # Read config file directly (avoid Pointcept's parser which has issues)
    config_text = Path(config_path).read_text()

    # Extract values using simple regex
    epochs = parse_config_value(config_text, 'epoch', 10)
    batch_size = parse_config_value(config_text, 'batch_size', 20)
    grad_accum = parse_config_value(config_text, 'gradient_accumulation_steps', 4)

    # PTv3 approximate parameter count
    # PT-v3m1 with enc_channels=(32, 64, 128, 256, 512) has ~46M params
    n_params_approx = 46.2e6

    # Estimate iterations per epoch (SemanticKITTI has ~19130 train scans)
    iters_per_epoch = 4782

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_total = torch.cuda.get_device_properties(device).total_memory

    return {
        'parameters': n_params_approx,
        'gpu_name': gpu_name,
        'gpu_memory': gpu_mem_total,
        'epochs': epochs,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'effective_batch': batch_size * grad_accum,
        'iters_per_epoch': iters_per_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description='Quick Model Profiler')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='Output file to append results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    try:
        metrics = quick_profile(args.config, gpu_id=args.gpu)

        output = f"""
================================================================================
MODEL PROFILE (PTv3-m1)
================================================================================
Parameters:     ~{metrics['parameters']/1e6:.1f}M trainable
GPU:            {metrics['gpu_name']}
GPU Memory:     {format_bytes(metrics['gpu_memory'])}
Batch Size:     {metrics['batch_size']} (effective: {metrics['effective_batch']} with grad_accum={metrics['grad_accum']})
Epochs:         {metrics['epochs']}
Iters/Epoch:    {metrics['iters_per_epoch']}
Est. Time:      ~{metrics['epochs'] * 2:.0f} hours ({metrics['epochs']} epochs x ~2h/epoch)
================================================================================
"""
        print(output)

        with open(args.output, 'a') as f:
            f.write(output)

    except Exception as e:
        error_msg = f"\n[PROFILE] Error: {e}\n"
        print(error_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
