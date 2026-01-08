#!/usr/bin/env python3
"""
Model Profiling Script for PTv3 SemanticKITTI Training

Calculates:
- GFLOPs (FLoating point OPerations)
- GPU Memory consumption (peak and allocated)
- Model parameters count
- Inference time

Usage:
    python profile_model.py --config CONFIG_FILE [--gpu GPU_ID] [--num_samples N]

Examples:
    python profile_model.py --config ../configs/semantickitti/generated/ptv3_semantickitti_IDIS_loss50_140gb.py
    python profile_model.py --config ../configs/semantickitti/generated/ptv3_semantickitti_RS_loss50_seed1_140gb.py --gpu 0
"""

import argparse
import os
import sys
import time
import gc
from pathlib import Path

import torch
import torch.nn as nn

# Add pointcept to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
PTv3_ROOT = PROJECT_ROOT.parent
POINTCEPT_DIR = PTv3_ROOT / "pointcept"
sys.path.insert(0, str(POINTCEPT_DIR))

from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model
from pointcept.datasets import build_dataset, collate_fn


def format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def profile_memory(model, input_dict, device, warmup_runs=3, profile_runs=10):
    """Profile GPU memory usage during forward pass."""
    model.eval()

    # Move input to device
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].to(device)

    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    # Record baseline memory (model loaded)
    baseline_memory = torch.cuda.memory_allocated(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_dict)
            torch.cuda.synchronize()

    # Reset after warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    # Profile runs
    allocated_memories = []
    peak_memories = []

    with torch.no_grad():
        for _ in range(profile_runs):
            torch.cuda.reset_peak_memory_stats(device)
            start_mem = torch.cuda.memory_allocated(device)

            _ = model(input_dict)
            torch.cuda.synchronize()

            peak_mem = torch.cuda.max_memory_allocated(device)
            end_mem = torch.cuda.memory_allocated(device)

            allocated_memories.append(end_mem - start_mem)
            peak_memories.append(peak_mem)

            torch.cuda.empty_cache()

    return {
        'baseline_memory': baseline_memory,
        'avg_peak_memory': sum(peak_memories) / len(peak_memories),
        'max_peak_memory': max(peak_memories),
        'avg_allocated_memory': sum(allocated_memories) / len(allocated_memories),
    }


def profile_flops_with_profiler(model, input_dict, device):
    """Profile FLOPs using PyTorch profiler."""
    model.eval()

    # Move input to device
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_dict)
            torch.cuda.synchronize()

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            _ = model(input_dict)
            torch.cuda.synchronize()

    # Extract FLOP count
    total_flops = 0
    for event in prof.key_averages():
        if event.flops is not None and event.flops > 0:
            total_flops += event.flops

    return total_flops, prof


def profile_inference_time(model, input_dict, device, warmup_runs=10, profile_runs=50):
    """Profile inference time."""
    model.eval()

    # Move input to device
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_dict)
            torch.cuda.synchronize()

    # Profile
    times = []
    with torch.no_grad():
        for _ in range(profile_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(input_dict)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        'avg_time_ms': sum(times) / len(times),
        'min_time_ms': min(times),
        'max_time_ms': max(times),
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def profile_training_memory(model, input_dict, device, optimizer_type='AdamW'):
    """Profile memory during training (forward + backward)."""
    model.train()

    # Move input to device
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].to(device)

    # Create optimizer
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        output = model(input_dict)
        loss = output['loss']
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Profile
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    optimizer.zero_grad()

    # Forward
    output = model(input_dict)
    after_forward = torch.cuda.memory_allocated(device)

    # Backward
    loss = output['loss']
    loss.backward()
    after_backward = torch.cuda.memory_allocated(device)

    # Optimizer step
    optimizer.step()
    after_optimizer = torch.cuda.memory_allocated(device)

    peak_memory = torch.cuda.max_memory_allocated(device)

    return {
        'after_forward': after_forward,
        'after_backward': after_backward,
        'after_optimizer': after_optimizer,
        'peak_training_memory': peak_memory,
    }


def get_sample_input(cfg, device, num_samples=1):
    """Get sample input from dataset."""
    print("Loading validation dataset for sample input...")
    val_data = build_dataset(cfg.data.val)

    # Get samples
    samples = []
    for i in range(min(num_samples, len(val_data))):
        samples.append(val_data[i])

    # Collate samples
    input_dict = collate_fn(samples)

    # Print input info
    print(f"\nInput statistics:")
    if 'coord' in input_dict:
        print(f"  - Number of points: {input_dict['coord'].shape[0]}")
    if 'feat' in input_dict:
        print(f"  - Feature dimension: {input_dict['feat'].shape[-1]}")
    if 'offset' in input_dict:
        print(f"  - Batch size: {len(input_dict['offset'])}")

    return input_dict


def main():
    parser = argparse.ArgumentParser(description='Profile PTv3 Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to use')
    parser.add_argument('--skip-training-profile', action='store_true',
                        help='Skip training memory profiling (faster)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (optional)')
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    print("=" * 70)
    print("PTv3 Model Profiler")
    print("=" * 70)
    print(f"\nConfig: {args.config}")
    print(f"GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
    print(f"GPU Memory: {format_size(torch.cuda.get_device_properties(device).total_memory)}")

    # Load config
    print("\n" + "-" * 70)
    print("Loading configuration...")
    cfg = default_config_parser(args.config, [])

    # Build model
    print("Building model...")
    model = build_model(cfg.model)
    model = model.to(device)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # Get sample input
    input_dict = get_sample_input(cfg, device, args.num_samples)

    # Profile inference memory
    print("\n" + "-" * 70)
    print("Profiling inference memory...")
    mem_results = profile_memory(model, input_dict.copy(), device)
    print(f"\nInference Memory Results:")
    print(f"  - Model memory (baseline): {format_size(mem_results['baseline_memory'])}")
    print(f"  - Average peak memory: {format_size(mem_results['avg_peak_memory'])}")
    print(f"  - Maximum peak memory: {format_size(mem_results['max_peak_memory'])}")

    # Profile FLOPs
    print("\n" + "-" * 70)
    print("Profiling FLOPs (this may take a moment)...")
    try:
        total_flops, prof = profile_flops_with_profiler(model, input_dict.copy(), device)
        gflops = total_flops / 1e9
        print(f"\nFLOPs Results:")
        print(f"  - Total FLOPs: {total_flops:,.0f}")
        print(f"  - GFLOPs: {gflops:.2f}")

        # Print top operations by FLOPs
        print(f"\nTop 10 Operations by FLOPs:")
        events = sorted(
            [e for e in prof.key_averages() if e.flops is not None and e.flops > 0],
            key=lambda x: x.flops,
            reverse=True
        )[:10]
        for event in events:
            print(f"  - {event.key}: {event.flops/1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"Warning: Could not profile FLOPs: {e}")
        gflops = None

    # Profile inference time
    print("\n" + "-" * 70)
    print("Profiling inference time...")
    time_results = profile_inference_time(model, input_dict.copy(), device)
    print(f"\nInference Time Results:")
    print(f"  - Average: {time_results['avg_time_ms']:.2f} ms")
    print(f"  - Min: {time_results['min_time_ms']:.2f} ms")
    print(f"  - Max: {time_results['max_time_ms']:.2f} ms")
    print(f"  - Std: {time_results['std_time_ms']:.2f} ms")

    # Profile training memory
    if not args.skip_training_profile:
        print("\n" + "-" * 70)
        print("Profiling training memory (forward + backward + optimizer)...")
        try:
            train_mem = profile_training_memory(model, input_dict.copy(), device)
            print(f"\nTraining Memory Results:")
            print(f"  - After forward pass: {format_size(train_mem['after_forward'])}")
            print(f"  - After backward pass: {format_size(train_mem['after_backward'])}")
            print(f"  - After optimizer step: {format_size(train_mem['after_optimizer'])}")
            print(f"  - Peak training memory: {format_size(train_mem['peak_training_memory'])}")
        except Exception as e:
            print(f"Warning: Could not profile training memory: {e}")
            train_mem = None
    else:
        train_mem = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: PTv3 ({cfg.model.type})")
    print(f"Parameters: {trainable_params/1e6:.2f}M trainable")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.2f}")
    print(f"Inference Memory (peak): {format_size(mem_results['max_peak_memory'])}")
    print(f"Inference Time: {time_results['avg_time_ms']:.2f} ms")
    if train_mem:
        print(f"Training Memory (peak): {format_size(train_mem['peak_training_memory'])}")
    print("=" * 70)

    # Save results to file if requested
    if args.output:
        results = {
            'config': args.config,
            'model_type': cfg.model.type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'gflops': gflops,
            'inference_memory_peak': mem_results['max_peak_memory'],
            'inference_time_ms': time_results['avg_time_ms'],
            'training_memory_peak': train_mem['peak_training_memory'] if train_mem else None,
        }

        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
