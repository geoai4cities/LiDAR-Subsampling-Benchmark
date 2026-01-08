#!/usr/bin/env python3
"""
Profile Memory and FLOPs During Training

This script runs a single training iteration with profiling enabled,
measuring actual GPU memory and FLOPs consumed during training.

Usage:
    python profile_during_training.py --config CONFIG_FILE [--gpu GPU_ID]

Example:
    python profile_during_training.py --config ../configs/semantickitti/generated/ptv3_semantickitti_IDIS_loss50_140gb.py

This gives you the ACTUAL memory consumption during training (not just inference).
"""

import argparse
import os
import sys
import gc
import time
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from packaging import version

# Add pointcept to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
PTv3_ROOT = PROJECT_ROOT.parent
POINTCEPT_DIR = PTv3_ROOT / "pointcept"
sys.path.insert(0, str(POINTCEPT_DIR))

from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.utils.optimizer import build_optimizer


def format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def profile_training_step(cfg, device, num_iterations=5):
    """Profile actual training step with gradients."""

    print("\n" + "=" * 70)
    print("TRAINING PROFILER")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU Total Memory: {format_size(torch.cuda.get_device_properties(device).total_memory)}")

    # Build model
    print("\n[1/4] Building model...")
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {n_parameters:,} ({n_parameters/1e6:.2f}M)")

    if cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    torch.cuda.empty_cache()
    model_memory = torch.cuda.memory_allocated(device)
    print(f"      Model memory: {format_size(model_memory)}")

    # Build optimizer
    print("\n[2/4] Building optimizer...")
    optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    torch.cuda.empty_cache()
    optimizer_memory = torch.cuda.memory_allocated(device) - model_memory
    print(f"      Optimizer overhead: {format_size(optimizer_memory)}")

    # Build dataloader
    print("\n[3/4] Building dataloader...")
    train_data = build_dataset(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=True,
        num_workers=2,  # Use fewer workers for profiling
        collate_fn=partial(point_collate_fn, mix_prob=cfg.mix_prob),
        pin_memory=True,
        drop_last=True,
    )
    print(f"      Batch size: {cfg.batch_size_per_gpu}")
    print(f"      Number of samples: {len(train_data)}")
    print(f"      Iterations per epoch: {len(train_loader)}")

    # Set up AMP
    enable_amp = cfg.enable_amp
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
    if version.parse(torch.__version__) >= version.parse("2.4"):
        auto_cast = partial(torch.amp.autocast, device_type="cuda")
        grad_scaler = partial(torch.amp.GradScaler, device="cuda")
    else:
        auto_cast = torch.cuda.amp.autocast
        grad_scaler = torch.cuda.amp.GradScaler
    scaler = grad_scaler() if enable_amp else None
    print(f"      AMP: {enable_amp} ({cfg.amp_dtype if enable_amp else 'N/A'})")

    # Profile training iterations
    print(f"\n[4/4] Profiling {num_iterations} training iterations...")

    model.train()
    data_iter = iter(train_loader)

    memory_stats = []
    time_stats = []
    flops_total = 0

    for i in range(num_iterations):
        print(f"\n      Iteration {i+1}/{num_iterations}")

        # Get batch
        try:
            input_dict = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_dict = next(data_iter)

        # Move to GPU
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        if 'coord' in input_dict:
            num_points = input_dict['coord'].shape[0]
            print(f"        Points in batch: {num_points:,}")

        # Clear cache and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        before_memory = torch.cuda.memory_allocated(device)

        # Time the iteration
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Forward pass
        optimizer.zero_grad()

        if i == 0:
            # Profile FLOPs on first iteration
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                with_flops=True,
            ) as prof:
                with auto_cast(enabled=enable_amp, dtype=amp_dtype):
                    output_dict = model(input_dict)
                    loss = output_dict["loss"]
                torch.cuda.synchronize()

            # Extract FLOPs
            for event in prof.key_averages():
                if event.flops is not None and event.flops > 0:
                    flops_total += event.flops
        else:
            with auto_cast(enabled=enable_amp, dtype=amp_dtype):
                output_dict = model(input_dict)
                loss = output_dict["loss"]

        after_forward = torch.cuda.memory_allocated(device)

        # Backward pass
        if enable_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        after_backward = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)

        iteration_time = (end_time - start_time) * 1000  # ms

        memory_stats.append({
            'before': before_memory,
            'after_forward': after_forward,
            'after_backward': after_backward,
            'peak': peak_memory,
            'loss': loss.item(),
        })
        time_stats.append(iteration_time)

        print(f"        Loss: {loss.item():.4f}")
        print(f"        Forward memory: {format_size(after_forward - before_memory)}")
        print(f"        Peak memory: {format_size(peak_memory)}")
        print(f"        Iteration time: {iteration_time:.2f} ms")

    # Aggregate results
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)

    avg_peak_memory = sum(s['peak'] for s in memory_stats) / len(memory_stats)
    max_peak_memory = max(s['peak'] for s in memory_stats)
    avg_time = sum(time_stats) / len(time_stats)

    print(f"\nMemory Usage:")
    print(f"  - Model parameters: {format_size(model_memory)}")
    print(f"  - Optimizer states: {format_size(optimizer_memory)}")
    print(f"  - Average peak memory: {format_size(avg_peak_memory)}")
    print(f"  - Maximum peak memory: {format_size(max_peak_memory)}")
    print(f"  - Available GPU memory: {format_size(torch.cuda.get_device_properties(device).total_memory)}")

    print(f"\nComputation:")
    if flops_total > 0:
        gflops = flops_total / 1e9
        print(f"  - GFLOPs (forward pass): {gflops:.2f}")
        print(f"  - TFLOPs (forward pass): {flops_total / 1e12:.4f}")
    else:
        print(f"  - GFLOPs: Could not measure (check torch version)")

    print(f"\nTiming:")
    print(f"  - Average iteration time: {avg_time:.2f} ms")
    print(f"  - Min iteration time: {min(time_stats):.2f} ms")
    print(f"  - Max iteration time: {max(time_stats):.2f} ms")

    # Estimate full training time
    iters_per_epoch = len(train_loader)
    epochs = cfg.epoch
    total_iters = iters_per_epoch * epochs
    estimated_hours = (total_iters * avg_time / 1000) / 3600

    print(f"\nTraining Estimates:")
    print(f"  - Iterations per epoch: {iters_per_epoch}")
    print(f"  - Total epochs: {epochs}")
    print(f"  - Total iterations: {total_iters:,}")
    print(f"  - Estimated training time: {estimated_hours:.1f} hours")

    print("\n" + "=" * 70)

    return {
        'model_memory_gb': model_memory / 1e9,
        'optimizer_memory_gb': optimizer_memory / 1e9,
        'peak_memory_gb': max_peak_memory / 1e9,
        'gflops': flops_total / 1e9 if flops_total > 0 else None,
        'avg_iteration_time_ms': avg_time,
        'estimated_training_hours': estimated_hours,
    }


def main():
    parser = argparse.ArgumentParser(description='Profile Training Step')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to profile')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file (optional)')
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    print(f"Loading config: {args.config}")
    cfg = default_config_parser(args.config, [])

    # Run profiling
    results = profile_training_step(cfg, device, num_iterations=args.iterations)

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
