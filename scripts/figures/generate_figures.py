#!/usr/bin/env python3
"""
Generate Academic Figures from Training Metrics

This script reads experiment results from all_experiments_detailed.txt
and generates publication-quality figures:

1. 01_ptv3_metric_grouped - Grouped bar chart with mIoU and GPU memory side by side
2. 02_ptv3_spatial_distribution_analysis - 2x2 analysis (Pareto, scatter, heatmaps)
3. 03_ptv3_ranking_bump_chart - Method ranking across loss levels

Output formats: PNG, SVG, PDF
"""

import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse

import matplotlib.pyplot as plt

# Script and project paths (for relative path resolution)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/figures -> scripts -> project root
DEFAULT_INPUT = PROJECT_ROOT / 'docs' / 'tables' / 'all_experiments_detailed.txt'
DEFAULT_OUTPUT = PROJECT_ROOT / 'docs' / 'figures'
import matplotlib.patches as mpatches
import numpy as np

# Set up academic-style plotting with improved readability
plt.rcParams.update({
    # Font settings - use serif fonts for academic papers (IEEE/ACM style)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'mathtext.fontset': 'stix',  # STIX fonts for math (compatible with Times)

    # Axes settings
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,

    # Tick settings
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,

    # Legend settings
    'legend.fontsize': 11,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',

    # Grid settings
    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    # Figure settings
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Use tight layout
    'figure.autolayout': False,
})

# Color scheme - matching main paper (Figure 8)
# Paper naming: RS, SB (Space-based/Poisson), VB (Voxel-based), IDIS, DEPOCO
# RGBA values from paper: RS:66b8ffff, SB:ff6752ff, VB:fff809ff, IDIS:50e17cff, DEPOCO:ff9c01ff
METHOD_COLORS = {
    'RS': '#66b8ff',        # Light Blue (from paper)
    'DBSCAN': '#f77189',    # Pink (new method)
    'FPS': '#97a431',       # Olive Green (new method)
    'VB': '#ff9c01',        # Orange (Voxel-based)
    'SB': '#ff6752',        # Salmon Red (Space-based/Poisson, from paper)
    'IDIS': '#2ca02c',      # Forest Green
    'DEPOCO': '#9467bd',    # Purple (distinct from others)
    'baseline': '#000000',  # Black
}

# Hatching patterns for methods (disabled for cleaner look)
METHOD_HATCHES = {
    'RS': '',
    'DBSCAN': '',
    'FPS': '',
    'VB': '',    # Voxel-based
    'SB': '',    # Space-based (Poisson)
    'IDIS': '',
    'DEPOCO': '',
}

# Method display order (for consistent plotting) - using paper naming
# Non-deterministic methods first (RS, FPS, SB), then deterministic (IDIS, DBSCAN, VB, DEPOCO)
METHOD_ORDER = ['RS', 'FPS', 'SB', 'IDIS', 'DBSCAN', 'VB', 'DEPOCO']

# Method name mapping: data source names -> paper names
METHOD_NAME_MAP = {
    'Voxel': 'VB',      # Voxel-based
    'Poisson': 'SB',    # Space-based (Poisson disk)
}

# Loss level order
LOSS_ORDER = [10, 30, 50, 70, 90]


def parse_experiments_file(filepath):
    """
    Parse the all_experiments_detailed.txt file and extract metrics.

    Returns:
        dict: {
            'baseline': {'mIoU': float, 'gpu_peak': float},
            'experiments': {
                (method, loss_level): {'mIoU': float, 'gpu_peak': float, 'seed': int, 'r_value': int}
            }
        }
    """
    results = {
        'baseline': None,
        'experiments': {}
    }

    with open(filepath, 'r') as f:
        content = f.read()

    # Find all experiment blocks using EXPERIMENT: pattern
    # Each experiment starts with ===...=== followed by EXPERIMENT: name
    experiment_pattern = r'={80,}\nEXPERIMENT:\s*(\S+)\n={80,}\n(.*?)(?=\n={80,}\nEXPERIMENT:|\Z)'
    matches = re.findall(experiment_pattern, content, re.DOTALL)

    for exp_name, section in matches:
        # Extract method
        method_match = re.search(r'Method:\s*(\S+)', section)
        method = method_match.group(1) if method_match else None

        # Apply name mapping (Voxel -> VB, Poisson -> SB) for paper consistency
        if method in METHOD_NAME_MAP:
            method = METHOD_NAME_MAP[method]

        # Extract loss level
        loss_match = re.search(r'Loss Level:\s*(\d+)%', section)
        loss_level = int(loss_match.group(1)) if loss_match else None

        # Extract seed (if present)
        seed_match = re.search(r'Seed:\s*(\d+)', section)
        seed = int(seed_match.group(1)) if seed_match else 1

        # Extract R value for IDIS (if present)
        r_match = re.search(r'Radius:\s*(\d+)m', section)
        r_value = int(r_match.group(1)) if r_match else None

        # Extract Test mIoU
        test_miou_match = re.search(r'Test mIoU:\s*([\d.]+)', section)
        test_miou = float(test_miou_match.group(1)) if test_miou_match else None

        # If no test mIoU, try to get best validation mIoU
        if test_miou is None:
            val_miou_match = re.search(r'Best Validation mIoU:\s*([\d.]+)', section)
            test_miou = float(val_miou_match.group(1)) if val_miou_match else None

        # Extract GPU peak from epoch data - look for the GPU column in epoch table
        # Format: Epoch | Train Loss | Val mIoU | Val mAcc | Val allAcc | GPU (GB) | Time
        # Example: 1    |    0.7959    |   0.5383   |   0.6620   |   0.9096   |    30.81     |  03:48:12
        # GPU is the 6th column - need to skip Epoch, Train Loss, Val mIoU, Val mAcc, Val allAcc
        gpu_peaks = re.findall(r'^\s*\d+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|', section, re.MULTILINE)
        gpu_peak = max([float(g) for g in gpu_peaks]) if gpu_peaks else None

        print(f"  Parsed: {exp_name} - Method={method}, Loss={loss_level}%, mIoU={test_miou}, GPU={gpu_peak}")

        # Store results
        if method == 'baseline' and loss_level == 0:
            results['baseline'] = {
                'mIoU': test_miou,
                'gpu_peak': gpu_peak
            }
        elif method and loss_level is not None and test_miou is not None:
            # Create key based on method and loss level
            # For IDIS with R-value ablation, include R in key
            if method == 'IDIS' and r_value and r_value != 10:
                key = (f'IDIS_R{r_value}', loss_level, seed)
            else:
                key = (method, loss_level, seed)

            results['experiments'][key] = {
                'mIoU': test_miou,
                'gpu_peak': gpu_peak,
                'seed': seed,
                'r_value': r_value,
                'exp_name': exp_name
            }

    return results


def aggregate_by_method_loss(results):
    """
    Aggregate results by method and loss level, averaging across seeds.
    For IDIS, also aggregate R-value variants (R=5, 10, 15) as variations.

    Returns:
        dict: {
            method: {
                loss_level: {'mIoU': float, 'mIoU_std': float, 'gpu_peak': float, 'n_seeds': int}
            }
        }
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: {'mIoUs': [], 'gpu_peaks': []}))

    for (method, loss, seed), data in results['experiments'].items():
        # Map IDIS R-value variants to main IDIS for aggregation
        if method.startswith('IDIS_R'):
            method_key = 'IDIS'  # Aggregate all R-values under IDIS
        else:
            method_key = method

        if data['mIoU'] is not None:
            aggregated[method_key][loss]['mIoUs'].append(data['mIoU'])
        if data['gpu_peak'] is not None:
            aggregated[method_key][loss]['gpu_peaks'].append(data['gpu_peak'])

    # Calculate mean and std
    final = {}
    for method, losses in aggregated.items():
        final[method] = {}
        for loss, values in losses.items():
            mious = values['mIoUs']
            gpus = values['gpu_peaks']
            final[method][loss] = {
                'mIoU': np.mean(mious) if mious else None,
                'mIoU_std': np.std(mious) if len(mious) > 1 else 0,
                'gpu_peak': np.mean(gpus) if gpus else None,
                'n_seeds': len(mious)
            }

    return final


def create_stacked_figures(results, output_dir, show_plot=False):
    """
    Create two stacked figures:
    - Top: mIoU vs Loss Level
    - Bottom: GPU Memory vs Loss Level
    """
    aggregated = aggregate_by_method_loss(results)
    baseline = results['baseline']

    # Determine which loss levels have data
    all_losses = set()
    for method_data in aggregated.values():
        all_losses.update(method_data.keys())
    available_losses = sorted([l for l in LOSS_ORDER if l in all_losses])

    if not available_losses:
        print("No experiment data found!")
        return

    # Determine which methods have data
    available_methods = [m for m in METHOD_ORDER if m in aggregated]

    print(f"Available methods: {available_methods}")
    print(f"Available loss levels: {available_losses}")

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.35)

    # Bar width and positions
    n_methods = len(available_methods)
    n_losses = len(available_losses)
    bar_width = 0.8 / n_methods

    # X positions for loss level groups
    x_positions = np.arange(n_losses)

    # ==================== Figure A: mIoU ====================
    ax1.set_title('(A) PTv3 Semantic Segmentation Performance on SemanticKITTI',
                  fontweight='bold', pad=10)

    # Plot baseline reference line
    if baseline and baseline['mIoU']:
        ax1.axhline(y=baseline['mIoU'], color='black', linestyle='--',
                    linewidth=1.5, label=f"Baseline ({baseline['mIoU']:.4f})", zorder=1)

    # Plot bars for each method
    for i, method in enumerate(available_methods):
        if method not in aggregated:
            continue

        x_offset = (i - n_methods/2 + 0.5) * bar_width

        mious = []
        stds = []
        positions = []

        for j, loss in enumerate(available_losses):
            if loss in aggregated[method] and aggregated[method][loss]['mIoU'] is not None:
                mious.append(aggregated[method][loss]['mIoU'])
                stds.append(aggregated[method][loss]['mIoU_std'])
                positions.append(x_positions[j] + x_offset)

        if mious:
            bars = ax1.bar(positions, mious, bar_width * 0.9,
                          label=method,
                          color=METHOD_COLORS.get(method, '#888888'),
                          hatch=METHOD_HATCHES.get(method, ''),
                          edgecolor='black',
                          linewidth=0.5,
                          yerr=stds if any(s > 0 for s in stds) else None,
                          capsize=2,
                          zorder=2)

            # Add value labels above bars - black text
            for bar, val in zip(bars, mious):
                height = bar.get_height()
                ax1.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7,
                           fontweight='normal',
                           color='black',
                           rotation=90)

    ax1.set_xlabel('Loss Level (%)', fontweight='bold')
    ax1.set_ylabel('mIoU', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{l}%' for l in available_losses])

    # Set y-axis limits - fixed minimum at 0.45
    all_mious = [aggregated[m][l]['mIoU'] for m in aggregated for l in aggregated[m]
                 if aggregated[m][l]['mIoU'] is not None]
    if all_mious:
        y_min = 0.45
        y_max = max(max(all_mious), baseline['mIoU'] if baseline else 0) + 0.05
        ax1.set_ylim(y_min, y_max)

    # Legend inside plot area - horizontal line above baseline
    ax1.legend(loc='upper center', ncol=6, framealpha=0.95, fontsize=8,
               bbox_to_anchor=(0.5, 0.98))
    ax1.grid(True, axis='y', alpha=0.3)

    # ==================== Figure B: GPU Memory ====================
    ax2.set_title('(B) GPU Memory Consumption During Training',
                  fontweight='bold', pad=10)

    # Plot baseline reference line
    if baseline and baseline['gpu_peak']:
        ax2.axhline(y=baseline['gpu_peak'], color='black', linestyle='--',
                    linewidth=1.5, label=f"Baseline ({baseline['gpu_peak']:.1f} GB)", zorder=1)

    # Plot bars for each method
    for i, method in enumerate(available_methods):
        if method not in aggregated:
            continue

        x_offset = (i - n_methods/2 + 0.5) * bar_width

        gpus = []
        positions = []

        for j, loss in enumerate(available_losses):
            if loss in aggregated[method] and aggregated[method][loss]['gpu_peak'] is not None:
                gpus.append(aggregated[method][loss]['gpu_peak'])
                positions.append(x_positions[j] + x_offset)

        if gpus:
            bars = ax2.bar(positions, gpus, bar_width * 0.9,
                          label=method,
                          color=METHOD_COLORS.get(method, '#888888'),
                          hatch=METHOD_HATCHES.get(method, ''),
                          edgecolor='black',
                          linewidth=0.5,
                          zorder=2)

            # Add value labels above bars - black text
            for bar, val in zip(bars, gpus):
                height = bar.get_height()
                ax2.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=7,
                           fontweight='normal',
                           color='black',
                           rotation=90)

    ax2.set_xlabel('Loss Level (%)', fontweight='bold')
    ax2.set_ylabel('GPU Memory (GB)', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{l}%' for l in available_losses])

    # Set y-axis limits - extra space for legend
    all_gpus = [aggregated[m][l]['gpu_peak'] for m in aggregated for l in aggregated[m]
                if aggregated[m][l]['gpu_peak'] is not None]
    if all_gpus:
        y_max = max(max(all_gpus), baseline['gpu_peak'] if baseline else 0) * 1.20
        ax2.set_ylim(0, y_max)

    # Legend inside plot area - horizontal line above baseline
    ax2.legend(loc='upper center', ncol=6, framealpha=0.95, fontsize=8,
               bbox_to_anchor=(0.5, 0.99))
    ax2.grid(True, axis='y', alpha=0.3)

    # Close the stacked figure without saving (we only want the 3 specific figures)
    plt.close(fig)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only create the 3 required figures:
    # 1. Metric-grouped figure (all mIoU | separator | all GPU)
    create_metric_grouped_figure(results, output_dir, show_plot)

    # 2. Spatial distribution analysis figure
    create_spatial_distribution_analysis(results, output_dir, show_plot)


def create_individual_figures(results, output_dir, show_plot=False):
    """
    Create individual figures for mIoU and GPU memory.
    """
    aggregated = aggregate_by_method_loss(results)
    baseline = results['baseline']

    # Determine available data
    all_losses = set()
    for method_data in aggregated.values():
        all_losses.update(method_data.keys())
    available_losses = sorted([l for l in LOSS_ORDER if l in all_losses])
    available_methods = [m for m in METHOD_ORDER if m in aggregated]

    if not available_losses:
        return

    n_methods = len(available_methods)
    n_losses = len(available_losses)
    bar_width = 0.8 / n_methods
    x_positions = np.arange(n_losses)

    output_dir = Path(output_dir)

    # ==================== Individual Figure A: mIoU ====================
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_title('PTv3 Semantic Segmentation Performance on SemanticKITTI',
                  fontweight='bold', pad=10)

    if baseline and baseline['mIoU']:
        ax1.axhline(y=baseline['mIoU'], color='black', linestyle='--',
                    linewidth=1.5, label=f"Baseline ({baseline['mIoU']:.4f})", zorder=1)

    for i, method in enumerate(available_methods):
        if method not in aggregated:
            continue
        x_offset = (i - n_methods/2 + 0.5) * bar_width

        mious, stds, positions = [], [], []
        for j, loss in enumerate(available_losses):
            if loss in aggregated[method] and aggregated[method][loss]['mIoU'] is not None:
                mious.append(aggregated[method][loss]['mIoU'])
                stds.append(aggregated[method][loss]['mIoU_std'])
                positions.append(x_positions[j] + x_offset)

        if mious:
            bars = ax1.bar(positions, mious, bar_width * 0.9,
                          label=method,
                          color=METHOD_COLORS.get(method, '#888888'),
                          hatch=METHOD_HATCHES.get(method, ''),
                          edgecolor='black', linewidth=0.5,
                          yerr=stds if any(s > 0 for s in stds) else None,
                          capsize=2, zorder=2)

            for bar, val in zip(bars, mious):
                ax1.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, -5), textcoords="offset points",
                           ha='center', va='top', fontsize=8, fontweight='normal',
                           color='black', rotation=90)

    ax1.set_xlabel('Loss Level (%)', fontweight='bold')
    ax1.set_ylabel('mIoU', fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{l}%' for l in available_losses])

    all_mious = [aggregated[m][l]['mIoU'] for m in aggregated for l in aggregated[m]
                 if aggregated[m][l]['mIoU'] is not None]
    if all_mious:
        ax1.set_ylim(min(all_mious) - 0.05,
                     max(max(all_mious), baseline['mIoU'] if baseline else 0) + 0.05)

    # Legend inside plot area - horizontal line above baseline
    ax1.legend(loc='upper center', ncol=6, framealpha=0.95, fontsize=8,
               bbox_to_anchor=(0.5, 0.98))
    ax1.grid(True, axis='y', alpha=0.3)

    for fmt in ['png', 'svg', 'pdf']:
        fig1.savefig(output_dir / f'ptv3_miou.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved individual mIoU figures")
    plt.close(fig1)

    # ==================== Individual Figure B: GPU Memory ====================
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.set_title('GPU Memory Consumption During Training', fontweight='bold', pad=10)

    if baseline and baseline['gpu_peak']:
        ax2.axhline(y=baseline['gpu_peak'], color='black', linestyle='--',
                    linewidth=1.5, label=f"Baseline ({baseline['gpu_peak']:.1f} GB)", zorder=1)

    for i, method in enumerate(available_methods):
        if method not in aggregated:
            continue
        x_offset = (i - n_methods/2 + 0.5) * bar_width

        gpus, positions = [], []
        for j, loss in enumerate(available_losses):
            if loss in aggregated[method] and aggregated[method][loss]['gpu_peak'] is not None:
                gpus.append(aggregated[method][loss]['gpu_peak'])
                positions.append(x_positions[j] + x_offset)

        if gpus:
            bars = ax2.bar(positions, gpus, bar_width * 0.9,
                          label=method,
                          color=METHOD_COLORS.get(method, '#888888'),
                          hatch=METHOD_HATCHES.get(method, ''),
                          edgecolor='black', linewidth=0.5, zorder=2)

            for bar, val in zip(bars, gpus):
                ax2.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, -5), textcoords="offset points",
                           ha='center', va='top', fontsize=8, fontweight='normal',
                           color='black', rotation=90)

    ax2.set_xlabel('Loss Level (%)', fontweight='bold')
    ax2.set_ylabel('GPU Memory (GB)', fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{l}%' for l in available_losses])

    all_gpus = [aggregated[m][l]['gpu_peak'] for m in aggregated for l in aggregated[m]
                if aggregated[m][l]['gpu_peak'] is not None]
    if all_gpus:
        ax2.set_ylim(0, max(max(all_gpus), baseline['gpu_peak'] if baseline else 0) * 1.20)

    # Legend inside plot area - horizontal line above baseline
    ax2.legend(loc='upper center', ncol=6, framealpha=0.95, fontsize=8,
               bbox_to_anchor=(0.5, 0.99))
    ax2.grid(True, axis='y', alpha=0.3)

    for fmt in ['png', 'svg', 'pdf']:
        fig2.savefig(output_dir / f'ptv3_gpu_memory.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved individual GPU memory figures")
    plt.close(fig2)


def create_dual_axis_figure(results, output_dir, show_plot=False):
    """
    Create a single figure with dual y-axes (style: solid bars + dashed bars side by side):
    - Left Y-axis: mIoU (solid bars)
    - Right Y-axis: GPU Memory (GB) (dashed border, lighter fill)
    - X-axis: Loss Level (%)
    """
    aggregated = aggregate_by_method_loss(results)
    baseline = results['baseline']

    # Determine available data
    all_losses = set()
    for method_data in aggregated.values():
        all_losses.update(method_data.keys())
    available_losses = sorted([l for l in LOSS_ORDER if l in all_losses])
    available_methods = [m for m in METHOD_ORDER if m in aggregated]

    if not available_losses:
        return

    n_methods = len(available_methods)
    n_losses = len(available_losses)

    # Bar positioning: each loss level has paired bars (mIoU solid + GPU dashed) per method
    group_width = 0.8  # Total width for each loss level group
    pair_width = group_width / n_methods  # Width for each method's pair
    single_bar_width = pair_width * 0.4  # Width of individual bar

    x_positions = np.arange(n_losses)

    output_dir = Path(output_dir)

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    fig.suptitle('PTv3 Performance vs GPU Memory on SemanticKITTI',
                 fontweight='bold', fontsize=14, y=0.98)

    # Plot bars for each method
    for i, method in enumerate(available_methods):
        if method not in aggregated:
            continue

        # Position offset for this method within the group
        method_offset = (i - n_methods/2 + 0.5) * pair_width

        for j, loss in enumerate(available_losses):
            if loss not in aggregated[method]:
                continue

            data = aggregated[method][loss]
            base_x = x_positions[j] + method_offset

            # mIoU bar (solid, left of pair)
            if data['mIoU'] is not None:
                miou_x = base_x - single_bar_width/2 - 0.01

                # Add error bar if std available
                yerr = data.get('mIoU_std', 0) if data.get('mIoU_std', 0) > 0 else None
                bar1 = ax1.bar(miou_x, data['mIoU'], single_bar_width,
                              color=METHOD_COLORS.get(method, '#888888'),
                              edgecolor='black', linewidth=1.0,
                              yerr=yerr, capsize=2, error_kw={'linewidth': 1},
                              zorder=3)

                # Calculate % drop from baseline
                miou_pct = ""
                if baseline and baseline['mIoU']:
                    pct_change = ((data['mIoU'] - baseline['mIoU']) / baseline['mIoU']) * 100
                    miou_pct = f" ({pct_change:+.0f}%)"

                # Add ± std if multiple seeds
                std_text = ""
                if data.get('mIoU_std', 0) > 0:
                    std_text = f"±{data['mIoU_std']:.3f}"

                # Value label - inside bar except for 90% loss
                label_text = f'{data["mIoU"]:.2f}{std_text}{miou_pct}'
                if loss == 90:
                    # Above bar for 90% loss (bars are shorter)
                    ax1.annotate(label_text,
                               xy=(miou_x, data['mIoU'] + 0.005 + (yerr if yerr else 0)),
                               ha='center', va='bottom', fontsize=8,
                               fontweight='normal', color='black', rotation=90)
                else:
                    # Inside bar for other loss levels
                    ax1.annotate(label_text,
                               xy=(miou_x, data['mIoU'] - 0.01),
                               ha='center', va='top', fontsize=8,
                               fontweight='normal', color='black', rotation=90)

            # GPU bar (dashed border, lighter fill, right of pair)
            if data['gpu_peak'] is not None:
                gpu_x = base_x + single_bar_width/2 + 0.01
                # Lighter version of the color (blend 40% towards white)
                import matplotlib.colors as mcolors
                base_color = METHOD_COLORS.get(method, '#888888')
                rgb = mcolors.to_rgb(base_color)
                # Blend towards white: light = color + (white - color) * factor
                blend_factor = 0.4
                light_rgb = tuple(c + (1.0 - c) * blend_factor for c in rgb)

                bar2 = ax2.bar(gpu_x, data['gpu_peak'], single_bar_width,
                              color=light_rgb,
                              edgecolor=base_color, linewidth=2.0,
                              linestyle='--', zorder=2)
                # Calculate % reduction from baseline
                gpu_pct = ""
                if baseline and baseline['gpu_peak']:
                    pct_change = ((data['gpu_peak'] - baseline['gpu_peak']) / baseline['gpu_peak']) * 100
                    gpu_pct = f" ({pct_change:+.0f}%)"
                # Value label - inside bar except for 90% loss
                label_text = f'{data["gpu_peak"]:.0f}{gpu_pct}'
                if loss == 90:
                    # Above bar for 90% loss (bars are shorter)
                    ax2.annotate(label_text,
                               xy=(gpu_x, data['gpu_peak'] + 1),
                               ha='center', va='bottom', fontsize=8,
                               fontweight='normal', color='black', rotation=90)
                else:
                    # Inside bar for other loss levels
                    ax2.annotate(label_text,
                               xy=(gpu_x, data['gpu_peak'] - 2),
                               ha='center', va='top', fontsize=8,
                               fontweight='normal', color='black', rotation=90)

    # Baseline reference lines
    if baseline:
        if baseline['mIoU']:
            ax1.axhline(y=baseline['mIoU'], color='black', linestyle='-',
                        linewidth=2, zorder=1)
            ax1.text(x_positions[-1] + 0.5, baseline['mIoU'] + 0.005,
                    f"Baseline mIoU: {baseline['mIoU']:.4f}",
                    fontsize=9, fontweight='bold', va='bottom', ha='right')
        if baseline['gpu_peak']:
            ax2.axhline(y=baseline['gpu_peak'], color='gray', linestyle='--',
                        linewidth=2, zorder=1)
            ax2.text(x_positions[-1] + 0.5, baseline['gpu_peak'] + 1,
                    f"Baseline GPU: {baseline['gpu_peak']:.0f} GB",
                    fontsize=9, fontweight='bold', va='bottom', ha='right', color='gray')

    # Configure left y-axis (mIoU)
    ax1.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('mIoU', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{l}%' for l in available_losses])
    ax1.set_ylim(0.45, 0.75)

    # Configure right y-axis (GPU Memory)
    ax2.set_ylabel('GPU Memory (GB)', fontweight='bold', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 100)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []

    # Method colors
    for method in available_methods:
        legend_elements.append(Patch(facecolor=METHOD_COLORS.get(method, '#888888'),
                                    edgecolor='black', linewidth=1, label=method))

    # Separator
    legend_elements.append(Patch(facecolor='white', edgecolor='white', label=''))

    # Bar type indicators
    legend_elements.append(Patch(facecolor='gray', edgecolor='black', linewidth=1, label='mIoU'))
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='gray', linewidth=2,
                                linestyle='--', label='GPU (GB)'))

    ax1.legend(handles=legend_elements, loc='upper center', ncol=len(available_methods) + 3,
               bbox_to_anchor=(0.5, -0.15), framealpha=0.95, fontsize=9)

    ax1.grid(True, axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    # Save
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(output_dir / f'ptv3_dual_axis.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved dual-axis figure")
    plt.close(fig)


def create_metric_grouped_figure(results, output_dir, show_plot=False):
    """
    Create a figure with two main sections separated by one vertical line:
    Left section: All mIoU bars (0% baseline, 30%, 50%, 70%, 90%)
    Right section: All GPU bars (0% baseline, 30%, 50%, 70%, 90%)
    With filled circle markers showing generalization (inference on original data)
    """
    aggregated = aggregate_by_method_loss(results)
    baseline = results['baseline']

    # Parse inference results for generalization markers
    inference_on_original = parse_inference_results(
        Path(output_dir).parent.parent / 'docs' / 'tables' / 'inference_on_original'
    )
    inference_on_original_agg = aggregate_inference_results(inference_on_original)

    # Determine available data
    all_losses = set()
    for method_data in aggregated.values():
        all_losses.update(method_data.keys())
    available_losses = sorted([l for l in LOSS_ORDER if l in all_losses])
    available_methods = [m for m in METHOD_ORDER if m in aggregated]

    if not available_losses:
        return

    # Use only actual loss levels (no 0% baseline bars)
    loss_levels = available_losses  # [30, 50, 70, 90]

    n_methods = len(available_methods)
    n_losses = len(loss_levels)

    # Bar positioning
    bar_width = 0.13
    loss_group_gap = 0.25  # Gap between loss level groups (constant for all)
    section_gap = 0.8  # Larger gap between mIoU and GPU sections

    output_dir = Path(output_dir)

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = ax1.twinx()

    import matplotlib.colors as mcolors

    # Calculate positions for mIoU section (left half)
    miou_positions = {}  # {loss: [x positions for each method]}
    current_x = 0
    for loss in loss_levels:
        miou_positions[loss] = []
        for i in range(n_methods):
            miou_positions[loss].append(current_x + i * bar_width)
        current_x += n_methods * bar_width + loss_group_gap

    miou_section_end = current_x - loss_group_gap

    # Separator position
    separator_x = miou_section_end + section_gap / 2

    # Calculate positions for GPU section (right half)
    gpu_positions = {}
    current_x = miou_section_end + section_gap
    for loss in loss_levels:
        gpu_positions[loss] = []
        for i in range(n_methods):
            gpu_positions[loss].append(current_x + i * bar_width)
        current_x += n_methods * bar_width + loss_group_gap

    # === Plot mIoU bars (left section) ===
    for loss in loss_levels:
        for i, method in enumerate(available_methods):
            if method not in aggregated or loss not in aggregated[method]:
                continue
            data = aggregated[method][loss]
            if data['mIoU'] is None:
                continue
            miou_val = data['mIoU']
            yerr = data.get('mIoU_std', 0) if data.get('mIoU_std', 0) > 0 else None

            x_pos = miou_positions[loss][i]

            ax1.bar(x_pos, miou_val, bar_width * 0.85,
                   color=METHOD_COLORS.get(method, '#888888'),
                   edgecolor='black', linewidth=0.5,
                   yerr=yerr, capsize=2, error_kw={'linewidth': 1},
                   zorder=3)

            # Calculate % drop from baseline
            miou_pct = ""
            if baseline and baseline['mIoU']:
                pct_change = ((miou_val - baseline['mIoU']) / baseline['mIoU']) * 100
                miou_pct = f" ({pct_change:+.0f}%)"

            # Add ± std if multiple seeds
            std_text = ""
            if data.get('mIoU_std', 0) > 0:
                std_text = f"±{data['mIoU_std']:.3f}"

            # Combined label text (inline format like dual_axis)
            label_text = f'{miou_val:.2f}{std_text}{miou_pct}'

            # Check if there's a generalization marker that might overlap
            has_gen_marker = False
            gen_miou = None
            if method in inference_on_original_agg and loss in inference_on_original_agg[method]:
                inf_data = inference_on_original_agg[method][loss]
                if inf_data['mIoU'] is not None:
                    has_gen_marker = True
                    gen_miou = inf_data['mIoU']

            # Value label - inside bar except for 90% loss (all black, normal weight)
            if loss == 90:
                label_y = miou_val + 0.005 + (yerr if yerr else 0)
                ax1.annotate(label_text,
                            xy=(x_pos, label_y),
                            ha='center', va='bottom', fontsize=9,
                            fontweight='normal', color='black', rotation=90)
            else:
                # Default: place label at top inside bar
                # If generalization marker would overlap, move label to middle
                y_min = 0.40  # y-axis minimum
                label_top = miou_val - 0.02
                label_bottom = miou_val - 0.07  # Approximate bottom of rotated text

                # Check if generalization marker is in the label area
                if has_gen_marker and gen_miou >= y_min and gen_miou > label_bottom and gen_miou < miou_val:
                    # Overlap detected - place label in middle of bar
                    label_y = (y_min + miou_val) / 2
                    ax1.annotate(label_text,
                                xy=(x_pos, label_y),
                                ha='center', va='center', fontsize=9,
                                fontweight='normal', color='black', rotation=90)
                else:
                    # No overlap - place label at top inside bar
                    label_y = miou_val - 0.02
                    ax1.annotate(label_text,
                                xy=(x_pos, label_y),
                                ha='center', va='top', fontsize=9,
                                fontweight='normal', color='black', rotation=90)

            # Add generalization marker (filled circle) if inference data available
            if method in inference_on_original_agg and loss in inference_on_original_agg[method]:
                inf_data = inference_on_original_agg[method][loss]
                if inf_data['mIoU'] is not None:
                    inf_miou = inf_data['mIoU']
                    inf_std = inf_data.get('mIoU_std', 0)
                    y_min = 0.40  # Match the y-axis minimum
                    baseline_miou = baseline['mIoU'] if baseline and baseline['mIoU'] else 0

                    # Build std text if multiple seeds
                    std_text = f'±{inf_std:.2f}' if inf_std > 0 else ''

                    if inf_miou >= y_min:
                        # Normal case: plot marker at actual position
                        ax1.scatter(x_pos, inf_miou, marker='o', s=50,
                                   c='black', edgecolors='black', linewidths=1,
                                   zorder=5, label='_nolegend_')
                        # If generalization mIoU >= baseline, show value on top of circle (2 decimals)
                        if inf_miou >= baseline_miou:
                            ax1.annotate(f'{inf_miou:.2f}',
                                        xy=(x_pos, inf_miou + 0.012),
                                        ha='center', va='bottom', fontsize=8,
                                        fontweight='bold', color='black', rotation=90)
                    else:
                        # Below visible range: plot marker at bottom with annotation
                        # Draw a small downward arrow at y_min
                        ax1.scatter(x_pos, y_min + 0.005, marker='v', s=40,
                                   c='black', edgecolors='black', linewidths=1,
                                   zorder=5, label='_nolegend_')
                        # Add text annotation ABOVE the marker showing the actual value with * prefix
                        ax1.annotate(f'*{inf_miou:.2f}',
                                    xy=(x_pos, y_min + 0.015),
                                    ha='center', va='bottom', fontsize=8,
                                    fontweight='normal', color='black', rotation=90)

    # === Plot GPU bars (right section) ===
    for loss in loss_levels:
        for i, method in enumerate(available_methods):
            if method not in aggregated or loss not in aggregated[method]:
                continue
            data = aggregated[method][loss]
            if data['gpu_peak'] is None:
                continue
            gpu_val = data['gpu_peak']

            x_pos = gpu_positions[loss][i]

            # Lighter color for GPU bars
            base_color = METHOD_COLORS.get(method, '#888888')
            rgb = mcolors.to_rgb(base_color)
            light_rgb = tuple(c + (1.0 - c) * 0.3 for c in rgb)

            ax2.bar(x_pos, gpu_val, bar_width * 0.85,
                   color=light_rgb,
                   edgecolor=base_color, linewidth=1.5,
                   linestyle='--', zorder=2)

            # Calculate % reduction from baseline
            gpu_pct = ""
            if baseline and baseline['gpu_peak']:
                pct_change = ((gpu_val - baseline['gpu_peak']) / baseline['gpu_peak']) * 100
                gpu_pct = f" ({pct_change:+.0f}%)"

            # Combined label text (inline format like dual_axis)
            label_text = f'{gpu_val:.0f}{gpu_pct}'

            # Value label - inside bar except for 90% loss (all black, normal weight)
            if loss == 90:
                ax2.annotate(label_text,
                            xy=(x_pos, gpu_val + 1),
                            ha='center', va='bottom', fontsize=9,
                            fontweight='normal', color='black', rotation=90)
            else:
                ax2.annotate(label_text,
                            xy=(x_pos, gpu_val - 3),
                            ha='center', va='top', fontsize=9,
                            fontweight='normal', color='black', rotation=90)

            # NOTE: Inference GPU memory markers removed - data is not reliable
            # The inference script captures GPU memory after inference completes,
            # not the actual peak during inference

    # Calculate x-axis limits - need to set before drawing lines
    x_left_edge = min(miou_positions[loss_levels[0]]) - bar_width
    x_right_edge = max(gpu_positions[loss_levels[-1]]) + bar_width
    ax1.set_xlim(x_left_edge, x_right_edge)

    # Draw vertical separator between mIoU and GPU sections (full height)
    ax1.axvline(x=separator_x, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=5)

    # Baseline reference lines - only in their respective sections, touching the y-axis and separator
    if baseline:
        # mIoU baseline - from y-axis (left edge) to separator
        if baseline['mIoU']:
            ax1.hlines(y=baseline['mIoU'], xmin=x_left_edge, xmax=separator_x,
                      color='black', linestyle='--', linewidth=2, zorder=1, alpha=0.7)
            # Add baseline value text near separator
            ax1.text(separator_x - 0.1, baseline['mIoU'] + 0.008,
                    f"Baseline mIoU: {baseline['mIoU']:.3f}",
                    fontsize=10, fontweight='bold', va='bottom', ha='right', color='black')

        # GPU baseline - from separator to right edge (y-axis on right)
        if baseline['gpu_peak']:
            ax2.hlines(y=baseline['gpu_peak'], xmin=separator_x, xmax=x_right_edge,
                      color='black', linestyle='--', linewidth=2, zorder=1, alpha=0.7)
            # Add baseline value text near separator
            ax2.text(separator_x + 0.1, baseline['gpu_peak'] + 1,
                    f"Baseline GPU: {baseline['gpu_peak']:.0f} GB",
                    fontsize=10, fontweight='bold', va='bottom', ha='left', color='black')

    # Configure axes
    ax1.set_ylabel('mIoU', fontweight='bold', fontsize=14)
    ax2.set_ylabel('GPU Memory (GB)', fontweight='bold', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # X-axis labels - loss level under each group
    all_tick_positions = []
    all_tick_labels = []

    for loss in loss_levels:
        # mIoU section tick
        miou_center = np.mean(miou_positions[loss])
        all_tick_positions.append(miou_center)
        all_tick_labels.append(f'{loss}%')

        # GPU section tick
        gpu_center = np.mean(gpu_positions[loss])
        all_tick_positions.append(gpu_center)
        all_tick_labels.append(f'{loss}%')

    ax1.set_xticks(all_tick_positions)
    ax1.set_xticklabels(all_tick_labels, fontsize=12)

    ax1.set_ylim(0.40, 0.75)
    ax2.set_ylim(0, 100)

    # Calculate section centers for legend positioning
    miou_section_center = np.mean([np.mean(miou_positions[l]) for l in loss_levels])
    gpu_section_center = np.mean([np.mean(gpu_positions[l]) for l in loss_levels])

    # Create separate legends for each section
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Legend elements for mIoU section (methods + generalization marker)
    miou_legend_elements = []
    for method in available_methods:
        miou_legend_elements.append(Patch(facecolor=METHOD_COLORS.get(method, '#888888'),
                                         edgecolor='black', linewidth=1, label=method))
    # Add generalization circle marker to legend
    miou_legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                       markeredgecolor='black', markersize=8, linewidth=0,
                                       label='Test → Original Data'))

    # Legend elements for GPU section (methods only - lighter colors with dotted edges)
    gpu_legend_elements = []
    for method in available_methods:
        base_color = METHOD_COLORS.get(method, '#888888')
        rgb = mcolors.to_rgb(base_color)
        light_rgb = tuple(c + (1.0 - c) * 0.3 for c in rgb)
        gpu_legend_elements.append(Patch(facecolor=light_rgb,
                                        edgecolor=base_color, linewidth=1.5,
                                        linestyle='--', label=method))

    # Calculate legend positions (center of each section in axes coordinates 0-1)
    miou_center_data = (x_left_edge + separator_x) / 2
    gpu_center_data = (separator_x + x_right_edge) / 2
    miou_legend_x = (miou_center_data - x_left_edge) / (x_right_edge - x_left_edge)
    gpu_legend_x = (gpu_center_data - x_left_edge) / (x_right_edge - x_left_edge)

    # mIoU legend in mIoU section
    miou_legend = ax1.legend(handles=miou_legend_elements, loc='upper center',
                            ncol=len(available_methods) + 1,
                            bbox_to_anchor=(miou_legend_x, 0.99),
                            framealpha=0.95, fontsize=9,
                            columnspacing=1.2, handletextpad=0.4, handlelength=1.5)
    ax1.add_artist(miou_legend)

    # GPU legend in GPU section
    ax2.legend(handles=gpu_legend_elements, loc='upper center',
              ncol=len(available_methods),
              bbox_to_anchor=(gpu_legend_x, 0.99),
              framealpha=0.95, fontsize=10,
              columnspacing=1.2, handletextpad=0.4, handlelength=1.5)

    ax1.grid(True, axis='y', alpha=0.3, zorder=0)

    # Add x-axis label
    ax1.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(output_dir / f'01_ptv3_metric_grouped.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 01_ptv3_metric_grouped figure")

    if show_plot:
        plt.show()
    plt.close(fig)


def create_spatial_distribution_analysis(results, output_dir, show_plot=False):
    """
    Create a figure with three heatmaps showing:
    - (A) Training mIoU (Method vs Loss Level)
    - (B) Generalization mIoU (Method vs Loss Level) - trained model on original data
    - (C) GPU Memory (Method vs Loss Level)
    """
    aggregated = aggregate_by_method_loss(results)

    output_dir = Path(output_dir)

    # Parse inference results for generalization data
    inference_on_original = parse_inference_results(
        Path(output_dir).parent / 'tables' / 'inference_on_original'
    )
    inference_on_original_agg = aggregate_inference_results(inference_on_original)

    # Spatial spread ordering for consistent method display
    # Includes all methods: classical geometric subsampling + DEPOCO (learned compression)
    SPATIAL_SPREAD = {
        'DBSCAN': 1,
        'IDIS': 2,
        'VB': 3,
        'RS': 4,
        'SB': 5,
        'FPS': 6,
        'DEPOCO': 7,
    }

    # Methods to include in this heatmap (all methods including DEPOCO)
    HEATMAP_METHODS = [m for m in METHOD_ORDER if m in SPATIAL_SPREAD]

    # Collect data for ALL loss levels
    all_data = []
    for method in HEATMAP_METHODS:
        if method not in aggregated:
            continue
        for loss in LOSS_ORDER:
            if loss not in aggregated[method]:
                continue
            method_data = aggregated[method][loss]
            if method_data['mIoU'] is not None:
                all_data.append({
                    'method': method,
                    'loss': loss,
                    'miou': method_data['mIoU'],
                    'gpu': method_data.get('gpu_peak'),
                    'spread': SPATIAL_SPREAD.get(method, 3),
                })

    if not all_data:
        print("No data available")
        return

    # Sort methods by spatial spread for consistent ordering
    methods_sorted = sorted([m for m in HEATMAP_METHODS if m in aggregated],
                           key=lambda x: SPATIAL_SPREAD.get(x, 3))

    # Create figure with 1x3 subplots (three heatmaps side by side)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Prepare data for heatmaps
    methods_for_heatmap = [m for m in methods_sorted if m in aggregated]
    losses_for_heatmap = sorted(set(d['loss'] for d in all_data))

    # === Plot 1: Heatmap - Training mIoU ===
    training_matrix = np.full((len(methods_for_heatmap), len(losses_for_heatmap)), np.nan)
    for i, method in enumerate(methods_for_heatmap):
        for j, loss in enumerate(losses_for_heatmap):
            if method in aggregated and loss in aggregated[method]:
                if aggregated[method][loss]['mIoU'] is not None:
                    training_matrix[i, j] = aggregated[method][loss]['mIoU']

    im1 = ax1.imshow(training_matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.7)

    # Add text annotations
    for i in range(len(methods_for_heatmap)):
        for j in range(len(losses_for_heatmap)):
            if not np.isnan(training_matrix[i, j]):
                ax1.text(j, i, f'{training_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')

    ax1.set_xticks(range(len(losses_for_heatmap)))
    ax1.set_xticklabels([f'{l}%' for l in losses_for_heatmap], fontsize=11)
    ax1.set_yticks(range(len(methods_for_heatmap)))
    ax1.set_yticklabels([f'{m}' for m in methods_for_heatmap], fontsize=10)
    ax1.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Method', fontweight='bold', fontsize=12)
    ax1.set_title('(A) mIoU (Test → Subsampled Data)', fontsize=13, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('mIoU', fontsize=10)

    # === Plot 2: Heatmap - Generalization mIoU ===
    generalization_matrix = np.full((len(methods_for_heatmap), len(losses_for_heatmap)), np.nan)
    for i, method in enumerate(methods_for_heatmap):
        for j, loss in enumerate(losses_for_heatmap):
            if method in inference_on_original_agg and loss in inference_on_original_agg[method]:
                if inference_on_original_agg[method][loss]['mIoU'] is not None:
                    generalization_matrix[i, j] = inference_on_original_agg[method][loss]['mIoU']

    im2 = ax2.imshow(generalization_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.7)

    # Add text annotations
    for i in range(len(methods_for_heatmap)):
        for j in range(len(losses_for_heatmap)):
            if not np.isnan(generalization_matrix[i, j]):
                ax2.text(j, i, f'{generalization_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')

    ax2.set_xticks(range(len(losses_for_heatmap)))
    ax2.set_xticklabels([f'{l}%' for l in losses_for_heatmap], fontsize=11)
    ax2.set_yticks(range(len(methods_for_heatmap)))
    ax2.set_yticklabels([f'{m}' for m in methods_for_heatmap], fontsize=10)
    ax2.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Method', fontweight='bold', fontsize=12)
    ax2.set_title('(B) mIoU (Test → Original Data)', fontsize=13, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('mIoU', fontsize=10)

    # === Plot 3: Heatmap - GPU Memory ===
    gpu_matrix = np.full((len(methods_for_heatmap), len(losses_for_heatmap)), np.nan)
    for i, method in enumerate(methods_for_heatmap):
        for j, loss in enumerate(losses_for_heatmap):
            if method in aggregated and loss in aggregated[method]:
                if aggregated[method][loss]['gpu_peak'] is not None:
                    gpu_matrix[i, j] = aggregated[method][loss]['gpu_peak']

    im3 = ax3.imshow(gpu_matrix, cmap='RdYlGn_r', aspect='auto')

    # Add text annotations
    for i in range(len(methods_for_heatmap)):
        for j in range(len(losses_for_heatmap)):
            if not np.isnan(gpu_matrix[i, j]):
                ax3.text(j, i, f'{gpu_matrix[i, j]:.0f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')

    ax3.set_xticks(range(len(losses_for_heatmap)))
    ax3.set_xticklabels([f'{l}%' for l in losses_for_heatmap], fontsize=11)
    ax3.set_yticks(range(len(methods_for_heatmap)))
    ax3.set_yticklabels([f'{m}' for m in methods_for_heatmap], fontsize=10)
    ax3.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Method', fontweight='bold', fontsize=12)
    ax3.set_title('(C) Training GPU Memory (GB)', fontsize=13, fontweight='bold')

    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('GPU (GB)', fontsize=10)

    plt.tight_layout()

    # Save
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(output_dir / f'02_ptv3_spatial_distribution_analysis.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 02_ptv3_spatial_distribution_analysis figure")

    if show_plot:
        plt.show()
    plt.close(fig)


def create_ranking_bump_chart(results, output_dir, inference_data=None, show_plot=False):
    """
    Create a ranking bump chart showing how method rankings change
    across different loss levels based on mIoU performance.
    Includes both training (solid) and inference/generalization (dotted) rankings.
    """
    aggregated = aggregate_by_method_loss(results)
    output_dir = Path(output_dir)

    # Collect mIoU data for ranking (training)
    # Structure: {loss: {method: miou}}
    loss_method_miou = {}
    for method in METHOD_ORDER:
        if method not in aggregated:
            continue
        for loss in LOSS_ORDER:
            if loss not in aggregated[method]:
                continue
            miou = aggregated[method][loss]['mIoU']
            if miou is not None:
                if loss not in loss_method_miou:
                    loss_method_miou[loss] = {}
                loss_method_miou[loss][method] = miou

    if not loss_method_miou:
        print("No data available for ranking bump chart")
        return

    # Calculate rankings at each loss level (1 = best mIoU)
    # Structure: {method: {loss: rank}}
    method_rankings = {m: {} for m in METHOD_ORDER}
    method_miou_at_loss = {m: {} for m in METHOD_ORDER}

    for loss in sorted(loss_method_miou.keys()):
        # Sort methods by mIoU (descending - higher is better)
        sorted_methods = sorted(loss_method_miou[loss].items(),
                               key=lambda x: x[1], reverse=True)
        for rank, (method, miou) in enumerate(sorted_methods, 1):
            method_rankings[method][loss] = rank
            method_miou_at_loss[method][loss] = miou

    # Collect inference data rankings if available
    inf_method_rankings = {m: {} for m in METHOD_ORDER}
    inf_method_miou_at_loss = {m: {} for m in METHOD_ORDER}
    has_inference = False

    if inference_data:
        # Structure: {loss: {method: miou}} for inference
        inf_loss_method_miou = {}
        for method in inference_data:
            for loss in inference_data[method]:
                if inference_data[method][loss]['mIoU'] is not None:
                    if loss not in inf_loss_method_miou:
                        inf_loss_method_miou[loss] = {}
                    inf_loss_method_miou[loss][method] = inference_data[method][loss]['mIoU']

        # Calculate inference rankings
        for loss in sorted(inf_loss_method_miou.keys()):
            sorted_methods = sorted(inf_loss_method_miou[loss].items(),
                                   key=lambda x: x[1], reverse=True)
            for rank, (method, miou) in enumerate(sorted_methods, 1):
                inf_method_rankings[method][loss] = rank
                inf_method_miou_at_loss[method][loss] = miou
                has_inference = True

    # Create the bump chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get available loss levels
    available_losses = sorted(loss_method_miou.keys())
    x_positions = {loss: i for i, loss in enumerate(available_losses)}

    baseline_miou = results['baseline']['mIoU'] if results['baseline'] else 0.6721

    # For each method, collect all data points (both training and inference)
    # and plot them with appropriate markers and connecting lines
    for method in METHOD_ORDER:
        if method not in method_rankings:
            continue

        losses_with_data = sorted(method_rankings[method].keys())
        if not losses_with_data:
            continue

        color = METHOD_COLORS.get(method, '#888888')

        # Collect all points for this method (for drawing connecting lines)
        all_x_vals = []
        all_y_vals = []

        # Separate training-only and inference points
        train_only_points = []  # [(x, y, miou), ...]
        inference_points = []   # [(x, y, miou), ...]

        for loss in losses_with_data:
            if loss not in x_positions:
                continue

            x_pos = x_positions[loss]
            # Check if we have inference data for this loss level
            has_inf_for_this = (inference_data and
                               method in inference_data and
                               loss in inference_data[method] and
                               inference_data[method][loss]['mIoU'] is not None)

            if has_inf_for_this:
                # Use inference ranking and mIoU
                y_pos = inf_method_rankings[method][loss]
                miou = inf_method_miou_at_loss[method][loss]
                inference_points.append((x_pos, y_pos, miou))
                all_x_vals.append(x_pos)
                all_y_vals.append(y_pos)
            else:
                # Use training ranking and mIoU
                y_pos = method_rankings[method][loss]
                miou = method_miou_at_loss[method][loss]
                train_only_points.append((x_pos, y_pos, miou))
                all_x_vals.append(x_pos)
                all_y_vals.append(y_pos)

        # Draw connecting line through ALL points (both training and inference)
        if len(all_x_vals) > 1:
            # Sort by x position to ensure proper line drawing
            sorted_points = sorted(zip(all_x_vals, all_y_vals), key=lambda p: p[0])
            sorted_x = [p[0] for p in sorted_points]
            sorted_y = [p[1] for p in sorted_points]
            ax.plot(sorted_x, sorted_y, '-', color=color, linewidth=3, alpha=0.8, zorder=2)

        # Draw square markers for training-only points
        if train_only_points:
            train_x = [p[0] for p in train_only_points]
            train_y = [p[1] for p in train_only_points]
            ax.scatter(train_x, train_y, c=color, s=200, edgecolors='black',
                      linewidths=2, zorder=4, marker='s')

            # Add mIoU annotations for training data
            for x, y, miou in train_only_points:
                offset_y = -0.3  # Above the marker
                font_weight = 'bold' if miou >= baseline_miou else 'normal'
                ax.annotate(f'{miou:.2f}', (x, y + offset_y), fontsize=10,
                           va='bottom', ha='center', fontweight=font_weight, color='black', zorder=5)

        # Draw circle markers for inference points
        if inference_points:
            inf_x = [p[0] for p in inference_points]
            inf_y = [p[1] for p in inference_points]
            ax.scatter(inf_x, inf_y, c=color, s=200, edgecolors='black',
                      linewidths=2, zorder=4, marker='o')

            # Add mIoU annotations for inference data
            for x, y, miou in inference_points:
                offset_y = -0.3  # Above the marker
                font_weight = 'bold' if miou >= baseline_miou else 'normal'
                ax.annotate(f'{miou:.2f}', (x, y + offset_y), fontsize=10,
                           va='bottom', ha='center', fontweight=font_weight, color='black', zorder=5)

    # Customize the plot
    ax.set_xlim(-0.5, len(available_losses) - 0.5 + 0.8)  # Extra space for annotations

    # Y-axis: Rank (1 at top, higher ranks at bottom)
    max_methods = max(len(loss_method_miou[loss]) for loss in available_losses)
    ax.set_ylim(max_methods + 0.5, 0.5)  # Inverted: rank 1 at top
    ax.set_yticks(range(1, max_methods + 1))
    ax.set_yticklabels([f'Rank {i}' for i in range(1, max_methods + 1)], fontsize=13)

    # X-axis: Loss levels
    ax.set_xticks(range(len(available_losses)))
    ax.set_xticklabels([f'{loss}%' for loss in available_losses], fontsize=13, fontweight='bold')

    ax.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=15)
    ax.set_ylabel('mIoU Ranking (1 = Best)', fontweight='bold', fontsize=15)

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = []

    # Method legends (show with circle marker - primary/original data style)
    for method in METHOD_ORDER:
        if method in aggregated and method_rankings[method]:
            legend_elements.append(Line2D([0], [0], marker='o', color=METHOD_COLORS.get(method, '#888888'),
                                         markerfacecolor=METHOD_COLORS.get(method, '#888888'),
                                         markeredgecolor='black', markersize=10, linewidth=3,
                                         linestyle='-', label=method))

    # Add line style explanation
    # Solid line with circle = Test → Original Data
    legend_elements.append(Line2D([0], [0], marker='o', color='black', linestyle='-',
                                 markerfacecolor='black', markeredgecolor='black',
                                 markersize=10, linewidth=3, label='Test → Original Data'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=4, fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(output_dir / f'03_ptv3_ranking_bump_chart.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 03_ptv3_ranking_bump_chart figure")

    if show_plot:
        plt.show()
    plt.close(fig)


def print_summary(results):
    """Print a summary of parsed results."""
    print("\n" + "="*60)
    print("PARSED RESULTS SUMMARY")
    print("="*60)

    if results['baseline']:
        print(f"\nBaseline (0% loss):")
        miou = results['baseline']['mIoU']
        gpu = results['baseline']['gpu_peak']
        print(f"  mIoU: {miou:.4f}" if miou else "  mIoU: N/A")
        print(f"  GPU Peak: {gpu:.1f} GB" if gpu else "  GPU Peak: N/A")

    print(f"\nExperiments found: {len(results['experiments'])}")

    # Group by method
    by_method = defaultdict(list)
    for (method, loss, seed), data in results['experiments'].items():
        by_method[method].append((loss, seed, data))

    for method in sorted(by_method.keys()):
        print(f"\n{method}:")
        for loss, seed, data in sorted(by_method[method]):
            miou = data['mIoU']
            gpu = data['gpu_peak']
            miou_str = f"{miou:.4f}" if miou is not None else "N/A"
            gpu_str = f"{gpu:.1f}" if gpu is not None else "N/A"
            print(f"  Loss {loss}%, Seed {seed}: mIoU={miou_str}, GPU={gpu_str} GB")


def parse_inference_results(inference_dir, inference_type='inference_on_original'):
    """
    Parse inference results from metrics.txt files.

    Args:
        inference_dir: Path to inference tables directory (e.g., docs/tables/inference_on_original)
        inference_type: 'inference' or 'inference_on_original'

    Returns:
        dict: {(method, loss, seed): {'mIoU': float, 'mAcc': float, 'allAcc': float}}
    """
    results = {}

    inference_path = Path(inference_dir)
    if not inference_path.exists():
        print(f"Warning: Inference directory not found: {inference_path}")
        return results

    # Parse each metrics.txt file
    for metrics_file in inference_path.glob('*_metrics.txt'):
        exp_name = metrics_file.stem.replace('_metrics', '')

        # Parse experiment name
        parts = exp_name.split('_')
        method = parts[0]

        # Apply name mapping
        if method in METHOD_NAME_MAP:
            method = METHOD_NAME_MAP[method]

        loss = None
        seed = 1
        r_value = None

        for part in parts:
            if part.startswith('loss'):
                loss = int(part.replace('loss', ''))
            elif part.startswith('seed'):
                seed = int(part.replace('seed', ''))
            elif part.startswith('R') and part[1:].isdigit():
                # Parse R-value for IDIS (e.g., R5, R10, R15, R20)
                r_value = int(part[1:])

        # For IDIS with R-value, include R in method name for proper aggregation
        if method == 'IDIS' and r_value is not None:
            method = f'IDIS_R{r_value}'

        if loss is None:
            continue

        # Parse metrics file for mIoU and GPU memory
        miou = None
        macc = None
        allacc = None
        gpu_peak = None

        with open(metrics_file, 'r') as f:
            content = f.read()
            # Pattern: mIoU:    0.6658
            miou_match = re.search(r'mIoU:\s*([\d.]+)', content)
            macc_match = re.search(r'mAcc:\s*([\d.]+)', content)
            allacc_match = re.search(r'allAcc:\s*([\d.]+)', content)
            # Pattern: Peak: 9173 MB
            gpu_match = re.search(r'Peak:\s*(\d+)\s*MB', content)

            if miou_match:
                miou = float(miou_match.group(1))
            if macc_match:
                macc = float(macc_match.group(1))
            if allacc_match:
                allacc = float(allacc_match.group(1))
            if gpu_match:
                gpu_peak = float(gpu_match.group(1)) / 1024  # Convert MB to GB

        if miou is not None:
            key = (method, loss, seed)
            results[key] = {
                'mIoU': miou,
                'mAcc': macc,
                'allAcc': allacc,
                'gpu_peak': gpu_peak,
                'exp_name': exp_name
            }
            print(f"  Parsed inference: {exp_name} - Method={method}, Loss={loss}%, mIoU={miou:.4f}")

    return results


def aggregate_inference_results(results):
    """
    Aggregate inference results by method and loss level, averaging across seeds.

    Returns:
        dict: {method: {loss_level: {'mIoU': float, 'mIoU_std': float, 'gpu_peak': float, 'gpu_std': float}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: {'mIoUs': [], 'gpus': []}))

    for (method, loss, seed), data in results.items():
        # Map IDIS R-value variants to main IDIS for aggregation
        if method.startswith('IDIS_R'):
            method_key = 'IDIS'
        else:
            method_key = method

        if data['mIoU'] is not None:
            aggregated[method_key][loss]['mIoUs'].append(data['mIoU'])
        if data.get('gpu_peak') is not None:
            aggregated[method_key][loss]['gpus'].append(data['gpu_peak'])

    # Calculate mean and std
    final = {}
    for method, losses in aggregated.items():
        final[method] = {}
        for loss, values in losses.items():
            mious = values['mIoUs']
            gpus = values['gpus']
            final[method][loss] = {
                'mIoU': np.mean(mious) if mious else None,
                'mIoU_std': np.std(mious) if len(mious) > 1 else 0,
                'gpu_peak': np.mean(gpus) if gpus else None,
                'gpu_std': np.std(gpus) if len(gpus) > 1 else 0,
                'n_seeds': len(mious)
            }

    return final


def create_combined_training_inference_figure(training_results, output_dir, show_plot=False):
    """
    Create a figure similar to 01_ptv3_metric_grouped but with inference markers:
    - Left section: mIoU bars (training) with diamond markers showing inference on original data
    - Right section: GPU Memory bars (training)

    Diamond markers (◆) on mIoU bars show inference performance (trained model → original data)
    This shows the "generalization gap" between training and inference.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Aggregate training results
    training_aggregated = aggregate_by_method_loss(training_results)
    baseline = training_results['baseline']

    # Parse inference results (only inference_on_original - trained models on original data)
    base_dir = PROJECT_ROOT
    inference_on_original_dir = base_dir / 'PTv3' / 'SemanticKITTI' / 'outputs' / 'inference_on_original'

    print("\nParsing inference_on_original results...")
    inference_on_original = parse_inference_results(inference_on_original_dir, 'inference_on_original')
    inference_on_original_agg = aggregate_inference_results(inference_on_original)

    # Determine available data from training
    all_losses = set()
    for method_data in training_aggregated.values():
        all_losses.update(method_data.keys())
    available_losses = sorted([l for l in LOSS_ORDER if l in all_losses])
    available_methods = [m for m in METHOD_ORDER if m in training_aggregated]

    if not available_losses:
        print("No training data found!")
        return

    output_dir = Path(output_dir)

    # Create figure with same layout as 01_ptv3_metric_grouped
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax2 = ax1.twinx()

    n_methods = len(available_methods)
    n_losses = len(available_losses)

    # Bar positioning - same as figure 01
    bar_width = 0.13
    loss_group_gap = 0.15
    section_gap = 0.8

    # Calculate positions for mIoU section (left half)
    miou_positions = {}
    current_x = 0
    for loss in available_losses:
        miou_positions[loss] = []
        for i in range(n_methods):
            miou_positions[loss].append(current_x + i * bar_width)
        current_x += n_methods * bar_width + loss_group_gap

    miou_section_end = current_x - loss_group_gap
    separator_x = miou_section_end + section_gap / 2

    # Calculate positions for GPU section (right half)
    gpu_positions = {}
    current_x = miou_section_end + section_gap
    for loss in available_losses:
        gpu_positions[loss] = []
        for i in range(n_methods):
            gpu_positions[loss].append(current_x + i * bar_width)
        current_x += n_methods * bar_width + loss_group_gap

    # === Plot mIoU bars (left section) with inference markers ===
    for loss in available_losses:
        for i, method in enumerate(available_methods):
            if method not in training_aggregated or loss not in training_aggregated[method]:
                continue
            data = training_aggregated[method][loss]
            if data['mIoU'] is None:
                continue

            x_pos = miou_positions[loss][i]
            miou_val = data['mIoU']
            yerr = data.get('mIoU_std', 0) if data.get('mIoU_std', 0) > 0 else None

            # Plot training mIoU bar
            ax1.bar(x_pos, miou_val, bar_width * 0.85,
                   color=METHOD_COLORS.get(method, '#888888'),
                   edgecolor='black', linewidth=0.5,
                   yerr=yerr, capsize=2, error_kw={'linewidth': 1},
                   zorder=3)

            # Training value label
            if loss == 90:
                label_y = miou_val + 0.005 + (yerr if yerr else 0)
                ax1.annotate(f'{miou_val:.2f}',
                            xy=(x_pos, label_y),
                            ha='center', va='bottom', fontsize=8,
                            fontweight='normal', color='black', rotation=90)
            else:
                label_y = miou_val - 0.02
                ax1.annotate(f'{miou_val:.2f}',
                            xy=(x_pos, label_y),
                            ha='center', va='top', fontsize=8,
                            fontweight='normal', color='black', rotation=90)

            # Add inference marker (filled circle) if data available
            if method in inference_on_original_agg and loss in inference_on_original_agg[method]:
                inf_data = inference_on_original_agg[method][loss]
                if inf_data['mIoU'] is not None:
                    inf_miou = inf_data['mIoU']
                    # Filled black circle marker at inference mIoU level
                    ax1.scatter(x_pos, inf_miou, marker='o', s=50,
                               c='black', edgecolors='black', linewidths=1,
                               zorder=5)

    # === Plot GPU bars (right section) ===
    for loss in available_losses:
        for i, method in enumerate(available_methods):
            if method not in training_aggregated or loss not in training_aggregated[method]:
                continue
            data = training_aggregated[method][loss]
            if data['gpu_peak'] is None:
                continue

            x_pos = gpu_positions[loss][i]
            gpu_val = data['gpu_peak']

            # Lighter color for GPU bars
            base_color = METHOD_COLORS.get(method, '#888888')
            rgb = mcolors.to_rgb(base_color)
            light_rgb = tuple(c + (1.0 - c) * 0.3 for c in rgb)

            ax2.bar(x_pos, gpu_val, bar_width * 0.85,
                   color=light_rgb,
                   edgecolor=base_color, linewidth=1.5,
                   linestyle='--', zorder=2)

            # Value label
            if loss == 90:
                ax2.annotate(f'{gpu_val:.0f}',
                            xy=(x_pos, gpu_val + 1),
                            ha='center', va='bottom', fontsize=9,
                            fontweight='normal', color='black', rotation=90)
            else:
                ax2.annotate(f'{gpu_val:.0f}',
                            xy=(x_pos, gpu_val - 3),
                            ha='center', va='top', fontsize=9,
                            fontweight='normal', color='black', rotation=90)

    # Calculate x-axis limits
    x_left_edge = min(miou_positions[available_losses[0]]) - bar_width
    x_right_edge = max(gpu_positions[available_losses[-1]]) + bar_width
    ax1.set_xlim(x_left_edge, x_right_edge)

    # Draw vertical separator
    ax1.axvline(x=separator_x, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=5)

    # Baseline reference lines
    if baseline:
        if baseline['mIoU']:
            ax1.hlines(y=baseline['mIoU'], xmin=x_left_edge, xmax=separator_x,
                      color='black', linestyle='--', linewidth=2, zorder=1, alpha=0.7)
            ax1.text(separator_x - 0.1, baseline['mIoU'] + 0.008,
                    f"Baseline mIoU: {baseline['mIoU']:.4f}",
                    fontsize=11, fontweight='bold', va='bottom', ha='right', color='black')

        if baseline['gpu_peak']:
            ax2.hlines(y=baseline['gpu_peak'], xmin=separator_x, xmax=x_right_edge,
                      color='black', linestyle='--', linewidth=2, zorder=1, alpha=0.7)
            ax2.text(separator_x + 0.1, baseline['gpu_peak'] + 1,
                    f"Baseline GPU: {baseline['gpu_peak']:.0f} GB",
                    fontsize=11, fontweight='bold', va='bottom', ha='left', color='black')

    # Configure axes
    ax1.set_ylabel('mIoU', fontweight='bold', fontsize=14)
    ax2.set_ylabel('GPU Memory (GB)', fontweight='bold', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # X-axis labels
    all_tick_positions = []
    all_tick_labels = []
    for loss in available_losses:
        miou_center = np.mean(miou_positions[loss])
        all_tick_positions.append(miou_center)
        all_tick_labels.append(f'{loss}%')
        gpu_center = np.mean(gpu_positions[loss])
        all_tick_positions.append(gpu_center)
        all_tick_labels.append(f'{loss}%')

    ax1.set_xticks(all_tick_positions)
    ax1.set_xticklabels(all_tick_labels, fontsize=12)
    ax1.set_xlabel('Loss Level (%)', fontweight='bold', fontsize=14)

    ax1.set_ylim(0.25, 0.75)
    ax2.set_ylim(0, 100)

    # Section labels at top
    miou_section_center = np.mean([np.mean(miou_positions[l]) for l in available_losses])
    gpu_section_center = np.mean([np.mean(gpu_positions[l]) for l in available_losses])

    ax1.text(miou_section_center, 0.76, 'mIoU (Training + Inference)', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='black')
    ax1.text(gpu_section_center, 0.76, 'GPU Memory (GB)', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='black')

    # Create legends
    # mIoU legend (methods + inference marker explanation)
    miou_legend_elements = []
    for method in available_methods:
        miou_legend_elements.append(Patch(facecolor=METHOD_COLORS.get(method, '#888888'),
                                         edgecolor='black', linewidth=1, label=method))
    # Add inference star marker to legend
    miou_legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                       markeredgecolor='black', markersize=8, linewidth=0,
                                       label='Test → Original Data'))

    # GPU legend (methods with lighter colors)
    gpu_legend_elements = []
    for method in available_methods:
        base_color = METHOD_COLORS.get(method, '#888888')
        rgb = mcolors.to_rgb(base_color)
        light_rgb = tuple(c + (1.0 - c) * 0.3 for c in rgb)
        gpu_legend_elements.append(Patch(facecolor=light_rgb,
                                        edgecolor=base_color, linewidth=1.5,
                                        linestyle='--', label=method))

    # Calculate legend positions
    miou_center_data = (x_left_edge + separator_x) / 2
    gpu_center_data = (separator_x + x_right_edge) / 2
    miou_legend_x = (miou_center_data - x_left_edge) / (x_right_edge - x_left_edge)
    gpu_legend_x = (gpu_center_data - x_left_edge) / (x_right_edge - x_left_edge)

    # Place legends
    miou_legend = ax1.legend(handles=miou_legend_elements, loc='upper center',
                            ncol=len(available_methods) + 1,
                            bbox_to_anchor=(miou_legend_x, 0.99),
                            framealpha=0.95, fontsize=9)
    ax1.add_artist(miou_legend)

    ax2.legend(handles=gpu_legend_elements, loc='upper center',
              ncol=len(available_methods),
              bbox_to_anchor=(gpu_legend_x, 0.99),
              framealpha=0.95, fontsize=9)

    ax1.grid(True, axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()

    # Save
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(output_dir / f'00_ptv3_training_inference_combined.{fmt}', format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 00_ptv3_training_inference_combined figure")

    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate academic figures from training metrics'
    )
    parser.add_argument(
        '--input', '-i',
        default=str(DEFAULT_INPUT),
        help='Path to all_experiments_detailed.txt'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT),
        help='Output directory for figures'
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show plots interactively'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of parsed results'
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Reading results from: {args.input}")
    results = parse_experiments_file(args.input)

    if args.summary:
        print_summary(results)

    print(f"\nGenerating figures...")

    # Parse inference on original data for bump chart
    inference_on_original_dir = Path(args.output).parent / 'tables' / 'inference_on_original'
    inference_on_original = parse_inference_results(inference_on_original_dir, 'inference_on_original')
    inference_on_original_agg = aggregate_inference_results(inference_on_original)

    # Generate all figures:
    # 0. Combined training + inference figure (NEW)
    create_combined_training_inference_figure(results, args.output, args.show)
    # 1. ptv3_metric_grouped (via create_stacked_figures -> create_metric_grouped_figure)
    # 2. ptv3_spatial_distribution_analysis (via create_stacked_figures -> create_spatial_distribution_analysis)
    # 3. ptv3_ranking_bump_chart
    create_stacked_figures(results, args.output, args.show)
    create_ranking_bump_chart(results, args.output, inference_on_original_agg, args.show)

    print(f"\nDone! Figures saved to: {args.output}")
    print("Generated figures:")
    print("  - 00_ptv3_training_inference_combined.{png,svg,pdf} (NEW)")
    print("  - 01_ptv3_metric_grouped.{png,svg,pdf}")
    print("  - 02_ptv3_spatial_distribution_analysis.{png,svg,pdf}")
    print("  - 03_ptv3_ranking_bump_chart.{png,svg,pdf}")


if __name__ == '__main__':
    main()
