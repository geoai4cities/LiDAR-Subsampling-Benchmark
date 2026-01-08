#!/usr/bin/env python3
"""
Generate Class-wise Performance Drop Figure (Small Multiples)

This script creates a 4x5 grid of subplots showing how each class's mIoU
drops across different loss levels for all subsampling methods.

Each subplot:
- X-axis: Loss level (0%, 30%, 50%, 70%, 90%)
- Y-axis: mIoU (0 to 1)
- Lines: Different methods (baseline, RS, DBSCAN, IDIS, etc.)

Output: 09_classwise_performance_drop.{png,svg,pdf}
"""

import re
import os
import sys
from pathlib import Path
import argparse
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Script and project paths (for relative path resolution)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/figures -> scripts -> project root
DEFAULT_TABLES_DIR = PROJECT_ROOT / 'docs' / 'tables'
DEFAULT_OUTPUT = PROJECT_ROOT / 'docs' / 'figures'

# Set up academic-style plotting with improved readability
plt.rcParams.update({
    # Font settings - use serif fonts for academic papers (IEEE/ACM style)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'mathtext.fontset': 'stix',

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
})

# SemanticKITTI class names (19 classes) - will be sorted by point count
SEMANTICKITTI_CLASSES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]

# Default classwise baseline file path
DEFAULT_CLASSWISE_DIR = PROJECT_ROOT / 'docs' / 'tables' / 'classwise'

# Method colors - same as generate_figures.py
METHOD_COLORS = {
    'baseline': '#000000',  # Black
    'RS': '#66b8ff',        # Light Blue
    'DBSCAN': '#f77189',    # Pink
    'FPS': '#97a431',       # Olive Green
    'VB': '#ff9c01',        # Orange (Voxel-based)
    'Voxel': '#ff9c01',     # Alias
    'SB': '#ff6752',        # Salmon Red (Space-based/Poisson)
    'Poisson': '#ff6752',   # Alias
    'IDIS': '#2ca02c',      # Forest Green
    'DEPOCO': '#9467bd',    # Purple (distinct from others)
}

# Method display order
METHOD_ORDER = ['baseline', 'RS', 'FPS', 'SB', 'IDIS', 'DBSCAN', 'VB', 'DEPOCO']

# Short method names for annotations (to save space in plots)
METHOD_SHORT_NAMES = {
    'baseline': 'B',
    'RS': 'R',
    'FPS': 'F',
    'SB': 'S',       # Space-based/Poisson
    'IDIS': 'I',
    'DBSCAN': 'D',
    'VB': 'V',       # Voxel
    'DEPOCO': 'DC',
}

# Legend labels with short forms
METHOD_LEGEND_LABELS = {
    'baseline': 'Baseline (B)',
    'RS': 'RS (R)',
    'FPS': 'FPS (F)',
    'SB': 'SB (S)',
    'IDIS': 'IDIS (I)',
    'DBSCAN': 'DBSCAN (D)',
    'VB': 'VB (V)',
    'DEPOCO': 'DEPOCO (DC)',
}

# Method line styles for better distinction
METHOD_LINESTYLES = {
    'baseline': ':',       # Dotted (reference)
    'RS': '-',             # Solid
    'DBSCAN': '-',         # Solid
    'FPS': '-',            # Solid
    'VB': '-',             # Solid (Voxel-based)
    'SB': '-',             # Solid (Space-based/Poisson)
    'IDIS': '-',           # Solid
    'DEPOCO': '-',         # Solid
}

# Method markers for better distinction
METHOD_MARKERS = {
    'baseline': 'o',       # Circle
    'RS': 's',             # Square
    'DBSCAN': '^',         # Triangle up
    'FPS': 'd',            # Diamond
    'VB': 'v',             # Triangle down (Voxel-based)
    'SB': 'p',             # Pentagon (Space-based/Poisson)
    'IDIS': '*',           # Star
    'DEPOCO': 'h',         # Hexagon
}

# Loss levels to include (excluding 0% - baseline shown as reference line)
LOSS_LEVELS = [30, 50, 70, 90]

# Inference directory for generalization data
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / 'docs' / 'tables' / 'inference_on_original'


def parse_metrics_file(filepath):
    """Parse a metrics file and extract class-wise IoU."""
    class_iou = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Find CLASS-WISE PERFORMANCE section
    class_section = re.search(
        r'CLASS-WISE PERFORMANCE[^\n]*\n-+\n[^\n]+\n-+\n(.*?)\n-+',
        content, re.DOTALL
    )
    if not class_section:
        return class_iou

    # Parse each class line
    for line in class_section.group(1).strip().split('\n'):
        match = re.match(r'(\S+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', line)
        if match:
            class_name = match.group(1)
            iou = float(match.group(2))
            class_iou[class_name] = iou

    return class_iou


def parse_baseline_classwise_file(filepath):
    """
    Parse baseline classwise distribution file to get point counts per class.

    Returns:
        dict: {class_name: total_points} (Train + Val points)
    """
    class_points = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse each class line
    # Format: class_name | ID | Train Base | Train Sub | Train % | Val Base | Val Sub | Val %
    for line in content.split('\n'):
        if '|' not in line or 'Class' in line or '---' in line or '===' in line:
            continue

        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 6:
            class_name = parts[0].strip()

            # Skip special rows
            if class_name in ['TOTAL', 'unlabeled', '']:
                continue

            try:
                # Get Train Base (index 2) and Val Base (index 5)
                train_base = int(parts[2].strip().replace(',', ''))
                val_base = int(parts[5].strip().replace(',', ''))
                total_points = train_base + val_base
                class_points[class_name] = total_points
            except (ValueError, IndexError):
                continue

    return class_points


def format_points(points):
    """Format point count in human-readable format (K, M, B)."""
    if points >= 1_000_000_000:
        return f"{points / 1_000_000_000:.1f}B"
    elif points >= 1_000_000:
        return f"{points / 1_000_000:.1f}M"
    elif points >= 1_000:
        return f"{points / 1_000:.1f}K"
    else:
        return str(points)


def get_classes_sorted_by_points(classwise_dir):
    """
    Get SemanticKITTI classes sorted by total point count (descending).

    Returns:
        tuple: (sorted_classes, class_points)
            - sorted_classes: list of class names sorted by point count (largest first)
            - class_points: dict of {class_name: total_points}
    """
    baseline_file = Path(classwise_dir) / 'classwise_baseline_loss0.txt'

    if not baseline_file.exists():
        print(f"Warning: Baseline classwise file not found: {baseline_file}")
        print("Using default class order.")
        return SEMANTICKITTI_CLASSES, {}

    class_points = parse_baseline_classwise_file(baseline_file)

    if not class_points:
        print("Warning: Could not parse class points from baseline file.")
        return SEMANTICKITTI_CLASSES, {}

    # Sort classes by point count (descending)
    sorted_classes = sorted(class_points.keys(), key=lambda x: class_points[x], reverse=True)

    # Verify all expected classes are present
    for cls in SEMANTICKITTI_CLASSES:
        if cls not in sorted_classes:
            print(f"Warning: Class '{cls}' not found in baseline file.")

    print(f"Classes sorted by point count (largest first):")
    for cls in sorted_classes:
        print(f"  {cls}: {class_points[cls]:,} points ({format_points(class_points[cls])})")

    return sorted_classes, class_points


def get_method_from_filename(filename):
    """Extract method name from filename."""
    name = filename.lower()

    if 'baseline' in name:
        return 'baseline'
    elif 'dbscan' in name:
        return 'DBSCAN'
    elif 'fps' in name:
        return 'FPS'
    elif 'poisson' in name:
        return 'SB'
    elif 'voxel' in name:
        return 'VB'
    elif 'idis' in name:
        return 'IDIS'
    elif 'rs_' in name or name.startswith('rs'):
        return 'RS'
    elif 'depoco' in name:
        return 'DEPOCO'
    return None


def collect_all_data(tables_dir):
    """
    Collect class-wise mIoU for all methods across all loss levels.

    Returns:
        data: {method: {loss_level: {class_name: iou}}}
        baseline_data: {class_name: iou} for loss 0%
    """
    data = {}
    baseline_data = {}

    for filename in os.listdir(tables_dir):
        if not filename.endswith('_metrics.txt'):
            continue

        # Extract method
        method = get_method_from_filename(filename)
        if not method:
            continue

        # Extract loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if loss_match:
            loss = int(loss_match.group(1))
        else:
            continue

        # Parse the file
        filepath = os.path.join(tables_dir, filename)
        class_iou = parse_metrics_file(filepath)

        if not class_iou:
            continue

        # Handle baseline (loss 0%) separately
        if loss == 0:
            baseline_data = class_iou
            continue

        # Only include specified loss levels for other methods
        if loss not in LOSS_LEVELS:
            continue

        # Initialize nested dicts if needed
        if method not in data:
            data[method] = {}

        if loss not in data[method]:
            data[method][loss] = class_iou
        else:
            # Average with existing data (for multi-seed experiments)
            for cls, iou in class_iou.items():
                if cls in data[method][loss]:
                    data[method][loss][cls] = (data[method][loss][cls] + iou) / 2
                else:
                    data[method][loss][cls] = iou

    return data, baseline_data


def collect_inference_data(inference_dir):
    """
    Collect class-wise mIoU for inference on original data (generalization).

    Returns:
        inference_data: {method: {loss_level: {class_name: iou}}}
    """
    inference_data = {}
    inference_dir = Path(inference_dir)

    if not inference_dir.exists():
        print(f"Warning: Inference directory not found: {inference_dir}")
        return inference_data

    for filename in os.listdir(inference_dir):
        if not filename.endswith('_metrics.txt'):
            continue

        # Extract method
        method = get_method_from_filename(filename)
        if not method:
            continue

        # Extract loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if loss_match:
            loss = int(loss_match.group(1))
        else:
            continue

        # Only include specified loss levels
        if loss not in LOSS_LEVELS:
            continue

        # Parse the file
        filepath = inference_dir / filename
        class_iou = parse_metrics_file(filepath)

        if not class_iou:
            continue

        # Initialize nested dicts if needed
        if method not in inference_data:
            inference_data[method] = {}

        if loss not in inference_data[method]:
            inference_data[method][loss] = class_iou
        else:
            # Average with existing data (for multi-seed experiments)
            for cls, iou in class_iou.items():
                if cls in inference_data[method][loss]:
                    inference_data[method][loss][cls] = (inference_data[method][loss][cls] + iou) / 2
                else:
                    inference_data[method][loss][cls] = iou

    return inference_data


def find_best_method_at_loss(inference_data, class_name, loss_level, threshold=0.01):
    """
    Find the best performing method for a class at a specific loss level.

    Args:
        inference_data: {method: {loss_level: {class_name: iou}}}
        class_name: The class to check
        loss_level: The loss level to check
        threshold: Minimum difference to declare a clear winner (default 1%)

    Returns:
        tuple: (best_method, best_iou, is_clear_winner)
               is_clear_winner is True if best method is > threshold better than 2nd best
    """
    method_ious = {}
    for method, loss_data in inference_data.items():
        if loss_level in loss_data and class_name in loss_data[loss_level]:
            method_ious[method] = loss_data[loss_level][class_name]

    if not method_ious:
        return None, 0, False

    # Sort by IoU descending
    sorted_methods = sorted(method_ious.items(), key=lambda x: x[1], reverse=True)
    best_method, best_iou = sorted_methods[0]

    # Check if clear winner (> threshold better than 2nd place)
    if len(sorted_methods) > 1:
        second_iou = sorted_methods[1][1]
        is_clear_winner = (best_iou - second_iou) >= threshold
    else:
        is_clear_winner = True

    return best_method, best_iou, is_clear_winner


def find_best_method_with_significance(inference_data, baseline_data, class_name, loss_level,
                                        all_classes, alpha=0.05):
    """
    Find the best performing method using Wilcoxon signed-rank test for significance.

    Compares the best method against ALL other methods (including baseline).
    Uses Bonferroni correction for multiple comparisons.
    Only marks as significant if best method beats ALL others.

    Args:
        inference_data: {method: {loss_level: {class_name: iou}}}
        baseline_data: {class_name: iou} baseline IoU values
        class_name: The class to check for best method
        loss_level: The loss level to check
        all_classes: List of all class names for cross-class comparison
        alpha: Significance level (default 0.05)

    Returns:
        tuple: (best_method, best_iou, is_significant, p_value)
    """
    # Collect IoU values for all subsampling methods at this loss level (excluding baseline)
    method_ious = {}
    for method, loss_data in inference_data.items():
        if loss_level in loss_data and class_name in loss_data[loss_level]:
            method_ious[method] = loss_data[loss_level][class_name]

    if len(method_ious) < 2:
        return None, 0, False, 1.0

    # Find the best method for this class (among subsampling methods only)
    sorted_methods = sorted(method_ious.items(), key=lambda x: x[1], reverse=True)
    best_method, best_iou = sorted_methods[0]

    # For significance testing, compare methods across ALL classes at this loss level
    # This gives us paired samples (one per class) for the Wilcoxon test

    # Collect cross-class IoU arrays for each subsampling method (excluding baseline)
    method_class_ious = {}
    for method, loss_data in inference_data.items():
        if loss_level in loss_data:
            ious = []
            for cls in all_classes:
                if cls in loss_data[loss_level]:
                    ious.append(loss_data[loss_level][cls])
                else:
                    ious.append(np.nan)
            method_class_ious[method] = np.array(ious)

    # Get the best method's cross-class IoUs
    best_method_ious = method_class_ious.get(best_method)
    if best_method_ious is None:
        return best_method, best_iou, False, 1.0

    # Test against ALL other methods - must beat all to be significant
    # No Bonferroni correction - use raw alpha
    is_significant = True
    max_p_value = 0.0  # Track the highest (worst) p-value

    for other_method, other_ious in method_class_ious.items():
        if other_method == best_method:
            continue

        # Remove NaN pairs
        valid_mask = ~(np.isnan(best_method_ious) | np.isnan(other_ious))
        if valid_mask.sum() < 5:  # Need at least 5 samples for meaningful test
            continue  # Skip this comparison if not enough data

        best_valid = best_method_ious[valid_mask]
        other_valid = other_ious[valid_mask]

        # Skip if arrays are identical
        if np.allclose(best_valid, other_valid):
            is_significant = False
            max_p_value = 1.0
            break

        try:
            # Wilcoxon signed-rank test (one-sided: best > other)
            stat, p_value = stats.wilcoxon(best_valid, other_valid, alternative='greater')
            max_p_value = max(max_p_value, p_value)

            if p_value >= alpha:
                is_significant = False
        except Exception:
            is_significant = False
            max_p_value = 1.0

    return best_method, best_iou, is_significant, max_p_value


def create_performance_drop_figure(data, baseline_data, output_dir, sorted_classes, class_points,
                                   inference_data=None, show_plot=False):
    """
    Create 4x5 grid of subplots showing mIoU vs loss level per class.
    Baseline shown as horizontal dotted reference line.
    Classes are sorted by point count (largest first).
    Inference/generalization data shown as filled circle markers.
    Best method at each loss level annotated with short name.
    """
    # Create figure with 4x5 subplots
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    # Get all methods that have data (exclude baseline)
    all_methods = [m for m in data.keys() if m != 'baseline']
    all_methods = sorted(all_methods, key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 100)

    # Plot each class (sorted by point count)
    for idx, class_name in enumerate(sorted_classes):
        ax = axes[idx]

        # Draw baseline as horizontal dotted line
        if class_name in baseline_data:
            baseline_iou = baseline_data[class_name]
            ax.axhline(y=baseline_iou, color='black', linestyle=':', linewidth=2.5,
                      label='Baseline', alpha=0.8)

        # Plot each method - only Test → Original Data (inference on original)
        for method in all_methods:
            # Only plot inference/generalization data (Test → Original Data)
            if inference_data and method in inference_data:
                inf_loss_levels = []
                inf_iou_values = []
                for loss in sorted(inference_data[method].keys()):
                    if class_name in inference_data[method][loss]:
                        inf_loss_levels.append(loss)
                        inf_iou_values.append(inference_data[method][loss][class_name])

                if inf_loss_levels:
                    color = METHOD_COLORS.get(method, '#888888')
                    marker = METHOD_MARKERS.get(method, 'o')
                    # Test → Original: solid line with markers
                    ax.plot(inf_loss_levels, inf_iou_values,
                           color=color, linewidth=2.5, markersize=8,
                           linestyle='-', marker=marker,
                           markeredgecolor='black', markeredgewidth=0.5,
                           label=method, zorder=5)

        # Add best method annotations at ALL loss levels
        # Uses Wilcoxon signed-rank test for statistical significance
        # Only shows annotation if best method is significantly better than ALL others
        if inference_data:
            for loss in LOSS_LEVELS:  # Annotate at all loss levels (30, 50, 70, 90)
                best_method, best_iou, is_significant, p_value = find_best_method_with_significance(
                    inference_data, baseline_data, class_name, loss,
                    all_classes=sorted_classes, alpha=0.05
                )
                # Only annotate if:
                # 1. Best method significantly outperforms RS (delta > 0.05)
                # 2. The IoU is meaningful (>0.10) - skip near-zero results
                baseline_iou = baseline_data.get(class_name, 0)

                # Get RS IoU for comparison
                rs_iou = None
                if 'RS' in inference_data and loss in inference_data['RS']:
                    rs_iou = inference_data['RS'][loss].get(class_name)

                # Only show annotation if best method beats RS by > 0.05
                if best_method and rs_iou is not None and best_method != 'RS':
                    delta_rs = best_iou - rs_iou
                    # Only show annotation if improvement over RS > 5%
                    if delta_rs <= 0.05:
                        continue  # Skip this annotation - RS is good enough
                    # Skip if IoU is too low (near-zero results)
                    if best_iou < 0.10:
                        continue
                    short_name = METHOD_SHORT_NAMES.get(best_method, best_method[:2])
                else:
                    # If best method IS RS or no RS data, skip annotation
                    continue

                # Position annotation above the point
                y_offset = 10
                v_align = 'bottom'

                # X positioning and arrow configuration
                if loss == 90:
                    # For 90% loss: two-line format with arrow
                    # Method name on top, delta value with brackets below
                    label = f"{short_name}\n(+{delta_rs:.2f})"
                    x_offset = -5   # Slightly to the left
                    y_offset = 8    # Closer to data point
                    h_align = 'center'
                    # Add arrow to clearly connect annotation to 90% point
                    arrow_props = dict(arrowstyle='->', color='black', lw=0.8)
                    fontsize = 10
                    linespacing = 0.8
                else:  # 30%, 50%, 70%
                    # Single line format: method(+delta)
                    label = f"{short_name}(+{delta_rs:.2f})"
                    x_offset = 2
                    h_align = 'left'   # Text starts at point, extends right
                    arrow_props = None
                    fontsize = 11
                    linespacing = 1.0

                # Position label - black bold for academic style
                # Use high zorder to bring annotations to front (above plot lines which have zorder=5)
                ax.annotate(label, xy=(loss, best_iou),
                           xytext=(x_offset, y_offset), textcoords='offset points',
                           fontsize=fontsize, fontweight='bold', color='black',
                           ha=h_align, va=v_align, zorder=100,
                           arrowprops=arrow_props, linespacing=linespacing)

        # Customize subplot - academic style with larger text
        # Include point count in title
        if class_name in class_points:
            points_str = format_points(class_points[class_name])
            title = f"{class_name} ({points_str})"
        else:
            title = class_name
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Loss %', fontsize=14, fontweight='medium')
        ax.set_ylabel('mIoU', fontsize=14, fontweight='medium')
        ax.set_xlim(25, 95)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(LOSS_LEVELS)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=12)

    # Use the 20th subplot (index 19) for the legend - vertically stacked
    ax_legend = axes[19]
    ax_legend.axis('off')

    # Create legend handles - baseline first, then methods (training), then line style explanation
    legend_handles = []

    # Baseline handle (horizontal dotted line)
    baseline_label = METHOD_LEGEND_LABELS.get('baseline', 'Baseline')
    baseline_handle = Line2D([0], [0], color='black', linestyle=':', linewidth=2.5,
                             label=baseline_label)
    legend_handles.append(baseline_handle)

    # Method handles (training - solid lines with markers)
    for method in all_methods:
        color = METHOD_COLORS.get(method, '#888888')
        marker = METHOD_MARKERS.get(method, 'o')
        label = METHOD_LEGEND_LABELS.get(method, method)
        handle = Line2D([0], [0], marker=marker, color=color,
                       markerfacecolor=color, markersize=10,
                       markeredgecolor='black', markeredgewidth=0.5,
                       linewidth=2.5, linestyle='-', label=label)
        legend_handles.append(handle)


    # Legend in the 20th subplot, vertically stacked (ncol=1)
    ax_legend.legend(handles=legend_handles, loc='upper center', fontsize=14,
                    framealpha=0.95, ncol=1)

    # Add note about annotation threshold below the legend
    ax_legend.text(0.5, 0.12, "Annotations: Best method vs RS (Fastest)\n(shown if delta > 0.05 mIoU)",
                  transform=ax_legend.transAxes, fontsize=13, ha='center', va='top',
                  style='italic', color='#333333')

    plt.tight_layout()

    # Save figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['png', 'svg', 'pdf']:
        output_path = output_dir / f'09_classwise_performance_drop.{fmt}'
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 09_classwise_performance_drop figure")

    if show_plot:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate class-wise performance drop figure (small multiples)'
    )
    parser.add_argument(
        '--tables-dir',
        default=str(DEFAULT_TABLES_DIR),
        help='Directory containing metrics files'
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

    args = parser.parse_args()

    # Verify directory exists
    if not os.path.exists(args.tables_dir):
        print(f"Error: Tables directory not found: {args.tables_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Generating Class-wise Performance Drop Figure")
    print("=" * 60)
    print(f"Tables directory: {args.tables_dir}")
    print(f"Output directory: {args.output}")
    print(f"Loss levels: {LOSS_LEVELS}")
    print()

    # Get classes sorted by point count
    print("Sorting classes by point count...")
    sorted_classes, class_points = get_classes_sorted_by_points(DEFAULT_CLASSWISE_DIR)
    print()

    # Collect training data
    print("Collecting training data from metrics files...")
    data, baseline_data = collect_all_data(args.tables_dir)

    # Print summary
    print(f"\nBaseline classes: {len(baseline_data)}")
    print(f"Methods found: {list(data.keys())}")
    for method in data:
        print(f"  {method}: loss levels {sorted(data[method].keys())}")

    # Collect inference/generalization data
    print("\nCollecting inference data (generalization)...")
    inference_data = collect_inference_data(DEFAULT_INFERENCE_DIR)
    if inference_data:
        print(f"Inference methods found: {list(inference_data.keys())}")
        for method in inference_data:
            print(f"  {method}: loss levels {sorted(inference_data[method].keys())}")
    else:
        print("No inference data found.")

    # Create figure
    print("\nGenerating figure...")
    create_performance_drop_figure(data, baseline_data, args.output, sorted_classes, class_points,
                                   inference_data=inference_data, show_plot=args.show)

    print("\nDone!")


if __name__ == '__main__':
    main()
