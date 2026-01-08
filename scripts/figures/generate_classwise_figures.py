#!/usr/bin/env python3
"""
Generate Class-wise Radar Charts for mIoU and Point Distribution

This script creates two-part figures (A + B) for each loss level:
- Part A: Radar chart showing class-wise mIoU (Test → Original Data)
- Part B: Radar chart showing class-wise point retention

Legend shows both Training (T) and Generalization (G) mIoU values.
Baseline (loss 0%) is included as reference in all figures.

Output figures (numbered to follow generate_figures.py):
- 04_classwise_loss30 - Class-wise radar charts at 30% loss
- 05_classwise_loss50 - Class-wise radar charts at 50% loss
- 06_classwise_loss70 - Class-wise radar charts at 70% loss
- 07_classwise_loss90 - Class-wise radar charts at 90% loss
- 08_classwise_loss90_idis_r_sensitivity - IDIS R-value comparison at 90% loss
"""

import re
import os
import sys
from pathlib import Path
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Script and project paths (for relative path resolution)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/figures -> scripts -> project root
DEFAULT_TABLES_DIR = PROJECT_ROOT / 'docs' / 'tables'
DEFAULT_CLASSWISE_DIR = PROJECT_ROOT / 'docs' / 'tables' / 'classwise'
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / 'docs' / 'tables' / 'inference_on_original'
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
    'savefig.pad_inches': 0.1,
})

# SemanticKITTI class names (19 classes)
SEMANTICKITTI_CLASSES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]

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

# IDIS R-value colors (distinct colors for better differentiation)
IDIS_R_COLORS = {
    'R5': '#E41A1C',   # Red
    'R10': '#377EB8',  # Blue
    'R15': '#4DAF4A',  # Green
    'R20': '#984EA3',  # Purple
}

# IDIS R-value line styles
IDIS_R_LINESTYLES = {
    'R5': '-',         # Solid
    'R10': '--',       # Dashed
    'R15': '-.',       # Dash-dot
    'R20': ':',        # Dotted
}

# IDIS R-value markers (different shapes for distinction)
IDIS_R_MARKERS = {
    'R5': 'o',         # Circle
    'R10': 's',        # Square
    'R15': '^',        # Triangle up
    'R20': 'D',        # Diamond
}

# Method display order
METHOD_ORDER = ['baseline', 'RS', 'FPS', 'SB', 'Poisson', 'IDIS', 'DBSCAN', 'VB', 'Voxel', 'DEPOCO']

# Figure number mapping for loss levels (starting from 04)
LOSS_TO_FIGURE_NUM = {
    30: '04',
    50: '05',
    70: '06',
    90: '07',
}


def parse_metrics_file(filepath):
    """Parse a metrics file and extract class-wise IoU."""
    class_iou = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Find CLASS-WISE PERFORMANCE section
    # Pattern: CLASS-WISE PERFORMANCE ... header line ... dashed line ... data ... dashed line
    class_section = re.search(
        r'CLASS-WISE PERFORMANCE[^\n]*\n-+\n[^\n]+\n-+\n(.*?)\n-+',
        content, re.DOTALL
    )
    if not class_section:
        return class_iou

    # Parse each class line
    # Format: class_name           |    IoU     |  Accuracy
    for line in class_section.group(1).strip().split('\n'):
        match = re.match(r'(\S+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', line)
        if match:
            class_name = match.group(1)
            iou = float(match.group(2))
            class_iou[class_name] = iou

    return class_iou


def parse_classwise_distribution_file(filepath):
    """
    Parse a classwise distribution file and extract Val % (retention) for each class.
    Also calculates actual class distribution for reference.

    Returns:
        retention_pct: {class_name: retention_percentage} (0-100)
        class_distribution: {class_name: distribution_percentage} (0-100)
    """
    retention_pct = {}
    class_points = {}
    total_points = 0
    unlabeled_points = 0

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse each class line
    # Format: class_name       |  ID  |   Train Base |    Train Sub |   Train % |     Val Base |      Val Sub |     Val %
    for line in content.split('\n'):
        if '|' not in line or 'Class' in line or '---' in line:
            continue

        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 8:
            class_name = parts[0].strip()

            # Get Val Sub column (index 6) for distribution calculation
            val_sub_str = parts[6].strip().replace(',', '')
            # Get Val % column (index 7) for retention
            val_pct_str = parts[7].strip().rstrip('%')

            try:
                val_sub = int(val_sub_str)
                val_pct = float(val_pct_str)

                if class_name == 'TOTAL':
                    total_points = val_sub
                elif class_name == 'unlabeled':
                    unlabeled_points = val_sub
                else:
                    retention_pct[class_name] = val_pct
                    class_points[class_name] = val_sub
            except ValueError:
                continue

    # Calculate actual class distribution (% of total excluding unlabeled)
    total_excluding_unlabeled = total_points - unlabeled_points
    class_distribution = {}
    if total_excluding_unlabeled > 0:
        for class_name, points in class_points.items():
            class_distribution[class_name] = (points / total_excluding_unlabeled) * 100.0

    return retention_pct, class_distribution


def parse_inference_metrics_file(filepath):
    """Parse an inference metrics file and extract class-wise IoU."""
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


def collect_inference_data_for_loss_level(inference_dir, loss_level):
    """
    Collect class-wise mIoU from inference (generalization) results.

    Returns:
        inference_data: {method: {class_name: iou}}
    """
    inference_data = {}

    if not os.path.exists(inference_dir):
        return inference_data

    for filename in os.listdir(inference_dir):
        if not filename.endswith('_metrics.txt'):
            continue

        # Check if this file is for the correct loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match:
            continue
        file_loss = int(loss_match.group(1))
        if file_loss != loss_level:
            continue

        method = get_method_from_filename(filename)
        if not method:
            continue

        filepath = os.path.join(inference_dir, filename)
        class_iou = parse_inference_metrics_file(filepath)

        if class_iou:
            if method not in inference_data:
                inference_data[method] = class_iou
            else:
                # Average with existing data (for multi-seed experiments)
                for cls, iou in class_iou.items():
                    if cls in inference_data[method]:
                        inference_data[method][cls] = (inference_data[method][cls] + iou) / 2
                    else:
                        inference_data[method][cls] = iou

    return inference_data


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


def collect_baseline_data(tables_dir, classwise_dir):
    """Collect baseline (loss 0%) data for mIoU and distribution."""
    miou_data = {}
    dist_data = {}
    actual_distribution = {}  # Actual % each class represents

    # mIoU from metrics file
    baseline_metrics = os.path.join(tables_dir, 'baseline_loss0_seed1_140gb_metrics.txt')
    if os.path.exists(baseline_metrics):
        miou_data['baseline'] = parse_metrics_file(baseline_metrics)

    # Distribution from classwise file
    baseline_dist = os.path.join(classwise_dir, 'classwise_baseline_loss0.txt')
    if os.path.exists(baseline_dist):
        retention, distribution = parse_classwise_distribution_file(baseline_dist)
        dist_data['baseline'] = retention
        actual_distribution['baseline'] = distribution

    return miou_data, dist_data, actual_distribution


def collect_data_for_loss_level(tables_dir, classwise_dir, loss_level):
    """
    Collect class-wise mIoU and point distribution data for a specific loss level.

    Returns:
        miou_data: {method: {class_name: iou}}
        dist_data: {method: {class_name: retention_pct}}
    """
    miou_data = {}
    dist_data = {}

    # mIoU data from metrics files
    for filename in os.listdir(tables_dir):
        if not filename.endswith('_metrics.txt'):
            continue

        # Check if this file is for the correct loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match:
            continue
        file_loss = int(loss_match.group(1))
        if file_loss != loss_level:
            continue

        method = get_method_from_filename(filename)
        if not method:
            continue

        filepath = os.path.join(tables_dir, filename)
        class_iou = parse_metrics_file(filepath)

        if class_iou:
            if method not in miou_data:
                miou_data[method] = class_iou
            else:
                # Average with existing data (for multi-seed experiments)
                for cls, iou in class_iou.items():
                    if cls in miou_data[method]:
                        miou_data[method][cls] = (miou_data[method][cls] + iou) / 2
                    else:
                        miou_data[method][cls] = iou

    # Distribution data from classwise files
    for filename in os.listdir(classwise_dir):
        if not filename.startswith('classwise_'):
            continue

        # Check if this file is for the correct loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match:
            continue
        file_loss = int(loss_match.group(1))
        if file_loss != loss_level:
            continue

        method = get_method_from_filename(filename)
        if not method:
            continue

        filepath = os.path.join(classwise_dir, filename)
        retention_pct, _ = parse_classwise_distribution_file(filepath)

        if retention_pct:
            if method not in dist_data:
                dist_data[method] = retention_pct
            else:
                # Average with existing data (for multi-seed experiments)
                for cls, pct in retention_pct.items():
                    if cls in dist_data[method]:
                        dist_data[method][cls] = (dist_data[method][cls] + pct) / 2
                    else:
                        dist_data[method][cls] = pct

    return miou_data, dist_data


def create_radar_chart(ax, data, title, classes, is_percentage=False,
                       baseline_data=None, show_values=True, max_scale=None,
                       actual_distribution=None):
    """
    Create a radar chart on the given axes with solid lines and markers.

    Args:
        ax: matplotlib axes
        data: {method: {class_name: value}}
        title: chart title
        classes: list of class names (full names will be used)
        is_percentage: if True, values are percentages (0-100), else IoU values (0-1)
        baseline_data: {class_name: value} for baseline reference
        show_values: whether to show values on labels
        max_scale: optional max value for y-axis (auto-calculated if None)
        actual_distribution: {class_name: actual_%} for showing actual class distribution in labels
    """
    # Number of variables
    num_vars = len(classes)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Sort methods by order preference (baseline first if present)
    methods = sorted(data.keys(), key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 100)

    # Plot baseline first (with fill) if provided and not already in data
    if baseline_data and 'baseline' not in data:
        values = []
        for cls in classes:
            val = baseline_data.get(cls, 0)
            # Keep IoU as-is (0-1 scale), percentages as-is (0-100 scale)
            values.append(val)
        values += values[:1]

        color = METHOD_COLORS.get('baseline', '#000000')
        # Baseline with SOLID line and fill
        ax.plot(angles, values, 'o-', linewidth=2.5, label='Baseline', color=color,
                markersize=6, markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Plot each method with SOLID lines and markers
    for method in methods:
        values = []
        for cls in classes:
            val = data[method].get(cls, 0)
            # Keep values as-is (IoU 0-1 or percentage 0-100)
            values.append(val)
        values += values[:1]  # Complete the loop

        color = METHOD_COLORS.get(method, '#888888')

        # Plot with SOLID line and circle markers (like reference image)
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color,
                markersize=5, markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set y-axis limits based on data type or custom max_scale
    if max_scale is not None:
        # Use custom scale (for class distribution)
        ax.set_ylim(0, max_scale * 1.05)
        # Create nice tick values
        tick_step = max_scale / 5
        ticks = [tick_step * i for i in range(1, 6)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.2f}' for t in ticks], size=9, fontweight='medium')
    elif is_percentage:
        ax.set_ylim(0, 105)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9, fontweight='medium')
    else:
        # IoU scale (0-1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, fontweight='medium')

    # Position radial labels (0.2, 0.4, etc.) on the inside, between class labels
    # With 19 classes, each spans ~18.9 degrees. Position labels at ~10 degrees to avoid overlap
    ax.set_rlabel_position(10)

    # Draw spoke lines extending slightly beyond the last ring - dotted black
    for angle in angles[:-1]:
        if max_scale is not None:
            ax.plot([angle, angle], [0, max_scale * 1.03], color='black', alpha=0.5, linewidth=1.2, linestyle=':', zorder=0)
        elif is_percentage:
            ax.plot([angle, angle], [0, 103], color='black', alpha=0.5, linewidth=1.2, linestyle=':', zorder=0)
        else:
            ax.plot([angle, angle], [0, 1.03], color='black', alpha=0.5, linewidth=1.2, linestyle=':', zorder=0)

    # Create labels with baseline values - value first, then FULL class name
    labels = []
    for i, cls in enumerate(classes):
        # Use full class name
        full_name = cls
        if baseline_data and show_values:
            val = baseline_data.get(cls, 0)
            # Always use 2 decimal places for consistency
            labels.append(f'{val:.2f}\n{full_name}')
        else:
            labels.append(full_name)

    # Set labels - larger text for academic readability
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, fontweight='medium')
    # Move labels slightly inward
    ax.tick_params(axis='x', pad=8)

    # Subplot title (A or B)
    ax.set_title(title, size=14, fontweight='bold', pad=20)

    # Grid appearance - dotted black lines
    ax.grid(True, color='black', alpha=0.5, linestyle=':', linewidth=1.2)
    ax.xaxis.grid(False)  # Turn off radial grid lines (we draw them manually)

    # Remove the outermost circle (polar spine)
    ax.spines['polar'].set_visible(False)


def create_combined_figure(miou_data, dist_data, loss_level, output_dir,
                           baseline_miou=None, baseline_dist=None,
                           baseline_actual_dist=None, inference_data=None,
                           show_plot=False):
    """
    Create a combined figure with two radar charts:
    - A: Class-wise mIoU (Test → Original Data)
    - B: Class-wise Point Retention
    Includes baseline as reference.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))

    # Part A: Class-wise mIoU (Test → Original Data)
    if inference_data or baseline_miou:
        create_radar_chart(
            ax1, inference_data if inference_data else {},
            '(A) Class-wise mIoU (Test → Original Data)',
            SEMANTICKITTI_CLASSES,
            is_percentage=False,
            baseline_data=baseline_miou,
            show_values=True
        )
    else:
        ax1.text(0.5, 0.5, 'No inference data available', ha='center', va='center', transform=ax1.transAxes)

    # Part B: Class-wise Point Retention (100% baseline = all points retained)
    # Convert retention data from percentage (0-100) to ratio (0-1)
    dist_data_normalized = {}
    for method, class_data in dist_data.items():
        dist_data_normalized[method] = {cls: val / 100.0 for cls, val in class_data.items()}

    baseline_dist_normalized = None
    if baseline_dist:
        baseline_dist_normalized = {cls: val / 100.0 for cls, val in baseline_dist.items()}

    if dist_data_normalized or baseline_dist_normalized:
        create_radar_chart(
            ax2, dist_data_normalized,
            '(B) Class-wise Point Retention',
            SEMANTICKITTI_CLASSES,
            is_percentage=False,  # 0-1 scale
            baseline_data=baseline_dist_normalized,
            show_values=False,  # Clean labels - just class names
            actual_distribution=None  # No percentages in labels
        )
    else:
        ax2.text(0.5, 0.5, 'No distribution data available', ha='center', va='center', transform=ax2.transAxes)

    # Create combined legend below the figures with mIoU values
    all_methods = set(miou_data.keys()) | set(dist_data.keys())
    if inference_data:
        all_methods |= set(inference_data.keys())
    if baseline_miou or baseline_dist:
        all_methods.add('baseline')
    methods_sorted = sorted(all_methods, key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 100)

    legend_elements = []
    for method in methods_sorted:
        color = METHOD_COLORS.get(method, '#888888')

        # Calculate mean mIoU for training and inference
        if method == 'baseline' and baseline_miou:
            mean_train = np.mean(list(baseline_miou.values()))
            label = f'Baseline ({mean_train:.2f})'
        elif method in miou_data:
            mean_train = np.mean(list(miou_data[method].values()))
            if inference_data and method in inference_data:
                mean_inf = np.mean(list(inference_data[method].values()))
                label = f'{method} (T:{mean_train:.2f}, G:{mean_inf:.2f})'
            else:
                label = f'{method} ({mean_train:.2f})'
        else:
            label = method

        legend_elements.append(Line2D([0], [0], marker='o', color=color,
                                      markerfacecolor=color, markersize=10,
                                      markeredgecolor='black', markeredgewidth=0.5,
                                      linewidth=2.5, linestyle='-', label=label))

    # Use 4 columns for legend to fit within figure width
    ncols = min(4, len(legend_elements))
    fig.legend(handles=legend_elements, loc='lower center', ncol=ncols,
               bbox_to_anchor=(0.5, -0.08), fontsize=10, framealpha=0.95)

    plt.tight_layout()

    # Save figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get figure number prefix
    fig_num = LOSS_TO_FIGURE_NUM.get(loss_level, '00')

    for fmt in ['png', 'svg', 'pdf']:
        output_path = output_dir / f'{fig_num}_classwise_loss{loss_level}.{fmt}'
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved {fig_num}_classwise_loss{loss_level} figure")

    if show_plot:
        plt.show()

    plt.close(fig)


def collect_idis_r_value_data(tables_dir, classwise_dir, loss_level=90):
    """
    Collect class-wise mIoU and retention data for IDIS R-value variants.

    Returns:
        miou_data: {r_value: {class_name: iou}}  e.g., {'R5': {...}, 'R10': {...}, ...}
        retention_data: {r_value: {class_name: retention_pct}}
    """
    miou_data = {}
    retention_data = {}

    # R-value variants to look for
    r_values = ['R5', 'R10', 'R15', 'R20']

    # mIoU data from metrics files
    for filename in os.listdir(tables_dir):
        if not filename.endswith('_metrics.txt'):
            continue
        if 'idis' not in filename.lower():
            continue

        # Check loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match or int(loss_match.group(1)) != loss_level:
            continue

        # Determine R-value
        r_match = re.search(r'_R(\d+)_', filename)
        if r_match:
            r_value = f'R{r_match.group(1)}'
        else:
            # Default is R10 for IDIS without explicit R value
            r_value = 'R10'

        filepath = os.path.join(tables_dir, filename)
        class_iou = parse_metrics_file(filepath)

        if class_iou and r_value in r_values:
            if r_value not in miou_data:
                miou_data[r_value] = class_iou
            else:
                # Average with existing (if multiple seeds)
                for cls, iou in class_iou.items():
                    if cls in miou_data[r_value]:
                        miou_data[r_value][cls] = (miou_data[r_value][cls] + iou) / 2
                    else:
                        miou_data[r_value][cls] = iou

    # Retention data from classwise files
    for filename in os.listdir(classwise_dir):
        if not filename.startswith('classwise_'):
            continue
        if 'idis' not in filename.lower():
            continue

        # Check loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match or int(loss_match.group(1)) != loss_level:
            continue

        # Determine R-value
        r_match = re.search(r'_R(\d+)', filename)
        if r_match:
            r_value = f'R{r_match.group(1)}'
        else:
            # Default is R10
            r_value = 'R10'

        filepath = os.path.join(classwise_dir, filename)
        retention_pct, _ = parse_classwise_distribution_file(filepath)

        if retention_pct and r_value in r_values:
            if r_value not in retention_data:
                retention_data[r_value] = retention_pct
            else:
                # Average with existing
                for cls, pct in retention_pct.items():
                    if cls in retention_data[r_value]:
                        retention_data[r_value][cls] = (retention_data[r_value][cls] + pct) / 2
                    else:
                        retention_data[r_value][cls] = pct

    return miou_data, retention_data


def collect_idis_r_value_inference_data(inference_dir, loss_level=90):
    """
    Collect class-wise mIoU from inference (generalization) results for IDIS R-value variants.

    Returns:
        inference_data: {r_value: {class_name: iou}}
    """
    inference_data = {}

    if not os.path.exists(inference_dir):
        return inference_data

    # R-value variants to look for
    r_values = ['R5', 'R10', 'R15', 'R20']

    for filename in os.listdir(inference_dir):
        if not filename.endswith('_metrics.txt'):
            continue
        if 'idis' not in filename.lower():
            continue

        # Check loss level
        loss_match = re.search(r'loss(\d+)', filename)
        if not loss_match or int(loss_match.group(1)) != loss_level:
            continue

        # Determine R-value
        r_match = re.search(r'_R(\d+)_', filename)
        if r_match:
            r_value = f'R{r_match.group(1)}'
        else:
            # Default is R10 for IDIS without explicit R value
            r_value = 'R10'

        filepath = os.path.join(inference_dir, filename)
        class_iou = parse_inference_metrics_file(filepath)

        if class_iou and r_value in r_values:
            if r_value not in inference_data:
                inference_data[r_value] = class_iou
            else:
                # Average with existing (if multiple seeds)
                for cls, iou in class_iou.items():
                    if cls in inference_data[r_value]:
                        inference_data[r_value][cls] = (inference_data[r_value][cls] + iou) / 2
                    else:
                        inference_data[r_value][cls] = iou

    return inference_data


def create_idis_r_radar_chart(ax, data, title, classes, colors, is_percentage=False,
                               baseline_data=None, show_values=True):
    """
    Create a radar chart for IDIS R-value comparison.

    Args:
        ax: matplotlib axes
        data: {r_value: {class_name: value}}
        title: chart title
        classes: list of class names
        colors: color mapping for R-values
        is_percentage: if True, values are percentages
        baseline_data: {class_name: value} for baseline reference
        show_values: whether to show values on labels
    """
    num_vars = len(classes)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Sort R-values in order
    r_order = ['R5', 'R10', 'R15', 'R20']
    r_values = sorted(data.keys(), key=lambda x: r_order.index(x) if x in r_order else 100)

    # Plot baseline first if provided
    if baseline_data:
        values = [baseline_data.get(cls, 0) for cls in classes]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label='Baseline', color='#000000',
                markersize=6, markerfacecolor='#000000', markeredgecolor='black', markeredgewidth=0.5)
        ax.fill(angles, values, alpha=0.1, color='#000000')

    # Plot each R-value with distinct line styles and markers
    for r_value in r_values:
        values = [data[r_value].get(cls, 0) for cls in classes]
        values += values[:1]

        color = colors.get(r_value, '#888888')
        marker = IDIS_R_MARKERS.get(r_value, 'o')
        linestyle = IDIS_R_LINESTYLES.get(r_value, '-')

        ax.plot(angles, values, linestyle=linestyle, marker=marker, linewidth=2.5, color=color,
                markersize=7, markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.8)
        ax.fill(angles, values, alpha=0.10, color=color)

    # Set y-axis limits
    if is_percentage:
        ax.set_ylim(0, 105)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9, fontweight='medium')
    else:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, fontweight='medium')

    ax.set_rlabel_position(10)

    # Draw spoke lines
    for angle in angles[:-1]:
        if is_percentage:
            ax.plot([angle, angle], [0, 103], color='black', alpha=0.5, linewidth=1.2, linestyle=':', zorder=0)
        else:
            ax.plot([angle, angle], [0, 1.03], color='black', alpha=0.5, linewidth=1.2, linestyle=':', zorder=0)

    # Create labels
    labels = []
    for cls in classes:
        if baseline_data and show_values:
            val = baseline_data.get(cls, 0)
            labels.append(f'{val:.2f}\n{cls}')
        else:
            labels.append(cls)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, fontweight='medium')
    ax.tick_params(axis='x', pad=8)

    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.grid(True, color='black', alpha=0.5, linestyle=':', linewidth=1.2)
    ax.xaxis.grid(False)
    ax.spines['polar'].set_visible(False)


def create_idis_r_comparison_figure(miou_data, retention_data, output_dir,
                                     baseline_miou=None, baseline_retention=None,
                                     inference_data=None,
                                     loss_level=90, show_plot=False):
    """
    Create a combined figure comparing IDIS R-values (R=5, 10, 15, 20).
    Two panels: (A) mIoU on Original Data, (B) Retention
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))

    # Part A: Class-wise mIoU (Test → Original Data)
    if inference_data:
        create_idis_r_radar_chart(
            ax1, inference_data,
            '(A) Class-wise mIoU (Test → Original Data)',
            SEMANTICKITTI_CLASSES,
            IDIS_R_COLORS,
            is_percentage=False,
            baseline_data=baseline_miou,
            show_values=True
        )
    else:
        ax1.text(0.5, 0.5, 'No inference data available', ha='center', va='center', transform=ax1.transAxes)

    # Part B: Class-wise Point Retention
    # Normalize retention to 0-1 scale
    retention_normalized = {}
    for r_val, class_data in retention_data.items():
        retention_normalized[r_val] = {cls: val / 100.0 for cls, val in class_data.items()}

    baseline_retention_normalized = None
    if baseline_retention:
        baseline_retention_normalized = {cls: val / 100.0 for cls, val in baseline_retention.items()}

    if retention_normalized:
        create_idis_r_radar_chart(
            ax2, retention_normalized,
            '(B) Class-wise Point Retention',
            SEMANTICKITTI_CLASSES,
            IDIS_R_COLORS,
            is_percentage=False,
            baseline_data=baseline_retention_normalized,
            show_values=False
        )
    else:
        ax2.text(0.5, 0.5, 'No retention data available', ha='center', va='center', transform=ax2.transAxes)

    # Create legend with clear labels including mIoU values (2 decimals)
    legend_elements = []

    # Calculate baseline mIoU (mean across all classes)
    if baseline_miou:
        baseline_mean_miou = np.mean(list(baseline_miou.values()))
        legend_elements.append(Line2D([0], [0], marker='o', color='#000000',
                                      markerfacecolor='#000000', markersize=10,
                                      markeredgecolor='black', markeredgewidth=1,
                                      linewidth=3, linestyle='-',
                                      label=f'Baseline ({baseline_mean_miou:.2f})'))
    elif baseline_retention:
        legend_elements.append(Line2D([0], [0], marker='o', color='#000000',
                                      markerfacecolor='#000000', markersize=10,
                                      markeredgecolor='black', markeredgewidth=1,
                                      linewidth=3, linestyle='-', label='Baseline'))

    for r_val in ['R5', 'R10', 'R15', 'R20']:
        if r_val in miou_data or r_val in retention_data or (inference_data and r_val in inference_data):
            color = IDIS_R_COLORS.get(r_val, '#888888')
            marker = IDIS_R_MARKERS.get(r_val, 'o')
            linestyle = IDIS_R_LINESTYLES.get(r_val, '-')

            # Get R value number for label
            r_num = r_val[1:]  # Extract number from 'R5', 'R10', etc.

            # Calculate mean mIoU for training and inference
            label_parts = [f'IDIS R={r_num}m']
            if r_val in miou_data:
                mean_train = np.mean(list(miou_data[r_val].values()))
                label_parts.append(f'T:{mean_train:.2f}')
            if inference_data and r_val in inference_data:
                mean_inf = np.mean(list(inference_data[r_val].values()))
                label_parts.append(f'G:{mean_inf:.2f}')

            if len(label_parts) > 1:
                label = f'{label_parts[0]} ({", ".join(label_parts[1:])})'
            else:
                label = label_parts[0]

            legend_elements.append(Line2D([0], [0], marker=marker, color=color,
                                          markerfacecolor=color, markersize=10,
                                          markeredgecolor='black', markeredgewidth=1,
                                          linewidth=3, linestyle=linestyle, label=label))

    # Use appropriate columns for legend to fit within figure width
    ncols = min(3, len(legend_elements))
    fig.legend(handles=legend_elements, loc='lower center', ncol=ncols,
               bbox_to_anchor=(0.5, -0.08), fontsize=10, framealpha=0.95,
               edgecolor='black', fancybox=False)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['png', 'svg', 'pdf']:
        output_path = output_dir / f'08_classwise_loss{loss_level}_idis_r_sensitivity.{fmt}'
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"Saved 08_classwise_loss{loss_level}_idis_r_sensitivity figure")

    if show_plot:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate class-wise radar charts for mIoU, inference, and point distribution'
    )
    parser.add_argument(
        '--tables-dir',
        default=str(DEFAULT_TABLES_DIR),
        help='Directory containing metrics files'
    )
    parser.add_argument(
        '--classwise-dir',
        default=str(DEFAULT_CLASSWISE_DIR),
        help='Directory containing classwise distribution files'
    )
    parser.add_argument(
        '--inference-dir',
        default=str(DEFAULT_INFERENCE_DIR),
        help='Directory containing inference metrics files'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT),
        help='Output directory for figures'
    )
    parser.add_argument(
        '--loss-levels',
        nargs='+',
        type=int,
        default=[30, 50, 70, 90],
        help='Loss levels to generate figures for (default: 30 50 70 90)'
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show plots interactively'
    )

    args = parser.parse_args()

    # Verify directories exist
    if not os.path.exists(args.tables_dir):
        print(f"Error: Tables directory not found: {args.tables_dir}")
        sys.exit(1)

    if not os.path.exists(args.classwise_dir):
        print(f"Error: Classwise directory not found: {args.classwise_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Generating Class-wise Radar Charts (A: Training, B: Inference, C: Retention)")
    print("=" * 60)
    print(f"Tables directory: {args.tables_dir}")
    print(f"Classwise directory: {args.classwise_dir}")
    print(f"Inference directory: {args.inference_dir}")
    print(f"Output directory: {args.output}")
    print(f"Loss levels: {args.loss_levels}")
    print()

    # First, collect baseline data (loss 0%) as reference
    print("--- Loading Baseline (Loss 0%) as Reference ---")
    baseline_miou, baseline_dist, baseline_actual_dist = collect_baseline_data(args.tables_dir, args.classwise_dir)
    print(f"  Baseline mIoU classes: {len(baseline_miou.get('baseline', {}))}")
    print(f"  Baseline distribution classes: {len(baseline_dist.get('baseline', {}))}")

    # Extract baseline class data
    baseline_miou_data = baseline_miou.get('baseline', {})
    baseline_dist_data = baseline_dist.get('baseline', {})
    baseline_actual_dist_data = baseline_actual_dist.get('baseline', {})

    # Generate figures for each loss level (excluding 0%)
    for loss_level in args.loss_levels:
        if loss_level == 0:
            print(f"\n--- Skipping Loss Level 0% (used as baseline reference) ---")
            continue

        print(f"\n--- Processing Loss Level: {loss_level}% ---")

        # Collect training data for this loss level
        miou_data, dist_data = collect_data_for_loss_level(
            args.tables_dir, args.classwise_dir, loss_level
        )

        # Collect inference (generalization) data for this loss level
        inference_data = collect_inference_data_for_loss_level(
            args.inference_dir, loss_level
        )

        print(f"  Methods with mIoU data (Training): {list(miou_data.keys())}")
        print(f"  Methods with mIoU data (Inference): {list(inference_data.keys())}")
        print(f"  Methods with distribution data: {list(dist_data.keys())}")

        if not miou_data and not baseline_miou_data:
            print(f"  Warning: No mIoU data found for loss level {loss_level}%")
            continue

        # Filter distribution data to only include methods with mIoU data
        # (baseline is always included as reference)
        methods_with_miou = set(miou_data.keys())
        filtered_dist_data = {m: d for m, d in dist_data.items() if m in methods_with_miou}
        print(f"  Filtered distribution data (methods with mIoU): {list(filtered_dist_data.keys())}")

        # Create figure with baseline as reference
        create_combined_figure(
            miou_data, filtered_dist_data, loss_level, args.output,
            baseline_miou=baseline_miou_data,
            baseline_dist=baseline_dist_data,
            baseline_actual_dist=baseline_actual_dist_data,
            inference_data=inference_data,
            show_plot=args.show
        )

    # Generate IDIS R-value sensitivity radar chart (same style as classwise_loss30.png)
    print("\n" + "=" * 60)
    print("Generating IDIS R-value Sensitivity Radar Chart")
    print("=" * 60)

    # Collect IDIS R-value data at 90% loss
    idis_miou_data, idis_retention_data = collect_idis_r_value_data(
        args.tables_dir, args.classwise_dir, loss_level=90
    )

    # Collect IDIS R-value inference (generalization) data at 90% loss
    idis_inference_data = collect_idis_r_value_inference_data(
        args.inference_dir, loss_level=90
    )

    print(f"  IDIS R-values with mIoU data (Training): {list(idis_miou_data.keys())}")
    print(f"  IDIS R-values with mIoU data (Inference): {list(idis_inference_data.keys())}")
    print(f"  IDIS R-values with retention data: {list(idis_retention_data.keys())}")

    if idis_miou_data or idis_inference_data:
        create_idis_r_comparison_figure(
            idis_miou_data, idis_retention_data, args.output,
            baseline_miou=baseline_miou_data,
            baseline_retention=baseline_dist_data,
            inference_data=idis_inference_data,
            loss_level=90,
            show_plot=args.show
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
