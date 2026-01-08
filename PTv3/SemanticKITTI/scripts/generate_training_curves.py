#!/usr/bin/env python3
"""
Generate Training Curves for PTv3 Models
=========================================

Parses train.log files and generates Loss vs Epoch and mIoU vs Epoch curves.
Saves figures in respective model output directories.

Usage:
    python generate_training_curves.py                    # Process all models
    python generate_training_curves.py --model RS_loss70  # Process specific model
    python generate_training_curves.py --list             # List available models
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"


def parse_train_log(log_path: Path) -> dict:
    """
    Parse train.log to extract training metrics.

    Returns:
        dict with keys:
            - train_loss: {epoch: [losses]}
            - val_miou: {epoch: miou}
            - best_miou: {epoch: best_miou_so_far}
            - lr: {epoch: lr}
    """
    data = {
        'train_loss': defaultdict(list),
        'val_miou': {},
        'val_loss': {},
        'best_miou': {},
        'lr': defaultdict(list),
        'iterations': defaultdict(list),
    }

    # Regex patterns
    # Train: [1/10][100/9565] ... loss: 4.3072 Lr: 0.00020
    train_pattern = re.compile(
        r'Train: \[(\d+)/\d+\]\[(\d+)/(\d+)\].*loss:\s*([\d.]+)\s+Lr:\s*([\d.]+)'
    )

    # Val: [4071/4071] Loss 0.6808 mIoU 0.6187
    val_pattern = re.compile(
        r'Val: \[(\d+)/\1\] Loss ([\d.]+) mIoU ([\d.]+)'
    )

    # Currently Best mIoU: 0.6187
    best_miou_pattern = re.compile(
        r'Currently Best mIoU:\s*([\d.]+)'
    )

    current_epoch = 0

    with open(log_path, 'r') as f:
        for line in f:
            # Parse training lines
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                iteration = int(train_match.group(2))
                total_iter = int(train_match.group(3))
                loss = float(train_match.group(4))
                lr = float(train_match.group(5))

                current_epoch = epoch
                data['train_loss'][epoch].append(loss)
                data['lr'][epoch].append(lr)
                data['iterations'][epoch].append(iteration)
                continue

            # Parse validation final line
            val_match = val_pattern.search(line)
            if val_match:
                val_loss = float(val_match.group(2))
                val_miou = float(val_match.group(3))
                data['val_miou'][current_epoch] = val_miou
                data['val_loss'][current_epoch] = val_loss
                continue

            # Parse best mIoU
            best_match = best_miou_pattern.search(line)
            if best_match:
                best_miou = float(best_match.group(1))
                data['best_miou'][current_epoch] = best_miou

    return data


def compute_epoch_averages(data: dict) -> dict:
    """Compute average loss per epoch."""
    epoch_avg = {}
    for epoch, losses in data['train_loss'].items():
        epoch_avg[epoch] = sum(losses) / len(losses)
    return epoch_avg


def generate_training_curves(model_dir: Path, show_plot: bool = False) -> bool:
    """
    Generate training curves for a single model.

    Args:
        model_dir: Path to model output directory
        show_plot: Whether to display the plot interactively

    Returns:
        True if successful, False otherwise
    """
    log_path = model_dir / "train.log"

    if not log_path.exists():
        print(f"  [SKIP] No train.log found in {model_dir.name}")
        return False

    print(f"  Parsing {model_dir.name}...")

    try:
        data = parse_train_log(log_path)
    except Exception as e:
        print(f"  [ERROR] Failed to parse log: {e}")
        return False

    # Check if we have data
    if not data['train_loss']:
        print(f"  [SKIP] No training data found in log")
        return False

    # Compute epoch averages
    epoch_avg_loss = compute_epoch_averages(data)

    # Get sorted epochs
    epochs = sorted(epoch_avg_loss.keys())

    if len(epochs) < 1:
        print(f"  [SKIP] Not enough epochs to plot")
        return False

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract model name for title
    model_name = model_dir.name

    # --- Plot 1: Training Loss vs Epoch ---
    avg_losses = [epoch_avg_loss[e] for e in epochs]
    ax1.plot(epochs, avg_losses, 'b-', linewidth=2, marker='o', markersize=6, label='Train Loss (avg)')

    # Add validation loss if available
    if data['val_loss']:
        val_epochs = sorted(data['val_loss'].keys())
        val_losses = [data['val_loss'][e] for e in val_epochs]
        ax1.plot(val_epochs, val_losses, 'r--', linewidth=2, marker='s', markersize=6, label='Val Loss')

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0.5)

    # --- Plot 2: mIoU vs Epoch ---
    if data['val_miou']:
        val_epochs = sorted(data['val_miou'].keys())
        val_mious = [data['val_miou'][e] for e in val_epochs]
        ax2.plot(val_epochs, val_mious, 'g-', linewidth=2, marker='o', markersize=6, label='Val mIoU')

    if data['best_miou']:
        best_epochs = sorted(data['best_miou'].keys())
        best_mious = [data['best_miou'][e] for e in best_epochs]
        ax2.plot(best_epochs, best_mious, 'purple', linewidth=2, linestyle='--',
                marker='^', markersize=6, label='Best mIoU')

        # Annotate final best mIoU
        final_best = best_mious[-1]
        ax2.axhline(y=final_best, color='purple', linestyle=':', alpha=0.5)
        ax2.annotate(f'Best: {final_best:.4f}',
                    xy=(best_epochs[-1], final_best),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='purple',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('mIoU', fontsize=12, fontweight='bold')
    ax2.set_title('Validation mIoU', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0.5)
    ax2.set_ylim(bottom=0, top=1.0)

    # Main title
    fig.suptitle(f'Training Curves: {model_name}', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figures
    output_path_png = model_dir / "training_curves.png"
    output_path_pdf = model_dir / "training_curves.pdf"

    fig.savefig(output_path_png, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"  Saved: {output_path_png.name}, {output_path_pdf.name}")

    if show_plot:
        plt.show()

    plt.close(fig)

    # Also save a summary text file
    summary_path = model_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Final Train Loss (avg): {avg_losses[-1]:.4f}\n")
        if data['val_loss']:
            f.write(f"Final Val Loss: {list(data['val_loss'].values())[-1]:.4f}\n")
        if data['val_miou']:
            f.write(f"Final Val mIoU: {list(data['val_miou'].values())[-1]:.4f}\n")
        if data['best_miou']:
            f.write(f"Best mIoU: {list(data['best_miou'].values())[-1]:.4f}\n")
        f.write("\n")
        f.write("Epoch-by-Epoch:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val mIoU':<12} {'Best mIoU':<12}\n")
        f.write("-" * 50 + "\n")
        for epoch in epochs:
            train_loss = epoch_avg_loss.get(epoch, float('nan'))
            val_loss = data['val_loss'].get(epoch, float('nan'))
            val_miou = data['val_miou'].get(epoch, float('nan'))
            best_miou = data['best_miou'].get(epoch, float('nan'))
            f.write(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_miou:<12.4f} {best_miou:<12.4f}\n")

    print(f"  Saved: {summary_path.name}")

    return True


def list_models() -> list:
    """List all model directories with train.log files."""
    models = []
    for model_dir in sorted(OUTPUTS_DIR.iterdir()):
        if model_dir.is_dir() and (model_dir / "train.log").exists():
            models.append(model_dir.name)
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Generate training curves for PTv3 models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Specific model to process (partial match supported)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available models')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Show plots interactively')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models with train.log:")
        print("=" * 50)
        for model in list_models():
            print(f"  {model}")
        print(f"\nTotal: {len(list_models())} models")
        return

    print("\n" + "=" * 60)
    print("Training Curves Generator")
    print("=" * 60)

    # Find models to process
    if args.model:
        # Find matching models
        matching = [m for m in list_models() if args.model.lower() in m.lower()]
        if not matching:
            print(f"No models matching '{args.model}' found.")
            return
        models_to_process = matching
    else:
        models_to_process = list_models()

    print(f"\nProcessing {len(models_to_process)} model(s)...\n")

    success_count = 0
    fail_count = 0

    for model_name in models_to_process:
        model_dir = OUTPUTS_DIR / model_name
        if generate_training_curves(model_dir, show_plot=args.show):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Done! Generated curves for {success_count} models.")
    if fail_count > 0:
        print(f"Skipped/Failed: {fail_count} models")
    print("=" * 60)


if __name__ == '__main__':
    main()
