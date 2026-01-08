#!/usr/bin/env python3
"""
DEPOCO Subsampling for SemanticKITTI
====================================

This script generates DEPOCO-subsampled SemanticKITTI data for the benchmark.

IMPORTANT: This script must be run with the DEPOCO virtual environment:
    source $DEPOCO_VENV/bin/activate

Usage:
    python generate_subsampled_depoco.py --loss-levels 10 30 50 70 90
    python generate_subsampled_depoco.py --loss 30

Available loss levels: 10, 30, 50, 70, 90 (mapping to DEPOCO models)
    - 10% → final_skitti_72.5 (actual: 9.5% loss) [VERIFIED]
    - 30% → final_skitti_82.5 (actual: 29.0% loss) [VERIFIED]
    - 50% → final_skitti_87.5 (actual: ~50% loss) [NEW MODEL - needs training]
    - 70% → final_skitti_92.5 (actual: 72.9% loss) [VERIFIED]
    - 90% → final_skitti_97.5 (actual: ~90% loss) [NEW MODEL - needs training]

NOTE: final_skitti_62.5 produces only 4.1% loss, NOT suitable for 50% target!
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# DEPOCO configuration (from environment variables)
# Set these environment variables before running:
#   export DEPOCO_BASE=/path/to/depoco
#   export DEPOCO_VENV=/path/to/depoco/venv
DEPOCO_BASE_PATH = os.environ.get("DEPOCO_BASE", "")
DEPOCO_VENV_PATH = os.environ.get("DEPOCO_VENV", "")

# SemanticKITTI configuration
SEMANTICKITTI_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# Loss level mapping (benchmark % → DEPOCO model)
# NOTE: Verified actual losses from verification reports:
#   final_skitti_72.5 → 9.5% loss (verified) - subsampling_dist=0.524
#   final_skitti_82.5 → 29.0% loss (verified) - subsampling_dist=0.85
#   final_skitti_87.5 → ~50% loss (NEW MODEL) - subsampling_dist=1.1
#   final_skitti_92.5 → 72.9% loss (verified) - subsampling_dist=1.8
#   final_skitti_97.5 → ~90% loss (NEW MODEL) - subsampling_dist=2.8
#
# IMPORTANT: final_skitti_62.5 (subsampling_dist=0.35) produces only 4.1% loss,
#            NOT suitable for 50% target. Use final_skitti_87.5 instead.
LOSS_LEVEL_MAPPING = {
    10: ("final_skitti_72.5", 9.5),    # 9.5% actual loss (verified)
    30: ("final_skitti_82.5", 29.0),   # 29.0% actual loss (verified)
    50: ("final_skitti_87.5", 50),     # ~50% actual loss (NEW MODEL - needs training)
    70: ("final_skitti_92.5", 72.9),   # 72.9% actual loss (verified)
    90: ("final_skitti_97.5", 90),     # ~90% actual loss (NEW MODEL - needs training)
}

VALID_LOSS_LEVELS = list(LOSS_LEVEL_MAPPING.keys())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DEPOCO-subsampled SemanticKITTI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available loss levels and their DEPOCO model mapping:
  10% → final_skitti_72.5 (actual: 9.5% loss) [VERIFIED]
  30% → final_skitti_82.5 (actual: 29.0% loss) [VERIFIED]
  50% → final_skitti_87.5 (actual: ~50% loss) [NEW MODEL - needs training]
  70% → final_skitti_92.5 (actual: 72.9% loss) [VERIFIED]
  90% → final_skitti_97.5 (actual: ~90% loss) [NEW MODEL - needs training]

NOTE: final_skitti_62.5 (subsampling_dist=0.35) produces only 4.1% loss!

Examples:
  # Process all available loss levels
  python generate_subsampled_depoco.py

  # Process specific loss levels
  python generate_subsampled_depoco.py --loss-levels 10 30 50 70 90

  # Process single loss level
  python generate_subsampled_depoco.py --loss 50

  # Dry run (show what would be done)
  python generate_subsampled_depoco.py --dry-run
        """
    )

    parser.add_argument(
        "--loss-levels",
        type=int,
        nargs="+",
        choices=VALID_LOSS_LEVELS,
        help=f"Loss levels to process (choices: {VALID_LOSS_LEVELS})"
    )
    parser.add_argument(
        "--loss",
        type=int,
        choices=VALID_LOSS_LEVELS,
        help="Single loss level to process"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "SemanticKITTI" / "original"),
        help="Input directory with original SemanticKITTI data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "SemanticKITTI" / "subsampled"),
        help="Output directory for subsampled data"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=SEMANTICKITTI_SEQUENCES,
        help="Sequences to process (default: 00-10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed scans (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )

    args = parser.parse_args()

    # Determine loss levels to process
    if args.loss is not None:
        args.loss_levels = [args.loss]
    elif args.loss_levels is None:
        args.loss_levels = VALID_LOSS_LEVELS

    if args.force:
        args.skip_existing = False

    return args


def check_environment():
    """Check if running in DEPOCO environment."""
    # Check environment variables
    if not DEPOCO_BASE_PATH:
        print("Error: DEPOCO_BASE environment variable not set.")
        print("  export DEPOCO_BASE=/path/to/depoco")
        sys.exit(1)

    if not DEPOCO_VENV_PATH:
        print("Error: DEPOCO_VENV environment variable not set.")
        print("  export DEPOCO_VENV=/path/to/depoco/venv")
        sys.exit(1)

    # Check if we can import DEPOCO dependencies
    try:
        import torch
        from ruamel.yaml import YAML
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print(f"\nPlease activate DEPOCO environment:")
        print(f"  source $DEPOCO_VENV/bin/activate")
        sys.exit(1)

    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, using CPU (will be slow)")

    # Check DEPOCO base path
    if not os.path.isdir(DEPOCO_BASE_PATH):
        print(f"Error: DEPOCO not found at {DEPOCO_BASE_PATH}")
        print(f"  Make sure DEPOCO_BASE points to valid DEPOCO installation")
        sys.exit(1)

    print(f"DEPOCO path: {DEPOCO_BASE_PATH}")


def convert_to_native_types(obj):
    """Convert ruamel.yaml CommentedSeq/CommentedMap to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def load_depoco_model(loss_level: int, device: str = "cuda"):
    """
    Load DEPOCO encoder-decoder for a given loss level.

    Returns encoder, decoder, and config.
    """
    import torch
    from ruamel.yaml import YAML

    # Get model info
    model_name, actual_loss = LOSS_LEVEL_MAPPING[loss_level]
    models_path = f"{DEPOCO_BASE_PATH}/main-scripts/paper-1/network_files"
    yamls_path = f"{DEPOCO_BASE_PATH}/yamls/paper-1"

    encoder_path = f"{models_path}/{model_name}/enc_best.pth"
    decoder_path = f"{models_path}/{model_name}/dec_best.pth"
    config_path = f"{yamls_path}/{model_name}.yaml"

    # Check files exist
    for path, name in [(encoder_path, "Encoder"), (decoder_path, "Decoder"), (config_path, "Config")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Add DEPOCO to path and import
    sys.path.insert(0, DEPOCO_BASE_PATH)
    import network_blocks as network

    # Load config (convert ruamel types to native Python types)
    with open(config_path, 'r') as f:
        yaml_loader = YAML()
        config = convert_to_native_types(yaml_loader.load(f))

    # Create models
    encoder = network.Network(config['network']['encoder_blocks'])
    decoder = network.Network(config['network']['decoder_blocks'])

    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    # Move to device and set eval mode
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    print(f"Loaded DEPOCO model: {model_name}")
    print(f"  Target loss: {loss_level}% → Actual loss: {actual_loss}%")

    return encoder, decoder, config, actual_loss


def process_scan(
    encoder,
    decoder,
    points: np.ndarray,
    intensity: np.ndarray,
    labels: np.ndarray,
    device: str = "cuda"
):
    """
    Process a single scan through DEPOCO encoder-decoder.

    Returns compressed points, intensity, and labels.
    """
    import torch
    from scipy.spatial import cKDTree

    # Prepare input
    points_tensor = torch.from_numpy(points).float().to(device)
    features = torch.ones((len(points), 1), dtype=torch.float32, device=device)

    input_dict = {
        'points': points_tensor,
        'features': features
    }

    # Forward pass
    with torch.no_grad():
        out_dict = encoder(input_dict.copy())
        out_dict = decoder(out_dict)

        if out_dict is None:
            # Point cloud too sparse, return original
            return points, intensity, labels

        translation = out_dict['features'][:, :3]
        samples = out_dict['points']
        compressed_points = (samples + translation).cpu().numpy()

    # Reassign labels and intensity via nearest neighbor
    tree = cKDTree(points)
    _, indices = tree.query(compressed_points, k=1)

    compressed_labels = labels[indices]
    compressed_intensity = intensity[indices]

    return compressed_points, compressed_intensity, compressed_labels


def process_loss_level(
    loss_level: int,
    input_dir: str,
    output_dir: str,
    sequences: List[str],
    device: str = "cuda",
    skip_existing: bool = True,
    dry_run: bool = False
):
    """Process all sequences for a given loss level."""
    import torch

    model_name, actual_loss = LOSS_LEVEL_MAPPING[loss_level]
    output_method_dir = os.path.join(output_dir, f"DEPOCO_loss{loss_level}")

    print(f"\n{'='*70}")
    print(f"Processing: DEPOCO loss{loss_level}%")
    print(f"  Model: {model_name} (actual: {actual_loss}% loss)")
    print(f"  Output: {output_method_dir}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Would create directory and process scans")
        return

    # Load model
    encoder, decoder, config, _ = load_depoco_model(loss_level, device)

    # Process each sequence
    total_scans = 0
    total_skipped = 0
    total_time = 0

    for seq in sequences:
        seq_input_dir = os.path.join(input_dir, "sequences", seq)
        seq_output_dir = os.path.join(output_method_dir, "sequences", seq)

        velodyne_input = os.path.join(seq_input_dir, "velodyne")
        labels_input = os.path.join(seq_input_dir, "labels")
        velodyne_output = os.path.join(seq_output_dir, "velodyne")
        labels_output = os.path.join(seq_output_dir, "labels")

        if not os.path.isdir(velodyne_input):
            print(f"  Sequence {seq}: Not found, skipping")
            continue

        # Get list of scans
        scan_files = sorted([f for f in os.listdir(velodyne_input) if f.endswith('.bin')])

        if not scan_files:
            print(f"  Sequence {seq}: No scans found, skipping")
            continue

        # Create output directories
        os.makedirs(velodyne_output, exist_ok=True)
        os.makedirs(labels_output, exist_ok=True)

        print(f"\n  Sequence {seq}: {len(scan_files)} scans")

        # Process scans with progress bar
        seq_time = 0
        seq_skipped = 0

        for scan_file in tqdm(scan_files, desc=f"    Seq {seq}", unit="scan"):
            scan_name = scan_file[:-4]  # Remove .bin

            # Check if already processed
            out_bin = os.path.join(velodyne_output, scan_file)
            out_label = os.path.join(labels_output, f"{scan_name}.label")

            if skip_existing and os.path.isfile(out_bin) and os.path.isfile(out_label):
                seq_skipped += 1
                continue

            # Load input
            bin_path = os.path.join(velodyne_input, scan_file)
            label_path = os.path.join(labels_input, f"{scan_name}.label")

            scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            points = scan[:, :3]
            intensity = scan[:, 3]
            labels = np.fromfile(label_path, dtype=np.uint32)

            # Process
            t_start = time.time()
            comp_points, comp_intensity, comp_labels = process_scan(
                encoder, decoder, points, intensity, labels, device
            )
            seq_time += time.time() - t_start

            # Save output
            comp_scan = np.hstack([comp_points, comp_intensity.reshape(-1, 1)]).astype(np.float32)
            comp_scan.tofile(out_bin)
            comp_labels.astype(np.uint32).tofile(out_label)

        processed = len(scan_files) - seq_skipped
        total_scans += processed
        total_skipped += seq_skipped
        total_time += seq_time

        if processed > 0:
            avg_time = seq_time / processed
            print(f"    Processed: {processed}, Skipped: {seq_skipped}, Avg time: {avg_time:.2f}s/scan")
        else:
            print(f"    All {seq_skipped} scans already exist (skipped)")

    print(f"\n  Summary for loss{loss_level}:")
    print(f"    Total processed: {total_scans} scans")
    print(f"    Total skipped: {total_skipped} scans")
    if total_scans > 0:
        print(f"    Total time: {total_time:.1f}s ({total_time/total_scans:.2f}s/scan avg)")
    print(f"    Output: {output_method_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("DEPOCO Subsampling for SemanticKITTI")
    print("=" * 70)
    print()

    # Check environment
    check_environment()
    print()

    # Show configuration
    print("Configuration:")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Loss levels: {args.loss_levels}")
    print(f"  Sequences: {args.sequences}")
    print(f"  Device: {args.device}")
    print(f"  Skip existing: {args.skip_existing}")
    if args.dry_run:
        print("  [DRY RUN MODE]")
    print()

    # Process each loss level
    start_time = time.time()

    for loss_level in args.loss_levels:
        process_loss_level(
            loss_level=loss_level,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sequences=args.sequences,
            device=args.device,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run
        )

    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print(f"DEPOCO subsampling complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
