#!/bin/bash
################################################################################
# Train DEPOCO Models for 50% and 90% Loss
#
# This script trains new DEPOCO encoder-decoder models with specific compression
# ratios for the benchmark.
#
################################################################################
# Model Details
################################################################################
#
# Verified Model Mappings (from actual verification reports):
#   Model Name       │ subsampling_dist │ Actual Loss │ Benchmark Target
#   ─────────────────┼──────────────────┼─────────────┼──────────────────
#   final_skitti_72.5│      0.524       │    9.5%     │    10%  [VERIFIED]
#   final_skitti_82.5│      0.85        │   29.0%     │    30%  [VERIFIED]
#   final_skitti_87.5│      1.3         │   55.2%     │    50%  [VERIFIED]
#   final_skitti_92.5│      1.8         │   72.9%     │    70%  [VERIFIED]
#   final_skitti_97.5│      2.65        │   ~90%      │    90%  [NEW - ADJUSTED]
#   ─────────────────┴──────────────────┴─────────────┴──────────────────
#
# ADJUSTMENT LOG (2026-01-05):
#   - 90% target: subsampling_dist=2.3 achieved only 83.0% loss [VERIFIED]
#   - Adjusted to 2.65 based on extrapolation: 1.8→72.9%, 2.3→83.0%
#
# PREVIOUS CONFIGURATIONS (for reference):
#   # final_skitti_97.5: subsampling_dist=2.3   → 83.0% loss [VERIFIED 2026-01-05]
#   # final_skitti_97.5: subsampling_dist=2.8   → ~99.6% loss (OLD - TOO HIGH)
#
# NOTE: final_skitti_62.5 (subsampling_dist=0.35) produces only 4.1% loss,
#       NOT suitable for 50% target. A new model (final_skitti_87.5) is needed.
#
################################################################################
# Requirements
################################################################################
#
# - DEPOCO virtual environment: Set via DEPOCO_VENV environment variable
# - DEPOCO project: Set via DEPOCO_BASE environment variable
# - Training data: Set via DEPOCO_DATA environment variable
#   (Generate with: ./preprocess_semantickitti_for_depoco.sh)
# - GPU with sufficient memory (recommended: 24GB+)
#
################################################################################
# Usage
################################################################################
#
#   ./train_depoco_loss50_loss90.sh [OPTIONS]
#
#   Options:
#     --loss LOSS       Loss level to train (50 or 90)
#     --resume          Resume training from last checkpoint (auto-detects epoch)
#     --start_epoch N   Override auto-detected epoch (0-indexed, use with --resume)
#     --dry-run         Show configuration without training
#     --epochs N        Override max epochs from config (useful for extending training)
#     --preprocess      Run preprocessing before training (if data missing)
#     -h, --help        Show this help message
#
#   Examples:
#     ./train_depoco_loss50_loss90.sh --loss 50                        # Train 50% loss model from scratch
#     ./train_depoco_loss50_loss90.sh --loss 90                        # Train 90% loss model from scratch
#     ./train_depoco_loss50_loss90.sh --loss 50 --resume               # Resume 50% from last checkpoint (auto-detect)
#     ./train_depoco_loss50_loss90.sh --loss 90 --resume               # Resume 90% from last checkpoint (auto-detect)
#     ./train_depoco_loss50_loss90.sh --loss 50 --resume --start_epoch 24  # Resume from specific epoch 25
#     ./train_depoco_loss50_loss90.sh --loss 50 --dry-run
#     ./train_depoco_loss50_loss90.sh --loss 50 --preprocess           # Preprocess first
#     ./train_depoco_loss50_loss90.sh --loss 50 --resume --epochs 200  # Extend training to 200 epochs
#
#   Resume Training:
#     --resume automatically:
#       - Detects the last completed epoch from checkpoint files
#       - Sets load_pretrained: True (no need to edit config)
#       - Loads model weights and continues training
#     Optionally use --start_epoch N to override auto-detection
#
#   Extend Training:
#     To extend training beyond the original max_epochs:
#     Use --epochs N to override max_epochs from config
#     Example: ./train_depoco_loss50_loss90.sh --loss 50 --resume --epochs 200
#
################################################################################

set -euo pipefail

# Configuration
# Default paths (comment out to use environment variables instead):
DEPOCO_VENV="/DATA/aakash/ms-project/venv/py38_depoco"
DEPOCO_BASE="/DATA/aakash/ms-project/depoco_for_transfer"
DEPOCO_DATA="/DATA/aakash/paper-1/skitti_depoco_new"
# Or set via environment variables:
#   export DEPOCO_VENV=/path/to/depoco/venv
#   export DEPOCO_BASE=/path/to/depoco
#   export DEPOCO_DATA=/path/to/preprocessed/data
# DEPOCO_VENV="${DEPOCO_VENV:-}"
# DEPOCO_BASE="${DEPOCO_BASE:-}"
# DEPOCO_DATA="${DEPOCO_DATA:-}"

# Default values
LOSS_LEVEL=""
RESUME=""
DRY_RUN=""
EPOCHS=""
PREPROCESS=""
START_EPOCH=""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

################################################################################
# Helper Functions
################################################################################

print_usage() {
    echo "Usage: $0 --loss LOSS [OPTIONS]"
    echo ""
    echo "Train DEPOCO model for 50% or 90% loss"
    echo ""
    echo "Options:"
    echo "  --loss LOSS       Loss level to train (50 or 90) [REQUIRED]"
    echo "  --resume          Resume training from last checkpoint (auto-detects epoch)"
    echo "  --start_epoch N   Override auto-detected epoch (0-indexed, use with --resume)"
    echo "  --dry-run         Show configuration without training"
    echo "  --epochs N        Override max epochs from config (useful for extending training)"
    echo "  --preprocess      Run preprocessing before training (if data missing)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available Models:"
    echo "  --loss 50 → final_skitti_87.5 (subsampling_dist=1.3, 55.2% loss) [VERIFIED]"
    echo "  --loss 90 → final_skitti_97.5 (subsampling_dist=2.65, ~90% loss) [ADJUSTED from 2.3→83%]"
    echo ""
    echo "Examples:"
    echo "  $0 --loss 50              # Train 50% loss model"
    echo "  $0 --loss 90              # Train 90% loss model"
    echo "  $0 --loss 50 --dry-run    # Show 50% config without training"
    echo "  $0 --loss 50 --preprocess # Preprocess data first, then train"
    echo ""
    echo "Training data: $DEPOCO_DATA"
    echo "  Generate with: ./preprocess_semantickitti_for_depoco.sh"
    echo ""
    echo "After training, run subsampling:"
    echo "  ./run_subsampling_phase3_semantickitti.sh --loss 50"
    echo "  ./run_subsampling_phase3_semantickitti.sh --loss 90"
}

get_model_config() {
    local loss="$1"
    case $loss in
        50)
            MODEL_NAME="final_skitti_87.5"
            SUBSAMPLING_DIST="1.3"
            TARGET_RETENTION="50%"
            ;;
        90)
            MODEL_NAME="final_skitti_97.5"
            SUBSAMPLING_DIST="2.65"  # ADJUSTED: was 2.3 (83% loss), now 2.65 (~90% loss)
            TARGET_RETENTION="10%"
            ;;
        *)
            echo "Error: Invalid loss level '$loss'. Must be 50 or 90."
            exit 1
            ;;
    esac
    CONFIG_FILE="$DEPOCO_BASE/yamls/paper-1/${MODEL_NAME}.yaml"
}

check_training_data() {
    # Check if training data exists
    if [[ ! -d "$DEPOCO_DATA/train" ]]; then
        echo "  [MISSING] Training data not found at $DEPOCO_DATA/train"
        return 1
    fi

    # Count training files
    local train_count=$(ls "$DEPOCO_DATA/train"/*.bin 2>/dev/null | wc -l)
    if [[ $train_count -eq 0 ]]; then
        echo "  [MISSING] No training files found in $DEPOCO_DATA/train"
        return 1
    fi

    echo "  [OK] Training data found: $train_count submaps in $DEPOCO_DATA/train"
    return 0
}

run_preprocessing() {
    echo ""
    echo "================================================================"
    echo "  Running Preprocessing"
    echo "================================================================"
    echo ""

    PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess_semantickitti_for_depoco.sh"
    if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
        echo "Error: Preprocessing script not found: $PREPROCESS_SCRIPT"
        exit 1
    fi

    echo "Executing: $PREPROCESS_SCRIPT --output $DEPOCO_DATA"
    echo ""

    "$PREPROCESS_SCRIPT" --output "$DEPOCO_DATA"

    if [[ $? -ne 0 ]]; then
        echo "Error: Preprocessing failed"
        exit 1
    fi

    echo ""
    echo "Preprocessing completed. Continuing with training..."
    echo ""
}

check_requirements() {
    echo "Checking requirements..."

    # Check environment variables
    if [[ -z "$DEPOCO_VENV" ]]; then
        echo "Error: DEPOCO_VENV environment variable not set."
        echo "  export DEPOCO_VENV=/path/to/depoco/venv"
        exit 1
    fi
    if [[ -z "$DEPOCO_BASE" ]]; then
        echo "Error: DEPOCO_BASE environment variable not set."
        echo "  export DEPOCO_BASE=/path/to/depoco"
        exit 1
    fi
    if [[ -z "$DEPOCO_DATA" ]]; then
        echo "Error: DEPOCO_DATA environment variable not set."
        echo "  export DEPOCO_DATA=/path/to/preprocessed/data"
        exit 1
    fi

    # Check DEPOCO venv
    if [[ ! -f "$DEPOCO_VENV/bin/activate" ]]; then
        echo "Error: DEPOCO venv not found at $DEPOCO_VENV"
        exit 1
    fi
    echo "  [OK] DEPOCO venv found"

    # Check DEPOCO base
    if [[ ! -d "$DEPOCO_BASE" ]]; then
        echo "Error: DEPOCO project not found at $DEPOCO_BASE"
        exit 1
    fi
    echo "  [OK] DEPOCO project found"

    # Check config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        echo ""
        echo "The config file needs to be created first."
        echo "Run this script with --dry-run to see the expected config."
        exit 1
    fi
    echo "  [OK] Config file found: $CONFIG_FILE"

    # Check training data
    if ! check_training_data; then
        if [[ -n "$PREPROCESS" ]]; then
            run_preprocessing
            # Re-check after preprocessing
            if ! check_training_data; then
                echo "Error: Training data still missing after preprocessing"
                exit 1
            fi
        else
            echo ""
            echo "Error: Training data not found!"
            echo ""
            echo "To generate training data, run:"
            echo "  ./scripts/preprocess_semantickitti_for_depoco.sh --output $DEPOCO_DATA"
            echo ""
            echo "Or use --preprocess flag to do it automatically:"
            echo "  $0 --loss $LOSS_LEVEL --preprocess"
            exit 1
        fi
    fi

    # Find training script
    TRAIN_SCRIPT="$DEPOCO_BASE/main-scripts/paper-1/final_train.py"
    if [[ ! -f "$TRAIN_SCRIPT" ]]; then
        TRAIN_SCRIPT="$DEPOCO_BASE/main-scripts/main2.py"
        if [[ ! -f "$TRAIN_SCRIPT" ]]; then
            TRAIN_SCRIPT="$DEPOCO_BASE/main.py"
            if [[ ! -f "$TRAIN_SCRIPT" ]]; then
                echo "Error: Training script not found"
                exit 1
            fi
        fi
    fi
    echo "  [OK] Training script found: $TRAIN_SCRIPT"

    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  [OK] GPU available:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/      /'
    else
        echo "  [WARN] nvidia-smi not found, training may be slow"
    fi

    echo ""
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --loss)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --loss requires a value (50 or 90)"
                exit 1
            fi
            LOSS_LEVEL="$2"
            if [[ "$LOSS_LEVEL" != "50" && "$LOSS_LEVEL" != "90" ]]; then
                echo "Error: --loss must be 50 or 90, got: $LOSS_LEVEL"
                exit 1
            fi
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --epochs)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --epochs requires a value"
                exit 1
            fi
            EPOCHS="$2"
            shift 2
            ;;
        --preprocess)
            PREPROCESS="true"
            shift
            ;;
        --start_epoch)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --start_epoch requires a value"
                exit 1
            fi
            START_EPOCH="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$LOSS_LEVEL" ]]; then
    echo "Error: --loss is required"
    echo ""
    print_usage
    exit 1
fi

# Get model configuration
get_model_config "$LOSS_LEVEL"

# Set log file name
LOG_FILE="$LOG_DIR/train_depoco_loss${LOSS_LEVEL}_${TIMESTAMP}.log"

################################################################################
# Main
################################################################################

echo ""
echo "================================================================"
echo "  DEPOCO Training: ${LOSS_LEVEL}% Loss Model (${MODEL_NAME})"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""

# Show configuration
echo "Configuration:"
echo "  Model ID:           $MODEL_NAME"
echo "  Target Loss:        ~${LOSS_LEVEL}% (${TARGET_RETENTION} retention)"
echo "  Subsampling Dist:   $SUBSAMPLING_DIST"
echo "  Config File:        $CONFIG_FILE"
echo "  Training Data:      $DEPOCO_DATA"
echo "  DEPOCO venv:        $DEPOCO_VENV"
if [[ -n "$RESUME" ]]; then
    echo "  Mode:               Resume from checkpoint"
fi
if [[ -n "$EPOCHS" ]]; then
    echo "  Epochs Override:    $EPOCHS"
fi
if [[ -n "$PREPROCESS" ]]; then
    echo "  Preprocess:         Yes (will run if data missing)"
fi
if [[ -n "$START_EPOCH" ]]; then
    echo "  Start Epoch:        $START_EPOCH (0-indexed)"
fi
echo "  Log File:           $LOG_FILE"
echo ""

# Show key parameters
echo "Key Parameters:"
echo "  subsampling_dist:   $SUBSAMPLING_DIST (controls compression ratio)"
echo "  max_epochs:         250"
echo "  batch_size:         10"
echo "  max_nr_pts:         500000"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "[DRY RUN] Would execute training with above configuration"
    echo ""
    echo "Config file contents ($CONFIG_FILE):"
    if [[ -f "$CONFIG_FILE" ]]; then
        head -80 "$CONFIG_FILE"
        echo "..."
    else
        echo "  (Config file does not exist yet - needs to be created)"
    fi
    echo ""
    echo "To start training, run without --dry-run flag"
    exit 0
fi

# Check requirements (will exit if config missing)
check_requirements

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  Starting DEPOCO Training"
echo "================================================================"
echo ""

# Build command
CMD="$DEPOCO_VENV/bin/python $TRAIN_SCRIPT --config $CONFIG_FILE"
if [[ -n "$RESUME" ]]; then
    CMD="$CMD $RESUME"
fi
if [[ -n "$EPOCHS" ]]; then
    CMD="$CMD --epochs $EPOCHS"
fi
if [[ -n "$START_EPOCH" ]]; then
    CMD="$CMD --start_epoch $START_EPOCH"
fi

echo "Command: $CMD"
echo ""

# Change to DEPOCO directory (some scripts expect this)
cd "$DEPOCO_BASE/main-scripts/paper-1"
echo "Working directory: $(pwd)"
echo ""

# Execute training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training started..."
echo ""

export PYTHONUNBUFFERED=1
export PYTHONPATH="$DEPOCO_BASE:$DEPOCO_BASE/chamfer-edited/chamfer3D:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

$CMD

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully!"
    echo ""
    echo "================================================================"
    echo "  Training Complete"
    echo "================================================================"
    echo ""
    echo "Model saved to:"
    echo "  $DEPOCO_BASE/main-scripts/paper-1/network_files/$MODEL_NAME/"
    echo ""
    echo "Next steps:"
    echo "  1. Verify model files exist:"
    echo "     ls -la $DEPOCO_BASE/main-scripts/paper-1/network_files/$MODEL_NAME/"
    echo ""
    echo "  2. Generate subsampled data:"
    echo "     cd $PROJECT_ROOT"
    echo "     ./scripts/run_subsampling_phase3_semantickitti.sh --loss $LOSS_LEVEL"
    echo ""
    echo "  3. Verify actual loss percentage:"
    echo "     python scripts/preprocessing/verify_subsampling.py --method DEPOCO --loss $LOSS_LEVEL"
    echo ""
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi

echo "Log file: $LOG_FILE"
echo ""
