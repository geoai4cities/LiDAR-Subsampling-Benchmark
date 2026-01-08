#!/bin/bash
################################################################################
# Train DEPOCO Models
#
# This script trains DEPOCO encoder-decoder models with specific compression
# ratios for the benchmark.
#
################################################################################
# Model Details
################################################################################
#
# Verified Model Mappings:
#   Model Name       │ subsampling_dist │ Actual Loss │ Benchmark Target
#   ─────────────────┼──────────────────┼─────────────┼──────────────────
#   final_skitti_72.5│      0.524       │    9.5%     │    10%  [VERIFIED]
#   final_skitti_82.5│      0.85        │   29.0%     │    30%  [VERIFIED]
#   final_skitti_87.5│      1.3         │   ~50%      │    50%  [NEW]
#   final_skitti_92.5│      1.8         │   72.9%     │    70%  [VERIFIED]
#   final_skitti_97.5│      2.3         │   ~90%      │    90%  [NEW]
#
################################################################################
# Requirements
################################################################################
#
# Environment Variables (set before running):
#   DEPOCO_VENV       Path to DEPOCO virtual environment
#   DEPOCO_BASE       Path to DEPOCO project directory
#   DEPOCO_DATA       Path to preprocessed training data
#
# Example:
#   export DEPOCO_VENV=/path/to/venv/py38_depoco
#   export DEPOCO_BASE=/path/to/depoco_for_transfer
#   export DEPOCO_DATA=/path/to/skitti_depoco_new
#
################################################################################
# Usage
################################################################################
#
#   ./train_depoco.sh [OPTIONS]
#
#   Options:
#     --loss LOSS       Loss level to train (10, 30, 50, 70, 90)
#     --resume          Resume training from last checkpoint
#     --dry-run         Show configuration without training
#     --epochs N        Override max epochs (default: 250)
#     -h, --help        Show this help message
#
#   Examples:
#     ./train_depoco.sh --loss 50         # Train 50% loss model
#     ./train_depoco.sh --loss 90         # Train 90% loss model
#     ./train_depoco.sh --loss 50 --dry-run
#
################################################################################

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check environment variables
if [[ -z "${DEPOCO_VENV:-}" ]]; then
    echo "Error: DEPOCO_VENV environment variable not set"
    echo "Set it to your DEPOCO virtual environment path, e.g.:"
    echo "  export DEPOCO_VENV=/path/to/venv/py38_depoco"
    exit 1
fi

if [[ -z "${DEPOCO_BASE:-}" ]]; then
    echo "Error: DEPOCO_BASE environment variable not set"
    echo "Set it to your DEPOCO project path, e.g.:"
    echo "  export DEPOCO_BASE=/path/to/depoco_for_transfer"
    exit 1
fi

if [[ -z "${DEPOCO_DATA:-}" ]]; then
    echo "Error: DEPOCO_DATA environment variable not set"
    echo "Set it to your preprocessed training data path, e.g.:"
    echo "  export DEPOCO_DATA=/path/to/skitti_depoco_new"
    exit 1
fi

# Default values
LOSS_LEVEL=""
RESUME=""
DRY_RUN=""
EPOCHS=""

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
    echo "Train DEPOCO model for specified loss level"
    echo ""
    echo "Options:"
    echo "  --loss LOSS       Loss level to train (10, 30, 50, 70, 90) [REQUIRED]"
    echo "  --resume          Resume training from last checkpoint"
    echo "  --dry-run         Show configuration without training"
    echo "  --epochs N        Override max epochs (default: 250)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available Models:"
    echo "  --loss 10 → final_skitti_72.5 (subsampling_dist=0.524)"
    echo "  --loss 30 → final_skitti_82.5 (subsampling_dist=0.85)"
    echo "  --loss 50 → final_skitti_87.5 (subsampling_dist=1.3)"
    echo "  --loss 70 → final_skitti_92.5 (subsampling_dist=1.8)"
    echo "  --loss 90 → final_skitti_97.5 (subsampling_dist=2.3)"
    echo ""
    echo "Environment Variables (required):"
    echo "  DEPOCO_VENV=$DEPOCO_VENV"
    echo "  DEPOCO_BASE=$DEPOCO_BASE"
    echo "  DEPOCO_DATA=$DEPOCO_DATA"
}

get_model_config() {
    local loss="$1"
    case $loss in
        10)
            MODEL_NAME="final_skitti_72.5"
            SUBSAMPLING_DIST="0.524"
            ;;
        30)
            MODEL_NAME="final_skitti_82.5"
            SUBSAMPLING_DIST="0.85"
            ;;
        50)
            MODEL_NAME="final_skitti_87.5"
            SUBSAMPLING_DIST="1.3"
            ;;
        70)
            MODEL_NAME="final_skitti_92.5"
            SUBSAMPLING_DIST="1.8"
            ;;
        90)
            MODEL_NAME="final_skitti_97.5"
            SUBSAMPLING_DIST="2.3"
            ;;
        *)
            echo "Error: Invalid loss level '$loss'. Must be 10, 30, 50, 70, or 90."
            exit 1
            ;;
    esac

    # Use config from this repo
    CONFIG_FILE="$SCRIPT_DIR/${MODEL_NAME}.yaml"
}

check_requirements() {
    echo "Checking requirements..."

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
        exit 1
    fi
    echo "  [OK] Config file found: $CONFIG_FILE"

    # Check training data
    if [[ ! -d "$DEPOCO_DATA/train" ]]; then
        echo "Error: Training data not found at $DEPOCO_DATA/train"
        echo "Generate with: python scripts/preprocessing/preprocess_semantickitti_for_depoco.py"
        exit 1
    fi
    echo "  [OK] Training data found"

    # Find training script
    TRAIN_SCRIPT="$DEPOCO_BASE/main-scripts/paper-1/final_train.py"
    if [[ ! -f "$TRAIN_SCRIPT" ]]; then
        TRAIN_SCRIPT="$DEPOCO_BASE/main.py"
        if [[ ! -f "$TRAIN_SCRIPT" ]]; then
            echo "Error: Training script not found"
            exit 1
        fi
    fi
    echo "  [OK] Training script found: $TRAIN_SCRIPT"

    echo ""
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --loss)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --loss requires a value"
                exit 1
            fi
            LOSS_LEVEL="$2"
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
echo "  Target Loss:        ~${LOSS_LEVEL}%"
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
echo "  Log File:           $LOG_FILE"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "[DRY RUN] Would execute training with above configuration"
    exit 0
fi

# Check requirements
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

echo "Command: $CMD"
echo ""

# Change to DEPOCO directory
cd "$DEPOCO_BASE/main-scripts/paper-1"
echo "Working directory: $(pwd)"
echo ""

# Execute training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training started..."
echo ""

export PYTHONUNBUFFERED=1
export PYTHONPATH="$DEPOCO_BASE:$DEPOCO_BASE/chamfer-edited/chamfer3D:${PYTHONPATH:-}"

$CMD

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Generate subsampled data:"
    echo "     ./scripts/run_subsampling_phase3_semantickitti.sh --loss $LOSS_LEVEL"
    echo ""
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Log file: $LOG_FILE"
echo ""
