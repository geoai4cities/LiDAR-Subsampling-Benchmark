#!/bin/bash
################################################################################
# Generate DEPOCO Subsampled Data
#
# This script generates DEPOCO-subsampled SemanticKITTI datasets using
# pre-trained encoder-decoder models.
#
################################################################################
# Requirements
################################################################################
#
# Environment Variables (set before running):
#   DEPOCO_VENV       Path to DEPOCO virtual environment
#   DEPOCO_BASE       Path to DEPOCO project directory
#
# Example:
#   export DEPOCO_VENV=/path/to/venv/py38_depoco
#   export DEPOCO_BASE=/path/to/depoco_for_transfer
#
################################################################################
# Available Loss Levels and Model Mapping
################################################################################
#
#   Benchmark Loss  │  DEPOCO Model      │  subsampling_dist │  Actual Loss
#   ────────────────┼────────────────────┼───────────────────┼───────────────
#       10%         │  final_skitti_72.5 │      0.524        │     9.5%
#       30%         │  final_skitti_82.5 │      0.85         │    29.0%
#       50%         │  final_skitti_87.5 │      1.3          │    ~50%
#       70%         │  final_skitti_92.5 │      1.8          │    72.9%
#       90%         │  final_skitti_97.5 │      2.3          │    ~90%
#
################################################################################
# Usage
################################################################################
#
#   ./generate_subsampled.sh [OPTIONS]
#
#   Options:
#     --loss LOSS       Loss percentage (10, 30, 50, 70, 90)
#                       Default: all available
#     --dry-run         Show what would be done without processing
#     --force           Overwrite existing files
#     -h, --help        Show this help message
#
#   Examples:
#     ./generate_subsampled.sh                    # All loss levels
#     ./generate_subsampled.sh --loss 30          # 30% loss only
#     ./generate_subsampled.sh --dry-run          # Show what would be done
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

# Valid options
VALID_LOSS=("10" "30" "50" "70" "90")

# Default values
LOSS_FILTER=""
DRY_RUN=""
FORCE=""

################################################################################
# Helper Functions
################################################################################

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate DEPOCO subsampled SemanticKITTI data"
    echo ""
    echo "Options:"
    echo "  --loss LOSS       Loss percentage (10, 30, 50, 70, 90)"
    echo "                    Default: all available"
    echo "  --dry-run         Show what would be done without processing"
    echo "  --force           Overwrite existing files"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available Loss Levels:"
    echo "  10% → final_skitti_72.5 (actual: 9.5% loss)"
    echo "  30% → final_skitti_82.5 (actual: 29.0% loss)"
    echo "  50% → final_skitti_87.5 (actual: ~50% loss)"
    echo "  70% → final_skitti_92.5 (actual: 72.9% loss)"
    echo "  90% → final_skitti_97.5 (actual: ~90% loss)"
    echo ""
    echo "Output structure:"
    echo "  data/SemanticKITTI/subsampled/DEPOCO_loss{XX}/"
}

check_depoco_available() {
    # Check DEPOCO venv
    if [[ ! -f "$DEPOCO_VENV/bin/activate" ]]; then
        echo "Error: DEPOCO venv not found at $DEPOCO_VENV"
        exit 1
    fi

    # Check DEPOCO base
    if [[ ! -d "$DEPOCO_BASE" ]]; then
        echo "Error: DEPOCO project not found at $DEPOCO_BASE"
        exit 1
    fi

    # Check models exist
    local models_path="$DEPOCO_BASE/main-scripts/paper-1/network_files"
    local found_models=0

    for model in "final_skitti_97.5" "final_skitti_92.5" "final_skitti_87.5" "final_skitti_82.5" "final_skitti_72.5"; do
        if [[ -f "$models_path/$model/enc_best.pth" ]]; then
            found_models=$((found_models + 1))
        fi
    done

    if [[ $found_models -eq 0 ]]; then
        echo "Error: No DEPOCO models found in $models_path"
        exit 1
    fi

    echo "Found $found_models DEPOCO model(s)"
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
            LOSS_FILTER="$2"
            if [[ ! " ${VALID_LOSS[@]} " =~ " ${LOSS_FILTER} " ]]; then
                echo "Error: Invalid loss '$LOSS_FILTER'. Must be one of: ${VALID_LOSS[*]}"
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
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

################################################################################
# Main
################################################################################

# Loss levels to process
if [[ -n "$LOSS_FILTER" ]]; then
    LOSS_LEVELS="$LOSS_FILTER"
else
    LOSS_LEVELS="${VALID_LOSS[*]}"
fi

# Change to project root
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/depoco_subsampling_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "================================================================"
echo "  DEPOCO Subsampling for SemanticKITTI"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""

# Check DEPOCO availability
echo "Checking DEPOCO installation..."
check_depoco_available
echo ""

echo "Configuration:"
echo "  Loss levels: $LOSS_LEVELS"
if [[ -n "$DRY_RUN" ]]; then
    echo "  Mode:        DRY RUN"
fi
if [[ -n "$FORCE" ]]; then
    echo "  Force:       Yes (overwrite existing)"
fi
echo "  Log file:    $LOG_FILE"
echo ""

# Build arguments
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/preprocessing/generate_subsampled_depoco.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Script not found: $PYTHON_SCRIPT"
    exit 1
fi

ARGS=""
if [[ -n "$LOSS_FILTER" ]]; then
    ARGS="$ARGS --loss $LOSS_FILTER"
else
    ARGS="$ARGS --loss-levels $LOSS_LEVELS"
fi

if [[ -n "$DRY_RUN" ]]; then
    ARGS="$ARGS --dry-run"
fi

if [[ -n "$FORCE" ]]; then
    ARGS="$ARGS --force"
fi

# Run with DEPOCO venv
echo "Running DEPOCO subsampling..."
echo "Command: $DEPOCO_VENV/bin/python $PYTHON_SCRIPT $ARGS"
echo ""

export PYTHONUNBUFFERED=1
"$DEPOCO_VENV/bin/python" "$PYTHON_SCRIPT" $ARGS

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "DEPOCO processing failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPOCO processing completed"
echo ""
echo "Output directories:"
for loss in $LOSS_LEVELS; do
    echo "  data/SemanticKITTI/subsampled/DEPOCO_loss${loss}/"
done
echo ""
echo "Log file: $LOG_FILE"
echo ""
