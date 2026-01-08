#!/bin/bash
################################################################################
# Preprocess SemanticKITTI for DEPOCO Training
#
# This script converts SemanticKITTI data into voxelized submaps required for
# training DEPOCO models. The output is used as training data for DEPOCO.
#
################################################################################
# Requirements
################################################################################
#
# Environment Variables (set before running):
#   DEPOCO_VENV       Path to DEPOCO virtual environment
#   DEPOCO_BASE       Path to DEPOCO project directory
#   DEPOCO_DATA       Output path for preprocessed data
#
# Example:
#   export DEPOCO_VENV=/path/to/venv/py38_depoco
#   export DEPOCO_BASE=/path/to/depoco_for_transfer
#   export DEPOCO_DATA=/path/to/output_submaps
#
################################################################################
# Output Structure
################################################################################
#
#   ${DEPOCO_DATA}/
#   ├── train/           # Submaps from sequences 00-07, 09-10
#   │   ├── 0.bin
#   │   ├── 1.bin
#   │   └── ...
#   ├── validation/      # Submaps from sequence 08
#   │   └── ...
#   └── test/            # (if available)
#       └── ...
#
################################################################################
# Usage
################################################################################
#
#   ./preprocess_semantickitti.sh [OPTIONS]
#
#   Options:
#     --input PATH        SemanticKITTI dataset path (default: data/SemanticKITTI/original)
#     --dry-run           Show what would be done without processing
#     -h, --help          Show this help message
#
################################################################################

set -euo pipefail

# Get script directory and project root
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
    echo "Set it to your output path for preprocessed data, e.g.:"
    echo "  export DEPOCO_DATA=/path/to/output_submaps"
    exit 1
fi

# Default paths
DEFAULT_INPUT="$PROJECT_ROOT/data/SemanticKITTI/original"

# Default values
INPUT_PATH=""
DRY_RUN=""

################################################################################
# Helper Functions
################################################################################

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Preprocess SemanticKITTI for DEPOCO training"
    echo ""
    echo "Options:"
    echo "  --input PATH        SemanticKITTI dataset path"
    echo "                      Default: $DEFAULT_INPUT"
    echo "  --dry-run           Show what would be done without processing"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment Variables (required):"
    echo "  DEPOCO_VENV=$DEPOCO_VENV"
    echo "  DEPOCO_BASE=$DEPOCO_BASE"
    echo "  DEPOCO_DATA=$DEPOCO_DATA"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use defaults"
    echo "  $0 --input /path/to/semantickitti"
    echo "  $0 --dry-run"
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

    # Check input path
    if [[ ! -d "$INPUT_PATH" ]]; then
        echo "Error: Input path not found: $INPUT_PATH"
        exit 1
    fi
    echo "  [OK] Input path found"

    # Check for sequences
    if [[ -d "$INPUT_PATH/sequences" ]]; then
        SEQ_PATH="$INPUT_PATH/sequences"
    elif [[ -d "$INPUT_PATH/00" ]]; then
        SEQ_PATH="$INPUT_PATH"
    else
        echo "Error: Cannot find sequences in $INPUT_PATH"
        exit 1
    fi
    echo "  [OK] Sequences found in $SEQ_PATH"

    # Count available sequences
    local seq_count=0
    for seq in 00 01 02 03 04 05 06 07 08 09 10; do
        if [[ -d "$SEQ_PATH/$seq" ]]; then
            seq_count=$((seq_count + 1))
        fi
    done
    echo "  [OK] Found $seq_count training/validation sequences"

    echo ""
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --input requires a path"
                exit 1
            fi
            INPUT_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
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

# Set defaults if not specified
INPUT_PATH="${INPUT_PATH:-$DEFAULT_INPUT}"

################################################################################
# Main
################################################################################

echo ""
echo "================================================================"
echo "  SemanticKITTI Preprocessing for DEPOCO"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""

# Check requirements
check_requirements

echo "Configuration:"
echo "  Input:       $INPUT_PATH"
echo "  Output:      $DEPOCO_DATA"
echo "  DEPOCO venv: $DEPOCO_VENV"
if [[ -n "$DRY_RUN" ]]; then
    echo "  Mode:        DRY RUN"
fi
echo ""

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/preprocess_depoco_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# Run preprocessing
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/preprocessing/preprocess_semantickitti_for_depoco.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Preprocessing script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build command
CMD="$DEPOCO_VENV/bin/python $PYTHON_SCRIPT"
CMD="$CMD --input $INPUT_PATH"
CMD="$CMD --output $DEPOCO_DATA"

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD --dry-run"
fi

echo "Command: $CMD"
echo ""

# Set PYTHONPATH for DEPOCO modules
export PYTHONPATH="$DEPOCO_BASE:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Execute
if [[ -z "$DRY_RUN" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

$CMD
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preprocessing completed successfully!"
    echo ""

    if [[ -z "$DRY_RUN" ]]; then
        echo "================================================================"
        echo "  Next Steps"
        echo "================================================================"
        echo ""
        echo "1. Train DEPOCO models:"
        echo "   cd configs/depoco"
        echo "   ./train_depoco.sh --loss 50"
        echo "   ./train_depoco.sh --loss 90"
        echo ""
        echo "2. Generate subsampled data:"
        echo "   ./generate_subsampled.sh --loss 10"
        echo "   ./generate_subsampled.sh --loss 30"
        echo ""
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preprocessing failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Log file: $LOG_FILE"
echo ""
