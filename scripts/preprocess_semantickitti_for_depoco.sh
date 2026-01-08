#!/bin/bash
################################################################################
# Preprocess SemanticKITTI for DEPOCO Training
#
# This script converts SemanticKITTI data into voxelized submaps required for
# training DEPOCO models. The output is used as training data for DEPOCO.
#
################################################################################
# Output Structure
################################################################################
#
#   output_dir/
#   ├── train/           # Submaps from sequences 00-07, 09-10
#   │   ├── 0.bin
#   │   ├── 1.bin
#   │   └── ...
#   ├── validation/      # Submaps from sequence 08
#   │   └── ...
#   └── test/            # Submaps from sequences 11-21
#       └── ...
#
################################################################################
# Usage
################################################################################
#
#   ./preprocess_semantickitti_for_depoco.sh [OPTIONS]
#
#   Options:
#     --input PATH        SemanticKITTI dataset path (default: data/SemanticKITTI/original)
#     --output PATH       Output directory (default: $DEPOCO_DATA or data/depoco_preprocessed)
#     --dry-run           Show what would be done without processing
#     -h, --help          Show this help message
#
################################################################################

set -euo pipefail

################################################################################
# Configuration
################################################################################

# DEPOCO virtual environment (has Open3D and dependencies)
# Set via environment variable: export DEPOCO_VENV=/path/to/depoco/venv
DEPOCO_VENV="${DEPOCO_VENV:-}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths
DEFAULT_INPUT="$PROJECT_ROOT/data/SemanticKITTI/original"
DEFAULT_OUTPUT="${DEPOCO_DATA:-$PROJECT_ROOT/data/depoco_preprocessed}"

# Default values
INPUT_PATH=""
OUTPUT_PATH=""
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
    echo "  --output PATH       Output directory for preprocessed data"
    echo "                      Default: $DEFAULT_OUTPUT"
    echo "  --dry-run           Show what would be done without processing"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use defaults"
    echo "  $0 --input /path/to/semantickitti"
    echo "  $0 --output /path/to/output --dry-run"
    echo ""
    echo "After preprocessing, update DEPOCO config files:"
    echo "  grid_output: <output_path>/"
}

check_requirements() {
    echo "Checking requirements..."

    # Check DEPOCO venv
    if [[ -z "$DEPOCO_VENV" ]]; then
        echo "Error: DEPOCO_VENV environment variable not set."
        echo "  export DEPOCO_VENV=/path/to/depoco/venv"
        exit 1
    fi
    if [[ ! -f "$DEPOCO_VENV/bin/activate" ]]; then
        echo "Error: DEPOCO venv not found at $DEPOCO_VENV"
        echo "Please set DEPOCO_VENV to a valid Python environment with:"
        echo "  - open3d"
        echo "  - numpy"
        echo "  - tqdm"
        exit 1
    fi
    echo "  [OK] DEPOCO venv found"

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
        --output)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --output requires a path"
                exit 1
            fi
            OUTPUT_PATH="$2"
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
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT}"

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
echo "  Output:      $OUTPUT_PATH"
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
CMD="$CMD --output $OUTPUT_PATH"

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD --dry-run"
fi

echo "Command: $CMD"
echo ""

# Set PYTHONPATH for octree_handler (if DEPOCO_BASE is set)
if [[ -n "${DEPOCO_BASE:-}" ]]; then
    export PYTHONPATH="${DEPOCO_BASE}:${PYTHONPATH:-}"
fi
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
        echo "1. Update DEPOCO config files to use the new data:"
        echo ""
        echo "   Edit: \$DEPOCO_BASE/yamls/paper-1/final_skitti_87.5.yaml"
        echo "   Edit: \$DEPOCO_BASE/yamls/paper-1/final_skitti_97.5.yaml"
        echo ""
        echo "   Change:"
        echo "     grid_output: \"$OUTPUT_PATH/\""
        echo ""
        echo "2. Train DEPOCO models:"
        echo "   ./scripts/train_depoco_loss50_loss90.sh --loss 50"
        echo "   ./scripts/train_depoco_loss50_loss90.sh --loss 90"
        echo ""
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preprocessing failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Log file: $LOG_FILE"
echo ""
