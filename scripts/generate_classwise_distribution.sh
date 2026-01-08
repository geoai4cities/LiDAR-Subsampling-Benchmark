#!/bin/bash
#
# Generate Class-wise Point Distribution Tables for SemanticKITTI
#
# Usage:
#   ./generate_classwise_distribution.sh --method RS --loss 90 --seed 1
#   ./generate_classwise_distribution.sh --method IDIS --loss 90 --r 10
#   ./generate_classwise_distribution.sh --method DBSCAN --loss 30
#   ./generate_classwise_distribution.sh --all
#   ./generate_classwise_distribution.sh --list
#
# Options:
#   --method, -m    Subsampling method (RS, FPS, Poisson, IDIS, DBSCAN, Voxel, DEPOCO, baseline)
#   --loss, -l      Loss level (0, 10, 30, 50, 70, 90)
#   --seed, -s      Seed for non-deterministic methods (RS, FPS, Poisson)
#   --r             R value for IDIS method (5, 10, 15, 20)
#   --all, -a       Analyze all available methods
#   --list          List all available subsampled datasets
#   --help, -h      Show this help message
#

set -e

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/analyze_classwise_distribution.py"
VENV_PYTHON="$BASE_DIR/ptv3_venv/bin/python"
DATA_PATH="$BASE_DIR/data/SemanticKITTI"
OUTPUT_DIR="$BASE_DIR/docs/tables/classwise"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Warning: Virtual environment not found at $VENV_PYTHON"
    echo "Falling back to system Python..."
    VENV_PYTHON="python3"
fi

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Function to show usage
show_help() {
    echo "Generate Class-wise Point Distribution Tables for SemanticKITTI"
    echo ""
    echo "Usage:"
    echo "  $0 --method <METHOD> --loss <LOSS> [--seed <SEED>] [--r <R>]"
    echo "  $0 --all"
    echo "  $0 --list"
    echo ""
    echo "Options:"
    echo "  --method, -m    Subsampling method:"
    echo "                    Non-deterministic: RS, FPS, Poisson (require --seed)"
    echo "                    Deterministic: IDIS, DBSCAN, Voxel, DEPOCO"
    echo "                    Special: baseline (for original data)"
    echo "  --loss, -l      Loss level: 0, 10, 30, 50, 70, 90"
    echo "  --seed, -s      Seed value for non-deterministic methods (1, 2, 3)"
    echo "  --r             R value for IDIS method (5, 10, 15, 20)"
    echo "  --all, -a       Analyze all available subsampled datasets"
    echo "  --list          List all available subsampled datasets"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Analyze Random Sampling at 90% loss with seed 1"
    echo "  $0 --method RS --loss 90 --seed 1"
    echo ""
    echo "  # Analyze IDIS at 90% loss with R=5"
    echo "  $0 --method IDIS --loss 90 --r 5"
    echo ""
    echo "  # Analyze DBSCAN at 30% loss"
    echo "  $0 --method DBSCAN --loss 30"
    echo ""
    echo "  # Analyze baseline (original data)"
    echo "  $0 --method baseline --loss 0"
    echo ""
    echo "  # Analyze all available methods"
    echo "  $0 --all"
    echo ""
    echo "  # List available subsampled datasets"
    echo "  $0 --list"
    echo ""
    echo "Output:"
    echo "  Tables are saved to: $OUTPUT_DIR"
    echo "  Format: classwise_<METHOD>_loss<LOSS>[_seed<SEED>][_R<R>].txt"
}

# Parse arguments
METHOD=""
LOSS=""
SEED=""
R_VALUE=""
ALL_FLAG=""
LIST_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method|-m)
            METHOD="$2"
            shift 2
            ;;
        --loss|-l)
            LOSS="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --r)
            R_VALUE="$2"
            shift 2
            ;;
        --all|-a)
            ALL_FLAG="--all"
            shift
            ;;
        --list)
            LIST_FLAG="--list"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="$VENV_PYTHON $PYTHON_SCRIPT"
CMD="$CMD --data-path $DATA_PATH"
CMD="$CMD --output $OUTPUT_DIR"

if [ -n "$ALL_FLAG" ]; then
    CMD="$CMD --all"
elif [ -n "$LIST_FLAG" ]; then
    CMD="$CMD --list"
else
    # Validate required arguments
    if [ -z "$METHOD" ] || [ -z "$LOSS" ]; then
        echo "Error: --method and --loss are required"
        echo ""
        show_help
        exit 1
    fi

    CMD="$CMD --method $METHOD --loss $LOSS"

    if [ -n "$SEED" ]; then
        CMD="$CMD --seed $SEED"
    fi

    if [ -n "$R_VALUE" ]; then
        CMD="$CMD --r $R_VALUE"
    fi
fi

# Print command
echo "Running: $CMD"
echo ""

# Execute
$CMD
