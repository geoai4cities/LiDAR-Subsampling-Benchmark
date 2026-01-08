#!/bin/bash
################################################################################
# Phase 2: DALES - GPU Accelerated Subsampling (IDIS / FPS)
#
# This script generates subsampled datasets using GPU-accelerated methods.
# Uses pointops CUDA kernels for fast processing.
#
################################################################################
# Usage
################################################################################
#
#   ./run_subsampling_phase2_dales.sh [OPTIONS]
#
#   Options:
#     --method METHOD   Subsampling method (IDIS, FPS)
#     --loss LOSS       Loss percentage (10, 30, 50, 70, 90)
#     --seed SEED       Random seed (1, 2, 3) - only for FPS (IDIS is deterministic)
#     -R RADIUS         IDIS radius in meters (default: 10, options: 5, 10, 15)
#
################################################################################
# Examples
################################################################################
#
#   # IDIS - deterministic (no seed needed)
#   ./run_subsampling_phase2_dales.sh --method IDIS --loss 50
#   ./run_subsampling_phase2_dales.sh --method IDIS -R 5 --loss 50
#   ./run_subsampling_phase2_dales.sh --method IDIS -R 15 --loss 50
#
#   # FPS - requires seed
#   ./run_subsampling_phase2_dales.sh --method FPS --loss 50 --seed 1
#
#   # Generate all loss levels
#   ./run_subsampling_phase2_dales.sh --method IDIS
#   ./run_subsampling_phase2_dales.sh --method FPS --seed 1
#
#   # Default: IDIS with R=10m, all loss levels
#   ./run_subsampling_phase2_dales.sh
#
################################################################################
# Method Properties
################################################################################
#
# IDIS (Inverse Distance Importance Sampling):
#   - DETERMINISTIC: Same input always produces same output
#   - No seed required
#   - Radius parameter (-R): 5m, 10m (default), 15m
#   - Output: IDIS_loss{XX}/ or IDIS_R{5,15}_loss{XX}/
#
# FPS (Farthest Point Sampling):
#   - NON-DETERMINISTIC: Starting point affects results
#   - Seed required for reproducibility
#   - Output: FPS_loss{XX}_seed{N}/
#
################################################################################
# Configuration
################################################################################
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DALES                                                                       │
# │   - Tiles: 40 (train: 23 | val: 6 | test: 11)                              │
# │   - Points per tile: ~11-14 million (much larger than SemanticKITTI)       │
# │   - Output: data/DALES/subsampled/{METHOD}_loss{XX}[_seed{N}]/             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Resume capability:
#   - Script automatically skips already-processed tiles
#   - Safe to re-run after interruption or partial completion
#
# Performance (GPU - NVIDIA A100/H100):
#   - IDIS: GPU-accelerated with pointops
#   - FPS:  GPU-accelerated with pointops
#   - Note: DALES tiles are much larger than SemanticKITTI scans
#
################################################################################

set -euo pipefail

# Trap Ctrl+C and kill entire process group
cleanup() {
    echo ""
    echo "Interrupted by user (Ctrl+C) - killing all processes..."
    kill -TERM -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

################################################################################
# Configuration
################################################################################

# Valid options
VALID_METHODS=("IDIS" "FPS")
VALID_LOSS=("10" "30" "50" "70" "90")
VALID_SEEDS=("1" "2" "3")
VALID_RADIUS=("5" "10" "15")

# Default values
METHOD_FILTER=""        # Default: IDIS
LOSS_FILTER=""          # Default: all loss levels (10, 30, 50, 70, 90)
SEED_FILTER=""          # Default: none (IDIS is deterministic)
RADIUS="10"             # Default: 10m for IDIS

################################################################################
# Helper Functions
################################################################################

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Phase 2: GPU-accelerated subsampling for DALES (IDIS / FPS)"
    echo ""
    echo "Options:"
    echo "  --method METHOD   Subsampling method (IDIS, FPS)"
    echo "                    Default: IDIS"
    echo "  --loss LOSS       Loss percentage (10, 30, 50, 70, 90)"
    echo "                    Default: all loss levels"
    echo "  --seed SEED       Random seed (1, 2, or 3)"
    echo "                    Required for FPS, ignored for IDIS (deterministic)"
    echo "  -R RADIUS         IDIS radius in meters (5, 10, 15)"
    echo "                    Default: 10 (only applies to IDIS method)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Method Properties:"
    echo "  IDIS: DETERMINISTIC - no seed needed, same input = same output"
    echo "  FPS:  NON-DETERMINISTIC - seed required for reproducibility"
    echo ""
    echo "Examples:"
    echo "  # IDIS - deterministic (no seed needed)"
    echo "  $0 --method IDIS --loss 50"
    echo "  $0 --method IDIS -R 5 --loss 50"
    echo "  $0 --method IDIS -R 15 --loss 50"
    echo ""
    echo "  # FPS - requires seed"
    echo "  $0 --method FPS --loss 50 --seed 1"
    echo "  $0 --method FPS --seed 1"
    echo ""
    echo "  # Default: IDIS with R=10m, all loss levels"
    echo "  $0"
    echo ""
    echo "Output structure:"
    echo "  IDIS: data/DALES/subsampled/IDIS_loss{XX}/"
    echo "  IDIS (R!=10): data/DALES/subsampled/IDIS_R{5,15}_loss{XX}/"
    echo "  FPS:  data/DALES/subsampled/FPS_loss{XX}_seed{N}/"
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --method requires a value"
                exit 1
            fi
            METHOD_FILTER="$2"
            if [[ ! " ${VALID_METHODS[@]} " =~ " ${METHOD_FILTER} " ]]; then
                echo "Error: Invalid method '$METHOD_FILTER'. Must be one of: ${VALID_METHODS[*]}"
                exit 1
            fi
            shift 2
            ;;
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
        --seed)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --seed requires a value"
                exit 1
            fi
            SEED_FILTER="$2"
            if [[ ! " ${VALID_SEEDS[@]} " =~ " ${SEED_FILTER} " ]]; then
                echo "Error: Invalid seed '$SEED_FILTER'. Must be one of: ${VALID_SEEDS[*]}"
                exit 1
            fi
            shift 2
            ;;
        -R)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -R requires a value"
                exit 1
            fi
            RADIUS="$2"
            if [[ ! " ${VALID_RADIUS[@]} " =~ " ${RADIUS} " ]]; then
                echo "Error: Invalid radius '$RADIUS'. Must be one of: ${VALID_RADIUS[*]}"
                exit 1
            fi
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

################################################################################
# Set defaults for unspecified options
################################################################################

# Method to process (default: IDIS)
if [[ -n "$METHOD_FILTER" ]]; then
    METHOD="$METHOD_FILTER"
else
    METHOD="IDIS"
fi

# Loss levels to process
if [[ -n "$LOSS_FILTER" ]]; then
    LOSS_LEVELS="$LOSS_FILTER"
else
    LOSS_LEVELS="${VALID_LOSS[*]}"
fi

# Seed handling - only required for FPS (non-deterministic)
# IDIS is deterministic and doesn't need seed
if [[ "$METHOD" == "FPS" ]]; then
    if [[ -z "$SEED_FILTER" ]]; then
        echo "Error: --seed is required for FPS (non-deterministic method)"
        echo "Example: $0 --method FPS --loss 50 --seed 1"
        exit 1
    fi
    SEED="$SEED_FILTER"
else
    # IDIS - ignore seed (deterministic)
    if [[ -n "$SEED_FILTER" ]]; then
        echo "Warning: --seed is ignored for IDIS (deterministic method)"
    fi
    SEED=""
fi

# Get script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate environment
echo "Activating environment..."
if [[ -f "$PROJECT_ROOT/ptv3_venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/ptv3_venv/bin/activate"
elif [[ -f "$PROJECT_ROOT/PTv3/activate.sh" ]]; then
    cd "$PROJECT_ROOT/PTv3"
    source activate.sh
else
    echo "Failed to activate environment"
    echo "Please run PTv3/setup_venv.sh first"
    exit 1
fi

# Change to project root for relative paths
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase2_dales_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

# Count items for display
LOSS_COUNT=$(echo $LOSS_LEVELS | wc -w)

# Determine radius suffix for IDIS (only add suffix for non-default radius)
if [[ "$METHOD" == "IDIS" && "$RADIUS" != "10" ]]; then
    RADIUS_SUFFIX="_R${RADIUS}"
    RADIUS_SUFFIX_ARG="--radius-suffix R${RADIUS}"
else
    RADIUS_SUFFIX=""
    RADIUS_SUFFIX_ARG=""
fi

echo ""
echo "================================================================"
echo "  Phase 2: DALES - GPU Accelerated Subsampling"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""
echo "Configuration:"
echo "  Method:      $METHOD"
if [[ "$METHOD" == "IDIS" ]]; then
    echo "  Radius:      ${RADIUS}m"
    echo "  Deterministic: Yes (no seed needed)"
else
    echo "  Seed:        $SEED"
fi
echo "  Loss levels: $LOSS_LEVELS ($LOSS_COUNT level(s))"
echo "  Log file:    $LOG_FILE"
echo ""
echo "Processing DALES: 40 tiles (train: 23, val: 6, test: 11)"
echo "  Note: Each tile contains ~11-14 million points"
echo ""

# ============================================================================
# Run GPU-accelerated subsampling
# ============================================================================
echo "================================================================"
if [[ "$METHOD" == "IDIS" ]]; then
    echo "  Processing: IDIS (R=${RADIUS}m) [GPU]"
else
    echo "  Processing: FPS [GPU]"
fi
echo "================================================================"
echo ""

if [[ "$METHOD" == "IDIS" ]]; then
    # IDIS - deterministic, no seed needed
    PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_dales_gpu.py \
        --method IDIS \
        --radius ${RADIUS}.0 \
        $RADIUS_SUFFIX_ARG \
        --loss-levels $LOSS_LEVELS
else
    # FPS - requires seed
    PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_dales_gpu.py \
        --method FPS \
        --loss-levels $LOSS_LEVELS \
        --seed $SEED
fi

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "$METHOD failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $METHOD processing completed"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "================================================================"
echo "  Phase 2 Complete - DALES [GPU]"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully"
echo ""
echo "Generated Data:"
echo "  Method:      $METHOD"
if [[ "$METHOD" == "IDIS" ]]; then
    echo "  Radius:      ${RADIUS}m"
    echo "  Deterministic: Yes"
else
    echo "  Seed:        $SEED"
fi
echo "  Loss levels: $LOSS_LEVELS"
echo ""
echo "Total: $LOSS_COUNT subsampled variant(s)"
echo ""
echo "Output directory:"
if [[ "$METHOD" == "IDIS" ]]; then
    echo "  data/DALES/subsampled/IDIS${RADIUS_SUFFIX}_loss{XX}/"
else
    echo "  data/DALES/subsampled/FPS_loss{XX}_seed${SEED}/"
fi
echo ""
echo "Note: DALES tiles are much larger (~11-14M points) than SemanticKITTI scans"
echo ""
echo "Log file: $LOG_FILE"
echo ""
