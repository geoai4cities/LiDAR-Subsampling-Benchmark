#!/bin/bash
################################################################################
# Phase 3: SemanticKITTI - DEPOCO (Deep Point Cloud Compression)
#
# This script generates DEPOCO-subsampled SemanticKITTI datasets.
# Uses pre-trained encoder-decoder models for neural compression.
#
################################################################################
# IMPORTANT: DEPOCO Requirements
################################################################################
#
# DEPOCO uses a separate Python environment with specific dependencies.
# This script will automatically activate the DEPOCO venv.
#
# DEPOCO Virtual Environment:
#   Set via: export DEPOCO_VENV=/path/to/depoco/venv
#
# DEPOCO Project Location:
#   Set via: export DEPOCO_BASE=/path/to/depoco
#
# Pre-trained Models:
#   $DEPOCO_BASE/main-scripts/paper-1/network_files/
#
################################################################################
# Usage
################################################################################
#
#   ./run_subsampling_phase3_semantickitti.sh [OPTIONS]
#
#   Options:
#     --loss LOSS       Loss percentage (10, 30, 50, 70, 90)
#                       Default: all available (10, 30, 50, 70, 90)
#     --dry-run         Show what would be done without processing
#     --force           Overwrite existing files
#     -h, --help        Show this help message
#
################################################################################
# Examples
################################################################################
#
#   # Process all available loss levels (10%, 30%, 50%, 70%, 90%)
#   ./run_subsampling_phase3_semantickitti.sh
#
#   # Process specific loss level
#   ./run_subsampling_phase3_semantickitti.sh --loss 30
#
#   # Dry run
#   ./run_subsampling_phase3_semantickitti.sh --dry-run
#
################################################################################
# Available Loss Levels and Model Mapping
################################################################################
#
# DEPOCO models are trained at fixed compression ratios:
#
#   Benchmark Loss  │  DEPOCO Model      │  subsampling_dist │  Actual Loss  │  Status
#   ────────────────┼────────────────────┼───────────────────┼───────────────┼────────────
#       10%         │  final_skitti_72.5 │      0.524        │     9.5%      │  VERIFIED
#       30%         │  final_skitti_82.5 │      0.85         │    29.0%      │  VERIFIED
#       50%         │  final_skitti_87.5 │      1.3          │    55.2%      │  VERIFIED
#       70%         │  final_skitti_92.5 │      1.8          │    72.9%      │  VERIFIED
#       90%         │  final_skitti_97.5 │      2.65         │    ~90%       │  NEEDS TRAINING (ADJUSTED)
#   ────────────────┴────────────────────┴───────────────────┴───────────────┴────────────
#
# ADJUSTMENT LOG (2026-01-05):
#   - 90% target: subsampling_dist=2.3 achieved only 83.0% loss
#   - Adjusted to 2.65 based on extrapolation: 1.8→72.9%, 2.3→83.0%
#   - 50% target: subsampling_dist=1.3 (interpolated from 0.85→29% and 1.8→72.9%)
#
# PREVIOUS CONFIGURATIONS (for reference):
#   # final_skitti_97.5: subsampling_dist=2.3   → 83.0% loss [VERIFIED 2026-01-05]
#   # final_skitti_97.5: subsampling_dist=2.8   → ~99.6% loss (OLD - TOO HIGH)
#
# IMPORTANT: final_skitti_62.5 (subsampling_dist=0.35) produces only 4.1% loss!
#            Do NOT use it for 50% target. Use final_skitti_87.5 instead.
#
# To train new models for 50% and 90%:
#   ./train_depoco_loss50_loss90.sh --loss 50
#   ./train_depoco_loss50_loss90.sh --loss 90
#
################################################################################
# Method Properties
################################################################################
#
# DEPOCO (Deep Point Cloud Compression):
#   - DETERMINISTIC: Neural network inference is deterministic
#   - Uses encoder-decoder architecture with Chamfer loss
#   - Labels reassigned via nearest neighbor (may introduce errors)
#   - GPU recommended for efficient inference
#   - Output: DEPOCO_loss{XX}/
#
################################################################################
# Output Structure
################################################################################
#
#   data/SemanticKITTI/subsampled/
#   ├── DEPOCO_loss10/
#   │   └── sequences/
#   │       ├── 00/
#   │       │   ├── velodyne/
#   │       │   │   ├── 000000.bin
#   │       │   │   └── ...
#   │       │   └── labels/
#   │       │       ├── 000000.label
#   │       │       └── ...
#   │       └── ...
#   ├── DEPOCO_loss30/
#   └── DEPOCO_loss70/
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

# DEPOCO paths
# Default paths (comment out to use environment variables instead):
export DEPOCO_VENV="/DATA/aakash/ms-project/venv/py38_depoco"
export DEPOCO_BASE="/DATA/aakash/ms-project/depoco_for_transfer"
# Or set via environment variables:
#   export DEPOCO_VENV=/path/to/depoco/venv
#   export DEPOCO_BASE=/path/to/depoco
# DEPOCO_VENV="${DEPOCO_VENV:-}"
# DEPOCO_BASE="${DEPOCO_BASE:-}"

# Valid options: 10%, 30%, 50%, 70%, 90% (mapped to closest DEPOCO models)
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
    echo "Phase 3: DEPOCO subsampling for SemanticKITTI"
    echo ""
    echo "Options:"
    echo "  --loss LOSS       Loss percentage (10, 30, 50, 70, 90)"
    echo "                    Default: all available"
    echo "  --dry-run         Show what would be done without processing"
    echo "  --force           Overwrite existing files"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Available Loss Levels:"
    echo "  10% → final_skitti_72.5 (subsampling_dist=0.524, actual: 9.5% loss)  [VERIFIED]"
    echo "  30% → final_skitti_82.5 (subsampling_dist=0.85, actual: 29.0% loss)  [VERIFIED]"
    echo "  50% → final_skitti_87.5 (subsampling_dist=1.3, actual: 55.2% loss)   [VERIFIED]"
    echo "  70% → final_skitti_92.5 (subsampling_dist=1.8, actual: 72.9% loss)   [VERIFIED]"
    echo "  90% → final_skitti_97.5 (subsampling_dist=2.65, actual: ~90% loss)   [NEEDS TRAINING - ADJUSTED]"
    echo ""
    echo "NOTE: final_skitti_62.5 produces only 4.1% loss, NOT for 50% target!"
    echo "      Previous 90% config (2.3) achieved only 83% loss, adjusted to 2.65"
    echo ""
    echo "Examples:"
    echo "  $0                    # Process all available loss levels"
    echo "  $0 --loss 30          # Process 30% loss only"
    echo "  $0 --dry-run          # Show what would be done"
    echo ""
    echo "Output structure:"
    echo "  data/SemanticKITTI/subsampled/DEPOCO_loss{XX}/"
}

check_depoco_available() {
    # Check environment variables are set
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

    # Check DEPOCO venv
    if [[ ! -f "$DEPOCO_VENV/bin/activate" ]]; then
        echo "Error: DEPOCO venv not found at $DEPOCO_VENV"
        echo "Please check DEPOCO installation"
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

    for model in "final_skitti_97.5" "final_skitti_92.5" "final_skitti_82.5" "final_skitti_72.5" "final_skitti_62.5"; do
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
# Set defaults for unspecified options
################################################################################

# Loss levels to process
if [[ -n "$LOSS_FILTER" ]]; then
    LOSS_LEVELS="$LOSS_FILTER"
else
    LOSS_LEVELS="${VALID_LOSS[*]}"
fi

# Get script directory and navigate to project root
# Default paths (comment out to use environment variables instead):
export PROJECT_ROOT="/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark"
# Or set via relative paths:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# Change to project root for relative paths
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase3_semantickitti_depoco_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

# Count items for display
LOSS_COUNT=$(echo $LOSS_LEVELS | wc -w)

echo ""
echo "================================================================"
echo "  Phase 3: SemanticKITTI - DEPOCO Subsampling"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""

# Check DEPOCO availability
echo "Checking DEPOCO installation..."
check_depoco_available
echo ""

echo "Configuration:"
echo "  Method:      DEPOCO (Deep Point Cloud Compression)"
echo "  Loss levels: $LOSS_LEVELS ($LOSS_COUNT level(s))"
echo "  Deterministic: Yes (neural network inference)"
if [[ -n "$DRY_RUN" ]]; then
    echo "  Mode:        DRY RUN"
fi
if [[ -n "$FORCE" ]]; then
    echo "  Force:       Yes (overwrite existing)"
fi
echo "  Log file:    $LOG_FILE"
echo ""
echo "Processing SemanticKITTI sequences: 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10"
echo ""

# ============================================================================
# Activate DEPOCO environment and run
# ============================================================================
echo "================================================================"
echo "  Activating DEPOCO Environment"
echo "================================================================"
echo ""
echo "DEPOCO venv: $DEPOCO_VENV"
echo ""

# Build the command
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/preprocessing/generate_subsampled_depoco.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build arguments
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

# Run with DEPOCO venv (use full path to avoid pyenv conflicts)
echo "Running DEPOCO subsampling..."
echo "Command: $DEPOCO_VENV/bin/python $PYTHON_SCRIPT $ARGS"
echo ""

# Execute with DEPOCO Python directly (avoids pyenv PATH issues)
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

# ============================================================================
# Summary
# ============================================================================
echo "================================================================"
echo "  Phase 3 Complete - SemanticKITTI DEPOCO"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully"
echo ""
echo "Generated Data:"
echo "  Method:      DEPOCO"
echo "  Loss levels: $LOSS_LEVELS"
echo "  Deterministic: Yes"
echo ""
echo "Total: $LOSS_COUNT subsampled variant(s)"
echo ""
echo "Output directories:"
for loss in $LOSS_LEVELS; do
    echo "  data/SemanticKITTI/subsampled/DEPOCO_loss${loss}/"
done
echo ""
echo "Loss Level Mapping:"
echo "  10% → final_skitti_72.5 (subsampling_dist=0.524, actual: 9.5% loss)  [VERIFIED]"
echo "  30% → final_skitti_82.5 (subsampling_dist=0.85, actual: 29.0% loss)  [VERIFIED]"
echo "  50% → final_skitti_87.5 (subsampling_dist=1.3, actual: 55.2% loss)   [VERIFIED]"
echo "  70% → final_skitti_92.5 (subsampling_dist=1.8, actual: 72.9% loss)   [VERIFIED]"
echo "  90% → final_skitti_97.5 (subsampling_dist=2.65, actual: ~90% loss)   [NEEDS TRAINING - ADJUSTED]"
echo ""
echo "Note: DEPOCO uses Chamfer loss for geometry, not semantic preservation."
echo "Labels are reassigned via nearest neighbor, which may introduce errors."
echo ""
echo "Next step: Start training experiments"
echo "  cd PTv3/SemanticKITTI/scripts"
echo "  ./train_semantickitti_140gb.sh --method DEPOCO --loss 30 start"
echo ""
echo "Log file: $LOG_FILE"
echo ""
