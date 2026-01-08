#!/bin/bash
################################################################################
# Phase 1: Subsampling Dataset Generation (CPU-based methods)
#
# This script generates subsampled datasets for the LiDAR subsampling benchmark.
# Uses CPU multiprocessing for parallel processing.
#
################################################################################
# Usage
################################################################################
#
#   ./run_subsampling_phase1_dales_kitti.sh [OPTIONS]
#
#   Options:
#     --method METHOD   Subsampling method (RS, DBSCAN, Voxel, Poisson)
#     --loss LOSS       Loss percentage (10, 30, 50, 70, 90)
#     --seed SEED       Random seed (1, 2, 3) - only for RS/Poisson
#     --dataset DATASET Dataset to process (semantickitti, dales, all)
#     --workers N       Number of parallel workers
#
################################################################################
# Examples
################################################################################
#
#   # Deterministic methods (no seed needed)
#   ./run_subsampling_phase1_dales_kitti.sh --method DBSCAN --loss 50
#   ./run_subsampling_phase1_dales_kitti.sh --method Voxel --loss 50
#
#   # Non-deterministic methods (seed required)
#   ./run_subsampling_phase1_dales_kitti.sh --method RS --loss 50 --seed 1
#   ./run_subsampling_phase1_dales_kitti.sh --method Poisson --loss 50 --seed 1
#
#   # Run with custom worker count
#   ./run_subsampling_phase1_dales_kitti.sh --method RS --loss 50 --seed 1 --workers 32
#
#   # Default: all methods, all loss levels, seed 1 (for non-deterministic)
#   ./run_subsampling_phase1_dales_kitti.sh
#
################################################################################
# Method Properties
################################################################################
#
# RS (Random Sampling):
#   - NON-DETERMINISTIC: Random point selection
#   - Seed required for reproducibility
#   - Output: RS_loss{XX}_seed{N}/
#
# DBSCAN (Density-based Clustering):
#   - DETERMINISTIC: Centroid selection from clusters
#   - No seed needed
#   - Output: DBSCAN_loss{XX}/
#
# Voxel (Voxel Grid Downsampling):
#   - DETERMINISTIC: Grid-based centroid computation
#   - No seed needed
#   - Output: Voxel_loss{XX}/
#
# Poisson (Poisson Disk Sampling):
#   - NON-DETERMINISTIC: Random point order with distance constraint
#   - Seed required for reproducibility
#   - Output: Poisson_loss{XX}_seed{N}/
#
################################################################################
# Configuration
################################################################################
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SemanticKITTI                                                               │
# │   - Sequences: 00-10 (train: 00-07,09,10 | val: 08)                        │
# │   - Output: data/SemanticKITTI/subsampled/{METHOD}_loss{XX}[_seed{N}]/     │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ DALES                                                                       │
# │   - Tiles: 40 tiles (train: 23 | val: 6 | test: 11)                        │
# │   - Output: data/DALES/subsampled/{METHOD}_loss{XX}[_seed{N}]/             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Resume capability:
#   - Script automatically skips already-processed scans/tiles
#   - Safe to re-run after interruption or partial completion
#
# Memory requirements (per worker):
#   - RS: ~1GB       (lightweight, can use many workers)
#   - DBSCAN: ~2GB   (moderate)
#   - Voxel: ~1GB    (lightweight)
#   - Poisson: ~2GB  (moderate)
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
VALID_METHODS=("RS" "DBSCAN" "Voxel" "Poisson")
VALID_LOSS=("10" "30" "50" "70" "90")
VALID_SEEDS=("1" "2" "3")
VALID_DATASETS=("semantickitti" "dales" "all")

# Default values
WORKERS_OVERRIDE=""
DATASET_FILTER="all"    # Default: run both datasets
METHOD_FILTER=""        # Default: all methods (RS, DBSCAN, Voxel, Poisson)
LOSS_FILTER=""          # Default: all loss levels (10, 30, 50, 70, 90)
SEED_FILTER=""          # Default: seed 1 only

################################################################################
# Helper Functions
################################################################################

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Phase 1: Generate subsampled datasets using RS, DBSCAN, Voxel, Poisson"
    echo ""
    echo "Options:"
    echo "  --method METHOD   Subsampling method (RS, DBSCAN, Voxel, Poisson)"
    echo "                    Default: all methods"
    echo "  --loss LOSS       Loss percentage (10, 30, 50, 70, 90)"
    echo "                    Default: all loss levels"
    echo "  --seed SEED       Random seed (1, 2, or 3)"
    echo "                    Required for RS/Poisson, ignored for DBSCAN/Voxel"
    echo "  --dataset DATASET Dataset to process (semantickitti, dales, all)"
    echo "                    Default: all"
    echo "  --workers N       Number of parallel workers (default: 75% of CPU cores)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Method Properties:"
    echo "  RS:      NON-DETERMINISTIC - seed required"
    echo "  DBSCAN:  DETERMINISTIC - no seed needed"
    echo "  Voxel:   DETERMINISTIC - no seed needed"
    echo "  Poisson: NON-DETERMINISTIC - seed required"
    echo ""
    echo "Examples:"
    echo "  # Deterministic methods (no seed needed)"
    echo "  $0 --method DBSCAN --loss 50"
    echo "  $0 --method Voxel --loss 50"
    echo ""
    echo "  # Non-deterministic methods (seed required)"
    echo "  $0 --method RS --loss 50 --seed 1"
    echo "  $0 --method Poisson --loss 50 --seed 1"
    echo ""
    echo "  # All methods (seed applies to RS/Poisson only)"
    echo "  $0 --seed 1"
    echo ""
    echo "  # Custom worker count"
    echo "  $0 --method RS --loss 50 --seed 1 --workers 32"
    echo ""
    echo "Output structure:"
    echo "  Deterministic:     {METHOD}_loss{XX}/"
    echo "  Non-deterministic: {METHOD}_loss{XX}_seed{N}/"
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
        --dataset)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --dataset requires a value"
                exit 1
            fi
            DATASET_FILTER="$2"
            if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET_FILTER} " ]]; then
                echo "Error: Invalid dataset '$DATASET_FILTER'. Must be one of: ${VALID_DATASETS[*]}"
                exit 1
            fi
            shift 2
            ;;
        --workers)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --workers requires a value"
                exit 1
            fi
            WORKERS_OVERRIDE="$2"
            if ! [[ "$WORKERS_OVERRIDE" =~ ^[0-9]+$ ]] || [[ "$WORKERS_OVERRIDE" -eq 0 ]]; then
                echo "Error: --workers must be a positive integer"
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

# Methods to process
if [[ -n "$METHOD_FILTER" ]]; then
    METHODS="$METHOD_FILTER"
else
    METHODS="${VALID_METHODS[*]}"
fi

# Loss levels to process
if [[ -n "$LOSS_FILTER" ]]; then
    LOSS_LEVELS="$LOSS_FILTER"
else
    LOSS_LEVELS="${VALID_LOSS[*]}"
fi

# Deterministic methods don't need seed
DETERMINISTIC_METHODS=("DBSCAN" "Voxel")
NON_DETERMINISTIC_METHODS=("RS" "Poisson")

# Seed handling - depends on method(s)
if [[ -n "$METHOD_FILTER" ]]; then
    # Single method specified
    if [[ " ${NON_DETERMINISTIC_METHODS[@]} " =~ " ${METHOD_FILTER} " ]]; then
        # Non-deterministic method requires seed
        if [[ -z "$SEED_FILTER" ]]; then
            echo "Error: --seed is required for $METHOD_FILTER (non-deterministic method)"
            echo "Example: $0 --method $METHOD_FILTER --loss 50 --seed 1"
            exit 1
        fi
        SEED="$SEED_FILTER"
    else
        # Deterministic method - ignore seed
        if [[ -n "$SEED_FILTER" ]]; then
            echo "Warning: --seed is ignored for $METHOD_FILTER (deterministic method)"
        fi
        SEED=""
    fi
else
    # All methods - seed applies to non-deterministic only
    if [[ -z "$SEED_FILTER" ]]; then
        SEED="1"  # Default seed for non-deterministic methods
    else
        SEED="$SEED_FILTER"
    fi
fi

# Auto-detect optimal worker count
TOTAL_CORES=$(nproc --all)
DEFAULT_WORKERS=$((TOTAL_CORES * 3 / 4))
if [[ $DEFAULT_WORKERS -lt 1 ]]; then
    DEFAULT_WORKERS=1
fi

if [[ -n "$WORKERS_OVERRIDE" ]]; then
    WORKERS=$WORKERS_OVERRIDE
else
    WORKERS=$DEFAULT_WORKERS
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

# Change to project root for relative paths in python scripts
cd "$PROJECT_ROOT"

# Setup logging
LOG_DIR="$PROJECT_ROOT/scripts/preprocessing/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase1_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

# Determine dataset description for output
case $DATASET_FILTER in
    semantickitti) DATASET_DESC="SemanticKITTI only" ;;
    dales)         DATASET_DESC="DALES only" ;;
    all)           DATASET_DESC="Both (SemanticKITTI + DALES)" ;;
esac

# Count items for display
METHOD_COUNT=$(echo $METHODS | wc -w)
LOSS_COUNT=$(echo $LOSS_LEVELS | wc -w)

echo ""
echo "================================================================"
echo "  Phase 1: Subsampling Dataset Generation (CPU Multiprocessing)"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started"
echo ""
echo "Configuration:"
echo "  Dataset:     $DATASET_DESC"
echo "  Methods:     $METHODS ($METHOD_COUNT method(s))"
echo "  Loss levels: $LOSS_LEVELS ($LOSS_COUNT level(s))"
if [[ -n "$SEED" ]]; then
    echo "  Seed:        $SEED (for RS/Poisson)"
else
    echo "  Seed:        N/A (deterministic method)"
fi
if [[ -n "$WORKERS_OVERRIDE" ]]; then
    echo "  Workers:     $WORKERS (user override)"
else
    echo "  Workers:     $WORKERS (auto-detected, 75% of $TOTAL_CORES cores)"
fi
echo "  Log file:    $LOG_FILE"
echo ""
echo "Method Properties:"
echo "  Deterministic (no seed):     DBSCAN, Voxel"
echo "  Non-deterministic (seed):    RS, Poisson"
echo ""

# ============================================================================
# Dataset 1: SemanticKITTI (V2 - All Sequences)
# ============================================================================
if [[ "$DATASET_FILTER" == "all" || "$DATASET_FILTER" == "semantickitti" ]]; then
    echo "================================================================"
    echo "  SemanticKITTI Processing"
    echo "================================================================"
    echo ""
    echo "Sequences: 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10"
    echo "Methods: $METHODS"
    echo "Loss levels: $LOSS_LEVELS"
    if [[ -n "$SEED" ]]; then
        echo "Seed: $SEED (for RS/Poisson)"
    fi
    echo "Output: data/SemanticKITTI/subsampled/{METHOD}_loss{XX}[_seed{N}]/"
    echo ""

    # Build command with optional seed
    if [[ -n "$SEED" ]]; then
        PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_semantickitti_v2.py \
            --methods $METHODS --loss-levels $LOSS_LEVELS --seeds $SEED --workers $WORKERS
    else
        PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_semantickitti_v2.py \
            --methods $METHODS --loss-levels $LOSS_LEVELS --workers $WORKERS
    fi

    SEMANTICKITTI_EXIT=$?
    if [[ $SEMANTICKITTI_EXIT -ne 0 ]]; then
        echo ""
        echo "SemanticKITTI processing failed with exit code: $SEMANTICKITTI_EXIT"
        exit $SEMANTICKITTI_EXIT
    fi

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SemanticKITTI processing completed"
    echo ""
else
    echo "Skipping SemanticKITTI (--dataset $DATASET_FILTER)"
    echo ""
fi

# ============================================================================
# Dataset 2: DALES
# ============================================================================
if [[ "$DATASET_FILTER" == "all" || "$DATASET_FILTER" == "dales" ]]; then
    echo "================================================================"
    echo "  DALES Processing"
    echo "================================================================"
    echo ""
    echo "Tiles: 40 (train: 23, val: 6, test: 11)"
    echo "Methods: $METHODS"
    echo "Loss levels: $LOSS_LEVELS"
    if [[ -n "$SEED" ]]; then
        echo "Seed: $SEED (for RS/Poisson)"
    fi
    echo "Output: data/DALES/subsampled/{METHOD}_loss{XX}[_seed{N}]/"
    echo ""

    # Build command with optional seed
    if [[ -n "$SEED" ]]; then
        PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_dales.py \
            --methods $METHODS --loss-levels $LOSS_LEVELS --seeds $SEED --workers $WORKERS
    else
        PYTHONUNBUFFERED=1 python scripts/preprocessing/generate_subsampled_dales.py \
            --methods $METHODS --loss-levels $LOSS_LEVELS --workers $WORKERS
    fi

    DALES_EXIT=$?
    if [[ $DALES_EXIT -ne 0 ]]; then
        echo ""
        echo "DALES processing failed with exit code: $DALES_EXIT"
        exit $DALES_EXIT
    fi

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DALES processing completed"
    echo ""
else
    echo "Skipping DALES (--dataset $DATASET_FILTER)"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "================================================================"
echo "  Processing Complete"
echo "================================================================"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully"
echo ""
echo "Completed ($DATASET_DESC):"
if [[ "$DATASET_FILTER" == "all" || "$DATASET_FILTER" == "semantickitti" ]]; then
    echo "  - SemanticKITTI: $METHODS ($LOSS_COUNT loss level(s))"
fi
if [[ "$DATASET_FILTER" == "all" || "$DATASET_FILTER" == "dales" ]]; then
    echo "  - DALES: $METHODS ($LOSS_COUNT loss level(s))"
fi
echo ""
echo "Output structure:"
echo "  Methods:     $METHODS"
echo "  Loss levels: $LOSS_LEVELS"
if [[ -n "$SEED" ]]; then
    echo "  Seed:        $SEED (for RS/Poisson only)"
fi
echo ""
echo "  Deterministic:     {METHOD}_loss{XX}/"
echo "  Non-deterministic: {METHOD}_loss{XX}_seed{N}/"
echo ""
echo "Next step: Run Phase 2 (IDIS, FPS - GPU accelerated)"
echo "  ./scripts/run_subsampling_phase2_semantickitti.sh --method IDIS --loss 50"
echo ""
echo "Log file: $LOG_FILE"
