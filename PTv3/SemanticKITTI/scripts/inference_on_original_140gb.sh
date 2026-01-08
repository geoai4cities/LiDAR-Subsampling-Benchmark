#!/bin/bash

################################################################################
# Inference Script for SemanticKITTI - Evaluate Models on ORIGINAL Data
#
# This script runs inference using models trained on SUBSAMPLED data and
# evaluates them on the ORIGINAL (full resolution) validation data.
# This measures how well models trained on subsampled data generalize to
# full-resolution point clouds.
#
# Key difference from inference_semantickitti_140gb.sh:
#   - That script: Fixed baseline MODEL, variable SUBSAMPLED data
#   - This script: Variable trained MODELS, fixed ORIGINAL data
#
# Usage:
#   ./inference_on_original_140gb.sh --method METHOD --loss LOSS [OPTIONS] COMMAND
#   ./inference_on_original_140gb.sh --all [OPTIONS] COMMAND
#
# Examples:
#   # Run ALL trained models on original data (sequentially)
#   ./inference_on_original_140gb.sh --all run
#   ./inference_on_original_140gb.sh --all status
#
#   # Deterministic methods (no seed needed)
#   ./inference_on_original_140gb.sh --method IDIS --loss 90 run
#   ./inference_on_original_140gb.sh --method DBSCAN --loss 50 run
#   ./inference_on_original_140gb.sh --method Voxel --loss 90 run
#
#   # Non-deterministic methods (seed required)
#   ./inference_on_original_140gb.sh --method RS --loss 90 --seed 1 run
#   ./inference_on_original_140gb.sh --method FPS --loss 90 --seed 1 run
#   ./inference_on_original_140gb.sh --method Poisson --loss 90 --seed 1 run
#
#   # IDIS R-value ablation
#   ./inference_on_original_140gb.sh --method IDIS_R5 --loss 90 run
#   ./inference_on_original_140gb.sh --method IDIS_R15 --loss 90 run
#   ./inference_on_original_140gb.sh --method IDIS_R20 --loss 90 run
#
#   # Run baseline model on original data (for comparison)
#   ./inference_on_original_140gb.sh --method baseline --loss 0 run
#
#   # Check status
#   ./inference_on_original_140gb.sh --method IDIS --loss 90 status
#
# Validation Set: Sequence 08 only (SemanticKITTI standard split)
# Data: ORIGINAL (full resolution) - /data/SemanticKITTI/original/
#
# Methods:
#   Deterministic:     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel, DEPOCO
#   Non-deterministic: RS, FPS, Poisson
#   Special:           baseline (0% loss, trained on original)
#
# Loss Levels: 0 (baseline), 30, 50, 70, 90
#
# Output:
#   Results saved to: outputs/inference_on_original/{METHOD}_loss{LOSS}[_seed{N}]/
#   - inference.log: Full inference log
#   - inference_metrics.txt: Timing and memory stats
#   - result/: mIoU and per-class results
#
# NOTE: GPU memory values reported in inference_metrics.txt are captured AFTER
#       the inference process completes (via nvidia-smi), not during execution.
#       These values may not reflect actual peak memory usage during inference.
#       For accurate inference memory profiling, use PyTorch's built-in memory
#       tracking (torch.cuda.max_memory_allocated()) within the inference code.
#
################################################################################

set -euo pipefail

################################################################################
# USER CONFIGURATION
################################################################################

# Default values
METHOD=""
LOSS=""
SEED="1"
GPU_ID="0"
COMMAND=""
RUN_ALL=false

################################################################################
# Parse Arguments
################################################################################

print_usage() {
    echo "Usage: $0 --method METHOD --loss LOSS [--seed SEED] [--gpu GPU_ID] COMMAND"
    echo "       $0 --all [--gpu GPU_ID] COMMAND"
    echo ""
    echo "Required Arguments (choose one):"
    echo "  --method METHOD   Method used for training the model"
    echo "  --loss LOSS       Loss percentage (0, 30, 50, 70, 90)"
    echo "  --all             Run ALL trained models on original data (sequential)"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed SEED       Random seed for non-deterministic methods (1, 2, or 3)"
    echo "  --gpu GPU_ID      GPU device ID (default: 0)"
    echo ""
    echo "Commands:"
    echo "  run               Run inference"
    echo "  stop              Stop running inference"
    echo "  status            Show inference status and results"
    echo "  logs              View inference logs"
    echo "  help              Show this help message"
    echo ""
    echo "Methods:"
    echo "  Deterministic:     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel, DEPOCO"
    echo "  Non-deterministic: RS, FPS, Poisson"
    echo "  Special:           baseline (use with --loss 0)"
    echo ""
    echo "Examples:"
    echo "  # Run ALL trained models"
    echo "  $0 --all run"
    echo "  $0 --all status"
    echo ""
    echo "  # Baseline model"
    echo "  $0 --method baseline --loss 0 run"
    echo ""
    echo "  # Deterministic methods"
    echo "  $0 --method IDIS --loss 90 run"
    echo "  $0 --method DBSCAN --loss 50 run"
    echo "  $0 --method Voxel --loss 90 run"
    echo ""
    echo "  # Non-deterministic methods (seed required)"
    echo "  $0 --method RS --loss 90 --seed 1 run"
    echo "  $0 --method FPS --loss 90 --seed 2 run"
    echo "  $0 --method Poisson --loss 90 --seed 3 run"
    echo ""
    echo "  # IDIS R-value ablation"
    echo "  $0 --method IDIS_R5 --loss 90 run"
    echo "  $0 --method IDIS_R15 --loss 90 run"
    echo "  $0 --method IDIS_R20 --loss 90 run"
    echo ""
    echo "Notes:"
    echo "  - Validation set: Sequence 08 only (SemanticKITTI standard split)"
    echo "  - Data: ORIGINAL (full resolution) validation data"
    echo "  - Data directory: data/SemanticKITTI/original/"
    echo ""
    echo "Output:"
    echo "  Results: outputs/inference_on_original/{METHOD}_loss{LOSS}[_seed{N}]/"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        run|stop|status|logs|help)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate arguments
VALID_METHODS=("baseline" "RS" "IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "FPS" "DBSCAN" "Voxel" "Poisson" "DEPOCO")
VALID_LOSS=("0" "30" "50" "70" "90")

# Deterministic methods don't need seed
DETERMINISTIC_METHODS=("baseline" "IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "DBSCAN" "Voxel" "DEPOCO")
NON_DETERMINISTIC_METHODS=("RS" "FPS" "Poisson")

# Handle help command
if [[ "$COMMAND" == "help" ]]; then
    print_usage
    exit 0
fi

# Show help if no arguments
if [[ -z "$COMMAND" && -z "$METHOD" && -z "$LOSS" && "$RUN_ALL" == false ]]; then
    print_usage
    exit 0
fi

# Validate command
if [[ -z "$COMMAND" ]]; then
    echo "Error: Command required (run|stop|status|logs|help)"
    print_usage
    exit 1
fi

# If --all is specified, skip individual method/loss validation
if [[ "$RUN_ALL" == false ]]; then
    # Validate required arguments
    if [[ -z "$METHOD" || -z "$LOSS" ]]; then
        echo "Error: --method and --loss are required (or use --all)"
        print_usage
        exit 1
    fi

    if [[ ! " ${VALID_METHODS[@]} " =~ " ${METHOD} " ]]; then
        echo "Error: Invalid method '$METHOD'. Must be one of: ${VALID_METHODS[*]}"
        exit 1
    fi

    if [[ ! " ${VALID_LOSS[@]} " =~ " ${LOSS} " ]]; then
        echo "Error: Invalid loss '$LOSS'. Must be one of: ${VALID_LOSS[*]}"
        exit 1
    fi

    # Seed validation based on method type
    if [[ " ${NON_DETERMINISTIC_METHODS[@]} " =~ " ${METHOD} " ]]; then
        if [[ ! "$SEED" =~ ^[123]$ ]]; then
            echo "Error: Invalid seed '$SEED'. Must be 1, 2, or 3"
            exit 1
        fi
    elif [[ " ${DETERMINISTIC_METHODS[@]} " =~ " ${METHOD} " ]]; then
        SEED=""
    fi
fi

################################################################################
# Configuration
################################################################################

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PTv3_ROOT="$(dirname "$PROJECT_ROOT")"
POINTCEPT_DIR="$PTv3_ROOT/pointcept"
VENV_PATH="$PTv3_ROOT/../ptv3_venv"
DATA_ROOT="$PTv3_ROOT/../data/SemanticKITTI"

# ORIGINAL data path (FIXED - always use original validation data)
ORIGINAL_DATA_PATH="$DATA_ROOT/original"
ORIGINAL_DATA_PATH_RELATIVE="../../data/SemanticKITTI/original"

# Model path (VARIABLE - based on method and loss)
# Model naming convention varies:
#   - Some: {METHOD}_loss{LOSS}_140gb (e.g., IDIS_loss30_140gb)
#   - Some: {METHOD}_loss{LOSS}_seed{N}_140gb (e.g., IDIS_loss90_seed1_140gb)
# This function tries both patterns and returns the one that exists
get_model_dir_name() {
    local method="$1"
    local loss="$2"
    local seed="$3"
    local models_dir="$PROJECT_ROOT/outputs"

    if [[ "$method" == "baseline" ]]; then
        echo "baseline_loss0_seed1_140gb"
        return
    fi

    # For non-deterministic methods with seed
    if [[ -n "$seed" ]]; then
        echo "${method}_loss${loss}_seed${seed}_140gb"
        return
    fi

    # For deterministic methods, check both naming patterns
    # Pattern 1: {METHOD}_loss{LOSS}_140gb (preferred for deterministic)
    local pattern1="${method}_loss${loss}_140gb"
    # Pattern 2: {METHOD}_loss{LOSS}_seed1_140gb (some older experiments)
    local pattern2="${method}_loss${loss}_seed1_140gb"

    if [ -d "$models_dir/$pattern1" ]; then
        echo "$pattern1"
    elif [ -d "$models_dir/$pattern2" ]; then
        echo "$pattern2"
    else
        # Default to pattern1 (will fail with helpful error if not found)
        echo "$pattern1"
    fi
}

# Set model directory name
if [[ "$RUN_ALL" == false ]]; then
    MODEL_DIR_NAME=$(get_model_dir_name "$METHOD" "$LOSS" "$SEED")
    MODEL_DIR="$PROJECT_ROOT/outputs/${MODEL_DIR_NAME}"
    MODEL_PATH="$MODEL_DIR/model/model_best.pth"

    # Experiment name for output
    if [[ -n "$SEED" ]]; then
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed${SEED}"
    else
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}"
    fi

    # Output directories (in inference_on_original subfolder)
    OUTPUT_DIR="$PROJECT_ROOT/outputs/inference_on_original/${EXPERIMENT_NAME}"
    OUTPUT_DIR_RELATIVE="../SemanticKITTI/outputs/inference_on_original/${EXPERIMENT_NAME}"
    INFERENCE_LOG="$OUTPUT_DIR/inference.log"
    METRICS_FILE="$OUTPUT_DIR/inference_metrics.txt"
    PID_FILE="$OUTPUT_DIR/inference.pid"
fi

# Config file (use baseline config, override data_root and weight)
BASELINE_CONFIG="$PROJECT_ROOT/configs/semantickitti/generated/ptv3_semantickitti_RS_loss0_seed1_140gb.py"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 Inference - Models on ORIGINAL Data            ${CYAN}║${NC}"
    if [[ -n "$SEED" ]]; then
        echo -e "${CYAN}║${NC}  Model: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Seed: ${GREEN}${SEED}${NC}                              ${CYAN}║${NC}"
    else
        echo -e "${CYAN}║${NC}  Model: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Deterministic                      ${CYAN}║${NC}"
    fi
    echo -e "${CYAN}║${NC}  Data: ${YELLOW}ORIGINAL${NC} (full resolution)                                 ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at: $VENV_PATH"
        return 1
    fi
    print_success "Virtual environment found"

    # Check trained model
    if [ ! -f "$MODEL_PATH" ]; then
        print_error "Trained model not found: $MODEL_PATH"
        print_info "Model directory resolved to: $MODEL_DIR_NAME"
        print_info "Train model first: ./train_semantickitti_140gb.sh --method $METHOD --loss $LOSS start"
        return 1
    fi
    print_success "Trained model found: $MODEL_DIR_NAME"

    # Check baseline config
    if [ ! -f "$BASELINE_CONFIG" ]; then
        print_error "Baseline config not found: $BASELINE_CONFIG"
        print_info "Generate configs: python scripts/generate_configs.py"
        return 1
    fi
    print_success "Config found"

    # Check original data path
    if [ ! -d "$ORIGINAL_DATA_PATH" ]; then
        print_error "Original data not found: $ORIGINAL_DATA_PATH"
        return 1
    fi
    print_success "Original data path found: $ORIGINAL_DATA_PATH"

    # Check validation sequence exists
    if [ ! -d "$ORIGINAL_DATA_PATH/sequences/08" ]; then
        print_error "Validation sequence 08 not found: $ORIGINAL_DATA_PATH/sequences/08"
        return 1
    fi
    print_success "Validation sequence 08 found"

    # Check pointcept
    if [ ! -d "$POINTCEPT_DIR" ]; then
        print_error "Pointcept not found: $POINTCEPT_DIR"
        return 1
    fi
    print_success "Pointcept found"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID 2>/dev/null || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null || echo "Unknown")
        print_success "GPU $GPU_ID: $GPU_NAME (${GPU_MEM}MB free)"
    else
        print_warning "nvidia-smi not found"
    fi

    return 0
}

################################################################################
# Inference Functions
################################################################################

run_inference() {
    print_header

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_error "Inference already running (PID: $PID)"
            print_info "Use '$0 --method $METHOD --loss $LOSS stop' to stop it first"
            return 1
        else
            print_warning "Stale PID file found, removing..."
            rm "$PID_FILE"
        fi
    fi

    # Check prerequisites
    check_prerequisites || return 1
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Print configuration
    print_info "Inference Configuration:"
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │ Model (trained on subsampled data)                              │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Method:         $METHOD"
    echo "  │  Loss:           $LOSS%"
    if [[ -n "$SEED" ]]; then
        echo "  │  Seed:           $SEED"
    else
        echo "  │  Seed:           N/A (deterministic)"
    fi
    echo "  │  Model Path:     $MODEL_PATH"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Data (ORIGINAL - full resolution)                               │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Data Path:      $ORIGINAL_DATA_PATH"
    echo "  │  Val Sequence:   08"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Output                                                          │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Results:        $OUTPUT_DIR"
    echo "  │  Log:            $INFERENCE_LOG"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    echo ""

    # Start inference
    print_info "Starting inference process..."

    (
        set -o pipefail

        source "$VENV_PATH/bin/activate"
        cd "$POINTCEPT_DIR"

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export PYTHONUNBUFFERED=1
        export PYTHONPATH="$POINTCEPT_DIR:${PYTHONPATH:-}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # Record start time
        START_TIME=$(date +%s)
        START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

        echo "========================================================================"
        echo "INFERENCE START: $START_DATETIME"
        echo "========================================================================"
        echo "Experiment:   $EXPERIMENT_NAME"
        echo "Model Method: $METHOD"
        echo "Model Loss:   $LOSS%"
        echo "Model Seed:   ${SEED:-N/A (deterministic)}"
        echo "Model Path:   $MODEL_PATH"
        echo "Data:         ORIGINAL (full resolution)"
        echo "Data Path:    $ORIGINAL_DATA_PATH_RELATIVE"
        echo "GPU:          $GPU_ID"
        echo "========================================================================"

        # Get initial GPU memory
        INITIAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

        # Run test.py with the trained model on ORIGINAL data
        # Using num_worker_test=4 for faster inference (default is 1)
        python tools/test.py \
            --config-file "$BASELINE_CONFIG" \
            --options \
                weight="$MODEL_PATH" \
                data_root="$ORIGINAL_DATA_PATH_RELATIVE" \
                save_path="$OUTPUT_DIR_RELATIVE" \
                num_worker_test=4

        # Record end time
        END_TIME=$(date +%s)
        END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
        ELAPSED=$((END_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))

        # Get peak GPU memory
        PEAK_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

        echo ""
        echo "========================================================================"
        echo "INFERENCE COMPLETE: $END_DATETIME"
        echo "========================================================================"
        echo "Total Time:     ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED} seconds)"
        echo "Initial GPU:    ${INITIAL_GPU_MEM} MB"
        echo "Peak GPU:       ${PEAK_GPU_MEM} MB"
        echo "========================================================================"

        # Save metrics to file
        cat > "$METRICS_FILE" << EOF
========================================================================
INFERENCE METRICS: ${EXPERIMENT_NAME} on ORIGINAL Data
========================================================================
Model (trained on subsampled data):
  Method:           $METHOD
  Loss Level:       $LOSS%
  Seed:             ${SEED:-N/A (deterministic)}
  Model Path:       $MODEL_PATH

Data (ORIGINAL - full resolution):
  Data Path:        $ORIGINAL_DATA_PATH_RELATIVE
  Val Sequence:     08

Timing:
  Start Time:       $START_DATETIME
  End Time:         $END_DATETIME
  Total Time:       ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED} seconds)

GPU Memory:
  GPU ID:           $GPU_ID
  Initial Memory:   ${INITIAL_GPU_MEM} MB
  Peak Memory:      ${PEAK_GPU_MEM} MB

========================================================================
EOF

        echo ""
        echo "Metrics saved to: $METRICS_FILE"

    ) 2>&1 | tee "$INFERENCE_LOG" &

    INFERENCE_PID=$!
    echo "$INFERENCE_PID" > "$PID_FILE"
    disown "$INFERENCE_PID"

    sleep 3

    if ps -p "$INFERENCE_PID" > /dev/null 2>&1; then
        print_success "Inference started (PID: $INFERENCE_PID)"
        echo ""
        print_info "Commands:"
        echo "  View logs:    $0 --method $METHOD --loss $LOSS logs"
        echo "  Status:       $0 --method $METHOD --loss $LOSS status"
        echo "  Stop:         $0 --method $METHOD --loss $LOSS stop"
    else
        print_error "Failed to start inference"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_inference() {
    print_header
    print_info "Stopping inference for: $EXPERIMENT_NAME"
    echo ""

    STOPPED=false

    # Kill by PID file if exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        print_info "Found PID file with PID: $PID"

        if ps -p "$PID" > /dev/null 2>&1; then
            print_info "Killing process $PID..."
            kill -TERM $PID 2>/dev/null || true
            sleep 2
            kill -9 $PID 2>/dev/null || true
            STOPPED=true
        else
            print_warning "Process $PID not running (already stopped)"
            STOPPED=true
        fi
        rm -f "$PID_FILE"
    else
        print_warning "No PID file found"
    fi

    # Also search for any test.py processes for this experiment
    PIDS=$(pgrep -f "test\.py.*inference_on_original.*${EXPERIMENT_NAME}" 2>/dev/null | tr '\n' ' ')
    if [ -n "$(echo $PIDS | tr -d ' ')" ]; then
        print_info "Found additional processes: $PIDS"
        for pid in $PIDS; do
            kill -9 $pid 2>/dev/null || true
        done
        STOPPED=true
    fi

    if [ "$STOPPED" = true ]; then
        print_success "Inference stopped"
    else
        print_info "No inference process was running"
    fi
}

show_status() {
    print_header

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_success "Inference is RUNNING (PID: $PID)"

            # GPU status
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                print_info "GPU $GPU_ID Status:"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                    --format=csv,noheader -i $GPU_ID | \
                    awk -F', ' '{printf "  Utilization: %s | Memory: %s / %s | Temp: %s\n", $1, $2, $3, $4}'
            fi

            # Show recent log
            if [ -f "$INFERENCE_LOG" ]; then
                echo ""
                print_info "Recent logs:"
                tail -5 "$INFERENCE_LOG" | sed 's/^/  /'
            fi
        else
            print_warning "Inference NOT RUNNING (stale PID: $PID)"
            rm "$PID_FILE"
        fi
    else
        print_info "Inference NOT RUNNING"
    fi

    # Show results if available
    if [ -f "$METRICS_FILE" ]; then
        echo ""
        print_success "Metrics file available: $METRICS_FILE"
        grep -E "(Total Time|Peak Memory)" "$METRICS_FILE" | sed 's/^/  /'
    fi

    # Check for test results
    if [ -d "$OUTPUT_DIR/result" ]; then
        echo ""
        print_success "Results available in: $OUTPUT_DIR/result/"
        if [ -f "$OUTPUT_DIR/result/test.log" ]; then
            echo ""
            print_info "Test Results:"
            grep -i "miou\|all_acc\|allAcc" "$OUTPUT_DIR/result/test.log" 2>/dev/null | tail -5 | sed 's/^/  /' || \
                tail -10 "$OUTPUT_DIR/result/test.log" | sed 's/^/  /'
        fi
    fi

    # Output info
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_info "Output Directory: $OUTPUT_DIR"
        DISK=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
        echo "  Disk usage: $DISK"
    fi
}

show_logs() {
    print_header

    if [ ! -f "$INFERENCE_LOG" ]; then
        print_warning "No log file found at: $INFERENCE_LOG"
        return 1
    fi

    print_info "Showing logs: $INFERENCE_LOG"
    print_info "Press Ctrl+C to exit"
    echo ""

    tail -f "$INFERENCE_LOG"
}

################################################################################
# Run All Functions
################################################################################

# Trained models directory
MODELS_DIR="$PROJECT_ROOT/outputs"

discover_trained_models() {
    # Discover all trained models (directories with model_best.pth)
    # Returns list of model directory names (e.g., "IDIS_loss90_140gb", "RS_loss90_seed1_140gb")
    if [ ! -d "$MODELS_DIR" ]; then
        print_error "Models directory not found: $MODELS_DIR"
        return 1
    fi

    local models=()
    for dir in "$MODELS_DIR"/*/; do
        local name=$(basename "$dir")
        # Skip non-model directories
        if [[ "$name" == "inference" || "$name" == "inference_on_original" || "$name" == "." || "$name" == ".." ]]; then
            continue
        fi
        # Check if it has model_best.pth
        if [ -f "$dir/model/model_best.pth" ]; then
            models+=("$name")
        fi
    done

    # Sort and print
    printf '%s\n' "${models[@]}" | sort
}

parse_model_name() {
    # Parse model directory name into method, loss, seed
    # Input: "IDIS_loss90_140gb" or "RS_loss90_seed1_140gb" or "IDIS_R5_loss90_140gb"
    # Output: Sets METHOD, LOSS, SEED variables
    local name="$1"

    # Remove _140gb suffix
    name="${name%_140gb}"

    # Extract seed if present
    if [[ "$name" =~ _seed([0-9]+)$ ]]; then
        SEED="${BASH_REMATCH[1]}"
        name="${name%_seed*}"
    else
        SEED=""
    fi

    # Extract loss level
    if [[ "$name" =~ _loss([0-9]+)$ ]]; then
        LOSS="${BASH_REMATCH[1]}"
        name="${name%_loss*}"
    else
        return 1
    fi

    # Remaining is the method
    METHOD="$name"
}

get_experiment_name_from_model() {
    # Convert model dir name to experiment name
    # Input: "IDIS_loss90_140gb" -> "IDIS_loss90"
    # Input: "RS_loss90_seed1_140gb" -> "RS_loss90_seed1"
    local model_name="$1"
    echo "${model_name%_140gb}"
}

run_all_inference() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 - ALL Models on ORIGINAL Data                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Validation Set: Sequence 08 (ORIGINAL - full resolution)          ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Check original data first
    if [ ! -d "$ORIGINAL_DATA_PATH/sequences/08" ]; then
        print_error "Original validation data not found: $ORIGINAL_DATA_PATH/sequences/08"
        return 1
    fi
    print_success "Original data: $ORIGINAL_DATA_PATH"
    echo ""

    # Discover all trained models
    print_info "Discovering trained models in: $MODELS_DIR"
    local models=()
    while IFS= read -r line; do
        [ -n "$line" ] && models+=("$line")
    done < <(discover_trained_models)

    if [ ${#models[@]} -eq 0 ]; then
        print_error "No trained models found!"
        return 1
    fi

    echo ""
    print_info "Found ${#models[@]} trained models:"
    for m in "${models[@]}"; do
        [ -z "$m" ] && continue
        echo "  - $m"
    done
    echo ""

    # Check which already have inference results
    local pending=()
    local completed=()
    local failed=()

    for m in "${models[@]}"; do
        [ -z "$m" ] && continue
        local exp_name=$(get_experiment_name_from_model "$m")
        local output_dir="$PROJECT_ROOT/outputs/inference_on_original/$exp_name"
        if [ -f "$output_dir/inference_metrics.txt" ]; then
            # Check if it completed successfully
            if grep -q "INFERENCE COMPLETE" "$output_dir/inference.log" 2>/dev/null; then
                completed+=("$m")
            else
                failed+=("$m")
            fi
        else
            pending+=("$m")
        fi
    done

    print_info "Status Summary:"
    echo "  Completed: ${#completed[@]}"
    echo "  Pending:   ${#pending[@]}"
    echo "  Failed:    ${#failed[@]}"
    echo ""

    if [ ${#pending[@]} -eq 0 ]; then
        print_success "All models have been evaluated on original data!"
        if [ ${#failed[@]} -gt 0 ]; then
            print_warning "Failed evaluations that may need retry:"
            for m in "${failed[@]}"; do
                echo "  - $m"
            done
        fi
        return 0
    fi

    print_info "Running inference on ${#pending[@]} pending models (sequential)..."
    echo ""

    local count=0
    local total=${#pending[@]}
    local success_count=0
    local fail_count=0

    for m in "${pending[@]}"; do
        ((count++))
        echo ""
        echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${MAGENTA}  [$count/$total] Processing: $m${NC}"
        echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"

        # Parse model name
        if ! parse_model_name "$m"; then
            print_error "Could not parse model name: $m"
            ((fail_count++))
            continue
        fi

        local exp_name=$(get_experiment_name_from_model "$m")
        local model_path="$MODELS_DIR/$m/model/model_best.pth"
        local output_dir="$PROJECT_ROOT/outputs/inference_on_original/$exp_name"
        local output_dir_relative="../SemanticKITTI/outputs/inference_on_original/$exp_name"
        local inference_log="$output_dir/inference.log"
        local metrics_file="$output_dir/inference_metrics.txt"

        print_info "Model: $m"
        print_info "Method: $METHOD | Loss: $LOSS% | Seed: ${SEED:-N/A}"
        print_info "Model Path: $model_path"
        print_info "Data: ORIGINAL (full resolution)"

        # Check model exists
        if [ ! -f "$model_path" ]; then
            print_error "Model not found: $model_path"
            ((fail_count++))
            continue
        fi

        # Create output directory
        mkdir -p "$output_dir"

        # Run inference (synchronously for --all mode)
        print_info "Starting inference..."

        (
            source "$VENV_PATH/bin/activate"
            cd "$POINTCEPT_DIR"

            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export PYTHONUNBUFFERED=1
            export PYTHONPATH="$POINTCEPT_DIR:${PYTHONPATH:-}"
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            # Record start time
            START_TIME=$(date +%s)
            START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

            echo "========================================================================"
            echo "INFERENCE START: $START_DATETIME"
            echo "========================================================================"
            echo "Model:        $m"
            echo "Method:       $METHOD"
            echo "Loss:         $LOSS%"
            echo "Seed:         ${SEED:-N/A}"
            echo "Model Path:   $model_path"
            echo "Data:         ORIGINAL (full resolution)"
            echo "Data Path:    $ORIGINAL_DATA_PATH_RELATIVE"
            echo "GPU:          $GPU_ID"
            echo "========================================================================"

            # Get initial GPU memory
            INITIAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

            # Run test.py with trained model on ORIGINAL data
            # Using num_worker_test=4 for faster inference (default is 1)
            python tools/test.py \
                --config-file "$BASELINE_CONFIG" \
                --options \
                    weight="$model_path" \
                    data_root="$ORIGINAL_DATA_PATH_RELATIVE" \
                    save_path="$output_dir_relative" \
                    num_worker_test=4

            # Record end time
            END_TIME=$(date +%s)
            END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
            ELAPSED=$((END_TIME - START_TIME))
            ELAPSED_MIN=$((ELAPSED / 60))
            ELAPSED_SEC=$((ELAPSED % 60))

            # Get peak GPU memory
            PEAK_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

            echo ""
            echo "========================================================================"
            echo "INFERENCE COMPLETE: $END_DATETIME"
            echo "========================================================================"
            echo "Total Time:     ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED} seconds)"
            echo "Initial GPU:    ${INITIAL_GPU_MEM} MB"
            echo "Peak GPU:       ${PEAK_GPU_MEM} MB"
            echo "========================================================================"

            # Save metrics
            cat > "$metrics_file" << EOF
========================================================================
INFERENCE METRICS: ${exp_name} on ORIGINAL Data
========================================================================
Model (trained on subsampled data):
  Model Dir:        $m
  Method:           $METHOD
  Loss Level:       $LOSS%
  Seed:             ${SEED:-N/A (deterministic)}
  Model Path:       $model_path

Data (ORIGINAL - full resolution):
  Data Path:        $ORIGINAL_DATA_PATH_RELATIVE
  Val Sequence:     08

Timing:
  Start Time:       $START_DATETIME
  End Time:         $END_DATETIME
  Total Time:       ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED} seconds)

GPU Memory:
  GPU ID:           $GPU_ID
  Initial Memory:   ${INITIAL_GPU_MEM} MB
  Peak Memory:      ${PEAK_GPU_MEM} MB

========================================================================
EOF

        ) 2>&1 | tee "$inference_log"

        # Check if inference succeeded
        if grep -q "INFERENCE COMPLETE" "$inference_log" 2>/dev/null; then
            print_success "Completed: $m"
            ((success_count++))
        else
            print_error "Failed: $m"
            ((fail_count++))
        fi
    done

    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ALL MODELS PROCESSING COMPLETE${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    print_info "Results:"
    echo "  Successful: $success_count"
    echo "  Failed:     $fail_count"
    echo "  Previously completed: ${#completed[@]}"
    echo ""
    print_info "Output directory: $PROJECT_ROOT/outputs/inference_on_original/"
}

show_all_status() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 - Status: Models on ORIGINAL Data              ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Discover all trained models
    local models=()
    while IFS= read -r line; do
        [ -n "$line" ] && models+=("$line")
    done < <(discover_trained_models)

    if [ ${#models[@]} -eq 0 ]; then
        print_error "No trained models found in: $MODELS_DIR"
        return 1
    fi

    print_info "Found ${#models[@]} trained models"
    echo ""

    # Table header
    printf "%-35s | %-12s | %-10s | %-10s\n" "Model" "Status" "Time" "mIoU"
    printf "%-35s-+-%-12s-+-%-10s-+-%-10s\n" "-----------------------------------" "------------" "----------" "----------"

    local completed=0
    local pending=0
    local running=0
    local failed=0

    for m in "${models[@]}"; do
        [ -z "$m" ] && continue

        local exp_name=$(get_experiment_name_from_model "$m")
        local output_dir="$PROJECT_ROOT/outputs/inference_on_original/$exp_name"
        local status="PENDING"
        local time_val="-"
        local miou="-"

        if [ -f "$output_dir/inference.pid" ]; then
            local pid
            pid=$(cat "$output_dir/inference.pid")
            if ps -p "$pid" > /dev/null 2>&1; then
                status="${YELLOW}RUNNING${NC}"
                running=$((running + 1))
            else
                # PID file exists but process not running
                if grep -q "INFERENCE COMPLETE" "$output_dir/inference.log" 2>/dev/null; then
                    status="${GREEN}DONE${NC}"
                    completed=$((completed + 1))
                else
                    status="${RED}FAILED${NC}"
                    failed=$((failed + 1))
                fi
            fi
        elif [ -f "$output_dir/inference_metrics.txt" ]; then
            if grep -q "INFERENCE COMPLETE" "$output_dir/inference.log" 2>/dev/null; then
                status="${GREEN}DONE${NC}"
                completed=$((completed + 1))
                # Extract time
                time_val=$(grep "Total Time:" "$output_dir/inference_metrics.txt" 2>/dev/null | head -1 | sed 's/.*Total Time:[ ]*//' | cut -d'(' -f1 | xargs) || true
                # Try to extract mIoU from test.log
                if [ -f "$output_dir/test.log" ]; then
                    miou=$(grep -i "miou" "$output_dir/test.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1) || true
                fi
            else
                status="${RED}FAILED${NC}"
                failed=$((failed + 1))
            fi
        else
            status="${BLUE}PENDING${NC}"
            pending=$((pending + 1))
        fi

        printf "%-35s | %-12b | %-10s | %-10s\n" "$m" "$status" "$time_val" "${miou:-N/A}"
    done

    echo ""
    printf "%-35s-+-%-12s-+-%-10s-+-%-10s\n" "-----------------------------------" "------------" "----------" "----------"
    echo ""
    print_info "Summary:"
    echo "  Completed: $completed"
    echo "  Running:   $running"
    echo "  Pending:   $pending"
    echo "  Failed:    $failed"
    echo "  Total:     ${#models[@]}"
}

################################################################################
# Main
################################################################################

# Handle --all mode
if [[ "$RUN_ALL" == true ]]; then
    case "$COMMAND" in
        run)
            run_all_inference
            ;;
        status)
            show_all_status
            ;;
        stop)
            print_warning "Stop command not supported for --all mode"
            print_info "Use --method and --loss to stop specific inference"
            ;;
        logs)
            print_warning "Logs command not supported for --all mode"
            print_info "Use --method and --loss to view specific logs"
            ;;
        *)
            print_error "Invalid command for --all mode: $COMMAND"
            print_info "Supported commands: run, status"
            exit 1
            ;;
    esac
    exit 0
fi

case "$COMMAND" in
    run)
        run_inference
        ;;
    stop)
        stop_inference
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    help)
        print_usage
        ;;
    *)
        print_error "Invalid command: $COMMAND"
        print_usage
        exit 1
        ;;
esac
