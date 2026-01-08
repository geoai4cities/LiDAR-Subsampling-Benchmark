#!/bin/bash

################################################################################
# Inference Script for SemanticKITTI Subsampling Benchmark - 140GB GPU
#
# This script runs inference using the trained BASELINE model on validation
# data from different subsampling methods. This evaluates how well the baseline
# model (trained on original data) generalizes to subsampled point clouds.
#
# Usage:
#   ./inference_semantickitti_140gb.sh --method METHOD --loss LOSS [OPTIONS] COMMAND
#   ./inference_semantickitti_140gb.sh --all [OPTIONS] COMMAND
#
# Examples:
#   # Run ALL available subsampled datasets (sequentially)
#   ./inference_semantickitti_140gb.sh --all run
#   ./inference_semantickitti_140gb.sh --all status
#
#   # Deterministic methods (no seed needed)
#   ./inference_semantickitti_140gb.sh --method IDIS --loss 90 run
#   ./inference_semantickitti_140gb.sh --method DBSCAN --loss 50 run
#   ./inference_semantickitti_140gb.sh --method Voxel --loss 90 run
#   ./inference_semantickitti_140gb.sh --method DEPOCO --loss 30 run
#
#   # Non-deterministic methods (seed required)
#   ./inference_semantickitti_140gb.sh --method RS --loss 90 --seed 1 run
#   ./inference_semantickitti_140gb.sh --method FPS --loss 90 --seed 1 run
#   ./inference_semantickitti_140gb.sh --method Poisson --loss 90 --seed 1 run
#
#   # IDIS R-value ablation
#   ./inference_semantickitti_140gb.sh --method IDIS_R5 --loss 90 run
#   ./inference_semantickitti_140gb.sh --method IDIS_R15 --loss 90 run
#   ./inference_semantickitti_140gb.sh --method IDIS_R20 --loss 90 run
#
#   # Check status
#   ./inference_semantickitti_140gb.sh --method IDIS --loss 90 status
#
# Validation Set: Sequence 08 only (SemanticKITTI standard split)
#
# Methods:
#   Deterministic:     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel, DEPOCO
#   Non-deterministic: RS, FPS, Poisson
#
# Loss Levels: 10, 30, 50, 70, 90 (use validation data from subsampled datasets)
#
# Output:
#   Results saved to: outputs/inference/{METHOD}_loss{LOSS}[_seed{N}]/
#   - inference.log: Full inference log
#   - inference_metrics.txt: Timing and memory stats
#   - result/: mIoU and per-class results
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

# Baseline model path (the trained model to use for inference)
# This is the model trained on ORIGINAL data (0% loss)
BASELINE_MODEL_NAME="baseline_loss0_seed1_140gb"

################################################################################
# Parse Arguments
################################################################################

print_usage() {
    echo "Usage: $0 --method METHOD --loss LOSS [--seed SEED] [--gpu GPU_ID] COMMAND"
    echo "       $0 --all [--gpu GPU_ID] COMMAND"
    echo ""
    echo "Required Arguments (choose one):"
    echo "  --method METHOD   Subsampling method for validation data"
    echo "  --loss LOSS       Loss percentage (10, 30, 50, 70, 90)"
    echo "  --all             Run on ALL available subsampled datasets (sequential)"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed SEED       Random seed for non-deterministic methods (1, 2, or 3)"
    echo "  --gpu GPU_ID      GPU device ID (default: 0)"
    echo "  --baseline NAME   Baseline model name (default: $BASELINE_MODEL_NAME)"
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
    echo ""
    echo "Examples:"
    echo "  # Run ALL available datasets"
    echo "  $0 --all run"
    echo "  $0 --all status"
    echo ""
    echo "  # Deterministic methods"
    echo "  $0 --method IDIS --loss 90 run"
    echo "  $0 --method DBSCAN --loss 50 run"
    echo "  $0 --method Voxel --loss 90 run"
    echo "  $0 --method DEPOCO --loss 30 run"
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
    echo "  - Data directory: data/SemanticKITTI/subsampled/"
    echo ""
    echo "Output:"
    echo "  Results: outputs/inference/{METHOD}_loss{LOSS}[_seed{N}]/"
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
        --baseline)
            BASELINE_MODEL_NAME="$2"
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
VALID_METHODS=("RS" "IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "FPS" "DBSCAN" "Voxel" "Poisson" "DEPOCO")
VALID_LOSS=("10" "30" "50" "70" "90")

# Deterministic methods don't need seed
DETERMINISTIC_METHODS=("IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "DBSCAN" "Voxel" "DEPOCO")
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

    # DEPOCO-specific note: currently has 10, 30, 70 (50, 90 to be generated)
    # The data path check will catch if data doesn't exist yet

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

# Baseline model path
BASELINE_OUTPUT_DIR="$PROJECT_ROOT/outputs/${BASELINE_MODEL_NAME}"
BASELINE_MODEL="$BASELINE_OUTPUT_DIR/model/model_best.pth"

# Baseline config (we'll override data_root for inference)
# Use a config that exists - RS_loss0_seed1 is the baseline config
BASELINE_CONFIG="$PROJECT_ROOT/configs/semantickitti/generated/ptv3_semantickitti_RS_loss0_seed1_140gb.py"

# Data path for subsampled validation data
if [[ -z "$SEED" ]]; then
    DATA_PATH="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}"
    EXPERIMENT_NAME="${METHOD}_loss${LOSS}"
else
    DATA_PATH="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed${SEED}"
fi

# Output directories
OUTPUT_DIR="$PROJECT_ROOT/outputs/inference/${EXPERIMENT_NAME}"
OUTPUT_DIR_RELATIVE="../SemanticKITTI/outputs/inference/${EXPERIMENT_NAME}"
INFERENCE_LOG="$OUTPUT_DIR/inference.log"
METRICS_FILE="$OUTPUT_DIR/inference_metrics.txt"
PID_FILE="$OUTPUT_DIR/inference.pid"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 Inference - Subsampling Benchmark              ${CYAN}║${NC}"
    if [[ -n "$SEED" ]]; then
        echo -e "${CYAN}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Seed: ${GREEN}${SEED}${NC}                              ${CYAN}║${NC}"
    else
        echo -e "${CYAN}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Deterministic                      ${CYAN}║${NC}"
    fi
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

    # Check baseline model
    if [ ! -f "$BASELINE_MODEL" ]; then
        print_error "Baseline model not found: $BASELINE_MODEL"
        print_info "Train baseline first: ./train_semantickitti_140gb.sh --method RS --loss 0 start"
        return 1
    fi
    print_success "Baseline model found: $BASELINE_MODEL"

    # Check baseline config
    if [ ! -f "$BASELINE_CONFIG" ]; then
        print_error "Baseline config not found: $BASELINE_CONFIG"
        print_info "Generate configs: python scripts/generate_configs.py"
        return 1
    fi
    print_success "Baseline config found"

    # Check data path
    if [ ! -d "$DATA_PATH" ]; then
        print_error "Subsampled data not found: $DATA_PATH"
        print_info "Run subsampling first for ${METHOD} at ${LOSS}% loss"
        return 1
    fi
    print_success "Data path found: $DATA_PATH"

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
    echo "  │ Experiment                                                      │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Method:         $METHOD"
    echo "  │  Loss:           $LOSS%"
    if [[ -n "$SEED" ]]; then
        echo "  │  Seed:           $SEED"
    else
        echo "  │  Seed:           N/A (deterministic)"
    fi
    echo "  │  Data Path:      $DATA_PATH"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Model                                                           │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Baseline Model: $BASELINE_MODEL"
    echo "  │  Config:         $BASELINE_CONFIG"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Output                                                          │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Results:        $OUTPUT_DIR"
    echo "  │  Log:            $INFERENCE_LOG"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    echo ""

    # Start inference
    print_info "Starting inference process..."

    # Get data path relative to pointcept directory
    DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}"
    if [[ -n "$SEED" ]]; then
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    fi

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
        echo "Method:     $METHOD"
        echo "Loss:       $LOSS%"
        echo "Data:       $DATA_PATH_RELATIVE"
        echo "Model:      $BASELINE_MODEL"
        echo "GPU:        $GPU_ID"
        echo "========================================================================"

        # Get initial GPU memory
        INITIAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

        # Run test.py with the baseline model on subsampled data
        # Override data_root to use subsampled validation data
        # Using num_worker_test=4 for faster inference (default is 1)
        python tools/test.py \
            --config-file "$BASELINE_CONFIG" \
            --options \
                weight="$BASELINE_MODEL" \
                data_root="$DATA_PATH_RELATIVE" \
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
INFERENCE METRICS: ${EXPERIMENT_NAME}
========================================================================
Experiment:
  Method:           $METHOD
  Loss Level:       $LOSS%
  Seed:             ${SEED:-N/A (deterministic)}
  Data Path:        $DATA_PATH_RELATIVE

Model:
  Baseline Model:   $BASELINE_MODEL
  Config:           $BASELINE_CONFIG

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
    PIDS=$(pgrep -f "test\.py.*${EXPERIMENT_NAME}" 2>/dev/null | tr '\n' ' ')
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
            # Try to find mIoU from log
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

# Subsampled data directory
SUBSAMPLED_DIR="$DATA_ROOT/subsampled"

discover_datasets() {
    # Discover all available subsampled datasets
    # Returns list of directory names (e.g., "IDIS_loss90", "RS_loss90_seed1")
    if [ ! -d "$SUBSAMPLED_DIR" ]; then
        print_error "Subsampled data directory not found: $SUBSAMPLED_DIR"
        return 1
    fi

    # Find all valid dataset directories (exclude 'reports' and other non-dataset dirs)
    local datasets=()
    for dir in "$SUBSAMPLED_DIR"/*/; do
        local name=$(basename "$dir")
        # Skip non-dataset directories
        if [[ "$name" == "reports" || "$name" == "." || "$name" == ".." ]]; then
            continue
        fi
        # Check if it has sequences directory (valid dataset)
        if [ -d "$dir/sequences" ]; then
            datasets+=("$name")
        fi
    done

    # Sort and print
    printf '%s\n' "${datasets[@]}" | sort
}

parse_dataset_name() {
    # Parse dataset name into method, loss, seed
    # Input: "IDIS_loss90" or "RS_loss90_seed1" or "IDIS_R5_loss90"
    # Output: Sets METHOD, LOSS, SEED variables
    local name="$1"

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

run_all_inference() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 Inference - Run ALL Datasets                   ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Validation Set: Sequence 08 only                                  ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Check baseline model first
    if [ ! -f "$BASELINE_MODEL" ]; then
        print_error "Baseline model not found: $BASELINE_MODEL"
        print_info "Train baseline first before running inference"
        return 1
    fi
    print_success "Baseline model: $BASELINE_MODEL"
    echo ""

    # Discover all datasets
    print_info "Discovering available datasets in: $SUBSAMPLED_DIR"
    local datasets=()
    while IFS= read -r line; do
        [ -n "$line" ] && datasets+=("$line")
    done < <(discover_datasets)

    if [ ${#datasets[@]} -eq 0 ]; then
        print_error "No datasets found!"
        return 1
    fi

    echo ""
    print_info "Found ${#datasets[@]} datasets:"
    for ds in "${datasets[@]}"; do
        [ -z "$ds" ] && continue
        echo "  - $ds"
    done
    echo ""

    # Check which already have inference results
    local pending=()
    local completed=()
    local failed=()

    for ds in "${datasets[@]}"; do
        [ -z "$ds" ] && continue
        local output_dir="$PROJECT_ROOT/outputs/inference/$ds"
        if [ -f "$output_dir/inference_metrics.txt" ]; then
            # Check if it completed successfully
            if grep -q "INFERENCE COMPLETE" "$output_dir/inference.log" 2>/dev/null; then
                completed+=("$ds")
            else
                failed+=("$ds")
            fi
        else
            pending+=("$ds")
        fi
    done

    print_info "Status Summary:"
    echo "  Completed: ${#completed[@]}"
    echo "  Pending:   ${#pending[@]}"
    echo "  Failed:    ${#failed[@]}"
    echo ""

    if [ ${#pending[@]} -eq 0 ]; then
        print_success "All datasets have been processed!"
        if [ ${#failed[@]} -gt 0 ]; then
            print_warning "Failed datasets that may need retry:"
            for ds in "${failed[@]}"; do
                echo "  - $ds"
            done
        fi
        return 0
    fi

    print_info "Running inference on ${#pending[@]} pending datasets (sequential)..."
    echo ""

    local count=0
    local total=${#pending[@]}
    local success_count=0
    local fail_count=0

    for ds in "${pending[@]}"; do
        ((count++))
        echo ""
        echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${MAGENTA}  [$count/$total] Processing: $ds${NC}"
        echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"

        # Parse dataset name
        if ! parse_dataset_name "$ds"; then
            print_error "Could not parse dataset name: $ds"
            ((fail_count++))
            continue
        fi

        # Set paths for this dataset
        DATA_PATH="$SUBSAMPLED_DIR/$ds"
        EXPERIMENT_NAME="$ds"
        OUTPUT_DIR="$PROJECT_ROOT/outputs/inference/${EXPERIMENT_NAME}"
        OUTPUT_DIR_RELATIVE="../SemanticKITTI/outputs/inference/${EXPERIMENT_NAME}"
        INFERENCE_LOG="$OUTPUT_DIR/inference.log"
        METRICS_FILE="$OUTPUT_DIR/inference_metrics.txt"

        print_info "Method: $METHOD | Loss: $LOSS% | Seed: ${SEED:-N/A}"
        print_info "Data: $DATA_PATH"

        # Check data exists
        if [ ! -d "$DATA_PATH/sequences" ]; then
            print_error "Data not found: $DATA_PATH/sequences"
            ((fail_count++))
            continue
        fi

        # Create output directory
        mkdir -p "$OUTPUT_DIR"

        # Run inference (synchronously for --all mode)
        print_info "Starting inference..."

        # Get data path relative to pointcept directory
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/$ds"

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
            echo "Dataset:    $ds"
            echo "Method:     $METHOD"
            echo "Loss:       $LOSS%"
            echo "Seed:       ${SEED:-N/A}"
            echo "Data:       $DATA_PATH_RELATIVE"
            echo "Model:      $BASELINE_MODEL"
            echo "GPU:        $GPU_ID"
            echo "========================================================================"

            # Get initial GPU memory
            INITIAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES 2>/dev/null || echo "0")

            # Run test.py
            # Using num_worker_test=4 for faster inference (default is 1)
            python tools/test.py \
                --config-file "$BASELINE_CONFIG" \
                --options \
                    weight="$BASELINE_MODEL" \
                    data_root="$DATA_PATH_RELATIVE" \
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

            # Save metrics
            cat > "$METRICS_FILE" << EOF
========================================================================
INFERENCE METRICS: ${EXPERIMENT_NAME}
========================================================================
Experiment:
  Dataset:          $ds
  Method:           $METHOD
  Loss Level:       $LOSS%
  Seed:             ${SEED:-N/A (deterministic)}
  Data Path:        $DATA_PATH_RELATIVE

Model:
  Baseline Model:   $BASELINE_MODEL
  Config:           $BASELINE_CONFIG

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

        ) 2>&1 | tee "$INFERENCE_LOG"

        # Check if inference succeeded
        if grep -q "INFERENCE COMPLETE" "$INFERENCE_LOG" 2>/dev/null; then
            print_success "Completed: $ds"
            ((success_count++))
        else
            print_error "Failed: $ds"
            ((fail_count++))
        fi
    done

    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ALL DATASETS PROCESSING COMPLETE${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    print_info "Results:"
    echo "  Successful: $success_count"
    echo "  Failed:     $fail_count"
    echo "  Previously completed: ${#completed[@]}"
    echo ""
    print_info "Output directory: $PROJECT_ROOT/outputs/inference/"
}

show_all_status() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  SemanticKITTI PTv3 Inference - Status of ALL Datasets             ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Discover all datasets
    local datasets=()
    while IFS= read -r line; do
        [ -n "$line" ] && datasets+=("$line")
    done < <(discover_datasets)

    if [ ${#datasets[@]} -eq 0 ]; then
        print_error "No datasets found in: $SUBSAMPLED_DIR"
        return 1
    fi

    print_info "Found ${#datasets[@]} subsampled datasets"
    echo ""

    # Table header
    printf "%-30s | %-12s | %-10s | %-20s\n" "Dataset" "Status" "Time" "mIoU"
    printf "%-30s-+-%-12s-+-%-10s-+-%-20s\n" "------------------------------" "------------" "----------" "--------------------"

    local completed=0
    local pending=0
    local running=0
    local failed=0

    for ds in "${datasets[@]}"; do
        [ -z "$ds" ] && continue

        local output_dir="$PROJECT_ROOT/outputs/inference/$ds"
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

        printf "%-30s | %-12b | %-10s | %-20s\n" "$ds" "$status" "$time_val" "${miou:-N/A}"
    done

    echo ""
    printf "%-30s-+-%-12s-+-%-10s-+-%-20s\n" "------------------------------" "------------" "----------" "--------------------"
    echo ""
    print_info "Summary:"
    echo "  Completed: $completed"
    echo "  Running:   $running"
    echo "  Pending:   $pending"
    echo "  Failed:    $failed"
    echo "  Total:     ${#datasets[@]}"
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
