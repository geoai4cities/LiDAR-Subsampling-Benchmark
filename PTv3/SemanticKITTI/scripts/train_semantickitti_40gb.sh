#!/bin/bash

################################################################################
# Training Script for SemanticKITTI Subsampling Benchmark - 40GB GPU
#
# Usage:
#   # Non-deterministic methods (RS, FPS, Poisson) - seed required
#   ./train_semantickitti_40gb.sh --method RS --loss 50 --seed 1 start
#   ./train_semantickitti_40gb.sh --method FPS --loss 50 --seed 1 resume
#
#   # Deterministic methods (IDIS, DBSCAN, Voxel, DEPOCO) - no seed needed
#   ./train_semantickitti_40gb.sh --method IDIS --loss 50 start
#   ./train_semantickitti_40gb.sh --method DBSCAN --loss 50 status
#
#   # Train on original data (0% loss)
#   ./train_semantickitti_40gb.sh --method RS --loss 0 --seed 1 start
#
# Methods: RS, IDIS, IDIS_R5, IDIS_R15, FPS, DBSCAN, Voxel, Poisson, DEPOCO
#   - IDIS: Default R=10m radius (deterministic)
#   - IDIS_R5: R=5m radius, denser sampling (deterministic)
#   - IDIS_R15: R=15m radius, sparser sampling (deterministic)
#
# Method Properties:
#   DETERMINISTIC (no seed needed): IDIS, IDIS_R5, IDIS_R15, DBSCAN, Voxel, DEPOCO
#   NON-DETERMINISTIC (seed required): RS, FPS, Poisson
#
# Loss Levels: 0, 10, 30, 50, 70, 90 (0 = original data)
# Seeds: 1, 2, 3 (only for non-deterministic methods)
#
# Configuration (Official Outdoor PTv3 Settings):
#   - GPU: 40GB (A100/A6000)
#   - CUDA: 11.8+
#   - Config: Generated from ptv3_40gb_official_template.py
#   - Model: PT-v3m1 with in_channels=4 (coord + strength)
#   - Batch Size: 6 (effective: 48 with gradient accumulation=8)
#   - Batch Size Val: 1 (REQUIRED for correct mIoU - inverse mapping bug fix)
#   - Grid Size: 0.05 (official outdoor setting)
#   - Learning Rate: 0.002, Weight Decay: 0.005
#   - feat_keys: ("coord", "strength") - 4 channel input
#   - Estimated Memory: ~30-35GB
#   - Epochs: 10 (default, can be extended with --epochs or extend command)
#
# Training Time Estimates (40GB GPU):
#   - Per epoch: ~3-4 hours
#   - 10 epochs: ~35-40 hours (default)
#   - 50 epochs: ~175-200 hours (full training)
#
# Extending Training:
#   # Extend completed training to 50 epochs
#   ./train_semantickitti_40gb.sh --method RS --loss 50 --epochs 50 extend
#
#   # Or modify config and resume
#   sed -i 's/epoch = 10/epoch = 50/' configs/semantickitti/generated/ptv3_*_40gb.py
#   ./train_semantickitti_40gb.sh --method RS --loss 50 resume
################################################################################

set -euo pipefail

################################################################################
# USER CONFIGURATION
################################################################################

# Default values
METHOD=""
LOSS=""
SEED="1"     # Default seed for non-deterministic methods (RS, FPS, Poisson)
GPU_ID="0"
EPOCHS=""  # Override epochs if specified
COMMAND=""  # Command to execute

# Method classification
# DETERMINISTIC: Same input always produces same output (no seed needed)
# NON-DETERMINISTIC: Requires seed for reproducibility
DETERMINISTIC_METHODS=("IDIS" "IDIS_R5" "IDIS_R15" "DBSCAN" "Voxel" "DEPOCO")
NON_DETERMINISTIC_METHODS=("RS" "FPS" "Poisson")

# WandB Configuration (Set to true to enable logging)
ENABLE_WANDB=false
WANDB_PROJECT="lidar-subsampling-benchmark"
WANDB_ENTITY=""

################################################################################
# Parse Arguments
################################################################################

print_usage() {
    echo "Usage: $0 --method METHOD --loss LOSS [--seed SEED] [--gpu GPU_ID] [--epochs N] {start|resume|extend|stop|status|logs|help}"
    echo ""
    echo "Required Arguments:"
    echo "  --method METHOD   Subsampling method (RS, IDIS, IDIS_R5, IDIS_R15, FPS, DBSCAN, Voxel, Poisson, DEPOCO)"
    echo "  --loss LOSS       Loss percentage (0, 10, 30, 50, 70, 90)"
    echo "                    NOTE: loss=0 trains on ORIGINAL data (no subsampling)"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed SEED       Random seed (1, 2, or 3)"
    echo "                    Required for non-deterministic methods: RS, FPS, Poisson"
    echo "                    Ignored for deterministic methods: IDIS, DBSCAN, Voxel, DEPOCO"
    echo "  --gpu GPU_ID      GPU device ID (default: 0)"
    echo "  --epochs N        Override number of epochs (default: 10)"
    echo ""
    echo "Method Properties:"
    echo "  DETERMINISTIC (no seed needed): IDIS, IDIS_R5, IDIS_R15, DBSCAN, Voxel, DEPOCO"
    echo "  NON-DETERMINISTIC (seed required): RS, FPS, Poisson"
    echo ""
    echo "Commands:"
    echo "  start             Start fresh training"
    echo "  resume            Resume from latest checkpoint (same epochs)"
    echo "  extend            Extend training with more epochs (use --epochs N)"
    echo "  stop              Stop training gracefully"
    echo "  status            Show training status"
    echo "  logs              View live training logs"
    echo "  report            Generate benchmark report"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Train on original data (0% loss)"
    echo "  $0 --method RS --loss 0 --seed 1 start"
    echo ""
    echo "  # Train on RS subsampled data with 50% loss (non-deterministic, needs seed)"
    echo "  $0 --method RS --loss 50 --seed 1 start"
    echo ""
    echo "  # Train on IDIS subsampled data (deterministic, no seed needed)"
    echo "  $0 --method IDIS --loss 50 start"
    echo ""
    echo "  # Resume training from checkpoint"
    echo "  $0 --method RS --loss 50 --seed 1 resume"
    echo ""
    echo "  # Extend training to 50 epochs total (from default 10)"
    echo "  $0 --method RS --loss 50 --seed 1 --epochs 50 extend"
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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        start|resume|extend|stop|status|logs|report|help)
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
VALID_METHODS=("RS" "IDIS" "IDIS_R5" "IDIS_R15" "FPS" "DBSCAN" "Voxel" "Poisson" "DEPOCO")
VALID_LOSS=("0" "10" "30" "50" "70" "90")

# Handle help command (doesn't need method/loss)
if [[ "$COMMAND" == "help" ]]; then
    print_usage
    exit 0
fi

# Show help if no arguments at all
if [[ -z "$COMMAND" && -z "$METHOD" && -z "$LOSS" ]]; then
    print_usage
    exit 0
fi

# Validate command is provided
if [[ -z "$COMMAND" ]]; then
    echo "Error: Command required (start|resume|extend|stop|status|logs|report|help)"
    print_usage
    exit 1
fi

# Validate required arguments for commands that need them
if [[ -z "$METHOD" || -z "$LOSS" ]]; then
    echo "Error: --method and --loss are required for '$COMMAND' command"
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
# Skip seed validation for loss=0 (original data - all methods use same baseline)
if [[ "$LOSS" != "0" ]]; then
    if [[ " ${NON_DETERMINISTIC_METHODS[@]} " =~ " ${METHOD} " ]]; then
        # Non-deterministic method - validate seed is 1, 2, or 3
        if [[ ! "$SEED" =~ ^[123]$ ]]; then
            echo "Error: Invalid seed '$SEED'. Must be 1, 2, or 3"
            exit 1
        fi
    elif [[ " ${DETERMINISTIC_METHODS[@]} " =~ " ${METHOD} " ]]; then
        # Deterministic method - clear seed (not needed)
        SEED=""
    fi
fi

# Extend command requires --epochs
if [[ "$COMMAND" == "extend" && -z "$EPOCHS" ]]; then
    echo "Error: 'extend' command requires --epochs N"
    echo "Example: $0 --method $METHOD --loss $LOSS --epochs 50 extend"
    exit 1
fi

################################################################################
# Configuration
################################################################################

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PTv3_ROOT="$(dirname "$PROJECT_ROOT")"
POINTCEPT_DIR="$PTv3_ROOT/pointcept"
VENV_PATH="$PTv3_ROOT/../ptv3_venv"
DATA_ROOT="$PTv3_ROOT/../data/SemanticKITTI"

# Experiment naming and data path - handle deterministic vs non-deterministic
# Also handles backward compatibility with old naming (_seed1 for all methods)
if [[ "$LOSS" == "0" ]]; then
    # Original data (no subsampling)
    DATA_PATH="$DATA_ROOT/original"
    if [[ -n "$SEED" ]]; then
        EXPERIMENT_NAME="baseline_loss0_seed${SEED}_40gb"
        CONFIG_SUFFIX="baseline_loss0_seed${SEED}_40gb"
    else
        EXPERIMENT_NAME="baseline_loss0_40gb"
        CONFIG_SUFFIX="baseline_loss0_40gb"
    fi
elif [[ -n "$SEED" ]]; then
    # Non-deterministic method with seed
    DATA_PATH="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed${SEED}_40gb"
    CONFIG_SUFFIX="${METHOD}_loss${LOSS}_seed${SEED}_40gb"
elif [[ -z "$SEED" ]]; then
    # Deterministic method - check new naming first, fallback to old
    DATA_PATH_NEW="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}"
    DATA_PATH_OLD="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}_seed1"

    if [[ -d "$DATA_PATH_NEW" ]]; then
        DATA_PATH="$DATA_PATH_NEW"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_40gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_40gb"
    elif [[ -d "$DATA_PATH_OLD" ]]; then
        # Fallback to old naming for backward compatibility
        DATA_PATH="$DATA_PATH_OLD"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed1_40gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_seed1_40gb"
        echo "Note: Using legacy naming (${METHOD}_loss${LOSS}_seed1)"
    else
        # Default to new naming (will create error later if not found)
        DATA_PATH="$DATA_PATH_NEW"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_40gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_40gb"
    fi
fi

# Config file - use pre-generated config from generate_configs.py
CONFIG_DIR="$PROJECT_ROOT/configs/semantickitti/generated"
CONFIG_FILE="$CONFIG_DIR/ptv3_semantickitti_${CONFIG_SUFFIX}.py"

# Fallback for config file - check old naming if new not found
if [[ ! -f "$CONFIG_FILE" && -z "$SEED" ]]; then
    CONFIG_FILE_OLD="$CONFIG_DIR/ptv3_semantickitti_${METHOD}_loss${LOSS}_seed1_40gb.py"
    if [[ -f "$CONFIG_FILE_OLD" ]]; then
        CONFIG_FILE="$CONFIG_FILE_OLD"
    fi
fi

# Output directories
OUTPUT_DIR="$PROJECT_ROOT/outputs/${EXPERIMENT_NAME}"
OUTPUT_DIR_RELATIVE="../SemanticKITTI/outputs/${EXPERIMENT_NAME}"
PID_FILE="$OUTPUT_DIR/train.pid"
# Note: Pointcept creates train.log in OUTPUT_DIR automatically
TRAIN_LOG="$OUTPUT_DIR/train.log"

################################################################################
# Helper Functions
################################################################################

print_header() {
    local SEED_DISPLAY="${SEED:-N/A (deterministic)}"
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  SemanticKITTI PTv3 Training - 40GB GPU                            ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Seed: ${GREEN}${SEED_DISPLAY}${NC}             ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

get_train_log() {
    # Pointcept writes to train.log in the output directory
    echo "$OUTPUT_DIR/train.log"
}

check_disk_space() {
    local required_gb=${1:-100}
    local available_gb=$(df -BG "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ -z "$available_gb" ]; then
        print_warning "Could not check disk space"
        return 0
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space: ${available_gb}GB available, ${required_gb}GB required"
        return 1
    fi
    print_success "Disk space: ${available_gb}GB available"
    return 0
}

check_gpu_memory() {
    local recommended_mb=${1:-35000}  # 35GB recommended
    local minimum_mb=20000  # 20GB absolute minimum
    local available_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null)

    if [ -z "$available_mb" ]; then
        print_warning "Could not query GPU $GPU_ID memory"
        return 0  # Continue anyway
    fi

    if [ "$available_mb" -lt "$minimum_mb" ]; then
        print_error "Insufficient GPU memory on GPU $GPU_ID: ${available_mb}MB free, ${minimum_mb}MB minimum required"
        return 1
    elif [ "$available_mb" -lt "$recommended_mb" ]; then
        print_warning "GPU $GPU_ID: ${available_mb}MB free (recommended: ${recommended_mb}MB)"
        print_warning "Training may run slower due to memory constraints. Consider reducing batch_size if OOM occurs."
    else
        print_success "GPU $GPU_ID memory: ${available_mb}MB free"
    fi
    return 0
}

generate_config_if_needed() {
    # Check if pre-generated config exists
    if [ -f "$CONFIG_FILE" ]; then
        print_success "Config exists: $CONFIG_FILE"
        return 0
    fi

    # Config not found - generate from template
    print_info "Pre-generated config not found, generating from template..."
    mkdir -p "$CONFIG_DIR"

    # Get template
    TEMPLATE_FILE="$PROJECT_ROOT/configs/semantickitti/ptv3_40gb_official_template.py"

    if [ ! -f "$TEMPLATE_FILE" ]; then
        print_error "Template config not found: $TEMPLATE_FILE"
        print_info "Run: python scripts/generate_configs.py --tier priority --gpu 40gb"
        return 1
    fi

    # Create config from template
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"

    # Replace placeholders - use relative path from pointcept
    if [[ "$LOSS" == "0" ]]; then
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/original"
    elif [[ -n "$SEED" ]]; then
        # Non-deterministic method with seed
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    else
        # Deterministic method - use new naming (no seed)
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}"
    fi

    sed -i "s|DATA_ROOT_PLACEHOLDER|$DATA_PATH_RELATIVE|g" "$CONFIG_FILE"
    sed -i "s|{METHOD}|$METHOD|g" "$CONFIG_FILE"
    sed -i "s|{LOSS}|$LOSS|g" "$CONFIG_FILE"
    if [[ -n "$SEED" ]]; then
        sed -i "s|{SEED}|$SEED|g" "$CONFIG_FILE"
        sed -i "s|seed=42|seed=$SEED|g" "$CONFIG_FILE"
    else
        sed -i "s|{SEED}||g" "$CONFIG_FILE"
    fi

    print_success "Config generated: $CONFIG_FILE"
    return 0
}

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at: $VENV_PATH"
        print_info "Run: cd $PTv3_ROOT && ./setup_venv.sh"
        return 1
    fi
    print_success "Virtual environment found"

    # Check data path
    if [ ! -d "$DATA_PATH" ]; then
        print_error "Data path not found: $DATA_PATH"
        if [[ "$LOSS" == "0" ]]; then
            print_info "Original data should be at: $DATA_ROOT/original/sequences/"
        else
            print_info "Run data generation: ./scripts/run_subsampling_phase1.sh"
        fi
        return 1
    fi
    print_success "Data path found: $DATA_PATH"

    # Generate config if needed
    generate_config_if_needed || return 1

    # Check pointcept
    if [ ! -d "$POINTCEPT_DIR" ]; then
        print_error "Pointcept not found: $POINTCEPT_DIR"
        return 1
    fi
    print_success "Pointcept found"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID 2>/dev/null || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null || echo "Unknown")
        print_success "GPU $GPU_ID: $GPU_NAME (${GPU_MEM}MB)"
    else
        print_warning "nvidia-smi not found"
    fi

    # Check disk space
    mkdir -p "$OUTPUT_DIR"
    check_disk_space 100 || return 1

    # Check GPU memory
    check_gpu_memory 35000 || return 1

    return 0
}

create_directories() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/model"
}

################################################################################
# Training Functions
################################################################################

start_training() {
    local RESUME_MODE=$1

    print_header

    if [ "$RESUME_MODE" = "resume" ]; then
        print_info "Resuming training from latest checkpoint..."
    elif [ "$RESUME_MODE" = "extend" ]; then
        print_info "Extending training to $EPOCHS epochs..."
    else
        print_info "Starting fresh training..."
    fi
    echo ""

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_error "Training already running (PID: $PID)"
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

    # Create directories
    create_directories

    # Check for existing checkpoints if resuming or extending
    # Pointcept saves: model_last.pth (most recent), model_best.pth (best metric), epoch_X.pth (periodic)
    RESUME_ARG=""
    EPOCHS_ARG=""
    LATEST_CHECKPOINT=""

    if [ "$RESUME_MODE" = "resume" ] || [ "$RESUME_MODE" = "extend" ]; then
        # Priority: model_last.pth > model_best.pth > epoch_*.pth
        if [ -f "$OUTPUT_DIR/model/model_last.pth" ]; then
            LATEST_CHECKPOINT="$OUTPUT_DIR/model/model_last.pth"
        elif [ -f "$OUTPUT_DIR/model/model_best.pth" ]; then
            LATEST_CHECKPOINT="$OUTPUT_DIR/model/model_best.pth"
        else
            # Find latest epoch checkpoint
            LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/model" -name "epoch_*.pth" 2>/dev/null | \
                sed 's/.*epoch_\([0-9]*\)\.pth/\1 &/' | sort -rn | head -n1 | cut -d' ' -f2-)
        fi

        if [ -z "$LATEST_CHECKPOINT" ]; then
            if [ "$RESUME_MODE" = "extend" ]; then
                print_error "No checkpoint found for extend. Run 'start' first."
                return 1
            else
                print_warning "No checkpoint found, starting fresh..."
            fi
        else
            print_success "Found checkpoint: $LATEST_CHECKPOINT"
            RESUME_ARG="resume=True weight=$LATEST_CHECKPOINT"

            # Show checkpoint epoch info
            if command -v python3 &> /dev/null; then
                CKPT_EPOCH=$(python3 -c "import torch; c=torch.load('$LATEST_CHECKPOINT', map_location='cpu', weights_only=False); print(c.get('epoch', '?'))" 2>/dev/null || echo "?")
                if [ "$CKPT_EPOCH" != "?" ]; then
                    print_info "Checkpoint at epoch: $((CKPT_EPOCH - 1)) completed (will resume from epoch $CKPT_EPOCH)"
                fi
            fi
        fi
    fi

    # Handle epochs override (for extend or custom training)
    if [ ! -z "$EPOCHS" ]; then
        print_info "Epochs override: $EPOCHS"
        EPOCHS_ARG="epoch=$EPOCHS"
        # Note: We pass epoch via command line options, not modifying config file
        # This keeps the config file unchanged for reproducibility
    fi

    # Print configuration
    print_info "Training Configuration:"
    echo "  Method:         $METHOD"
    echo "  Loss:           $LOSS%"
    if [[ -n "$SEED" ]]; then
        echo "  Seed:           $SEED"
    else
        echo "  Seed:           N/A (deterministic method)"
    fi
    echo "  Data Path:      $DATA_PATH"
    echo "  Config:         $CONFIG_FILE"
    echo "  Output:         $OUTPUT_DIR"
    echo "  GPU:            $GPU_ID"
    echo "  Log:            $TRAIN_LOG"
    if [[ "$LOSS" == "0" ]]; then
        echo -e "  ${YELLOW}NOTE: Training on ORIGINAL data (0% loss)${NC}"
    fi
    if [[ "$RESUME_MODE" == "extend" ]]; then
        echo -e "  ${GREEN}MODE: Extending training to $EPOCHS epochs${NC}"
    elif [[ "$RESUME_MODE" == "resume" ]]; then
        echo -e "  ${GREEN}MODE: Resuming from checkpoint${NC}"
    fi
    echo ""

    # Start training
    print_info "Starting training process..."

    (
        set -o pipefail

        source "$VENV_PATH/bin/activate"
        cd "$POINTCEPT_DIR"

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export PYTHONUNBUFFERED=1
        export PYTHONPATH="$POINTCEPT_DIR:${PYTHONPATH:-}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # ============================================================
        # Quick profiling (GFLOPs, Memory) - only on fresh start
        # ============================================================
        PROFILE_SCRIPT="$SCRIPT_DIR/quick_profile.py"
        if [ -f "$PROFILE_SCRIPT" ] && [ "$RESUME_MODE" = "fresh" ]; then
            echo ""
            echo "========================================================================"
            echo "Running quick profiler (GFLOPs, Memory)..."
            echo "========================================================================"
            python "$PROFILE_SCRIPT" \
                --config "$CONFIG_FILE" \
                --output "$TRAIN_LOG" \
                --gpu "$CUDA_VISIBLE_DEVICES" \
                2>&1 || echo "[PROFILE] Skipped due to error"
            echo ""
        fi

        # WandB configuration
        if [ "$ENABLE_WANDB" = "true" ]; then
            export WANDB_PROJECT="$WANDB_PROJECT"
            [ ! -z "$WANDB_ENTITY" ] && export WANDB_ENTITY="$WANDB_ENTITY"
            WANDB_OPTION="enable_wandb=True"
        else
            WANDB_OPTION="enable_wandb=False"
        fi

        # Build options string
        OPTIONS="save_path=$OUTPUT_DIR_RELATIVE $WANDB_OPTION"
        [ ! -z "$RESUME_ARG" ] && OPTIONS="$OPTIONS $RESUME_ARG"
        [ ! -z "$EPOCHS_ARG" ] && OPTIONS="$OPTIONS $EPOCHS_ARG"

        # Note: Pointcept trainer writes to train.log in save_path automatically
        python tools/train.py \
            --config-file "$CONFIG_FILE" \
            --options $OPTIONS
    ) &

    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"
    disown "$TRAIN_PID"

    sleep 3

    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        print_success "Training started (PID: $TRAIN_PID)"
        echo ""
        print_info "Commands:"
        echo "  View logs:    $0 --method $METHOD --loss $LOSS logs"
        echo "  Status:       $0 --method $METHOD --loss $LOSS status"
        echo "  Stop:         $0 --method $METHOD --loss $LOSS stop"
        echo "  Resume:       $0 --method $METHOD --loss $LOSS resume"
    else
        print_error "Failed to start training"
        rm "$PID_FILE"
        return 1
    fi
}

stop_training() {
    print_header
    print_info "Stopping training for: $EXPERIMENT_NAME"
    echo ""

    STOPPED=false

    # Helper function to recursively kill all descendants of a process
    kill_descendants() {
        local parent_pid=$1
        local children=$(pgrep -P $parent_pid 2>/dev/null)
        for child in $children; do
            kill_descendants $child
        done
        if [ -n "$parent_pid" ] && ps -p $parent_pid > /dev/null 2>&1; then
            kill -9 $parent_pid 2>/dev/null || true
        fi
    }

    # Step 1: Kill by PID file if exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        print_info "Found PID file with PID: $PID"

        if ps -p "$PID" > /dev/null 2>&1; then
            print_info "Killing process tree for PID $PID..."

            # First, try graceful termination
            kill -TERM $PID 2>/dev/null || true
            sleep 2

            # Kill all descendants recursively (grandchildren, etc.)
            kill_descendants $PID

            STOPPED=true
        else
            print_warning "Process $PID not running (already stopped)"
            STOPPED=true
        fi
        rm -f "$PID_FILE"
    else
        print_warning "No PID file found at: $PID_FILE"
    fi

    # Step 2: Find and kill any processes using this experiment's config or output path
    print_info "Searching for training processes..."

    # Search patterns - match config file name or output directory
    if [[ -n "$SEED" ]]; then
        CONFIG_PATTERN="ptv3_semantickitti_${METHOD}_loss${LOSS}_seed${SEED}"
    else
        CONFIG_PATTERN="ptv3_semantickitti_${METHOD}_loss${LOSS}"
    fi

    # Find processes matching our experiment
    EXPERIMENT_PIDS=""

    # Method 1: Search by config file pattern
    PIDS1=$(pgrep -f "${CONFIG_PATTERN}" 2>/dev/null | tr '\n' ' ')

    # Method 2: Search by output directory pattern
    PIDS2=$(pgrep -f "save_path=.*${EXPERIMENT_NAME}" 2>/dev/null | tr '\n' ' ')

    # Method 3: Search by train.py with experiment name
    PIDS3=$(ps aux | grep -E "(python|python3).*train\.py.*${EXPERIMENT_NAME}" | grep -v grep | awk '{print $2}' | tr '\n' ' ')

    # Method 4: Search for any python train.py with our config
    PIDS4=$(ps aux | grep -E "python.*train\.py.*${CONFIG_PATTERN}" | grep -v grep | awk '{print $2}' | tr '\n' ' ')

    # Combine all found PIDs (unique)
    EXPERIMENT_PIDS=$(echo "$PIDS1 $PIDS2 $PIDS3 $PIDS4" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

    if [ -n "$(echo $EXPERIMENT_PIDS | tr -d ' ')" ]; then
        print_info "Found processes: $EXPERIMENT_PIDS"
        for pid in $EXPERIMENT_PIDS; do
            if [ -n "$pid" ]; then
                print_info "Killing process tree for PID $pid..."
                kill_descendants $pid
            fi
        done
        STOPPED=true
        sleep 1
    else
        print_info "No matching processes found via pattern search"
    fi

    # Step 3: Verify processes are stopped
    sleep 1
    REMAINING=$(pgrep -f "${CONFIG_PATTERN}" 2>/dev/null | tr '\n' ' ')
    if [ -z "$(echo $REMAINING | tr -d ' ')" ]; then
        if [ "$STOPPED" = true ]; then
            print_success "Training stopped successfully"
        else
            print_info "No training process was running for this experiment"
        fi
    else
        print_error "Some processes may still be running: $REMAINING"
        print_info "Try manually: kill -9 $REMAINING"
        return 1
    fi
}

show_status() {
    print_header

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_success "Training is RUNNING (PID: $PID)"

            # GPU status
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                print_info "GPU $GPU_ID Status:"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                    --format=csv,noheader -i $GPU_ID | \
                    awk -F', ' '{printf "  Utilization: %s | Memory: %s / %s | Temp: %s\n", $1, $2, $3, $4}'
            fi

            # Training progress
            LATEST_LOG=$(get_train_log)
            if [ -f "$LATEST_LOG" ]; then
                echo ""
                print_info "Training Progress:"
                LAST_EPOCH=$(grep -oP 'Train: \[\K[0-9]+(?=/[0-9]+\])' "$LATEST_LOG" 2>/dev/null | tail -n1)
                LAST_LOSS=$(grep -oP 'loss: \K[0-9.]+' "$LATEST_LOG" 2>/dev/null | tail -n1)
                LAST_MIOU=$(grep -oP 'mIoU.*?:\s*\K[0-9.]+' "$LATEST_LOG" 2>/dev/null | tail -n1)
                TOTAL_EPOCHS=$(grep -oP 'Train: \[[0-9]+/\K[0-9]+' "$LATEST_LOG" 2>/dev/null | tail -n1 || echo "?")

                [ ! -z "$LAST_EPOCH" ] && echo "  Epoch: $LAST_EPOCH / $TOTAL_EPOCHS"
                [ ! -z "$LAST_LOSS" ] && echo "  Loss: $LAST_LOSS"
                [ ! -z "$LAST_MIOU" ] && echo "  mIoU: ${LAST_MIOU}%"

                echo ""
                print_info "Recent logs:"
                tail -5 "$LATEST_LOG" | sed 's/^/  /'
            fi
        else
            print_warning "Training NOT RUNNING (stale PID: $PID)"
            rm "$PID_FILE"
        fi
    else
        print_warning "Training NOT RUNNING"
    fi

    # Output info
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_info "Output: $OUTPUT_DIR"
        if [ -d "$OUTPUT_DIR/model" ]; then
            CKPT_COUNT=$(find "$OUTPUT_DIR/model" -name "*.pth" 2>/dev/null | wc -l)
            echo "  Checkpoints: $CKPT_COUNT"
        fi
        DISK=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
        echo "  Disk usage: $DISK"
    fi
}

show_logs() {
    print_header

    LATEST_LOG=$(get_train_log)

    if [ ! -f "$LATEST_LOG" ]; then
        print_warning "No log file found at: $LATEST_LOG"
        return 1
    fi

    print_info "Showing logs: $LATEST_LOG"
    print_info "Press Ctrl+C to exit"
    echo ""

    tail -f "$LATEST_LOG"
}

generate_report() {
    print_header
    print_info "Generating benchmark report..."
    echo ""

    REPORT_FILE="$OUTPUT_DIR/benchmark_report_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "============================================"
        echo "SemanticKITTI Benchmark Report - 40GB GPU"
        echo "============================================"
        echo "Generated: $(date)"
        echo ""

        echo "Experiment Configuration (Official Outdoor PTv3):"
        echo "  Method:          $METHOD"
        echo "  Loss Level:      $LOSS%"
        if [[ -n "$SEED" ]]; then
            echo "  Seed:            $SEED"
        else
            echo "  Seed:            N/A (deterministic method)"
        fi
        echo "  Batch Size:      6 (effective: 48)"
        echo "  Grid Size:       0.05"
        echo "  in_channels:     4 (coord + strength)"
        echo "  LR:              0.002, WD: 0.005"
        echo "  GPU Memory:      ~30-35GB"
        echo ""

        echo "Training Status:"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "  Status:          RUNNING (PID: $PID)"
            else
                echo "  Status:          STOPPED"
            fi
        else
            echo "  Status:          NOT STARTED"
        fi
        echo ""

        LATEST_LOG=$(get_train_log)
        if [ -f "$LATEST_LOG" ]; then
            echo "Training Progress:"
            LAST_EPOCH=$(grep -oP 'Train: \[\K[0-9]+(?=/[0-9]+\])' "$LATEST_LOG" 2>/dev/null | tail -n1)
            BEST_MIOU=$(grep -oP 'Best.*mIoU.*:\K[0-9.]+' "$LATEST_LOG" 2>/dev/null | tail -n1)
            LATEST_MIOU=$(grep -oP 'mIoU.*:\K[0-9.]+' "$LATEST_LOG" 2>/dev/null | tail -n1)
            LATEST_LOSS=$(grep -oP 'loss: \K[0-9.]+' "$LATEST_LOG" 2>/dev/null | tail -n1)

            TOTAL_EPOCHS=$(grep -oP 'Train: \[[0-9]+/\K[0-9]+' "$LATEST_LOG" 2>/dev/null | tail -n1 || echo "?")
            [ ! -z "$LAST_EPOCH" ] && echo "  Epochs:          $LAST_EPOCH / $TOTAL_EPOCHS"
            [ ! -z "$BEST_MIOU" ] && echo "  Best mIoU:       ${BEST_MIOU}%"
            [ ! -z "$LATEST_MIOU" ] && echo "  Latest mIoU:     ${LATEST_MIOU}%"
            [ ! -z "$LATEST_LOSS" ] && echo "  Latest Loss:     $LATEST_LOSS"
        fi

        echo ""
        echo "Output Directory:"
        echo "  Location:        $OUTPUT_DIR"
        echo "  Disk Usage:      $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"

        if [ -d "$OUTPUT_DIR/model" ]; then
            CHECKPOINT_COUNT=$(find "$OUTPUT_DIR/model" -name "*.pth" 2>/dev/null | wc -l)
            echo "  Checkpoints:     $CHECKPOINT_COUNT"
        fi

        echo ""
        echo "============================================"

    } | tee "$REPORT_FILE"

    echo ""
    print_success "Report saved to: $REPORT_FILE"
}

################################################################################
# Main
################################################################################

case "$COMMAND" in
    start)
        start_training "fresh"
        ;;
    resume)
        start_training "resume"
        ;;
    extend)
        start_training "extend"
        ;;
    stop)
        stop_training
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    report)
        generate_report
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
