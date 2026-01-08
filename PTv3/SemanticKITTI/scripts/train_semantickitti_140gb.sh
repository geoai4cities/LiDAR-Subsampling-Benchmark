#!/bin/bash

################################################################################
# Training Script for SemanticKITTI Subsampling Benchmark - 140GB GPU
#
# Usage:
#   ./train_semantickitti_140gb.sh --method METHOD --loss LOSS [--seed SEED] COMMAND
#
# Examples:
#   # Deterministic methods (IDIS, DBSCAN, Voxel) - no seed needed
#   ./train_semantickitti_140gb.sh --method IDIS --loss 50 start
#   ./train_semantickitti_140gb.sh --method DBSCAN --loss 50 start
#   ./train_semantickitti_140gb.sh --method Voxel --loss 50 start
#
#   # Non-deterministic methods (RS, FPS, Poisson) - seed required
#   ./train_semantickitti_140gb.sh --method RS --loss 50 --seed 1 start
#   ./train_semantickitti_140gb.sh --method FPS --loss 50 --seed 1 start
#   ./train_semantickitti_140gb.sh --method Poisson --loss 50 --seed 1 start
#
#   # Train on original data (0% loss)
#   ./train_semantickitti_140gb.sh --method RS --loss 0 start
#
# Extending Training:
#   ./train_semantickitti_140gb.sh --method IDIS --loss 50 --epochs 50 extend
#
# Methods:
#   Deterministic (no seed):     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel
#   Non-deterministic (seed):    RS, FPS, Poisson
#   Special:                     DEPOCO (external model)
#
# Loss Levels: 0, 10, 30, 50, 70, 90 (0 = original data)
#
################################################################################
# Model Configuration (Official Outdoor PTv3 Settings)
################################################################################
#   Model:            PT-v3m1 (DefaultSegmentorV2)
#   in_channels:      4 (coord: 3 + strength: 1)
#   num_classes:      19 (SemanticKITTI classes)
#   Encoder depths:   (2, 2, 2, 6, 2)
#   Encoder channels: (32, 64, 128, 256, 512)
#   Decoder depths:   (2, 2, 2, 2)
#   Decoder channels: (64, 64, 128, 256)
#   Patch size:       1024
#   drop_path:        0.3
#   Flash Attention:  Enabled
#
################################################################################
# Training Configuration
################################################################################
#   GPU:              H200 140GB / H100 80GB
#   Memory Usage:     ~80GB
#   Epochs:           10 (default), extend to 50 for full training
#   Batch Size:       20 (train), 1 (val), 1 (test)
#   Grad Accumulation: 4 (effective batch: 80)
#   Grid Size:        0.05 (official outdoor setting)
#   PointClip Range:  (-35.2, -35.2, -4) to (35.2, 35.2, 2)
#
#   NOTE: batch_size_val=1 is REQUIRED for correct inverse mapping
#         in SemSegEvaluator (fixes mIoU computation bug)
#
################################################################################
# Optimizer & Scheduler
################################################################################
#   Optimizer:        AdamW
#   Learning Rate:    0.002 (backbone: 0.0002)
#   Weight Decay:     0.005
#   Scheduler:        OneCycleLR
#   pct_start:        0.04
#   final_div_factor: 100.0
#   Gradient Clip:    0.5
#   AMP:              bfloat16
#
################################################################################
# Loss Functions
################################################################################
#   CrossEntropyLoss: Weighted (class-balanced)
#   LovaszLoss:       multiclass, weight=1.0
#
################################################################################
# Hooks & Evaluation
################################################################################
#   SemSegEvaluator:  Softmax accumulation + inverse mapping (batch_size=1)
#   EarlyStopping:    patience=15, min_delta=0.001, metric=mIoU
#   CheckpointSaver:  model_best.pth + model_last.pth only (no epoch_*.pth)
#   PreciseEvaluator: TTA with 4 rotations (0, 90, 180, 270 deg)
#
################################################################################
# Time & Resource Estimates (from actual training logs)
################################################################################
#   Time per Epoch:   ~2 hours (training + validation)
#   Total 10 Epochs:  ~22 hours (incl. final PreciseEvaluator testing)
#   Total 50 Epochs:  ~105 hours (~4.4 days)
#   Iterations/Epoch: 4782
#   Disk Space:       ~100GB recommended
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
EPOCHS=""  # Override epochs if specified
COMMAND=""  # Command to execute

# WandB Configuration (Set to true to enable logging)
ENABLE_WANDB=false
WANDB_PROJECT="lidar-subsampling-benchmark"
WANDB_ENTITY=""

################################################################################
# Parse Arguments
################################################################################

print_usage() {
    echo "Usage: $0 --method METHOD --loss LOSS [--seed SEED] [--gpu GPU_ID] [--epochs N] COMMAND"
    echo ""
    echo "Required Arguments:"
    echo "  --method METHOD   Subsampling method"
    echo "  --loss LOSS       Loss percentage (0, 10, 30, 50, 70, 90)"
    echo "                    NOTE: loss=0 trains on ORIGINAL data (no subsampling)"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed SEED       Random seed (1, 2, or 3)"
    echo "                    Required for: RS, FPS, Poisson (non-deterministic)"
    echo "                    Ignored for:  IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel (deterministic)"
    echo "  --gpu GPU_ID      GPU device ID (default: 0)"
    echo "  --epochs N        Override number of epochs (default: 10)"
    echo ""
    echo "Method Properties:"
    echo "  Deterministic (no seed):     IDIS, IDIS_R5, IDIS_R15, IDIS_R20, DBSCAN, Voxel"
    echo "  Non-deterministic (seed):    RS, FPS, Poisson"
    echo "  Special:                     DEPOCO"
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
    echo "  # Deterministic methods (no seed needed)"
    echo "  $0 --method IDIS --loss 50 start"
    echo "  $0 --method DBSCAN --loss 50 start"
    echo "  $0 --method Voxel --loss 50 start"
    echo ""
    echo "  # Non-deterministic methods (seed required)"
    echo "  $0 --method RS --loss 50 --seed 1 start"
    echo "  $0 --method FPS --loss 50 --seed 1 start"
    echo "  $0 --method Poisson --loss 50 --seed 1 start"
    echo ""
    echo "  # IDIS with different radii (ablation study)"
    echo "  $0 --method IDIS_R5 --loss 90 start"
    echo "  $0 --method IDIS_R15 --loss 90 start"
    echo "  $0 --method IDIS_R20 --loss 90 start"
    echo ""
    echo "  # Train on original data (0% loss)"
    echo "  $0 --method RS --loss 0 start"
    echo ""
    echo "  # Extend training to 50 epochs"
    echo "  $0 --method IDIS --loss 50 --epochs 50 extend"
    echo ""
    echo "Output Directories:"
    echo "  Deterministic:     {METHOD}_loss{XX}_140gb/"
    echo "  Non-deterministic: {METHOD}_loss{XX}_seed{N}_140gb/"
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
VALID_METHODS=("RS" "IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "FPS" "DBSCAN" "Voxel" "Poisson" "DEPOCO")
VALID_LOSS=("0" "10" "30" "50" "70" "90")

# Deterministic methods don't need seed
DETERMINISTIC_METHODS=("IDIS" "IDIS_R5" "IDIS_R15" "IDIS_R20" "DBSCAN" "Voxel" "DEPOCO")
NON_DETERMINISTIC_METHODS=("RS" "FPS" "Poisson")

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
MAGENTA='\033[0;35m'
NC='\033[0m'

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PTv3_ROOT="$(dirname "$PROJECT_ROOT")"
POINTCEPT_DIR="$PTv3_ROOT/pointcept"
VENV_PATH="$PTv3_ROOT/../ptv3_venv"
DATA_ROOT="$PTv3_ROOT/../data/SemanticKITTI"

# Experiment naming and data path
# Deterministic methods: prefer no seed in path, fallback to _seed1 for compatibility
# Non-deterministic methods: always include seed in path
if [[ "$LOSS" == "0" ]]; then
    # Training on original data
    DATA_PATH="$DATA_ROOT/original"
    EXPERIMENT_NAME="baseline_loss0_140gb"
    CONFIG_SUFFIX="baseline_loss0_140gb"
elif [[ -z "$SEED" ]]; then
    # Deterministic method (IDIS, DBSCAN, Voxel)
    # Try new naming first (no seed), fallback to old naming (_seed1)
    DATA_PATH_NEW="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}"
    DATA_PATH_OLD="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}_seed1"

    if [[ -d "$DATA_PATH_NEW" ]]; then
        DATA_PATH="$DATA_PATH_NEW"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_140gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_140gb"
    elif [[ -d "$DATA_PATH_OLD" ]]; then
        # Fallback to old naming with _seed1
        DATA_PATH="$DATA_PATH_OLD"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed1_140gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_seed1_140gb"
        echo "Note: Using legacy naming (${METHOD}_loss${LOSS}_seed1)"
    else
        # Default to new naming (will be created or error in prerequisites check)
        DATA_PATH="$DATA_PATH_NEW"
        EXPERIMENT_NAME="${METHOD}_loss${LOSS}_140gb"
        CONFIG_SUFFIX="${METHOD}_loss${LOSS}_140gb"
    fi
else
    # Non-deterministic method (RS, FPS, Poisson) - always include seed
    DATA_PATH="$DATA_ROOT/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed${SEED}_140gb"
    CONFIG_SUFFIX="${METHOD}_loss${LOSS}_seed${SEED}_140gb"
fi

# Config file - use pre-generated config from generate_configs.py
CONFIG_DIR="$PROJECT_ROOT/configs/semantickitti/generated"
CONFIG_FILE="$CONFIG_DIR/ptv3_semantickitti_${CONFIG_SUFFIX}.py"

# Fallback: check for config with alternative naming (legacy _seed1 suffix)
if [[ ! -f "$CONFIG_FILE" && -z "$SEED" ]]; then
    CONFIG_FILE_OLD="$CONFIG_DIR/ptv3_semantickitti_${METHOD}_loss${LOSS}_seed1_140gb.py"
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
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}  SemanticKITTI PTv3 Training - 140GB GPU                           ${MAGENTA}║${NC}"
    if [[ -n "$SEED" ]]; then
        echo -e "${MAGENTA}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Seed: ${GREEN}${SEED}${NC}                              ${MAGENTA}║${NC}"
    else
        echo -e "${MAGENTA}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | Deterministic                      ${MAGENTA}║${NC}"
    fi
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════╝${NC}"
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
    local recommended_mb=${1:-70000}  # 70GB recommended (actual usage ~80GB)
    local minimum_mb=40000  # 40GB absolute minimum
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
    TEMPLATE_FILE="$PROJECT_ROOT/configs/semantickitti/ptv3_140gb_official_template.py"

    if [ ! -f "$TEMPLATE_FILE" ]; then
        print_error "Template config not found: $TEMPLATE_FILE"
        print_info "Run: python scripts/generate_configs.py --tier priority --gpu 140gb"
        return 1
    fi

    # Create config from template
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"

    # Replace placeholders - use relative path from pointcept
    if [[ "$LOSS" == "0" ]]; then
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/original"
    elif [[ -z "$SEED" ]]; then
        # Deterministic method - no seed in path
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}"
    else
        # Non-deterministic method - seed in path
        DATA_PATH_RELATIVE="../../data/SemanticKITTI/subsampled/${METHOD}_loss${LOSS}_seed${SEED}"
    fi

    sed -i "s|DATA_ROOT_PLACEHOLDER|$DATA_PATH_RELATIVE|g" "$CONFIG_FILE"
    sed -i "s|{METHOD}|$METHOD|g" "$CONFIG_FILE"
    sed -i "s|{LOSS}|$LOSS|g" "$CONFIG_FILE"
    if [[ -n "$SEED" ]]; then
        sed -i "s|{SEED}|$SEED|g" "$CONFIG_FILE"
        sed -i "s|seed=42|seed=$SEED|g" "$CONFIG_FILE"
    else
        sed -i "s|{SEED}||g" "$CONFIG_FILE"
        sed -i "s|_seed||g" "$CONFIG_FILE"
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

    # Check GPU memory (70GB minimum, actual usage ~80GB)
    check_gpu_memory 70000 || return 1

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

    if [ "$RESUME_MODE" = "extend" ]; then
        print_info "Extending training with more epochs..."
    elif [ "$RESUME_MODE" = "resume" ]; then
        print_info "Resuming training from latest checkpoint..."
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
    EPOCHS_ARG=""
    if [ ! -z "$EPOCHS" ]; then
        print_info "Epochs override: $EPOCHS"
        EPOCHS_ARG="epoch=$EPOCHS"
        # Note: We pass epoch via command line options, not modifying config file
        # This keeps the config file unchanged for reproducibility
    fi

    # Print configuration
    echo ""
    print_info "Training Configuration:"
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
    echo "  │  Config:         $CONFIG_FILE"
    echo "  │  Output:         $OUTPUT_DIR"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Model (PT-v3m1)                                                 │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  in_channels:    4 (coord + strength)"
    echo "  │  num_classes:    19"
    echo "  │  Grid Size:      0.05"
    echo "  │  PointClip:      (-35.2, -35.2, -4) to (35.2, 35.2, 2)"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Training                                                        │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  GPU:            $GPU_ID"
    echo "  │  Batch Size:     20 (effective: 80 with grad_accum=4)"
    echo "  │  Optimizer:      AdamW (lr=0.002, wd=0.005)"
    echo "  │  Scheduler:      OneCycleLR (pct_start=0.04)"
    echo "  │  AMP:            bfloat16"
    echo "  │  Loss:           CrossEntropyLoss + LovaszLoss"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │ Resources                                                       │"
    echo "  ├─────────────────────────────────────────────────────────────────┤"
    echo "  │  Memory:         ~80GB"
    echo "  │  Time/Epoch:     ~2 hours"
    echo "  │  Iters/Epoch:    4782"
    echo "  │  Log:            $TRAIN_LOG"
    echo "  └─────────────────────────────────────────────────────────────────┘"
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
        CONFIG_PATTERN="ptv3_semantickitti_${METHOD}_loss${LOSS}_140gb"
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
        echo "SemanticKITTI Benchmark Report - 140GB GPU"
        echo "============================================"
        echo "Generated: $(date)"
        echo ""

        echo "Experiment Configuration:"
        echo "  Method:          $METHOD"
        echo "  Loss Level:      $LOSS%"
        if [[ -n "$SEED" ]]; then
            echo "  Seed:            $SEED"
        else
            echo "  Deterministic:   Yes (no seed needed)"
        fi
        echo ""

        echo "Model Configuration (PT-v3m1):"
        echo "  in_channels:     4 (coord + strength)"
        echo "  num_classes:     19"
        echo "  Grid Size:       0.05"
        echo "  PointClip:       (-35.2, -35.2, -4) to (35.2, 35.2, 2)"
        echo ""

        echo "Training Configuration:"
        echo "  Batch Size:      20 (effective: 80 with grad_accum=4)"
        echo "  Optimizer:       AdamW (lr=0.002, wd=0.005)"
        echo "  Scheduler:       OneCycleLR (pct_start=0.04)"
        echo "  AMP:             bfloat16"
        echo "  Loss:            CrossEntropyLoss + LovaszLoss"
        echo ""

        echo "Resource Usage:"
        echo "  GPU Memory:      ~80GB"
        echo "  Time/Epoch:      ~2 hours"
        echo "  Iters/Epoch:     4782"
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
