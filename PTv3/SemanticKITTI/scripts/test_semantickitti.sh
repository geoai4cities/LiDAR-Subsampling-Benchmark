#!/bin/bash

################################################################################
# Test Script for SemanticKITTI Subsampling Benchmark
#
# This script runs inference on a trained model and saves predictions for:
# - Benchmark submission to SemanticKITTI
# - Visualization and analysis
# - Comparison between methods
#
# Usage:
#   ./test_semantickitti.sh --method RS --loss 50 --gpu 140gb test
#   ./test_semantickitti.sh --method RS --loss 0 --gpu 140gb test
#
# Methods: RS, IDIS, FPS, DBSCAN, Voxel, Poisson, DEPOCO
# Loss Levels: 0, 10, 30, 50, 70, 90 (0 = original/baseline)
#
# Output:
#   - {output_dir}/result/*.npy       - Per-frame predictions
#   - {output_dir}/result/submit/     - SemanticKITTI submission format
#
################################################################################

set -euo pipefail

################################################################################
# Configuration
################################################################################

METHOD=""
LOSS=""
SEED="1"
GPU_ID="0"
GPU_TYPE="140gb"  # 40gb or 140gb
MODEL_TYPE="best"  # best or last

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

################################################################################
# Parse Arguments
################################################################################

print_usage() {
    echo "Usage: $0 --method METHOD --loss LOSS --gpu GPU_TYPE [options] {test|help}"
    echo ""
    echo "Required Arguments:"
    echo "  --method METHOD   Subsampling method (RS, IDIS, FPS, DBSCAN, Voxel, Poisson, DEPOCO)"
    echo "  --loss LOSS       Loss percentage (0, 10, 30, 50, 70, 90)"
    echo "  --gpu GPU_TYPE    GPU configuration (40gb or 140gb)"
    echo ""
    echo "Optional Arguments:"
    echo "  --seed SEED       Random seed (default: 1)"
    echo "  --gpu-id ID       GPU device ID (default: 0)"
    echo "  --model TYPE      Model to test: 'best' or 'last' (default: best)"
    echo ""
    echo "Commands:"
    echo "  test              Run inference and save predictions"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Test baseline model (trained on original data)"
    echo "  $0 --method RS --loss 0 --gpu 140gb test"
    echo ""
    echo "  # Test model trained on RS subsampled data with 50% loss"
    echo "  $0 --method RS --loss 50 --gpu 140gb test"
    echo ""
    echo "  # Test using last checkpoint instead of best"
    echo "  $0 --method RS --loss 50 --gpu 140gb --model last test"
    echo ""
    echo "Output:"
    echo "  Predictions saved to: outputs/{experiment}/result/"
    echo "  Submit files saved to: outputs/{experiment}/result/submit/"
}

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
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        test|help)
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

# Validate
VALID_METHODS=("RS" "IDIS" "FPS" "DBSCAN" "Voxel" "Poisson" "DEPOCO")
VALID_LOSS=("0" "10" "30" "50" "70" "90")
VALID_GPU=("40gb" "140gb")

if [[ "${COMMAND:-}" == "help" ]]; then
    print_usage
    exit 0
fi

if [[ -z "$METHOD" || -z "$LOSS" || -z "$GPU_TYPE" ]]; then
    echo "Error: --method, --loss, and --gpu are required"
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

if [[ ! " ${VALID_GPU[@]} " =~ " ${GPU_TYPE} " ]]; then
    echo "Error: Invalid gpu '$GPU_TYPE'. Must be one of: ${VALID_GPU[*]}"
    exit 1
fi

if [[ -z "${COMMAND:-}" ]]; then
    echo "Error: Command required (test|help)"
    print_usage
    exit 1
fi

################################################################################
# Setup Paths
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PTv3_ROOT="$(dirname "$PROJECT_ROOT")"
POINTCEPT_DIR="$PTv3_ROOT/pointcept"
VENV_PATH="$PTv3_ROOT/../ptv3_venv"

# Experiment naming
if [[ "$LOSS" == "0" ]]; then
    EXPERIMENT_NAME="baseline_loss0_seed${SEED}_${GPU_TYPE}"
else
    EXPERIMENT_NAME="${METHOD}_loss${LOSS}_seed${SEED}_${GPU_TYPE}"
fi

CONFIG_FILE="$PROJECT_ROOT/configs/semantickitti/generated/ptv3_semantickitti_${METHOD}_loss${LOSS}_seed${SEED}_${GPU_TYPE}.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/${EXPERIMENT_NAME}"
MODEL_DIR="$OUTPUT_DIR/model"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}  SemanticKITTI PTv3 Testing - Save Predictions                     ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}║${NC}  Method: ${GREEN}${METHOD}${NC} | Loss: ${GREEN}${LOSS}%${NC} | GPU: ${GREEN}${GPU_TYPE}${NC}                         ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

################################################################################
# Test Function
################################################################################

run_test() {
    print_header

    # Check prerequisites
    print_info "Checking prerequisites..."

    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found: $VENV_PATH"
        exit 1
    fi
    print_success "Virtual environment found"

    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Config file found"

    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Model directory not found: $MODEL_DIR"
        print_info "Train the model first using train_semantickitti_${GPU_TYPE}.sh"
        exit 1
    fi

    # Find model checkpoint
    if [[ "$MODEL_TYPE" == "best" ]]; then
        MODEL_PATH="$MODEL_DIR/model_best.pth"
    else
        MODEL_PATH="$MODEL_DIR/model_last.pth"
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        print_error "Model checkpoint not found: $MODEL_PATH"
        exit 1
    fi
    print_success "Model checkpoint found: $MODEL_PATH"

    echo ""
    print_info "Test Configuration:"
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  Method:         $METHOD"
    echo "  │  Loss:           $LOSS%"
    echo "  │  Seed:           $SEED"
    echo "  │  GPU Type:       $GPU_TYPE"
    echo "  │  GPU ID:         $GPU_ID"
    echo "  │  Model:          $MODEL_TYPE ($MODEL_PATH)"
    echo "  │  Config:         $CONFIG_FILE"
    echo "  │  Output:         $OUTPUT_DIR/result/"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    echo ""

    # Create temporary config with save_predictions=True
    TEMP_CONFIG="$OUTPUT_DIR/test_config_temp.py"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"

    # Update save_predictions to True
    if grep -q "save_predictions = False" "$TEMP_CONFIG"; then
        sed -i 's/save_predictions = False/save_predictions = True/' "$TEMP_CONFIG"
    else
        # Add if not present
        echo "save_predictions = True" >> "$TEMP_CONFIG"
    fi
    print_success "Created temp config with save_predictions=True"

    # Run test
    print_info "Starting inference..."
    echo ""

    (
        source "$VENV_PATH/bin/activate"
        cd "$POINTCEPT_DIR"

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export PYTHONUNBUFFERED=1
        export PYTHONPATH="$POINTCEPT_DIR:${PYTHONPATH:-}"

        python tools/test.py \
            --config-file "$TEMP_CONFIG" \
            --options save_path="$OUTPUT_DIR" weight="$MODEL_PATH"
    )

    # Cleanup temp config
    rm -f "$TEMP_CONFIG"

    echo ""
    print_success "Testing completed!"
    echo ""

    # Show output info
    if [ -d "$OUTPUT_DIR/result" ]; then
        print_info "Results saved:"
        PRED_COUNT=$(find "$OUTPUT_DIR/result" -name "*_pred.npy" 2>/dev/null | wc -l)
        RESULT_SIZE=$(du -sh "$OUTPUT_DIR/result" 2>/dev/null | cut -f1)
        echo "  Predictions: $PRED_COUNT files"
        echo "  Total size: $RESULT_SIZE"

        if [ -d "$OUTPUT_DIR/result/submit" ]; then
            echo ""
            print_info "Submit files ready at:"
            echo "  $OUTPUT_DIR/result/submit/"
        fi
    fi
}

################################################################################
# Main
################################################################################

case "$COMMAND" in
    test)
        run_test
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
