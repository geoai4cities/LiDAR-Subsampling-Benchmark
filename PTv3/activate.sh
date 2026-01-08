#!/bin/bash
################################################################################
# PTv3 LiDAR Subsampling Benchmark - Environment Activation Script
################################################################################

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$BENCHMARK_ROOT/ptv3_venv"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found at $VENV_PATH"
    echo "  Run ./setup_venv.sh first"
    return 1
fi

# Cache directory configuration (keeps all cache on disk, not home)
export XDG_CACHE_HOME="$VENV_PATH/cache"
export TORCH_HOME="$VENV_PATH/cache/torch"
export HF_HOME="$VENV_PATH/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$VENV_PATH/cache/torch_extensions"
export CUDA_CACHE_PATH="$VENV_PATH/cache/cuda"
echo "✓ Cache directories configured (not in home)"

# Set PYTHONPATH (handle case where PYTHONPATH is not set)
export PYTHONPATH="$SCRIPT_DIR/pointcept:${PYTHONPATH:-}"
echo "✓ PYTHONPATH configured"

# Display environment info
echo ""
echo "PTv3 LiDAR Subsampling Benchmark Environment"
echo "============================================="
echo "Python:   $(python --version 2>&1)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA:     $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "Venv:     $VENV_PATH"
echo "Cache:    $VENV_PATH/cache"
echo ""
