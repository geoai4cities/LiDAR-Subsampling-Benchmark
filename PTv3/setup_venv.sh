#!/bin/bash
################################################################################
# PTv3 LiDAR Subsampling Benchmark - Automated Setup Script
#
# This script automates the complete environment setup for PTv3 benchmarking
# across multiple datasets (SemanticKITTI, DALES) with subsampling methods.
#
# Usage:
#   ./setup_venv.sh [--cuda-version 11.8|12.x|cpu] [--skip-compile]
#
# Options:
#   --cuda-version    Specify CUDA version (default: auto-detect)
#   --skip-compile    Skip CUDA extension compilation (faster, for testing)
#   --help           Show this help message
#
# Requirements:
#   - Python 3.9-3.11
#   - uv package manager (recommended) or pip
#   - CUDA toolkit (optional, for GPU training)
#
################################################################################
#
# CRITICAL LESSONS LEARNED (November 2025):
# =========================================
#
# 1. SPCONV VERSION: Always use spconv-cu118, NOT spconv-cu120/cu124
#    - Problem: spconv-cu120 causes "Floating point exception (core dumped)"
#      crash on H200 GPU during training, even though imports work fine
#    - Solution: Use spconv-cu118==2.3.8 for ALL CUDA versions
#    - Discovered by comparing working BhopalMLS venv vs broken ptv3_venv
#
# 2. FLASH ATTENTION: Let it find CUDA naturally, don't set CUDA_HOME
#    - Problem: Complex CUDA_HOME detection logic caused build failures
#    - Solution: Simple "uv pip install flash-attn --no-build-isolation"
#      Flash Attention finds the correct CUDA automatically
#
# 3. SETUPTOOLS: Must be installed before flash-attn and pointops
#    - Problem: "ModuleNotFoundError: No module named 'setuptools'" during build
#    - Solution: Install setuptools, wheel, ninja, packaging before building
#
# 4. VENV CREATION: Use "uv venv" instead of "python -m venv"
#    - uv creates cleaner environments that work better with CUDA extensions
#
# 5. POINTOPS: Use venv pip for fallback, NOT system pip
#    - Problem: "pip install" falls back to system pip, installs to wrong location
#    - Solution: Always use "$VENV_PATH/bin/pip" explicitly
#
# 6. NUM_WORKERS: Keep at 12 or less for 20-core systems
#    - Rule: num_workers = min(nproc - 4, 12) to leave cores for system
#
# Reference: Working environment from BhopalMLS project
# - Python 3.9.13, PyTorch 2.5.0+cu124, Flash Attention 2.8.3
# - spconv-cu118==2.3.8, triton 3.1.0
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$BENCHMARK_ROOT/ptv3_venv"
PYTHON_VERSION="3.11"
SKIP_COMPILE=false
CUDA_VERSION_ARG=""
TOTAL_STEPS=10

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}  $1${MAGENTA}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "\n${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

show_help() {
    cat << EOF
PTv3 LiDAR Subsampling Benchmark - Automated Setup Script

Usage: $0 [OPTIONS]

Options:
  --cuda-version VERSION    Specify CUDA version (11.8, 12.x, or cpu)
                            Default: auto-detect from nvcc
  --skip-compile            Skip CUDA extension compilation
  --help                    Show this help message

Examples:
  $0                        # Auto-detect CUDA and setup everything
  $0 --cuda-version 11.8    # Force CUDA 11.8
  $0 --cuda-version cpu     # CPU-only installation
  $0 --skip-compile         # Skip CUDA extensions (faster)

This script will:
  1. Check prerequisites (Python, CUDA, Pointcept)
  2. Setup virtual environment at $BENCHMARK_ROOT/ptv3_venv
  3. Configure cache directories (keeps cache on disk, not home)
  4. Install Python dependencies (numpy, scipy, sklearn, etc.)
  5. Install PyTorch with CUDA support
  6. Install PyTorch Geometric + Flash Attention
  7. Compile CUDA extensions (pointops)
  8. Register DALES dataset
  9. Verify installation
  10. Create activation script

Datasets supported:
  - SemanticKITTI (already in Pointcept)
  - DALES (will be registered)

Subsampling methods:
  - RS (Random Sampling)
  - IDIS (Inverse Distance Importance Sampling)
  - FPS (Farthest Point Sampling) - NEW
  - DBSCAN - NEW

CRITICAL NOTES (from debugging Nov 2025):
  - spconv: ALWAYS use spconv-cu118 (cu120 causes FPE crash on H200)
  - flash-attn: Let it find CUDA naturally, don't set CUDA_HOME
  - setuptools: Must be installed before building extensions
  - Reference working env: BhopalMLS with Python 3.9.13, PyTorch 2.5.0+cu124

EOF
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION_ARG="$2"
            shift 2
            ;;
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

################################################################################
# Step 1: Check Prerequisites
################################################################################

print_header "Step 1/$TOTAL_STEPS: Checking Prerequisites"

# Check for package manager (prefer uv, install if not found)
if command -v uv &> /dev/null; then
    PKG_MANAGER="uv"
    print_success "uv found: $(uv --version)"
else
    print_warning "uv not found, installing uv..."
    print_info "uv is a fast Python package manager (10-100x faster than pip)"

    # Install uv
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        print_success "uv installed successfully"

        # Add uv to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"

        # Also try cargo bin path (alternative install location)
        if [ -d "$HOME/.cargo/bin" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        # Verify uv is now available
        if command -v uv &> /dev/null; then
            PKG_MANAGER="uv"
            print_success "uv ready: $(uv --version)"
        else
            print_warning "uv installed but not in PATH, using pip as fallback"
            print_info "Add to your ~/.bashrc: export PATH=\"\$HOME/.local/bin:\$PATH\""
            PKG_MANAGER="pip"
        fi
    else
        print_warning "Failed to install uv, using pip as fallback"
        PKG_MANAGER="pip"
    fi
fi

# Check Python version
if command -v python$PYTHON_VERSION &> /dev/null; then
    print_success "Python $PYTHON_VERSION found"
else
    print_warning "Python $PYTHON_VERSION not found, will use available Python"
    PYTHON_VERSION=""
fi

# Detect CUDA version if not specified
if [ -z "$CUDA_VERSION_ARG" ]; then
    CUDA_VERSION=""

    # Method 1: Try nvcc first (CUDA toolkit in PATH)
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        print_success "CUDA detected via nvcc: $CUDA_VERSION"
    fi

    # Method 2: Check common CUDA toolkit locations if nvcc not in PATH
    if [ -z "$CUDA_VERSION" ]; then
        for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-11.8 /opt/cuda; do
            if [ -x "$cuda_path/bin/nvcc" ]; then
                CUDA_VERSION=$("$cuda_path/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
                print_success "CUDA detected at $cuda_path: $CUDA_VERSION"
                # Add to PATH for later use
                export PATH="$cuda_path/bin:$PATH"
                export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
                print_info "Added $cuda_path to PATH"
                break
            fi
        done
    fi

    # Method 3: Fallback to nvidia-smi (driver provides CUDA version even without toolkit)
    if [ -z "$CUDA_VERSION" ] && command -v nvidia-smi &> /dev/null; then
        # Extract CUDA version from nvidia-smi output (e.g., "CUDA Version: 12.8")
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        if [ -n "$CUDA_VERSION" ]; then
            print_success "CUDA detected via nvidia-smi (driver): $CUDA_VERSION"
            print_info "Note: nvcc not in PATH, but GPU driver supports CUDA $CUDA_VERSION"
            print_info "For compiling extensions, you may need: export PATH=/usr/local/cuda/bin:\$PATH"
        fi
    fi

    # Determine PyTorch CUDA version based on detected CUDA
    if [ -n "$CUDA_VERSION" ]; then
        if [[ "$CUDA_VERSION" =~ ^11\. ]]; then
            PYTORCH_CUDA="cu118"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        elif [[ "$CUDA_VERSION" =~ ^12\. ]]; then
            PYTORCH_CUDA="cu124"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        else
            print_warning "Unknown CUDA version $CUDA_VERSION, defaulting to cu124"
            PYTORCH_CUDA="cu124"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        fi
    else
        print_warning "No CUDA detected, will use CPU-only PyTorch"
        PYTORCH_CUDA="cpu"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
else
    # Use user-specified CUDA version
    case $CUDA_VERSION_ARG in
        11.8)
            PYTORCH_CUDA="cu118"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            ;;
        12.x|12.*|12)
            PYTORCH_CUDA="cu124"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            ;;
        cpu)
            PYTORCH_CUDA="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            ;;
        *)
            print_error "Invalid CUDA version: $CUDA_VERSION_ARG"
            exit 1
            ;;
    esac
    print_info "Using specified CUDA version: $CUDA_VERSION_ARG (PyTorch: $PYTORCH_CUDA)"
fi

# Check if Pointcept exists
if [ ! -d "$SCRIPT_DIR/pointcept" ]; then
    print_error "Pointcept not found at $SCRIPT_DIR/pointcept"
    echo "Please clone it first:"
    echo "  cd $SCRIPT_DIR"
    echo "  git clone https://github.com/Pointcept/Pointcept.git pointcept"
    exit 1
fi
print_success "Pointcept found"

################################################################################
# Step 2: Setup Virtual Environment
################################################################################

print_header "Step 2/$TOTAL_STEPS: Setting Up Virtual Environment"

if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at $VENV_PATH"
    print_info "Using existing environment (will update packages)"
    print_info "To recreate: rm -rf $VENV_PATH && bash setup_venv.sh"
else
    print_step "Creating independent PTv3 virtual environment..."
    print_info "Location: $VENV_PATH"
    # Use uv venv if available (matches working BhopalMLS setup)
    if [ "$PKG_MANAGER" = "uv" ]; then
        if [ -z "$PYTHON_VERSION" ]; then
            uv venv "$VENV_PATH"
        else
            uv venv "$VENV_PATH" --python $PYTHON_VERSION || uv venv "$VENV_PATH"
        fi
    else
        python3 -m venv "$VENV_PATH"
    fi
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
print_success "Environment activated: $(which python)"

################################################################################
# Step 3: Configure Cache Directories
################################################################################

print_header "Step 3/$TOTAL_STEPS: Configuring Cache Directories"

print_step "Setting up cache directories on disk (not home)..."
CACHE_DIR="$VENV_PATH/cache"
mkdir -p "$CACHE_DIR"/{torch,huggingface,torch_extensions,cuda}

# Add cache configuration to environment (will be persisted in activate script later)
export XDG_CACHE_HOME="$VENV_PATH/cache"
export TORCH_HOME="$VENV_PATH/cache/torch"
export HF_HOME="$VENV_PATH/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$VENV_PATH/cache/torch_extensions"
export CUDA_CACHE_PATH="$VENV_PATH/cache/cuda"

print_success "Cache directories configured at $CACHE_DIR"
print_info "Models and cache will be stored in venv, not home directory"

################################################################################
# Step 4: Install Core Python Dependencies
################################################################################

print_header "Step 4/$TOTAL_STEPS: Installing Core Python Dependencies"

print_step "Installing core scientific computing libraries..."
print_info "numpy, scipy, scikit-learn, pandas, h5py, tqdm..."

if [ "$PKG_MANAGER" = "uv" ]; then
    uv pip install numpy scipy scikit-learn pandas h5py tqdm pyyaml
else
    pip install numpy scipy scikit-learn pandas h5py tqdm pyyaml
fi

print_success "Core dependencies installed"

################################################################################
# Step 4: Install PyTorch
################################################################################

print_header "Step 5/$TOTAL_STEPS: Installing PyTorch"

print_step "Installing PyTorch 2.5.0 for $PYTORCH_CUDA..."
if [ "$PYTORCH_CUDA" = "cpu" ]; then
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install torch==2.5.0 torchvision==0.20.0 --index-url $TORCH_INDEX
    else
        pip install torch==2.5.0 torchvision==0.20.0 --index-url $TORCH_INDEX
    fi
else
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install torch==2.5.0+$PYTORCH_CUDA torchvision==0.20.0+$PYTORCH_CUDA --index-url $TORCH_INDEX
    else
        pip install torch==2.5.0+$PYTORCH_CUDA torchvision==0.20.0+$PYTORCH_CUDA --index-url $TORCH_INDEX
    fi
fi

print_success "PyTorch installed for $PYTORCH_CUDA"

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" || print_warning "PyTorch verification warning"

################################################################################
# Step 5: Install PyTorch Geometric and Extensions
################################################################################

print_header "Step 6/$TOTAL_STEPS: Installing PyTorch Geometric"

print_step "Installing PyTorch Geometric and extensions..."
if [ "$PYTORCH_CUDA" = "cpu" ]; then
    print_info "Installing torch-geometric, torch-scatter, torch-cluster, torch-sparse (CPU)..."
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install torch-geometric
        uv pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
    else
        pip install torch-geometric
        pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
    fi
else
    print_info "Installing torch-geometric, torch-scatter, torch-cluster, torch-sparse ($PYTORCH_CUDA)..."
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install torch-geometric
        uv pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+$PYTORCH_CUDA.html
    else
        pip install torch-geometric
        pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+$PYTORCH_CUDA.html
    fi
fi
print_success "PyTorch Geometric and extensions installed"

# Install spconv (sparse convolutions for 3D point clouds)
# NOTE: Always use spconv-cu118 regardless of CUDA version - cu120 causes "Floating point exception"
# This was discovered by comparing working BhopalMLS venv (cu118) vs broken ptv3_venv (cu120)
print_step "Installing spconv (sparse convolutions)..."
if [ "$PYTORCH_CUDA" = "cpu" ]; then
    print_warning "Skipping spconv (requires CUDA)"
else
    # CRITICAL: Use spconv-cu118 for ALL CUDA versions (cu120 causes FPE crash on H200)
    print_info "Installing spconv-cu118 (works with all CUDA versions)..."
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install spconv-cu118==2.3.8
    else
        pip install spconv-cu118==2.3.8
    fi
    print_success "spconv-cu118 installed"
fi

################################################################################
# Step 6: Install Additional Deep Learning Libraries
################################################################################

print_header "Step 7/$TOTAL_STEPS: Installing Additional Libraries"

# Install Flash Attention (CUDA only)
# NOTE: Simplified to match working BhopalMLS setup - no CUDA_HOME manipulation
if [ "$PYTORCH_CUDA" != "cpu" ]; then
    print_step "Installing Flash Attention 2..."
    print_warning "CRITICAL: Flash Attention is REQUIRED for PT-v3m1 configs!"
    print_info "This compiles CUDA kernels and takes 15-30 minutes"
    print_info "You'll see 'Building flash-attn' multiple times - this is normal"
    echo ""

    # Install build dependencies first (required for flash-attn)
    print_info "Installing build dependencies (setuptools, wheel, ninja)..."
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install setuptools wheel ninja packaging --upgrade
    else
        pip install setuptools wheel ninja packaging --upgrade
    fi

    # Simple installation - let flash-attn find CUDA naturally (like BhopalMLS)
    if [ "$PKG_MANAGER" = "uv" ]; then
        if uv pip install flash-attn --no-build-isolation -v; then
            print_success "Flash Attention installed successfully!"
            python -c "import flash_attn; print(f'   Version: {flash_attn.__version__}')" 2>/dev/null || true
        else
            print_warning "Primary installation failed, trying from GitHub..."
            if uv pip install "git+https://github.com/Dao-AILab/flash-attention.git" --no-build-isolation -v; then
                print_success "Flash Attention installed from GitHub!"
                python -c "import flash_attn; print(f'   Version: {flash_attn.__version__}')" 2>/dev/null || true
            else
                print_warning "Flash Attention installation failed"
                print_info "PT-v3m1 configs require Flash Attention"
                print_info "Manual fix: pip install flash-attn --no-build-isolation"
            fi
        fi
    else
        if pip install flash-attn --no-build-isolation -v; then
            print_success "Flash Attention installed successfully!"
            python -c "import flash_attn; print(f'   Version: {flash_attn.__version__}')" 2>/dev/null || true
        else
            print_warning "Flash Attention installation failed"
            print_info "Try: pip install flash-attn --no-build-isolation"
        fi
    fi
else
    print_info "Skipping Flash Attention (CPU-only installation)"
fi

# Install Weights & Biases for experiment tracking
print_step "Installing Weights & Biases..."
if [ "$PKG_MANAGER" = "uv" ]; then
    uv pip install wandb
else
    pip install wandb
fi
print_success "Weights & Biases installed"

# Install additional tools
print_step "Installing additional tools (termcolor, easydict, einops, addict, yapf, tensorboardX, peft, timm)..."
if [ "$PKG_MANAGER" = "uv" ]; then
    uv pip install termcolor easydict einops addict yapf tensorboardX peft timm
else
    pip install termcolor easydict einops addict yapf tensorboardX peft timm
fi
print_success "Additional tools installed"

# Install Open3D for visualization (used by PointCept visualization utilities)
print_step "Installing Open3D for point cloud visualization..."
if [ "$PKG_MANAGER" = "uv" ]; then
    uv pip install open3d
else
    pip install open3d
fi
print_success "Open3D installed"

################################################################################
# Step 7: Setup Environment and Install Pointcept
################################################################################

print_header "Step 8/$TOTAL_STEPS: Setting Up Pointcept"

cd "$SCRIPT_DIR"

print_step "Setting PYTHONPATH..."
export PYTHONPATH="$SCRIPT_DIR/pointcept:$PYTHONPATH"
print_success "PYTHONPATH configured"

# Create activate.sh if it doesn't exist
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate.sh"
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    print_step "Creating activate.sh..."
    cat > "$ACTIVATE_SCRIPT" << ENVEOF
#!/bin/bash
################################################################################
# PTv3 LiDAR Subsampling Benchmark - Environment Activation Script
################################################################################

# Get the script directory
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="\$(dirname "\$SCRIPT_DIR")"
VENV_PATH="\$BENCHMARK_ROOT/ptv3_venv"

# Activate virtual environment
if [ -f "\$VENV_PATH/bin/activate" ]; then
    source "\$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found at \$VENV_PATH"
    echo "  Run ./setup_venv.sh first"
    return 1
fi

# Cache directory configuration (keeps all cache on disk, not home)
export XDG_CACHE_HOME="\$VENV_PATH/cache"
export TORCH_HOME="\$VENV_PATH/cache/torch"
export HF_HOME="\$VENV_PATH/cache/huggingface"
export TORCH_EXTENSIONS_DIR="\$VENV_PATH/cache/torch_extensions"
export CUDA_CACHE_PATH="\$VENV_PATH/cache/cuda"
echo "✓ Cache directories configured (not in home)"

# Set PYTHONPATH
export PYTHONPATH="\$SCRIPT_DIR/pointcept:\$PYTHONPATH"
echo "✓ PYTHONPATH configured"

# Display environment info
echo ""
echo "PTv3 LiDAR Subsampling Benchmark Environment"
echo "============================================="
echo "Python:   \$(python --version 2>&1)"
echo "PyTorch:  \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA:     \$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "Venv:     \$VENV_PATH"
echo "Cache:    \$VENV_PATH/cache"
echo ""
ENVEOF
    chmod +x "$ACTIVATE_SCRIPT"
    print_success "activate.sh created at $ACTIVATE_SCRIPT"
fi

################################################################################
# Step 8: Compile CUDA Extensions
################################################################################

print_header "Step 9/$TOTAL_STEPS: Compiling CUDA Extensions"

if [ "$SKIP_COMPILE" = true ]; then
    print_warning "Skipping CUDA extension compilation (--skip-compile flag)"
elif [ "$PYTORCH_CUDA" = "cpu" ]; then
    print_warning "Skipping CUDA extensions (CPU-only installation)"
else
    print_step "Compiling pointops extension..."
    print_info "This may take 30-60 seconds..."

    # Ensure setuptools is available (required for pointops build)
    print_info "Ensuring build dependencies are available..."
    if [ "$PKG_MANAGER" = "uv" ]; then
        uv pip install setuptools wheel --quiet
    else
        "$VENV_PATH/bin/pip" install setuptools wheel --quiet
    fi

    cd "$SCRIPT_DIR/pointcept/libs/pointops"

    # Use venv pip explicitly to avoid using system pip
    VENV_PIP="$VENV_PATH/bin/pip"

    # Use regular install instead of editable (-e) to avoid setuptools build_editable issues
    if [ "$PKG_MANAGER" = "uv" ]; then
        if uv pip install --no-build-isolation .; then
            print_success "pointops compiled successfully"
        else
            # Fallback: try with venv pip directly (NOT system pip)
            print_warning "uv install failed, trying venv pip..."
            if "$VENV_PIP" install --no-build-isolation .; then
                print_success "pointops compiled successfully (via pip)"
            else
                print_error "Failed to compile pointops"
                print_warning "You may need to compile it manually later"
                print_info "cd $SCRIPT_DIR/pointcept/libs/pointops && $VENV_PIP install --no-build-isolation ."
            fi
        fi
    else
        if "$VENV_PIP" install --no-build-isolation .; then
            print_success "pointops compiled successfully"
        else
            print_error "Failed to compile pointops"
            print_warning "You may need to compile it manually later"
            print_info "cd $SCRIPT_DIR/pointcept/libs/pointops && $VENV_PIP install --no-build-isolation ."
        fi
    fi
fi

################################################################################
# Step 9: Register DALES Dataset
################################################################################

print_header "Step 10/$TOTAL_STEPS: Registering DALES Dataset"

cd "$SCRIPT_DIR/pointcept"

# Create symlink for DALES dataset
DATASET_SYMLINK="pointcept/datasets/dales.py"
DATASET_SOURCE="$SCRIPT_DIR/DALES/dales_dataset.py"

if [ ! -f "$DATASET_SOURCE" ]; then
    print_warning "DALES dataset source not found at:"
    print_warning "  $DATASET_SOURCE"
    print_info "DALES dataset will need to be registered manually"
    print_info "Expected location: $SCRIPT_DIR/DALES/dales_dataset.py"
else
    if [ -L "$DATASET_SYMLINK" ] || [ -f "$DATASET_SYMLINK" ]; then
        print_info "DALES dataset file already exists"
    else
        print_step "Creating DALES dataset symlink..."
        ln -sf "$DATASET_SOURCE" "$DATASET_SYMLINK"
        print_success "Symlink created"
    fi

    # Check if dataset is imported in __init__.py
    INIT_FILE="pointcept/datasets/__init__.py"
    if grep -q "from .dales import DALESDataset" "$INIT_FILE"; then
        print_success "DALES dataset already registered in __init__.py"
    else
        print_step "Adding DALES import to __init__.py..."

        # Find appropriate line to insert (after other outdoor datasets)
        LINE_NUM=$(grep -n "from .waymo import WaymoDataset" "$INIT_FILE" | cut -d: -f1)

        if [ -n "$LINE_NUM" ]; then
            # Insert after WaymoDataset line
            sed -i "${LINE_NUM}a from .dales import DALESDataset" "$INIT_FILE"
            print_success "DALES dataset registered in __init__.py"
        else
            print_warning "Could not auto-register DALES dataset in __init__.py"
            print_info "Please add manually: from .dales import DALESDataset"
        fi
    fi
fi

################################################################################
# Verification
################################################################################

print_header "Verification"

cd "$SCRIPT_DIR"

print_step "Testing imports..."

# Test 1: Import torch
if python -c "import torch; print('✓ PyTorch')" 2>/dev/null; then
    print_success "PyTorch imports successfully"
else
    print_error "Failed to import PyTorch"
fi

# Test 2: Import scipy (for FPS/DBSCAN)
if python -c "import scipy; from scipy.spatial.distance import pdist; print('✓ SciPy')" 2>/dev/null; then
    print_success "SciPy imports successfully"
else
    print_error "Failed to import SciPy"
fi

# Test 3: Import pointcept
if python -c "from pointcept.datasets import SemanticKITTIDataset; print('✓ SemanticKITTI')" 2>/dev/null; then
    print_success "SemanticKITTI dataset imports successfully"
else
    print_error "Failed to import SemanticKITTI dataset"
fi

# Test 4: Import DALES (if registered)
if python -c "from pointcept.datasets import DALESDataset; print('✓ DALES')" 2>/dev/null; then
    print_success "DALES dataset imports successfully"
else
    print_warning "DALES dataset not yet registered (expected if source not available)"
fi

# Test 5: Check Flash Attention (if CUDA)
if [ "$PYTORCH_CUDA" != "cpu" ]; then
    if python -c "import flash_attn; print('✓ Flash Attention')" 2>/dev/null; then
        print_success "Flash Attention available"
    else
        print_warning "Flash Attention not available (configs with enable_flash=True will fail)"
    fi
fi

################################################################################
# Summary
################################################################################

print_header "Setup Complete!"

echo -e "\n${GREEN}✓ Environment is ready for benchmarking!${NC}\n"

echo "Quick Start:"
echo "  1. Activate environment:"
echo "     ${BLUE}source $SCRIPT_DIR/activate.sh${NC}"
echo ""
echo "  2. Test subsampling methods:"
echo "     ${BLUE}cd $BENCHMARK_ROOT/src/subsampling${NC}"
echo "     ${BLUE}python fps.py${NC}                    # Test FPS"
echo "     ${BLUE}python dbscan.py${NC}                 # Test DBSCAN"
echo ""
echo "  3. Generate subsampled datasets:"
echo "     ${BLUE}cd $SCRIPT_DIR/SemanticKITTI/scripts${NC}"
echo "     ${BLUE}python generate_subsampled.py${NC}    # Generate all combinations"
echo ""
echo "  4. Start training:"
echo "     ${BLUE}cd $SCRIPT_DIR/SemanticKITTI/scripts${NC}"
echo "     ${BLUE}./train_experiments.sh${NC}           # Run experiments"
echo ""

echo "Datasets:"
echo "  - SemanticKITTI: $BENCHMARK_ROOT/data/SemanticKITTI/"
echo "  - DALES:         $BENCHMARK_ROOT/data/DALES/"
echo ""

echo "Configurations:"
echo "  - SemanticKITTI: $SCRIPT_DIR/SemanticKITTI/configs/semantickitti/generated/ (18 configs)"
echo "  - DALES:         $SCRIPT_DIR/DALES/configs/dales/generated/ (18 configs)"
echo ""

echo "Documentation:"
echo "  - $BENCHMARK_ROOT/docs/ACTION_PLAN.md"
echo "  - $BENCHMARK_ROOT/docs/PTv3m1_CONFIG_VERIFICATION_SUMMARY.md"
echo "  - $SCRIPT_DIR/SemanticKITTI/README.md"
echo "  - $SCRIPT_DIR/DALES/README.md"
echo ""

echo "Environment Details:"
echo "  Python:      $(python --version)"
echo "  PyTorch:     $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not loaded")"
echo "  CUDA:        $PYTORCH_CUDA"
echo "  Venv:        $VENV_PATH"
echo "  Cache:       $VENV_PATH/cache"
echo ""

if [ "$PYTORCH_CUDA" != "cpu" ] && [ "$SKIP_COMPILE" = false ]; then
    print_info "CUDA extensions compiled - GPU training ready!"
elif [ "$PYTORCH_CUDA" = "cpu" ]; then
    print_warning "CPU-only installation - GPU training not available"
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Setup completed successfully! 🚀${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
