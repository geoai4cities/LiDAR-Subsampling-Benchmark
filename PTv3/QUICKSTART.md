# PTv3 Environment Setup - Quick Start Guide

**Project:** LiDAR Subsampling Benchmark
**Last Updated:** November 24, 2025

---

## Prerequisites

✅ **Python 3.9-3.11** (3.11 recommended)
✅ **CUDA Toolkit** (11.8 or 12.x, optional for GPU)
✅ **~10-15 GB disk space** (for venv + cache)
✅ **Pointcept** repository cloned

---

## One-Time Setup

### Step 1: Clone Pointcept (if not done)

```bash
cd PTv3
git clone https://github.com/Pointcept/Pointcept.git pointcept
```

### Step 2: Run Setup Script

```bash
# Auto-detect CUDA and setup everything
./setup_venv.sh

# Or specify CUDA version
./setup_venv.sh --cuda-version 11.8    # For CUDA 11.x
./setup_venv.sh --cuda-version 12.x    # For CUDA 12.x
./setup_venv.sh --cuda-version cpu     # CPU-only

# Fast setup (skip CUDA extensions)
./setup_venv.sh --skip-compile
```

**Time:** 30-45 minutes (includes Flash Attention compilation)

---

## Daily Usage

### Activate Environment

```bash
cd PTv3
source activate.sh
```

**You'll see:**
```
✓ Virtual environment activated
✓ Cache directories configured (not in home)
✓ PYTHONPATH configured

PTv3 LiDAR Subsampling Benchmark Environment
=============================================
Python:   Python 3.11.x
PyTorch:  2.5.0+cu118
CUDA:     11.8
Venv:     <project_root>/ptv3_venv
Cache:    <project_root>/ptv3_venv/cache
```

### Verify Installation

```bash
# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Flash Attention
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Test datasets
python -c "from pointcept.datasets import SemanticKITTIDataset, DALESDataset; print('✓ Datasets OK')"
```

---

## Test Subsampling Methods

```bash
cd src/subsampling

# Test all methods
python random_sampling.py
python idis.py
python fps.py
python dbscan.py
python voxel_grid.py
python poisson_disk.py
```

**Expected output:** "ALL TESTS PASSED! ✓"

---

## Generate Subsampled Datasets

### Test Mode (Quick)

```bash
# From project root

# SemanticKITTI (10 scans, ~5 minutes)
python scripts/preprocessing/generate_subsampled_semantickitti.py --test

# DALES (3 tiles, ~2 minutes)
python scripts/preprocessing/generate_subsampled_dales.py --test
```

### Full Generation (Production)

```bash
# SemanticKITTI (~486,000 files, 1-2 weeks with 4 workers)
nohup python scripts/preprocessing/generate_subsampled_semantickitti.py --workers 4 > semantickitti.log 2>&1 &

# DALES (~4,320 files, 0.5-2 days with 4 workers)
nohup python scripts/preprocessing/generate_subsampled_dales.py --workers 4 > dales.log 2>&1 &

# Monitor progress
tail -f semantickitti.log
```

---

## Train PTv3 Models

### Test Training (5 epochs)

```bash
cd PTv3

python pointcept/tools/train.py \
  --config SemanticKITTI/configs/semantickitti/generated/ptv3_semantickitti_RS_loss0_seed1.py \
  --options epoch=5
```

### Full Training (50 epochs)

```bash
# Single experiment
python pointcept/tools/train.py \
  --config SemanticKITTI/configs/semantickitti/generated/ptv3_semantickitti_IDIS_loss50_seed1.py
```

---

## Experiment Configurations

### Available Configs

**SemanticKITTI:** 18 configs in `PTv3/SemanticKITTI/configs/semantickitti/generated/`
- Methods: RS, IDIS
- Loss levels: 0%, 50%, 90%
- Seeds: 1, 2, 3

**DALES:** 18 configs in `PTv3/DALES/configs/dales/generated/`
- Methods: RS, IDIS
- Loss levels: 0%, 50%, 90%
- Seeds: 1, 2, 3

**Total:** 36 configs (Tier 1: RS + IDIS)

### List All Configs

```bash
ls PTv3/SemanticKITTI/configs/semantickitti/generated/
ls PTv3/DALES/configs/dales/generated/
```

---

## Directory Structure

```
LiDAR-Subsampling-Benchmark/
├── ptv3_venv/                      # Virtual environment (external)
│   ├── bin/activate
│   └── cache/                      # All cache here (not home)
│       ├── torch/
│       ├── huggingface/
│       ├── torch_extensions/
│       └── cuda/
│
├── PTv3/
│   ├── setup_venv.sh              # Setup script
│   ├── activate.sh                # Activation script
│   ├── pyproject.toml             # Project configuration
│   ├── pointcept/                 # Pointcept framework
│   ├── SemanticKITTI/
│   │   └── configs/semantickitti/generated/  # 18 configs
│   └── DALES/
│       └── configs/dales/generated/          # 18 configs
│
├── src/
│   └── subsampling/               # 6 subsampling methods
│       ├── random_sampling.py
│       ├── idis.py
│       ├── fps.py
│       ├── dbscan.py
│       ├── voxel_grid.py
│       └── poisson_disk.py
│
├── scripts/
│   ├── convert_ply_to_txt.py     # DALES PLY converter
│   ├── generate_subsampled_semantickitti.py
│   └── generate_subsampled_dales.py
│
├── data/
│   ├── SemanticKITTI/
│   │   ├── original/sequences/00/
│   │   └── subsampled/            # Generated datasets
│   └── DALES/
│       ├── original/              # train/ and test/
│       └── subsampled/            # Generated datasets
│
└── docs/
    ├── DAY1_COMPLETION_SUMMARY.md
    ├── DAY2_COMPLETION_SUMMARY.md
    ├── ACTION_PLAN.md
    └── SETUP_SCRIPT_FIXES.md
```

---

## Common Commands

### Environment Management

```bash
# Activate
source PTv3/activate.sh

# Deactivate
deactivate

# Recreate environment
rm -rf ptv3_venv
./PTv3/setup_venv.sh
```

### Check Installation

```bash
# Python packages
pip list | grep torch
pip list | grep flash

# CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# GPU info
nvidia-smi
```

### Monitor GPU

```bash
# Watch GPU usage (update every 1 second)
watch -n 1 nvidia-smi

# Or use GPUstat
pip install gpustat
gpustat -i 1
```

---

## Troubleshooting

### Flash Attention Failed

**Symptom:** Flash Attention compilation fails during setup

**Solution:**
```bash
# Skip for now
./setup_venv.sh --skip-compile

# Install later manually
source activate.sh
pip install flash-attn --no-build-isolation -v
```

### CUDA Not Detected

**Symptom:** Setup detects CPU-only despite having CUDA

**Solution:**
```bash
# Check CUDA
nvcc --version

# Force CUDA version
./setup_venv.sh --cuda-version 11.8
```

### Import Errors

**Symptom:** "ModuleNotFoundError" when importing

**Solution:**
```bash
# Make sure environment is activated
source PTv3/activate.sh

# Check PYTHONPATH
echo $PYTHONPATH  # Should include pointcept

# Reinstall if needed
cd PTv3/pointcept/libs/pointops
pip install -e .
```

### Disk Space Issues

**Symptom:** Setup fails with "No space left on device"

**Solution:**
```bash
# Check space
df -h

# Clean pip cache
pip cache purge

# Clean conda cache (if using conda)
conda clean --all
```

---

## Getting Help

### Check Documentation

```bash
# Setup script help
./PTv3/setup_venv.sh --help

# Data generation help
python scripts/preprocessing/generate_subsampled_semantickitti.py --help
python scripts/preprocessing/generate_subsampled_dales.py --help

# List subsampling methods
cd src && python -c "from subsampling import list_available_methods; list_available_methods()"
```

### View Logs

```bash
# Training logs
ls PTv3/SemanticKITTI/outputs/
ls PTv3/DALES/outputs/

# Data generation logs
tail -f semantickitti.log
tail -f dales.log
```

---

## Summary

### What You Get

✅ **6 Subsampling Methods:** RS, IDIS, FPS, DBSCAN, Voxel, Poisson
✅ **2 Datasets:** SemanticKITTI, DALES
✅ **36 PTv3 Configs:** Ready for experiments
✅ **Automated Scripts:** Data generation and training
✅ **External Venv:** Clean, isolated environment
✅ **Cache Management:** All cache on disk, not home

### Quick Commands

```bash
# Setup (one-time)
./PTv3/setup_venv.sh

# Activate (daily)
source PTv3/activate.sh

# Test (quick verification)
python scripts/generate_subsampled_semantickitti.py --test

# Generate (full datasets)
python scripts/generate_subsampled_semantickitti.py --workers 4

# Train (start experiments)
python PTv3/PointTransformerV3/Pointcept/tools/train.py --config <config.py>
```

---

**Ready to start?** Run `./PTv3/setup_venv.sh` now! 🚀
