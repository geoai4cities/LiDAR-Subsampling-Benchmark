#!/bin/bash
################################################################################
# Compare BhopalMLS venv vs ptv3_venv to debug Flash Attention issue
################################################################################

echo "========================================"
echo "=== BhopalMLS venv (WORKING) ==="
echo "========================================"
source /home/vaibhavk/vaibhav/pyare/Bhopal/Benchmarking/PTv3/.venv/bin/activate

echo "Python: $(python --version 2>&1)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA built with: {torch.version.cuda}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention: NOT INSTALLED"
python -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || echo "Triton: NOT INSTALLED"

echo ""
echo "Key packages:"
pip list 2>/dev/null | grep -E "^torch |^flash-attn|^nvidia|^triton|^setuptools|^wheel|^packaging|^ninja"

echo ""
echo "Flash-attn location:"
pip show flash-attn 2>/dev/null | grep -E "Version|Location|Requires" || echo "Not found"

deactivate 2>/dev/null

echo ""
echo "========================================"
echo "=== ptv3_venv (BROKEN) ==="
echo "========================================"
source /home/vaibhavk/vaibhav/pyare/LiDAR-Subsampling-Benchmark/ptv3_venv/bin/activate

echo "Python: $(python --version 2>&1)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA built with: {torch.version.cuda}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention: NOT INSTALLED"
python -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || echo "Triton: NOT INSTALLED"

echo ""
echo "Key packages:"
pip list 2>/dev/null | grep -E "^torch |^flash-attn|^nvidia|^triton|^setuptools|^wheel|^packaging|^ninja"

echo ""
echo "Flash-attn location:"
pip show flash-attn 2>/dev/null | grep -E "Version|Location|Requires" || echo "Not found"

deactivate 2>/dev/null

echo ""
echo "========================================"
echo "=== System Info ==="
echo "========================================"
echo "CUDA (nvcc): $(nvcc --version 2>&1 | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CPU Cores: $(nproc)"
