#!/bin/bash
################################################################################
# Test Flash Attention runtime in both environments
################################################################################

TEST_SCRIPT='
import torch
import torch.nn as nn
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test basic CUDA
print("\n=== Testing basic CUDA ===")
x = torch.randn(100, 100, device="cuda")
y = torch.matmul(x, x)
print(f"Basic CUDA matmul: OK (result shape: {y.shape})")

# Test Flash Attention import
print("\n=== Testing Flash Attention import ===")
try:
    from flash_attn import flash_attn_func
    print("flash_attn_func imported: OK")
except Exception as e:
    print(f"Import failed: {e}")
    exit(1)

# Test Flash Attention execution
print("\n=== Testing Flash Attention execution ===")
try:
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

    output = flash_attn_func(q, k, v)
    print(f"Flash Attention output shape: {output.shape}")
    print("Flash Attention execution: OK")
except Exception as e:
    print(f"Flash Attention execution FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== ALL TESTS PASSED ===")
'

echo "========================================"
echo "=== Testing BhopalMLS venv ==="
echo "========================================"
source /home/vaibhavk/vaibhav/pyare/Bhopal/Benchmarking/PTv3/.venv/bin/activate
python -c "$TEST_SCRIPT"
RESULT1=$?
deactivate 2>/dev/null

echo ""
echo "========================================"
echo "=== Testing ptv3_venv ==="
echo "========================================"
source /home/vaibhavk/vaibhav/pyare/LiDAR-Subsampling-Benchmark/ptv3_venv/bin/activate
python -c "$TEST_SCRIPT"
RESULT2=$?
deactivate 2>/dev/null

echo ""
echo "========================================"
echo "=== Summary ==="
echo "========================================"
if [ $RESULT1 -eq 0 ]; then
    echo "BhopalMLS venv: PASSED"
else
    echo "BhopalMLS venv: FAILED"
fi

if [ $RESULT2 -eq 0 ]; then
    echo "ptv3_venv: PASSED"
else
    echo "ptv3_venv: FAILED"
fi
