#!/bin/bash
################################################################################
# Comprehensive Debug Script: Compare BhopalMLS venv vs ptv3_venv
# Find the root cause of "Floating point exception" crash
################################################################################

BHOPAL_VENV="/home/vaibhavk/vaibhav/pyare/Bhopal/Benchmarking/PTv3/.venv"
PTV3_VENV="/home/vaibhavk/vaibhav/pyare/LiDAR-Subsampling-Benchmark/ptv3_venv"
OUTPUT_DIR="/tmp/venv_debug"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "=== Comprehensive venv Debug Script ==="
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

################################################################################
# 1. Basic Environment Info
################################################################################
echo ">>> 1. Basic Environment Info"

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    echo "--- $VENV_NAME ---"
    source "$VENV_PATH/bin/activate"

    {
        echo "=== $VENV_NAME Environment ==="
        echo "Python: $(python --version 2>&1)"
        echo "Python path: $(which python)"
        echo "Pip path: $(which pip)"
        echo ""

        echo "=== PyTorch ==="
        python -c "
import torch
print(f'Version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA arch list: {torch.cuda.get_arch_list()}')
print(f'Torch file: {torch.__file__}')
"
        echo ""

        echo "=== Flash Attention ==="
        python -c "
import flash_attn
print(f'Version: {flash_attn.__version__}')
print(f'File: {flash_attn.__file__}')
# Check compiled extensions
import os
fa_dir = os.path.dirname(flash_attn.__file__)
for f in os.listdir(fa_dir):
    if f.endswith('.so'):
        print(f'Extension: {f}')
" 2>/dev/null || echo "Flash Attention not available"
        echo ""

        echo "=== Triton ==="
        python -c "
import triton
print(f'Version: {triton.__version__}')
print(f'File: {triton.__file__}')
" 2>/dev/null || echo "Triton not available"
        echo ""

        echo "=== pointops ==="
        python -c "
import pointops
print(f'File: {pointops.__file__}')
import os
po_dir = os.path.dirname(pointops.__file__)
for f in os.listdir(po_dir):
    if f.endswith('.so'):
        print(f'Extension: {f}')
" 2>/dev/null || echo "pointops not available"
        echo ""

        echo "=== spconv ==="
        python -c "
import spconv
print(f'Version: {spconv.__version__}')
print(f'File: {spconv.__file__}')
" 2>/dev/null || echo "spconv not available"
        echo ""

    } > "$OUTPUT_DIR/${VENV_NAME}_info.txt" 2>&1

    deactivate 2>/dev/null
done

echo "Saved to: $OUTPUT_DIR/bhopal_info.txt and $OUTPUT_DIR/ptv3_info.txt"
echo ""

################################################################################
# 2. Compare .so files (compiled extensions)
################################################################################
echo ">>> 2. Compiled Extensions (.so files)"

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    echo "--- $VENV_NAME .so files ---"
    find "$VENV_PATH/lib" -name "*.so" -type f 2>/dev/null | while read f; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "$(basename $f) : $SIZE bytes"
    done | sort > "$OUTPUT_DIR/${VENV_NAME}_so_files.txt"
done

echo "Comparing .so file sizes..."
diff "$OUTPUT_DIR/bhopal_so_files.txt" "$OUTPUT_DIR/ptv3_so_files.txt" > "$OUTPUT_DIR/so_diff.txt"
if [ -s "$OUTPUT_DIR/so_diff.txt" ]; then
    echo "DIFFERENCES FOUND:"
    cat "$OUTPUT_DIR/so_diff.txt" | head -30
else
    echo "No differences in .so files"
fi
echo ""

################################################################################
# 3. Check CUDA-related environment variables
################################################################################
echo ">>> 3. CUDA Environment Variables"

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    source "$VENV_PATH/bin/activate"

    {
        echo "=== $VENV_NAME CUDA Environment ==="
        echo "CUDA_HOME: ${CUDA_HOME:-not set}"
        echo "CUDA_PATH: ${CUDA_PATH:-not set}"
        echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
        echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-not set}"
        echo ""

        echo "=== PyTorch CUDA info ==="
        python -c "
import torch
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Total memory: {props.total_memory / 1024**3:.1f} GB')
"
    } > "$OUTPUT_DIR/${VENV_NAME}_cuda_env.txt" 2>&1

    deactivate 2>/dev/null
done

echo "Comparing CUDA environments..."
diff "$OUTPUT_DIR/bhopal_cuda_env.txt" "$OUTPUT_DIR/ptv3_cuda_env.txt"
echo ""

################################################################################
# 4. Check flash_attn compiled extensions in detail
################################################################################
echo ">>> 4. Flash Attention Extension Details"

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    echo "--- $VENV_NAME flash_attn ---"
    FA_DIR="$VENV_PATH/lib/python3.9/site-packages/flash_attn"
    if [ -d "$FA_DIR" ]; then
        ls -la "$FA_DIR"/*.so 2>/dev/null | awk '{print $5, $9}' || echo "No .so files"

        # Check symbols in the .so file
        echo "Checking symbols..."
        for so in "$FA_DIR"/*.so; do
            if [ -f "$so" ]; then
                echo "File: $(basename $so)"
                nm -D "$so" 2>/dev/null | grep -i cuda | head -5 || echo "  No CUDA symbols found"
            fi
        done
    else
        echo "flash_attn directory not found"
    fi
    echo ""
done > "$OUTPUT_DIR/flash_attn_details.txt" 2>&1
cat "$OUTPUT_DIR/flash_attn_details.txt"

################################################################################
# 5. Check pointops compiled extensions in detail
################################################################################
echo ">>> 5. pointops Extension Details"

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    echo "--- $VENV_NAME pointops ---"
    PO_DIR="$VENV_PATH/lib/python3.9/site-packages/pointops"
    if [ -d "$PO_DIR" ]; then
        ls -la "$PO_DIR"/*.so 2>/dev/null | awk '{print $5, $9}' || echo "No .so files"
    else
        echo "pointops directory not found at $PO_DIR"
        # Try to find it
        find "$VENV_PATH" -name "pointops*" -type d 2>/dev/null | head -3
    fi
    echo ""
done

################################################################################
# 6. Test specific operations that might cause FPE
################################################################################
echo ">>> 6. Testing Operations That Might Cause FPE"

TEST_SCRIPT='
import torch
import sys

print(f"Testing with PyTorch {torch.__version__}")

# Test 1: Basic operations
print("\n[Test 1] Basic CUDA operations...")
try:
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print(f"  MatMul: OK")
except Exception as e:
    print(f"  MatMul FAILED: {e}")

# Test 2: Division (common FPE cause)
print("\n[Test 2] Division operations...")
try:
    a = torch.randn(100, device="cuda")
    b = torch.zeros(100, device="cuda")
    c = a / (b + 1e-8)  # Safe division
    print(f"  Safe division: OK")

    # Unsafe division (might cause FPE on some systems)
    # d = a / b  # Uncomment to test
except Exception as e:
    print(f"  Division FAILED: {e}")

# Test 3: Flash Attention with edge cases
print("\n[Test 3] Flash Attention edge cases...")
try:
    from flash_attn import flash_attn_func

    # Normal case
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    print(f"  Normal case: OK")

    # Larger batch (like training)
    q = torch.randn(24, 1024, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(24, 1024, 8, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(24, 1024, 8, 64, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    print(f"  Large batch: OK")

except Exception as e:
    print(f"  Flash Attention FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: pointops operations
print("\n[Test 4] pointops operations...")
try:
    import pointops

    # Test basic pointops function
    xyz = torch.randn(2, 10000, 3, device="cuda")
    # If there are any test functions available
    print(f"  pointops import: OK")
    print(f"  Available functions: {[x for x in dir(pointops) if not x.startswith(\"_\")][:10]}")
except Exception as e:
    print(f"  pointops FAILED: {e}")

# Test 5: Memory pressure test
print("\n[Test 5] Memory pressure test...")
try:
    tensors = []
    for i in range(10):
        t = torch.randn(1000, 1000, device="cuda")
        tensors.append(t)
    del tensors
    torch.cuda.empty_cache()
    print(f"  Memory allocation: OK")
except Exception as e:
    print(f"  Memory FAILED: {e}")

# Test 6: bfloat16 operations (used in training config)
print("\n[Test 6] bfloat16 operations...")
try:
    x = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
    y = torch.matmul(x, x)
    print(f"  bfloat16 MatMul: OK")
except Exception as e:
    print(f"  bfloat16 FAILED: {e}")

print("\n=== All tests completed ===")
'

for VENV_NAME in "bhopal" "ptv3"; do
    if [ "$VENV_NAME" = "bhopal" ]; then
        VENV_PATH="$BHOPAL_VENV"
    else
        VENV_PATH="$PTV3_VENV"
    fi

    echo ""
    echo "========================================"
    echo "=== Testing $VENV_NAME ==="
    echo "========================================"
    source "$VENV_PATH/bin/activate"
    python -c "$TEST_SCRIPT" 2>&1 | tee "$OUTPUT_DIR/${VENV_NAME}_tests.txt"
    deactivate 2>/dev/null
done

################################################################################
# 7. Check activation scripts for differences
################################################################################
echo ""
echo ">>> 7. Activation Script Differences"

diff "$BHOPAL_VENV/bin/activate" "$PTV3_VENV/bin/activate" > "$OUTPUT_DIR/activate_diff.txt" 2>&1
if [ -s "$OUTPUT_DIR/activate_diff.txt" ]; then
    echo "Differences in activate scripts:"
    head -20 "$OUTPUT_DIR/activate_diff.txt"
else
    echo "No differences in activate scripts"
fi

################################################################################
# 8. Check site-packages structure
################################################################################
echo ""
echo ">>> 8. Site-packages Structure"

echo "--- BhopalMLS site-packages ---"
ls "$BHOPAL_VENV/lib/python3.9/site-packages/" | grep -E "flash|point|torch|spconv|triton" | sort > "$OUTPUT_DIR/bhopal_packages.txt"
cat "$OUTPUT_DIR/bhopal_packages.txt"

echo ""
echo "--- ptv3_venv site-packages ---"
ls "$PTV3_VENV/lib/python3.9/site-packages/" | grep -E "flash|point|torch|spconv|triton" | sort > "$OUTPUT_DIR/ptv3_packages.txt"
cat "$OUTPUT_DIR/ptv3_packages.txt"

echo ""
echo "Differences:"
diff "$OUTPUT_DIR/bhopal_packages.txt" "$OUTPUT_DIR/ptv3_packages.txt"

################################################################################
# 9. Summary
################################################################################
echo ""
echo "========================================"
echo "=== Debug Summary ==="
echo "========================================"
echo "All output saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Key files to check:"
echo "  - $OUTPUT_DIR/so_diff.txt (compiled extension differences)"
echo "  - $OUTPUT_DIR/bhopal_tests.txt (test results)"
echo "  - $OUTPUT_DIR/ptv3_tests.txt (test results)"
echo ""
echo "To view full comparison:"
echo "  diff $OUTPUT_DIR/bhopal_info.txt $OUTPUT_DIR/ptv3_info.txt"
