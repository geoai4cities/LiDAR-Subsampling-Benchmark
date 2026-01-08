#!/bin/bash
################################################################################
# Test Data Generation Scripts
################################################################################

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  LiDAR Subsampling Benchmark - Test Data Generation       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate environment
cd "$PROJECT_ROOT/PTv3"
echo "▶ Activating environment..."
if ! source activate.sh; then
    echo "✗ Failed to activate environment"
    echo "  Please run PTv3/setup_venv.sh first"
    exit 1
fi
cd "$PROJECT_ROOT"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Test 1: SemanticKITTI Data Generation (Quick Test)"
echo "═══════════════════════════════════════════════════════════"
echo "Expected time: ~5 minutes"
echo "Processing: 10 scans with 6 methods"
echo ""

python scripts/preprocessing/generate_subsampled_semantickitti.py --test

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Test 2: DALES Data Generation (Quick Test)"
echo "═══════════════════════════════════════════════════════════"
echo "Expected time: ~2 minutes"
echo "Processing: 3 tiles with 6 methods"
echo ""

python scripts/preprocessing/generate_subsampled_dales.py --test

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All Tests Complete!                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
