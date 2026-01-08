#!/bin/bash
#
# Generate All Figures Pipeline
#
# This script runs the complete pipeline to extract metrics and generate all figures:
# 1. Extract training metrics from experiment outputs
# 2. Extract inference metrics from inference outputs
# 3. Generate main figures (metric grouped, spatial distribution, ranking)
# 4. Generate class-wise performance drop figures
# 5. Generate class-wise detailed figures
#
# Optional (with --with-pointcloud flag):
# 6. Find good scans for point cloud visualization
# 7. Generate point cloud comparison figures (requires xvfb-run)
#
# Usage:
#   ./generate_all.sh                  # Generate all figures (excluding point cloud)
#   ./generate_all.sh --with-pointcloud  # Also generate point cloud visualizations
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Python interpreter from ptv3_venv
PYTHON="$PROJECT_ROOT/ptv3_venv/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Please ensure ptv3_venv is set up correctly."
    exit 1
fi

echo "=============================================="
echo "Generate All Figures Pipeline"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON"
echo ""

# Step 1: Extract training metrics
echo "----------------------------------------------"
echo "Step 1: Extracting training metrics..."
echo "----------------------------------------------"
$PYTHON "$PROJECT_ROOT/scripts/extract_training_metrics.py"
echo "Done."
echo ""

# Step 2: Extract inference metrics
echo "----------------------------------------------"
echo "Step 2: Extracting inference metrics..."
echo "----------------------------------------------"
$PYTHON "$PROJECT_ROOT/scripts/extract_inference_metrics.py"
echo "Done."
echo ""

# Step 3: Generate main figures
echo "----------------------------------------------"
echo "Step 3: Generating main figures..."
echo "----------------------------------------------"
$PYTHON "$SCRIPT_DIR/generate_figures.py"
echo "Done."
echo ""

# Step 4: Generate class-wise performance drop figures
echo "----------------------------------------------"
echo "Step 4: Generating class-wise performance drop figures..."
echo "----------------------------------------------"
$PYTHON "$SCRIPT_DIR/generate_classwise_performance_drop.py"
echo "Done."
echo ""

# Step 5: Generate class-wise detailed figures
echo "----------------------------------------------"
echo "Step 5: Generating class-wise detailed figures..."
echo "----------------------------------------------"
$PYTHON "$SCRIPT_DIR/generate_classwise_figures.py"
echo "Done."
echo ""

echo "=============================================="
echo "All figures generated successfully!"
echo "=============================================="
echo "Output directory: $PROJECT_ROOT/docs/figures"
echo ""

# Optional: Point cloud visualization (requires xvfb-run)
if [ "$1" = "--with-pointcloud" ]; then
    echo "----------------------------------------------"
    echo "Step 6 (Optional): Finding good scans for visualization..."
    echo "----------------------------------------------"
    if [ -f "$SCRIPT_DIR/good_scans.txt" ]; then
        echo "good_scans.txt already exists. Skipping scan search."
    else
        $PYTHON "$SCRIPT_DIR/find_good_scans.py"
    fi
    echo "Done."
    echo ""

    echo "----------------------------------------------"
    echo "Step 7 (Optional): Generating point cloud comparisons..."
    echo "----------------------------------------------"
    if command -v xvfb-run &> /dev/null; then
        xvfb-run -a $PYTHON "$SCRIPT_DIR/generate_pointcloud_comparison_o3d.py"
        echo "Done."
    else
        echo "Warning: xvfb-run not found. Skipping point cloud visualization."
        echo "Install with: sudo apt-get install xvfb"
    fi
    echo ""
fi

echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo ""
echo "To also generate point cloud visualizations, run:"
echo "  $0 --with-pointcloud"
echo ""
