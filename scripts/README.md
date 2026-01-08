# Scripts Directory

**Purpose:** Scripts for subsampling data generation, metrics extraction, and figure generation.

---

## Directory Structure

```
scripts/
├── README.md                              # This file
│
├── preprocessing/                          # Data preprocessing & subsampling
│   ├── generate_subsampled_semantickitti.py    # SemanticKITTI subsampling (CPU methods)
│   ├── generate_subsampled_semantickitti_v2.py # SemanticKITTI v2 with improvements
│   ├── generate_subsampled_gpu.py              # GPU-based subsampling (IDIS, FPS)
│   ├── generate_subsampled_dales.py            # DALES subsampling (CPU methods)
│   ├── generate_subsampled_dales_gpu.py        # DALES GPU subsampling
│   ├── generate_subsampled_depoco.py           # DEPOCO neural compression
│   ├── generate_idis_r_ablation.py             # IDIS R-value ablation study
│   ├── verify_subsampling.py                   # Verify subsampling loss percentages
│   ├── preprocess_semantickitti_for_depoco.py  # DEPOCO preprocessing
│   ├── convert_ply_to_txt.py                   # PLY to TXT converter
│   ├── restructure_subsampled_semantickitti.py # Restructure output directories
│   └── logs/                                   # Preprocessing logs
│
├── figures/                                # Figure generation scripts
│   ├── generate_figures.py                     # Main figures (mIoU, rankings, etc.)
│   ├── generate_classwise_figures.py           # Class-wise radar charts
│   ├── generate_classwise_performance_drop.py  # Performance drop analysis
│   ├── generate_pointcloud_comparison.py       # Point cloud visualizations
│   ├── generate_pointcloud_comparison_o3d.py   # Open3D-based visualizations
│   └── find_good_scans.py                      # Find representative scans
│
├── run_subsampling_phase1_dales_kitti.sh   # Phase 1: CPU methods (RS, DBSCAN, Voxel, Poisson)
├── run_subsampling_phase2_semantickitti.sh # Phase 2: GPU methods (IDIS, FPS) - SemanticKITTI
├── run_subsampling_phase2_dales.sh         # Phase 2: GPU methods - DALES
├── run_subsampling_phase3_semantickitti.sh # Phase 3: DEPOCO subsampling
│
├── train_depoco_loss50_loss90.sh           # Train DEPOCO models for 50%/90% loss
├── preprocess_semantickitti_for_depoco.sh  # Preprocess data for DEPOCO training
│
├── extract_training_metrics.py             # Extract mIoU metrics from training logs
├── extract_inference_metrics.py            # Extract inference metrics
├── analyze_classwise_distribution.py       # Class distribution analysis
├── generate_classwise_distribution.sh      # Generate class distribution reports
├── benchmark_subsampling_efficiency.sh     # Benchmark time/memory efficiency
└── RUN_TESTS.sh                            # Run unit tests
```

---

## Subsampling Pipeline

### Phase 1: CPU-based Methods

Generates subsampled data using CPU methods (RS, DBSCAN, Voxel, Poisson).

```bash
# SemanticKITTI (sequences 00-10)
./scripts/run_subsampling_phase1_dales_kitti.sh --dataset semantickitti

# DALES dataset
./scripts/run_subsampling_phase1_dales_kitti.sh --dataset dales

# Specific loss level
./scripts/run_subsampling_phase1_dales_kitti.sh --dataset semantickitti --loss 90
```

**Methods:** RS (Random Sampling), DBSCAN, Voxel Grid, Poisson Disk

### Phase 2: GPU-based Methods

Generates subsampled data using GPU methods (IDIS, FPS).

```bash
# SemanticKITTI - IDIS
./scripts/run_subsampling_phase2_semantickitti.sh --method IDIS --loss 90

# SemanticKITTI - FPS with seed
./scripts/run_subsampling_phase2_semantickitti.sh --method FPS --loss 90 --seed 1

# DALES
./scripts/run_subsampling_phase2_dales.sh --method IDIS --loss 90
```

**Methods:** IDIS (Inverse Distance Importance Sampling), FPS (Farthest Point Sampling)

### Phase 3: DEPOCO (Deep Compression)

Generates subsampled data using pre-trained DEPOCO encoder-decoder models.

```bash
# Generate DEPOCO subsampled data
./scripts/run_subsampling_phase3_semantickitti.sh --loss 30

# Train new DEPOCO models (for 50%/90% loss)
./scripts/train_depoco_loss50_loss90.sh --loss 50
./scripts/train_depoco_loss50_loss90.sh --loss 90
```

**Available Models:**
| Target Loss | Model | subsampling_dist | Status |
|-------------|-------|------------------|--------|
| 30% | final_skitti_82.5 | 0.85 | VERIFIED |
| 50% | final_skitti_87.5 | 1.3 | TRAINING |
| 70% | final_skitti_92.5 | 1.8 | VERIFIED |
| 90% | final_skitti_97.5 | 2.3 | TRAINING |

---

## Verification

Verify subsampling produces correct loss percentages:

```bash
# Verify specific method/loss
python scripts/preprocessing/verify_subsampling.py --method RS --loss 90

# Verify DEPOCO
python scripts/preprocessing/verify_subsampling.py --method DEPOCO --loss 30

# Verify all
python scripts/preprocessing/verify_subsampling.py --all
```

---

## Metrics Extraction

### Training Metrics

Extract mIoU and training metrics from PTv3 logs:

```bash
# Auto-detect all experiments
python scripts/extract_training_metrics.py --auto

# Specific experiment
python scripts/extract_training_metrics.py --method RS --loss 90 --seed 1

# Output to CSV
python scripts/extract_training_metrics.py --auto --output results.csv
```

### Inference Metrics

Extract inference metrics (throughput, memory):

```bash
python scripts/extract_inference_metrics.py --method RS --loss 90
```

---

## Figure Generation

All figure scripts are in `scripts/figures/`.

### Main Figures

```bash
cd scripts/figures

# Generate all main figures
python generate_figures.py

# Outputs:
#   01_ptv3_metric_grouped.{png,svg,pdf}
#   02_ptv3_spatial_distribution_analysis.{png,svg,pdf}
#   03_ptv3_ranking_bump_chart.{png,svg,pdf}
```

### Class-wise Figures

```bash
# Class-wise radar charts per loss level
python generate_classwise_figures.py

# Outputs:
#   04_classwise_loss30.{png,svg,pdf}
#   05_classwise_loss50.{png,svg,pdf}
#   06_classwise_loss70.{png,svg,pdf}
#   07_classwise_loss90.{png,svg,pdf}
#   08_classwise_loss90_idis_r_sensitivity.{png,svg,pdf}

# Class-wise performance drop analysis
python generate_classwise_performance_drop.py

# Outputs:
#   09_classwise_performance_drop.{png,svg,pdf}
```

### Point Cloud Visualizations

```bash
# Generate point cloud comparison figures
python generate_pointcloud_comparison.py

# Using Open3D (better quality)
python generate_pointcloud_comparison_o3d.py
```

---

## Efficiency Benchmarks

Benchmark subsampling methods for time and memory efficiency:

```bash
./scripts/benchmark_subsampling_efficiency.sh

# Results saved to: benchmark_results/
```

**Sample Results (SemanticKITTI seq 08):**

| Method | Time/Scan (s) | Throughput (scans/s) | Peak RAM (GB) |
|--------|---------------|----------------------|---------------|
| RS | 0.001 | 834.1 | 3.84 |
| Poisson | 0.056 | 17.75 | 4.79 |
| Voxel | 0.077 | 12.67 | 4.67 |
| DBSCAN | 0.161 | 6.20 | 30.80 |
| IDIS | 1.008 | 1.00 | 0.64 |
| FPS | 1.572 | 0.64 | 0.56 |

---

## Preprocessing Scripts

### generate_subsampled_semantickitti.py

Main script for SemanticKITTI subsampling with CPU methods.

```bash
cd scripts/preprocessing

# Full generation
python generate_subsampled_semantickitti.py --workers 4

# Specific method and loss
python generate_subsampled_semantickitti.py --method RS --loss 90 --seed 1

# Dry run
python generate_subsampled_semantickitti.py --dry-run
```

### generate_subsampled_gpu.py

GPU-accelerated subsampling (IDIS, FPS).

```bash
python generate_subsampled_gpu.py --method IDIS --loss 90

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python generate_subsampled_gpu.py --method FPS --loss 90
```

### generate_subsampled_depoco.py

DEPOCO neural compression subsampling.

```bash
python generate_subsampled_depoco.py --loss 30
python generate_subsampled_depoco.py --loss 50
python generate_subsampled_depoco.py --loss 70
python generate_subsampled_depoco.py --loss 90
```

### generate_idis_r_ablation.py

Generate IDIS data for R-value ablation study.

```bash
python generate_idis_r_ablation.py --r-values 5 10 15 20 --loss 90
```

---

## Environment Setup

```bash
# Activate PTv3 environment
cd PTv3
source activate.sh
cd ../scripts
```

For DEPOCO scripts, set environment variables and use separate environment:
```bash
# Set DEPOCO paths (adjust for your installation)
export DEPOCO_BASE=/path/to/depoco_for_transfer
export DEPOCO_VENV=/path/to/venv/py38_depoco
export DEPOCO_DATA=/path/to/depoco_training_data

# Activate DEPOCO environment
source $DEPOCO_VENV/bin/activate
```

See `configs/depoco/README.md` for detailed DEPOCO setup instructions.

---

## Output Structure

```
data/SemanticKITTI/subsampled/
├── RS_loss90_seed1/
│   └── sequences/
│       ├── 00/
│       │   ├── velodyne/*.bin
│       │   └── labels/*.label
│       └── ...
├── IDIS_loss90/
├── FPS_loss90_seed1/
├── DBSCAN_loss90/
├── Voxel_loss90/
├── Poisson_loss90_seed1/
├── DEPOCO_loss30/
├── DEPOCO_loss50/
├── DEPOCO_loss70/
├── DEPOCO_loss90/
└── reports/
    └── verify_*.txt
```

---

**Last Updated:** December 29, 2025
