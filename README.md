# LiDAR Subsampling Benchmark

**Performance Analysis of Subsampled LiDAR Point Clouds Using Deep Learning Based Semantic Segmentation**

*Official code repository for the paper submitted to Applied Intelligence (APIN)*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


---

## Overview

This repository provides a comprehensive benchmarking framework for evaluating point cloud subsampling methods on outdoor LiDAR semantic segmentation tasks. We evaluate **7 subsampling methods** across **multiple point loss levels** using state-of-the-art Point Transformer V3 (PTv3) architecture on the complete SemanticKITTI dataset (sequences 00-10, ~23,201 scans).

### Key Contributions

- Comprehensive evaluation of 7 subsampling methods at 4 loss levels (30%, 50%, 70%, 90%)
- Multi-seed experiments (3 seeds) for stochastic methods ensuring statistical reliability
- IDIS R-value ablation study (R = 5, 10, 15, 20m)
- Computational efficiency benchmarks (time, memory, throughput)
- Class-wise performance analysis across 19 semantic categories
- Generalization testing (models trained on subsampled data, evaluated on original data)

---

## Subsampling Methods

| Method | Strategy |
|--------|----------|
| **RS** (Random Sampling) | Uniform random selection |
| **SB** (Poisson Disk) | Space-based blue noise distribution |
| **VB** (Voxel Grid) | Deterministic grid downsampling |
| **DBSCAN** | Density-based clustering centroids |
| **IDIS** (Inverse Distance Importance) | Feature-preserving importance sampling |
| **FPS** (Farthest Point Sampling) | Maximum spatial coverage |
| **DEPOCO** | Deep learning compression |

---

## Project Structure

```
LiDAR-Subsampling-Benchmark/
├── data/                              # Datasets
│   └── SemanticKITTI/
│       ├── original/                  # Original SemanticKITTI data
│       │   └── sequences/             # 00-10 sequences
│       └── subsampled/                # Generated subsampled datasets
│           ├── RS_loss90_seed1/
│           ├── IDIS_loss90/
│           ├── FPS_loss30/
│           └── ...
│
├── configs/                           # Configuration files
│   └── depoco/                        # DEPOCO model configurations
│       ├── README.md                  # DEPOCO setup instructions
│       ├── final_skitti_*.yaml        # Model configs (30%, 50%, 70%, 90%)
│       ├── preprocess_semantickitti.sh  # Data preprocessing
│       ├── train_depoco.sh            # Model training
│       └── generate_subsampled.sh     # Generate subsampled data
│
├── PTv3/                              # Point Transformer V3 workspace
│   ├── setup_venv.sh                  # Environment setup script
│   ├── activate.sh                    # Environment activation
│   ├── pointcept/                     # Pointcept framework
│   └── SemanticKITTI/
│       ├── configs/                   # Training configurations
│       ├── outputs/                   # Training outputs & checkpoints
│       └── scripts/                   # Training & inference scripts
│
├── [RandLANet/](RandLANet/README.md)  # RandLA-Net workspace
│
├── src/subsampling/                   # Subsampling method implementations
│   ├── random_sampling.py
│   ├── idis.py / idis_gpu.py         # 55x GPU speedup
│   ├── fps.py / fps_gpu.py
│   ├── dbscan.py
│   ├── voxel_grid.py
│   ├── poisson_disk.py
│   └── depoco.py
│
├── scripts/                           # Pipeline scripts
│   ├── preprocessing/                 # Data generation scripts
│   ├── figures/                       # Figure generation scripts
│   │   ├── generate_all.sh           # One-command pipeline
│   │   ├── generate_figures.py       # Main figures
│   │   ├── generate_classwise_figures.py
│   │   └── generate_classwise_performance_drop.py
│   ├── extract_training_metrics.py   # Metrics extraction
│   ├── extract_inference_metrics.py  # Inference metrics
│   └── benchmark_subsampling_efficiency.sh
│
└── docs/                              # Documentation & results
    ├── tables/                        # Extracted metrics tables
    │   ├── all_experiments_detailed.txt
    │   ├── inference/                 # Inference metrics
    │   └── inference_on_original/     # Generalization metrics
    └── figures/                       # Generated figures (PNG, SVG, PDF)
```

---

## Quick Start

### Step 1: Dataset Setup

Download SemanticKITTI from: [http://semantic-kitti.org/](http://semantic-kitti.org/)

Place data in the `data/` directory:

```
data/SemanticKITTI/original/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/          # .bin point cloud files
│   │   └── labels/            # .label semantic labels
│   ├── 01/
│   ...
│   └── 10/
```

### Step 2: Environment Setup

```bash
cd PTv3

# Run automated setup (30-45 minutes)
./setup_venv.sh

# Activate environment
source activate.sh
```

### Step 3: Generate Subsampled Data

```bash
# Phase 1: CPU-based methods (RS, DBSCAN, VB, SB)
./scripts/run_subsampling_phase1_dales_kitti.sh --dataset semantickitti

# Phase 2: GPU-based methods (IDIS, FPS)
./scripts/run_subsampling_phase2_semantickitti.sh --method IDIS --loss 90
./scripts/run_subsampling_phase2_semantickitti.sh --method FPS --loss 90 --seed 1

# Phase 3: DEPOCO (Deep Learning Compression)
# Requires DEPOCO environment - see configs/depoco/README.md
./scripts/run_subsampling_phase3_semantickitti.sh --loss 30
./scripts/run_subsampling_phase3_semantickitti.sh --loss 50
./scripts/run_subsampling_phase3_semantickitti.sh --loss 70
```

> **See:** [scripts/README.md](scripts/README.md) for detailed subsampling documentation.
> **See:** [configs/depoco/README.md](configs/depoco/README.md) for DEPOCO setup and configuration.

### Step 4: Train Models

```bash
cd PTv3/SemanticKITTI/scripts

# Train baseline
./train_semantickitti_140gb.sh --method RS --loss 0 start

# Train on subsampled data
./train_semantickitti_140gb.sh --method RS --loss 90 --seed 1 start
./train_semantickitti_140gb.sh --method IDIS --loss 90 start
```

> **See:** [PTv3/SemanticKITTI/README.md](PTv3/SemanticKITTI/README.md) for detailed training documentation.

### Step 5: Evaluate & Generate Results for figures

```bash
# Generate all figures (extracts metrics + generates all figures)
./scripts/figures/generate_all.sh

# Or run individual steps:
python scripts/extract_training_metrics.py
python scripts/extract_inference_metrics.py
python scripts/figures/generate_figures.py
python scripts/figures/generate_classwise_figures.py
python scripts/figures/generate_classwise_performance_drop.py

# With point cloud visualization (requires xvfb-run):
./scripts/figures/generate_all.sh --with-pointcloud
```

### Step 6: Benchmark Subsampling Efficiency (Optional)

Measure computational efficiency (time, memory, throughput) of subsampling methods:

```bash
# Benchmark all methods on validation sequence (default: seq 08)
./scripts/benchmark_subsampling_efficiency.sh

# Benchmark specific methods
./scripts/benchmark_subsampling_efficiency.sh --methods RS,FPS,IDIS

# Benchmark with custom settings
./scripts/benchmark_subsampling_efficiency.sh --sequences "00 01 02" --loss 90 --workers 16
```

**Metrics collected:**
- Wall-clock time (total and per-scan)
- Peak memory usage (RAM for CPU, VRAM for GPU)
- CPU/GPU utilization
- Throughput (scans/second)

Results saved to `benchmark_results/` directory.

---

## Citation

*To be updated after publication.*

---

## Acknowledgments

- **Pointcept Framework:** [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept)
- **Point Transformer V3:** [Paper](https://arxiv.org/abs/2312.10035)
- **SemanticKITTI Dataset:** [Website](http://semantic-kitti.org/)
- **DEPOCO:** [PRBonn/deep-point-map-compression](https://github.com/PRBonn/deep-point-map-compression)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact


For questions or issues, please open a GitHub issue.
