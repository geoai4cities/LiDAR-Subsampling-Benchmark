# SemanticKITTI PTv3 Training

**Purpose:** Training and inference scripts for Point Transformer V3 on SemanticKITTI dataset.

---

## Directory Structure

```
PTv3/SemanticKITTI/
├── README.md                           # This file
├── configs/
│   ├── _base_/
│   │   └── default_runtime.py          # Base runtime configuration
│   └── semantickitti/
│       ├── ptv3_40gb_official_template.py    # Template for 40GB GPU
│       ├── ptv3_140gb_official_template.py   # Template for 140GB GPU
│       └── generated/                        # Auto-generated experiment configs
├── scripts/
│   ├── generate_configs.py             # Generate all experiment configs
│   ├── train_semantickitti_140gb.sh    # Main training script (140GB GPU)
│   ├── train_semantickitti_40gb.sh     # Training script (40GB GPU)
│   ├── inference_semantickitti_140gb.sh    # Inference on subsampled data
│   ├── inference_on_original_140gb.sh      # Test models on original data
│   ├── test_semantickitti.sh           # Quick testing script
│   ├── profile_model.py                # Model profiling
│   ├── profile_during_training.py      # Training profiling
│   └── quick_profile.py                # Quick profiling
├── outputs/                            # Training outputs
│   ├── {experiment_name}/
│   │   ├── model/
│   │   │   ├── model_best.pth
│   │   │   └── model_last.pth
│   │   ├── log.txt
│   │   └── config.py
└── docs/                               # Documentation
```

---

## Scripts

### train_semantickitti_140gb.sh

Main training script for H200 140GB GPU.

```bash
cd PTv3/SemanticKITTI/scripts

# Start training
./train_semantickitti_140gb.sh --method RS --loss 90 start

# With seed (for stochastic methods)
./train_semantickitti_140gb.sh --method RS --loss 90 --seed 1 start

# Resume training
./train_semantickitti_140gb.sh --method RS --loss 90 resume

# Check status
./train_semantickitti_140gb.sh --method RS --loss 90 status

# Stop training
./train_semantickitti_140gb.sh --method RS --loss 90 stop
```

**Arguments:**
| Argument | Description | Values |
|----------|-------------|--------|
| `--method` | Subsampling method | RS, FPS, IDIS, DBSCAN, Voxel, Poisson, DEPOCO |
| `--loss` | Loss percentage | 0, 10, 30, 50, 70, 90 |
| `--seed` | Random seed (for stochastic methods) | 1, 2, 3 |
| `--r` | IDIS R-value (for ablation) | 5, 10, 15, 20 |

**Actions:**
| Action | Description |
|--------|-------------|
| `start` | Start new training |
| `resume` | Resume from checkpoint |
| `status` | Check training status |
| `stop` | Stop training |

### train_semantickitti_40gb.sh

Training script for A100 40GB GPU (smaller batch size).

```bash
./train_semantickitti_40gb.sh --method RS --loss 90 start
```

### inference_semantickitti_140gb.sh

Run inference on subsampled test data.

```bash
# Inference on subsampled data
./inference_semantickitti_140gb.sh --method RS --loss 90 --seed 1

# IDIS with R-value
./inference_semantickitti_140gb.sh --method IDIS --loss 90 --r 10

# All methods at specific loss
./inference_semantickitti_140gb.sh --loss 90 --all
```

### inference_on_original_140gb.sh

Test trained models on original (non-subsampled) data for generalization analysis.

```bash
# Test RS model on original data
./inference_on_original_140gb.sh --method RS --loss 90 --seed 1

# Test all models on original
./inference_on_original_140gb.sh --loss 90 --all
```

### generate_configs.py

Auto-generate experiment configurations.

```bash
# Generate all configs
python generate_configs.py

# Generate specific config
python generate_configs.py --method RS --loss 90 --seed 1 --gpu 140gb
```

---

## Training Workflow

### 1. Generate Configurations

```bash
cd PTv3/SemanticKITTI/scripts
python generate_configs.py
```

### 2. Start Training

```bash
# Baseline (0% loss)
./train_semantickitti_140gb.sh --method RS --loss 0 start

# Subsampled experiments
./train_semantickitti_140gb.sh --method RS --loss 90 --seed 1 start
./train_semantickitti_140gb.sh --method IDIS --loss 90 start
./train_semantickitti_140gb.sh --method FPS --loss 90 --seed 1 start
```

### 3. Monitor Training

```bash
# Check status
./train_semantickitti_140gb.sh --method RS --loss 90 status

# View live log
tail -f ../outputs/ptv3-v1m1-0-base-RS-loss90-seed1/log.txt

# GPU usage
nvidia-smi -l 1
```

### 4. Run Inference

```bash
# Test on subsampled data
./inference_semantickitti_140gb.sh --method RS --loss 90 --seed 1

# Test on original data (generalization)
./inference_on_original_140gb.sh --method RS --loss 90 --seed 1
```

---

## Experiments Summary

### Completed Experiments

| Method | Loss Levels | Seeds | Status |
|--------|-------------|-------|--------|
| **Baseline** | 0% | - | Complete |
| **RS** | 30%, 50%, 70%, 90% | 1, 2, 3 | Complete |
| **FPS** | 30%, 50%, 70%, 90% | 1, 2, 3 | Complete |
| **Poisson** | 30%, 50%, 70%, 90% | 1, 2, 3 | Complete |
| **IDIS** | 30%, 50%, 70%, 90% | - | Complete |
| **DBSCAN** | 30%, 50%, 70%, 90% | - | Complete |
| **Voxel** | 30%, 50%, 70%, 90% | - | Complete |
| **IDIS R-ablation** | 90% (R=5,10,15,20) | - | Complete |
| **DEPOCO** | 10%, 30%, 70% | - | Complete |

### Key Results (90% Loss)

| Method | Test mIoU | Std Dev | Ranking |
|--------|-----------|---------|---------|
| Baseline | 0.6721 | - | - |
| RS (3 seeds) | 0.5882 | ±0.0036 | 1st |
| DBSCAN | 0.5528 | - | 2nd |
| Voxel | 0.5390 | - | 3rd |
| FPS (3 seeds) | 0.5292 | ±0.0064 | 4th |
| Poisson (3 seeds) | 0.4813 | ±0.0020 | 5th |
| IDIS | 0.4627 | - | 6th |

---

## Configuration Details

### Model Architecture

- **Model:** Point Transformer V3 (PT-v3m1)
- **Classes:** 19 (SemanticKITTI standard)
- **Input channels:** 4 (x, y, z, intensity)
- **Grid size:** 0.05m
- **Flash Attention:** Enabled

### Training Parameters

| Parameter | 140GB GPU | 40GB GPU |
|-----------|-----------|----------|
| Batch size | 20 | 6 |
| Grad accumulation | 4 | 8 |
| Effective batch | 80 | 48 |
| Learning rate | 0.002 | 0.002 |
| Epochs | 50 | 50 |
| Early stopping | 15 epochs | 15 epochs |

### Data Augmentation

- RandomRotate (z: ±1)
- PointClip (range: -35.2 to 35.2 x/y, -4 to 2 z)
- RandomScale ([0.9, 1.1])
- RandomFlip (p=0.5)
- RandomJitter (σ=0.005, clip=0.02)
- GridSample (0.05m)

---

## Output Structure

```
outputs/
├── ptv3-v1m1-0-base-RS-loss0/           # Baseline
│   ├── model/
│   │   ├── model_best.pth
│   │   └── model_last.pth
│   ├── log.txt
│   ├── config.py
│   └── result/
│       └── test_results.json
├── ptv3-v1m1-0-base-RS-loss90-seed1/    # RS 90% seed 1
├── ptv3-v1m1-0-base-RS-loss90-seed2/    # RS 90% seed 2
├── ptv3-v1m1-0-base-RS-loss90-seed3/    # RS 90% seed 3
├── ptv3-v1m1-0-base-IDIS-loss90/        # IDIS 90%
├── ptv3-v1m1-0-base-FPS-loss90-seed1/   # FPS 90% seed 1
└── ...
```

---

## Troubleshooting

### NaN Loss
- Flash Attention is enabled by default
- If NaN occurs, reduce learning rate: `--lr 0.003`

### Out of Memory
- Reduce batch_size in config
- Use 40GB config for smaller GPUs

### Slow Training
- Expected: ~35-40 hours per experiment (140GB)
- Check GPU utilization: `nvidia-smi`

### Resume from Checkpoint
```bash
./train_semantickitti_140gb.sh --method RS --loss 90 resume
```

---

## Environment

```bash
# Activate environment (from project root)
cd PTv3
source activate.sh

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pointcept; print('Pointcept OK')"
```

---

## References

- **PTv3 Paper:** [Point Transformer V3](https://arxiv.org/abs/2312.10035)
- **SemanticKITTI:** [http://semantic-kitti.org/](http://semantic-kitti.org/)
- **Pointcept:** [https://github.com/Pointcept/Pointcept](https://github.com/Pointcept/Pointcept)

---

**Last Updated:** December 30, 2025
