# DALES PTv3 Experiments

This directory contains configurations and scripts for running PTv3 experiments on DALES (Dayton Annotated LiDAR Earth Scan) dataset with different subsampling methods.

## Directory Structure

```
DALES/
├── configs/
│   ├── _base_/
│   │   └── default_runtime.py          # Base runtime configuration
│   └── dales/
│       ├── ptv3_40gb_official_template.py   # Template for A100 40GB
│       ├── ptv3_140gb_official_template.py  # Template for H200 140GB
│       └── generated/                       # Auto-generated experiment configs
├── scripts/
│   ├── generate_configs.py             # Generate all experiment configs
│   ├── train_40gb.sh                   # Training script for 40GB GPU
│   ├── train_140gb.sh                  # Training script for 140GB GPU
│   └── test.sh                         # Testing script
├── outputs/
│   ├── checkpoints/                    # Model checkpoints
│   ├── logs/                           # Training logs
│   └── results/                        # Evaluation results
├── data/                               # Symlink to actual data
├── docs/                               # Experiment documentation
└── notebooks/                          # Analysis notebooks
```

## Quick Start

### 1. Generate Experiment Configs

```bash
cd PTv3/DALES/scripts
python generate_configs.py
```

This will create configs for all experiments:
- Methods: RS, IDIS (Tier 1); FPS, DBSCAN (baselines)
- Loss levels: 0%, 10%, 30%, 50%, 70%, 90%
- Seeds: 1, 2, 3
- GPUs: 40GB and 140GB configs

Total configs generated: **54 for Tier 1** (3 seeds × 3 loss × 2 methods × 3 configs)

### 2. Train a Single Experiment

**On 40GB A100 (GPU 0) - Note: DALES typically runs on GPU 1:**
```bash
./scripts/train_40gb.sh start RS 0 1  # Method, Loss, Seed
```

**On 140GB H200 (GPU 1) - Primary GPU for DALES:**
```bash
./scripts/train_140gb.sh start RS 0 1
```

### 3. Run All Experiments (Managed Queue)

```bash
# From main benchmark directory
python scripts/experiment_queue.py --dataset DALES
```

## Dataset Requirements

**Data Location:** `data/DALES/`

**Structure:**
```
DALES/
├── original/
│   ├── 5030001.txt                     # Train tile 1
│   ├── 5030002.txt                     # Train tile 2
│   ├── 5030020.txt                     # Validation tile
│   └── 5030040.txt                     # Test tile
└── subsampled/
    ├── RS_loss0/
    │   ├── 5030001.txt
    │   ├── 5030002.txt
    │   ├── 5030020.txt
    │   └── 5030040.txt
    ├── RS_loss10/
    ├── RS_loss50/
    ├── IDIS_loss0/
    ├── IDIS_loss50/
    ├── FPS_loss0/
    ├── FPS_loss50/
    └── ...
```

**File Format:** Each .txt file contains space/tab-separated values:
```
x  y  z  intensity  return_num  num_returns  class
```
- Columns 0-2: XYZ coordinates (float)
- Column 3: Intensity (float)
- Column 4-5: Return information (int)
- Column 6: Class label (int, 0-7)

## Dataset Characteristics

### Tiles (Consistent with Paper)
- **Train:** 5030001, 5030002 (2 tiles)
- **Val:** 5030020 (1 tile)
- **Test:** 5030040 (1 tile)

### Classes (8 total)
0. Ground
1. Vegetation
2. Cars
3. Trucks
4. Power lines
5. Fences
6. Poles
7. Buildings

### Data Type
- **Source:** Airborne Laser Scanning (ALS)
- **Point density:** ~1-5 million points per tile
- **Characteristics:** Large-scale outdoor scenes, ~100m × 100m per tile

## Experiments

### Tier 1 (Critical - Week 2-7)

| Loss | Method | Seeds | GPU | Est. Time | Priority |
|------|--------|-------|-----|-----------|----------|
| 0%   | RS     | 1,2,3 | 1   | 90 hr     | P1       |
| 0%   | IDIS   | 1,2,3 | 1   | 90 hr     | P1       |
| 50%  | RS     | 1,2,3 | 1   | 90 hr     | P1       |
| 50%  | IDIS   | 1,2,3 | 1   | 90 hr     | P1       |
| 90%  | RS     | 1,2,3 | 1   | 90 hr     | P1       |
| 90%  | IDIS   | 1,2,3 | 1   | 90 hr     | P1       |

**Plus FPS/DBSCAN baselines:** 6 experiments (loss 0%, 50%, 90% × 2 methods)

**Total GPU 1 time:** ~720 hours (30 days) - **Fits in 42 days ✓**

### Configuration Details

**Model:** Point Transformer V3 (PT-v3m1)
- **Classes:** 8 (DALES)
- **Input channels:** 1 (intensity/strength)
- **Grid size:** 0.10m (larger for ALS data vs 0.05m for mobile LiDAR)
- **Flash Attention:** Enabled (prevents NaN)

**Training (OFFICIAL PTv3):**
- **Learning rate:** 0.006
- **Weight decay:** 0.05
- **Epochs:** 50 (with early stopping patience=15)
- **Optimizer:** AdamW
- **Scheduler:** OneCycleLR
- **Batch size (40GB):** 8 (effective: 32 with grad_accum×4)
- **Batch size (140GB):** 12 (effective: 24 with grad_accum×2)
- **Loop:** 10 (dataset has only 4 tiles, loop for balanced training)

**Data Augmentation:**
- CenterShift (apply_z=True)
- RandomDropout (0.2, p=0.2)
- RandomRotate (z: ±1, x/y: ±1/64)
- RandomScale ([0.9, 1.1])
- RandomFlip (p=0.5)
- RandomJitter (σ=0.005, clip=0.02)
- ElasticDistortion (official strength)
- GridSample (0.10m - adapted for ALS)

**Class Weights:**
```python
[1.0, 1.2, 5.0, 8.0, 10.0, 6.0, 7.0, 2.0]
# Ground, Vegetation, Cars, Trucks, Power lines, Fences, Poles, Buildings
```

## Key Differences from SemanticKITTI

| Aspect | SemanticKITTI | DALES |
|--------|---------------|-------|
| **Data type** | Mobile LiDAR | Airborne LiDAR (ALS) |
| **Classes** | 19 | 8 |
| **Grid size** | 0.05m | 0.10m |
| **File format** | .bin (binary) | .txt (text) |
| **Point density** | ~120k per scan | ~1-5M per tile |
| **Coverage** | Street-level | Bird's-eye view |
| **Loop** | 1 (large dataset) | 10 (small dataset) |
| **Primary GPU** | GPU 0 (A100 40GB) | GPU 1 (H200 140GB) |

## Results

Results will be saved to:
- **Checkpoints:** `outputs/checkpoints/{method}_loss{loss}_seed{seed}/`
- **Logs:** `outputs/logs/{method}_loss{loss}_seed{seed}.log`
- **Metrics:** `outputs/results/{method}_loss{loss}_seed{seed}_metrics.json`

## Monitoring

**Check training status:**
```bash
./scripts/monitor.sh
```

**View live logs:**
```bash
tail -f outputs/logs/RS_loss0_seed1.log
```

**Check GPU utilization:**
```bash
nvidia-smi -l 1
```

## Troubleshooting

### NaN Loss Issues
- ✅ **Prevented:** Using official PTv3 settings with Flash Attention enabled
- Config has `enable_flash=True`, `upcast_attention=False`, `upcast_softmax=False`
- If NaN still occurs: reduce learning rate by 50%

### OOM (Out of Memory)
- **40GB config:** Optimized for ~28-35GB usage
- **140GB config:** Optimized for ~60-70GB usage
- DALES tiles can be large (1-5M points)
- If OOM: reduce `batch_size` or increase `grid_size` to 0.15m

### Slow Training
- **Expected time (40GB):** ~35-40 hours with early stopping
- **Expected time (140GB):** ~28-30 hours with early stopping
- Early stopping saves 25-33% time vs full 50 epochs
- Loop=10 means each tile seen 10× per epoch (necessary for small dataset)

### File Not Found Errors
- Ensure .txt files are in correct location: `data/DALES/original/`
- File names must match exactly: 5030001.txt, 5030002.txt, 5030020.txt, 5030040.txt
- Check file format: 7 columns (x, y, z, intensity, return_num, num_returns, class)

## Dataset Loader Implementation

The DALES dataset loader (`pointcept/datasets/dales.py`) handles:
- ✅ .txt file loading (7-column format)
- ✅ Coordinate extraction (columns 0-2)
- ✅ Intensity extraction (column 3)
- ✅ Label extraction and remapping (column 6)
- ✅ Flexible path resolution (original and subsampled)
- ✅ Proper train/val/test splits

**Key features:**
```python
# Load .txt file
data = np.loadtxt(tile_file)

# Extract components
coord = data[:, :3]      # XYZ
strength = data[:, 3:4]  # Intensity
segment = data[:, 6]     # Labels (0-7)
```

## References

- **DALES Dataset:** [University of Dayton](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php)
- **PTv3 Paper:** [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
- **Original Paper:** LiDAR Subsampling Benchmark (uses 4 tiles: 5030001, 5030002, 5030020, 5030040)
- **BhopalMLS Config:** Based on proven `ptv3_hdf5_40gb_12class_official.py`

---

*Created: November 24, 2025*
*Status: Ready for experiments*
