# Preprocessing Scripts

**Purpose:** Data preprocessing and subsampling scripts for LiDAR datasets.

---

## 📁 Scripts Overview

### Data Generation (Main Scripts)

#### `generate_subsampled_semantickitti.py`
**Purpose:** Generate all subsampled versions of SemanticKITTI dataset

**Usage:**
```bash
# Test mode (10 scans, ~5 minutes)
python generate_subsampled_semantickitti.py --test

# Full generation (~405,000 files, faster without 0% loss)
python generate_subsampled_semantickitti.py --workers 16

# Resume from interruption
python generate_subsampled_semantickitti.py --workers 4 --resume
```

**Output:** 90 subdirectories in `data/SemanticKITTI/subsampled/`
- 6 methods × 5 loss levels × 3 seeds
- Each subdirectory contains `.bin` files (point clouds) and `.label` files (labels)

**Methods:** RS, IDIS, FPS, DBSCAN, Voxel, Poisson

**Loss Levels:** 10%, 30%, 50%, 70%, 90%

**Baseline (0% loss):** Uses original data from `data/SemanticKITTI/original/sequences/00/velodyne/`

**Seeds:** 1, 2, 3

**Note:** DEPOCO (deep learning-based) is documented but not implemented. Pre-computed results available in `_archive/`. See `docs/DEPOCO_REFERENCE.md`

---

#### `generate_subsampled_dales.py`
**Purpose:** Generate all subsampled versions of DALES dataset

**Usage:**
```bash
# Test mode (3 tiles, ~2 minutes)
python generate_subsampled_dales.py --test

# Full generation (~3,600 files, faster without 0% loss)
python generate_subsampled_dales.py --workers 16

# Resume from interruption
python generate_subsampled_dales.py --workers 4 --resume
```

**Output:** 90 subdirectories in `data/DALES/subsampled/`
- 6 methods × 5 loss levels × 3 seeds
- Each subdirectory contains `.txt` files (point clouds + labels)

**Methods:** RS, IDIS, FPS, DBSCAN, Voxel, Poisson

**Loss Levels:** 10%, 30%, 50%, 70%, 90%

**Baseline (0% loss):** Uses original data from `data/DALES/original/`

**Seeds:** 1, 2, 3

**Note:** DEPOCO (deep learning-based) is documented but not implemented. Pre-computed results available in `_archive/`. See `docs/DEPOCO_REFERENCE.md`

---

### Data Conversion

#### `convert_ply_to_txt.py`
**Purpose:** Convert DALES PLY files to TXT format for Pointcept

**Usage:**
```bash
# Convert all PLY files in DALES original directory
python convert_ply_to_txt.py \
  --input data/DALES/original/dales_ply/ \
  --output data/DALES/original/

# Convert specific split
python convert_ply_to_txt.py \
  --input data/DALES/original/dales_ply/train/ \
  --output data/DALES/original/train/
```

**Input:** PLY files (binary point cloud format)
**Output:** TXT files (X Y Z R G B Label format)

---

### Legacy Scripts

#### `subsample_data.py`
**Purpose:** Simple CLI for single subsampling operations (outdated)

**Status:** ⚠️ Superseded by `generate_subsampled_*.py` scripts

**Usage:**
```bash
python subsample_data.py \
  --dataset SemanticKITTI \
  --method RS \
  --loss_level 50 \
  --seed 42
```

**Note:** Use `generate_subsampled_*.py` for production data generation.

---

## 📊 Generated Data Structure

### SemanticKITTI Subsampled Data
```
data/SemanticKITTI/subsampled/
├── RS_loss0_seed1/
│   ├── velodyne/           # Subsampled point clouds (.bin)
│   └── labels/             # Corresponding labels (.label)
├── RS_loss0_seed2/
├── RS_loss0_seed3/
├── RS_loss10_seed1/
├── ...
├── IDIS_loss50_seed1/
├── IDIS_loss50_seed2/
├── ...
└── Poisson_loss90_seed3/
```

### DALES Subsampled Data
```
data/DALES/subsampled/
├── RS_loss0_seed1/
│   ├── train/              # Training tiles (.txt)
│   └── test/               # Test tiles (.txt)
├── RS_loss0_seed2/
├── ...
├── FPS_loss30_seed1/
└── Voxel_loss70_seed3/
```

---

## 🔧 Technical Details

### Subsampling Methods

| Method | File | Complexity | Strategy |
|--------|------|------------|----------|
| **RS** | random_sampling.py | O(N) | Uniform probability |
| **IDIS** | idis.py | O(N log N) | Inverse distance importance |
| **FPS** | fps.py | O(N×M×D) | Farthest point sampling |
| **DBSCAN** | dbscan.py | O(N log N) | Density-based clustering |
| **Voxel** | voxel_grid.py | O(N) | Grid-based downsampling |
| **Poisson** | poisson_disk.py | O(N) | Blue noise sampling |

**Implementation Location:** `src/subsampling/`

### Loss Levels

| Loss % | Ratio | Points (SemanticKITTI ~130k) | Points (DALES variable) |
|--------|-------|------------------------------|-------------------------|
| 0% | 1.00 | ~130,000 | Original |
| 10% | 0.90 | ~117,000 | 90% retained |
| 30% | 0.70 | ~91,000 | 70% retained |
| 50% | 0.50 | ~65,000 | 50% retained |
| 70% | 0.30 | ~39,000 | 30% retained |
| 90% | 0.10 | ~13,000 | 10% retained |

### Seeds

Three random seeds for reproducibility:
- Seed 1: Initial experiments
- Seed 2: Validation experiments
- Seed 3: Statistical significance

---

## ⚡ Performance

### SemanticKITTI Generation
- **Test mode:** ~5 minutes (10 scans)
- **Full generation:** Several hours (16 workers, ~17% faster without 0% loss)
- **Total files:** ~405,000 files (was 486,000)
- **Disk space:** ~80-90GB (saved ~10-15GB by skipping 0% loss)

### DALES Generation
- **Test mode:** ~2 minutes (3 tiles)
- **Full generation:** Several hours (16 workers, ~17% faster without 0% loss)
- **Total files:** ~3,600 files (was 4,320)
- **Disk space:** ~40-45GB (saved ~5-8GB by skipping 0% loss)

### Parallel Processing
```bash
# Single worker (slow)
python generate_subsampled_semantickitti.py --workers 1

# Multi-worker (recommended)
python generate_subsampled_semantickitti.py --workers 4

# Maximum parallelism (if resources available)
python generate_subsampled_semantickitti.py --workers 8
```

---

## 🐛 Troubleshooting

### Missing Dependencies
```bash
# Activate environment first
cd ../../PTv3
source activate.sh
cd ../scripts/preprocessing
```

### Out of Memory
```bash
# Reduce workers
python generate_subsampled_semantickitti.py --workers 1

# Process smaller batches (modify script)
```

### Resume Interrupted Generation
```bash
# Scripts automatically skip existing files
python generate_subsampled_semantickitti.py --workers 4
```

### Verify Output
```bash
# Count generated files
find ../../data/SemanticKITTI/subsampled/ -name "*.bin" | wc -l
find ../../data/DALES/subsampled/ -name "*.txt" | wc -l

# Check specific subdirectory
ls ../../data/SemanticKITTI/subsampled/IDIS_loss50_seed1/velodyne/ | wc -l
```

---

## 📚 Related Documentation

- **Subsampling Methods:** [../../src/subsampling/README.md](../../src/subsampling/README.md) (if exists)
- **Data Structure:** [../../FOLDER_STRUCTURE.md](../../FOLDER_STRUCTURE.md)
- **Quick Start:** [../../PTv3/QUICKSTART.md](../../PTv3/QUICKSTART.md)

---

**Last Updated:** November 24, 2025
**Location:** `scripts/preprocessing/`
**Status:** Production-ready
