# Data Directory

## Structure

```
data/
├── DALES/
│   ├── original/          # Original DALES tiles
│   │   ├── 5019.ply
│   │   ├── 5029.ply
│   │   ├── 5030.ply
│   │   └── 5039.ply
│   └── subsampled/        # Subsampled versions
│       ├── RS_50-55/
│       ├── IDIS_50-55/
│       └── ...
└── SemanticKITTI/
    ├── original/          # Original SemanticKITTI
    │   └── sequences/
    │       └── 00/
    │           ├── velodyne/
    │           └── labels/
    └── subsampled/        # Subsampled versions
        └── ...
```

## Download Instructions

### DALES
1. Visit: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php
2. Download tiles: 5019, 5029, 5030, 5039
3. Place in `data/DALES/original/`

### SemanticKITTI
1. Visit: http://www.semantic-kitti.org/dataset.html
2. Download sequence 00 (velodyne + labels)
3. Extract to `data/SemanticKITTI/original/sequences/00/`

## Data Format

### DALES (.ply files)
- Coordinates: x, y, z (float32)
- Colors: red, green, blue (uint8)
- Labels: class (uint8, 0-7)

### SemanticKITTI (.bin files)
- Points: x, y, z, intensity (float32)
- Labels: semantic_label (uint32, in .label files)
