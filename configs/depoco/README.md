# DEPOCO Configuration Files

**DEPOCO: Deep Point Cloud Compression**

Reference: [Wiesmann et al., IEEE RA-L 2021](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf)

---

## Model Configurations

These YAML configurations define the DEPOCO encoder-decoder network architecture
for different compression levels on SemanticKITTI dataset.

### Model Configurations

| Config File | Target Loss | subsampling_dist | Verified Loss | Status |
|-------------|-------------|------------------|---------------|--------|
| `final_skitti_82.5.yaml` | 30% | 0.85 | 29.0% | VERIFIED |
| `final_skitti_87.5.yaml` | 50% | 1.3 | ~50% | - |
| `final_skitti_92.5.yaml` | 70% | 1.8 | 72.9% | VERIFIED |
| `final_skitti_97.5.yaml` | 90% | 2.3 | ~90% | - |

---

## Key Parameter: subsampling_dist

The `subsampling_dist` parameter in the encoder controls the compression ratio:

- **Lower value** = Less compression = More points retained
- **Higher value** = More compression = Fewer points retained

### Verified Mapping

```
subsampling_dist  →  Actual Loss (%)
     0.85         →     29.0%
     1.3          →     ~50%  (interpolated)
     1.8          →     72.9%
     2.3          →     ~90%  (extrapolated)
```

---

## Usage

### 1. Setup DEPOCO Environment

```bash
# Clone DEPOCO repository
git clone https://github.com/PRBonn/deep-point-map-compression.git depoco

# Create virtual environment (Python 3.8 required)
python3.8 -m venv venv_depoco
source venv_depoco/bin/activate

# Install dependencies
pip install torch==1.9.0 torchvision torchaudio
pip install ruamel.yaml tensorboard scipy scikit-learn open3d tqdm
```

### 2. Set Environment Variables

```bash
# Set paths (adjust for your installation)
export DEPOCO_BASE=/path/to/depoco_for_transfer
export DEPOCO_VENV=/path/to/venv_depoco
export DEPOCO_DATA=/path/to/output_submaps
```

### 3. Preprocess Data

Convert SemanticKITTI to DEPOCO submap format:

```bash
cd configs/depoco

# Preprocess training data
./preprocess_semantickitti.sh --input ../../data/SemanticKITTI/original

# Or with custom paths
./preprocess_semantickitti.sh --input /path/to/semantickitti
```

### 4. Train DEPOCO Model

```bash
# Train for specific loss level
./train_depoco.sh --loss 30    # 30% loss model
./train_depoco.sh --loss 50    # 50% loss model
./train_depoco.sh --loss 70    # 70% loss model
./train_depoco.sh --loss 90    # 90% loss model
```

### 5. Generate Subsampled Data

```bash
# Generate subsampled SemanticKITTI data
./generate_subsampled.sh --loss 30
./generate_subsampled.sh --loss 50
./generate_subsampled.sh --loss 70
./generate_subsampled.sh --loss 90

# Or all at once
./generate_subsampled.sh
```

---

## Configuration Structure

```yaml
train:
  experiment_id: "final_skitti_XX.X"
  max_epochs: 250
  batch_size: 10
  # ... training parameters

grid:
  pose_distance: 15
  size: [40.0, 40.0, 15.0]
  voxel_size: 0.1
  # ... voxel grid parameters

network:
  encoder_blocks:
    - type: "GridSampleConv"
      parameters:
        subsampling_dist: X.XX  # KEY PARAMETER for compression
        # ... encoder parameters
  decoder_blocks:
    - type: "AdaptiveDeconv"
      # ... decoder parameters

dataset:
  data_folders:
    grid_output: "path/to/submaps"
    train: ["train"]
    valid: ["validation"]
    test: ["validation"]
```

---

## Network Architecture

### Encoder
- 3x GridSampleConv blocks with progressive subsampling
- Linear layer to compress features to 3D offsets

### Decoder
- 4x AdaptiveDeconv blocks for upsampling
- Linear layer to output 3D point coordinates

---

## References

- **Paper**: [Deep Compression for Dense Point Cloud Maps](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf)
- **Code**: [PRBonn/deep-point-map-compression](https://github.com/PRBonn/deep-point-map-compression)
- **Authors**: Louis Wiesmann, Andres Milioto, Xieyuanli Chen, Cyrill Stachniss, Jens Behley

---

**Last Updated:** December 29, 2025
