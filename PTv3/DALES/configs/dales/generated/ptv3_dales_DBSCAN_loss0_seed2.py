"""
PTv3 Configuration for DALES Dataset - A100 40GB OFFICIAL SETTINGS

Based on: BhopalMLS ptv3_hdf5_40gb_12class_official.py
Adapted for: DALES 8-class semantic segmentation
Date: 2025-11-24
Status: TEMPLATE - Use generate_configs.py to create experiment-specific configs

Key Adaptations for DALES:
1. ✅ Dataset: DALESDataset (custom loader for .txt files)
2. ✅ Classes: 8 (Ground, Vegetation, Cars, Trucks, Power lines, Fences, Poles, Buildings)
3. ✅ Grid size: 0.10 (vs 0.05 for SemanticKITTI) - ALS data characteristics
4. ✅ Channels: 4 (x, y, z, intensity) - using 1 feature (intensity)
5. ✅ Tiles: 5030001, 5030002, 5030020, 5030040 (consistent with paper)
6. ✅ Official PTv3 settings: Flash Attention enabled, lr=0.006, weight_decay=0.05

CRITICAL SETTINGS (Prevents NaN):
- enable_flash=True
- upcast_attention=False
- upcast_softmax=False
- Early stopping with patience=15

GPU: A100 40GB (or A6000 48GB)
CUDA: 11.8+
Estimated Memory: ~28-35GB
Training Time: ~35-40 hours with early stopping
"""

_base_ = ["../_base_/default_runtime.py"]

# Experiment settings - WILL BE REPLACED BY GENERATE SCRIPT
experiment = dict(
    name="ptv3_dales_DBSCAN_loss0_seed2_40gb",
    seed=2,
)

# Data root - WILL BE REPLACED BY GENERATE SCRIPT
# Example: "../../data/DALES/subsampled/RS_loss0"
data_root = "../../../../../data/DALES/original"

# Model configuration - OFFICIAL PTv3 SETTINGS
model = dict(
    type="DefaultSegmentorV2",
    num_classes=8,  # DALES has 8 classes
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=1,  # intensity only
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,

        # CRITICAL: OFFICIAL FLASH ATTENTION SETTINGS
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,

        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("DALES",),
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            # Class weights for DALES (computed from dataset statistics)
            weight=[
                1.0,    # Ground (abundant)
                1.2,    # Vegetation (abundant)
                5.0,    # Cars (moderate)
                8.0,    # Trucks (rare)
                10.0,   # Power lines (rare, thin)
                6.0,    # Fences (moderate)
                7.0,    # Poles (moderate, thin)
                2.0,    # Buildings (moderate)
            ],
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Dataset configuration
dataset_type = "DALESDataset"
num_classes = 8
class_names = [
    "Ground",
    "Vegetation",
    "Cars",
    "Trucks",
    "Power lines",
    "Fences",
    "Poles",
    "Buildings",
]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,

    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # Similar augmentations but adapted for ALS data
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),

            # ElasticDistortion at official strength
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            dict(
                type="GridSample",
                grid_size=0.10,  # Larger for ALS data (vs 0.05 for mobile LiDAR)
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
        loop=10,  # DALES has only 4 tiles, use loop for balanced training
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.10,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.10,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("strength",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
        ignore_index=-1,
    ),
)

# Training configuration - OFFICIAL PTv3 SETTINGS
epoch = 10
lr = 0.006
weight_decay = 0.05

optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=weight_decay,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

param_dicts = [dict(keyword="block", lr=lr * 0.1)]

# Training settings - A100 40GB optimized
batch_size = 8
batch_size_val = 8
batch_size_test = 1
num_worker = 16
num_worker_test = 1
gradient_accumulation_steps = 4  # Effective batch = 32
mix_prob = 0.8
empty_cache = False
empty_cache_per_epoch = False
enable_amp = True
amp_dtype = "bfloat16"

# Hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(
        type="EarlyStoppingHook",
        patience=15,
        min_delta=0.001,
        metric_name="mIoU",
        mode="max",
        restore_best_weights=True,
        verbose=True,
    ),
    dict(type="CheckpointSaver", save_freq=10),
    dict(type="PreciseEvaluator", test_last=False),
]

# Wandb configuration
wandb = dict(
    project="lidar-subsampling-benchmark",
    name="ptv3_dales_DBSCAN_loss0_seed2_40gb",
    entity=None,
    tags=["dales", "a100-40gb", "ptv3", "DBSCAN", "loss0"],
)
