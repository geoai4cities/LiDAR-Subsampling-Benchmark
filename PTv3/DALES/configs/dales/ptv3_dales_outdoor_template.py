"""
PTv3 Configuration for DALES - Based on Outdoor LiDAR Pattern

Base: Similar to SemanticKITTI/NuScenes official configs (outdoor LiDAR)
Modified: Adapted for DALES ALS characteristics
Date: 2025-11-24

RATIONALE:
- DALES is outdoor LiDAR (airborne)
- More similar to SemanticKITTI/NuScenes (outdoor) than BhopalMLS (indoor MLS)
- Use outdoor LiDAR hyperparameter pattern: lr=0.002, weight_decay=0.005
- Add Flash Attention for PT-v3m1 upgrade

KEY CHANGES FROM PREVIOUS (BhopalMLS-based):
1. lr: 0.006 → 0.002 (outdoor LiDAR standard)
2. weight_decay: 0.05 → 0.005 (outdoor LiDAR standard)
3. pct_start: 0.05 → 0.04 (outdoor LiDAR standard)
4. Simpler augmentations (outdoor pattern)
5. KEPT: Flash Attention, Early stopping, PT-v3m1

GPU: Flexible (works on 40GB or 140GB)
CUDA: 11.8+
Estimated Memory: ~28-35GB (batch_size=8)
Training Time: ~35-40 hours with early stopping
"""

_base_ = ["../_base_/default_runtime.py"]

# Experiment settings - WILL BE REPLACED BY GENERATE SCRIPT
experiment = dict(
    name="ptv3_dales_{METHOD}_loss{LOSS}_seed{SEED}",
    seed=42,  # WILL BE REPLACED
)

# misc custom setting (outdoor LiDAR pattern)
batch_size = 8  # Works for both 40GB and 140GB
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings - PT-v3m1 WITH FLASH ATTENTION
model = dict(
    type="DefaultSegmentorV2",
    num_classes=8,  # DALES has 8 classes
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,  # x, y, z, intensity
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

        # CRITICAL: FLASH ATTENTION SETTINGS
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
            # Class weights for DALES (can be adjusted based on class frequencies)
            weight=[
                1.0,    # Ground
                1.2,    # Vegetation
                5.0,    # Cars
                8.0,    # Trucks
                10.0,   # Power lines
                6.0,    # Fences
                7.0,    # Poles
                2.0,    # Buildings
            ],
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings - OUTDOOR LIDAR STANDARD (from SemanticKITTI/NuScenes)
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)  # Outdoor LiDAR standard
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,  # Outdoor LiDAR standard
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "DALESDataset"
data_root = "DATA_ROOT_PLACEHOLDER"  # Will be replaced by generate script
ignore_index = -1
names = [
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
    num_classes=8,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",  # tiles 5030001, 5030002
        data_root=data_root,
        transform=[
            # Outdoor LiDAR augmentation pattern (simpler than indoor)
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.10,  # Larger for ALS (vs 0.05 for mobile LiDAR)
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # Note: No PointClip for DALES (tile-based, not range-based)
            # Note: No SphereCrop (ALS has different spatial characteristics)
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        loop=10,  # DALES has only 4 tiles, need loop for sufficient training
    ),
    val=dict(
        type=dataset_type,
        split="val",  # tile 5030020
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.10,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",  # tile 5030040
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
        ignore_index=ignore_index,
    ),
)

# Hooks - WITH EARLY STOPPING
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
    name="ptv3_dales_{METHOD}_loss{LOSS}_seed{SEED}",
    entity=None,
    tags=["dales", "ptv3", "{METHOD}", "loss{LOSS}", "outdoor-lidar"],
)
