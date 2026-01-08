"""
PTv3 Configuration for SemanticKITTI Dataset - H200 140GB (Official Outdoor Settings)

Based on: Official Pointcept PTv3 Waymo/nuScenes configs
Reference: https://github.com/Pointcept/Pointcept/blob/main/configs/waymo/semseg-pt-v3m1-0-base.py
Date: 2025-12-02
Status: TEMPLATE - Use generate_configs.py to create experiment-specific configs

IMPORTANT: SemanticKITTI is an OUTDOOR driving dataset like Waymo/nuScenes.
Uses official outdoor PTv3 settings, NOT indoor (ScanNet) settings.

Official Outdoor PTv3 Settings:
1. in_channels=4 (coord + strength features)
2. lr=0.002, weight_decay=0.005
3. grid_size=0.05 (outdoor standard)
4. pct_start=0.04, final_div_factor=100.0
5. PointClip to limit point cloud range
6. NO SphereCrop, NO ElasticDistortion (outdoor configs don't use these)
7. feat_keys=("coord", "strength") for 4-channel input

GPU: H200 140GB / H100 80GB
CUDA: 12.x+
Memory Usage: ~80GB
Training Time: ~2h per epoch, ~12h for 5 epochs (incl. final testing)
"""

_base_ = ["../_base_/default_runtime.py"]

# Experiment settings - WILL BE REPLACED BY GENERATE SCRIPT
experiment = dict(
    name="ptv3_semantickitti_Poisson_loss90_seed1_140gb",
    seed=1,
)

# Data root - WILL BE REPLACED BY GENERATE SCRIPT
data_root = "../../data/SemanticKITTI/subsampled/Poisson_loss90_seed1"

# Model configuration - OFFICIAL PTv3 OUTDOOR SETTINGS (from Waymo/nuScenes)
model = dict(
    type="DefaultSegmentorV2",
    num_classes=19,  # SemanticKITTI has 19 classes
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,  # OFFICIAL: coord (3) + strength (1) = 4 channels
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

        # OFFICIAL FLASH ATTENTION SETTINGS
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
        pdnorm_conditions=("SemanticKITTI",),
    ),
    criteria=[
        # Weighted CrossEntropyLoss + LovaszLoss (official outdoor config)
        dict(
            type="CrossEntropyLoss",
            weight=[
                3.1557,   # car
                8.7029,   # bicycle
                7.8281,   # motorcycle
                6.1354,   # truck
                6.3161,   # other-vehicle
                7.9937,   # person
                8.9704,   # bicyclist
                10.1922,  # motorcyclist
                1.6155,   # road
                4.2187,   # parking
                1.9385,   # sidewalk
                5.5455,   # other-ground
                2.0198,   # building
                2.6261,   # fence
                1.3212,   # vegetation
                5.1102,   # trunk
                2.5492,   # terrain
                5.8585,   # pole
                7.3929,   # traffic-sign
            ],
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# Dataset configuration
dataset_type = "SemanticKITTIDataset"
num_classes = 19
class_names = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
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
            # OFFICIAL OUTDOOR TRANSFORM PIPELINE (from Waymo/nuScenes PTv3)
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # PointClip - OFFICIAL outdoor setting to limit point cloud range
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # GridSample at 0.05 - OFFICIAL outdoor grid size
            dict(
                type="GridSample",
                grid_size=0.05,  # OFFICIAL outdoor grid size (not 0.02)
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),  # OFFICIAL: 4 channels (coord + strength)
            ),
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            # PointClip for validation too
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,  # OFFICIAL outdoor grid size
                hash_type="fnv",
                mode="train",  # Must be "train" when test_mode=False (returns dict, not list)
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "strength"),  # OFFICIAL: 4 channels
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),

    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # PointClip first
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,  # Finer grid for test (half of train)
                hash_type="fnv",
                mode="train",  # MUST be "train" here
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,  # OFFICIAL outdoor grid size
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),  # OFFICIAL: 4 channels
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

# Training configuration - OFFICIAL OUTDOOR PTv3 SETTINGS (from Waymo/nuScenes)
epoch = 10  # Default run (extend to 50 for full training)
eval_epoch = 1
lr = 0.002  # OFFICIAL outdoor setting
weight_decay = 0.005  # OFFICIAL outdoor setting

optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=weight_decay,
)

# OFFICIAL outdoor scheduler settings
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr * 0.1],  # Backbone gets 10% of head LR
    pct_start=0.04,  # OFFICIAL outdoor setting
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,  # OFFICIAL outdoor setting
)

# Layer-wise learning rate decay
param_dicts = [dict(keyword="block", lr=lr * 0.1)]

# Training settings - 140GB GPU optimized (actual usage: ~80GB)
# Training time: ~2h per epoch, ~12h for 5 epochs (incl. final testing)
batch_size = 20  # Batch size 20 uses ~80GB GPU memory
batch_size_val = 1  # MUST be 1 for correct inverse mapping in SemSegEvaluator
batch_size_test = 1
num_worker = 8
num_worker_test = 1
gradient_accumulation_steps = 4  # Effective batch = 80
mix_prob = 0.8
empty_cache = False  # OFFICIAL setting
empty_cache_per_epoch = False
enable_amp = True
amp_dtype = "bfloat16"

# Evaluation
evaluate = True
save_predictions = False  # Don't save predictions during training (saves ~5GB per experiment)

# Gradient clipping
clip_grad = 0.5

# Hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="GPUMemoryMonitor", log_per_step=False),  # Log GPU memory and epoch time
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
    dict(type="CheckpointSaver", save_freq=None),  # Only save model_best.pth and model_last.pth
    dict(type="PreciseEvaluator", test_last=False),
]

# Wandb configuration
wandb = dict(
    project="lidar-subsampling-benchmark",
    name="ptv3_semantickitti_Poisson_loss90_seed1_140gb",
    entity=None,
    tags=["semantickitti", "h200-140gb", "ptv3", "Poisson", "loss90"],
)
