"""
Default runtime configuration
Based on Pointcept's runtime settings
"""

# Training settings
num_worker = 16
batch_size = 12
batch_size_val = 12
batch_size_test = 1
num_worker_test = 1
mix_prob = 0.0  # Probability of mixing samples (0.0 = disabled)
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients (1 = no accumulation)
clip_grad = 1.0  # Gradient clipping max norm (CRITICAL: prevents NaN/CUDA errors)

# Checkpoint and logging
weight = None  # Path to pretrained weights
resume = False
evaluate = False
test_only = False

# Output directory
save_path = "outputs"

# Distributed training
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"  # Mixed precision dtype: "bfloat16" (more stable) or "float16" (faster but can overflow)
find_unused_parameters = False  # Set to True if you have unused parameters in model

# Evaluation
eval_epoch = 1  # Evaluate after every epoch for close monitoring (50 epochs total)

# Random seed
seed = None

# Runtime hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# Wandb configuration (optional)
# SECURITY: Set wandb key via environment variable, NOT in code!
# Export WANDB_API_KEY in your shell or use wandb login
enable_wandb = False  # Set to True to enable Weights & Biases logging
wandb_project = "bhopal-mls-benchmark"  # Wandb project name
wandb = dict(
    project="bhopal-mls-benchmark",
    name=None,  # Will be set automatically from experiment name
    entity=None,  # Set your wandb username/team if needed
)

# Optimizer parameter groups (for different learning rates per layer)
param_dicts = None  # Set to list of dicts to use different LR for different params

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)
