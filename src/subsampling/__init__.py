"""
LiDAR Subsampling Methods - Unified Interface

This module provides a unified interface to all subsampling methods used in the
LiDAR Subsampling Benchmark project.

Available Methods:
1. RS (Random Sampling) - Baseline uniform random subsampling
2. IDIS (Inverse Distance Importance Sampling) - Our proposed method
3. FPS (Farthest Point Sampling) - Greedy spatial coverage
4. DBSCAN (Density-Based Clustering) - Density-aware subsampling
5. Voxel Grid - Grid-based downsampling (deterministic)
6. Poisson Disk - Space-based sampling with minimum distance
7. DEPOCO (Deep Point Cloud Compression) - Deep learning-based (reference only)

Usage Example:
    from subsampling import get_sampler, AVAILABLE_METHODS

    # Get a sampler by name
    sampler = get_sampler('IDIS', loss_percentage=50, seed=42)
    sampled_points = sampler(points)

    # Or use directly
    from subsampling import idis_subsample_with_loss
    sampled = idis_subsample_with_loss(points, loss_percentage=50, seed=42)

"""

# Import all sampling methods
from .random_sampling import (
    random_sampling,
    random_subsample_with_loss,
    random_batch_subsample,
    RandomSampling  # Legacy class API
)

from .idis import (
    idis_sampling,
    idis_subsample_with_loss,
    idis_batch_subsample,
    compute_importance_scores,
    IDIS  # Legacy class API
)

from .fps import (
    farthest_point_sampling,
    fps_subsample_with_loss,
    fps_batch_subsample
)

from .dbscan import (
    dbscan_subsampling,
    dbscan_subsample_with_loss,
    dbscan_batch_subsample,
    estimate_dbscan_eps
)

from .voxel_grid import (
    voxel_grid_subsampling,
    voxel_subsample_with_loss,
    voxel_batch_subsample
)

from .poisson_disk import (
    poisson_disk_sampling,
    poisson_subsample_with_loss,
    poisson_batch_subsample
)

# GPU-accelerated versions (optional - requires pointops and CUDA)
try:
    from .fps_gpu import (
        fps_gpu,
        fps_subsample_with_loss_gpu
    )
    GPU_FPS_AVAILABLE = True
except ImportError:
    GPU_FPS_AVAILABLE = False

try:
    from .idis_gpu import (
        idis_sampling_gpu,
        idis_subsample_with_loss_gpu,
        compute_importance_scores_gpu
    )
    GPU_IDIS_AVAILABLE = True
except ImportError:
    GPU_IDIS_AVAILABLE = False


# ============================================================================
# Unified Interface
# ============================================================================

AVAILABLE_METHODS = {
    'RS': {
        'name': 'Random Sampling',
        'function': random_subsample_with_loss,
        'description': 'Uniform random subsampling (baseline)',
        'parameters': {'seed': 42},
        'deterministic': True,  # With seed
        'complexity': 'O(N)',
        'characteristics': ['unbiased', 'fast', 'no spatial structure']
    },
    'IDIS': {
        'name': 'Inverse Distance Importance Sampling',
        'function': idis_subsample_with_loss,
        'description': 'Density-aware sampling favoring sparse regions (proposed method)',
        'parameters': {'radius': 10.0, 'distance_exponent': -2.0, 'seed': 42},  # R=10m default for SemanticKITTI
        'deterministic': True,  # With seed
        'complexity': 'O(N log N)',
        'characteristics': ['density-aware', 'preserves sparse regions', 'probabilistic']
    },
    'FPS': {
        'name': 'Farthest Point Sampling',
        'function': fps_subsample_with_loss,
        'description': 'Greedy algorithm for maximum spatial coverage',
        'parameters': {'seed': 42},
        'deterministic': False,  # Start point affects result
        'complexity': 'O(N × M × D)',
        'characteristics': ['uniform coverage', 'greedy', 'slow for large N']
    },
    'DBSCAN': {
        'name': 'DBSCAN Clustering',
        'function': dbscan_subsample_with_loss,
        'description': 'Density-based clustering with representative selection',
        'parameters': {'sampling_strategy': 'centroid', 'seed': 42, 'n_jobs': 1},
        'deterministic': True,  # With seed (for random strategy)
        'complexity': 'O(N log N)',
        'characteristics': ['density-based', 'clusters', 'noise-robust']
    },
    'Voxel': {
        'name': 'Voxel Grid Downsampling',
        'function': voxel_subsample_with_loss,
        'description': 'Grid-based downsampling with voxel centroids',
        'parameters': {'dataset': 'semantickitti'},  # or 'dales'
        'deterministic': True,
        'complexity': 'O(N)',
        'characteristics': ['deterministic', 'fast', 'uniform grid']
    },
    'Poisson': {
        'name': 'Poisson Disk Sampling',
        'function': poisson_subsample_with_loss,
        'description': 'Space-based sampling with minimum distance constraint',
        'parameters': {'dataset': 'semantickitti', 'seed': 42},  # or 'dales'
        'deterministic': True,  # With seed
        'complexity': 'O(N)',
        'characteristics': ['blue noise', 'minimum distance', 'uniform spatial']
    },
    'DEPOCO': {
        'name': 'Deep Point Cloud Compression',
        'function': None,  # Requires separate environment - use scripts/preprocessing/generate_subsampled_depoco.py
        'description': 'Deep learning encoder-decoder compression (Wiesmann et al. 2021)',
        'parameters': {
            'model_path': '${DEPOCO_BASE}/main-scripts/paper-1/network_files/',
            'device': 'cuda'
        },
        'deterministic': True,  # Given same model
        'complexity': 'O(N) - GPU inference',
        'characteristics': ['deep learning', 'geometric reconstruction', 'requires pre-trained model'],
        'status': 'Separate environment required - use generate_subsampled_depoco.py',
        'paper': 'https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf',
        'note': 'See configs/depoco/README.md for setup and usage instructions'
    }
}


def get_sampler(method_name, loss_percentage, **kwargs):
    """
    Get a sampler function with configured parameters.

    Args:
        method_name: Name of method ('RS', 'IDIS', 'FPS', 'DBSCAN', 'Voxel', 'Poisson')
        loss_percentage: Target loss percentage (0-100)
        **kwargs: Additional parameters to override defaults

    Returns:
        Callable sampler function: sampler(points, features=None, labels=None) -> results

    Example:
        >>> sampler = get_sampler('IDIS', loss_percentage=50, seed=42)
        >>> sampled_points = sampler(points)
        >>>
        >>> # With features and labels
        >>> sampled_points, sampled_features, sampled_labels = sampler(
        ...     points, features=intensity, labels=labels
        ... )
    """
    if method_name not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. Available methods: {list(AVAILABLE_METHODS.keys())}"
        )

    method_info = AVAILABLE_METHODS[method_name]
    subsample_func = method_info['function']

    # Handle reference-only methods (e.g., DEPOCO)
    if subsample_func is None:
        raise NotImplementedError(
            f"Method '{method_name}' is not implemented (reference only). "
            f"See docs/DEPOCO_REFERENCE.md for details."
        )

    # Merge default parameters with user-provided kwargs
    params = method_info['parameters'].copy()
    params.update(kwargs)

    # Create wrapper function
    def sampler(points, features=None, labels=None, return_indices=False, verbose=False):
        return subsample_func(
            points=points,
            loss_percentage=loss_percentage,
            features=features,
            labels=labels,
            return_indices=return_indices,
            verbose=verbose,
            **params
        )

    # Add metadata to function
    sampler.method_name = method_name
    sampler.method_full_name = method_info['name']
    sampler.loss_percentage = loss_percentage
    sampler.parameters = params

    return sampler


def list_available_methods():
    """
    Print information about all available subsampling methods.
    """
    print("="*80)
    print("AVAILABLE SUBSAMPLING METHODS")
    print("="*80)

    for method_code, info in AVAILABLE_METHODS.items():
        print(f"\n{method_code}: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Complexity: {info['complexity']}")
        print(f"  Deterministic: {info['deterministic']}")
        print(f"  Characteristics: {', '.join(info['characteristics'])}")
        print(f"  Default Parameters: {info['parameters']}")

    print("\n" + "="*80)


def subsample_with_method(
    points,
    method_name,
    loss_percentage,
    features=None,
    labels=None,
    seed=None,
    verbose=False,
    **kwargs
):
    """
    Convenience function to subsample with any method by name.

    Args:
        points: (N, 3) array of 3D coordinates
        method_name: Name of method ('RS', 'IDIS', 'FPS', 'DBSCAN', 'Voxel', 'Poisson')
        loss_percentage: Target loss percentage (0-100)
        features: Optional (N, F) array of features
        labels: Optional (N,) array of labels
        seed: Random seed (if applicable)
        verbose: Print progress
        **kwargs: Additional method-specific parameters

    Returns:
        Subsampled results (same format as individual methods)

    Example:
        >>> sampled = subsample_with_method(
        ...     points, 'IDIS', loss_percentage=50,
        ...     labels=labels, seed=42, radius=5.0
        ... )
    """
    sampler = get_sampler(method_name, loss_percentage, seed=seed, **kwargs)
    return sampler(points, features=features, labels=labels, verbose=verbose)


# ============================================================================
# Export all
# ============================================================================

__all__ = [
    # Random Sampling
    'random_sampling',
    'random_subsample_with_loss',
    'random_batch_subsample',
    'RandomSampling',

    # IDIS (CPU)
    'idis_sampling',
    'idis_subsample_with_loss',
    'idis_batch_subsample',
    'compute_importance_scores',
    'IDIS',

    # IDIS (GPU)
    'idis_sampling_gpu',
    'idis_subsample_with_loss_gpu',
    'compute_importance_scores_gpu',
    'GPU_IDIS_AVAILABLE',

    # FPS (CPU)
    'farthest_point_sampling',
    'fps_subsample_with_loss',
    'fps_batch_subsample',

    # FPS (GPU)
    'fps_gpu',
    'fps_subsample_with_loss_gpu',
    'GPU_FPS_AVAILABLE',

    # DBSCAN
    'dbscan_subsampling',
    'dbscan_subsample_with_loss',
    'dbscan_batch_subsample',
    'estimate_dbscan_eps',

    # Voxel Grid
    'voxel_grid_subsampling',
    'voxel_subsample_with_loss',
    'voxel_batch_subsample',

    # Poisson Disk
    'poisson_disk_sampling',
    'poisson_subsample_with_loss',
    'poisson_batch_subsample',

    # Unified Interface
    'AVAILABLE_METHODS',
    'get_sampler',
    'list_available_methods',
    'subsample_with_method',
]


if __name__ == '__main__':
    # Display available methods when run as script
    list_available_methods()

    print("\nUsage Example:")
    print("-" * 80)
    print("""
    from subsampling import get_sampler
    import numpy as np

    # Generate sample data
    points = np.random.rand(10000, 3) * 100

    # Get IDIS sampler with 50% loss
    sampler = get_sampler('IDIS', loss_percentage=50, seed=42)

    # Apply sampling
    sampled_points = sampler(points, verbose=True)

    print(f"Reduced from {len(points)} to {len(sampled_points)} points")
    """)
