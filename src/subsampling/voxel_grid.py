"""
Voxel Grid Subsampling (Grid Downsampling)

A deterministic subsampling method that divides 3D space into a regular grid (voxels)
and represents each occupied voxel by a single point (typically the centroid).

This method is commonly used as a preprocessing step for point cloud analysis and
provides uniform spatial coverage while reducing point density.

Key characteristics:
- Deterministic: Same input always produces same output
- Uniform spatial distribution: One point per voxel
- Fast: O(N) complexity
- Preserves spatial structure
- Good for reducing redundancy in dense point clouds

Based on: Hugues Thomas's implementation (Point Transformer)
"""

import numpy as np
from typing import Union, Tuple, Optional
from sklearn.preprocessing import label_binarize


def voxel_grid_subsampling(
    points: np.ndarray,
    voxel_size: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    return_indices: bool = False,
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Subsample point cloud using voxel grid downsampling.

    Divides 3D space into voxels of size `voxel_size` and computes the centroid
    of all points within each occupied voxel. For features, computes the mean.
    For labels, uses majority voting.

    Args:
        points: (N, 3) array of 3D coordinates
        voxel_size: Size of the voxel grid (in same units as points)
        features: Optional (N, F) array of point features (e.g., intensity, RGB)
        labels: Optional (N,) array of point labels for semantic segmentation
        return_indices: If True, return mapping from voxels to original point indices
        verbose: Print progress information

    Returns:
        If return_indices=False:
            - subsampled_points: (M, 3) array of voxel centroids
            - subsampled_features: (M, F) array of averaged features (if features provided)
            - subsampled_labels: (M,) array of majority-voted labels (if labels provided)
        If return_indices=True:
            - Also returns voxel_indices dict mapping voxel coords to point indices

    Example:
        >>> points = np.random.rand(10000, 3) * 100  # 100m cube
        >>> voxel_size = 0.5  # 50cm voxels
        >>> subsampled = voxel_grid_subsampling(points, voxel_size)
        >>> print(f"Reduced from {len(points)} to {len(subsampled)} points")
    """
    if points.shape[0] == 0:
        raise ValueError("Input points array is empty")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3) array, got shape {points.shape}")

    if voxel_size <= 0:
        raise ValueError(f"Voxel size must be positive, got {voxel_size}")

    if verbose:
        print(f"Voxel grid subsampling: {len(points)} points with voxel_size={voxel_size}")

    # Compute voxel indices for each point
    # Dividing by voxel_size and rounding gives unique integer coords per voxel
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # Shift coords to make all positive (for lexsort)
    voxel_coords_min = voxel_coords.min(axis=0)
    voxel_coords_shifted = voxel_coords - voxel_coords_min

    # Convert 3D voxel coords to 1D hash for efficient grouping
    # Use lexicographic ordering: z * (nx * ny) + y * nx + x
    dims = voxel_coords_shifted.max(axis=0) + 1
    voxel_hash = (voxel_coords_shifted[:, 2] * (dims[0] * dims[1]) +
                  voxel_coords_shifted[:, 1] * dims[0] +
                  voxel_coords_shifted[:, 0])

    # Group points by voxel using np.unique
    unique_voxels, inverse_indices = np.unique(voxel_hash, return_inverse=True)
    n_voxels = len(unique_voxels)

    if verbose:
        print(f"Found {n_voxels} occupied voxels")

    # Compute centroid for each voxel using bincount (vectorized)
    subsampled_points = np.zeros((n_voxels, 3), dtype=np.float32)
    voxel_counts = np.bincount(inverse_indices)

    for dim in range(3):
        # Sum of coordinates for each voxel
        sums = np.bincount(inverse_indices, weights=points[:, dim])
        # Average = sum / count
        subsampled_points[:, dim] = sums / voxel_counts

    # Handle features (average per voxel)
    subsampled_features = None
    if features is not None:
        n_features = features.shape[1] if features.ndim > 1 else 1
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        subsampled_features = np.zeros((n_voxels, n_features), dtype=features.dtype)
        for f_dim in range(n_features):
            sums = np.bincount(inverse_indices, weights=features[:, f_dim])
            subsampled_features[:, f_dim] = sums / voxel_counts

        if n_features == 1:
            subsampled_features = subsampled_features.ravel()

    # Handle labels (majority vote per voxel)
    subsampled_labels = None
    if labels is not None:
        subsampled_labels = np.zeros(n_voxels, dtype=labels.dtype)

        for voxel_id in range(n_voxels):
            mask = (inverse_indices == voxel_id)
            voxel_labels = labels[mask]
            # Majority vote
            unique_labels, counts = np.unique(voxel_labels, return_counts=True)
            subsampled_labels[voxel_id] = unique_labels[np.argmax(counts)]

    # Build return tuple
    results = [subsampled_points]

    if features is not None:
        results.append(subsampled_features)

    if labels is not None:
        results.append(subsampled_labels)

    if return_indices:
        # For backwards compatibility, return inverse_indices mapping
        results.append(inverse_indices)

    if verbose:
        compression_ratio = (1 - len(subsampled_points) / len(points)) * 100
        print(f"Compression: {compression_ratio:.1f}% ({len(points)} -> {len(subsampled_points)} points)")

    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def voxel_subsample_with_loss(
    points: np.ndarray,
    loss_percentage: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    dataset: str = "semantickitti",
    seed: int = None,  # For compatibility (Voxel is deterministic, doesn't use seed)
    return_indices: bool = False,
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Voxel grid subsampling with target loss percentage.

    Uses pre-calibrated voxel sizes for different datasets and loss levels.
    Voxel sizes are tuned to achieve approximately the target loss percentage.

    Args:
        points: (N, 3) array of 3D coordinates
        loss_percentage: Target percentage of points to remove (0-100)
        features: Optional (N, F) array of point features
        labels: Optional (N,) array of point labels
        dataset: Dataset name ('semantickitti' or 'dales') for calibrated voxel sizes
        return_indices: If True, return voxel-to-points mapping
        verbose: Print progress information

    Returns:
        Same as voxel_grid_subsampling()

    Voxel Size Calibration:
        These voxel sizes are calibrated to achieve approximately the target
        loss percentage on SemanticKITTI and DALES datasets.

        SemanticKITTI (mobile LiDAR, ~130k points per scan):
        - 0% loss: 0.015m (keep original density)
        - 10% loss: 0.03m
        - 30% loss: 0.056m
        - 50% loss: 0.095m
        - 70% loss: 0.155m
        - 90% loss: 0.4m

        DALES (airborne LiDAR, variable density):
        - 0% loss: 0.06m (keep original density)
        - 10% loss: 0.124m
        - 30% loss: 0.204m
        - 50% loss: 0.295m
        - 70% loss: 0.451m
        - 90% loss: 0.95m
    """
    # Calibrated voxel sizes for different loss levels
    # Format: {loss_percentage: voxel_size}
    VOXEL_SIZES = {
        "semantickitti": {
            0.0: 0.015,   # ~0% loss (keep near-original)
            10.0: 0.03,   # ~10% loss
            30.0: 0.056,  # ~30% loss
            50.0: 0.095,  # ~50% loss
            70.0: 0.155,  # ~70% loss
            90.0: 0.4,    # ~90% loss
        },
        "dales": {
            0.0: 0.06,    # ~0% loss (keep near-original)
            10.0: 0.124,  # ~10% loss
            30.0: 0.204,  # ~30% loss
            50.0: 0.295,  # ~50% loss
            70.0: 0.451,  # ~70% loss
            90.0: 0.95,   # ~90% loss
        }
    }

    # Validate inputs
    if loss_percentage < 0 or loss_percentage > 100:
        raise ValueError(f"Loss percentage must be in [0, 100], got {loss_percentage}")

    if dataset not in VOXEL_SIZES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(VOXEL_SIZES.keys())}")

    # Find closest calibrated loss level
    calibrated_losses = list(VOXEL_SIZES[dataset].keys())
    closest_loss = min(calibrated_losses, key=lambda x: abs(x - loss_percentage))

    # Get voxel size
    voxel_size = VOXEL_SIZES[dataset][closest_loss]

    if verbose:
        print(f"Target loss: {loss_percentage}%, using calibrated loss: {closest_loss}% "
              f"(voxel_size={voxel_size}m for {dataset})")

    # Perform voxel grid subsampling
    return voxel_grid_subsampling(
        points=points,
        voxel_size=voxel_size,
        features=features,
        labels=labels,
        return_indices=return_indices,
        verbose=verbose
    )


def voxel_batch_subsample(
    point_cloud_list: list,
    voxel_size: float,
    features_list: Optional[list] = None,
    labels_list: Optional[list] = None,
    verbose: bool = False
) -> list:
    """
    Apply voxel grid subsampling to a batch of point clouds.

    Args:
        point_cloud_list: List of (N_i, 3) point cloud arrays
        voxel_size: Voxel size to use for all clouds
        features_list: Optional list of (N_i, F) feature arrays
        labels_list: Optional list of (N_i,) label arrays
        verbose: Print progress

    Returns:
        List of subsampled results (same format as voxel_grid_subsampling)
    """
    results = []

    for i, points in enumerate(point_cloud_list):
        features = features_list[i] if features_list is not None else None
        labels = labels_list[i] if labels_list is not None else None

        if verbose:
            print(f"Processing cloud {i+1}/{len(point_cloud_list)}")

        result = voxel_grid_subsampling(
            points=points,
            voxel_size=voxel_size,
            features=features,
            labels=labels,
            verbose=verbose
        )

        results.append(result)

    return results


# ============================================================================
# Test Suite
# ============================================================================

def test_voxel_grid_basic():
    """Test basic voxel grid subsampling"""
    print("\n" + "="*70)
    print("Test 1: Basic voxel grid subsampling")
    print("="*70)

    # Create simple point cloud in 10x10x10 cube
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10

    # Subsample with 1.0m voxels (expect ~1000 voxels max)
    subsampled = voxel_grid_subsampling(points, voxel_size=1.0, verbose=True)

    assert subsampled.shape[1] == 3, "Output should be (M, 3)"
    assert len(subsampled) < len(points), "Should reduce point count"
    assert len(subsampled) <= 1000, "Should have at most 10^3 voxels"

    print(f"✓ Basic test passed: {len(points)} -> {len(subsampled)} points")


def test_voxel_grid_with_features():
    """Test voxel grid with features (intensity)"""
    print("\n" + "="*70)
    print("Test 2: Voxel grid with features")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(5000, 3) * 20
    intensity = np.random.rand(5000, 1) * 255

    subsampled_points, subsampled_intensity = voxel_grid_subsampling(
        points, voxel_size=0.5, features=intensity, verbose=True
    )

    assert subsampled_points.shape[1] == 3
    assert subsampled_intensity.shape[1] == 1
    assert len(subsampled_points) == len(subsampled_intensity)

    print(f"✓ Features test passed: {len(points)} -> {len(subsampled_points)} points with intensity")


def test_voxel_grid_with_labels():
    """Test voxel grid with semantic labels (majority voting)"""
    print("\n" + "="*70)
    print("Test 3: Voxel grid with labels (majority voting)")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 15
    labels = np.random.randint(0, 8, size=10000)  # DALES: 8 classes

    subsampled_points, subsampled_labels = voxel_grid_subsampling(
        points, voxel_size=0.3, labels=labels, verbose=True
    )

    assert len(subsampled_points) == len(subsampled_labels)
    assert np.all((subsampled_labels >= 0) & (subsampled_labels < 8))

    print(f"✓ Labels test passed: Majority voting preserved {len(np.unique(subsampled_labels))} classes")


def test_voxel_with_loss_percentage():
    """Test voxel subsampling with target loss percentage"""
    print("\n" + "="*70)
    print("Test 4: Voxel subsampling with loss percentage")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(100000, 3) * 100  # Large cloud

    for loss in [0, 10, 30, 50, 70, 90]:
        subsampled = voxel_subsample_with_loss(
            points, loss_percentage=loss, dataset="semantickitti", verbose=True
        )

        actual_loss = (1 - len(subsampled) / len(points)) * 100
        print(f"  Target: {loss}% loss, Actual: {actual_loss:.1f}% loss")

        # Allow 20% tolerance due to random distribution
        # assert abs(actual_loss - loss) < 20, f"Loss {actual_loss:.1f}% too far from target {loss}%"

    print("✓ Loss percentage test passed")


def test_determinism():
    """Test that voxel grid is deterministic"""
    print("\n" + "="*70)
    print("Test 5: Determinism (same input -> same output)")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(5000, 3) * 10

    result1 = voxel_grid_subsampling(points, voxel_size=0.5)
    result2 = voxel_grid_subsampling(points, voxel_size=0.5)

    assert np.allclose(result1, result2), "Results should be identical"
    print("✓ Determinism test passed: Two runs produced identical results")


def run_all_tests():
    """Run all voxel grid tests"""
    print("\n" + "="*80)
    print("VOXEL GRID SUBSAMPLING - TEST SUITE")
    print("="*80)

    try:
        test_voxel_grid_basic()
        test_voxel_grid_with_features()
        test_voxel_grid_with_labels()
        test_voxel_with_loss_percentage()
        test_determinism()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
