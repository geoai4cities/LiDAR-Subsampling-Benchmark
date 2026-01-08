"""
Inverse Distance Importance Sampling (IDIS)

This implementation matches the manuscript description:
1. Shuffle points randomly
2. For each point, find all neighbors within ball radius R
3. Compute importance = sum(1/d²) for all neighbors (inverse-square weighting)
4. Sort by importance DESCENDING (higher sum = denser region = more important)
5. Select TOP N points deterministically

Algorithm behavior:
- Points in DENSE regions (many close neighbors) get HIGHER importance scores
- Points in SPARSE regions (few/distant neighbors) get LOWER importance scores
- Selects points from dense, feature-rich areas (edges, clusters, boundaries)

Formula: S_j = Σ(1/d_ij²) for all neighbors x_i within radius R

Key characteristics:
- Density-aware: Preserves points in dense/feature-rich regions
- Deterministic: TOP-K selection based on importance scores
- Configurable: Radius controls neighborhood size
- Seed-deterministic: Same seed produces same shuffle -> same result

This is the proposed method from the LiDAR Subsampling Benchmark paper.

"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Union, Tuple, Optional


def compute_importance_scores(
    points: np.ndarray,
    radius: float = 10.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute importance score for each point based on local density.

    This matches the manuscript description:
    - importance = sum(1/d²) for all neighbors within radius (inverse-square weighting)
    - Higher value = more close neighbors = denser region = MORE important

    Formula: S_j = Σ(1/d_ij²) for all neighbors x_i within radius R

    Args:
        points: (N, 3) array of 3D coordinates
        radius: Neighborhood radius for importance computation (meters)
        verbose: Print progress information

    Returns:
        importance_scores: (N,) array. Higher = denser region.
        Sum of inverse-square distances to neighbors.

    Algorithm:
        For each point p_j:
        1. Find all neighbors within radius R
        2. Compute distances d_ij to each neighbor
        3. Compute importance = sum(1/d_ij²)

    Higher importance = denser region (more close neighbors)
    Lower importance = sparser region (fewer/distant neighbors)
    """
    if verbose:
        print(f"Computing importance scores with radius={radius}")

    n_points = len(points)

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(points)

    # Compute importance for each point
    importance = np.zeros(n_points)

    for i in range(n_points):
        pt = points[i, :3]
        # Find all neighbors within radius
        idxs = tree.query_ball_point(pt, radius)

        # Compute distances to neighbors (excluding self)
        if len(idxs) > 1:
            neighbors = points[idxs]
            distances = np.linalg.norm(neighbors - pt, axis=1)
            distances = distances[distances > 1e-10]  # Remove self (distance ~0)

            if len(distances) > 0:
                # importance = sum(1/d²) matching manuscript description
                # This implements: S_j = Σ(1/d_ij²) → favors DENSE regions
                # Higher importance = more close neighbors = denser = selected first
                importance[i] = np.sum(1.0 / (distances ** 2))

    if verbose:
        print(f"Importance scores: min={importance.min():.4f}, max={importance.max():.4f}, "
              f"mean={importance.mean():.4f}")

    return importance


def idis_sampling(
    points: np.ndarray,
    n_samples: int,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    radius: float = 10.0,
    return_indices: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Inverse Distance Importance Sampling (IDIS).

    Matches the manuscript description:
    1. Shuffle points randomly
    2. Compute importance = sum(1/d²) for neighbors (inverse-square weighting)
    3. Sort by importance DESCENDING (higher = denser = selected first)
    4. Select TOP N points deterministically

    Args:
        points: (N, 3) array of 3D coordinates
        n_samples: Number of points to sample
        features: Optional (N, F) array of point features
        labels: Optional (N,) array of point labels
        radius: Neighborhood radius for importance computation (meters)
        return_indices: If True, return indices of selected points
        seed: Random seed for shuffling
        verbose: Print progress information

    Returns:
        If return_indices=False:
            - sampled_points: (n_samples, 3) array
            - sampled_features: (n_samples, F) if features provided
            - sampled_labels: (n_samples,) if labels provided
        If return_indices=True:
            - Also returns sampled_indices: (n_samples,) array

    Example:
        >>> points = np.random.rand(10000, 3) * 100
        >>> # Sample 5000 points, favoring dense regions
        >>> sampled = idis_sampling(points, n_samples=5000, radius=10.0, seed=42)
    """
    if points.shape[0] == 0:
        raise ValueError("Input points array is empty")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3) array, got shape {points.shape}")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if n_samples > len(points):
        raise ValueError(f"n_samples ({n_samples}) cannot exceed number of points ({len(points)})")

    # Shuffle points before computing importance (matching original algorithm)
    if seed is not None:
        np.random.seed(seed)

    # Create shuffle indices
    shuffle_indices = np.random.permutation(len(points))
    shuffled_points = points[shuffle_indices]

    if verbose:
        print(f"IDIS sampling: {len(points)} -> {n_samples} points")

    # Compute importance scores (average distance to neighbors)
    importance = compute_importance_scores(
        shuffled_points, radius=radius, verbose=verbose
    )

    # Sort by importance DESCENDING and take TOP N (deterministic selection)
    # Higher importance = more close neighbors = denser region = selected first
    sorted_indices = np.argsort(importance)[::-1]  # Descending order
    selected_shuffled_indices = sorted_indices[:n_samples]

    # Map back to original indices
    sampled_indices = shuffle_indices[selected_shuffled_indices]
    sampled_indices_sorted = np.sort(sampled_indices)  # Sort to maintain order

    # Extract sampled data
    sampled_points = points[sampled_indices_sorted]

    results = [sampled_points]

    if features is not None:
        sampled_features = features[sampled_indices_sorted]
        results.append(sampled_features)

    if labels is not None:
        sampled_labels = labels[sampled_indices_sorted]
        results.append(sampled_labels)

    if return_indices:
        results.append(sampled_indices_sorted)

    if verbose:
        compression_ratio = (1 - n_samples / len(points)) * 100
        print(f"Compression: {compression_ratio:.1f}%")

    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def idis_subsample_with_loss(
    points: np.ndarray,
    loss_percentage: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    radius: float = 10.0,
    seed: Optional[int] = None,
    dataset: str = None,  # noqa: ARG001 - For API compatibility (not used by IDIS)
    return_indices: bool = False,
    verbose: bool = False,
    **kwargs  # Accept but ignore extra parameters like distance_exponent
) -> Union[np.ndarray, Tuple]:
    """
    IDIS subsampling with target loss percentage.

    Matches the ORIGINAL algorithm (idws_sampler.py).

    Args:
        points: (N, 3) array of 3D coordinates
        loss_percentage: Percentage of points to remove (0-100)
        features: Optional (N, F) array of point features
        labels: Optional (N,) array of point labels
        radius: Neighborhood radius (meters)
        seed: Random seed for shuffling
        return_indices: If True, return indices of selected points
        verbose: Print progress information

    Returns:
        Same as idis_sampling()

    Example:
        >>> points = np.random.rand(10000, 3)
        >>> # Remove 50% of points using IDIS
        >>> sampled = idis_subsample_with_loss(points, loss_percentage=50, seed=42)
        >>> assert len(sampled) == 5000
    """
    if loss_percentage < 0 or loss_percentage > 100:
        raise ValueError(f"Loss percentage must be in [0, 100], got {loss_percentage}")

    # Calculate number of points to keep
    n_keep = int(len(points) * (1 - loss_percentage / 100))

    if n_keep == 0:
        raise ValueError(f"Loss percentage {loss_percentage}% would remove all points")

    if verbose:
        print(f"Target loss: {loss_percentage}% -> keeping {n_keep}/{len(points)} points")

    return idis_sampling(
        points=points,
        n_samples=n_keep,
        features=features,
        labels=labels,
        radius=radius,
        seed=seed,
        return_indices=return_indices,
        verbose=verbose
    )


def idis_batch_subsample(
    point_cloud_list: list,
    n_samples: int,
    features_list: Optional[list] = None,
    labels_list: Optional[list] = None,
    radius: float = 10.0,
    seed: Optional[int] = None,
    verbose: bool = False
) -> list:
    """
    Apply IDIS to a batch of point clouds.

    Args:
        point_cloud_list: List of (N_i, 3) point cloud arrays
        n_samples: Number of points to sample from each cloud
        features_list: Optional list of (N_i, F) feature arrays
        labels_list: Optional list of (N_i,) label arrays
        radius: Neighborhood radius (meters)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        List of subsampled results
    """
    results = []

    for i, points in enumerate(point_cloud_list):
        features = features_list[i] if features_list is not None else None
        labels = labels_list[i] if labels_list is not None else None

        if verbose:
            print(f"Processing cloud {i+1}/{len(point_cloud_list)}")

        # Use seed + i for deterministic but different results per cloud
        cloud_seed = (seed + i) if seed is not None else None

        result = idis_sampling(
            points=points,
            n_samples=n_samples,
            features=features,
            labels=labels,
            radius=radius,
            seed=cloud_seed,
            verbose=verbose
        )

        results.append(result)

    return results


# ============================================================================
# Legacy Class API (for backward compatibility)
# ============================================================================

class IDIS:
    """
    Inverse Distance Importance Sampling (legacy class API)

    For new code, use the functional API: idis_sampling() and idis_subsample_with_loss()
    """
    def __init__(self, target_ratio=0.5, radius=10.0, seed=42):
        """
        Args:
            target_ratio: Fraction of points to keep
            radius: Neighborhood radius (meters)
            seed: Random seed
        """
        self.target_ratio = target_ratio
        self.radius = radius
        self.seed = seed

    def compute_importance(self, points):
        """
        Compute importance score for each point

        Args:
            points: (N, 3) array

        Returns:
            importance_scores: (N,) array
        """
        return compute_importance_scores(points, radius=self.radius)

    def subsample(self, points, labels=None, features=None):
        """
        Subsample using IDIS

        Args:
            points: (N, 3) array
            labels: (N,) array (optional)
            features: (N, F) array (optional)

        Returns:
            subsampled_points, subsampled_labels, subsampled_features, indices
        """
        n_keep = int(len(points) * self.target_ratio)

        result = idis_sampling(
            points=points,
            n_samples=n_keep,
            features=features,
            labels=labels,
            radius=self.radius,
            seed=self.seed,
            return_indices=True
        )

        # Unpack based on what was provided
        if features is not None and labels is not None:
            subsampled_points, subsampled_features, subsampled_labels, indices = result
            return subsampled_points, subsampled_labels, subsampled_features, indices
        elif labels is not None:
            subsampled_points, subsampled_labels, indices = result
            return subsampled_points, subsampled_labels, None, indices
        elif features is not None:
            subsampled_points, subsampled_features, indices = result
            return subsampled_points, None, subsampled_features, indices
        else:
            subsampled_points, indices = result
            return subsampled_points, None, None, indices

    def __repr__(self):
        return f"IDIS(target_ratio={self.target_ratio}, radius={self.radius}, seed={self.seed})"


# ============================================================================
# Test Suite
# ============================================================================

def test_importance_computation():
    """Test importance score computation"""
    print("\n" + "="*70)
    print("Test 1: Importance score computation (inverse-square weighting)")
    print("="*70)

    # Create point cloud with known density structure
    # Dense cluster at origin, sparse points farther away
    np.random.seed(42)
    dense_points = np.random.rand(500, 3) * 5  # Dense cluster
    sparse_points = np.random.rand(100, 3) * 30 + 30  # Sparse points far away
    points = np.vstack([dense_points, sparse_points])

    importance = compute_importance_scores(points, radius=10.0, verbose=True)

    # Dense points should have higher importance (inverse-square weighting)
    dense_importance = importance[:500].mean()
    sparse_importance = importance[500:].mean()

    print(f"Dense region importance: {dense_importance:.4f}")
    print(f"Sparse region importance: {sparse_importance:.4f}")

    assert dense_importance > sparse_importance, "Dense points should have higher importance (inverse-square)"
    print("✓ Importance computation test passed")


def test_idis_basic():
    """Test basic IDIS sampling"""
    print("\n" + "="*70)
    print("Test 2: Basic IDIS sampling")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100

    n_samples = 5000
    sampled = idis_sampling(points, n_samples=n_samples, radius=10.0, seed=42, verbose=True)

    assert len(sampled) == n_samples
    assert sampled.shape[1] == 3

    print(f"✓ Basic test passed: {len(points)} -> {n_samples} points")


def test_idis_determinism():
    """Test that IDIS is deterministic with seed"""
    print("\n" + "="*70)
    print("Test 3: Determinism with seed")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(5000, 3) * 10

    result1 = idis_sampling(points, n_samples=2500, radius=5.0, seed=123)
    result2 = idis_sampling(points, n_samples=2500, radius=5.0, seed=123)

    assert np.array_equal(result1, result2), "Results should be identical with same seed"
    print("✓ Determinism test passed")


def test_dense_preservation():
    """Test that dense regions are preferentially preserved"""
    print("\n" + "="*70)
    print("Test 4: Dense region preservation")
    print("="*70)

    np.random.seed(42)
    dense_points = np.random.rand(500, 3) * 5  # Dense cluster
    sparse_points = np.random.rand(100, 3) * 30 + 30  # Sparse points
    points = np.vstack([dense_points, sparse_points])

    # Sample 50% = 300 points
    sampled, indices = idis_subsample_with_loss(
        points, loss_percentage=50, radius=10.0, seed=42,
        return_indices=True, verbose=True
    )

    dense_selected = np.sum(indices < 500)
    sparse_selected = np.sum(indices >= 500)

    print(f"Selected from dense: {dense_selected}/500 ({dense_selected/500*100:.1f}%)")
    print(f"Selected from sparse: {sparse_selected}/100 ({sparse_selected/100*100:.1f}%)")

    # Dense points should be preferentially selected with manuscript algorithm
    assert dense_selected > sparse_selected, f"Expected more dense points selected, got {dense_selected} vs {sparse_selected}"
    print("✓ Dense preservation test passed")


def test_idis_with_loss_percentage():
    """Test IDIS with loss percentage"""
    print("\n" + "="*70)
    print("Test 5: IDIS with loss percentage")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100

    for loss in [0, 10, 30, 50, 70, 90]:
        sampled = idis_subsample_with_loss(points, loss_percentage=loss, seed=42, verbose=False)

        expected_keep = int(len(points) * (1 - loss / 100))
        assert len(sampled) == expected_keep, f"Expected {expected_keep}, got {len(sampled)}"

        actual_loss = (1 - len(sampled) / len(points)) * 100
        print(f"  Target: {loss}%, Actual: {actual_loss:.1f}%")

    print("✓ Loss percentage test passed")


def test_legacy_class_api():
    """Test legacy class API"""
    print("\n" + "="*70)
    print("Test 6: Legacy class API")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10
    labels = np.random.randint(0, 8, size=1000)

    sampler = IDIS(target_ratio=0.5, radius=5.0, seed=42)
    subsampled_points, subsampled_labels, _, _ = sampler.subsample(points, labels=labels)

    assert len(subsampled_points) == 500
    assert len(subsampled_labels) == 500
    print("✓ Legacy class API test passed")


def run_all_tests():
    """Run all IDIS tests"""
    print("\n" + "="*80)
    print("IDIS (Inverse Distance Importance Sampling) - TEST SUITE")
    print("Manuscript Algorithm Implementation (inverse-square weighting)")
    print("="*80)

    try:
        test_importance_computation()
        test_idis_basic()
        test_idis_determinism()
        test_dense_preservation()
        test_idis_with_loss_percentage()
        test_legacy_class_api()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
