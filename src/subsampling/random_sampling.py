"""
Random Sampling (RS)

The simplest subsampling method that randomly selects a subset of points
without any spatial or semantic considerations.

Key characteristics:
- Unbiased: Each point has equal probability of selection
- Fast: O(N) complexity
- Seed-deterministic: Same seed produces same result
- No spatial structure preservation
- Can create spatial gaps and clusters

This serves as a baseline method for comparison with more sophisticated
sampling techniques.

"""

import numpy as np
from typing import Union, Tuple, Optional


def random_sampling(
    points: np.ndarray,
    n_samples: int,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    return_indices: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Random subsampling of point cloud.

    Randomly selects n_samples points from the input cloud with uniform
    probability (each point has equal chance of selection).

    Args:
        points: (N, 3) array of 3D coordinates
        n_samples: Number of points to sample
        features: Optional (N, F) array of point features
        labels: Optional (N,) array of point labels
        return_indices: If True, return indices of selected points
        seed: Random seed for reproducibility
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
        >>> sampled = random_sampling(points, n_samples=5000, seed=42)
        >>> print(f"Sampled {len(sampled)}/{len(points)} points randomly")
    """
    if points.shape[0] == 0:
        raise ValueError("Input points array is empty")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3) array, got shape {points.shape}")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if n_samples > len(points):
        raise ValueError(f"n_samples ({n_samples}) cannot exceed number of points ({len(points)})")

    if seed is not None:
        np.random.seed(seed)

    if verbose:
        print(f"Random sampling: {len(points)} -> {n_samples} points")

    # Randomly select indices
    sampled_indices = np.random.choice(len(points), size=n_samples, replace=False)
    sampled_indices.sort()  # Sort to maintain some order

    # Extract sampled data
    sampled_points = points[sampled_indices]

    results = [sampled_points]

    if features is not None:
        sampled_features = features[sampled_indices]
        results.append(sampled_features)

    if labels is not None:
        sampled_labels = labels[sampled_indices]
        results.append(sampled_labels)

    if return_indices:
        results.append(sampled_indices)

    if verbose:
        compression_ratio = (1 - n_samples / len(points)) * 100
        print(f"Compression: {compression_ratio:.1f}%")

    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def random_subsample_with_loss(
    points: np.ndarray,
    loss_percentage: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    dataset: str = None,  # For API compatibility (not used by RS)
    return_indices: bool = False,
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    Random subsampling with target loss percentage.

    Args:
        points: (N, 3) array of 3D coordinates
        loss_percentage: Percentage of points to remove (0-100)
        features: Optional (N, F) array of point features
        labels: Optional (N,) array of point labels
        seed: Random seed for reproducibility
        dataset: Dataset name (for API compatibility, not used by RS)
        return_indices: If True, return indices of selected points
        verbose: Print progress information

    Returns:
        Same as random_sampling()

    Example:
        >>> points = np.random.rand(10000, 3)
        >>> # Remove 50% of points
        >>> sampled = random_subsample_with_loss(points, loss_percentage=50, seed=42)
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

    return random_sampling(
        points=points,
        n_samples=n_keep,
        features=features,
        labels=labels,
        seed=seed,
        return_indices=return_indices,
        verbose=verbose
    )


def random_batch_subsample(
    point_cloud_list: list,
    n_samples: int,
    features_list: Optional[list] = None,
    labels_list: Optional[list] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> list:
    """
    Apply random sampling to a batch of point clouds.

    Args:
        point_cloud_list: List of (N_i, 3) point cloud arrays
        n_samples: Number of points to sample from each cloud
        features_list: Optional list of (N_i, F) feature arrays
        labels_list: Optional list of (N_i,) label arrays
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

        result = random_sampling(
            points=points,
            n_samples=n_samples,
            features=features,
            labels=labels,
            seed=cloud_seed,
            verbose=verbose
        )

        results.append(result)

    return results


# ============================================================================
# Legacy Class API (for backward compatibility)
# ============================================================================

class RandomSampling:
    """
    Random sampling of point clouds (legacy class API)

    For new code, use the functional API: random_sampling() and random_subsample_with_loss()
    """
    def __init__(self, target_ratio=0.5, seed=42):
        """
        Args:
            target_ratio: Fraction of points to keep (0-1)
            seed: Random seed for reproducibility
        """
        self.target_ratio = target_ratio
        self.seed = seed

    def subsample(self, points, labels=None, features=None):
        """
        Subsample point cloud randomly

        Args:
            points: (N, 3) array of point coordinates
            labels: (N,) array of labels (optional)
            features: (N, D) array of features (optional)

        Returns:
            subsampled_points, subsampled_labels, subsampled_features, indices
        """
        n_keep = int(len(points) * self.target_ratio)

        result = random_sampling(
            points=points,
            n_samples=n_keep,
            features=features,
            labels=labels,
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
        return f"RandomSampling(target_ratio={self.target_ratio}, seed={self.seed})"


# ============================================================================
# Test Suite
# ============================================================================

def test_random_basic():
    """Test basic random sampling"""
    print("\n" + "="*70)
    print("Test 1: Basic random sampling")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100

    n_samples = 5000
    sampled = random_sampling(points, n_samples=n_samples, seed=42, verbose=True)

    assert len(sampled) == n_samples
    assert sampled.shape[1] == 3

    print(f"✓ Basic test passed: {len(points)} -> {n_samples} points")


def test_random_with_features():
    """Test random sampling with features"""
    print("\n" + "="*70)
    print("Test 2: Random sampling with features and labels")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100
    intensity = np.random.rand(10000, 1) * 255
    labels = np.random.randint(0, 19, size=10000)  # SemanticKITTI: 19 classes

    sampled_points, sampled_intensity, sampled_labels = random_sampling(
        points, n_samples=5000, features=intensity, labels=labels, seed=42, verbose=True
    )

    assert len(sampled_points) == 5000
    assert len(sampled_intensity) == 5000
    assert len(sampled_labels) == 5000

    print(f"✓ Features test passed")


def test_random_determinism():
    """Test that random sampling is deterministic with seed"""
    print("\n" + "="*70)
    print("Test 3: Determinism with seed")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(5000, 3) * 10

    result1 = random_sampling(points, n_samples=2500, seed=123)
    result2 = random_sampling(points, n_samples=2500, seed=123)

    assert np.array_equal(result1, result2), "Results should be identical with same seed"
    print("✓ Determinism test passed")


def test_random_with_loss_percentage():
    """Test random sampling with loss percentage"""
    print("\n" + "="*70)
    print("Test 4: Random sampling with loss percentage")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(10000, 3) * 100

    for loss in [0, 10, 30, 50, 70, 90]:
        sampled = random_subsample_with_loss(points, loss_percentage=loss, seed=42, verbose=True)

        expected_keep = int(len(points) * (1 - loss / 100))
        assert len(sampled) == expected_keep, f"Expected {expected_keep}, got {len(sampled)}"

        actual_loss = (1 - len(sampled) / len(points)) * 100
        print(f"  Target: {loss}%, Actual: {actual_loss:.1f}%")

    print("✓ Loss percentage test passed")


def test_return_indices():
    """Test return_indices parameter"""
    print("\n" + "="*70)
    print("Test 5: Return indices")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10

    sampled_points, indices = random_sampling(
        points, n_samples=500, return_indices=True, seed=42, verbose=True
    )

    # Verify indices are correct
    assert np.array_equal(sampled_points, points[indices])
    print("✓ Return indices test passed")


def test_legacy_class_api():
    """Test legacy class API for backward compatibility"""
    print("\n" + "="*70)
    print("Test 6: Legacy class API")
    print("="*70)

    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10
    labels = np.random.randint(0, 8, size=1000)

    sampler = RandomSampling(target_ratio=0.5, seed=42)
    subsampled_points, subsampled_labels, _, indices = sampler.subsample(points, labels=labels)

    assert len(subsampled_points) == 500
    assert len(subsampled_labels) == 500
    print("✓ Legacy class API test passed")


def run_all_tests():
    """Run all random sampling tests"""
    print("\n" + "="*80)
    print("RANDOM SAMPLING - TEST SUITE")
    print("="*80)

    try:
        test_random_basic()
        test_random_with_features()
        test_random_determinism()
        test_random_with_loss_percentage()
        test_return_indices()
        test_legacy_class_api()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
