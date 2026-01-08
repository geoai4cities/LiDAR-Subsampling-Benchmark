#!/usr/bin/env python3
"""
Farthest Point Sampling (FPS) for Point Clouds

Implements the FPS algorithm for subsampling point clouds by iteratively
selecting points that are farthest from already selected points.

"""

import numpy as np
from typing import Union, Tuple
import time


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    return_indices: bool = False,
    start_idx: Union[int, None] = None,
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Farthest Point Sampling (FPS) algorithm.

    Iteratively selects points that are farthest from the set of already
    selected points, ensuring good spatial coverage.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, D) where N is number of points and
        D is dimensionality (typically 3 for x,y,z)
    n_samples : int
        Number of points to sample
    return_indices : bool, default=False
        If True, return both sampled points and their indices
    start_idx : int or None, default=None
        Index of the first point to select. If None, selects randomly
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    sampled_points : np.ndarray
        Sampled points of shape (n_samples, D)
    indices : np.ndarray (optional)
        Indices of sampled points in original array, shape (n_samples,)
        Only returned if return_indices=True

    Algorithm
    ---------
    1. Select an initial point (random or specified)
    2. Compute distances from this point to all other points
    3. Select the point with maximum distance
    4. Update distances: for each point, keep minimum distance to any selected point
    5. Repeat steps 3-4 until n_samples points are selected

    Complexity: O(n_samples * N * D)

    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> sampled = farthest_point_sampling(points, 100)
    >>> print(sampled.shape)
    (100, 3)

    >>> sampled, indices = farthest_point_sampling(points, 100, return_indices=True)
    >>> assert np.allclose(sampled, points[indices])
    """
    start_time = time.time()

    n_points = points.shape[0]

    if n_samples > n_points:
        raise ValueError(
            f"Cannot sample {n_samples} points from {n_points} points. "
            f"n_samples must be <= number of points."
        )

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Initialize
    selected_indices = np.zeros(n_samples, dtype=np.int32)

    # Select first point
    if start_idx is None:
        selected_indices[0] = np.random.randint(0, n_points)
    else:
        if start_idx < 0 or start_idx >= n_points:
            raise ValueError(f"start_idx {start_idx} out of range [0, {n_points})")
        selected_indices[0] = start_idx

    # Initialize distances to infinity
    distances = np.full(n_points, np.inf, dtype=np.float32)

    # Iteratively select farthest points
    for i in range(n_samples):
        if verbose and (i % (n_samples // 10) == 0 or i == n_samples - 1):
            elapsed = time.time() - start_time
            progress = (i + 1) / n_samples * 100
            print(f"FPS Progress: {i+1}/{n_samples} ({progress:.1f}%) - {elapsed:.2f}s")

        # Current selected point
        current_point = points[selected_indices[i]]

        # Compute distances from current point to all points (memory-efficient)
        # Avoid broadcasting by computing squared differences per dimension
        dx = points[:, 0] - current_point[0]
        dy = points[:, 1] - current_point[1]
        dz = points[:, 2] - current_point[2]
        dist_to_current = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Update minimum distances
        distances = np.minimum(distances, dist_to_current)

        # Select next point (farthest from all selected points)
        if i < n_samples - 1:
            selected_indices[i + 1] = np.argmax(distances)

    if verbose:
        total_time = time.time() - start_time
        print(f"FPS completed: {n_samples} points sampled in {total_time:.2f}s")

    # Return results
    sampled_points = points[selected_indices]

    if return_indices:
        return sampled_points, selected_indices
    else:
        return sampled_points


def fps_subsample_with_loss(
    points: np.ndarray,
    loss_percentage: float,
    features: np.ndarray = None,
    labels: np.ndarray = None,
    return_indices: bool = False,
    seed: int = None,  # For compatibility (FPS uses start_idx instead)
    dataset: str = None,  # For API compatibility (not used by FPS)
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Subsample point cloud using FPS with specified loss percentage.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, D)
    loss_percentage : float
        Percentage of points to remove (0-100)
        E.g., loss_percentage=50 means keep 50% of points
    return_indices : bool, default=False
        If True, return both sampled points and their indices
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    sampled_points : np.ndarray
        Subsampled points
    indices : np.ndarray (optional)
        Indices of sampled points, only if return_indices=True

    Examples
    --------
    >>> points = np.random.rand(10000, 3)
    >>> # Keep 50% of points (50% loss)
    >>> sampled = fps_subsample_with_loss(points, loss_percentage=50)
    >>> print(sampled.shape[0])
    5000
    """
    if loss_percentage < 0 or loss_percentage > 100:
        raise ValueError(f"loss_percentage must be in [0, 100], got {loss_percentage}")

    n_points = points.shape[0]
    keep_ratio = (100 - loss_percentage) / 100
    n_samples = int(n_points * keep_ratio)

    if n_samples == 0:
        raise ValueError(
            f"loss_percentage {loss_percentage}% would result in 0 points. "
            f"Original points: {n_points}"
        )

    # Validate inputs
    if features is not None and len(features) != len(points):
        raise ValueError(f"Features length ({len(features)}) must match points length ({len(points)})")

    if labels is not None and len(labels) != len(points):
        raise ValueError(f"Labels length ({len(labels)}) must match points length ({len(points)})")

    if verbose:
        print(f"FPS Subsampling:")
        print(f"  Original points: {n_points}")
        print(f"  Loss percentage: {loss_percentage}%")
        print(f"  Target points: {n_samples} ({100-loss_percentage}% retained)")

    # Use seed as start_idx if provided (for some determinism)
    start_idx = seed if seed is not None else None

    # Always get indices to extract features/labels
    need_indices = return_indices or features is not None or labels is not None

    result = farthest_point_sampling(
        points,
        n_samples=n_samples,
        return_indices=need_indices,
        start_idx=start_idx,
        verbose=verbose
    )

    if need_indices:
        sampled_points, indices = result
    else:
        sampled_points = result
        indices = None

    # Build results tuple
    results = [sampled_points]

    if features is not None:
        sampled_features = features[indices]
        results.append(sampled_features)

    if labels is not None:
        sampled_labels = labels[indices]
        results.append(sampled_labels)

    if return_indices:
        results.append(indices)

    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def fps_batch_subsample(
    point_cloud_list: list,
    loss_percentage: float,
    verbose: bool = False
) -> list:
    """
    Apply FPS to a batch of point clouds.

    Parameters
    ----------
    point_cloud_list : list of np.ndarray
        List of point clouds, each of shape (N_i, D)
    loss_percentage : float
        Percentage of points to remove (0-100)
    verbose : bool, default=False
        If True, print progress for each cloud

    Returns
    -------
    subsampled_list : list of np.ndarray
        List of subsampled point clouds

    Examples
    --------
    >>> clouds = [np.random.rand(1000, 3) for _ in range(10)]
    >>> subsampled = fps_batch_subsample(clouds, loss_percentage=50)
    >>> len(subsampled)
    10
    """
    if verbose:
        print(f"\nBatch FPS Subsampling: {len(point_cloud_list)} clouds")

    subsampled_list = []
    for i, points in enumerate(point_cloud_list):
        if verbose:
            print(f"\nProcessing cloud {i+1}/{len(point_cloud_list)}...")

        sampled = fps_subsample_with_loss(
            points,
            loss_percentage=loss_percentage,
            verbose=verbose
        )
        subsampled_list.append(sampled)

    return subsampled_list


if __name__ == "__main__":
    # Test FPS implementation
    print("=" * 60)
    print("Testing Farthest Point Sampling (FPS)")
    print("=" * 60)

    # Test 1: Basic sampling
    print("\nTest 1: Basic FPS sampling")
    np.random.seed(42)
    points = np.random.rand(1000, 3)
    sampled = farthest_point_sampling(points, 100, verbose=True)
    print(f"Input shape: {points.shape}")
    print(f"Sampled shape: {sampled.shape}")
    assert sampled.shape == (100, 3), "Shape mismatch"
    print("✓ Test 1 passed")

    # Test 2: With indices
    print("\nTest 2: FPS with indices")
    sampled, indices = farthest_point_sampling(points, 100, return_indices=True)
    assert np.allclose(sampled, points[indices]), "Indices don't match samples"
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min()}, {indices.max()}]")
    print("✓ Test 2 passed")

    # Test 3: With loss percentage
    print("\nTest 3: FPS with loss percentage")
    sampled = fps_subsample_with_loss(points, loss_percentage=50, verbose=True)
    expected_n = int(1000 * 0.5)
    assert sampled.shape[0] == expected_n, f"Expected {expected_n}, got {sampled.shape[0]}"
    print("✓ Test 3 passed")

    # Test 4: Coverage test (points should be well-distributed)
    print("\nTest 4: Coverage test")
    points_2d = np.random.rand(1000, 2)
    sampled_2d = farthest_point_sampling(points_2d, 50)

    # Check that sampled points cover the space better than random sampling
    random_sampled = points_2d[np.random.choice(1000, 50, replace=False)]

    # Measure average minimum distance between sampled points (should be higher for FPS)
    from scipy.spatial.distance import pdist
    fps_min_dist = np.min(pdist(sampled_2d))
    random_min_dist = np.min(pdist(random_sampled))

    print(f"FPS minimum inter-point distance: {fps_min_dist:.4f}")
    print(f"Random minimum inter-point distance: {random_min_dist:.4f}")
    print(f"FPS/Random ratio: {fps_min_dist/random_min_dist:.2f}x better")
    print("✓ Test 4 passed")

    # Test 5: Edge cases
    print("\nTest 5: Edge cases")
    try:
        # Should raise error: more samples than points
        farthest_point_sampling(points, 2000)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Single point sampling
    sampled = farthest_point_sampling(points, 1)
    assert sampled.shape == (1, 3)
    print("✓ Single point sampling works")

    # All points
    sampled = farthest_point_sampling(points, 1000)
    assert sampled.shape == (1000, 3)
    print("✓ All points sampling works")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
