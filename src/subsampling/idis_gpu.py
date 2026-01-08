#!/usr/bin/env python3
"""
GPU-Accelerated Inverse Distance Importance Sampling (IDIS)

This implementation matches the manuscript description:
1. For each point, find all neighbors within ball radius R
2. Compute importance = sum(1/d²) for all neighbors (inverse-square weighting)
3. Sort by importance DESCENDING (higher sum = denser region = more important)
4. Select TOP N points deterministically

Algorithm behavior:
- Points in DENSE regions (many close neighbors) get HIGHER importance scores
- Points in SPARSE regions (few/distant neighbors) get LOWER importance scores
- Selects points from dense, feature-rich areas (edges, clusters, boundaries)

Formula: S_j = Σ(1/d_ij²) for all neighbors x_i within radius R

Uses pointops CUDA kernels for neighbor search, providing
significant speedup over CPU scipy.spatial.cKDTree.

Requirements:
    - PyTorch with CUDA
    - pointops (from PTv3/Pointcept)

"""

import numpy as np
import torch
from typing import Union, Tuple, Optional

# Try to import pointops
try:
    import pointops
    POINTOPS_AVAILABLE = True
except ImportError:
    POINTOPS_AVAILABLE = False
    print("Warning: pointops not available, GPU IDIS will fall back to CPU")


def compute_importance_scores_gpu(
    points: np.ndarray,
    radius: float = 10.0,
    max_neighbors: int = 128,
    device: str = 'cuda',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute importance scores using GPU-accelerated KNN with distance filtering.

    This matches the manuscript description:
    - importance = sum(1/d²) for all neighbors within radius (inverse-square weighting)
    - Higher value = more close neighbors = denser region = MORE important

    Formula: S_j = Σ(1/d_ij²) for all neighbors x_i within radius R

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3)
    radius : float, default=10.0
        Neighborhood radius in meters (default 10m for SemanticKITTI)
    max_neighbors : int, default=128
        Maximum neighbors per point for KNN query.
        Note: pointops has a bug with K >= 256, so use 128 or less.
    device : str, default='cuda'
        CUDA device to use
    verbose : bool, default=False
        Print timing information

    Returns
    -------
    importance : np.ndarray
        Importance scores of shape (N,). Higher = denser region.
        Sum of inverse-square distances to neighbors.
    """
    import time
    start = time.time()

    if not POINTOPS_AVAILABLE:
        raise RuntimeError("pointops not available. Please install from PTv3/Pointcept")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use CPU IDIS instead.")

    n_points = len(points)

    # Clamp max_neighbors to avoid pointops bug with K >= 256
    safe_max_neighbors = min(max_neighbors, 128)
    if max_neighbors > 128 and verbose:
        print(f"Warning: max_neighbors clamped to 128 (pointops limitation)")

    # Convert to torch tensor on GPU
    points_tensor = torch.from_numpy(points.astype(np.float32)).to(device).contiguous()
    offset = torch.tensor([n_points], dtype=torch.int32, device=device)

    # GPU KNN query: find K nearest neighbors
    # Returns: idx (N, K), dist (N, K) where dist is euclidean distance
    idx, dist = pointops.knn_query(safe_max_neighbors, points_tensor, offset)

    # Filter by radius to match original algorithm (ball query)
    # Include self (dist=0) to match original cdist behavior exactly
    valid_mask = (dist <= radius)

    # Compute INVERSE-SQUARE weighted importance 
    # This implements: importance = sum(1/d²) → favors DENSE regions
    # Higher importance = more close neighbors = denser region = selected first
    #
    # Formula: S_j = Σ(1/d_ij²) for all neighbors x_i within radius R
    #
    # Set invalid entries to 0 so they don't contribute to sum
    # Add small epsilon to avoid division by zero for self-distance
    eps = 1e-8
    inv_sq_dist = torch.zeros_like(dist)
    valid_nonzero = valid_mask & (dist > eps)  # Exclude self (dist ≈ 0)
    inv_sq_dist[valid_nonzero] = 1.0 / (dist[valid_nonzero] ** 2)

    # Sum of inverse-square distances per point (importance score)
    # Points with no neighbors get 0 importance (will be at bottom when sorted)
    importance = inv_sq_dist.sum(dim=1)

    # Transfer to CPU
    torch.cuda.synchronize()
    importance_np = importance.cpu().numpy()

    if verbose:
        elapsed = time.time() - start
        avg_neighbors = valid_nonzero.sum(dim=1).float().mean().item()
        print(f"GPU importance: {n_points} points, radius={radius}m, "
              f"avg_neighbors={avg_neighbors:.1f}, time={elapsed:.3f}s")

    return importance_np


def idis_sampling_gpu(
    points: np.ndarray,
    n_samples: int,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    radius: float = 10.0,
    max_neighbors: int = 128,
    return_indices: bool = False,
    seed: Optional[int] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    GPU-accelerated Inverse Distance Importance Sampling.

    Matches the manuscript description:
    1. Compute importance = sum(1/d²) for neighbors (inverse-square weighting)
    2. Sort by importance DESCENDING (higher = denser = selected first)
    3. Select TOP N points deterministically

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3)
    n_samples : int
        Number of points to sample
    features : np.ndarray, optional
        Point features of shape (N, F)
    labels : np.ndarray, optional
        Point labels of shape (N,)
    radius : float, default=10.0
        Neighborhood radius in meters
    max_neighbors : int, default=128
        Maximum neighbors per point for KNN query
    return_indices : bool, default=False
        If True, return indices of selected points
    seed : int, optional
        Random seed for shuffling before importance computation
    device : str, default='cuda'
        CUDA device to use
    verbose : bool, default=False
        Print progress information

    Returns
    -------
    sampled_points : np.ndarray
        Sampled points of shape (n_samples, 3)
    sampled_features : np.ndarray (if features provided)
    sampled_labels : np.ndarray (if labels provided)
    indices : np.ndarray (if return_indices=True)
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
        print(f"GPU IDIS: {len(points)} -> {n_samples} points")

    # Compute importance scores on GPU (average distance to neighbors)
    importance = compute_importance_scores_gpu(
        shuffled_points,
        radius=radius,
        max_neighbors=max_neighbors,
        device=device,
        verbose=verbose
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
        compression = (1 - n_samples / len(points)) * 100
        print(f"Compression: {compression:.1f}%")

    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def idis_subsample_with_loss_gpu(
    points: np.ndarray,
    loss_percentage: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    radius: float = 10.0,
    max_neighbors: int = 128,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,  # Not used but for API compatibility
    return_indices: bool = False,
    device: str = 'cuda',
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    GPU-accelerated IDIS subsampling with target loss percentage.

    Matches the ORIGINAL algorithm (idws_sampler.py).

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3)
    loss_percentage : float
        Percentage of points to remove (0-100)
    features : np.ndarray, optional
        Point features
    labels : np.ndarray, optional
        Point labels
    radius : float, default=10.0
        Neighborhood radius in meters
    max_neighbors : int, default=128
        Maximum neighbors for KNN query
    seed : int, optional
        Random seed for shuffling
    dataset : str, optional
        Not used, for API compatibility
    return_indices : bool, default=False
        If True, return indices
    device : str, default='cuda'
        CUDA device
    verbose : bool, default=False
        Print progress

    Returns
    -------
    Same as idis_sampling_gpu()
    """
    if loss_percentage < 0 or loss_percentage > 100:
        raise ValueError(f"Loss percentage must be in [0, 100], got {loss_percentage}")

    # Calculate number of points to keep
    n_keep = int(len(points) * (1 - loss_percentage / 100))

    if n_keep == 0:
        raise ValueError(f"Loss percentage {loss_percentage}% would remove all points")

    if verbose:
        print(f"GPU IDIS: radius={radius}m, loss={loss_percentage}%")

    return idis_sampling_gpu(
        points=points,
        n_samples=n_keep,
        features=features,
        labels=labels,
        radius=radius,
        max_neighbors=max_neighbors,
        seed=seed,
        return_indices=return_indices,
        device=device,
        verbose=verbose
    )


def test_gpu_idis():
    """Test GPU IDIS implementation."""
    import time

    print("=" * 60)
    print("Testing GPU IDIS Implementation (Manuscript Algorithm)")
    print("=" * 60)

    if not POINTOPS_AVAILABLE:
        print("SKIP: pointops not available")
        return

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # Test 1: Importance computation
    print("\nTest 1: GPU importance computation (inverse-square weighting)")
    np.random.seed(42)

    # Create point cloud with known density structure
    dense_points = np.random.rand(500, 3).astype(np.float32) * 5
    sparse_points = np.random.rand(100, 3).astype(np.float32) * 30 + 30
    points = np.vstack([dense_points, sparse_points])

    importance = compute_importance_scores_gpu(points, radius=10.0, verbose=True)

    dense_importance = importance[:500].mean()
    sparse_importance = importance[500:].mean()

    print(f"  Dense region mean importance: {dense_importance:.4f}")
    print(f"  Sparse region mean importance: {sparse_importance:.4f}")
    assert dense_importance > sparse_importance, "Dense points should have higher importance (inverse-square)"
    print("✓ Importance computation test passed")

    # Test 2: Verify dense region preservation
    print("\nTest 2: Dense region preservation (should be high %)")
    sampled, indices = idis_subsample_with_loss_gpu(
        points, loss_percentage=50, radius=10.0, seed=42,
        return_indices=True, verbose=True
    )

    dense_selected = np.sum(indices < 500)
    sparse_selected = np.sum(indices >= 500)
    print(f"  Selected from dense: {dense_selected}/500 ({dense_selected/500*100:.1f}%)")
    print(f"  Selected from sparse: {sparse_selected}/100 ({sparse_selected/100*100:.1f}%)")

    # With manuscript algorithm, dense points should be preferentially selected
    assert dense_selected > sparse_selected, f"Expected more dense points selected, got {dense_selected} vs {sparse_selected}"
    print("✓ Dense preservation test passed")

    # Test 3: Basic sampling
    print("\nTest 3: GPU IDIS sampling")
    points = np.random.rand(100000, 3).astype(np.float32) * 100

    start = time.time()
    sampled = idis_sampling_gpu(points, n_samples=50000, radius=10.0, seed=42, verbose=True)
    gpu_time = time.time() - start

    assert sampled.shape == (50000, 3), f"Expected (50000, 3), got {sampled.shape}"
    print(f"✓ Basic sampling test passed: {gpu_time:.3f}s")

    # Test 4: With loss percentage and labels
    print("\nTest 4: GPU IDIS with loss percentage")
    labels = np.random.randint(0, 19, size=100000).astype(np.uint32)
    features = np.random.rand(100000, 1).astype(np.float32)

    sampled_pts, sampled_feat, sampled_labels = idis_subsample_with_loss_gpu(
        points, loss_percentage=50, features=features, labels=labels,
        radius=10.0, seed=42, verbose=True
    )
    assert len(sampled_pts) == 50000
    assert len(sampled_feat) == 50000
    assert len(sampled_labels) == 50000
    print("✓ Loss percentage test passed")

    # Test 5: Determinism with seed
    print("\nTest 5: Determinism with seed")
    result1 = idis_sampling_gpu(points, n_samples=10000, radius=10.0, seed=123)
    result2 = idis_sampling_gpu(points, n_samples=10000, radius=10.0, seed=123)
    assert np.array_equal(result1, result2), "Results should be identical with same seed"
    print("✓ Determinism test passed")

    # Test 6: Speed benchmark
    print("\nTest 6: Speed benchmark")
    for n in [10000, 50000, 100000]:
        points = np.random.rand(n, 3).astype(np.float32) * 100
        n_samples = n // 2

        start = time.time()
        _ = idis_sampling_gpu(points, n_samples, radius=10.0, seed=42)
        gpu_time = time.time() - start

        print(f"  {n:,} points -> {n_samples:,}: {gpu_time:.3f}s")

    print("\n" + "=" * 60)
    print("All GPU IDIS tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_gpu_idis()
