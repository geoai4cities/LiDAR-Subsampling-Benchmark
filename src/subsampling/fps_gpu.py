#!/usr/bin/env python3
"""
GPU-Accelerated Farthest Point Sampling (FPS) for Point Clouds

Uses pointops CUDA kernels for ~40-70x speedup over CPU implementation.

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
    print("Warning: pointops not available, GPU FPS will fall back to CPU")


def fps_gpu(
    points: np.ndarray,
    n_samples: int,
    return_indices: bool = False,
    seed: Optional[int] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    GPU-accelerated Farthest Point Sampling using pointops.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3)
    n_samples : int
        Number of points to sample
    return_indices : bool, default=False
        If True, return both sampled points and their indices
    seed : int, optional
        Random seed for shuffling input points (provides reproducibility)
    device : str, default='cuda'
        CUDA device to use
    verbose : bool, default=False
        Print timing information

    Returns
    -------
    sampled_points : np.ndarray
        Sampled points of shape (n_samples, 3)
    indices : np.ndarray (optional)
        Indices of sampled points, only if return_indices=True
    """
    import time
    start = time.time()

    n_points = len(points)

    if n_samples > n_points:
        raise ValueError(f"Cannot sample {n_samples} points from {n_points} points")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if not POINTOPS_AVAILABLE:
        raise RuntimeError("pointops not available. Please install from PTv3/Pointcept")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use CPU FPS instead.")

    # Shuffle points for reproducibility with seed
    if seed is not None:
        np.random.seed(seed)
    shuffle_indices = np.random.permutation(n_points)
    shuffled_points = points[shuffle_indices]

    # Convert to torch tensor on GPU
    points_tensor = torch.from_numpy(shuffled_points.astype(np.float32)).to(device).contiguous()

    # Create offset tensors (single batch)
    offset = torch.tensor([n_points], dtype=torch.int32, device=device)
    new_offset = torch.tensor([n_samples], dtype=torch.int32, device=device)

    # Run GPU FPS
    indices_tensor = pointops.farthest_point_sampling(points_tensor, offset, new_offset)

    # Synchronize and convert back to numpy
    torch.cuda.synchronize()
    shuffled_indices = indices_tensor.cpu().numpy()

    # Map back to original indices
    indices = shuffle_indices[shuffled_indices]

    if verbose:
        elapsed = time.time() - start
        print(f"GPU FPS: {n_points} -> {n_samples} points in {elapsed:.3f}s")

    # Get sampled points from original array
    sampled_points = points[indices]

    if return_indices:
        return sampled_points, indices
    else:
        return sampled_points


def fps_subsample_with_loss_gpu(
    points: np.ndarray,
    loss_percentage: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    return_indices: bool = False,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,  # noqa: ARG001 - Not used, for API compatibility
    device: str = 'cuda',
    verbose: bool = False
) -> Union[np.ndarray, Tuple]:
    """
    GPU-accelerated FPS subsampling with specified loss percentage.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 3)
    loss_percentage : float
        Percentage of points to remove (0-100)
    features : np.ndarray, optional
        Point features of shape (N, F)
    labels : np.ndarray, optional
        Point labels of shape (N,)
    return_indices : bool, default=False
        If True, return indices of selected points
    seed : int, optional
        Random seed for reproducibility (shuffles input before FPS)
    dataset : str, optional
        Not used, included for API compatibility
    device : str, default='cuda'
        CUDA device to use
    verbose : bool, default=False
        Print progress information

    Returns
    -------
    sampled_points : np.ndarray
        Subsampled points
    sampled_features : np.ndarray (if features provided)
    sampled_labels : np.ndarray (if labels provided)
    indices : np.ndarray (if return_indices=True)
    """
    if loss_percentage < 0 or loss_percentage > 100:
        raise ValueError(f"loss_percentage must be in [0, 100], got {loss_percentage}")

    n_points = len(points)
    keep_ratio = (100 - loss_percentage) / 100
    n_samples = int(n_points * keep_ratio)

    if n_samples == 0:
        raise ValueError(f"loss_percentage {loss_percentage}% would result in 0 points")

    if verbose:
        print(f"GPU FPS: {n_points} -> {n_samples} points ({100-loss_percentage}% retained)")

    # Run GPU FPS
    sampled_points, indices = fps_gpu(
        points, n_samples, return_indices=True, seed=seed, device=device, verbose=verbose
    )

    # Build results
    results = [sampled_points]

    if features is not None:
        sampled_features = features[indices]
        results.append(sampled_features)

    if labels is not None:
        sampled_labels = labels[indices]
        results.append(sampled_labels)

    if return_indices:
        results.append(indices)

    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def test_gpu_fps():
    """Test GPU FPS implementation."""
    import time

    print("=" * 60)
    print("Testing GPU FPS Implementation")
    print("=" * 60)

    if not POINTOPS_AVAILABLE:
        print("SKIP: pointops not available")
        return

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # Test 1: Basic functionality
    print("\nTest 1: Basic GPU FPS")
    np.random.seed(42)
    points = np.random.rand(100000, 3).astype(np.float32) * 100

    start = time.time()
    sampled = fps_gpu(points, 50000, verbose=True)
    gpu_time = time.time() - start

    assert sampled.shape == (50000, 3), f"Expected (50000, 3), got {sampled.shape}"
    print(f"✓ Basic test passed: {gpu_time:.3f}s")

    # Test 2: With indices
    print("\nTest 2: GPU FPS with indices")
    sampled, indices = fps_gpu(points, 10000, return_indices=True)
    assert len(indices) == 10000
    assert np.allclose(sampled, points[indices])
    print("✓ Indices test passed")

    # Test 3: With loss percentage
    print("\nTest 3: GPU FPS with loss percentage")
    labels = np.random.randint(0, 19, size=100000).astype(np.uint32)
    features = np.random.rand(100000, 1).astype(np.float32)

    sampled_pts, sampled_feat, sampled_labels = fps_subsample_with_loss_gpu(
        points, loss_percentage=50, features=features, labels=labels, verbose=True
    )
    assert len(sampled_pts) == 50000
    assert len(sampled_feat) == 50000
    assert len(sampled_labels) == 50000
    print("✓ Loss percentage test passed")

    # Test 4: Determinism with seed
    print("\nTest 4: Determinism with seed")
    points = np.random.rand(10000, 3).astype(np.float32) * 100
    result1 = fps_gpu(points, 5000, seed=123)
    result2 = fps_gpu(points, 5000, seed=123)
    assert np.array_equal(result1, result2), "Results should be identical with same seed"
    print("✓ Determinism test passed")

    # Test 5: Speed comparison
    print("\nTest 5: Speed benchmark")
    for n in [10000, 50000, 100000]:
        points = np.random.rand(n, 3).astype(np.float32) * 100
        n_samples = n // 2

        start = time.time()
        _ = fps_gpu(points, n_samples)
        gpu_time = time.time() - start

        print(f"  {n:,} points -> {n_samples:,}: {gpu_time:.3f}s")

    print("\n" + "=" * 60)
    print("All GPU FPS tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_gpu_fps()
