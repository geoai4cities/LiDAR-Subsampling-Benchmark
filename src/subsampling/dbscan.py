#!/usr/bin/env python3
"""
DBSCAN-based Subsampling for Point Clouds

Implements density-based clustering (DBSCAN) to subsample point clouds by
selecting representative points from each cluster.

"""

import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN
from typing import Union, Tuple
import time


def dbscan_subsampling(
    points: np.ndarray,
    n_samples: int,
    eps: float = 0.5,
    min_samples: int = 5,
    features: np.ndarray = None,
    labels: np.ndarray = None,
    return_indices: bool = False,
    sampling_strategy: str = "centroid",
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    DBSCAN-based subsampling of point clouds.

    Uses DBSCAN clustering to group points, then selects representatives from
    each cluster. Handles noise points separately.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, D) where N is number of points and
        D is dimensionality (typically 3 for x,y,z)
    n_samples : int
        Target number of points to sample
    eps : float, default=0.5
        DBSCAN epsilon parameter - maximum distance between points in a cluster
        Smaller eps creates more, smaller clusters
    min_samples : int, default=5
        DBSCAN min_samples parameter - minimum points to form a cluster
    features : np.ndarray, optional
        Input features of shape (N, F) where F is number of features
    labels : np.ndarray, optional
        Input labels of shape (N,)
    return_indices : bool, default=False
        If True, return both sampled points and their indices
    sampling_strategy : str, default="centroid"
        Strategy for selecting points from clusters:
        - "centroid": Select point closest to cluster centroid
        - "random": Random point from each cluster
        - "density": Select highest density point (closest to most neighbors)
    n_jobs : int, default=1
        Number of parallel jobs for DBSCAN clustering
        Use 1 for multiprocessing contexts to avoid oversubscription
        Use -1 to use all available cores (only when not in multiprocessing)
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    sampled_points : np.ndarray
        Sampled points of shape (n_samples, D) or fewer if clusters < n_samples
    indices : np.ndarray (optional)
        Indices of sampled points in original array
        Only returned if return_indices=True

    Algorithm
    ---------
    1. Run DBSCAN clustering on input points
    2. Identify clusters and noise points
    3. Allocate sampling budget: prioritize clusters by size
    4. Sample from each cluster according to strategy
    5. If needed, add random noise points to reach n_samples

    Notes
    -----
    - DBSCAN parameters (eps, min_samples) greatly affect clustering quality
    - For dense point clouds, smaller eps recommended (e.g., 0.1-0.3)
    - For sparse point clouds, larger eps recommended (e.g., 0.5-1.0)
    - Actual sampled points may be < n_samples if too few clusters

    Examples
    --------
    >>> points = np.random.rand(10000, 3)
    >>> sampled = dbscan_subsampling(points, 1000, eps=0.2, min_samples=5)
    >>> print(sampled.shape[0])
    1000  # or fewer if insufficient clusters
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

    if sampling_strategy not in ["centroid", "random", "density"]:
        raise ValueError(
            f"Invalid sampling_strategy: {sampling_strategy}. "
            f"Must be one of: centroid, random, density"
        )

    if verbose:
        print(f"DBSCAN Subsampling:")
        print(f"  Input points: {n_points}")
        print(f"  Target samples: {n_samples}")
        print(f"  eps={eps}, min_samples={min_samples}")
        print(f"  Strategy: {sampling_strategy}")

    # Validate inputs
    if features is not None and len(features) != len(points):
        raise ValueError(f"Features length ({len(features)}) must match points length ({len(points)})")

    if labels is not None and len(labels) != len(points):
        raise ValueError(f"Labels length ({len(labels)}) must match points length ({len(points)})")

    # Run DBSCAN clustering
    if verbose:
        print(f"  Running DBSCAN clustering...")

    dbscan = SKDBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    cluster_labels = dbscan.fit_predict(points)

    # Analyze clusters
    unique_cluster_labels = np.unique(cluster_labels)
    n_clusters = len(unique_cluster_labels[unique_cluster_labels >= 0])  # Exclude noise (-1)
    n_noise = np.sum(cluster_labels == -1)

    if verbose:
        print(f"  Found {n_clusters} clusters and {n_noise} noise points")

    # Handle edge case: no clusters found
    if n_clusters == 0:
        if verbose:
            print(f"  Warning: No clusters found, using random sampling")
        indices = np.random.choice(n_points, size=min(n_samples, n_points), replace=False)
        sampled_points = points[indices]

        # Build results tuple (same as normal path)
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

    # Calculate cluster sizes (excluding noise)
    valid_cluster_ids = unique_cluster_labels[unique_cluster_labels >= 0]
    cluster_sizes = np.array([np.sum(cluster_labels == cid) for cid in valid_cluster_ids])

    # Allocate samples to clusters proportionally to their size
    # Reserve some samples for noise points if they exist
    if n_noise > 0:
        n_samples_clusters = int(n_samples * 0.9)  # 90% to clusters
        n_samples_noise = n_samples - n_samples_clusters  # 10% to noise
    else:
        n_samples_clusters = n_samples
        n_samples_noise = 0

    # Distribute samples across clusters proportionally
    cluster_sample_counts = np.round(
        (cluster_sizes / cluster_sizes.sum()) * n_samples_clusters
    ).astype(int)

    # Ensure we have at least 1 sample per cluster (if budget allows)
    cluster_sample_counts = np.maximum(cluster_sample_counts, 1)

    # Adjust if total exceeds budget
    while cluster_sample_counts.sum() > n_samples_clusters:
        # Reduce from largest clusters
        max_idx = np.argmax(cluster_sample_counts)
        cluster_sample_counts[max_idx] -= 1

    # Adjust if total is less than budget (distribute remainder)
    remainder = n_samples_clusters - cluster_sample_counts.sum()
    if remainder > 0:
        # Give to largest clusters first
        largest_clusters = np.argsort(cluster_sizes)[::-1][:remainder]
        cluster_sample_counts[largest_clusters] += 1

    if verbose:
        print(f"  Sampling from clusters:")
        for cid, size, n_sample in zip(valid_cluster_ids, cluster_sizes, cluster_sample_counts):
            print(f"    Cluster {cid}: {size} points → {n_sample} samples")

    # Sample from each cluster
    selected_indices = []

    for cid, n_sample in zip(valid_cluster_ids, cluster_sample_counts):
        if n_sample == 0:
            continue

        # Get points in this cluster
        cluster_mask = cluster_labels == cid
        cluster_points = points[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        if n_sample >= len(cluster_indices):
            # Take all points from this cluster
            selected_indices.extend(cluster_indices)
        else:
            # Sample according to strategy
            if sampling_strategy == "random":
                # Random sampling
                sample_idx = np.random.choice(len(cluster_indices), size=n_sample, replace=False)
                selected_indices.extend(cluster_indices[sample_idx])

            elif sampling_strategy == "centroid":
                # Select points closest to cluster centroid
                centroid = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx = np.argsort(distances)[:n_sample]
                selected_indices.extend(cluster_indices[closest_idx])

            elif sampling_strategy == "density":
                # Select points with highest local density
                # (closest to most neighbors within cluster)
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(10, len(cluster_points))).fit(cluster_points)
                distances, _ = nbrs.kneighbors(cluster_points)
                # Lower mean distance = higher density
                density = 1.0 / (distances.mean(axis=1) + 1e-6)
                highest_density_idx = np.argsort(density)[::-1][:n_sample]
                selected_indices.extend(cluster_indices[highest_density_idx])

    # Add noise points if needed
    if n_noise > 0 and n_samples_noise > 0:
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0]
        if len(noise_indices) > 0:
            n_noise_sample = min(n_samples_noise, len(noise_indices))
            noise_sample_idx = np.random.choice(noise_indices, size=n_noise_sample, replace=False)
            selected_indices.extend(noise_sample_idx)
            if verbose:
                print(f"  Added {n_noise_sample} noise points")

    # Convert to numpy array
    selected_indices = np.array(selected_indices, dtype=np.int32)

    # Handle case where we got fewer samples than requested
    actual_samples = len(selected_indices)
    if actual_samples < n_samples:
        if verbose:
            print(f"  Warning: Only {actual_samples}/{n_samples} samples obtained")
            print(f"  Consider adjusting eps or min_samples parameters")

    # Get sampled points
    sampled_points = points[selected_indices]

    # Build results tuple
    results = [sampled_points]

    if features is not None:
        sampled_features = features[selected_indices]
        results.append(sampled_features)

    if labels is not None:
        sampled_labels = labels[selected_indices]
        results.append(sampled_labels)

    if return_indices:
        results.append(selected_indices)

    total_time = time.time() - start_time
    if verbose:
        print(f"  DBSCAN subsampling completed in {total_time:.2f}s")
        print(f"  Output: {len(sampled_points)} points")

    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def dbscan_subsample_with_loss(
    points: np.ndarray,
    loss_percentage: float,
    eps: float = 0.5,
    min_samples: int = 5,
    features: np.ndarray = None,
    labels: np.ndarray = None,
    sampling_strategy: str = "centroid",
    strategy: str = None,  # Alias for sampling_strategy (backwards compat)
    seed: int = None,  # For compatibility with unified interface
    dataset: str = None,  # For API compatibility (not used by DBSCAN)
    n_jobs: int = 1,
    return_indices: bool = False,
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Subsample point cloud using DBSCAN with specified loss percentage.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, D)
    loss_percentage : float
        Percentage of points to remove (0-100)
        E.g., loss_percentage=50 means keep 50% of points
    eps : float, default=0.5
        DBSCAN epsilon parameter
    min_samples : int, default=5
        DBSCAN min_samples parameter
    sampling_strategy : str, default="centroid"
        Strategy for selecting points ("centroid", "random", "density")
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
    >>> # Keep 50% of points (50% loss) using DBSCAN
    >>> sampled = dbscan_subsample_with_loss(points, loss_percentage=50, eps=0.3)
    >>> print(sampled.shape[0])
    ~5000  # May vary based on clustering
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

    # Handle backwards compatibility: strategy is alias for sampling_strategy
    if strategy is not None:
        sampling_strategy = strategy

    # Set random seed if provided (for reproducible random strategy)
    if seed is not None and sampling_strategy == "random":
        np.random.seed(seed)

    if verbose:
        print(f"DBSCAN Subsampling:")
        print(f"  Original points: {n_points}")
        print(f"  Loss percentage: {loss_percentage}%")
        print(f"  Target points: {n_samples} ({100-loss_percentage}% retained)")

    return dbscan_subsampling(
        points,
        n_samples=n_samples,
        eps=eps,
        min_samples=min_samples,
        features=features,
        labels=labels,
        sampling_strategy=sampling_strategy,
        n_jobs=n_jobs,
        return_indices=return_indices,
        verbose=verbose
    )


def estimate_dbscan_eps(points: np.ndarray, k: int = 5) -> float:
    """
    Estimate a good eps value for DBSCAN using k-nearest neighbors.

    Uses the k-distance method: find the "elbow" in the k-distance graph.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud
    k : int, default=5
        Number of nearest neighbors to consider

    Returns
    -------
    eps : float
        Estimated eps value (90th percentile of k-distances)

    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> eps = estimate_dbscan_eps(points, k=5)
    >>> sampled = dbscan_subsampling(points, 500, eps=eps)
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(points) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)

    # k-th nearest neighbor distance for each point
    k_distances = distances[:, k]
    k_distances = np.sort(k_distances)

    # Use 90th percentile as eps (conservative)
    eps = np.percentile(k_distances, 90)

    return eps


def dbscan_batch_subsample(
    point_cloud_list: list,
    loss_percentage: float,
    features_list: list = None,
    labels_list: list = None,
    eps: float = 0.5,
    min_samples: int = 5,
    sampling_strategy: str = "centroid",
    n_jobs: int = 1,
    seed: int = None,
    verbose: bool = False
) -> list:
    """
    Apply DBSCAN subsampling to a batch of point clouds.

    Args:
        point_cloud_list: List of (N_i, 3) point cloud arrays
        loss_percentage: Percentage of points to remove (0-100)
        features_list: Optional list of (N_i, F) feature arrays
        labels_list: Optional list of (N_i,) label arrays
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        sampling_strategy: Strategy for selecting points ("centroid", "random", "density")
        n_jobs: Number of parallel jobs for DBSCAN
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

        result = dbscan_subsample_with_loss(
            points=points,
            loss_percentage=loss_percentage,
            eps=eps,
            min_samples=min_samples,
            features=features,
            labels=labels,
            sampling_strategy=sampling_strategy,
            n_jobs=n_jobs,
            seed=cloud_seed,
            verbose=verbose
        )

        results.append(result)

    return results


if __name__ == "__main__":
    # Test DBSCAN implementation
    print("=" * 60)
    print("Testing DBSCAN-based Subsampling")
    print("=" * 60)

    # Test 1: Basic DBSCAN sampling
    print("\nTest 1: Basic DBSCAN sampling")
    np.random.seed(42)

    # Create point cloud with clear clusters
    cluster1 = np.random.randn(500, 3) * 0.3 + [0, 0, 0]
    cluster2 = np.random.randn(500, 3) * 0.3 + [2, 2, 2]
    cluster3 = np.random.randn(500, 3) * 0.3 + [-2, -2, -2]
    noise = np.random.rand(100, 3) * 6 - 3  # Random noise points

    points = np.vstack([cluster1, cluster2, cluster3, noise])
    print(f"Input shape: {points.shape}")

    sampled = dbscan_subsampling(points, 500, eps=0.5, min_samples=5, verbose=True)
    print(f"Sampled shape: {sampled.shape}")
    assert sampled.shape[0] <= 500, "Too many samples"
    print("✓ Test 1 passed")

    # Test 2: Different sampling strategies
    print("\nTest 2: Different sampling strategies")
    for strategy in ["centroid", "random", "density"]:
        print(f"  Testing strategy: {strategy}")
        sampled = dbscan_subsampling(
            points, 400, eps=0.5, min_samples=5,
            sampling_strategy=strategy
        )
        assert sampled.shape[0] <= 400
        print(f"    ✓ {strategy}: {sampled.shape[0]} points")
    print("✓ Test 2 passed")

    # Test 3: With indices
    print("\nTest 3: DBSCAN with indices")
    sampled, indices = dbscan_subsampling(
        points, 300, eps=0.5, min_samples=5,
        return_indices=True
    )
    assert np.allclose(sampled, points[indices]), "Indices don't match samples"
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min()}, {indices.max()}]")
    print("✓ Test 3 passed")

    # Test 4: With loss percentage
    print("\nTest 4: DBSCAN with loss percentage")
    sampled = dbscan_subsample_with_loss(
        points, loss_percentage=50, eps=0.5, min_samples=5, verbose=True
    )
    expected_n = int(len(points) * 0.5)
    print(f"Expected: ~{expected_n}, Got: {sampled.shape[0]}")
    print("✓ Test 4 passed")

    # Test 5: Automatic eps estimation
    print("\nTest 5: Automatic eps estimation")
    estimated_eps = estimate_dbscan_eps(points, k=5)
    print(f"Estimated eps: {estimated_eps:.4f}")

    sampled = dbscan_subsampling(
        points, 400, eps=estimated_eps, min_samples=5, verbose=True
    )
    print(f"Sampled with auto eps: {sampled.shape[0]} points")
    print("✓ Test 5 passed")

    # Test 6: Edge case - very small eps (many noise points)
    print("\nTest 6: Edge case - small eps")
    sampled = dbscan_subsampling(
        points, 200, eps=0.1, min_samples=5, verbose=True
    )
    print(f"Small eps result: {sampled.shape[0]} points")
    print("✓ Test 6 passed")

    # Test 7: Edge case - large eps (few clusters)
    print("\nTest 7: Edge case - large eps")
    sampled = dbscan_subsampling(
        points, 200, eps=5.0, min_samples=5, verbose=True
    )
    print(f"Large eps result: {sampled.shape[0]} points")
    print("✓ Test 7 passed")

    # Test 8: Comparison with FPS
    print("\nTest 8: DBSCAN vs FPS comparison")
    from fps import farthest_point_sampling

    test_points = np.random.rand(1000, 3)

    # Time DBSCAN
    start = time.time()
    dbscan_result = dbscan_subsampling(test_points, 100, eps=0.1, min_samples=3)
    dbscan_time = time.time() - start

    # Time FPS
    start = time.time()
    fps_result = farthest_point_sampling(test_points, 100)
    fps_time = time.time() - start

    print(f"DBSCAN: {dbscan_result.shape[0]} points in {dbscan_time:.3f}s")
    print(f"FPS:    {fps_result.shape[0]} points in {fps_time:.3f}s")
    print(f"DBSCAN/FPS time ratio: {dbscan_time/fps_time:.2f}x")
    print("✓ Test 8 passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
