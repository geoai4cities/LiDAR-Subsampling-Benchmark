#!/usr/bin/env python3
"""
PCA-based Curvature Preservation Analysis for Subsampled Point Clouds

This script analyzes how well different subsampling methods preserve geometric
features by comparing local surface properties before and after subsampling.

================================================================================
GEOMETRIC FEATURE DEFINITIONS
================================================================================

For each point, PCA is performed on the k-nearest neighbors (default k=30) to
compute eigenvalues (λ1 ≥ λ2 ≥ λ3) of the local covariance matrix. From these,
the following geometric features are derived:

| Feature     | Formula                  | Interpretation                        |
|-------------|--------------------------|---------------------------------------|
| Curvature   | λ3 / (λ1 + λ2 + λ3)      | High = edges/corners                  |
| Linearity   | (λ1 - λ2) / λ1           | High = linear structures (poles)      |
| Planarity   | (λ2 - λ3) / λ1           | High = planar surfaces (walls, ground)|
| Sphericity  | λ3 / λ1                  | High = scattered/volumetric (foliage) |

PRESERVATION RATIO INTERPRETATION:
  - Ratio > 1.0: Subsampled data OVER-represents that feature (edge bias)
  - Ratio = 1.0: Perfect proportional preservation (ideal)
  - Ratio < 1.0: Subsampled data UNDER-represents that feature

================================================================================
INSTALLATION
================================================================================

Required dependencies:
    pip install numpy scikit-learn

Optional (for .ply/.pcd file support):
    pip install open3d

Or use the project virtual environment:
    source /NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/ptv3_venv/bin/activate

================================================================================
USAGE EXAMPLES
================================================================================

1. Default: All methods, all losses, single sequence (08):
   python pca_curvature_analysis.py

2. All methods, all losses, ALL sequences (comprehensive benchmark):
   python pca_curvature_analysis.py --all

3. Specific methods and loss level:
   python pca_curvature_analysis.py --methods RS FPS IDIS --loss 90

4. Specific sequences only:
   python pca_curvature_analysis.py --all --sequences 00 02 05 08

5. Single loss level, all sequences:
   python pca_curvature_analysis.py --loss 90 --all

6. Faster run with fewer scans per sequence:
   python pca_curvature_analysis.py --all --n-scans 3

================================================================================
COMMAND-LINE ARGUMENTS
================================================================================

--original-dir      Directory containing original SemanticKITTI data
                    Default: /NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/data/SemanticKITTI/original

--subsampled-dir    Base directory for subsampled data
                    Default: /NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/data/SemanticKITTI/subsampled

--output-dir        Output directory for JSON results
                    Default: /NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/analysis_results/pca_curvature

--methods           Subsampling methods to compare
                    Options: RS, FPS, IDIS, DBSCAN, VB, SB, DEPOCO, IDIS_R5, IDIS_R15, IDIS_R20
                    Default: all (runs RS, FPS, IDIS, DBSCAN, VB, SB, DEPOCO)

--loss              Loss percentage
                    Options: 30, 50, 70, 90
                    Default: all (runs 30, 50, 70, 90)

--sequence          SemanticKITTI sequence number (ignored if --all is set)
                    Default: 08

--all               Run analysis on ALL sequences (00-10) and aggregate results

--sequences         Specific sequences to analyze (e.g., --sequences 00 02 05 08)

--k                 Number of neighbors for PCA computation
                    Default: 30

--n-scans           Number of scans to sample per sequence
                    Default: 5

--scan-ids          Specific scan IDs to analyze (only for single sequence mode)

================================================================================
OUTPUT FILES
================================================================================

Single sequence mode:
  - pca_curvature_analysis_loss{LOSS}.json

All sequences mode (--all):
  - pca_curvature_analysis_loss{LOSS}_seq{SEQ}.json  (per sequence)
  - pca_curvature_analysis_loss{LOSS}_all_sequences.json  (aggregated)

================================================================================
EXPECTED RESULTS (based on paper findings)
================================================================================

| Method  | 30% Loss | 50% Loss | 70% Loss | 90% Loss | Interpretation           |
|---------|----------|----------|----------|----------|--------------------------|
| RS      | ~1.00    | ~1.00    | ~1.00    | ~1.00    | Perfect baseline         |
| FPS     | ~1.18    | ~1.40    | ~1.50    | ~1.60    | Increasing edge bias     |
| VB      | ~1.18    | ~1.34    | ~1.47    | ~1.57    | Increasing edge bias     |
| SB      | ~1.17    | ~1.41    | ~1.53    | ~1.57    | Increasing edge bias     |
| DBSCAN  | ~0.92    | ~0.85    | ~0.57    | ~0.85    | Variable, cluster-based  |
| IDIS    | ~0.84    | N/A      | ~0.36    | ~0.72    | Under-represents edges   |
| DEPOCO  | ~1.12    | ~1.34    | ~1.40    | ~1.17    | Moderate, best at 90%    |

Key finding: FPS, VB, SB show monotonically increasing edge over-representation,
explaining their catastrophic failure (0.11-0.20 mIoU) on original data at 90% loss.

"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def compute_local_pca(points: np.ndarray, k: int = 30, radius: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Compute local PCA features for each point in the point cloud.

    Args:
        points: Nx3 array of point coordinates
        k: Number of nearest neighbors for PCA computation
        radius: Optional radius for neighborhood (if None, uses k-NN)

    Returns:
        Dictionary containing:
        - eigenvalues: Nx3 array of eigenvalues (λ1, λ2, λ3) sorted descending
        - curvature: Nx1 array of curvature values λ3/(λ1+λ2+λ3)
        - linearity: Nx1 array of linearity (λ1-λ2)/λ1
        - planarity: Nx1 array of planarity (λ2-λ3)/λ1
        - sphericity: Nx1 array of sphericity λ3/λ1
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for PCA analysis. Install with: pip install scikit-learn")

    n_points = len(points)

    # Build k-NN tree
    nn = NearestNeighbors(n_neighbors=min(k, n_points), algorithm='kd_tree')
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    # Initialize output arrays
    eigenvalues = np.zeros((n_points, 3))
    curvature = np.zeros(n_points)
    linearity = np.zeros(n_points)
    planarity = np.zeros(n_points)
    sphericity = np.zeros(n_points)

    for i in range(n_points):
        # Get neighborhood points
        neighbor_idx = indices[i]
        if radius is not None:
            # Filter by radius
            neighbor_idx = neighbor_idx[distances[i] <= radius]

        if len(neighbor_idx) < 3:
            # Not enough neighbors for PCA
            continue

        neighborhood = points[neighbor_idx]

        # Center the neighborhood
        centered = neighborhood - np.mean(neighborhood, axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # Sort descending

        # Avoid division by zero
        eigsum = np.sum(eigvals) + 1e-10
        λ1, λ2, λ3 = eigvals[0] + 1e-10, eigvals[1] + 1e-10, eigvals[2] + 1e-10

        eigenvalues[i] = eigvals
        curvature[i] = λ3 / eigsum
        linearity[i] = (λ1 - λ2) / λ1
        planarity[i] = (λ2 - λ3) / λ1
        sphericity[i] = λ3 / λ1

    return {
        'eigenvalues': eigenvalues,
        'curvature': curvature,
        'linearity': linearity,
        'planarity': planarity,
        'sphericity': sphericity
    }


def load_point_cloud(path: str) -> np.ndarray:
    """Load point cloud from .bin or .npy file."""
    path = Path(path)

    if path.suffix == '.bin':
        # SemanticKITTI format: x, y, z, intensity
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # Return only xyz
    elif path.suffix == '.npy':
        points = np.load(str(path))
        if points.shape[1] >= 3:
            return points[:, :3]
        return points
    elif path.suffix in ['.ply', '.pcd']:
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for .ply/.pcd files")
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def analyze_curvature_preservation(
    original_path: str,
    subsampled_path: str,
    k: int = 30,
    sample_size: int = 10000
) -> Dict:
    """
    Compare curvature metrics between original and subsampled point clouds.

    Args:
        original_path: Path to original point cloud
        subsampled_path: Path to subsampled point cloud
        k: Number of neighbors for PCA
        sample_size: Number of points to sample for analysis (for efficiency)

    Returns:
        Dictionary with preservation metrics
    """
    print(f"Loading original: {original_path}")
    original = load_point_cloud(original_path)

    print(f"Loading subsampled: {subsampled_path}")
    subsampled = load_point_cloud(subsampled_path)

    # Sample points for efficiency
    if len(original) > sample_size:
        idx = np.random.choice(len(original), sample_size, replace=False)
        original_sample = original[idx]
    else:
        original_sample = original

    if len(subsampled) > sample_size:
        idx = np.random.choice(len(subsampled), sample_size, replace=False)
        subsampled_sample = subsampled[idx]
    else:
        subsampled_sample = subsampled

    print(f"Computing PCA for original ({len(original_sample)} points)...")
    orig_features = compute_local_pca(original_sample, k=k)

    print(f"Computing PCA for subsampled ({len(subsampled_sample)} points)...")
    sub_features = compute_local_pca(subsampled_sample, k=k)

    # Compute statistics
    results = {
        'original': {
            'n_points': len(original),
            'curvature_mean': float(np.mean(orig_features['curvature'])),
            'curvature_std': float(np.std(orig_features['curvature'])),
            'curvature_median': float(np.median(orig_features['curvature'])),
            'linearity_mean': float(np.mean(orig_features['linearity'])),
            'planarity_mean': float(np.mean(orig_features['planarity'])),
            'sphericity_mean': float(np.mean(orig_features['sphericity'])),
        },
        'subsampled': {
            'n_points': len(subsampled),
            'curvature_mean': float(np.mean(sub_features['curvature'])),
            'curvature_std': float(np.std(sub_features['curvature'])),
            'curvature_median': float(np.median(sub_features['curvature'])),
            'linearity_mean': float(np.mean(sub_features['linearity'])),
            'planarity_mean': float(np.mean(sub_features['planarity'])),
            'sphericity_mean': float(np.mean(sub_features['sphericity'])),
        },
        'retention_ratio': len(subsampled) / len(original),
    }

    # Compute preservation metrics
    results['preservation'] = {
        'curvature_ratio': results['subsampled']['curvature_mean'] / (results['original']['curvature_mean'] + 1e-10),
        'linearity_ratio': results['subsampled']['linearity_mean'] / (results['original']['linearity_mean'] + 1e-10),
        'planarity_ratio': results['subsampled']['planarity_mean'] / (results['original']['planarity_mean'] + 1e-10),
        'sphericity_ratio': results['subsampled']['sphericity_mean'] / (results['original']['sphericity_mean'] + 1e-10),
    }

    return results


def compare_methods(
    original_dir: str,
    subsampled_base_dir: str,
    methods: List[str],
    loss: int,
    sequence: str = "08",
    scan_ids: List[str] = None,
    k: int = 30,
    output_dir: str = None,
    n_scans_per_seq: int = 5
) -> Dict:
    """
    Compare curvature preservation across multiple subsampling methods.

    Args:
        original_dir: Directory containing original point clouds
        subsampled_base_dir: Base directory for subsampled data
        methods: List of method names (RS, FPS, IDIS, etc.)
        loss: Loss percentage (30, 50, 70, 90)
        sequence: SemanticKITTI sequence number
        scan_ids: List of scan IDs to analyze (if None, samples 5 random scans)
        k: Number of neighbors for PCA
        output_dir: Directory to save results
        n_scans_per_seq: Number of scans to sample per sequence

    Returns:
        Dictionary with comparison results
    """
    original_path = Path(original_dir) / "sequences" / sequence / "velodyne"

    if scan_ids is None:
        # Sample n_scans_per_seq random scans
        all_scans = sorted([f.stem for f in original_path.glob("*.bin")])
        if len(all_scans) > n_scans_per_seq:
            scan_ids = np.random.choice(all_scans, n_scans_per_seq, replace=False).tolist()
        else:
            scan_ids = all_scans

    results = {
        'metadata': {
            'sequence': sequence,
            'loss': loss,
            'k_neighbors': k,
            'scan_ids': scan_ids,
            'timestamp': datetime.now().isoformat(),
        },
        'methods': {}
    }

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Analyzing method: {method}")
        print(f"{'='*60}")

        method_results = []

        for scan_id in scan_ids:
            orig_file = original_path / f"{scan_id}.bin"

            # Map method names to directory names
            method_dir_map = {
                "RS": f"RS_loss{loss}_seed1",
                "FPS": f"FPS_loss{loss}_seed1",
                "SB": f"Poisson_loss{loss}_seed1",
                "Poisson": f"Poisson_loss{loss}_seed1",
                "VB": f"Voxel_loss{loss}",
                "Voxel": f"Voxel_loss{loss}",
                "IDIS": f"IDIS_loss{loss}",
                "IDIS_R5": f"IDIS_R5_loss{loss}",
                "IDIS_R15": f"IDIS_R15_loss{loss}",
                "IDIS_R20": f"IDIS_R20_loss{loss}",
                "DBSCAN": f"DBSCAN_loss{loss}",
                "DEPOCO": f"DEPOCO_loss{loss}",
            }

            dir_name = method_dir_map.get(method, f"{method}_loss{loss}")
            sub_dir = Path(subsampled_base_dir) / dir_name / "sequences" / sequence / "velodyne"

            sub_file = sub_dir / f"{scan_id}.bin"

            if not sub_file.exists():
                # Try .npy extension
                sub_file = sub_dir / f"{scan_id}.npy"

            if not orig_file.exists() or not sub_file.exists():
                print(f"  Skipping scan {scan_id}: files not found")
                continue

            print(f"\n  Scan {scan_id}:")
            try:
                scan_results = analyze_curvature_preservation(
                    str(orig_file), str(sub_file), k=k, sample_size=5000
                )
                method_results.append(scan_results)

                print(f"    Original: {scan_results['original']['n_points']} pts, "
                      f"curvature={scan_results['original']['curvature_mean']:.4f}")
                print(f"    Subsampled: {scan_results['subsampled']['n_points']} pts, "
                      f"curvature={scan_results['subsampled']['curvature_mean']:.4f}")
                print(f"    Curvature preservation ratio: {scan_results['preservation']['curvature_ratio']:.3f}")
            except Exception as e:
                print(f"    Error: {e}")
                continue

        if method_results:
            # Aggregate results
            results['methods'][method] = {
                'n_scans': len(method_results),
                'avg_retention_ratio': np.mean([r['retention_ratio'] for r in method_results]),
                'avg_curvature_preservation': np.mean([r['preservation']['curvature_ratio'] for r in method_results]),
                'avg_linearity_preservation': np.mean([r['preservation']['linearity_ratio'] for r in method_results]),
                'avg_planarity_preservation': np.mean([r['preservation']['planarity_ratio'] for r in method_results]),
                'avg_sphericity_preservation': np.mean([r['preservation']['sphericity_ratio'] for r in method_results]),
                'original_curvature_mean': np.mean([r['original']['curvature_mean'] for r in method_results]),
                'subsampled_curvature_mean': np.mean([r['subsampled']['curvature_mean'] for r in method_results]),
                'per_scan': method_results,
            }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"pca_curvature_analysis_loss{loss}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Generate summary table
        print_summary_table(results)

    return results


def compare_methods_all_sequences(
    original_dir: str,
    subsampled_base_dir: str,
    methods: List[str],
    loss: int,
    sequences: List[str] = None,
    k: int = 30,
    output_dir: str = None,
    n_scans_per_seq: int = 3
) -> Dict:
    """
    Compare curvature preservation across multiple subsampling methods and ALL sequences.

    Args:
        original_dir: Directory containing original point clouds
        subsampled_base_dir: Base directory for subsampled data
        methods: List of method names (RS, FPS, IDIS, etc.)
        loss: Loss percentage (30, 50, 70, 90)
        sequences: List of sequence numbers (if None, uses all 00-10)
        k: Number of neighbors for PCA
        output_dir: Directory to save results
        n_scans_per_seq: Number of scans to sample per sequence

    Returns:
        Dictionary with aggregated comparison results across all sequences
    """
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    all_sequence_results = {}

    for seq in sequences:
        original_path = Path(original_dir) / "sequences" / seq / "velodyne"
        if not original_path.exists():
            print(f"\nSequence {seq}: Original data not found, skipping...")
            continue

        print(f"\n{'#'*80}")
        print(f"# SEQUENCE {seq}")
        print(f"{'#'*80}")

        # Sample scans for this sequence
        all_scans = sorted([f.stem for f in original_path.glob("*.bin")])
        if len(all_scans) > n_scans_per_seq:
            scan_ids = np.random.choice(all_scans, n_scans_per_seq, replace=False).tolist()
        else:
            scan_ids = all_scans

        seq_results = {
            'metadata': {
                'sequence': seq,
                'loss': loss,
                'k_neighbors': k,
                'scan_ids': scan_ids,
            },
            'methods': {}
        }

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Analyzing method: {method} (Sequence {seq})")
            print(f"{'='*60}")

            method_results = []

            for scan_id in scan_ids:
                orig_file = original_path / f"{scan_id}.bin"

                # Map method names to directory names
                method_dir_map = {
                    "RS": f"RS_loss{loss}_seed1",
                    "FPS": f"FPS_loss{loss}_seed1",
                    "SB": f"Poisson_loss{loss}_seed1",
                    "Poisson": f"Poisson_loss{loss}_seed1",
                    "VB": f"Voxel_loss{loss}",
                    "Voxel": f"Voxel_loss{loss}",
                    "IDIS": f"IDIS_loss{loss}",
                    "IDIS_R5": f"IDIS_R5_loss{loss}",
                    "IDIS_R15": f"IDIS_R15_loss{loss}",
                    "IDIS_R20": f"IDIS_R20_loss{loss}",
                    "DBSCAN": f"DBSCAN_loss{loss}",
                    "DEPOCO": f"DEPOCO_loss{loss}",
                }

                dir_name = method_dir_map.get(method, f"{method}_loss{loss}")
                sub_dir = Path(subsampled_base_dir) / dir_name / "sequences" / seq / "velodyne"

                sub_file = sub_dir / f"{scan_id}.bin"

                if not sub_file.exists():
                    sub_file = sub_dir / f"{scan_id}.npy"

                if not orig_file.exists() or not sub_file.exists():
                    print(f"  Skipping scan {scan_id}: files not found")
                    continue

                print(f"\n  Scan {scan_id}:")
                try:
                    scan_results = analyze_curvature_preservation(
                        str(orig_file), str(sub_file), k=k, sample_size=5000
                    )
                    method_results.append(scan_results)

                    print(f"    Original: {scan_results['original']['n_points']} pts, "
                          f"curvature={scan_results['original']['curvature_mean']:.4f}")
                    print(f"    Subsampled: {scan_results['subsampled']['n_points']} pts, "
                          f"curvature={scan_results['subsampled']['curvature_mean']:.4f}")
                    print(f"    Curvature preservation ratio: {scan_results['preservation']['curvature_ratio']:.3f}")
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            if method_results:
                seq_results['methods'][method] = {
                    'n_scans': len(method_results),
                    'avg_retention_ratio': np.mean([r['retention_ratio'] for r in method_results]),
                    'avg_curvature_preservation': np.mean([r['preservation']['curvature_ratio'] for r in method_results]),
                    'avg_linearity_preservation': np.mean([r['preservation']['linearity_ratio'] for r in method_results]),
                    'avg_planarity_preservation': np.mean([r['preservation']['planarity_ratio'] for r in method_results]),
                    'avg_sphericity_preservation': np.mean([r['preservation']['sphericity_ratio'] for r in method_results]),
                    'per_scan': method_results,
                }

        all_sequence_results[seq] = seq_results

    # Aggregate results across all sequences
    aggregated = aggregate_sequence_results(all_sequence_results, methods, loss, k)

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save per-sequence results
        for seq, seq_results in all_sequence_results.items():
            seq_file = output_path / f"pca_curvature_analysis_loss{loss}_seq{seq}.json"
            with open(seq_file, 'w') as f:
                json.dump(seq_results, f, indent=2)

        # Save aggregated results
        agg_file = output_path / f"pca_curvature_analysis_loss{loss}_all_sequences.json"
        with open(agg_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated results saved to: {agg_file}")

        # Print summary table
        print_summary_table_with_std(aggregated)

    return aggregated


def aggregate_sequence_results(all_sequence_results: Dict, methods: List[str], loss: int, k: int) -> Dict:
    """Aggregate results across all sequences with mean ± std."""
    aggregated = {
        'metadata': {
            'sequences': list(all_sequence_results.keys()),
            'loss': loss,
            'k_neighbors': k,
            'timestamp': datetime.now().isoformat(),
        },
        'methods': {}
    }

    for method in methods:
        curvature_vals = []
        linearity_vals = []
        planarity_vals = []
        sphericity_vals = []
        retention_vals = []

        for seq_results in all_sequence_results.values():
            if method in seq_results['methods']:
                m_data = seq_results['methods'][method]
                curvature_vals.append(m_data['avg_curvature_preservation'])
                linearity_vals.append(m_data['avg_linearity_preservation'])
                planarity_vals.append(m_data['avg_planarity_preservation'])
                sphericity_vals.append(m_data['avg_sphericity_preservation'])
                retention_vals.append(m_data['avg_retention_ratio'])

        if curvature_vals:
            aggregated['methods'][method] = {
                'n_sequences': len(curvature_vals),
                'retention_mean': float(np.mean(retention_vals)),
                'retention_std': float(np.std(retention_vals)),
                'curvature_mean': float(np.mean(curvature_vals)),
                'curvature_std': float(np.std(curvature_vals)),
                'linearity_mean': float(np.mean(linearity_vals)),
                'linearity_std': float(np.std(linearity_vals)),
                'planarity_mean': float(np.mean(planarity_vals)),
                'planarity_std': float(np.std(planarity_vals)),
                'sphericity_mean': float(np.mean(sphericity_vals)),
                'sphericity_std': float(np.std(sphericity_vals)),
                'per_sequence_curvature': {s: all_sequence_results[s]['methods'].get(method, {}).get('avg_curvature_preservation', None)
                                           for s in all_sequence_results.keys()},
            }

    return aggregated


def print_summary_table_with_std(results: Dict):
    """Print a formatted summary table with mean ± std from all sequences."""
    print("\n" + "="*100)
    print("PCA CURVATURE PRESERVATION ANALYSIS SUMMARY (ALL SEQUENCES)")
    print(f"Loss Level: {results['metadata']['loss']}%")
    print(f"Sequences: {', '.join(results['metadata']['sequences'])}")
    print("="*100)

    print(f"\n{'Method':<10} {'N_Seq':<6} {'Retention':<14} {'Curvature':<16} {'Linearity':<16} {'Planarity':<16} {'Sphericity':<16}")
    print("-"*100)

    for method, data in results['methods'].items():
        print(f"{method:<10} "
              f"{data['n_sequences']:<6} "
              f"{data['retention_mean']*100:>5.1f}% ± {data['retention_std']*100:>4.1f}% "
              f"{data['curvature_mean']:>5.2f} ± {data['curvature_std']:>5.3f} "
              f"{data['linearity_mean']:>5.2f} ± {data['linearity_std']:>5.3f} "
              f"{data['planarity_mean']:>5.2f} ± {data['planarity_std']:>5.3f} "
              f"{data['sphericity_mean']:>5.2f} ± {data['sphericity_std']:>5.3f}")

    print("-"*100)
    print("\nInterpretation:")
    print("  - Preservation ratio > 1.0: Over-representation of that feature type")
    print("  - Preservation ratio = 1.0: Perfect preservation")
    print("  - Preservation ratio < 1.0: Under-representation of that feature type")


def print_summary_table(results: Dict):
    """Print a formatted summary table of curvature preservation results."""
    print("\n" + "="*80)
    print("PCA CURVATURE PRESERVATION ANALYSIS SUMMARY")
    print(f"Loss Level: {results['metadata']['loss']}%")
    print("="*80)

    print(f"\n{'Method':<10} {'Retention':<12} {'Curvature':<12} {'Linearity':<12} {'Planarity':<12} {'Sphericity':<12}")
    print("-"*80)

    for method, data in results['methods'].items():
        print(f"{method:<10} "
              f"{data['avg_retention_ratio']*100:>10.1f}% "
              f"{data['avg_curvature_preservation']:>11.3f} "
              f"{data['avg_linearity_preservation']:>11.3f} "
              f"{data['avg_planarity_preservation']:>11.3f} "
              f"{data['avg_sphericity_preservation']:>11.3f}")

    print("-"*80)
    print("\nInterpretation:")
    print("  - Preservation ratio > 1.0: Subsampled data has HIGHER metric (over-retention of that feature type)")
    print("  - Preservation ratio = 1.0: Perfect preservation")
    print("  - Preservation ratio < 1.0: Subsampled data has LOWER metric (under-retention of that feature type)")
    print("\n  - Curvature: High values indicate edge/corner regions")
    print("  - Linearity: High values indicate linear structures (poles, edges)")
    print("  - Planarity: High values indicate planar surfaces (walls, ground)")
    print("  - Sphericity: High values indicate scattered/volumetric regions (vegetation)")


def main():
    parser = argparse.ArgumentParser(description="PCA Curvature Preservation Analysis")

    parser.add_argument("--original-dir", type=str,
                        default="/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/data/SemanticKITTI/original",
                        help="Directory containing original SemanticKITTI data")
    parser.add_argument("--subsampled-dir", type=str,
                        default="/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/data/SemanticKITTI/subsampled",
                        help="Base directory for subsampled data")
    parser.add_argument("--output-dir", type=str,
                        default="/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/analysis_results/pca_curvature",
                        help="Output directory for results")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["all"],
                        help="Subsampling methods to compare (RS, FPS, IDIS, DBSCAN, VB, SB, DEPOCO). Default: all")
    parser.add_argument("--loss", type=str, default="all",
                        help="Loss percentage (30, 50, 70, 90). Default: all")
    parser.add_argument("--sequence", type=str, default="08",
                        help="SemanticKITTI sequence number (ignored if --all is set)")
    parser.add_argument("--all", action="store_true",
                        help="Run analysis on ALL sequences (00-10) and aggregate results")
    parser.add_argument("--sequences", type=str, nargs="+",
                        help="Specific sequences to analyze (e.g., 00 02 05 08)")
    parser.add_argument("--k", type=int, default=30,
                        help="Number of neighbors for PCA computation")
    parser.add_argument("--n-scans", type=int, default=5,
                        help="Number of scans to analyze per sequence")
    parser.add_argument("--scan-ids", type=str, nargs="*",
                        help="Specific scan IDs to analyze (only for single sequence mode)")

    args = parser.parse_args()

    # Handle "all" keyword for methods
    ALL_METHODS = ["RS", "FPS", "IDIS", "DBSCAN", "VB", "SB", "DEPOCO"]
    if args.methods == ["all"] or args.methods == "all":
        methods = ALL_METHODS
    else:
        methods = args.methods

    # Handle "all" keyword for loss levels
    if args.loss.lower() == "all":
        loss_levels = [30, 50, 70, 90]
    else:
        loss_levels = [int(args.loss)]

    print("PCA Curvature Preservation Analysis")
    print("="*60)
    print(f"Original data: {args.original_dir}")
    print(f"Subsampled data: {args.subsampled_dir}")
    print(f"Methods: {methods}")
    print(f"Loss levels: {loss_levels}")

    all_results = {}

    for loss in loss_levels:
        print(f"\n{'#'*80}")
        print(f"# LOSS LEVEL: {loss}%")
        print(f"{'#'*80}")

        if args.all:
            sequences = args.sequences if args.sequences else None
            seq_str = ", ".join(sequences) if sequences else "00-10 (all)"
            print(f"Sequences: {seq_str}")
            print(f"Scans per sequence: {args.n_scans}")
            print(f"k-neighbors: {args.k}")
            print("="*60)

            results = compare_methods_all_sequences(
                original_dir=args.original_dir,
                subsampled_base_dir=args.subsampled_dir,
                methods=methods,
                loss=loss,
                sequences=sequences,
                k=args.k,
                output_dir=args.output_dir,
                n_scans_per_seq=args.n_scans
            )
        else:
            print(f"Sequence: {args.sequence}")
            print(f"k-neighbors: {args.k}")
            print("="*60)

            results = compare_methods(
                original_dir=args.original_dir,
                subsampled_base_dir=args.subsampled_dir,
                methods=methods,
                loss=loss,
                sequence=args.sequence,
                scan_ids=args.scan_ids,
                k=args.k,
                output_dir=args.output_dir,
                n_scans_per_seq=args.n_scans
            )

        all_results[loss] = results

    # Print final summary if running multiple losses
    if len(loss_levels) > 1:
        print_all_losses_summary(all_results, methods, args.all)

    return all_results


def print_all_losses_summary(all_results: Dict, methods: List[str], is_all_sequences: bool):
    """Print a comprehensive summary table across all loss levels."""
    print("\n" + "="*100)
    print("COMPREHENSIVE PCA CURVATURE PRESERVATION SUMMARY")
    print("="*100)

    # Header
    print(f"\n{'Method':<10} | {'30% Loss':<12} | {'50% Loss':<12} | {'70% Loss':<12} | {'90% Loss':<12} | {'Trend':<20}")
    print("-"*100)

    for method in methods:
        values = []
        for loss in [30, 50, 70, 90]:
            if loss in all_results:
                if is_all_sequences and 'methods' in all_results[loss]:
                    if method in all_results[loss]['methods']:
                        val = all_results[loss]['methods'][method].get('curvature_mean', None)
                        std = all_results[loss]['methods'][method].get('curvature_std', 0)
                        values.append((val, std))
                    else:
                        values.append((None, None))
                elif 'methods' in all_results[loss]:
                    if method in all_results[loss]['methods']:
                        val = all_results[loss]['methods'][method].get('avg_curvature_preservation', None)
                        values.append((val, 0))
                    else:
                        values.append((None, None))
                else:
                    values.append((None, None))
            else:
                values.append((None, None))

        # Format values
        val_strs = []
        for val, std in values:
            if val is not None:
                if std and std > 0:
                    val_strs.append(f"{val:.2f}±{std:.2f}")
                else:
                    val_strs.append(f"{val:.2f}")
            else:
                val_strs.append("N/A")

        # Determine trend
        valid_vals = [v[0] for v in values if v[0] is not None]
        if len(valid_vals) >= 2:
            if valid_vals[-1] > valid_vals[0] + 0.1:
                trend = "↑ Increasing bias"
            elif valid_vals[-1] < valid_vals[0] - 0.1:
                trend = "↓ Decreasing"
            else:
                trend = "→ Stable"
        else:
            trend = "-"

        print(f"{method:<10} | {val_strs[0]:<12} | {val_strs[1]:<12} | {val_strs[2]:<12} | {val_strs[3]:<12} | {trend:<20}")

    print("-"*100)
    print("\nInterpretation:")
    print("  - Curvature ratio > 1.0: Over-represents edges/corners (can cause generalization failure)")
    print("  - Curvature ratio = 1.0: Perfect proportional preservation (ideal)")
    print("  - Curvature ratio < 1.0: Under-represents edges/corners")


if __name__ == "__main__":
    main()
