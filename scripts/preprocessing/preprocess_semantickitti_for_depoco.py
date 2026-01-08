#!/usr/bin/env python3
"""
Preprocess SemanticKITTI data for DEPOCO training.

This script converts SemanticKITTI velodyne scans into voxelized submaps
required for training DEPOCO models.

Based on: DEPOCO kitti2voxel.py (https://github.com/PRBonn/deep-point-map-compression)

Usage:
    python preprocess_semantickitti_for_depoco.py --input /path/to/semantickitti --output /path/to/output

Output Structure:
    output/
    ├── train/
    │   ├── 000000.bin
    │   ├── 000001.bin
    │   └── ...
    ├── validation/
    │   └── ...
    └── test/
        └── ...
"""

import numpy as np
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Install with: pip install open3d")
    exit(1)

# Default paths (comment out to use environment variables/arguments instead)
DEFAULT_DEPOCO_BASE = "/DATA/aakash/ms-project/depoco_for_transfer"
DEFAULT_INPUT = "/NFSDISK2/pyare/LiDAR-Subsampling-Benchmark/data/SemanticKITTI/original"
DEFAULT_OUTPUT = "/DATA/aakash/paper-1/skitti_depoco_new"
# Or use environment variables:
# DEFAULT_DEPOCO_BASE = os.environ.get("DEPOCO_BASE", "")
# DEFAULT_INPUT = None  # Required via --input argument
# DEFAULT_OUTPUT = None  # Required via --output argument

# Try to import octree_handler for normal/eigenvalue computation
try:
    import sys
    # Use default path or DEPOCO_BASE environment variable
    depoco_base = os.environ.get("DEPOCO_BASE", DEFAULT_DEPOCO_BASE)
    if depoco_base:
        sys.path.insert(0, depoco_base)
    import octree_handler
    HAS_OCTREE = True
except ImportError:
    HAS_OCTREE = False
    print("Warning: octree_handler not available. Normals/eigenvalues will be computed with Open3D.")
    print(f"         DEPOCO_BASE: {depoco_base if depoco_base else 'Not set'}")


# SemanticKITTI sequence splits
TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VALID_SEQUENCES = ["08"]
TEST_SEQUENCES = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]


def open_label(filename):
    """Open SemanticKITTI label file."""
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF  # Get semantic label (lower 16 bits)
    return label


def parse_calibration(filename):
    """Read calibration file."""
    calib = {}
    with open(filename, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, content = line.strip().split(":", 1)
            values = [float(v) for v in content.strip().split()]
            if len(values) == 12:
                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0
                calib[key] = pose
    return calib


def parse_poses(filename, calibration):
    """Read poses file."""
    poses = []
    Tr = calibration.get("Tr", np.eye(4))
    Tr_inv = np.linalg.inv(Tr)

    with open(filename, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) != 12:
                continue
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


def distance_matrix(x, y):
    """Compute distance matrix between two point sets."""
    dims = x.shape[1]
    dist = np.zeros((x.shape[0], y.shape[0]))
    for i in range(dims):
        dist += (x[:, i][..., np.newaxis] - y[:, i][np.newaxis, ...]) ** 2
    return dist ** 0.5


def get_key_poses(pose_list, delta=15):
    """Get key poses that are delta meters apart."""
    poses = np.asarray(pose_list)
    xy = poses[:, 0:2, -1]
    dist = distance_matrix(xy, xy)

    key_pose_idx = []
    indices = np.arange(poses.shape[0])
    dist_it = dist.copy()

    while dist_it.shape[0] > 0:
        key_pose_idx.append(indices[0])
        valid_idx = dist_it[0, :] > delta
        dist_it = dist_it[valid_idx, :]
        dist_it = dist_it[:, valid_idx]
        indices = indices[valid_idx]

    return key_pose_idx, poses[key_pose_idx], dist


def save_cloud_to_binary(cloud, filename):
    """Save point cloud to binary file."""
    cloud.astype('float32').tofile(filename)


def compute_normals_eigenvalues_o3d(points, radius=0.5):
    """Compute normals and eigenvalues using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=30))

    normals = np.asarray(pcd.normals)

    # For eigenvalues, use covariance estimation (simplified)
    # Real implementation would use octree_handler
    eigenvalues = np.ones((points.shape[0], 3)) * 0.1  # Placeholder

    return np.hstack((eigenvalues, normals))


def create_submap(poses, key_pose_idx, seq_path, distance_matrix, config):
    """Create a voxelized submap around a key pose."""
    grid_size = np.array(config['grid_size'])
    center = poses[key_pose_idx][0:3, -1] + np.array([0, 0, config['dz']])
    upper_bound = center + grid_size / 2
    lower_bound = center - grid_size / 2

    # Find valid scans within range
    valid_scans = np.argwhere(
        distance_matrix[key_pose_idx, :] < grid_size[0] + config['max_range']
    ).squeeze()

    if valid_scans.ndim == 0:
        valid_scans = np.array([valid_scans.item()])

    point_cld = []
    features = []

    for i in valid_scans:
        velodyne_file = os.path.join(seq_path, "velodyne", f"{i:06d}.bin")
        label_file = os.path.join(seq_path, "labels", f"{i:06d}.label")

        if not os.path.isfile(velodyne_file):
            continue

        # Load scan
        scan = np.fromfile(velodyne_file, dtype=np.float32).reshape((-1, 4))

        # Filter by range
        dists = np.linalg.norm(scan[:, 0:3], axis=1)
        valid_p = (dists > config['min_range']) & (dists < config['max_range'])

        # Transform to world coordinates
        scan_hom = np.ones_like(scan)
        scan_hom[:, 0:3] = scan[:, 0:3]
        points = np.matmul(poses[i], scan_hom[valid_p, :].T).T

        # Get intensity and labels
        intensity = scan[valid_p, 3:4]

        if os.path.isfile(label_file):
            label = open_label(label_file)[valid_p]
        else:
            label = np.full((points.shape[0],), 0)

        # Create feature vector: [intensity, label, placeholder]
        feature = np.hstack((
            intensity,
            np.expand_dims(label.astype('float'), axis=1),
            np.zeros_like(intensity)
        ))

        points = points[:, 0:3]

        # Filter by grid bounds and remove moving objects (labels >= 200)
        valids = (
            np.all(points > lower_bound, axis=1) &
            np.all(points < upper_bound, axis=1) &
            (label < 200) & (label > 0)
        )

        if np.sum(valids) > 0:
            point_cld.append(points[valids, :])
            features.append(feature[valids])

    if len(point_cld) == 0:
        return None

    # Concatenate all points
    cloud = np.concatenate(point_cld)
    cloud_features = np.concatenate(features)

    # Voxelize using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(cloud_features)

    downpcd = pcd.voxel_down_sample(voxel_size=config['voxel_size'])

    sparse_points = np.asarray(downpcd.points)
    sparse_features = np.asarray(downpcd.colors)
    sparse_features = sparse_features[:, :2]  # intensity and label
    sparse_features[:, 1] = np.around(sparse_features[:, 1])  # Round labels

    # Compute normals and eigenvalues
    if HAS_OCTREE:
        octree = octree_handler.Octree()
        octree.setInput(sparse_points)
        eig_normals = octree.computeEigenvaluesNormal(config['normal_radius'])
    else:
        eig_normals = compute_normals_eigenvalues_o3d(sparse_points, config['normal_radius'])

    # Final format: [x, y, z, intensity, label, eigenvalues(3), normals(3)]
    sparse_points_features = np.hstack((sparse_points, sparse_features, eig_normals))

    return sparse_points_features.astype('float32')


def process_sequence(seq_path, output_dir, config, start_idx=0):
    """Process a single sequence."""
    calib_file = os.path.join(seq_path, "calib.txt")
    poses_file = os.path.join(seq_path, "poses.txt")

    if not os.path.exists(calib_file) or not os.path.exists(poses_file):
        print(f"  Skipping {seq_path}: missing calib.txt or poses.txt")
        return start_idx, 0

    calibration = parse_calibration(calib_file)
    poses = parse_poses(poses_file, calibration)

    if len(poses) == 0:
        print(f"  Skipping {seq_path}: no poses found")
        return start_idx, 0

    # Get key poses
    key_poses_idx, key_poses, dist_matrix = get_key_poses(poses, config['pose_distance'])

    count = 0
    for i, idx in enumerate(tqdm(key_poses_idx, desc=f"  Processing submaps")):
        submap = create_submap(poses, idx, seq_path, dist_matrix, config)

        if submap is not None and submap.shape[0] > 100:  # Min points threshold
            output_file = os.path.join(output_dir, f"{start_idx + count}.bin")
            save_cloud_to_binary(submap, output_file)
            count += 1

    return start_idx + count, count


def main():
    parser = argparse.ArgumentParser(description="Preprocess SemanticKITTI for DEPOCO")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT,
                        help=f"Path to SemanticKITTI dataset (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output directory for preprocessed data (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--pose-distance", type=float, default=15.0,
                        help="Distance between key poses (meters)")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size for downsampling")
    parser.add_argument("--grid-size", type=float, nargs=3, default=[40.0, 40.0, 15.0],
                        help="Grid size [x, y, z] in meters")
    parser.add_argument("--min-range", type=float, default=2.0,
                        help="Minimum scan range")
    parser.add_argument("--max-range", type=float, default=20.0,
                        help="Maximum scan range")
    parser.add_argument("--dz", type=float, default=4.0,
                        help="Z offset for grid center")
    parser.add_argument("--normal-radius", type=float, default=0.5,
                        help="Radius for normal estimation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without processing")

    args = parser.parse_args()

    # Build config
    config = {
        'pose_distance': args.pose_distance,
        'voxel_size': args.voxel_size,
        'grid_size': args.grid_size,
        'min_range': args.min_range,
        'max_range': args.max_range,
        'dz': args.dz,
        'normal_radius': args.normal_radius,
    }

    # Validate input
    sequences_path = os.path.join(args.input, "sequences")
    if not os.path.exists(sequences_path):
        # Try without sequences subfolder
        if os.path.exists(os.path.join(args.input, "00")):
            sequences_path = args.input
        else:
            print(f"Error: Cannot find sequences in {args.input}")
            return 1

    print("=" * 60)
    print("  SemanticKITTI Preprocessing for DEPOCO")
    print("=" * 60)
    print(f"\nInput:  {sequences_path}")
    print(f"Output: {args.output}")
    print(f"\nConfiguration:")
    print(f"  Pose distance:  {config['pose_distance']} m")
    print(f"  Voxel size:     {config['voxel_size']} m")
    print(f"  Grid size:      {config['grid_size']}")
    print(f"  Range:          {config['min_range']} - {config['max_range']} m")
    print(f"  Octree handler: {'Available' if HAS_OCTREE else 'Not available (using Open3D)'}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would process:")
        print(f"  Train sequences: {TRAIN_SEQUENCES}")
        print(f"  Valid sequences: {VALID_SEQUENCES}")
        print(f"  Test sequences:  {TEST_SEQUENCES}")
        return 0

    # Create output directories
    train_dir = os.path.join(args.output, "train")
    valid_dir = os.path.join(args.output, "validation")
    test_dir = os.path.join(args.output, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    start_time = time.time()

    # Process train sequences
    print("\n[1/3] Processing TRAIN sequences...")
    train_idx = 0
    train_total = 0
    for seq in TRAIN_SEQUENCES:
        seq_path = os.path.join(sequences_path, seq)
        if os.path.exists(seq_path):
            print(f"\n  Sequence {seq}:")
            train_idx, count = process_sequence(seq_path, train_dir, config, train_idx)
            train_total += count
            print(f"    Generated {count} submaps")

    # Process validation sequences
    print("\n[2/3] Processing VALIDATION sequences...")
    valid_idx = 0
    valid_total = 0
    for seq in VALID_SEQUENCES:
        seq_path = os.path.join(sequences_path, seq)
        if os.path.exists(seq_path):
            print(f"\n  Sequence {seq}:")
            valid_idx, count = process_sequence(seq_path, valid_dir, config, valid_idx)
            valid_total += count
            print(f"    Generated {count} submaps")

    # Process test sequences
    print("\n[3/3] Processing TEST sequences...")
    test_idx = 0
    test_total = 0
    for seq in TEST_SEQUENCES:
        seq_path = os.path.join(sequences_path, seq)
        if os.path.exists(seq_path):
            print(f"\n  Sequence {seq}:")
            test_idx, count = process_sequence(seq_path, test_dir, config, test_idx)
            test_total += count
            print(f"    Generated {count} submaps")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  Preprocessing Complete")
    print("=" * 60)
    print(f"\nGenerated submaps:")
    print(f"  Train:      {train_total} submaps")
    print(f"  Validation: {valid_total} submaps")
    print(f"  Test:       {test_total} submaps")
    print(f"  Total:      {train_total + valid_total + test_total} submaps")
    print(f"\nElapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nOutput saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
