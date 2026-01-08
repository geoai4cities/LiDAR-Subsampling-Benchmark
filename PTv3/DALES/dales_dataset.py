"""
DALES (Dayton Annotated LiDAR Earth Scan) dataset

Adapted from: semantic_kitti.py by Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

Dataset: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php
"""

import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class DALESDataset(DefaultDataset):
    """
    DALES dataset for airborne LiDAR semantic segmentation.

    Format: .txt files with columns: x, y, z, intensity, return_num, num_returns, class
    Classes: 8 classes (Ground, Vegetation, Cars, Trucks, Power lines, Fences, Poles, Buildings)
    Total tiles: 40 (29 from train directory + 11 from test directory)

    Splits (based on actual data directory structure):
    - train: 23 tiles (~80% of train directory, spatially distributed)
    - val: 6 tiles (~20% of train directory, spatially distributed for good coverage)
    - test: 11 tiles (test directory, as provided)

    Data source: data/DALES/original/dales_ply/{train,test}/
    """

    def __init__(
        self,
        split="train",
        data_root="data/DALES",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        """
        Get list of .txt files for DALES tiles.

        Splits (based on actual data directory structure):
        - train: 23 tiles (80% of train directory)
        - val: 6 tiles (20% of train directory)
        - test: 11 tiles (test directory)

        Total: 40 tiles (29 train + 11 test from original data)
        """
        split2tiles = dict(
            train=[
                # Training tiles (23 tiles, ~80% of train dir)
                # Excluded tiles used for validation: 5080_54435, 5100_54495, 5130_54355, 5145_54470, 5165_54395, 5190_54400
                "5085_54320", "5095_54440", "5095_54455", "5105_54405", "5105_54460",
                "5110_54320", "5110_54460", "5110_54475", "5110_54495", "5115_54480",
                "5135_54495", "5140_54445", "5145_54340", "5145_54405", "5145_54460",
                "5145_54480", "5150_54340", "5160_54330", "5165_54390", "5180_54435",
                "5180_54485", "5185_54390", "5185_54485"
            ],
            val=[
                # Validation tiles (6 tiles, ~20% of train dir, spatially distributed)
                "5080_54435",  # Northwest
                "5100_54495",  # North-central
                "5130_54355",  # Central
                "5145_54470",  # South-central
                "5165_54395",  # South
                "5190_54400"   # Southeast
            ],
            test=[
                # Test tiles (11 tiles from test directory)
                "5080_54400", "5080_54470", "5100_54440", "5100_54490", "5120_54445",
                "5135_54430", "5135_54435", "5140_54390", "5150_54325", "5155_54335",
                "5175_54395"
            ],
        )

        if isinstance(self.split, str):
            tile_list = split2tiles[self.split]
        elif isinstance(self.split, list):
            tile_list = []
            for split in self.split:
                tile_list += split2tiles[split]
        else:
            raise NotImplementedError(f"Split type {type(self.split)} not supported")

        data_list = []
        for tile in tile_list:
            # Look for .txt file in data_root
            # Support both direct path and with subsampling subdirectory structure
            tile_file = os.path.join(self.data_root, f"{tile}.txt")

            if not os.path.exists(tile_file):
                # Try alternative path structure (for subsampled datasets)
                # e.g., data/DALES/subsampled/RS_loss0/5030001.txt
                alt_tile_file = os.path.join(self.data_root, tile, f"{tile}.txt")
                if os.path.exists(alt_tile_file):
                    tile_file = alt_tile_file
                else:
                    raise FileNotFoundError(
                        f"Tile file not found: {tile_file} or {alt_tile_file}"
                    )

            data_list.append(tile_file)

        return data_list

    def get_data(self, idx):
        """
        Load DALES .txt file and return point cloud data.

        Format: x, y, z, intensity, return_num, num_returns, class
        Columns: 7 total (0-indexed: 0=x, 1=y, 2=z, 3=intensity, 6=class)

        Returns:
            dict with keys:
                - coord: (N, 3) - x, y, z coordinates
                - strength: (N, 1) - intensity values
                - segment: (N,) - semantic labels (remapped to 0-7)
        """
        data_path = self.data_list[idx % len(self.data_list)]

        # Load .txt file
        # Format: x, y, z, intensity, return_num, num_returns, class
        try:
            data = np.loadtxt(data_path)
        except Exception as e:
            raise IOError(f"Error loading {data_path}: {e}")

        # Validate shape
        if data.ndim != 2 or data.shape[1] < 7:
            raise ValueError(
                f"Invalid data shape {data.shape} for {data_path}. "
                f"Expected (N, 7) with columns: x, y, z, intensity, return_num, num_returns, class"
            )

        # Extract coordinates (columns 0-2)
        coord = data[:, :3].astype(np.float32)

        # Extract intensity (column 3)
        strength = data[:, 3:4].astype(np.float32)

        # Extract labels (column 6) and remap
        raw_labels = data[:, 6].astype(np.int32)
        segment = np.vectorize(self.learning_map.__getitem__)(raw_labels).astype(np.int32)

        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        """
        Get unique name for this data point.

        Returns: tile name (e.g., "5030001")
        """
        file_path = self.data_list[idx % len(self.data_list)]
        file_name = os.path.basename(file_path)
        data_name = os.path.splitext(file_name)[0]  # Remove .txt extension
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        """
        Get label remapping for DALES dataset.

        DALES has 8 classes (0-7):
        0: Ground
        1: Vegetation
        2: Cars
        3: Trucks
        4: Power lines
        5: Fences
        6: Poles
        7: Buildings

        Returns:
            dict: Mapping from raw labels to remapped labels (0-7)
        """
        # DALES labels are already 0-7, so identity mapping
        # But we handle any potential outliers
        learning_map = {
            0: 0,  # Ground
            1: 1,  # Vegetation
            2: 2,  # Cars
            3: 3,  # Trucks
            4: 4,  # Power lines
            5: 5,  # Fences
            6: 6,  # Poles
            7: 7,  # Buildings
            # Handle potential outliers
            -1: ignore_index,  # Unknown/unlabeled
            255: ignore_index,  # Sometimes used as unlabeled
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        """
        Get inverse label remapping (for visualization/saving).

        Returns:
            dict: Mapping from remapped labels (0-7) to original labels
        """
        learning_map_inv = {
            ignore_index: ignore_index,  # unlabeled
            0: 0,  # Ground
            1: 1,  # Vegetation
            2: 2,  # Cars
            3: 3,  # Trucks
            4: 4,  # Power lines
            5: 5,  # Fences
            6: 6,  # Poles
            7: 7,  # Buildings
        }
        return learning_map_inv
