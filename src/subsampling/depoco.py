"""
DEPOCO: Deep Point Cloud Compression
=====================================

This module provides the DEPOCO (DEep POint Cloud COmpression) subsampling method
for the LiDAR Subsampling Benchmark.

Paper: "Deep Compression for Dense Point Cloud Maps"
       Louis Wiesmann et al., IEEE RA-L 2021
       https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf

Configuration:
--------------
Set these environment variables before using DEPOCO:
  - DEPOCO_BASE: Path to DEPOCO project directory
  - DEPOCO_VENV: Path to DEPOCO virtual environment

Example:
  export DEPOCO_BASE=/path/to/depoco_for_transfer
  export DEPOCO_VENV=/path/to/venv/py38_depoco

Available SemanticKITTI Models:
  - final_skitti_72.5: ~10% loss (subsampling_dist=0.524)
  - final_skitti_82.5: ~30% loss (subsampling_dist=0.85)
  - final_skitti_87.5: ~50% loss (subsampling_dist=1.3)
  - final_skitti_92.5: ~70% loss (subsampling_dist=1.8)
  - final_skitti_97.5: ~90% loss (subsampling_dist=2.3)

See configs/depoco/README.md for detailed documentation.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from scipy.spatial import cKDTree

# ==============================================================================
# Configuration
# ==============================================================================

# DEPOCO project paths - from environment variables
# Set these before using DEPOCO:
#   export DEPOCO_BASE=/path/to/depoco_for_transfer
#   export DEPOCO_VENV=/path/to/venv/py38_depoco
DEPOCO_BASE_PATH = os.environ.get("DEPOCO_BASE", "")
DEPOCO_VENV_PATH = os.environ.get("DEPOCO_VENV", "")
DEPOCO_MODELS_PATH = f"{DEPOCO_BASE_PATH}/main-scripts/paper-1/network_files" if DEPOCO_BASE_PATH else ""
DEPOCO_YAMLS_PATH = f"{DEPOCO_BASE_PATH}/yamls/paper-1" if DEPOCO_BASE_PATH else ""

# Alternative: Get configs from this repository
def _get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs" / "depoco").exists():
            return parent
    return None

PROJECT_ROOT = _get_project_root()
REPO_CONFIGS_PATH = str(PROJECT_ROOT / "configs" / "depoco") if PROJECT_ROOT else ""

# Available DEPOCO models for SemanticKITTI
# Maps retention percentage to model name
AVAILABLE_MODELS = {
    92.5: "final_skitti_92.5",  # 7.5% loss
    82.5: "final_skitti_82.5",  # 17.5% loss
    72.5: "final_skitti_72.5",  # 27.5% loss
    62.5: "final_skitti_62.5",  # 37.5% loss
}

# Benchmark loss level to DEPOCO model mapping
# Maps benchmark loss percentage to (model_name, actual_loss)
# NOTE: Model names are counterintuitive! Higher number = MORE compression
LOSS_LEVEL_MAPPING = {
    10: ("final_skitti_72.5", 12),     # ~12% actual loss (closest to 10%)
    30: ("final_skitti_82.5", 33),     # ~33% actual loss (closest to 30%)
    70: ("final_skitti_92.5", 73),     # ~73% actual loss (closest to 70%)
}


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_model_for_loss_level(loss_percentage: int) -> Tuple[Optional[str], Optional[float]]:
    """
    Get the appropriate DEPOCO model for a target loss level.

    Parameters
    ----------
    loss_percentage : int
        Target loss percentage (10, 30, 50, 70, 90)

    Returns
    -------
    model_name : str or None
        Name of the DEPOCO model to use, or None if not available
    actual_loss : float or None
        Actual loss percentage of the model

    Examples
    --------
    >>> model, actual = get_model_for_loss_level(30)
    >>> print(f"Using {model} with {actual}% loss")
    Using final_skitti_72.5 with 27.5% loss
    """
    if loss_percentage not in LOSS_LEVEL_MAPPING:
        # Find closest available loss level
        available_losses = [k for k, v in LOSS_LEVEL_MAPPING.items() if v is not None]
        if not available_losses:
            return None, None
        closest = min(available_losses, key=lambda x: abs(x - loss_percentage))
        mapping = LOSS_LEVEL_MAPPING[closest]
    else:
        mapping = LOSS_LEVEL_MAPPING[loss_percentage]

    if mapping is None:
        return None, None

    return mapping[0], mapping[1]


def get_model_paths(model_name: str) -> Dict[str, str]:
    """
    Get paths to encoder and decoder model files.

    Parameters
    ----------
    model_name : str
        Name of the DEPOCO model (e.g., "final_skitti_72.5")

    Returns
    -------
    paths : dict
        Dictionary with 'encoder', 'decoder', 'config' paths
    """
    model_dir = f"{DEPOCO_MODELS_PATH}/{model_name}"
    return {
        'encoder': f"{model_dir}/enc_best.pth",
        'decoder': f"{model_dir}/dec_best.pth",
        'config': f"{DEPOCO_YAMLS_PATH}/{model_name}.yaml",
        'model_dir': model_dir,
    }


def check_depoco_available() -> Tuple[bool, str]:
    """
    Check if DEPOCO is available and properly configured.

    Returns
    -------
    available : bool
        True if DEPOCO is ready to use
    message : str
        Status message
    """
    checks = []

    # Check base path
    if not os.path.isdir(DEPOCO_BASE_PATH):
        return False, f"DEPOCO base path not found: {DEPOCO_BASE_PATH}"
    checks.append("Base path OK")

    # Check venv
    venv_python = f"{DEPOCO_VENV_PATH}/bin/python"
    if not os.path.isfile(venv_python):
        return False, f"DEPOCO venv not found: {DEPOCO_VENV_PATH}"
    checks.append("Venv OK")

    # Check at least one model exists
    models_found = []
    for retention, model_name in AVAILABLE_MODELS.items():
        paths = get_model_paths(model_name)
        if os.path.isfile(paths['encoder']) and os.path.isfile(paths['decoder']):
            models_found.append(model_name)

    if not models_found:
        return False, f"No DEPOCO models found in {DEPOCO_MODELS_PATH}"
    checks.append(f"Models OK ({len(models_found)} found)")

    return True, f"DEPOCO available: {', '.join(checks)}"


def list_available_loss_levels() -> List[int]:
    """
    List loss levels that have available DEPOCO models.

    Returns
    -------
    levels : list of int
        Available loss levels (subset of [10, 30, 50, 70, 90])
    """
    available = []
    for loss_level, mapping in LOSS_LEVEL_MAPPING.items():
        if mapping is not None:
            model_name = mapping[0]
            paths = get_model_paths(model_name)
            if os.path.isfile(paths['encoder']) and os.path.isfile(paths['decoder']):
                available.append(loss_level)
    return sorted(available)


# ==============================================================================
# DEPOCO Inference Functions
# ==============================================================================

def reassign_labels_nn(
    original_points: np.ndarray,
    original_labels: np.ndarray,
    compressed_points: np.ndarray
) -> np.ndarray:
    """
    Reassign labels to compressed points using nearest neighbor.

    DEPOCO reconstructs geometry but doesn't preserve point-to-point
    correspondence. Labels must be reassigned from original points.

    Parameters
    ----------
    original_points : np.ndarray, shape (N, 3)
        Original point cloud coordinates
    original_labels : np.ndarray, shape (N,)
        Labels for original points
    compressed_points : np.ndarray, shape (M, 3)
        Compressed point cloud coordinates

    Returns
    -------
    labels : np.ndarray, shape (M,)
        Labels for compressed points (from nearest original point)
    """
    tree = cKDTree(original_points)
    _, indices = tree.query(compressed_points, k=1)
    return original_labels[indices]


def reassign_features_nn(
    original_points: np.ndarray,
    original_features: np.ndarray,
    compressed_points: np.ndarray
) -> np.ndarray:
    """
    Reassign features to compressed points using nearest neighbor.

    Parameters
    ----------
    original_points : np.ndarray, shape (N, 3)
        Original point cloud coordinates
    original_features : np.ndarray, shape (N, F)
        Features for original points (e.g., intensity)
    compressed_points : np.ndarray, shape (M, 3)
        Compressed point cloud coordinates

    Returns
    -------
    features : np.ndarray, shape (M, F)
        Features for compressed points
    """
    tree = cKDTree(original_points)
    _, indices = tree.query(compressed_points, k=1)
    return original_features[indices]


# ==============================================================================
# SemanticKITTI I/O Functions
# ==============================================================================

def load_semantickitti_scan(bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a SemanticKITTI scan from binary file.

    Parameters
    ----------
    bin_path : str
        Path to .bin file

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        XYZ coordinates
    intensity : np.ndarray, shape (N,)
        Intensity values
    """
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]
    intensity = scan[:, 3]
    return points, intensity


def load_semantickitti_labels(label_path: str) -> np.ndarray:
    """
    Load SemanticKITTI labels from binary file.

    Parameters
    ----------
    label_path : str
        Path to .label file

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Semantic labels (lower 16 bits) and instance IDs (upper 16 bits)
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels


def save_semantickitti_scan(
    bin_path: str,
    points: np.ndarray,
    intensity: np.ndarray
) -> None:
    """
    Save a SemanticKITTI scan to binary file.

    Parameters
    ----------
    bin_path : str
        Output path for .bin file
    points : np.ndarray, shape (N, 3)
        XYZ coordinates
    intensity : np.ndarray, shape (N,)
        Intensity values
    """
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    scan = np.hstack([points, intensity.reshape(-1, 1)]).astype(np.float32)
    scan.tofile(bin_path)


def save_semantickitti_labels(label_path: str, labels: np.ndarray) -> None:
    """
    Save SemanticKITTI labels to binary file.

    Parameters
    ----------
    label_path : str
        Output path for .label file
    labels : np.ndarray, shape (N,)
        Label values
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    labels.astype(np.uint32).tofile(label_path)


# ==============================================================================
# Main DEPOCO Class
# ==============================================================================

class DEPOCOSubsampling:
    """
    DEPOCO: Deep Point Cloud Compression

    This class provides the interface for DEPOCO-based subsampling.
    It requires running in the DEPOCO virtual environment with proper
    dependencies installed.

    Parameters
    ----------
    loss_percentage : int
        Target loss percentage (10, 30, or 50)
    device : str, default='cuda'
        Device for inference ('cuda' or 'cpu')

    Attributes
    ----------
    model_name : str
        Name of the DEPOCO model being used
    actual_loss : float
        Actual loss percentage of the model
    encoder : torch.nn.Module
        Trained encoder network
    decoder : torch.nn.Module
        Trained decoder network

    Examples
    --------
    >>> # Must run in DEPOCO venv with proper dependencies
    >>> depoco = DEPOCOSubsampling(loss_percentage=30)
    >>> points_out, labels_out = depoco.subsample(points, labels)

    Notes
    -----
    This class requires the DEPOCO environment and dependencies.
    For standalone usage, use the preprocessing script instead:

        scripts/preprocessing/generate_subsampled_depoco.py

    See Also
    --------
    docs/DEPOCO_REFERENCE.md : Complete documentation
    docs/DEPOCO_MODEL_INVENTORY.md : Model inventory
    """

    def __init__(
        self,
        loss_percentage: int = 30,
        device: str = 'cuda'
    ):
        """Initialize DEPOCO with pre-trained models."""
        # Get model for loss level
        self.model_name, self.actual_loss = get_model_for_loss_level(loss_percentage)

        if self.model_name is None:
            raise ValueError(
                f"No DEPOCO model available for {loss_percentage}% loss.\n"
                f"Available loss levels: {list_available_loss_levels()}"
            )

        self.loss_percentage = loss_percentage
        self.device = device
        self.paths = get_model_paths(self.model_name)

        # Check model files exist
        if not os.path.isfile(self.paths['encoder']):
            raise FileNotFoundError(f"Encoder not found: {self.paths['encoder']}")
        if not os.path.isfile(self.paths['decoder']):
            raise FileNotFoundError(f"Decoder not found: {self.paths['decoder']}")

        # Load models (requires DEPOCO environment)
        self._load_models()

    def _load_models(self):
        """Load encoder and decoder models."""
        try:
            import torch
            import yaml

            # Add DEPOCO to path
            sys.path.insert(0, DEPOCO_BASE_PATH)
            import network_blocks as network

            # Load config
            with open(self.paths['config'], 'r') as f:
                from ruamel.yaml import YAML
                yaml_loader = YAML()
                self.config = yaml_loader.load(f)

            # Create models
            self.encoder = network.Network(self.config['network']['encoder_blocks'])
            self.decoder = network.Network(self.config['network']['decoder_blocks'])

            # Load weights
            self.encoder.load_state_dict(torch.load(
                self.paths['encoder'],
                map_location=lambda storage, loc: storage
            ))
            self.decoder.load_state_dict(torch.load(
                self.paths['decoder'],
                map_location=lambda storage, loc: storage
            ))

            # Move to device
            self.encoder.to(self.device)
            self.decoder.to(self.device)

            # Set to evaluation mode
            self.encoder.eval()
            self.decoder.eval()

            print(f"Loaded DEPOCO model: {self.model_name}")
            print(f"  Target loss: {self.loss_percentage}%")
            print(f"  Actual loss: {self.actual_loss}%")

        except ImportError as e:
            raise ImportError(
                f"Failed to import DEPOCO dependencies: {e}\n"
                f"Make sure you're running in the DEPOCO environment:\n"
                f"  source {DEPOCO_VENV_PATH}/bin/activate"
            )

    def subsample(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Subsample point cloud using DEPOCO encoder-decoder.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Input 3D point coordinates
        labels : np.ndarray, shape (N,), optional
            Point labels (will be reassigned via nearest neighbor)
        intensity : np.ndarray, shape (N,), optional
            Point intensity values

        Returns
        -------
        compressed_points : np.ndarray, shape (M, 3)
            Compressed point coordinates
        compressed_labels : np.ndarray, shape (M,), optional
            Labels for compressed points (if labels provided)
        compressed_intensity : np.ndarray, shape (M,), optional
            Intensity for compressed points (if intensity provided)
        """
        import torch

        # Prepare input
        points_tensor = torch.from_numpy(points).float().to(self.device)
        features = torch.ones((len(points), 1), dtype=torch.float32, device=self.device)

        input_dict = {
            'points': points_tensor,
            'features': features
        }

        # Forward pass
        with torch.no_grad():
            out_dict = self.encoder(input_dict.copy())
            out_dict = self.decoder(out_dict)

            if out_dict is None:
                # Point cloud too sparse
                return (points, labels, intensity) if labels is not None else (points,)

            translation = out_dict['features'][:, :3]
            samples = out_dict['points']
            compressed = (samples + translation).cpu().numpy()

        # Reassign labels and intensity via nearest neighbor
        results = [compressed]

        if labels is not None:
            compressed_labels = reassign_labels_nn(points, labels, compressed)
            results.append(compressed_labels)

        if intensity is not None:
            compressed_intensity = reassign_features_nn(
                points, intensity.reshape(-1, 1), compressed
            ).flatten()
            results.append(compressed_intensity)

        return tuple(results) if len(results) > 1 else results[0]


# ==============================================================================
# Functional Interface
# ==============================================================================

def depoco_subsample(
    points: np.ndarray,
    loss_percentage: int = 30,
    labels: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, ...]:
    """
    Functional interface for DEPOCO subsampling.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Input point cloud
    loss_percentage : int, default=30
        Target loss percentage (10, 30, or 50)
    labels : np.ndarray, optional
        Point labels
    intensity : np.ndarray, optional
        Point intensity values
    device : str, default='cuda'
        Inference device

    Returns
    -------
    results : tuple
        (compressed_points,) or (compressed_points, compressed_labels, ...)

    Examples
    --------
    >>> compressed, labels_out = depoco_subsample(points, 30, labels)
    """
    depoco = DEPOCOSubsampling(loss_percentage=loss_percentage, device=device)
    return depoco.subsample(points, labels, intensity)


# ==============================================================================
# Module Info
# ==============================================================================

__all__ = [
    'DEPOCOSubsampling',
    'depoco_subsample',
    'get_model_for_loss_level',
    'get_model_paths',
    'check_depoco_available',
    'list_available_loss_levels',
    'reassign_labels_nn',
    'reassign_features_nn',
    'load_semantickitti_scan',
    'load_semantickitti_labels',
    'save_semantickitti_scan',
    'save_semantickitti_labels',
    'DEPOCO_BASE_PATH',
    'DEPOCO_VENV_PATH',
    'LOSS_LEVEL_MAPPING',
]


if __name__ == '__main__':
    print("=" * 80)
    print("DEPOCO: Deep Point Cloud Compression")
    print("=" * 80)
    print()

    # Check availability
    available, message = check_depoco_available()
    print(f"Status: {message}")
    print()

    if available:
        print("Available loss levels for SemanticKITTI:")
        for loss in list_available_loss_levels():
            model, actual = get_model_for_loss_level(loss)
            print(f"  {loss}% → {model} (actual: {actual}% loss)")
    else:
        print("DEPOCO is not available. Check:")
        print(f"  - Base path: {DEPOCO_BASE_PATH}")
        print(f"  - Venv path: {DEPOCO_VENV_PATH}")
        print(f"  - Models path: {DEPOCO_MODELS_PATH}")

    print()
    print("To use DEPOCO:")
    print(f"  1. Activate venv: source {DEPOCO_VENV_PATH}/bin/activate")
    print("  2. Run: python scripts/preprocessing/generate_subsampled_depoco.py")
    print()
    print("See docs/DEPOCO_REFERENCE.md for complete documentation")
    print("=" * 80)
