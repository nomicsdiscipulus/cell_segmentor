"""
PyTorch Dataset for BCCD cell segmentation.

Supports two inference approaches:
1. Padding: Process entire image at once
2. Tiling: Extract overlapping tiles for inference, random crops for training
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset

from .tiling import (
    calculate_tile_positions,
    extract_tiles,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP
)
from .transforms import get_transform


# =============================================================================
# Base Dataset
# =============================================================================

class BCCDDataset(Dataset):
    """
    Base BCCD dataset for cell segmentation.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        file_list: List of filenames to include (from split JSON)
        mode: "padding" or "tiling"
        train: If True, use random crops for tiling (vs full coverage)
        augment: If True, apply augmentations. Default: same as train
        tile_size: Tile size for tiling mode
        overlap: Overlap fraction for tiling mode (inference only)
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        file_list: Optional[List[str]] = None,
        mode: str = "padding",
        train: bool = False,
        augment: bool = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = DEFAULT_OVERLAP
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.train = train
        self.augment = augment if augment is not None else train
        self.tile_size = tile_size
        self.overlap = overlap

        # Get file list
        if file_list is not None:
            self.files = file_list
        else:
            # Use all files in directory
            self.files = sorted(
                [f.name for f in self.image_dir.glob("*.png")] +
                [f.name for f in self.image_dir.glob("*.jpg")]
            )

        # Validate files exist
        self._validate_files()

        # Get transform
        self.transform = get_transform(
            mode=mode,
            train=train,
            augment=self.augment,
            tile_size=tile_size
        )

    def _validate_files(self):
        """Check that all image and mask files exist."""
        missing_images = []
        missing_masks = []

        for f in self.files:
            # Check image
            img_path = self.image_dir / f
            if not img_path.exists():
                missing_images.append(f)

            # Check mask
            stem = Path(f).stem
            mask_path = self.mask_dir / f"{stem}.png"
            if not mask_path.exists():
                missing_masks.append(f)

        if missing_images:
            raise FileNotFoundError(
                f"Missing {len(missing_images)} images: {missing_images[:5]}..."
            )
        if missing_masks:
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks: {missing_masks[:5]}..."
            )

    def _get_image_path(self, filename: str) -> Path:
        """Get full path for image."""
        return self.image_dir / filename

    def _get_mask_path(self, filename: str) -> Path:
        """Get mask path corresponding to image (assumes .png mask)."""
        stem = Path(filename).stem
        return self.mask_dir / f"{stem}.png"

    def _load_image(self, filename: str) -> Image.Image:
        """Load image as RGB PIL Image."""
        path = self._get_image_path(filename)
        return Image.open(path).convert('RGB')

    def _load_mask(self, filename: str) -> np.ndarray:
        """Load mask as binary numpy array."""
        path = self._get_mask_path(filename)
        mask = Image.open(path).convert('L')
        mask_arr = np.array(mask) / 255.0
        return (mask_arr > 0.5).astype(np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item based on mode and train/inference.

        Returns dict with keys depending on mode:
        - padding: image, mask, padding_info, filename
        - tiling (train): image, mask, position, filename
        - tiling (inference): tiles, positions, grid_size, original_size, filename
        """
        filename = self.files[idx]
        img = self._load_image(filename)
        mask = self._load_mask(filename)

        if self.mode == "padding":
            return self._get_padding_item(img, mask, filename)
        else:  # tiling
            if self.train:
                return self._get_tiling_train_item(img, mask, filename)
            else:
                return self._get_tiling_inference_item(img, mask, filename)

    def _get_padding_item(
        self,
        img: Image.Image,
        mask: np.ndarray,
        filename: str
    ) -> Dict[str, Any]:
        """Get item for padding mode."""
        tensor, padding_info, padded_mask = self.transform(img, mask)

        return {
            "image": tensor,
            "mask": torch.from_numpy(padded_mask).float(),
            "padding_info": padding_info,
            "filename": filename,
            "original_size": img.size
        }

    def _get_tiling_train_item(
        self,
        img: Image.Image,
        mask: np.ndarray,
        filename: str
    ) -> Dict[str, Any]:
        """Get item for tiling mode (training with random crop)."""
        tensor, mask_tensor, position = self.transform(img, mask)

        return {
            "image": tensor,
            "mask": mask_tensor,
            "position": position,
            "filename": filename,
            "original_size": img.size
        }

    def _get_tiling_inference_item(
        self,
        img: Image.Image,
        mask: np.ndarray,
        filename: str
    ) -> Dict[str, Any]:
        """Get item for tiling mode (inference with full coverage)."""
        # Calculate tile positions
        positions, grid_size = calculate_tile_positions(
            img.size, self.tile_size, self.overlap
        )

        # Extract tiles
        tiles = extract_tiles(img, positions, self.tile_size)

        # Transform tiles
        tile_tensors = self.transform(tiles)

        return {
            "tiles": tile_tensors,
            "positions": positions,
            "grid_size": grid_size,
            "mask": torch.from_numpy(mask).float(),
            "filename": filename,
            "original_size": img.size
        }


# =============================================================================
# Dataset Factory
# =============================================================================

def create_datasets(
    data_root: Path,
    split_path: Path,
    mode: str = "padding",
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    val_random_crop: bool = True
) -> Tuple[BCCDDataset, BCCDDataset]:
    """
    Create train and validation datasets from split JSON.

    Args:
        data_root: Root directory containing train/original and train/mask
        split_path: Path to split JSON file
        mode: "padding" or "tiling"
        tile_size: Tile size for tiling mode
        overlap: Overlap for tiling inference
        val_random_crop: If True, validation uses random crops (for training).
            If False, uses full tile coverage (for evaluation).

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load split
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    train_files = split_data["train"]
    val_files = split_data["val"]

    # Paths
    image_dir = data_root / "train" / "original"
    mask_dir = data_root / "train" / "mask"

    # Create datasets
    train_dataset = BCCDDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=train_files,
        mode=mode,
        train=True,
        augment=True,
        tile_size=tile_size,
        overlap=overlap
    )

    # For validation during training: random crops, no augmentation
    # For evaluation: full tile coverage (set val_random_crop=False)
    val_dataset = BCCDDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=val_files,
        mode=mode,
        train=val_random_crop,  # True = random crops, False = full coverage
        augment=False,  # Never augment validation
        tile_size=tile_size,
        overlap=overlap
    )

    return train_dataset, val_dataset


def create_test_dataset(
    data_root: Path,
    mode: str = "padding",
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP
) -> BCCDDataset:
    """
    Create test dataset (for final evaluation only).

    Args:
        data_root: Root directory containing test/original and test/mask
        mode: "padding" or "tiling"
        tile_size: Tile size for tiling mode
        overlap: Overlap for tiling inference

    Returns:
        Test dataset
    """
    image_dir = data_root / "test" / "original"
    mask_dir = data_root / "test" / "mask"

    return BCCDDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=None,  # Use all files
        mode=mode,
        train=False,
        tile_size=tile_size,
        overlap=overlap
    )


# =============================================================================
# Collate Functions
# =============================================================================

def padding_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for padding mode.

    Note: Images may have different sizes after padding, so we can't
    stack them directly. Returns list of tensors.
    """
    return {
        "images": [item["image"] for item in batch],
        "masks": [item["mask"] for item in batch],
        "padding_infos": [item["padding_info"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch]
    }


def tiling_train_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for tiling training mode.

    All crops are same size, so we can stack them.
    """
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "masks": torch.stack([item["mask"] for item in batch]),
        "positions": [item["position"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch]
    }


def tiling_inference_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for tiling inference mode.

    Each image has different number of tiles, so we can't stack.
    """
    return {
        "tiles": [item["tiles"] for item in batch],
        "positions": [item["positions"] for item in batch],
        "grid_sizes": [item["grid_size"] for item in batch],
        "masks": [item["mask"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch]
    }


def get_collate_fn(mode: str, train: bool):
    """Get appropriate collate function."""
    if mode == "padding":
        return padding_collate_fn
    else:  # tiling
        if train:
            return tiling_train_collate_fn
        else:
            return tiling_inference_collate_fn
