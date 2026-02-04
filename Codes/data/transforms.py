"""
Transforms and augmentations for BCCD cell segmentation.

Provides preprocessing for both Padding and Tiling approaches,
plus training augmentations.
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional, Callable

from .tiling import (
    pad_to_multiple,
    random_crop,
    PATCH_SIZE,
    DEFAULT_TILE_SIZE
)


# =============================================================================
# ImageNet Normalization (for DinoBloom)
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Base Transforms
# =============================================================================

def get_normalize_transform() -> transforms.Compose:
    """Get standard ImageNet normalization transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_denormalize_transform() -> Callable:
    """Get function to denormalize tensor back to [0, 1] range."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor from ImageNet normalization."""
        return tensor * std + mean

    return denormalize


# =============================================================================
# Padding Approach Transforms
# =============================================================================

class PaddingTransform:
    """
    Transform for padding approach.

    Pads image to be divisible by patch_size, then normalizes.
    """

    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        pad_mode: str = "reflect"
    ):
        self.patch_size = patch_size
        self.pad_mode = pad_mode
        self.normalize = get_normalize_transform()

    def __call__(
        self,
        img: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, dict, Optional[np.ndarray]]:
        """
        Apply padding transform.

        Args:
            img: PIL Image
            mask: Optional binary mask (H, W)

        Returns:
            Tuple of (tensor, padding_info, padded_mask)
        """
        # Pad image
        img_padded, padding_info = pad_to_multiple(
            img, self.patch_size, self.pad_mode
        )

        # Normalize
        tensor = self.normalize(img_padded)

        # Pad mask if provided
        padded_mask = None
        if mask is not None:
            pad_top, pad_bottom, pad_left, pad_right = padding_info["padding"]
            padded_mask = np.pad(
                mask,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )

        return tensor, padding_info, padded_mask


# =============================================================================
# Tiling Approach Transforms
# =============================================================================

class TileTransform:
    """
    Transform for tiling approach (inference).

    Extracts tiles at specified positions and normalizes each.
    """

    def __init__(self, tile_size: int = DEFAULT_TILE_SIZE):
        self.tile_size = tile_size
        self.normalize = get_normalize_transform()

    def __call__(
        self,
        tiles: list
    ) -> torch.Tensor:
        """
        Transform list of tile images to tensor batch.

        Args:
            tiles: List of PIL Image tiles

        Returns:
            Tensor of shape (N, C, H, W)
        """
        tensors = [self.normalize(tile) for tile in tiles]
        return torch.stack(tensors)


class RandomCropTransform:
    """
    Transform for tiling approach (training).

    Extracts random crop and applies augmentations.
    """

    def __init__(
        self,
        crop_size: int = DEFAULT_TILE_SIZE,
        augment: bool = True
    ):
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = get_normalize_transform()

        # Augmentation pipeline
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
            ])
        else:
            self.augmentation = None

    def __call__(
        self,
        img: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
        """
        Apply random crop transform with augmentation.

        Args:
            img: PIL Image
            mask: Optional binary mask (H, W)

        Returns:
            Tuple of (image_tensor, mask_tensor, (x, y) position)
        """
        # Random crop
        crop, (x, y) = random_crop(img, self.crop_size)

        # Crop mask if provided
        mask_crop = None
        if mask is not None:
            mask_crop = mask[y:y + self.crop_size, x:x + self.crop_size]

        # Apply augmentations
        if self.augment and self.augmentation is not None:
            # Set random state for synchronized augmentation
            seed = np.random.randint(2147483647)

            # Augment image
            torch.manual_seed(seed)
            crop = self.augmentation(crop)

            # Augment mask with same transforms (flip, rotation only)
            if mask_crop is not None:
                torch.manual_seed(seed)
                mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
                # Only geometric augmentations for mask
                mask_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=90),
                ])(mask_pil)
                mask_crop = np.array(mask_aug) / 255.0

        # Normalize image
        tensor = self.normalize(crop)

        # Convert mask to tensor
        mask_tensor = None
        if mask_crop is not None:
            mask_tensor = torch.from_numpy(mask_crop).float()

        return tensor, mask_tensor, (x, y)


# =============================================================================
# Combined Transform
# =============================================================================

def get_transform(
    mode: str = "padding",
    train: bool = False,
    augment: bool = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    patch_size: int = PATCH_SIZE,
    pad_mode: str = "reflect"
) -> Callable:
    """
    Factory function to get appropriate transform.

    Args:
        mode: "padding" or "tiling"
        train: If True, use random crops for tiling (vs full coverage)
        augment: If True, apply augmentations. Default: same as train
        tile_size: Tile size for tiling mode
        patch_size: Patch size for padding mode
        pad_mode: Padding mode for padding approach

    Returns:
        Transform callable
    """
    # Default: augment same as train
    if augment is None:
        augment = train

    if mode == "padding":
        return PaddingTransform(patch_size=patch_size, pad_mode=pad_mode)
    elif mode == "tiling":
        if train:
            # Random crops (with or without augmentation)
            return RandomCropTransform(crop_size=tile_size, augment=augment)
        else:
            # Full tile coverage for inference
            return TileTransform(tile_size=tile_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'padding' or 'tiling'.")
