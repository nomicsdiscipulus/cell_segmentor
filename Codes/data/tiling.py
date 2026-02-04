"""
Tiling and stitching utilities for BCCD cell segmentation.

Handles:
1. Padding images to be divisible by patch size (14)
2. Extracting overlapping tiles for inference
3. Random cropping for training
4. Stitching tiles back with Gaussian blending
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional


# =============================================================================
# Constants
# =============================================================================

PATCH_SIZE = 14  # DinoBloom patch size
DEFAULT_TILE_SIZE = 224  # Standard input size for DinoBloom
DEFAULT_OVERLAP = 0.75  # 75% overlap (stride=56) - reduces boundary artifacts
DEFAULT_GAUSSIAN_SIGMA = 0.7  # Higher sigma for smoother blending at tile edges


# =============================================================================
# Padding Functions
# =============================================================================

def pad_to_multiple(
    img: Image.Image,
    multiple: int = PATCH_SIZE,
    mode: str = "reflect"
) -> Tuple[Image.Image, dict]:
    """
    Pad image so both dimensions are divisible by multiple.

    Args:
        img: PIL Image to pad
        multiple: Dimensions must be divisible by this (default: 14)
        mode: Padding mode ('reflect', 'constant', 'edge')

    Returns:
        Tuple of (padded_image, padding_info)
        padding_info contains: original_size, padded_size, padding (top, bottom, left, right)
    """
    w, h = img.size

    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Distribute padding evenly (more on bottom/right if odd)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Convert to numpy for padding
    img_arr = np.array(img)

    # Handle grayscale vs RGB
    if img_arr.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    # Pad
    if mode == "constant":
        padded_arr = np.pad(img_arr, pad_width, mode='constant', constant_values=0)
    else:
        padded_arr = np.pad(img_arr, pad_width, mode=mode)

    # Convert back to PIL
    padded_img = Image.fromarray(padded_arr)

    padding_info = {
        "original_size": (w, h),
        "padded_size": padded_img.size,
        "padding": (pad_top, pad_bottom, pad_left, pad_right)
    }

    return padded_img, padding_info


def unpad(
    arr: np.ndarray,
    padding_info: dict
) -> np.ndarray:
    """
    Remove padding from array to restore original size.

    Args:
        arr: Padded array (H, W) or (H, W, C)
        padding_info: Dict from pad_to_multiple containing padding values

    Returns:
        Unpadded array with original dimensions
    """
    pad_top, pad_bottom, pad_left, pad_right = padding_info["padding"]
    h, w = arr.shape[:2]

    # Calculate slice indices
    top = pad_top
    bottom = h - pad_bottom if pad_bottom > 0 else h
    left = pad_left
    right = w - pad_right if pad_right > 0 else w

    return arr[top:bottom, left:right]


# =============================================================================
# Tile Position Calculation
# =============================================================================

def calculate_tile_positions(
    image_size: Tuple[int, int],
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Calculate positions for overlapping tiles to cover entire image.

    Args:
        image_size: (width, height) of image
        tile_size: Size of each tile (square)
        overlap: Overlap fraction between tiles (0.5 = 50%)

    Returns:
        Tuple of (positions, grid_size)
        positions: List of (x, y) top-left coordinates for each tile
        grid_size: (num_cols, num_rows) of tile grid
    """
    w, h = image_size
    stride = int(tile_size * (1 - overlap))

    # Calculate number of tiles needed
    # Ensure we cover the entire image
    n_cols = max(1, int(np.ceil((w - tile_size) / stride)) + 1)
    n_rows = max(1, int(np.ceil((h - tile_size) / stride)) + 1)

    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = min(col * stride, w - tile_size)
            y = min(row * stride, h - tile_size)
            # Clamp to valid range
            x = max(0, x)
            y = max(0, y)
            positions.append((x, y))

    return positions, (n_cols, n_rows)


# =============================================================================
# Tile Extraction
# =============================================================================

def extract_tiles(
    img: Image.Image,
    positions: List[Tuple[int, int]],
    tile_size: int = DEFAULT_TILE_SIZE
) -> List[Image.Image]:
    """
    Extract tiles from image at specified positions.

    Args:
        img: PIL Image
        positions: List of (x, y) top-left coordinates
        tile_size: Size of each tile

    Returns:
        List of PIL Image tiles
    """
    tiles = []
    for (x, y) in positions:
        tile = img.crop((x, y, x + tile_size, y + tile_size))
        tiles.append(tile)
    return tiles


def random_crop(
    img: Image.Image,
    crop_size: int = DEFAULT_TILE_SIZE
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Extract a random crop from image for training.

    Args:
        img: PIL Image
        crop_size: Size of crop (square)

    Returns:
        Tuple of (cropped_image, (x, y) position)
    """
    w, h = img.size

    # Handle case where image is smaller than crop size
    if w < crop_size or h < crop_size:
        # Pad image first
        img, _ = pad_to_multiple(img, crop_size, mode="reflect")
        w, h = img.size

    # Random position
    max_x = w - crop_size
    max_y = h - crop_size

    x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
    y = np.random.randint(0, max_y + 1) if max_y > 0 else 0

    crop = img.crop((x, y, x + crop_size, y + crop_size))

    return crop, (x, y)


# =============================================================================
# Gaussian Blending
# =============================================================================

def create_gaussian_weight_map(
    tile_size: int = DEFAULT_TILE_SIZE,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA
) -> np.ndarray:
    """
    Create 2D Gaussian weight map for tile blending.

    Pixels near tile center get weight ~1, edges get weight ~0.
    Higher sigma (0.7) provides smoother blending to reduce
    "weaved cable" artifacts at tile boundaries.

    Args:
        tile_size: Size of tile (square)
        sigma: Gaussian sigma (default 0.7, smaller = sharper falloff)

    Returns:
        Weight map of shape (tile_size, tile_size)
    """
    x = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, x)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weights.astype(np.float32)


def stitch_tiles(
    tiles: List[np.ndarray],
    positions: List[Tuple[int, int]],
    output_size: Tuple[int, int],
    tile_size: int = DEFAULT_TILE_SIZE,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA
) -> np.ndarray:
    """
    Stitch tiles back together with Gaussian blending.

    Args:
        tiles: List of tile arrays, shape (H, W) or (H, W, C)
        positions: List of (x, y) top-left coordinates
        output_size: (width, height) of output image
        tile_size: Size of each tile
        sigma: Gaussian sigma for blending

    Returns:
        Stitched array of shape (H, W) or (H, W, C)
    """
    w, h = output_size
    weight_map = create_gaussian_weight_map(tile_size, sigma)

    # Determine output shape
    sample_tile = tiles[0]
    if sample_tile.ndim == 2:
        output = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
    else:
        c = sample_tile.shape[2]
        output = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

    # Accumulate weighted tiles
    for tile, (x, y) in zip(tiles, positions):
        # Handle edge cases where tile extends beyond image
        tile_h, tile_w = tile.shape[:2]

        # Clip to output bounds
        out_x_end = min(x + tile_w, w)
        out_y_end = min(y + tile_h, h)
        tile_x_end = out_x_end - x
        tile_y_end = out_y_end - y

        if sample_tile.ndim == 2:
            output[y:out_y_end, x:out_x_end] += (
                tile[:tile_y_end, :tile_x_end] *
                weight_map[:tile_y_end, :tile_x_end]
            )
        else:
            output[y:out_y_end, x:out_x_end] += (
                tile[:tile_y_end, :tile_x_end] *
                weight_map[:tile_y_end, :tile_x_end, np.newaxis]
            )
        weight_sum[y:out_y_end, x:out_x_end] += weight_map[:tile_y_end, :tile_x_end]

    # Normalize by weight sum (avoid division by zero)
    weight_sum = np.maximum(weight_sum, 1e-8)
    if sample_tile.ndim == 2:
        output = output / weight_sum
    else:
        output = output / weight_sum[:, :, np.newaxis]

    return output


# =============================================================================
# Mask Utilities
# =============================================================================

def resize_mask_to_patches(
    mask: np.ndarray,
    patch_size: int = PATCH_SIZE
) -> np.ndarray:
    """
    Resize mask to patch resolution using max pooling.

    For a 224x224 mask with patch_size=14, output is 16x16.
    Each output pixel is 1 if any pixel in the corresponding patch is 1.

    Args:
        mask: Binary mask of shape (H, W)
        patch_size: Size of each patch

    Returns:
        Downsampled mask of shape (H//patch_size, W//patch_size)
    """
    h, w = mask.shape
    new_h = h // patch_size
    new_w = w // patch_size

    # Reshape and take max over each patch
    reshaped = mask[:new_h * patch_size, :new_w * patch_size]
    reshaped = reshaped.reshape(new_h, patch_size, new_w, patch_size)
    pooled = reshaped.max(axis=(1, 3))

    return pooled.astype(np.float32)


def extract_tile_masks(
    mask: np.ndarray,
    positions: List[Tuple[int, int]],
    tile_size: int = DEFAULT_TILE_SIZE
) -> List[np.ndarray]:
    """
    Extract mask tiles at specified positions.

    Args:
        mask: Full mask array of shape (H, W)
        positions: List of (x, y) top-left coordinates
        tile_size: Size of each tile

    Returns:
        List of mask tiles
    """
    tiles = []
    for (x, y) in positions:
        tile = mask[y:y + tile_size, x:x + tile_size]
        tiles.append(tile)
    return tiles
