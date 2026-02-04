"""
Test DinoBloom feature extraction on BCCD dataset - TILING approach.

This script tests the DinoBloom model using sliding window tiling:
1. Cut BCCD images (1600x1200) into overlapping 224x224 tiles
2. Extract patch features for each tile
3. Stitch features back using Gaussian blending
4. Visualize full stitched feature map

This preserves cell scale and is consistent with DinoBloom training.

Designed to run on CPU for local testing.
"""

import sys
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
UNITTEST_DIR = SCRIPT_DIR.parent
CODES_DIR = UNITTEST_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

sys.path.insert(0, str(CODES_DIR))

DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TEST_IMAGES_DIR = DATA_ROOT / "test" / "original"
TEST_MASKS_DIR = DATA_ROOT / "test" / "mask"

SCRIPT_NAME = Path(__file__).stem
# Note: OUTPUT_DIR is set after MODEL_SIZE is defined

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import random
from math import ceil

from backbone import load_dinobloom, extract_features, IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
# Configuration
# =============================================================================

MODEL_SIZE = "large"
TILE_SIZE = 224
PATCH_SIZE = 14
OVERLAP = 0.75  # 75% overlap (higher overlap reduces boundary artifacts)
STRIDE = int(TILE_SIZE * (1 - OVERLAP))  # 56 pixels

OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME / MODEL_SIZE

# =============================================================================
# Tiling Functions
#
# NOTE: Tile-based inference can create boundary artifacts ("weaved cable" effect)
# because:
#   1. ViT edge effects: Attention at tile edges has less context (no pixels
#      beyond the tile boundary), causing features to differ systematically
#   2. Systematic pattern: Tile boundaries create repeating vertical/horizontal
#      artifacts at intervals equal to the stride
#   3. Blending limitations: Gaussian blending may not fully smooth edges
#
# Mitigations applied:
#   - 75% overlap (more redundancy at boundaries, was 50%)
#   - Higher Gaussian sigma=0.7 (smoother blending, was 0.5)
#   - Consider additional: crop tile edges, post-processing median filter
# =============================================================================

def calculate_tile_positions(img_size: tuple, tile_size: int = TILE_SIZE, stride: int = STRIDE) -> list:
    """
    Calculate tile positions to cover the entire image.

    Args:
        img_size: (width, height) of image
        tile_size: Size of each tile (square)
        stride: Step between tiles

    Returns:
        List of (x, y) positions for each tile's top-left corner
    """
    W, H = img_size
    positions = []

    # Calculate number of tiles needed
    n_tiles_x = ceil((W - tile_size) / stride) + 1
    n_tiles_y = ceil((H - tile_size) / stride) + 1

    for j in range(n_tiles_y):
        for i in range(n_tiles_x):
            x = min(i * stride, W - tile_size)
            y = min(j * stride, H - tile_size)
            positions.append((x, y))

    return positions, (n_tiles_x, n_tiles_y)


def extract_tiles(img: Image.Image, positions: list, tile_size: int = TILE_SIZE) -> list:
    """Extract tiles from image at given positions."""
    tiles = []
    for x, y in positions:
        tile = img.crop((x, y, x + tile_size, y + tile_size))
        tiles.append(tile)
    return tiles


def create_gaussian_weight_map(size: int = TILE_SIZE, sigma: float = 0.7) -> np.ndarray:
    """
    Create Gaussian weight map for blending.
    Center has weight ~1, edges have weight ~0.

    Higher sigma (0.7) provides smoother blending at tile boundaries
    to reduce "weaved cable" artifacts. Lower sigma (0.5) gives sharper
    center weighting but may show more boundary effects.
    """
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weight


def stitch_feature_maps(
    feature_maps: list,
    positions: list,
    img_size: tuple,
    patch_size: int = PATCH_SIZE,
    tile_size: int = TILE_SIZE
) -> np.ndarray:
    """
    Stitch tile feature maps into full image using Gaussian blending.

    Args:
        feature_maps: List of (H, W, 3) PCA feature maps per tile
        positions: List of (x, y) tile positions
        img_size: Original image size (W, H)
        patch_size: DinoBloom patch size (14)
        tile_size: Tile size (224)

    Returns:
        Stitched feature map at patch resolution
    """
    W, H = img_size
    patches_per_tile = tile_size // patch_size  # 16

    # Output size in patches
    out_w = ceil(W / patch_size)
    out_h = ceil(H / patch_size)

    # Initialize output and weight accumulator
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float32)

    # Create weight map for tile (in patch coordinates)
    weight_map = create_gaussian_weight_map(patches_per_tile, sigma=0.5)

    for feat_map, (x, y) in zip(feature_maps, positions):
        # Convert pixel position to patch position
        px = x // patch_size
        py = y // patch_size

        # Add weighted features
        for i in range(patches_per_tile):
            for j in range(patches_per_tile):
                out_y = py + i
                out_x = px + j
                if out_y < out_h and out_x < out_w:
                    w = weight_map[i, j]
                    output[out_y, out_x] += feat_map[i, j] * w
                    weight_sum[out_y, out_x] += w

    # Normalize by weight
    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(3):
        output[:, :, c] /= weight_sum

    return output


def get_image_paths(images_dir: Path, num_samples: int = 1) -> list:
    """Get random sample of image paths from directory."""
    all_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if len(all_images) < num_samples:
        return all_images
    return random.sample(all_images, num_samples)


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_tile_features(model, tiles: list, transform) -> list:
    """
    Extract features for all tiles.

    Returns list of (16, 16, embed_dim) feature tensors.
    """
    features_list = []
    patches_per_tile = TILE_SIZE // PATCH_SIZE

    for tile in tiles:
        tile_tensor = transform(tile).unsqueeze(0)
        features = extract_features(model, tile_tensor)  # (1, 256, embed_dim)
        features = features[0].cpu().numpy()  # (256, embed_dim)
        features = features.reshape(patches_per_tile, patches_per_tile, -1)
        features_list.append(features)

    return features_list


def features_to_pca_rgb(features_list: list) -> list:
    """
    Convert all tile features to PCA RGB images.
    PCA is fit on all tiles together for consistent colors.
    """
    # Stack all features for joint PCA
    all_features = []
    for feat in features_list:
        all_features.append(feat.reshape(-1, feat.shape[-1]))
    all_features = np.vstack(all_features)

    # Fit PCA on all features
    pca = PCA(n_components=3)
    pca.fit(all_features)

    # Transform each tile
    pca_maps = []
    idx = 0
    patches_per_tile = TILE_SIZE // PATCH_SIZE

    for feat in features_list:
        n_patches = patches_per_tile * patches_per_tile
        feat_flat = feat.reshape(-1, feat.shape[-1])
        pca_feat = pca.transform(feat_flat)

        # Normalize to [0, 1] using global min/max
        for i in range(3):
            pca_feat[:, i] = (pca_feat[:, i] - all_features[:, :3].min()) / \
                             (all_features[:, :3].max() - all_features[:, :3].min() + 1e-8)

        pca_feat = np.clip(pca_feat, 0, 1)
        pca_map = pca_feat.reshape(patches_per_tile, patches_per_tile, 3)
        pca_maps.append(pca_map)

    return pca_maps


# =============================================================================
# Visualization
# =============================================================================

def visualize_tiling_results(
    img_original: Image.Image,
    tiles: list,
    positions: list,
    stitched_features: np.ndarray,
    grid_size: tuple,
    output_path: Path
):
    """Create comprehensive visualization of tiling approach."""
    n_tiles_x, n_tiles_y = grid_size

    fig = plt.figure(figsize=(16, 12))

    # Original image with tile grid overlay
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_original)
    # Draw tile boundaries
    for x, y in positions[::4]:  # Show every 4th tile to avoid clutter
        rect = plt.Rectangle((x, y), TILE_SIZE, TILE_SIZE,
                              fill=False, edgecolor='red', linewidth=0.5, alpha=0.5)
        ax1.add_patch(rect)
    ax1.set_title(f"Original with tile grid\n{img_original.size[0]}x{img_original.size[1]}, {len(positions)} tiles")
    ax1.axis('off')

    # Sample tiles
    ax2 = fig.add_subplot(2, 2, 2)
    # Show 4 sample tiles in a 2x2 grid
    sample_indices = [0, len(tiles)//3, 2*len(tiles)//3, len(tiles)-1]
    sample_grid = np.zeros((TILE_SIZE*2, TILE_SIZE*2, 3), dtype=np.uint8)
    for idx, tile_idx in enumerate(sample_indices):
        r, c = idx // 2, idx % 2
        sample_grid[r*TILE_SIZE:(r+1)*TILE_SIZE, c*TILE_SIZE:(c+1)*TILE_SIZE] = np.array(tiles[tile_idx])
    ax2.imshow(sample_grid)
    ax2.set_title(f"Sample tiles (224x224 each)\nTile indices: {sample_indices}")
    ax2.axis('off')

    # Stitched features (raw resolution)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(stitched_features)
    ax3.set_title(f"Stitched PCA features (raw)\n{stitched_features.shape[1]}x{stitched_features.shape[0]} patches")
    ax3.axis('off')

    # Stitched features upsampled
    ax4 = fig.add_subplot(2, 2, 4)
    stitched_img = Image.fromarray((stitched_features * 255).astype(np.uint8))
    stitched_upsampled = stitched_img.resize(img_original.size, Image.NEAREST)
    ax4.imshow(stitched_upsampled)
    ax4.set_title(f"Stitched PCA features (upsampled)\n{img_original.size[0]}x{img_original.size[1]}")
    ax4.axis('off')

    plt.suptitle(f"DinoBloom Feature Extraction - Tiling Approach\n"
                 f"Tile: {TILE_SIZE}x{TILE_SIZE}, Stride: {STRIDE} ({int(OVERLAP*100)}% overlap), "
                 f"Grid: {n_tiles_x}x{n_tiles_y} = {len(positions)} tiles", fontsize=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DinoBloom Feature Extraction - TILING Approach")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Model size:  {MODEL_SIZE}")
    print(f"  Tile size:   {TILE_SIZE}x{TILE_SIZE}")
    print(f"  Stride:      {STRIDE} ({int(OVERLAP*100)}% overlap)")
    print(f"  Patch size:  {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Output:      {OUTPUT_DIR}")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test images not found: {TEST_IMAGES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading DinoBloom model (size={MODEL_SIZE})...")
    model = load_dinobloom(size=MODEL_SIZE, device="cpu")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Get test images
    image_paths = get_image_paths(TEST_IMAGES_DIR, num_samples=1)
    print(f"\nProcessing {len(image_paths)} images...")

    for img_path in image_paths:
        print(f"\n{'='*40}")
        print(f"Image: {img_path.name}")

        # Load image
        img_original = Image.open(img_path).convert('RGB')
        print(f"  Original size: {img_original.size}")

        # Calculate tile positions
        positions, grid_size = calculate_tile_positions(img_original.size)
        print(f"  Tile grid:     {grid_size[0]}x{grid_size[1]} = {len(positions)} tiles")

        # Extract tiles
        print("  Extracting tiles...")
        tiles = extract_tiles(img_original, positions)

        # Extract features for each tile
        print(f"  Extracting features ({len(tiles)} tiles)...")
        features_list = extract_tile_features(model, tiles, transform)
        print(f"  Features per tile: {features_list[0].shape}")

        # Convert to PCA RGB
        print("  Computing PCA...")
        pca_maps = features_to_pca_rgb(features_list)

        # Stitch together
        print("  Stitching with Gaussian blending...")
        stitched = stitch_feature_maps(pca_maps, positions, img_original.size)
        print(f"  Stitched shape: {stitched.shape}")

        # Visualize
        output_path = OUTPUT_DIR / f"{img_path.stem}_tiling_features.png"
        visualize_tiling_results(img_original, tiles, positions, stitched, grid_size, output_path)

    print("\n" + "=" * 60)
    print("Tiling approach completed!")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
