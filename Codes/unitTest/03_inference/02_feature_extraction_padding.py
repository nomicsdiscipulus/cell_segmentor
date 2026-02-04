"""
Test DinoBloom feature extraction on BCCD dataset - PADDING approach.

This script tests the DinoBloom model using image padding:
1. Pad BCCD images (1600x1200) to next multiple of 14 (1610x1204)
2. Extract patch features at original scale
3. Visualize features using PCA

Note: This preserves cell scale but requires processing much larger images.
The model may not have been trained with such large inputs.

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

from backbone import load_dinobloom, extract_features, IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
# Configuration
# =============================================================================

MODEL_SIZE = "large"
PATCH_SIZE = 14

OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME / MODEL_SIZE

# =============================================================================
# Padding Functions
# =============================================================================

def pad_to_multiple(img: Image.Image, multiple: int = 14, mode: str = "reflect") -> tuple:
    """
    Pad image so both dimensions are divisible by multiple.

    Args:
        img: PIL Image
        multiple: Pad to this multiple (default 14 for DinoBloom)
        mode: Padding mode - "reflect", "edge", or "constant"

    Returns:
        tuple: (padded_image, padding_info)
            padding_info: dict with original size and padding amounts
    """
    w, h = img.size

    # Calculate padding needed
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple

    new_w = w + pad_w
    new_h = h + pad_h

    # Convert to numpy for padding
    img_arr = np.array(img)

    # Pad: ((top, bottom), (left, right), (channels))
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if mode == "reflect":
        padded = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
    elif mode == "edge":
        padded = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
    else:  # constant (black)
        padded = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    padding_info = {
        "original_size": (w, h),
        "padded_size": (new_w, new_h),
        "padding": (pad_top, pad_bottom, pad_left, pad_right),
        "patch_grid": (new_w // multiple, new_h // multiple)
    }

    return Image.fromarray(padded), padding_info


def get_image_paths(images_dir: Path, num_samples: int = 2) -> list:
    """Get random sample of image paths from directory."""
    all_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if len(all_images) < num_samples:
        return all_images
    return random.sample(all_images, num_samples)


# =============================================================================
# Visualization
# =============================================================================

def visualize_pca_features_fullres(features, img_original, img_padded, padding_info, output_path: Path):
    """
    Visualize PCA features at full resolution.
    """
    embed_dim = features.shape[-1]
    patch_grid_w, patch_grid_h = padding_info["patch_grid"]
    num_patches = patch_grid_w * patch_grid_h

    # Reshape features for PCA
    features_flat = features.reshape(num_patches, embed_dim).cpu().numpy()

    # PCA to 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)

    # Normalize to [0, 1]
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min() + 1e-8)

    # Reshape to grid
    pca_features_rgb = pca_features.reshape(patch_grid_h, patch_grid_w, 3)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original image
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title(f"Original\n{img_original.size[0]}x{img_original.size[1]}")
    axes[0, 0].axis('off')

    # Padded image
    axes[0, 1].imshow(img_padded)
    axes[0, 1].set_title(f"Padded (reflect)\n{img_padded.size[0]}x{img_padded.size[1]}")
    axes[0, 1].axis('off')

    # PCA features (raw grid)
    axes[1, 0].imshow(pca_features_rgb)
    axes[1, 0].set_title(f"PCA Features (raw)\n{patch_grid_w}x{patch_grid_h} patches")
    axes[1, 0].axis('off')

    # PCA features upsampled to padded image size
    pca_img = Image.fromarray((pca_features_rgb * 255).astype(np.uint8))
    pca_upsampled = pca_img.resize(img_padded.size, Image.NEAREST)
    axes[1, 1].imshow(pca_upsampled)
    axes[1, 1].set_title(f"PCA Features (upsampled)\n{img_padded.size[0]}x{img_padded.size[1]}")
    axes[1, 1].axis('off')

    plt.suptitle(f"DinoBloom Feature Extraction - Padding Approach\nPatch grid: {patch_grid_w}x{patch_grid_h} = {num_patches} patches", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return pca_features_rgb


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DinoBloom Feature Extraction - PADDING Approach")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Model size: {MODEL_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Output:     {OUTPUT_DIR}")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test images not found: {TEST_IMAGES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading DinoBloom model (size={MODEL_SIZE})...")
    model = load_dinobloom(size=MODEL_SIZE, device="cpu")

    # Transform (no resize, just normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Get test images
    image_paths = get_image_paths(TEST_IMAGES_DIR, num_samples=2)
    print(f"\nProcessing {len(image_paths)} images...")

    for img_path in image_paths:
        print(f"\n{'='*40}")
        print(f"Image: {img_path.name}")

        # Load image
        img_original = Image.open(img_path).convert('RGB')
        print(f"  Original size: {img_original.size}")

        # Pad to multiple of 14
        img_padded, padding_info = pad_to_multiple(img_original, multiple=PATCH_SIZE)
        print(f"  Padded size:   {padding_info['padded_size']}")
        print(f"  Patch grid:    {padding_info['patch_grid'][0]}x{padding_info['patch_grid'][1]}")
        print(f"  Total patches: {padding_info['patch_grid'][0] * padding_info['patch_grid'][1]}")

        # Transform
        img_tensor = transform(img_padded).unsqueeze(0)  # Add batch dim
        print(f"  Tensor shape:  {img_tensor.shape}")

        # Extract features
        print("  Extracting features...")
        features = extract_features(model, img_tensor)
        print(f"  Features shape: {features.shape}")

        # Visualize
        output_path = OUTPUT_DIR / f"{img_path.stem}_padding_features.png"
        visualize_pca_features_fullres(features[0], img_original, img_padded, padding_info, output_path)

    print("\n" + "=" * 60)
    print("Padding approach completed!")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
