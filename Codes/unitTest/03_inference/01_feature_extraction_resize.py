"""
Test DinoBloom feature extraction on BCCD dataset - RESIZE approach.

This script tests the DinoBloom model using direct image resizing:
1. Resize BCCD images (1600x1200) to 224x224
2. Extract patch features using DinoBloom
3. Visualize features using PCA

Note: This approach distorts aspect ratio and makes cells much smaller
than DinoBloom's training data. See 02_padding and 03_tile for alternatives.

Designed to run on CPU for local testing.
"""

import sys
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
# Script location: Codes/unitTest/03_inference/01_feature_extraction.py
# Project root:    Fd-To-Sg/

SCRIPT_DIR = Path(__file__).parent                    # 03_inference/
UNITTEST_DIR = SCRIPT_DIR.parent                      # unitTest/
CODES_DIR = UNITTEST_DIR.parent                       # Codes/
PROJECT_ROOT = CODES_DIR.parent                       # Fd-To-Sg/

# Add Codes to path for backbone import
sys.path.insert(0, str(CODES_DIR))

# Data paths
DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TEST_IMAGES_DIR = DATA_ROOT / "test" / "original"
TEST_MASKS_DIR = DATA_ROOT / "test" / "mask"

# Output to script-specific folder: outputs/01_feature_extraction/
SCRIPT_NAME = Path(__file__).stem  # "01_feature_extraction"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import random

# Import from our backbone wrapper
from backbone import load_dinobloom, extract_features, IMAGENET_MEAN, IMAGENET_STD

# =============================================================================
# Configuration
# =============================================================================

MODEL_SIZE = "small"  # "small", "base", or "large"
IMG_SIZE = 224  # Must be divisible by 14. Options: 224, 518, 728, etc.
PATCH_SIZE = 14
PATCH_NUM = IMG_SIZE // PATCH_SIZE  # 16x16 patches for 224, 37x37 for 518

# Output options
SAVE_RESIZED_IMAGES = True  # Save individual resized images
SAVE_COMPARISON = True      # Save side-by-side comparison (original vs resized vs features)


# =============================================================================
# Image Size Handling
# =============================================================================

def resize_image(img: Image.Image, target_size: int, method: str = "resize") -> Image.Image:
    """
    Resize image to target size using specified method.

    Args:
        img: PIL Image
        target_size: Target size (square output)
        method:
            - "resize": Direct resize (may distort aspect ratio)
            - "center_crop": Resize shortest side, then center crop
            - "pad": Resize longest side, then pad to square

    Returns:
        Resized PIL Image (target_size × target_size)
    """
    if method == "resize":
        return img.resize((target_size, target_size), Image.BILINEAR)

    elif method == "center_crop":
        # Resize so shortest side = target_size, then center crop
        w, h = img.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        return img.crop((left, top, left + target_size, top + target_size))

    elif method == "pad":
        # Resize so longest side = target_size, then pad
        w, h = img.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Pad to square (black padding)
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        left = (target_size - new_w) // 2
        top = (target_size - new_h) // 2
        padded.paste(img, (left, top))
        return padded

    else:
        raise ValueError(f"Unknown resize method: {method}")


def get_image_info(img_path: Path) -> dict:
    """Get image metadata."""
    img = Image.open(img_path)
    return {
        "path": img_path,
        "size": img.size,
        "aspect_ratio": img.size[0] / img.size[1],
        "format": img.format
    }


# =============================================================================
# Data Loading
# =============================================================================

def get_image_paths(images_dir: Path, num_samples: int = 4) -> list:
    """Get random sample of image paths from directory."""
    all_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if len(all_images) < num_samples:
        return all_images
    return random.sample(all_images, num_samples)


def load_and_preprocess_images(
    image_paths: list,
    transform,
    target_size: int = IMG_SIZE,
    resize_method: str = "resize"
) -> tuple:
    """
    Load images and apply preprocessing.

    Args:
        image_paths: List of image file paths
        transform: Torchvision transform for normalization
        target_size: Target image size (must be divisible by 14)
        resize_method: "resize", "center_crop", or "pad"

    Returns:
        tuple: (tensor of transformed images, list of original PIL images, list of resized PIL images)
    """
    images_original = []
    images_resized = []
    tensors = []

    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images_original.append(img.copy())

        img_resized = resize_image(img, target_size, method=resize_method)
        images_resized.append(img_resized)
        tensors.append(transform(img_resized))

    return torch.stack(tensors), images_original, images_resized


# =============================================================================
# Visualization
# =============================================================================

def save_resized_images(
    image_paths: list,
    images_resized: list,
    output_dir: Path,
    resize_method: str = "resize"
):
    """
    Save resized images to disk for inspection.

    Args:
        image_paths: Original image paths (for naming)
        images_resized: List of resized PIL images
        output_dir: Directory to save images
        resize_method: Method used for resizing (for filename)
    """
    resized_dir = output_dir / "resized_images"
    resized_dir.mkdir(parents=True, exist_ok=True)

    for path, img in zip(image_paths, images_resized):
        output_name = f"{path.stem}_{resize_method}_{img.size[0]}x{img.size[1]}.png"
        output_path = resized_dir / output_name
        img.save(output_path)
        print(f"  Saved: {output_path.name}")

    print(f"Resized images saved to: {resized_dir}")


def visualize_resize_comparison(
    image_paths: list,
    images_original: list,
    images_resized: list,
    output_path: Path,
    resize_method: str = "resize"
):
    """
    Create side-by-side comparison of original vs resized images.

    Args:
        image_paths: Original image paths (for titles)
        images_original: List of original PIL images
        images_resized: List of resized PIL images
        output_path: Path to save the comparison figure
        resize_method: Method used for resizing
    """
    num_images = len(images_original)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i, (path, orig, resized) in enumerate(zip(image_paths, images_original, images_resized)):
        # Original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original: {path.name}\n{orig.size[0]}x{orig.size[1]}")
        axes[i, 0].axis('off')

        # Resized
        axes[i, 1].imshow(resized)
        axes[i, 1].set_title(f"Resized ({resize_method})\n{resized.size[0]}x{resized.size[1]}")
        axes[i, 1].axis('off')

    plt.suptitle(f"Original vs Resized Images (method: {resize_method})", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()


def visualize_pca_features(features, images_for_plotting, output_path: Path, upsample: bool = True):
    """
    Visualize patch features using PCA reduction to RGB.

    Args:
        features: Tensor of shape (N, num_patches, embed_dim)
        images_for_plotting: List of PIL images
        output_path: Path to save the visualization
        upsample: If True, upsample feature map to match image size for better visualization
    """
    num_images = len(images_for_plotting)
    embed_dim = features.shape[-1]
    img_size = images_for_plotting[0].size[0]  # Assuming square images
    patch_num = int(np.sqrt(features.shape[1]))  # Infer patch grid size

    # Reshape features for PCA: (N * H * W, embed_dim)
    features_flat = features.reshape(num_images * patch_num * patch_num, embed_dim).cpu().numpy()

    # Fit PCA to reduce to 3 components (RGB)
    pca = PCA(n_components=3)
    pca.fit(features_flat)
    pca_features = pca.transform(features_flat)

    # Normalize each component to [0, 1]
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min() + 1e-8)

    # Reshape back to images
    pca_features_rgb = pca_features.reshape(num_images, patch_num, patch_num, 3)

    # Create visualization
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

    for i in range(num_images):
        # Resized input image
        axes[0, i].imshow(images_for_plotting[i])
        axes[0, i].set_title(f"Input ({img_size}x{img_size})")
        axes[0, i].axis('off')

        # PCA features (optionally upsampled)
        if upsample:
            # Upsample feature map to match input image size
            pca_img = Image.fromarray((pca_features_rgb[i] * 255).astype(np.uint8))
            pca_img_upsampled = pca_img.resize((img_size, img_size), Image.NEAREST)
            axes[1, i].imshow(pca_img_upsampled)
            axes[1, i].set_title(f"PCA Features\n({patch_num}x{patch_num} → {img_size}x{img_size})")
        else:
            axes[1, i].imshow(pca_features_rgb[i])
            axes[1, i].set_title(f"PCA Features ({patch_num}x{patch_num})")
        axes[1, i].axis('off')

    plt.suptitle(f"DinoBloom Feature Extraction (patch_size={PATCH_SIZE}, grid={patch_num}x{patch_num})", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

    return pca_features_rgb, patch_num


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DinoBloom Feature Extraction Test on BCCD Dataset")
    print("=" * 60)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model size:    {MODEL_SIZE}")
    print(f"  Image size:    {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Patch size:    {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Patch grid:    {PATCH_NUM}x{PATCH_NUM} = {PATCH_NUM**2} patches")

    print(f"\nPath Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data root:    {DATA_ROOT}")
    print(f"  Output:       {OUTPUT_DIR}")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test images directory not found: {TEST_IMAGES_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model using backbone wrapper
    print(f"\nLoading DinoBloom model (size={MODEL_SIZE})...")
    model = load_dinobloom(size=MODEL_SIZE, device="cpu")

    # Setup transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load test images
    resize_method = "resize"  # Options: "resize", "center_crop", "pad"
    print(f"\nLoading test images (resize_method={resize_method})...")
    image_paths = get_image_paths(TEST_IMAGES_DIR, num_samples=4)
    print(f"Selected {len(image_paths)} images:")
    for p in image_paths:
        info = get_image_info(p)
        print(f"  - {p.name} ({info['size'][0]}x{info['size'][1]})")

    imgs_tensor, images_original, images_resized = load_and_preprocess_images(
        image_paths, transform, target_size=IMG_SIZE, resize_method=resize_method
    )
    print(f"Input tensor shape: {imgs_tensor.shape}")

    # Save resized images if enabled
    if SAVE_RESIZED_IMAGES:
        print("\nSaving resized images...")
        save_resized_images(image_paths, images_resized, OUTPUT_DIR, resize_method)

    # Save comparison visualization if enabled
    if SAVE_COMPARISON:
        print("\nGenerating resize comparison...")
        comparison_path = OUTPUT_DIR / "resize_comparison.png"
        visualize_resize_comparison(
            image_paths, images_original, images_resized, comparison_path, resize_method
        )

    # Extract features using backbone wrapper
    print("\nExtracting features...")
    features = extract_features(model, imgs_tensor)
    print(f"Features shape: {features.shape}")
    print(f"  - {features.shape[0]} images")
    print(f"  - {features.shape[1]} patches per image ({PATCH_NUM}x{PATCH_NUM})")
    print(f"  - {features.shape[2]} embedding dimension")

    # Visualize PCA features
    print("\nGenerating PCA visualization...")
    output_path = OUTPUT_DIR / "dinobloom_bccd_features.png"
    visualize_pca_features(features, images_resized, output_path, upsample=True)

    # Also save non-upsampled version for comparison
    output_path_raw = OUTPUT_DIR / "dinobloom_bccd_features_raw.png"
    visualize_pca_features(features, images_resized, output_path_raw, upsample=False)

    print("\nTest completed successfully!")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
