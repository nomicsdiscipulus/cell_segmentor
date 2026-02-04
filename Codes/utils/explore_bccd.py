"""
BCCD Dataset Exploration Script

This script analyzes the BCCD (Blood Cell Count and Detection) dataset:
1. Dataset structure and file counts
2. Image and mask properties (size, format, value range)
3. Mask analysis (binary vs instance, channel info)
4. Cell statistics (count per image, cell sizes)
5. Visualization of samples

Usage:
    python explore_bccd.py

Output:
    - Console statistics
    - bccd_samples.png: Visualization of sample images, masks, and overlays
    - bccd_cell_stats.png: Histogram of cell counts and sizes
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent          # utils/
CODES_DIR = SCRIPT_DIR.parent               # Codes/
PROJECT_ROOT = CODES_DIR.parent             # Fd-To-Sg/

DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TRAIN_IMAGES = DATA_ROOT / "train" / "original"
TRAIN_MASKS = DATA_ROOT / "train" / "mask"
TEST_IMAGES = DATA_ROOT / "test" / "original"
TEST_MASKS = DATA_ROOT / "test" / "mask"

# Output to script-specific folder: outputs/exploration/explore_bccd/
SCRIPT_NAME = Path(__file__).stem  # "explore_bccd"
OUTPUT_DIR = CODES_DIR / "outputs" / "exploration" / SCRIPT_NAME


# =============================================================================
# Dataset Structure Analysis
# =============================================================================

def count_files():
    """Count images and masks in train/test splits."""
    print("=" * 60)
    print("BCCD Dataset Structure")
    print("=" * 60)

    train_imgs = list(TRAIN_IMAGES.glob("*.png")) + list(TRAIN_IMAGES.glob("*.jpg"))
    train_masks = list(TRAIN_MASKS.glob("*.png"))
    test_imgs = list(TEST_IMAGES.glob("*.png")) + list(TEST_IMAGES.glob("*.jpg"))
    test_masks = list(TEST_MASKS.glob("*.png"))

    print(f"\nTrain set:")
    print(f"  Images: {len(train_imgs)}")
    print(f"  Masks:  {len(train_masks)}")

    print(f"\nTest set:")
    print(f"  Images: {len(test_imgs)}")
    print(f"  Masks:  {len(test_masks)}")

    print(f"\nTotal: {len(train_imgs) + len(test_imgs)} images")

    return train_imgs, train_masks, test_imgs, test_masks


# =============================================================================
# Image and Mask Properties
# =============================================================================

def analyze_image_properties(image_paths, name="Images"):
    """Analyze image sizes and formats."""
    print(f"\n{name} Properties:")
    print("-" * 40)

    sizes = []
    modes = []

    for img_path in image_paths:
        img = Image.open(img_path)
        sizes.append(img.size)
        modes.append(img.mode)

    # Size distribution
    size_counts = Counter(sizes)
    print(f"  Size distribution:")
    for size, count in size_counts.most_common():
        print(f"    {size[0]}x{size[1]}: {count} images ({100*count/len(sizes):.1f}%)")

    # Mode distribution
    mode_counts = Counter(modes)
    print(f"  Mode distribution:")
    for mode, count in mode_counts.most_common():
        print(f"    {mode}: {count} images")

    return sizes, modes


def analyze_sample_image_mask(img_path, mask_path):
    """Analyze a single image-mask pair in detail."""
    print("\n" + "=" * 60)
    print("Sample Image-Mask Analysis")
    print("=" * 60)

    # Load image
    img = Image.open(img_path)
    img_arr = np.array(img)

    print(f"\nImage: {img_path.name}")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    print(f"  Array shape: {img_arr.shape}")
    print(f"  Dtype: {img_arr.dtype}")
    print(f"  Value range: [{img_arr.min()}, {img_arr.max()}]")

    # Load mask
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)

    print(f"\nMask: {mask_path.name}")
    print(f"  Size: {mask.size}")
    print(f"  Mode: {mask.mode}")
    print(f"  Array shape: {mask_arr.shape}")
    print(f"  Dtype: {mask_arr.dtype}")
    print(f"  Value range: [{mask_arr.min()}, {mask_arr.max()}]")
    print(f"  Unique values: {np.unique(mask_arr)}")

    # Check if mask channels are identical
    if len(mask_arr.shape) == 3:
        channels_equal = (
            np.allclose(mask_arr[:,:,0], mask_arr[:,:,1]) and
            np.allclose(mask_arr[:,:,1], mask_arr[:,:,2])
        )
        print(f"  All channels equal: {channels_equal}")

        # Unique colors
        unique_colors = np.unique(mask_arr.reshape(-1, 3), axis=0)
        print(f"  Unique colors: {len(unique_colors)}")
        for color in unique_colors:
            print(f"    RGB{tuple(color)}")

    # Coverage statistics
    if len(mask_arr.shape) == 3:
        binary_mask = mask_arr[:,:,0] > 127
    else:
        binary_mask = mask_arr > 127

    cell_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    print(f"\nCoverage:")
    print(f"  Cell pixels: {cell_pixels} ({100*cell_pixels/total_pixels:.2f}%)")
    print(f"  Background pixels: {total_pixels - cell_pixels} ({100*(total_pixels-cell_pixels)/total_pixels:.2f}%)")

    return img_arr, mask_arr


# =============================================================================
# Cell Statistics
# =============================================================================

def analyze_cell_statistics(mask_paths, num_samples=None):
    """Analyze cell counts and sizes using connected components."""
    print("\n" + "=" * 60)
    print("Cell Statistics (Connected Components)")
    print("=" * 60)

    if num_samples:
        mask_paths = mask_paths[:num_samples]

    cell_counts = []
    all_cell_sizes = []

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        binary_mask = (mask > 127).astype(np.uint8)
        labeled_array, num_features = label(binary_mask)

        cell_counts.append(num_features)

        # Get cell sizes
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            all_cell_sizes.append(size)

    print(f"\nAnalyzed {len(mask_paths)} images")

    print(f"\nCells per image:")
    print(f"  Min:  {min(cell_counts)}")
    print(f"  Max:  {max(cell_counts)}")
    print(f"  Mean: {np.mean(cell_counts):.1f}")
    print(f"  Std:  {np.std(cell_counts):.1f}")

    print(f"\nCell sizes (pixels):")
    print(f"  Min:  {min(all_cell_sizes)}")
    print(f"  Max:  {max(all_cell_sizes)}")
    print(f"  Mean: {np.mean(all_cell_sizes):.0f}")
    print(f"  Median: {np.median(all_cell_sizes):.0f}")

    # Approximate diameters (assuming circular cells)
    diameters = [2 * np.sqrt(s / np.pi) for s in all_cell_sizes]
    print(f"\nApproximate cell diameters (pixels):")
    print(f"  Min:  {min(diameters):.1f}")
    print(f"  Max:  {max(diameters):.1f}")
    print(f"  Mean: {np.mean(diameters):.1f}")

    return cell_counts, all_cell_sizes


# =============================================================================
# Visualization
# =============================================================================

def visualize_samples(image_paths, mask_dir, output_path, num_samples=4):
    """Create visualization of sample images, masks, and overlays."""
    print(f"\nGenerating sample visualization...")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, img_path in enumerate(image_paths[:num_samples]):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # Try with .png extension
            mask_path = mask_dir / (img_path.stem + ".png")

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\n{img_path.name[:20]}...')
        axes[i, 0].axis('off')

        # Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Mask (Binary)')
        axes[i, 1].axis('off')

        # Overlay: blend original with red mask
        overlay = img.copy()
        overlay[mask > 127] = [255, 0, 0]  # Red for cells
        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title('Overlay (60% img + 40% red)')
        axes[i, 2].axis('off')

    plt.suptitle('BCCD Dataset: Images, Masks, and Overlays', fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def visualize_cell_statistics(cell_counts, cell_sizes, output_path):
    """Create histogram visualization of cell statistics."""
    print(f"\nGenerating cell statistics visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Cell counts histogram
    axes[0].hist(cell_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Cells')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title(f'Cells per Image\n(mean={np.mean(cell_counts):.1f})')
    axes[0].axvline(np.mean(cell_counts), color='r', linestyle='--', label='Mean')
    axes[0].legend()

    # Cell sizes histogram (log scale for better visualization)
    axes[1].hist(cell_sizes, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Cell Size (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Cell Size Distribution\n(median={np.median(cell_sizes):.0f} pixels)')
    axes[1].set_xscale('log')
    axes[1].axvline(np.median(cell_sizes), color='r', linestyle='--', label='Median')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("BCCD Dataset Exploration")
    print("=" * 60)
    print(f"\nData root: {DATA_ROOT}")

    # Check if data exists
    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        return

    # 1. Count files
    train_imgs, train_masks, test_imgs, test_masks = count_files()

    # 2. Analyze image properties
    print("\n" + "=" * 60)
    print("Image Properties Analysis")
    print("=" * 60)
    analyze_image_properties(train_imgs[:100], "Train Images (sample of 100)")
    analyze_image_properties(test_imgs, "Test Images (all)")

    # 3. Analyze sample image-mask pair
    if train_imgs and train_masks:
        sample_img = train_imgs[0]
        sample_mask = TRAIN_MASKS / sample_img.name
        if sample_mask.exists():
            analyze_sample_image_mask(sample_img, sample_mask)

    # 4. Cell statistics
    cell_counts, cell_sizes = analyze_cell_statistics(train_masks, num_samples=100)

    # 5. Visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    visualize_samples(
        train_imgs,
        TRAIN_MASKS,
        OUTPUT_DIR / "bccd_samples.png",
        num_samples=4
    )

    visualize_cell_statistics(
        cell_counts,
        cell_sizes,
        OUTPUT_DIR / "bccd_cell_stats.png"
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
Dataset: BCCD (Blood Cell Count and Detection)
- Train: {len(train_imgs)} images
- Test: {len(test_imgs)} images
- Image size: 1600x1200 (most common)
- Mask type: Binary (0=background, 255=cell)
- Cells per image: {min(cell_counts)}-{max(cell_counts)} (mean={np.mean(cell_counts):.0f})
- Cell sizes: {min(cell_sizes)}-{max(cell_sizes)} pixels

Key observations:
1. Masks are BINARY, not instance-labeled
2. Need connected components for instance separation
3. Large images (1600x1200) require resizing for model input
4. High cell counts (~80/image) with variable sizes
""")


if __name__ == "__main__":
    main()
