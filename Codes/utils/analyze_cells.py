"""
BCCD Cell Analysis Script

Analyzes individual cells in the BCCD dataset:
1. Connected components analysis with size filtering
2. Cell type estimation based on size
3. Visualization of cells colored by size/type
4. Corrected size distribution (excluding noise)

Blood Cell Types:
- RBC (Red Blood Cells): ~6-8 μm, appear as medium-sized pink discs
- WBC (White Blood Cells): ~10-15 μm, larger with visible purple nucleus
- Platelets: ~2-3 μm, very small fragments

Usage:
    python analyze_cells.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import label, find_objects

# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TRAIN_IMAGES = DATA_ROOT / "train" / "original"
TRAIN_MASKS = DATA_ROOT / "train" / "mask"
# Output to script-specific folder: outputs/exploration/analyze_cells/
SCRIPT_NAME = Path(__file__).stem  # "analyze_cells"
OUTPUT_DIR = CODES_DIR / "outputs" / "exploration" / SCRIPT_NAME


# =============================================================================
# Cell Size Thresholds (in pixels, for 1600x1200 images)
# =============================================================================

# These thresholds are estimated based on typical blood cell sizes
# Actual values may need adjustment based on image magnification

NOISE_THRESHOLD = 100        # Below this = noise/artifact
PLATELET_MAX = 1000          # Platelets are very small
RBC_MIN = 1000               # RBC minimum size
RBC_MAX = 12000              # RBC maximum size
WBC_MIN = 8000               # WBC minimum (overlaps with large RBC)
MERGED_THRESHOLD = 20000     # Above this = likely merged/overlapping cells


# =============================================================================
# Analysis Functions
# =============================================================================

def get_connected_components(mask):
    """
    Get connected components from binary mask.

    Returns:
        labeled_array: Array with each component labeled 1, 2, 3, ...
        num_features: Number of components found
        sizes: Array of component sizes (in pixels)
        centroids: List of (row, col) centroid for each component
    """
    binary_mask = (mask > 127).astype(np.uint8)
    labeled_array, num_features = label(binary_mask)

    sizes = []
    centroids = []

    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i)
        sizes.append(np.sum(component_mask))

        # Calculate centroid
        rows, cols = np.where(component_mask)
        centroids.append((rows.mean(), cols.mean()))

    return labeled_array, num_features, np.array(sizes), centroids


def classify_cell_by_size(size):
    """
    Estimate cell type based on size in pixels.

    Returns:
        str: 'noise', 'platelet', 'rbc', 'wbc', or 'merged'
    """
    if size < NOISE_THRESHOLD:
        return 'noise'
    elif size < PLATELET_MAX:
        return 'platelet'
    elif size < RBC_MAX:
        return 'rbc'
    elif size < MERGED_THRESHOLD:
        return 'wbc'
    else:
        return 'merged'


def analyze_single_image(img_path, mask_path):
    """Analyze cells in a single image."""
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    labeled, num_cells, sizes, centroids = get_connected_components(mask)

    # Classify each cell
    classifications = [classify_cell_by_size(s) for s in sizes]

    # Count by type
    type_counts = {}
    for cls in ['noise', 'platelet', 'rbc', 'wbc', 'merged']:
        type_counts[cls] = classifications.count(cls)

    return {
        'image': img,
        'mask': mask,
        'labeled': labeled,
        'num_cells': num_cells,
        'sizes': sizes,
        'centroids': centroids,
        'classifications': classifications,
        'type_counts': type_counts
    }


def analyze_dataset_sizes(mask_paths, num_samples=100):
    """Analyze cell sizes across multiple images."""
    all_sizes = []
    all_classifications = []

    for mask_path in mask_paths[:num_samples]:
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        _, _, sizes, _ = get_connected_components(mask)
        all_sizes.extend(sizes)
        all_classifications.extend([classify_cell_by_size(s) for s in sizes])

    return np.array(all_sizes), all_classifications


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_cells_by_type(analysis_result, output_path):
    """
    Create visualization showing cells colored by estimated type.
    """
    img = analysis_result['image']
    labeled = analysis_result['labeled']
    sizes = analysis_result['sizes']
    classifications = analysis_result['classifications']
    type_counts = analysis_result['type_counts']

    # Color map for cell types
    colors = {
        'noise': [128, 128, 128],     # Gray
        'platelet': [255, 255, 0],    # Yellow
        'rbc': [255, 0, 0],           # Red
        'wbc': [0, 0, 255],           # Blue
        'merged': [255, 0, 255]       # Magenta
    }

    # Create colored overlay
    overlay = np.zeros_like(img)
    for i, (size, cls) in enumerate(zip(sizes, classifications), start=1):
        mask_i = (labeled == i)
        overlay[mask_i] = colors[cls]

    # Blend with original
    blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Colored by type
    axes[1].imshow(blended)
    axes[1].set_title('Cells Colored by Estimated Type')
    axes[1].axis('off')

    # Add legend
    legend_elements = [
        Patch(facecolor=np.array(colors['rbc'])/255, label=f"RBC ({type_counts['rbc']})"),
        Patch(facecolor=np.array(colors['wbc'])/255, label=f"WBC ({type_counts['wbc']})"),
        Patch(facecolor=np.array(colors['platelet'])/255, label=f"Platelet ({type_counts['platelet']})"),
        Patch(facecolor=np.array(colors['merged'])/255, label=f"Merged ({type_counts['merged']})"),
        Patch(facecolor=np.array(colors['noise'])/255, label=f"Noise ({type_counts['noise']})"),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Size histogram for this image
    valid_sizes = sizes[sizes >= NOISE_THRESHOLD]
    axes[2].hist(valid_sizes, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(RBC_MAX, color='r', linestyle='--', label=f'RBC/WBC boundary ({RBC_MAX})')
    axes[2].set_xlabel('Cell Size (pixels)')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Cell Size Distribution\n(excluding noise, n={len(valid_sizes)})')
    axes[2].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_size_distribution(all_sizes, all_classifications, output_path):
    """
    Create corrected size distribution visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Raw size distribution (log scale)
    axes[0, 0].hist(all_sizes, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cell Size (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Raw Size Distribution (All Components)')
    axes[0, 0].set_xscale('log')

    # 2. Filtered size distribution (excluding noise)
    valid_sizes = all_sizes[all_sizes >= NOISE_THRESHOLD]
    axes[0, 1].hist(valid_sizes, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Cell Size (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Filtered Distribution (size >= {NOISE_THRESHOLD})\nn={len(valid_sizes)} cells')

    # Add threshold lines
    for threshold, name in [(RBC_MAX, 'RBC/WBC'), (MERGED_THRESHOLD, 'Merged')]:
        if threshold < valid_sizes.max():
            axes[0, 1].axvline(threshold, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].text(threshold, axes[0, 1].get_ylim()[1]*0.9, name, rotation=90, va='top')

    # 3. Count by classification
    from collections import Counter
    class_counts = Counter(all_classifications)
    classes = ['noise', 'platelet', 'rbc', 'wbc', 'merged']
    counts = [class_counts.get(c, 0) for c in classes]
    colors = ['gray', 'yellow', 'red', 'blue', 'magenta']

    bars = axes[1, 0].bar(classes, counts, color=colors, edgecolor='black')
    axes[1, 0].set_xlabel('Cell Type (Estimated)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Components by Estimated Type')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        str(count), ha='center', va='bottom')

    # 4. Size statistics table
    axes[1, 1].axis('off')

    # Calculate statistics for valid cells only
    valid_mask = all_sizes >= NOISE_THRESHOLD
    valid = all_sizes[valid_mask]

    stats_text = f"""
    Size Statistics (excluding noise < {NOISE_THRESHOLD} px)

    Total components: {len(all_sizes)}
    Valid cells: {len(valid)} ({100*len(valid)/len(all_sizes):.1f}%)
    Noise removed: {len(all_sizes) - len(valid)}

    Valid cell sizes:
      Min:    {valid.min():,.0f} pixels
      Max:    {valid.max():,.0f} pixels
      Mean:   {valid.mean():,.0f} pixels
      Median: {np.median(valid):,.0f} pixels
      Std:    {valid.std():,.0f} pixels

    Estimated cell diameters:
      Min:    {2*np.sqrt(valid.min()/np.pi):.0f} pixels
      Max:    {2*np.sqrt(valid.max()/np.pi):.0f} pixels
      Mean:   {2*np.sqrt(valid.mean()/np.pi):.0f} pixels

    Classification thresholds:
      Noise:    < {NOISE_THRESHOLD} px
      Platelet: {NOISE_THRESHOLD} - {PLATELET_MAX} px
      RBC:      {RBC_MIN} - {RBC_MAX} px
      WBC:      > {WBC_MIN} px
      Merged:   > {MERGED_THRESHOLD} px
    """

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_sample_cells(analysis_result, output_path, num_samples=12):
    """
    Show cropped examples of individual cells.
    """
    img = analysis_result['image']
    labeled = analysis_result['labeled']
    sizes = analysis_result['sizes']
    classifications = analysis_result['classifications']

    # Get bounding boxes for each component
    objects = find_objects(labeled)

    # Filter to valid cells and sort by size
    valid_indices = [i for i, cls in enumerate(classifications) if cls not in ['noise']]
    valid_indices = sorted(valid_indices, key=lambda i: sizes[i], reverse=True)[:num_samples]

    # Create figure
    cols = 4
    rows = (len(valid_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes

    for ax_idx, cell_idx in enumerate(valid_indices):
        if ax_idx >= len(axes):
            break

        slices = objects[cell_idx]
        if slices is None:
            continue

        # Add padding
        pad = 20
        r_start = max(0, slices[0].start - pad)
        r_end = min(img.shape[0], slices[0].stop + pad)
        c_start = max(0, slices[1].start - pad)
        c_end = min(img.shape[1], slices[1].stop + pad)

        cell_img = img[r_start:r_end, c_start:c_end]

        axes[ax_idx].imshow(cell_img)
        axes[ax_idx].set_title(f"{classifications[cell_idx].upper()}\n{sizes[cell_idx]:,} px", fontsize=9)
        axes[ax_idx].axis('off')

    # Hide empty axes
    for ax_idx in range(len(valid_indices), len(axes)):
        axes[ax_idx].axis('off')

    plt.suptitle('Sample Cells (sorted by size, largest first)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("BCCD Cell Analysis")
    print("=" * 60)

    # Get sample image
    img_paths = list(TRAIN_IMAGES.glob("*.png"))
    if not img_paths:
        print("ERROR: No images found")
        return

    img_path = img_paths[0]
    mask_path = TRAIN_MASKS / img_path.name

    print(f"\nAnalyzing: {img_path.name}")

    # Analyze single image
    result = analyze_single_image(img_path, mask_path)

    print(f"\nSingle Image Results:")
    print(f"  Total components: {result['num_cells']}")
    print(f"  By type: {result['type_counts']}")

    # Analyze dataset
    print(f"\nAnalyzing dataset (100 samples)...")
    mask_paths = list(TRAIN_MASKS.glob("*.png"))
    all_sizes, all_classifications = analyze_dataset_sizes(mask_paths, num_samples=100)

    print(f"  Total components: {len(all_sizes)}")
    print(f"  Size range: {all_sizes.min()} - {all_sizes.max()}")

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visualize_cells_by_type(result, OUTPUT_DIR / "bccd_cells_by_type.png")
    visualize_size_distribution(all_sizes, all_classifications, OUTPUT_DIR / "bccd_size_distribution.png")
    visualize_sample_cells(result, OUTPUT_DIR / "bccd_sample_cells.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
