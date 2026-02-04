"""
Test data pipeline for BCCD cell segmentation.

Tests both Padding and Tiling modes to ensure:
1. Data loading works correctly
2. Transforms apply properly
3. Collate functions work with DataLoader
4. Visualize sample outputs
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
SPLIT_PATH = DATA_ROOT / "splits.json"

SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import (
    create_datasets,
    create_test_dataset,
    create_train_val_split,
    get_collate_fn,
    IMAGENET_MEAN,
    IMAGENET_STD
)


# =============================================================================
# Utilities
# =============================================================================

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize tensor from ImageNet normalization to [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


# =============================================================================
# Test Functions
# =============================================================================

def test_split_creation():
    """Test train/val split creation."""
    print("\n" + "=" * 60)
    print("Testing Split Creation")
    print("=" * 60)

    train_dir = DATA_ROOT / "train" / "original"

    if not train_dir.exists():
        print(f"ERROR: Training directory not found: {train_dir}")
        return False

    # Create split
    train_files, val_files = create_train_val_split(
        image_dir=train_dir,
        output_path=SPLIT_PATH,
        val_ratio=0.2,
        seed=42
    )

    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Split saved to: {SPLIT_PATH}")

    return True


def test_padding_mode():
    """Test padding mode dataset."""
    print("\n" + "=" * 60)
    print("Testing Padding Mode")
    print("=" * 60)

    # Create datasets
    train_ds, val_ds = create_datasets(
        data_root=DATA_ROOT,
        split_path=SPLIT_PATH,
        mode="padding"
    )

    print(f"  Train dataset: {len(train_ds)} samples")
    print(f"  Val dataset:   {len(val_ds)} samples")

    # Test single item
    sample = train_ds[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape:  {sample['mask'].shape}")
    print(f"  Padding info: {sample['padding_info']}")

    # Test with DataLoader
    collate_fn = get_collate_fn(mode="padding", train=True)
    loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\n  Batch keys: {list(batch.keys())}")
    print(f"  Batch images: {len(batch['images'])} tensors")

    # Visualize
    visualize_padding_samples(train_ds, OUTPUT_DIR / "padding_samples.png")

    return True


def test_tiling_train_mode():
    """Test tiling mode for training (random crop)."""
    print("\n" + "=" * 60)
    print("Testing Tiling Mode (Training)")
    print("=" * 60)

    # Create datasets
    train_ds, val_ds = create_datasets(
        data_root=DATA_ROOT,
        split_path=SPLIT_PATH,
        mode="tiling"
    )

    print(f"  Train dataset: {len(train_ds)} samples")
    print(f"  Val dataset:   {len(val_ds)} samples")

    # Test single item
    sample = train_ds[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape:  {sample['mask'].shape}")
    print(f"  Position:    {sample['position']}")

    # Test with DataLoader
    collate_fn = get_collate_fn(mode="tiling", train=True)
    loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\n  Batch keys: {list(batch.keys())}")
    print(f"  Batch images shape: {batch['images'].shape}")
    print(f"  Batch masks shape:  {batch['masks'].shape}")

    # Visualize
    visualize_tiling_train_samples(train_ds, OUTPUT_DIR / "tiling_train_samples.png")

    return True


def test_tiling_inference_mode():
    """Test tiling mode for inference (full coverage)."""
    print("\n" + "=" * 60)
    print("Testing Tiling Mode (Inference)")
    print("=" * 60)

    # Create test dataset
    test_ds = create_test_dataset(
        data_root=DATA_ROOT,
        mode="tiling"
    )

    print(f"  Test dataset: {len(test_ds)} samples")

    # Test single item
    sample = test_ds[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    print(f"  Tiles shape: {sample['tiles'].shape}")
    print(f"  Num positions: {len(sample['positions'])}")
    print(f"  Grid size: {sample['grid_size']}")
    print(f"  Original size: {sample['original_size']}")

    # Test with DataLoader
    collate_fn = get_collate_fn(mode="tiling", train=False)
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\n  Batch keys: {list(batch.keys())}")
    print(f"  Batch tiles: {len(batch['tiles'])} images, first has {batch['tiles'][0].shape[0]} tiles")

    # Visualize
    visualize_tiling_inference_samples(test_ds, OUTPUT_DIR / "tiling_inference_samples.png")

    return True


# =============================================================================
# Visualization
# =============================================================================

def visualize_padding_samples(dataset, output_path: Path, num_samples: int = 2):
    """Visualize samples from padding mode dataset."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]
        img = denormalize(sample['image'])
        mask = sample['mask'].numpy()

        # Original (denormalized)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image (padded)\n{sample['padding_info']['padded_size']}")
        axes[i, 0].axis('off')

        # Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Mask\n{mask.shape}")
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(mask, alpha=0.3, cmap='Reds')
        axes[i, 2].set_title(f"Overlay\n{sample['filename']}")
        axes[i, 2].axis('off')

    plt.suptitle("Padding Mode Samples", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def visualize_tiling_train_samples(dataset, output_path: Path, num_samples: int = 4):
    """Visualize samples from tiling training mode."""
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    for i in range(num_samples):
        sample = dataset[i]
        img = denormalize(sample['image'])
        mask = sample['mask'].numpy()

        # Image crop
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Crop {sample['position']}")
        axes[0, i].axis('off')

        # Mask crop
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask")
        axes[1, i].axis('off')

    plt.suptitle("Tiling Mode - Training Crops (224x224)", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def visualize_tiling_inference_samples(dataset, output_path: Path):
    """Visualize tiling for inference."""
    sample = dataset[0]
    tiles = sample['tiles']
    positions = sample['positions']
    grid_size = sample['grid_size']

    # Show subset of tiles
    n_show = min(16, tiles.shape[0])
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_show):
        img = denormalize(tiles[i])
        axes[i].imshow(img)
        axes[i].set_title(f"Tile {i}: {positions[i]}")
        axes[i].axis('off')

    # Hide empty axes
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Tiling Mode - Inference Tiles\n"
                 f"Grid: {grid_size[0]}x{grid_size[1]} = {len(positions)} tiles, "
                 f"Original: {sample['original_size']}", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Data Pipeline Test")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Data root:   {DATA_ROOT}")
    print(f"  Split path:  {SPLIT_PATH}")
    print(f"  Output:      {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tests
    results = {}

    # 1. Create split if needed
    if not SPLIT_PATH.exists():
        results["split_creation"] = test_split_creation()
    else:
        print(f"\nSplit already exists: {SPLIT_PATH}")
        results["split_creation"] = True

    # 2. Test padding mode
    results["padding_mode"] = test_padding_mode()

    # 3. Test tiling training mode
    results["tiling_train"] = test_tiling_train_mode()

    # 4. Test tiling inference mode
    results["tiling_inference"] = test_tiling_inference_mode()

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
