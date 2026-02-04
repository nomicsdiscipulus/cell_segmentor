"""
Test script for data/dataset.py module.

Tests:
1. BCCDDataset initialization
2. File validation (images and masks)
3. Padding mode - single item
4. Tiling training mode - single item
5. Tiling inference mode - single item
6. create_datasets factory
7. create_test_dataset factory
8. Collate functions with DataLoader
9. Batch iteration

Usage:
    python Codes/unitTest/02_data_pipeline/03_test_dataset.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    BCCDDataset,
    create_datasets,
    create_test_dataset,
    get_collate_fn,
    padding_collate_fn,
    tiling_train_collate_fn,
    tiling_inference_collate_fn
)
from data.splits import create_train_val_split, load_split
from data.tiling import DEFAULT_TILE_SIZE, DEFAULT_OVERLAP
from data.transforms import get_denormalize_transform

# Paths
DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
SPLIT_PATH = DATA_ROOT / "splits.json"
TRAIN_IMAGE_DIR = DATA_ROOT / "train" / "original"
TRAIN_MASK_DIR = DATA_ROOT / "train" / "mask"
TEST_IMAGE_DIR = DATA_ROOT / "test" / "original"
TEST_MASK_DIR = DATA_ROOT / "test" / "mask"

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


def ensure_split_exists():
    """Ensure train/val split exists."""
    if not SPLIT_PATH.exists():
        print("  Creating train/val split...")
        create_train_val_split(
            image_dir=TRAIN_IMAGE_DIR,
            output_path=SPLIT_PATH,
            val_ratio=0.2,
            seed=2026
        )
    else:
        print(f"  Split exists: {SPLIT_PATH}")


def test_dataset_init():
    """Test 1: BCCDDataset initialization."""
    print("\n[Test 1] BCCDDataset initialization...")

    # Test with test set (no split needed)
    dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="padding",
        train=False
    )

    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.mode == "padding"
    assert dataset.train == False

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Mode: {dataset.mode}")
    print(f"  Train: {dataset.train}")
    print("  PASS")
    return True


def test_file_validation():
    """Test 2: File validation catches missing files."""
    print("\n[Test 2] File validation...")

    # Test with valid directory
    dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="padding"
    )
    print(f"  Valid dataset created with {len(dataset)} files")

    # Test with invalid file list
    try:
        invalid_dataset = BCCDDataset(
            image_dir=TEST_IMAGE_DIR,
            mask_dir=TEST_MASK_DIR,
            file_list=["nonexistent_file.png"],
            mode="padding"
        )
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  Invalid file list correctly raises error: {str(e)[:50]}...")

    print("  PASS")
    return True


def test_padding_mode():
    """Test 3: Padding mode returns correct structure."""
    print("\n[Test 3] Padding mode...")

    dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="padding",
        train=False
    )

    item = dataset[0]

    # Check keys
    expected_keys = {"image", "mask", "padding_info", "filename", "original_size"}
    assert set(item.keys()) == expected_keys, f"Wrong keys: {item.keys()}"

    # Check types
    assert isinstance(item["image"], torch.Tensor), "Image should be tensor"
    assert isinstance(item["mask"], torch.Tensor), "Mask should be tensor"
    assert isinstance(item["padding_info"], dict), "padding_info should be dict"
    assert isinstance(item["filename"], str), "filename should be str"

    # Check shapes
    assert item["image"].dim() == 3, "Image should be 3D (C, H, W)"
    assert item["mask"].dim() == 2, "Mask should be 2D (H, W)"

    # Check padding makes dimensions divisible by 14
    _, h, w = item["image"].shape
    assert h % 14 == 0, f"Height {h} not divisible by 14"
    assert w % 14 == 0, f"Width {w} not divisible by 14"

    # Check image and mask have same spatial dims
    assert item["image"].shape[1:] == item["mask"].shape, \
        f"Image {item['image'].shape[1:]} != Mask {item['mask'].shape}"

    print(f"  Image shape: {item['image'].shape}")
    print(f"  Mask shape: {item['mask'].shape}")
    print(f"  Original size: {item['original_size']}")
    print(f"  Padding: {item['padding_info']['padding']}")
    print("  PASS")
    return True


def test_tiling_train_mode():
    """Test 4: Tiling training mode returns correct structure."""
    print("\n[Test 4] Tiling training mode...")

    dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="tiling",
        train=True
    )

    item = dataset[0]

    # Check keys
    expected_keys = {"image", "mask", "position", "filename", "original_size"}
    assert set(item.keys()) == expected_keys, f"Wrong keys: {item.keys()}"

    # Check types
    assert isinstance(item["image"], torch.Tensor), "Image should be tensor"
    assert isinstance(item["mask"], torch.Tensor), "Mask should be tensor"
    assert isinstance(item["position"], tuple), "Position should be tuple"

    # Check shapes - should be tile size
    assert item["image"].shape == (3, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
        f"Wrong image shape: {item['image'].shape}"
    assert item["mask"].shape == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
        f"Wrong mask shape: {item['mask'].shape}"

    # Check position is valid
    x, y = item["position"]
    orig_w, orig_h = item["original_size"]
    assert 0 <= x <= orig_w - DEFAULT_TILE_SIZE, f"Invalid x: {x}"
    assert 0 <= y <= orig_h - DEFAULT_TILE_SIZE, f"Invalid y: {y}"

    print(f"  Image shape: {item['image'].shape}")
    print(f"  Mask shape: {item['mask'].shape}")
    print(f"  Position: {item['position']}")
    print(f"  Original size: {item['original_size']}")
    print("  PASS")
    return True


def test_tiling_inference_mode():
    """Test 5: Tiling inference mode returns correct structure."""
    print("\n[Test 5] Tiling inference mode...")

    dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="tiling",
        train=False
    )

    item = dataset[0]

    # Check keys
    expected_keys = {"tiles", "positions", "grid_size", "mask", "filename", "original_size"}
    assert set(item.keys()) == expected_keys, f"Wrong keys: {item.keys()}"

    # Check types
    assert isinstance(item["tiles"], torch.Tensor), "Tiles should be tensor"
    assert isinstance(item["positions"], list), "Positions should be list"
    assert isinstance(item["grid_size"], tuple), "Grid size should be tuple"
    assert isinstance(item["mask"], torch.Tensor), "Mask should be tensor"

    # Check tiles shape
    n_tiles = len(item["positions"])
    assert item["tiles"].shape == (n_tiles, 3, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
        f"Wrong tiles shape: {item['tiles'].shape}"

    # Check grid size matches positions count
    n_cols, n_rows = item["grid_size"]
    assert n_cols * n_rows == n_tiles, \
        f"Grid {n_cols}x{n_rows} != {n_tiles} positions"

    # Check mask is original size (not padded)
    orig_w, orig_h = item["original_size"]
    assert item["mask"].shape == (orig_h, orig_w), \
        f"Mask shape {item['mask'].shape} != original {(orig_h, orig_w)}"

    print(f"  Tiles shape: {item['tiles'].shape}")
    print(f"  Grid size: {item['grid_size']}")
    print(f"  Num positions: {len(item['positions'])}")
    print(f"  Mask shape: {item['mask'].shape}")
    print(f"  Original size: {item['original_size']}")
    print("  PASS")
    return True


def test_create_datasets():
    """Test 6: create_datasets factory function."""
    print("\n[Test 6] create_datasets factory...")

    ensure_split_exists()

    train_dataset, val_dataset = create_datasets(
        data_root=DATA_ROOT,
        split_path=SPLIT_PATH,
        mode="tiling"
    )

    # Load split to verify counts
    train_files, val_files = load_split(SPLIT_PATH)

    assert len(train_dataset) == len(train_files), \
        f"Train size mismatch: {len(train_dataset)} != {len(train_files)}"
    assert len(val_dataset) == len(val_files), \
        f"Val size mismatch: {len(val_dataset)} != {len(val_files)}"

    # Check train has augmentation, val doesn't
    assert train_dataset.train == True, "Train dataset should have train=True"
    assert val_dataset.train == False, "Val dataset should have train=False"

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")
    print(f"  Train augmentation: {train_dataset.train}")
    print(f"  Val augmentation: {val_dataset.train}")
    print("  PASS")
    return True


def test_create_test_dataset():
    """Test 7: create_test_dataset factory function."""
    print("\n[Test 7] create_test_dataset factory...")

    test_dataset = create_test_dataset(
        data_root=DATA_ROOT,
        mode="padding"
    )

    # Count files in test directory
    test_files = list(TEST_IMAGE_DIR.glob("*.png")) + list(TEST_IMAGE_DIR.glob("*.jpg"))

    assert len(test_dataset) == len(test_files), \
        f"Test size mismatch: {len(test_dataset)} != {len(test_files)}"
    assert test_dataset.train == False, "Test dataset should have train=False"

    print(f"  Test dataset: {len(test_dataset)} samples")
    print(f"  Mode: {test_dataset.mode}")
    print("  PASS")
    return True


def test_collate_functions():
    """Test 8: Collate functions work correctly."""
    print("\n[Test 8] Collate functions...")

    # Test padding collate
    print("  Testing padding_collate_fn...")
    padding_dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="padding"
    )
    padding_loader = DataLoader(
        padding_dataset,
        batch_size=2,
        collate_fn=padding_collate_fn
    )
    padding_batch = next(iter(padding_loader))

    assert "images" in padding_batch and len(padding_batch["images"]) == 2
    assert "masks" in padding_batch and len(padding_batch["masks"]) == 2
    print(f"    Batch has {len(padding_batch['images'])} images (list, not stacked)")

    # Test tiling train collate
    print("  Testing tiling_train_collate_fn...")
    tiling_train_dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="tiling",
        train=True
    )
    tiling_train_loader = DataLoader(
        tiling_train_dataset,
        batch_size=4,
        collate_fn=tiling_train_collate_fn
    )
    tiling_train_batch = next(iter(tiling_train_loader))

    assert tiling_train_batch["images"].shape[0] == 4, "Should stack 4 images"
    assert tiling_train_batch["masks"].shape[0] == 4, "Should stack 4 masks"
    print(f"    Images shape: {tiling_train_batch['images'].shape} (stacked)")
    print(f"    Masks shape: {tiling_train_batch['masks'].shape} (stacked)")

    # Test tiling inference collate
    print("  Testing tiling_inference_collate_fn...")
    tiling_inf_dataset = BCCDDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        mode="tiling",
        train=False
    )
    tiling_inf_loader = DataLoader(
        tiling_inf_dataset,
        batch_size=2,
        collate_fn=tiling_inference_collate_fn
    )
    tiling_inf_batch = next(iter(tiling_inf_loader))

    assert len(tiling_inf_batch["tiles"]) == 2, "Should have 2 tile sets"
    assert len(tiling_inf_batch["masks"]) == 2, "Should have 2 masks"
    print(f"    Batch has {len(tiling_inf_batch['tiles'])} tile sets (list)")
    print(f"    First tile set shape: {tiling_inf_batch['tiles'][0].shape}")

    # Test get_collate_fn factory
    print("  Testing get_collate_fn...")
    assert get_collate_fn("padding", True) == padding_collate_fn
    assert get_collate_fn("tiling", True) == tiling_train_collate_fn
    assert get_collate_fn("tiling", False) == tiling_inference_collate_fn

    print("  PASS")
    return True


def test_batch_iteration():
    """Test 9: Can iterate through full dataset."""
    print("\n[Test 9] Batch iteration...")

    ensure_split_exists()

    train_dataset, val_dataset = create_datasets(
        data_root=DATA_ROOT,
        split_path=SPLIT_PATH,
        mode="tiling"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=get_collate_fn("tiling", train=True),
        shuffle=True
    )

    # Iterate through a few batches
    n_batches = 3
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        assert batch["images"].shape[0] <= 8
        assert batch["masks"].shape[0] <= 8

    print(f"  Successfully iterated through {n_batches} batches")
    print(f"  Batch image shape: {batch['images'].shape}")
    print(f"  Batch mask shape: {batch['masks'].shape}")
    print("  PASS")
    return True


def visualize_dataset_samples():
    """Generate visualization of dataset samples with original images for comparison."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    denorm = get_denormalize_transform()

    # Test both modes
    for mode in ["padding", "tiling"]:
        print(f"  Generating visualization for {mode} mode...")

        if mode == "tiling":
            # Training mode (random crops)
            dataset = BCCDDataset(
                image_dir=TEST_IMAGE_DIR,
                mask_dir=TEST_MASK_DIR,
                mode="tiling",
                train=True
            )

            # 2 rows x 4 columns: Original+rect | Orig Mask | Cropped | Cropped Mask
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            for row in range(2):
                item = dataset[row % len(dataset)]
                filename = item["filename"]

                # Load original image and mask
                orig_img = np.array(Image.open(dataset._get_image_path(filename)).convert('RGB'))
                orig_mask = dataset._load_mask(filename)

                # Denormalize cropped image
                crop_img = denorm(item["image"]).permute(1, 2, 0).numpy()
                crop_img = np.clip(crop_img, 0, 1)
                crop_mask = item["mask"].numpy()

                # Get crop position
                x, y = item["position"]
                tile_size = DEFAULT_TILE_SIZE

                # Original image with crop rectangle
                axes[row, 0].imshow(orig_img)
                rect = patches.Rectangle((x, y), tile_size, tile_size,
                                          linewidth=2, edgecolor='red', facecolor='none')
                axes[row, 0].add_patch(rect)
                axes[row, 0].set_title(f"Original {orig_img.shape[1]}x{orig_img.shape[0]}\ncrop at ({x}, {y})")
                axes[row, 0].axis('off')

                # Original mask
                axes[row, 1].imshow(orig_mask, cmap='gray')
                axes[row, 1].set_title(f"Original Mask")
                axes[row, 1].axis('off')

                # Cropped image
                axes[row, 2].imshow(crop_img)
                axes[row, 2].set_title(f"Cropped {tile_size}x{tile_size}")
                axes[row, 2].axis('off')

                # Cropped mask
                axes[row, 3].imshow(crop_mask, cmap='gray')
                axes[row, 3].set_title(f"Cropped Mask")
                axes[row, 3].axis('off')

            plt.suptitle("Tiling Training Mode: Original vs Random 224x224 Crops", fontsize=12)

        else:
            # Padding mode - 2 rows x 3 columns: Original | Padded Image | Padded Mask
            dataset = BCCDDataset(
                image_dir=TEST_IMAGE_DIR,
                mask_dir=TEST_MASK_DIR,
                mode="padding",
                train=False
            )

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for row in range(2):
                item = dataset[row]
                filename = item["filename"]

                # Load original image
                orig_img = np.array(Image.open(dataset._get_image_path(filename)).convert('RGB'))
                orig_mask = dataset._load_mask(filename)

                # Denormalize padded image
                padded_img = denorm(item["image"]).permute(1, 2, 0).numpy()
                padded_img = np.clip(padded_img, 0, 1)
                padded_mask = item["mask"].numpy()

                # Original image
                axes[row, 0].imshow(orig_img)
                axes[row, 0].set_title(f"Original\n{orig_img.shape[1]}x{orig_img.shape[0]}")
                axes[row, 0].axis('off')

                # Padded image
                axes[row, 1].imshow(padded_img)
                axes[row, 1].set_title(f"Padded Image\n{item['image'].shape[2]}x{item['image'].shape[1]}")
                axes[row, 1].axis('off')

                # Padded mask
                axes[row, 2].imshow(padded_mask, cmap='gray')
                axes[row, 2].set_title(f"Padded Mask\n{padded_mask.shape[1]}x{padded_mask.shape[0]}")
                axes[row, 2].axis('off')

            plt.suptitle("Padding Mode: Original vs Padded (to multiple of 14)", fontsize=12)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f"dataset_{mode}_samples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing data/dataset.py")
    print("=" * 60)

    results = []

    tests = [
        ("Dataset init", test_dataset_init),
        ("File validation", test_file_validation),
        ("Padding mode", test_padding_mode),
        ("Tiling train mode", test_tiling_train_mode),
        ("Tiling inference mode", test_tiling_inference_mode),
        ("create_datasets", test_create_datasets),
        ("create_test_dataset", test_create_test_dataset),
        ("Collate functions", test_collate_functions),
        ("Batch iteration", test_batch_iteration),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except AssertionError as e:
            results.append((test_name, f"FAIL: {e}"))
            print(f"  FAIL: {e}")
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
            print(f"  ERROR: {e}")

    # Generate visualization
    try:
        print("\n[Visualization] Generating dataset samples...")
        visualize_dataset_samples()
        results.append(("Visualization", "PASS"))
    except Exception as e:
        results.append(("Visualization", f"ERROR: {e}"))
        print(f"  Visualization ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for test_name, status in results:
        symbol = "+" if status == "PASS" else "-"
        print(f"  [{symbol}] {test_name}: {status}")

    print(f"\n  {passed}/{total} tests passed")

    if OUTPUT_DIR.exists():
        print(f"  Outputs: {OUTPUT_DIR}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
