"""
Test script for data/tiling.py module.

Tests:
1. Constants - verify expected values
2. pad_to_multiple - pad image to multiple of 14
3. unpad - restore original dimensions
4. calculate_tile_positions - verify grid coverage
5. extract_tiles - verify tile extraction
6. random_crop - verify crop size and position
7. create_gaussian_weight_map - verify shape and center weighting
8. stitch_tiles - verify reconstruction
9. resize_mask_to_patches - verify downsampling
10. extract_tile_masks - verify mask tile extraction

Usage:
    python Codes/unitTest/02_data_pipeline/01_test_tiling.py
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

from data.tiling import (
    PATCH_SIZE,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_GAUSSIAN_SIGMA,
    pad_to_multiple,
    unpad,
    calculate_tile_positions,
    extract_tiles,
    random_crop,
    create_gaussian_weight_map,
    stitch_tiles,
    resize_mask_to_patches,
    extract_tile_masks
)

# Test data paths
DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TEST_IMAGES_DIR = DATA_ROOT / "test" / "original"

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


def get_sample_images() -> dict:
    """Load sample BCCD images for testing (one jpg, one png if available)."""
    png_images = list(TEST_IMAGES_DIR.glob("*.png"))
    jpg_images = list(TEST_IMAGES_DIR.glob("*.jpg"))

    samples = {}
    if png_images:
        img = Image.open(png_images[0]).convert('RGB')
        samples["png"] = img
        print(f"  Loaded PNG: {png_images[0].name}, size: {img.size}")
    if jpg_images:
        img = Image.open(jpg_images[0]).convert('RGB')
        samples["jpg"] = img
        print(f"  Loaded JPG: {jpg_images[0].name}, size: {img.size}")

    if not samples:
        raise FileNotFoundError(f"No images found in {TEST_IMAGES_DIR}")

    print(f"  Available formats: {list(samples.keys())}")

    return samples


def get_sample_image() -> Image.Image:
    """Load a sample BCCD image for testing (for backward compatibility)."""
    samples = get_sample_images()
    return list(samples.values())[0]


def test_constants():
    """Test 1: Verify constants have expected values."""
    print("\n[Test 1] Constants...")

    assert PATCH_SIZE == 14, f"PATCH_SIZE should be 14, got {PATCH_SIZE}"
    assert DEFAULT_TILE_SIZE == 224, f"DEFAULT_TILE_SIZE should be 224, got {DEFAULT_TILE_SIZE}"
    assert DEFAULT_OVERLAP == 0.75, f"DEFAULT_OVERLAP should be 0.75, got {DEFAULT_OVERLAP}"
    assert DEFAULT_GAUSSIAN_SIGMA == 0.7, f"DEFAULT_GAUSSIAN_SIGMA should be 0.7, got {DEFAULT_GAUSSIAN_SIGMA}"

    stride = int(DEFAULT_TILE_SIZE * (1 - DEFAULT_OVERLAP))
    assert stride == 56, f"Stride should be 56, got {stride}"

    print(f"  PATCH_SIZE = {PATCH_SIZE}")
    print(f"  DEFAULT_TILE_SIZE = {DEFAULT_TILE_SIZE}")
    print(f"  DEFAULT_OVERLAP = {DEFAULT_OVERLAP} (stride={stride})")
    print(f"  DEFAULT_GAUSSIAN_SIGMA = {DEFAULT_GAUSSIAN_SIGMA}")
    print("  PASS")
    return True


def test_pad_to_multiple():
    """Test 2: Pad image to multiple of 14."""
    print("\n[Test 2] pad_to_multiple...")

    # Test with non-divisible size
    img = Image.new('RGB', (100, 100), color='red')
    padded, info = pad_to_multiple(img, multiple=14)

    # Check padded size is divisible by 14
    pw, ph = padded.size
    assert pw % 14 == 0, f"Padded width {pw} not divisible by 14"
    assert ph % 14 == 0, f"Padded height {ph} not divisible by 14"

    # Check info dict
    assert info["original_size"] == (100, 100)
    assert info["padded_size"] == (pw, ph)
    assert len(info["padding"]) == 4

    print(f"  Input: 100x100 -> Output: {pw}x{ph}")
    print(f"  Padding: {info['padding']}")

    # Test with BCCD-like size (1600x1200)
    img2 = Image.new('RGB', (1600, 1200), color='blue')
    padded2, info2 = pad_to_multiple(img2, multiple=14)
    pw2, ph2 = padded2.size
    assert pw2 % 14 == 0 and ph2 % 14 == 0

    print(f"  BCCD size: 1600x1200 -> {pw2}x{ph2}")
    print("  PASS")
    return True


def test_unpad():
    """Test 3: Restore original dimensions after padding."""
    print("\n[Test 3] unpad...")

    # Create image, pad, then unpad
    original = np.random.rand(100, 150, 3).astype(np.float32)
    img = Image.fromarray((original * 255).astype(np.uint8))

    padded, info = pad_to_multiple(img, multiple=14)
    padded_arr = np.array(padded).astype(np.float32) / 255.0

    # Unpad
    restored = unpad(padded_arr, info)

    assert restored.shape == original.shape, \
        f"Shape mismatch: {restored.shape} != {original.shape}"

    print(f"  Padded shape: {padded_arr.shape}")
    print(f"  Restored shape: {restored.shape}")
    print("  PASS")
    return True


def test_calculate_tile_positions():
    """Test 4: Verify tile grid covers entire image."""
    print("\n[Test 4] calculate_tile_positions...")

    # BCCD-like image
    img_size = (1600, 1200)
    positions, grid_size = calculate_tile_positions(img_size)

    n_cols, n_rows = grid_size
    print(f"  Image: {img_size[0]}x{img_size[1]}")
    print(f"  Grid: {n_cols}x{n_rows} = {len(positions)} tiles")
    print(f"  Overlap: {DEFAULT_OVERLAP*100}%, Stride: {int(DEFAULT_TILE_SIZE * (1 - DEFAULT_OVERLAP))}")

    # Check all positions are valid
    for x, y in positions:
        assert 0 <= x <= img_size[0] - DEFAULT_TILE_SIZE, f"Invalid x position: {x}"
        assert 0 <= y <= img_size[1] - DEFAULT_TILE_SIZE, f"Invalid y position: {y}"

    # Check coverage: first and last tiles should cover corners
    first_x, first_y = positions[0]
    last_x, last_y = positions[-1]

    assert first_x == 0 and first_y == 0, "First tile should be at (0, 0)"
    assert last_x + DEFAULT_TILE_SIZE >= img_size[0], "Last column doesn't reach right edge"
    assert last_y + DEFAULT_TILE_SIZE >= img_size[1], "Last row doesn't reach bottom edge"

    print(f"  First tile: ({first_x}, {first_y})")
    print(f"  Last tile: ({last_x}, {last_y})")
    print("  PASS")
    return True


def test_extract_tiles():
    """Test 5: Extract tiles from image (both jpg and png)."""
    print("\n[Test 5] extract_tiles...")

    samples = get_sample_images()

    for fmt, img in samples.items():
        print(f"  Testing {fmt.upper()} image ({img.size[0]}x{img.size[1]})...")

        positions, grid_size = calculate_tile_positions(img.size)
        tiles = extract_tiles(img, positions)

        assert len(tiles) == len(positions), \
            f"[{fmt}] Tile count mismatch: {len(tiles)} != {len(positions)}"

        # Check tile sizes
        for i, tile in enumerate(tiles[:5]):  # Check first 5
            assert tile.size == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
                f"[{fmt}] Tile {i} has wrong size: {tile.size}"

        print(f"    Extracted {len(tiles)} tiles, size: {tiles[0].size}")

    print("  PASS")
    return True


def test_random_crop():
    """Test 6: Random crop produces correct size (both jpg and png)."""
    print("\n[Test 6] random_crop...")

    samples = get_sample_images()

    for fmt, img in samples.items():
        print(f"  Testing {fmt.upper()} image...")

        # Test multiple crops
        positions_seen = set()
        for _ in range(10):
            crop, pos = random_crop(img)
            assert crop.size == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
                f"[{fmt}] Wrong crop size: {crop.size}"
            positions_seen.add(pos)

        # Should have some variety in positions
        assert len(positions_seen) > 1, f"[{fmt}] Random crop always returns same position"

        print(f"    Crop size: {DEFAULT_TILE_SIZE}x{DEFAULT_TILE_SIZE}, unique positions: {len(positions_seen)}")

    print("  PASS")
    return True


def test_gaussian_weight_map():
    """Test 7: Gaussian weight map has correct properties."""
    print("\n[Test 7] create_gaussian_weight_map...")

    weight_map = create_gaussian_weight_map()

    assert weight_map.shape == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
        f"Wrong shape: {weight_map.shape}"

    # Center should have highest weight
    center = DEFAULT_TILE_SIZE // 2
    center_weight = weight_map[center, center]
    corner_weight = weight_map[0, 0]

    assert center_weight > corner_weight, \
        f"Center ({center_weight}) should be > corner ({corner_weight})"

    # Check value range
    assert weight_map.min() >= 0, "Weights should be non-negative"
    assert weight_map.max() <= 1.0 + 1e-6, "Weights should be <= 1"

    print(f"  Shape: {weight_map.shape}")
    print(f"  Center weight: {center_weight:.4f}")
    print(f"  Corner weight: {corner_weight:.4f}")
    print(f"  Range: [{weight_map.min():.4f}, {weight_map.max():.4f}]")
    print("  PASS")
    return True


def test_stitch_tiles():
    """Test 8: Stitch tiles back together."""
    print("\n[Test 8] stitch_tiles...")

    # Create a simple test image with gradient
    w, h = 448, 336  # Smaller for testing
    img_arr = np.zeros((h, w, 3), dtype=np.float32)
    img_arr[:, :, 0] = np.linspace(0, 1, w)  # Red gradient horizontal
    img_arr[:, :, 1] = np.linspace(0, 1, h)[:, np.newaxis]  # Green gradient vertical

    img = Image.fromarray((img_arr * 255).astype(np.uint8))

    # Extract tiles
    positions, grid_size = calculate_tile_positions(img.size)
    tiles = extract_tiles(img, positions)

    # Convert to numpy arrays
    tile_arrays = [np.array(t).astype(np.float32) / 255.0 for t in tiles]

    # Stitch back
    stitched = stitch_tiles(tile_arrays, positions, img.size)

    assert stitched.shape == (h, w, 3), \
        f"Wrong shape: {stitched.shape} != {(h, w, 3)}"

    # Check reconstruction quality (should be close to original)
    mse = np.mean((stitched - img_arr) ** 2)
    print(f"  Original: {w}x{h}")
    print(f"  Tiles: {len(tiles)}")
    print(f"  Stitched: {stitched.shape}")
    print(f"  MSE: {mse:.6f}")

    assert mse < 0.01, f"MSE too high: {mse}"
    print("  PASS")
    return True


def test_resize_mask_to_patches():
    """Test 9: Downsample mask to patch resolution."""
    print("\n[Test 9] resize_mask_to_patches...")

    # Test 1: Shape check
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[50:100, 50:100] = 1.0  # Square "cell"

    patch_mask = resize_mask_to_patches(mask)

    expected_size = 224 // PATCH_SIZE  # 16
    assert patch_mask.shape == (expected_size, expected_size), \
        f"Wrong shape: {patch_mask.shape}"

    print(f"  Shape: {mask.shape} -> {patch_mask.shape}")

    # Test 2: Verify specific patch locations
    # Put a single pixel at (14, 14) - should affect patch (1, 1)
    mask2 = np.zeros((224, 224), dtype=np.float32)
    mask2[14, 14] = 1.0  # Single pixel in patch (1, 1)

    patch_mask2 = resize_mask_to_patches(mask2)

    assert patch_mask2[1, 1] == 1.0, "Single pixel not detected in correct patch"
    assert patch_mask2[0, 0] == 0.0, "Wrong patch marked (0,0 should be 0)"
    assert patch_mask2.sum() == 1.0, f"Should have exactly 1 patch marked, got {patch_mask2.sum()}"

    print(f"  Single pixel test: pixel (14,14) -> patch (1,1)")

    # Test 3: Verify patch boundaries
    # Fill exactly one patch: rows [28:42], cols [42:56] = patch (2, 3)
    mask3 = np.zeros((224, 224), dtype=np.float32)
    mask3[28:42, 42:56] = 1.0  # Exactly patch (2, 3)

    patch_mask3 = resize_mask_to_patches(mask3)

    assert patch_mask3[2, 3] == 1.0, "Full patch not detected"
    assert patch_mask3.sum() == 1.0, f"Should mark exactly 1 patch, got {patch_mask3.sum()}"

    print(f"  Full patch test: pixels [28:42, 42:56] -> patch (2,3)")

    # Test 4: Empty mask should give all zeros
    mask4 = np.zeros((224, 224), dtype=np.float32)
    patch_mask4 = resize_mask_to_patches(mask4)

    assert patch_mask4.sum() == 0, "Empty mask should give all zeros"

    print(f"  Empty mask test: sum = 0")

    print("  PASS")
    return True


def test_extract_tile_masks():
    """Test 10: Extract mask tiles at positions."""
    print("\n[Test 10] extract_tile_masks...")

    # Create full-size mask
    mask = np.random.rand(1200, 1600).astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)

    # Get positions
    positions, _ = calculate_tile_positions((1600, 1200))

    # Extract mask tiles
    mask_tiles = extract_tile_masks(mask, positions)

    assert len(mask_tiles) == len(positions), \
        f"Count mismatch: {len(mask_tiles)} != {len(positions)}"

    # Check tile sizes
    for i, mt in enumerate(mask_tiles[:5]):
        assert mt.shape == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), \
            f"Mask tile {i} wrong shape: {mt.shape}"

    print(f"  Extracted {len(mask_tiles)} mask tiles")
    print(f"  Tile shape: {mask_tiles[0].shape}")
    print("  PASS")
    return True


def visualize_results():
    """Generate visualization of tiling operations for both jpg and png."""
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_sample_images()

    for fmt, img in samples.items():
        print(f"  Generating visualization for {fmt.upper()}...")

        positions, grid_size = calculate_tile_positions(img.size)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Original with tile grid
        ax = axes[0, 0]
        ax.imshow(img)
        n_cols, n_rows = grid_size
        # Show corner tiles + sampled tiles from actual positions
        # Corners: first, last of first row, first of last row, last
        corner_indices = [
            0,                          # top-left
            n_cols - 1,                 # top-right
            (n_rows - 1) * n_cols,      # bottom-left
            len(positions) - 1          # bottom-right
        ]
        # Add some middle samples
        middle_indices = [
            len(positions) // 4,
            len(positions) // 2,
            3 * len(positions) // 4
        ]
        sample_indices = list(set(corner_indices + middle_indices))

        for idx in sample_indices:
            if 0 <= idx < len(positions):
                x, y = positions[idx]
                rect = plt.Rectangle((x, y), DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE,
                                      fill=False, edgecolor='red', linewidth=1.5, alpha=0.8)
                ax.add_patch(rect)
        ax.set_title(f"Original with sample tiles\n{img.size[0]}x{img.size[1]}, grid {n_cols}x{n_rows}={len(positions)} tiles")
        ax.axis('off')

        # 2. Sample tiles
        ax = axes[0, 1]
        tiles = extract_tiles(img, positions)
        sample_idx = [0, len(tiles)//2, len(tiles)-1]
        tile_strip = np.hstack([np.array(tiles[i]) for i in sample_idx])
        ax.imshow(tile_strip)
        ax.set_title(f"Sample tiles (indices {sample_idx})")
        ax.axis('off')

        # 3. Gaussian weight map
        ax = axes[0, 2]
        weight_map = create_gaussian_weight_map()
        im = ax.imshow(weight_map, cmap='hot')
        ax.set_title(f"Gaussian weight map\nsigma={DEFAULT_GAUSSIAN_SIGMA}")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 4. Padding demonstration
        ax = axes[1, 0]
        small_img = Image.new('RGB', (100, 80), color='lightblue')
        padded, info = pad_to_multiple(small_img, multiple=14)
        padded_arr = np.array(padded)
        ax.imshow(padded_arr)
        pad = info["padding"]
        ax.axhline(pad[0], color='red', linestyle='--', label='padding')
        ax.axhline(padded_arr.shape[0] - pad[1], color='red', linestyle='--')
        ax.axvline(pad[2], color='red', linestyle='--')
        ax.axvline(padded_arr.shape[1] - pad[3], color='red', linestyle='--')
        ax.set_title(f"Padding: {info['original_size']} -> {info['padded_size']}")
        ax.legend(loc='lower right')

        # 5. Stitch reconstruction test
        ax = axes[1, 1]
        # Use smaller image for faster test
        small = img.resize((448, 336))
        pos_small, _ = calculate_tile_positions(small.size)
        tiles_small = extract_tiles(small, pos_small)
        tile_arrays = [np.array(t).astype(np.float32) / 255.0 for t in tiles_small]
        stitched = stitch_tiles(tile_arrays, pos_small, small.size)
        ax.imshow(stitched)
        ax.set_title(f"Stitched reconstruction\n{small.size[0]}x{small.size[1]}")
        ax.axis('off')

        # 6. Mask to patches demonstration - side by side comparison
        ax = axes[1, 2]

        # Create synthetic mask with two cell regions
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[28:84, 28:98] = 1.0    # Top-left cell (not patch-aligned)
        mask[112:182, 98:182] = 1.0  # Bottom-right cell (not patch-aligned)

        patch_mask = resize_mask_to_patches(mask)  # 16x16

        # Upscale patch mask back to 224x224 for visual comparison
        patch_upscaled = np.repeat(np.repeat(patch_mask, 14, axis=0), 14, axis=1)

        # Create side-by-side: pixel mask (left 224) | patch mask (right 224)
        side_by_side = np.zeros((224, 448), dtype=np.float32)
        side_by_side[:, :224] = mask
        side_by_side[:, 224:] = patch_upscaled

        ax.imshow(side_by_side, cmap='gray')
        ax.axvline(224, color='red', linewidth=2)  # Separator
        ax.set_title(f"Pixel mask (left) vs Patch mask (right)\n224x224 -> 16x16 -> 224x224")
        ax.set_xticks([112, 336])
        ax.set_xticklabels(['Pixel', 'Patch'])
        ax.set_yticks([])

        plt.suptitle(f"Tiling Module Test Visualization ({fmt.upper()})", fontsize=14)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"tiling_test_visualization_{fmt}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing data/tiling.py")
    print("=" * 60)

    results = []

    tests = [
        ("Constants", test_constants),
        ("pad_to_multiple", test_pad_to_multiple),
        ("unpad", test_unpad),
        ("calculate_tile_positions", test_calculate_tile_positions),
        ("extract_tiles", test_extract_tiles),
        ("random_crop", test_random_crop),
        ("gaussian_weight_map", test_gaussian_weight_map),
        ("stitch_tiles", test_stitch_tiles),
        ("resize_mask_to_patches", test_resize_mask_to_patches),
        ("extract_tile_masks", test_extract_tile_masks),
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

    # Generate visualization for both formats
    try:
        print("\n[Visualization] Generating for jpg and png...")
        visualize_results()
        results.append(("Visualization (jpg+png)", "PASS"))
    except Exception as e:
        results.append(("Visualization (jpg+png)", f"ERROR: {e}"))
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
