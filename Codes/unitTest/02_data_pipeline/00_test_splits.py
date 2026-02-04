"""
Test script for data/splits.py module.

Tests:
1. Create split - verify train + val counts match expected ratio
2. Reproducibility - run twice with same seed, verify identical results
3. Load split - load from JSON, verify matches created split
4. No overlap - verify no files appear in both train and val
5. Coverage - verify train + val = all files in directory

Usage:
    python Codes/unitTest/04_data_pipeline/00_test_splits.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

from data.splits import create_train_val_split, load_split

# Paths
DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
IMAGE_DIR = DATA_ROOT / "train" / "original"
SPLIT_PATH = SCRIPT_DIR / "splits.json"


def get_all_images() -> list:
    """Get all image files in the training directory."""
    return sorted(
        [f.name for f in IMAGE_DIR.glob("*.png")] +
        [f.name for f in IMAGE_DIR.glob("*.jpg")]
    )


def test_create_split():
    """Test 1: Create split and verify counts match expected ratio."""
    print("\n[Test 1] Create split...")

    train_files, val_files = create_train_val_split(
        image_dir=IMAGE_DIR,
        output_path=SPLIT_PATH,
        val_ratio=0.2,
        seed=2026
    )

    total = len(train_files) + len(val_files)
    expected_val = int(total * 0.2)

    # Check counts
    assert len(val_files) == expected_val, \
        f"Val count mismatch: {len(val_files)} != {expected_val}"
    assert len(train_files) == total - expected_val, \
        f"Train count mismatch: {len(train_files)} != {total - expected_val}"

    print(f"  Total: {total}, Train: {len(train_files)}, Val: {len(val_files)}")
    print("  PASS: Counts match expected 80/20 ratio")
    return train_files, val_files


def test_reproducibility(original_train, original_val):
    """Test 2: Run twice with same seed, verify identical results."""
    print("\n[Test 2] Reproducibility...")

    # Create again with same seed
    train_files2, val_files2 = create_train_val_split(
        image_dir=IMAGE_DIR,
        output_path=SPLIT_PATH,
        val_ratio=0.2,
        seed=2026
    )

    assert train_files2 == original_train, "Train files differ on second run"
    assert val_files2 == original_val, "Val files differ on second run"

    print("  PASS: Identical results with same seed")


def test_load_split(original_train, original_val):
    """Test 3: Load from JSON and verify matches created split."""
    print("\n[Test 3] Load split...")

    loaded_train, loaded_val = load_split(SPLIT_PATH)

    assert loaded_train == original_train, "Loaded train != created train"
    assert loaded_val == original_val, "Loaded val != created val"

    print(f"  Loaded: {len(loaded_train)} train, {len(loaded_val)} val")
    print("  PASS: Loaded split matches created split")


def test_no_overlap(train_files, val_files):
    """Test 4: Verify no files appear in both train and val."""
    print("\n[Test 4] No overlap...")

    train_set = set(train_files)
    val_set = set(val_files)
    overlap = train_set & val_set

    assert len(overlap) == 0, f"Overlap found: {overlap}"

    print("  PASS: No overlap between train and val")


def test_coverage(train_files, val_files):
    """Test 5: Verify train + val = all files in directory."""
    print("\n[Test 5] Coverage...")

    all_images = set(get_all_images())
    split_files = set(train_files) | set(val_files)

    missing = all_images - split_files
    extra = split_files - all_images

    assert len(missing) == 0, f"Missing files: {missing}"
    assert len(extra) == 0, f"Extra files: {extra}"

    print(f"  All {len(all_images)} images accounted for")
    print("  PASS: Complete coverage")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing data/splits.py")
    print("=" * 60)

    results = []

    try:
        # Test 1: Create split
        train_files, val_files = test_create_split()
        results.append(("Create split", "PASS"))

        # Test 2: Reproducibility
        test_reproducibility(train_files, val_files)
        results.append(("Reproducibility", "PASS"))

        # Test 3: Load split
        test_load_split(train_files, val_files)
        results.append(("Load split", "PASS"))

        # Test 4: No overlap
        test_no_overlap(train_files, val_files)
        results.append(("No overlap", "PASS"))

        # Test 5: Coverage
        test_coverage(train_files, val_files)
        results.append(("Coverage", "PASS"))

    except AssertionError as e:
        results.append(("FAILED", str(e)))
        print(f"\n  FAIL: {e}")
    except Exception as e:
        results.append(("ERROR", str(e)))
        print(f"\n  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for test_name, status in results:
        print(f"  {test_name}: {status}")

    print(f"\n  {passed}/{total} tests passed")
    print(f"  Split saved to: {SPLIT_PATH}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
