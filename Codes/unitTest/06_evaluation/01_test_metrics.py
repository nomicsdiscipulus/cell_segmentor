"""
Test script for evaluation/metrics.py module.

Tests:
1. SegmentationMetrics initialization
2. Single sample update
3. Batch update
4. Metric computation (IoU, Dice, precision, recall)
5. Perfect predictions (should give 1.0)
6. Completely wrong predictions
7. Per-image metrics

Usage:
    python Codes/unitTest/06_evaluation/01_test_metrics.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import torch
import numpy as np

from evaluation import (
    SegmentationMetrics,
    compute_iou_single,
    compute_dice_single,
    compute_per_image_metrics,
)


def test_metrics_init():
    """Test 1: SegmentationMetrics initialization."""
    print("\n[Test 1] SegmentationMetrics initialization...")

    metrics = SegmentationMetrics(num_classes=2)

    assert metrics.num_classes == 2
    assert metrics.class_names == ["background", "cell"]
    assert metrics.confusion_matrix.shape == (2, 2)
    assert metrics.num_samples == 0

    print("  Initialized with 2 classes")
    print("  PASS")
    return True


def test_single_update():
    """Test 2: Single sample update."""
    print("\n[Test 2] Single sample update...")

    metrics = SegmentationMetrics(num_classes=2)

    # Create simple prediction and target
    pred = torch.tensor([[0, 0, 1, 1],
                         [0, 0, 1, 1]])
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])

    metrics.update(pred, target)

    assert metrics.num_samples == 1
    assert metrics.confusion_matrix.sum() == 8  # 8 pixels

    print(f"  Updated with {pred.numel()} pixels")
    print(f"  Confusion matrix sum: {metrics.confusion_matrix.sum()}")
    print("  PASS")
    return True


def test_batch_update():
    """Test 3: Batch update."""
    print("\n[Test 3] Batch update...")

    metrics = SegmentationMetrics(num_classes=2)

    # Create batch of predictions
    preds = torch.randint(0, 2, (4, 8, 8))
    targets = torch.randint(0, 2, (4, 8, 8))

    metrics.update_batch(preds, targets)

    assert metrics.num_samples == 4
    assert metrics.confusion_matrix.sum() == 4 * 8 * 8

    print(f"  Updated with batch of {preds.shape[0]} samples")
    print(f"  Total pixels: {metrics.confusion_matrix.sum()}")
    print("  PASS")
    return True


def test_metric_computation():
    """Test 4: Metric computation."""
    print("\n[Test 4] Metric computation...")

    metrics = SegmentationMetrics(num_classes=2)

    # Known prediction with some errors
    # Ground truth: half background, half cell
    # Prediction: mostly correct with a few errors
    pred = torch.tensor([[0, 0, 1, 1],
                         [0, 1, 1, 1]])  # One FP in position (1,1)
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])

    metrics.update(pred, target)
    results = metrics.compute()

    # Check that all expected keys exist
    expected_keys = ['iou', 'dice', 'precision', 'recall', 'f1',
                     'iou_background', 'dice_background',
                     'iou_cell', 'dice_cell', 'miou', 'mdice', 'accuracy']
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"

    print(f"  IoU (cell): {results['iou']:.4f}")
    print(f"  Dice (cell): {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print("  PASS")
    return True


def test_perfect_prediction():
    """Test 5: Perfect predictions should give 1.0."""
    print("\n[Test 5] Perfect predictions...")

    metrics = SegmentationMetrics(num_classes=2)

    # Perfect prediction
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])
    pred = target.clone()

    metrics.update(pred, target)
    results = metrics.compute()

    assert abs(results['iou'] - 1.0) < 1e-6, f"IoU should be 1.0, got {results['iou']}"
    assert abs(results['dice'] - 1.0) < 1e-6, f"Dice should be 1.0, got {results['dice']}"
    assert abs(results['accuracy'] - 1.0) < 1e-6, f"Accuracy should be 1.0, got {results['accuracy']}"

    print(f"  IoU: {results['iou']:.4f} (expected 1.0)")
    print(f"  Dice: {results['dice']:.4f} (expected 1.0)")
    print(f"  Accuracy: {results['accuracy']:.4f} (expected 1.0)")
    print("  PASS")
    return True


def test_completely_wrong():
    """Test 6: Completely wrong predictions."""
    print("\n[Test 6] Completely wrong predictions...")

    metrics = SegmentationMetrics(num_classes=2)

    # Completely inverted prediction
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])
    pred = 1 - target  # Inverted

    metrics.update(pred, target)
    results = metrics.compute()

    # IoU should be 0 (no overlap)
    assert results['iou'] < 0.01, f"IoU should be ~0, got {results['iou']}"
    assert results['dice'] < 0.01, f"Dice should be ~0, got {results['dice']}"

    print(f"  IoU: {results['iou']:.4f} (expected ~0)")
    print(f"  Dice: {results['dice']:.4f} (expected ~0)")
    print("  PASS")
    return True


def test_per_image_metrics():
    """Test 7: Per-image metrics function."""
    print("\n[Test 7] Per-image metrics...")

    pred = torch.tensor([[0, 0, 1, 1],
                         [0, 0, 1, 1]])
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])

    results = compute_per_image_metrics(pred, target, num_classes=2)

    assert 'iou' in results
    assert 'dice' in results
    assert 'iou_cell' in results
    assert 'dice_cell' in results

    # Perfect prediction
    assert abs(results['iou'] - 1.0) < 1e-6
    assert abs(results['dice'] - 1.0) < 1e-6

    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Dice: {results['dice']:.4f}")
    print(f"  Background IoU: {results['iou_background']:.4f}")
    print(f"  Cell IoU: {results['iou_cell']:.4f}")
    print("  PASS")
    return True


def test_iou_dice_functions():
    """Test 8: Individual IoU and Dice functions."""
    print("\n[Test 8] Individual IoU and Dice functions...")

    pred = torch.tensor([[0, 0, 1, 1],
                         [0, 0, 1, 1]])
    target = torch.tensor([[0, 0, 1, 1],
                           [0, 0, 1, 1]])

    iou = compute_iou_single(pred, target, num_classes=2)
    dice = compute_dice_single(pred, target, num_classes=2)

    assert iou.shape == (2,)
    assert dice.shape == (2,)
    assert torch.allclose(iou, torch.ones(2))
    assert torch.allclose(dice, torch.ones(2))

    print(f"  IoU per class: {iou.tolist()}")
    print(f"  Dice per class: {dice.tolist()}")
    print("  PASS")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing evaluation/metrics.py")
    print("=" * 60)

    results = []
    tests = [
        ("Metrics init", test_metrics_init),
        ("Single update", test_single_update),
        ("Batch update", test_batch_update),
        ("Metric computation", test_metric_computation),
        ("Perfect prediction", test_perfect_prediction),
        ("Completely wrong", test_completely_wrong),
        ("Per-image metrics", test_per_image_metrics),
        ("IoU/Dice functions", test_iou_dice_functions),
    ]

    for name, func in tests:
        try:
            func()
            results.append((name, "PASS"))
        except AssertionError as e:
            results.append((name, f"FAIL: {e}"))
            print(f"  FAIL: {e}")
        except Exception as e:
            results.append((name, f"ERROR: {e}"))
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for name, status in results:
        symbol = "+" if status == "PASS" else "-"
        print(f"  [{symbol}] {name}: {status}")

    print(f"\n  {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
