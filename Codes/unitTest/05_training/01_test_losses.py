"""
Test script for training/losses.py module.

Tests:
1. DiceLoss forward pass
2. FocalLoss forward pass
3. BCEDiceLoss forward pass
4. CrossEntropyLoss forward pass
5. Factory function get_loss_function()
6. Loss values are valid (non-negative, finite)
7. Gradient flow
8. Different batch sizes
9. Loss comparison (relative values)

Usage:
    python Codes/unitTest/05_training/01_test_losses.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import torch
import torch.nn.functional as F

from training.losses import (
    DiceLoss,
    FocalLoss,
    BCEDiceLoss,
    CrossEntropyLoss,
    get_loss_function,
    list_available_losses,
)

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


def create_dummy_data(batch_size=2, num_classes=2, height=224, width=224):
    """Create dummy logits and targets for testing."""
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    return logits, targets


# =============================================================================
# Test Functions
# =============================================================================

def test_dice_loss():
    """Test 1: DiceLoss forward pass."""
    print("\n[Test 1] DiceLoss forward pass...")

    loss_fn = DiceLoss()
    logits, targets = create_dummy_data()

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert loss.item() <= 1, f"Dice loss should be <= 1, got {loss.item()}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    print(f"  Loss value: {loss.item():.4f}")
    print("  PASS")
    return True


def test_focal_loss():
    """Test 2: FocalLoss forward pass."""
    print("\n[Test 2] FocalLoss forward pass...")

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits, targets = create_dummy_data()

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    print(f"  Loss value: {loss.item():.4f}")
    print("  PASS")
    return True


def test_bce_dice_loss():
    """Test 3: BCEDiceLoss forward pass."""
    print("\n[Test 3] BCEDiceLoss forward pass...")

    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits, targets = create_dummy_data()

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    print(f"  Loss value: {loss.item():.4f}")
    print("  PASS")
    return True


def test_cross_entropy_loss():
    """Test 4: CrossEntropyLoss forward pass."""
    print("\n[Test 4] CrossEntropyLoss forward pass...")

    loss_fn = CrossEntropyLoss()
    logits, targets = create_dummy_data()

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    print(f"  Loss value: {loss.item():.4f}")
    print("  PASS")
    return True


def test_factory_function():
    """Test 5: Factory function get_loss_function()."""
    print("\n[Test 5] Factory function get_loss_function()...")

    available = list_available_losses()
    print(f"  Available losses: {available}")

    logits, targets = create_dummy_data()

    for name in available:
        loss_fn = get_loss_function(name)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"Loss for {name} should be finite"
        print(f"  {name}: {loss.item():.4f}")

    # Test invalid name
    try:
        get_loss_function("invalid_loss")
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Invalid name correctly raises: {str(e)[:40]}...")

    print("  PASS")
    return True


def test_loss_validity():
    """Test 6: Loss values are valid (non-negative, finite)."""
    print("\n[Test 6] Loss value validity...")

    losses = [
        ("dice", DiceLoss()),
        ("focal", FocalLoss()),
        ("bce_dice", BCEDiceLoss()),
        ("ce", CrossEntropyLoss()),
    ]

    # Test with various inputs
    test_cases = [
        ("random", *create_dummy_data()),
        ("all_zeros", torch.zeros(2, 2, 32, 32), torch.zeros(2, 32, 32).long()),
        ("all_ones", torch.ones(2, 2, 32, 32), torch.ones(2, 32, 32).long()),
    ]

    for case_name, logits, targets in test_cases:
        print(f"  Case: {case_name}")
        for loss_name, loss_fn in losses:
            loss = loss_fn(logits, targets)
            assert torch.isfinite(loss), f"{loss_name} not finite for {case_name}"
            assert loss.item() >= 0, f"{loss_name} negative for {case_name}"
        print(f"    All losses valid")

    print("  PASS")
    return True


def test_gradient_flow():
    """Test 7: Gradient flow (backpropagation works)."""
    print("\n[Test 7] Gradient flow...")

    losses = [
        ("dice", DiceLoss()),
        ("focal", FocalLoss()),
        ("bce_dice", BCEDiceLoss()),
        ("ce", CrossEntropyLoss()),
    ]

    for loss_name, loss_fn in losses:
        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 32, 32))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None, f"No gradient for {loss_name}"
        assert torch.isfinite(logits.grad).all(), f"Non-finite gradient for {loss_name}"

        print(f"  {loss_name}: gradient flows correctly")

    print("  PASS")
    return True


def test_different_batch_sizes():
    """Test 8: Different batch sizes."""
    print("\n[Test 8] Different batch sizes...")

    loss_fn = BCEDiceLoss()
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        logits, targets = create_dummy_data(batch_size=bs, height=64, width=64)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"Loss not finite for batch_size={bs}"
        print(f"  batch_size={bs}: loss={loss.item():.4f}")

    print("  PASS")
    return True


def test_loss_comparison():
    """Test 9: Loss comparison (perfect vs random predictions)."""
    print("\n[Test 9] Loss comparison (perfect vs random)...")

    loss_fn = BCEDiceLoss()

    # Create targets
    targets = torch.randint(0, 2, (2, 64, 64))

    # Perfect predictions (high confidence for correct class)
    perfect_logits = torch.zeros(2, 2, 64, 64)
    for b in range(2):
        for h in range(64):
            for w in range(64):
                c = targets[b, h, w].item()
                perfect_logits[b, c, h, w] = 10.0  # High logit for correct class

    # Random predictions
    random_logits = torch.randn(2, 2, 64, 64)

    # Bad predictions (high confidence for wrong class)
    bad_logits = torch.zeros(2, 2, 64, 64)
    for b in range(2):
        for h in range(64):
            for w in range(64):
                c = targets[b, h, w].item()
                bad_logits[b, 1-c, h, w] = 10.0  # High logit for wrong class

    perfect_loss = loss_fn(perfect_logits, targets)
    random_loss = loss_fn(random_logits, targets)
    bad_loss = loss_fn(bad_logits, targets)

    print(f"  Perfect predictions: {perfect_loss.item():.4f}")
    print(f"  Random predictions:  {random_loss.item():.4f}")
    print(f"  Bad predictions:     {bad_loss.item():.4f}")

    # Perfect should have lowest loss
    assert perfect_loss < random_loss, "Perfect should be better than random"
    assert random_loss < bad_loss, "Random should be better than bad"

    print("  Loss ordering correct: perfect < random < bad")
    print("  PASS")
    return True


def test_dice_loss_perfect_overlap():
    """Test 10: Dice loss with perfect overlap should be ~0."""
    print("\n[Test 10] Dice loss with perfect overlap...")

    loss_fn = DiceLoss(smooth=1e-6)

    # Create targets
    targets = torch.randint(0, 2, (2, 64, 64))

    # Perfect predictions
    perfect_logits = torch.zeros(2, 2, 64, 64)
    for b in range(2):
        for h in range(64):
            for w in range(64):
                c = targets[b, h, w].item()
                perfect_logits[b, c, h, w] = 100.0  # Very high logit

    loss = loss_fn(perfect_logits, targets)

    print(f"  Dice loss with perfect overlap: {loss.item():.6f}")
    assert loss.item() < 0.01, f"Perfect overlap should have near-zero loss, got {loss.item()}"

    print("  PASS")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing training/losses.py")
    print("=" * 60)

    results = []

    tests = [
        ("DiceLoss forward", test_dice_loss),
        ("FocalLoss forward", test_focal_loss),
        ("BCEDiceLoss forward", test_bce_dice_loss),
        ("CrossEntropyLoss forward", test_cross_entropy_loss),
        ("Factory function", test_factory_function),
        ("Loss validity", test_loss_validity),
        ("Gradient flow", test_gradient_flow),
        ("Different batch sizes", test_different_batch_sizes),
        ("Loss comparison", test_loss_comparison),
        ("Dice perfect overlap", test_dice_loss_perfect_overlap),
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

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
