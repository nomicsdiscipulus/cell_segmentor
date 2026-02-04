"""
Test script for training/trainer.py module.

Tests:
1. Trainer initialization
2. Single training step
3. Single validation step
4. Multiple epochs training
5. Checkpointing (save/load)
6. Optimizer and scheduler creation
7. Metrics computation (IoU, Dice)
8. Early stopping

Usage:
    python Codes/unitTest/05_training/02_test_trainer.py
"""

import sys
from pathlib import Path
import shutil

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    compute_iou,
    compute_dice,
    get_loss_function,
)

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


# =============================================================================
# Dummy Model for Testing
# =============================================================================

class DummySegmentationModel(nn.Module):
    """Simple model for testing trainer."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)


def create_dummy_dataloader(batch_size=4, num_samples=16, img_size=64):
    """Create dummy dataloader for testing."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    masks = torch.randint(0, 2, (num_samples, img_size, img_size))

    dataset = TensorDataset(images, masks)

    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        return {"images": images, "masks": masks}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# =============================================================================
# Test Functions
# =============================================================================

def test_trainer_init():
    """Test 1: Trainer initialization."""
    print("\n[Test 1] Trainer initialization...")

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device="cpu"
    )

    assert trainer.model is not None
    assert trainer.train_loader is not None
    assert trainer.loss_fn is not None
    assert trainer.optimizer is not None

    print(f"  Model: {type(model).__name__}")
    print(f"  Loss: {type(trainer.loss_fn).__name__}")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print("  PASS")
    return True


def test_single_train_step():
    """Test 2: Single training step."""
    print("\n[Test 2] Single training step...")

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader(batch_size=2, num_samples=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device="cpu",
        log_interval=0  # Disable batch logging
    )

    # Run one epoch
    metrics = trainer.train_epoch()

    assert "train_loss" in metrics
    assert "train_iou" in metrics
    assert "train_dice" in metrics
    assert metrics["train_loss"] >= 0

    print(f"  Train loss: {metrics['train_loss']:.4f}")
    print(f"  Train IoU: {metrics['train_iou']:.4f}")
    print(f"  Train Dice: {metrics['train_dice']:.4f}")
    print("  PASS")
    return True


def test_single_val_step():
    """Test 3: Single validation step."""
    print("\n[Test 3] Single validation step...")

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader(batch_size=2, num_samples=4)
    val_loader = create_dummy_dataloader(batch_size=2, num_samples=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu"
    )

    # Run validation
    metrics = trainer.validate()

    assert "val_loss" in metrics
    assert "val_iou" in metrics
    assert "val_dice" in metrics

    print(f"  Val loss: {metrics['val_loss']:.4f}")
    print(f"  Val IoU: {metrics['val_iou']:.4f}")
    print(f"  Val Dice: {metrics['val_dice']:.4f}")
    print("  PASS")
    return True


def test_multi_epoch_training():
    """Test 4: Multiple epochs training."""
    print("\n[Test 4] Multiple epochs training...")

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader(batch_size=2, num_samples=8)
    val_loader = create_dummy_dataloader(batch_size=2, num_samples=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        log_interval=0
    )

    # Train for 3 epochs
    history = trainer.fit(num_epochs=3, save_best=False, save_last=False)

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert len(history["lr"]) == 3

    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print("  PASS")
    return True


def test_checkpointing():
    """Test 5: Checkpointing (save/load)."""
    print("\n[Test 5] Checkpointing...")

    # Create output directory
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader(batch_size=2, num_samples=4)
    val_loader = create_dummy_dataloader(batch_size=2, num_samples=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        log_interval=0
    )

    # Train and save
    trainer.fit(num_epochs=2, save_best=True, save_last=True)

    # Check files exist
    assert (checkpoint_dir / "best.pth").exists(), "best.pth should exist"
    assert (checkpoint_dir / "last.pth").exists(), "last.pth should exist"
    assert (checkpoint_dir / "history.json").exists(), "history.json should exist"

    # Load checkpoint into new model
    model2 = DummySegmentationModel()
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        device="cpu"
    )
    epoch = trainer2.load_checkpoint(checkpoint_dir / "best.pth")

    assert epoch > 0, "Should load epoch number"

    print(f"  Saved checkpoints: best.pth, last.pth, history.json")
    print(f"  Loaded epoch: {epoch}")
    print("  PASS")

    # Cleanup
    shutil.rmtree(checkpoint_dir)
    return True


def test_optimizer_creation():
    """Test 6: Optimizer creation."""
    print("\n[Test 6] Optimizer creation...")

    model = DummySegmentationModel()

    # Test different optimizers
    optimizers = ["adam", "adamw", "sgd"]

    for opt_name in optimizers:
        optimizer = create_optimizer(model, name=opt_name, lr=1e-3)
        assert optimizer is not None
        print(f"  {opt_name}: {type(optimizer).__name__}")

    # Test invalid optimizer
    try:
        create_optimizer(model, name="invalid")
        assert False, "Should raise ValueError"
    except ValueError:
        print("  Invalid optimizer raises ValueError")

    print("  PASS")
    return True


def test_scheduler_creation():
    """Test 7: Scheduler creation."""
    print("\n[Test 7] Scheduler creation...")

    model = DummySegmentationModel()
    optimizer = create_optimizer(model)

    # Test different schedulers
    schedulers = ["cosine", "step", "plateau", "none"]

    for sched_name in schedulers:
        scheduler = create_scheduler(optimizer, name=sched_name, num_epochs=100)
        if sched_name == "none":
            assert scheduler is None
            print(f"  {sched_name}: None")
        else:
            assert scheduler is not None
            print(f"  {sched_name}: {type(scheduler).__name__}")

    print("  PASS")
    return True


def test_metrics_computation():
    """Test 8: Metrics computation (IoU, Dice)."""
    print("\n[Test 8] Metrics computation...")

    # Perfect predictions
    pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

    iou = compute_iou(pred, target, num_classes=2)
    dice = compute_dice(pred, target, num_classes=2)

    assert torch.allclose(iou, torch.ones(2)), f"Perfect IoU should be 1, got {iou}"
    assert torch.allclose(dice, torch.ones(2)), f"Perfect Dice should be 1, got {dice}"

    print(f"  Perfect prediction IoU: {iou.tolist()}")
    print(f"  Perfect prediction Dice: {dice.tolist()}")

    # Random predictions (should be less than perfect)
    pred_random = torch.randint(0, 2, (2, 4))
    target_random = torch.randint(0, 2, (2, 4))

    iou_random = compute_iou(pred_random, target_random, num_classes=2)
    dice_random = compute_dice(pred_random, target_random, num_classes=2)

    assert (iou_random >= 0).all() and (iou_random <= 1).all()
    assert (dice_random >= 0).all() and (dice_random <= 1).all()

    print(f"  Random prediction IoU: {iou_random.tolist()}")
    print(f"  Random prediction Dice: {dice_random.tolist()}")
    print("  PASS")
    return True


def test_early_stopping():
    """Test 9: Early stopping."""
    print("\n[Test 9] Early stopping...")

    model = DummySegmentationModel()
    train_loader = create_dummy_dataloader(batch_size=2, num_samples=4)
    val_loader = create_dummy_dataloader(batch_size=2, num_samples=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        log_interval=0
    )

    # Set best_val_loss to very low value to trigger early stopping
    trainer.best_val_loss = -float("inf")

    # Train with early stopping (patience=2)
    history = trainer.fit(
        num_epochs=10,
        early_stopping_patience=2,
        save_best=False,
        save_last=False
    )

    # Should stop before 10 epochs
    epochs_run = len(history["train_loss"])
    assert epochs_run <= 3, f"Should stop early, but ran {epochs_run} epochs"

    print(f"  Requested epochs: 10")
    print(f"  Actual epochs: {epochs_run} (stopped early)")
    print("  PASS")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing training/trainer.py")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    tests = [
        ("Trainer init", test_trainer_init),
        ("Single train step", test_single_train_step),
        ("Single val step", test_single_val_step),
        ("Multi-epoch training", test_multi_epoch_training),
        ("Checkpointing", test_checkpointing),
        ("Optimizer creation", test_optimizer_creation),
        ("Scheduler creation", test_scheduler_creation),
        ("Metrics computation", test_metrics_computation),
        ("Early stopping", test_early_stopping),
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
            import traceback
            traceback.print_exc()

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
