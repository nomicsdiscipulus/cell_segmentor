"""
Training loop and utilities for cell segmentation.

This module provides:
- Trainer class for training and validation
- Optimizer and scheduler setup
- Checkpointing and logging
- Early stopping

Usage:
    from training import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cuda"
    )

    history = trainer.fit(num_epochs=100)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import json
import time


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred: Predicted masks (B, H, W) with class indices
        target: Ground truth masks (B, H, W) with class indices
        num_classes: Number of classes

    Returns:
        IoU for each class, shape (num_classes,)
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union == 0:
            iou = torch.tensor(1.0)  # Empty class, perfect score
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.stack(ious)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute Dice coefficient for each class.

    Args:
        pred: Predicted masks (B, H, W) with class indices
        target: Ground truth masks (B, H, W) with class indices
        num_classes: Number of classes

    Returns:
        Dice for each class, shape (num_classes,)
    """
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        cardinality = pred_c.sum() + target_c.sum()

        if cardinality == 0:
            dice = torch.tensor(1.0)  # Empty class, perfect score
        else:
            dice = 2 * intersection / cardinality

        dices.append(dice)

    return torch.stack(dices)


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer for cell segmentation models.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        model: The segmentation model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer (if None, creates AdamW)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ("cuda" or "cpu")
        num_classes: Number of segmentation classes
        checkpoint_dir: Directory to save checkpoints
        log_interval: Log every N batches (0 to disable batch logging)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        num_classes: int = 2,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.log_interval = log_interval

        # Loss function
        if loss_fn is None:
            from .losses import BCEDiceLoss
            loss_fn = BCEDiceLoss()
        self.loss_fn = loss_fn

        # Optimizer
        if optimizer is None:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
                weight_decay=1e-4
            )
        self.optimizer = optimizer

        # Scheduler
        self.scheduler = scheduler

        # Checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_iou": [],
            "val_iou": [],
            "train_dice": [],
            "val_dice": [],
            "lr": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with train_loss, train_iou, train_dice
        """
        self.model.train()

        total_loss = 0.0
        total_iou = torch.zeros(self.num_classes)
        total_dice = torch.zeros(self.num_classes)
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            images = self._get_images(batch).to(self.device)
            masks = self._get_masks(batch).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                iou = compute_iou(pred, masks, self.num_classes)
                dice = compute_dice(pred, masks, self.num_classes)

            total_loss += loss.item()
            total_iou += iou.cpu()
            total_dice += dice.cpu()
            num_batches += 1

            # Log batch progress
            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                print(f"    Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"loss={loss.item():.4f}")

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches

        return {
            "train_loss": avg_loss,
            "train_iou": avg_iou.mean().item(),
            "train_dice": avg_dice.mean().item(),
            "train_iou_per_class": avg_iou.tolist(),
            "train_dice_per_class": avg_dice.tolist(),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dict with val_loss, val_iou, val_dice
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_iou = torch.zeros(self.num_classes)
        total_dice = torch.zeros(self.num_classes)
        num_batches = 0

        for batch in self.val_loader:
            # Get data
            images = self._get_images(batch).to(self.device)
            masks = self._get_masks(batch).to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)

            # Compute metrics
            pred = logits.argmax(dim=1)
            iou = compute_iou(pred, masks, self.num_classes)
            dice = compute_dice(pred, masks, self.num_classes)

            total_loss += loss.item()
            total_iou += iou.cpu()
            total_dice += dice.cpu()
            num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches

        return {
            "val_loss": avg_loss,
            "val_iou": avg_iou.mean().item(),
            "val_dice": avg_dice.mean().item(),
            "val_iou_per_class": avg_iou.tolist(),
            "val_dice_per_class": avg_dice.tolist(),
        }

    def fit(
        self,
        num_epochs: int,
        early_stopping_patience: int = 0,
        save_best: bool = True,
        save_last: bool = True
    ) -> Dict[str, List]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if val_loss doesn't improve for N epochs
                (0 to disable)
            save_best: Save checkpoint when val_loss improves
            save_last: Save checkpoint after last epoch

        Returns:
            Training history dict
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print()

        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()

            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_iou"].append(train_metrics["train_iou"])
            self.history["train_dice"].append(train_metrics["train_dice"])
            self.history["lr"].append(current_lr)

            if val_metrics:
                self.history["val_loss"].append(val_metrics["val_loss"])
                self.history["val_iou"].append(val_metrics["val_iou"])
                self.history["val_dice"].append(val_metrics["val_dice"])

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, num_epochs, train_metrics, val_metrics, current_lr, epoch_time)

            # Check for improvement
            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0

                if save_best and self.checkpoint_dir:
                    self.save_checkpoint("best.pth", epoch, val_metrics)
            else:
                patience_counter += 1

            # Early stopping
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break

        # Save last checkpoint
        if save_last and self.checkpoint_dir:
            self.save_checkpoint("last.pth", epoch, val_metrics)

        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {total_time / 60:.1f} min")
        print(f"  Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.4f})")

        # Save history
        if self.checkpoint_dir:
            self._save_history()

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint["epoch"]

    def _get_images(self, batch: Dict) -> torch.Tensor:
        """Extract images from batch (handles different collate formats)."""
        if "images" in batch:
            images = batch["images"]
            if isinstance(images, list):
                return torch.stack(images)
            return images
        elif "image" in batch:
            return batch["image"]
        else:
            raise KeyError("Batch must contain 'images' or 'image' key")

    def _get_masks(self, batch: Dict) -> torch.Tensor:
        """Extract masks from batch (handles different collate formats)."""
        if "masks" in batch:
            masks = batch["masks"]
            if isinstance(masks, list):
                return torch.stack(masks)
            return masks
        elif "mask" in batch:
            return batch["mask"]
        else:
            raise KeyError("Batch must contain 'masks' or 'mask' key")

    def _log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float
    ):
        """Log epoch summary."""
        msg = f"Epoch {epoch:3d}/{num_epochs}"
        msg += f" | train_loss: {train_metrics['train_loss']:.4f}"
        msg += f" | train_dice: {train_metrics['train_dice']:.4f}"

        if val_metrics:
            msg += f" | val_loss: {val_metrics['val_loss']:.4f}"
            msg += f" | val_dice: {val_metrics['val_dice']:.4f}"

        msg += f" | lr: {lr:.2e}"
        msg += f" | time: {epoch_time:.1f}s"

        print(msg)

    def _save_history(self):
        """Save training history to JSON."""
        if self.checkpoint_dir is None:
            return

        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# Optimizer and Scheduler Factories
# =============================================================================

def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer.

    Args:
        model: Model to optimize
        name: Optimizer name ("adam", "adamw", "sgd")
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    # Only optimize trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())

    if name.lower() == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name.lower() == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name.lower() == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    num_epochs: int = 100,
    **kwargs
) -> Any:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        name: Scheduler name ("cosine", "step", "plateau", "none")
        num_epochs: Total number of epochs (for cosine)
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance or None
    """
    if name.lower() == "none":
        return None
    elif name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, **kwargs
        )
    elif name.lower() == "step":
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, **kwargs
        )
    elif name.lower() == "plateau":
        patience = kwargs.pop("patience", 10)
        factor = kwargs.pop("factor", 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
