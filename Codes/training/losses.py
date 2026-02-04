"""
Loss functions for cell segmentation.

This module provides various loss functions for training segmentation models:
- DiceLoss: Based on Dice coefficient, good for imbalanced classes
- FocalLoss: Focuses on hard examples
- BCEDiceLoss: Combined BCE + Dice (recommended)

Usage:
    from training import get_loss_function

    loss_fn = get_loss_function("bce_dice", bce_weight=0.5, dice_weight=0.5)
    loss_fn = get_loss_function("dice")
    loss_fn = get_loss_function("focal", alpha=0.25, gamma=2.0)

    loss = loss_fn(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


# =============================================================================
# Dice Loss
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.

    Dice = 2 * |A intersect B| / (|A| + |B|)
    Loss = 1 - Dice

    Good for imbalanced classes as it measures overlap rather than
    pixel-wise accuracy.

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: 'mean' or 'none' (default: 'mean')
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Raw predictions (B, C, H, W) where C = num_classes
            targets: Ground truth masks (B, H, W) with class indices

        Returns:
            Dice loss value
        """
        num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets.long(), num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute Dice for each class
        dims = (2, 3)  # Spatial dimensions
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over classes and batch
        if self.reduction == 'mean':
            return 1.0 - dice_score.mean()
        else:
            return 1.0 - dice_score


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    FL = -alpha * (1 - p)^gamma * log(p)

    Down-weights easy examples and focuses on hard ones.

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
            - gamma = 0: equivalent to cross entropy
            - gamma > 0: reduces loss for well-classified examples
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            logits: Raw predictions (B, C, H, W)
            targets: Ground truth masks (B, H, W) with class indices

        Returns:
            Focal loss value
        """
        num_classes = logits.shape[1]

        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')

        # Get probabilities for the target class
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of true class

        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# BCE + Dice Combined Loss
# =============================================================================

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice loss.

    Loss = bce_weight * BCE + dice_weight * Dice

    This is the recommended loss for cell segmentation:
    - BCE ensures pixel-level accuracy
    - Dice handles class imbalance and measures overlap

    Args:
        bce_weight: Weight for BCE loss (default: 0.5)
        dice_weight: Weight for Dice loss (default: 0.5)
        smooth: Smoothing factor for Dice (default: 1.0)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined BCE + Dice loss.

        Args:
            logits: Raw predictions (B, C, H, W)
            targets: Ground truth masks (B, H, W) with class indices

        Returns:
            Combined loss value
        """
        # BCE loss
        bce = F.cross_entropy(logits, targets.long())

        # Dice loss
        dice = self.dice_loss(logits, targets)

        # Combined
        return self.bce_weight * bce + self.dice_weight * dice


# =============================================================================
# Simple Cross Entropy (Baseline)
# =============================================================================

class CrossEntropyLoss(nn.Module):
    """
    Standard cross entropy loss wrapper.

    Wrapper around F.cross_entropy for consistent interface.

    Args:
        weight: Class weights for imbalanced classes (optional)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross entropy loss.

        Args:
            logits: Raw predictions (B, C, H, W)
            targets: Ground truth masks (B, H, W) with class indices

        Returns:
            Cross entropy loss value
        """
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.weight,
            reduction=self.reduction
        )


# =============================================================================
# Factory Function
# =============================================================================

LOSS_REGISTRY = {
    "ce": CrossEntropyLoss,
    "cross_entropy": CrossEntropyLoss,
    "dice": DiceLoss,
    "focal": FocalLoss,
    "bce_dice": BCEDiceLoss,
}


def get_loss_function(
    name: Literal["ce", "cross_entropy", "dice", "focal", "bce_dice"],
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function by name.

    Args:
        name: Loss function name
            - "ce" or "cross_entropy": Standard cross entropy
            - "dice": Dice loss
            - "focal": Focal loss
            - "bce_dice": Combined BCE + Dice (recommended)
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Loss function module

    Example:
        >>> loss_fn = get_loss_function("bce_dice", bce_weight=0.5, dice_weight=0.5)
        >>> loss_fn = get_loss_function("focal", alpha=0.25, gamma=2.0)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[name](**kwargs)


def list_available_losses() -> list:
    """List available loss function names."""
    return list(LOSS_REGISTRY.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def compute_class_weights(
    dataset,
    num_classes: int = 2,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute class weights from dataset for handling imbalance.

    Weights are inversely proportional to class frequency.

    Args:
        dataset: PyTorch dataset with masks
        num_classes: Number of classes
        device: Device to put weights on

    Returns:
        Class weights tensor of shape (num_classes,)
    """
    class_counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        item = dataset[i]
        mask = item.get("mask", item.get("masks"))
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum()

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return weights.to(device)
