"""
Evaluation metrics for cell segmentation.

Provides:
- IoU (Intersection over Union)
- Dice coefficient
- Precision, Recall, F1
- Per-class and mean metrics

Usage:
    from evaluation import SegmentationMetrics

    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(predictions, targets)
    results = metrics.compute()
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics.

    Computes metrics over multiple batches/images, then returns
    aggregated results.

    Args:
        num_classes: Number of segmentation classes
        class_names: Optional names for each class
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        if num_classes == 2:
            self.class_names = ["background", "cell"]

        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        # Confusion matrix: [num_classes, num_classes]
        # confusion[i, j] = pixels with true class i predicted as class j
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )
        self.num_samples = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """
        Update metrics with new predictions.

        Args:
            pred: Predicted masks (B, H, W) or (H, W) with class indices
            target: Ground truth masks (B, H, W) or (H, W) with class indices
        """
        # Ensure tensors
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        # Flatten to 1D
        pred = pred.flatten().long()
        target = target.flatten().long()

        # Update confusion matrix
        for t, p in zip(target, pred):
            self.confusion_matrix[t, p] += 1

        self.num_samples += 1

    def update_batch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with batch of predictions.

        Args:
            preds: Predicted masks (B, H, W)
            targets: Ground truth masks (B, H, W)
        """
        batch_size = preds.shape[0]
        for i in range(batch_size):
            self.update(preds[i], targets[i])

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated confusion matrix.

        Returns:
            Dict with IoU, Dice, precision, recall for each class and mean
        """
        results = {}
        cm = self.confusion_matrix.float()

        # Per-class metrics
        ious = []
        dices = []
        precisions = []
        recalls = []

        for c in range(self.num_classes):
            tp = cm[c, c]  # True positives
            fp = cm[:, c].sum() - tp  # False positives
            fn = cm[c, :].sum() - tp  # False negatives

            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-8)
            ious.append(iou.item())

            # Dice = 2*TP / (2*TP + FP + FN)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            dices.append(dice.item())

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp + 1e-8)
            precisions.append(precision.item())

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn + 1e-8)
            recalls.append(recall.item())

            # Store per-class results
            class_name = self.class_names[c]
            results[f"iou_{class_name}"] = ious[-1]
            results[f"dice_{class_name}"] = dices[-1]
            results[f"precision_{class_name}"] = precisions[-1]
            results[f"recall_{class_name}"] = recalls[-1]

        # Mean metrics (excluding background for segmentation)
        if self.num_classes == 2:
            # For binary, report cell class as primary metric
            results["iou"] = ious[1]  # Cell IoU
            results["dice"] = dices[1]  # Cell Dice
            results["precision"] = precisions[1]
            results["recall"] = recalls[1]
        else:
            # For multi-class, use mean
            results["iou"] = np.mean(ious)
            results["dice"] = np.mean(dices)
            results["precision"] = np.mean(precisions)
            results["recall"] = np.mean(recalls)

        # Mean IoU (all classes)
        results["miou"] = np.mean(ious)
        results["mdice"] = np.mean(dices)

        # F1 score
        p, r = results["precision"], results["recall"]
        results["f1"] = 2 * p * r / (p + r + 1e-8)

        # Accuracy
        total = cm.sum()
        correct = cm.diag().sum()
        results["accuracy"] = (correct / (total + 1e-8)).item()

        # Sample count
        results["num_samples"] = self.num_samples

        return results

    def get_confusion_matrix(self) -> torch.Tensor:
        """Return the confusion matrix."""
        return self.confusion_matrix


def compute_iou_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2
) -> torch.Tensor:
    """
    Compute IoU for a single prediction.

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        IoU per class, shape (num_classes,)
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.stack(ious)


def compute_dice_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2
) -> torch.Tensor:
    """
    Compute Dice for a single prediction.

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dice per class, shape (num_classes,)
    """
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        cardinality = pred_c.sum() + target_c.sum()

        if cardinality == 0:
            dice = torch.tensor(1.0)
        else:
            dice = 2 * intersection / cardinality

        dices.append(dice)

    return torch.stack(dices)


def compute_per_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute metrics for a single image.

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dict with iou, dice per class and mean
    """
    iou = compute_iou_single(pred, target, num_classes)
    dice = compute_dice_single(pred, target, num_classes)

    results = {
        "iou_background": iou[0].item(),
        "iou_cell": iou[1].item() if num_classes > 1 else iou[0].item(),
        "dice_background": dice[0].item(),
        "dice_cell": dice[1].item() if num_classes > 1 else dice[0].item(),
        "miou": iou.mean().item(),
        "mdice": dice.mean().item(),
    }

    # Primary metrics (cell class for binary)
    if num_classes == 2:
        results["iou"] = results["iou_cell"]
        results["dice"] = results["dice_cell"]
    else:
        results["iou"] = results["miou"]
        results["dice"] = results["mdice"]

    return results
