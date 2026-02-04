"""
Evaluation module for cell segmentation.

Provides metrics computation and visualization utilities.

Usage:
    from evaluation import SegmentationMetrics, visualize_prediction

    # Compute metrics
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(predictions, targets)
    results = metrics.compute()

    # Visualize
    fig = visualize_prediction(image, pred_mask, gt_mask)
"""

from .metrics import (
    SegmentationMetrics,
    compute_iou_single,
    compute_dice_single,
    compute_per_image_metrics,
)

from .visualize import (
    visualize_prediction,
    visualize_error_map,
    create_grid_visualization,
    save_prediction_visualization,
    denormalize_image,
)

__all__ = [
    # Metrics
    "SegmentationMetrics",
    "compute_iou_single",
    "compute_dice_single",
    "compute_per_image_metrics",
    # Visualization
    "visualize_prediction",
    "visualize_error_map",
    "create_grid_visualization",
    "save_prediction_visualization",
    "denormalize_image",
]
