"""
Visualization utilities for cell segmentation evaluation.

Provides:
- Prediction overlay on original image
- Side-by-side comparison (image, ground truth, prediction)
- Error map visualization
- Batch visualization grid

Usage:
    from evaluation import visualize_prediction, create_comparison_figure

    fig = visualize_prediction(image, pred_mask, gt_mask)
    fig.savefig("prediction.png")
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional, List, Tuple, Union
from PIL import Image


# Color maps for visualization
MASK_CMAP = ListedColormap(['black', 'red'])  # Background=black, Cell=red
OVERLAY_ALPHA = 0.4


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image tensor back to [0, 1] range.

    Args:
        tensor: Normalized image tensor (C, H, W) or (H, W, C)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Image array in [0, 1] range, shape (H, W, C)
    """
    img = tensor_to_numpy(tensor)

    # Handle channel dimension
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean

    # Clip to valid range
    img = np.clip(img, 0, 1)

    return img


def visualize_prediction(
    image: Union[torch.Tensor, np.ndarray],
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    denormalize: bool = True,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create visualization comparing prediction with ground truth.

    Args:
        image: Input image (C, H, W) or (H, W, C)
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W), optional
        title: Figure title
        denormalize: Whether to denormalize image from ImageNet normalization
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if denormalize:
        img = denormalize_image(image)
    else:
        img = tensor_to_numpy(image)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    pred = tensor_to_numpy(pred_mask)
    gt = tensor_to_numpy(gt_mask) if gt_mask is not None else None

    # Create figure
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # 1. Original image
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # 2. Ground truth (if provided)
    col = 1
    if gt is not None:
        axes[col].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=1)
        axes[col].set_title("Ground Truth")
        axes[col].axis('off')
        col += 1

    # 3. Prediction
    axes[col].imshow(pred, cmap=MASK_CMAP, vmin=0, vmax=1)
    axes[col].set_title("Prediction")
    axes[col].axis('off')
    col += 1

    # 4. Overlay
    overlay = img.copy()
    pred_bool = pred > 0.5
    overlay[pred_bool] = overlay[pred_bool] * (1 - OVERLAY_ALPHA) + \
                         np.array([1, 0, 0]) * OVERLAY_ALPHA  # Red overlay
    axes[col].imshow(overlay)
    axes[col].set_title("Prediction Overlay")
    axes[col].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def visualize_error_map(
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Visualize prediction errors.

    Colors:
    - Green: True positive (correct cell)
    - Red: False positive (predicted cell, actually background)
    - Blue: False negative (missed cell)
    - Black: True negative (correct background)

    Args:
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W)
        image: Optional input image for reference
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    pred = tensor_to_numpy(pred_mask) > 0.5
    gt = tensor_to_numpy(gt_mask) > 0.5

    # Create error map
    # 0: TN (black), 1: TP (green), 2: FP (red), 3: FN (blue)
    error_map = np.zeros(pred.shape, dtype=np.uint8)
    error_map[(gt == 1) & (pred == 1)] = 1  # TP
    error_map[(gt == 0) & (pred == 1)] = 2  # FP
    error_map[(gt == 1) & (pred == 0)] = 3  # FN

    # Custom colormap
    colors = ['black', 'green', 'red', 'blue']
    cmap = ListedColormap(colors)

    # Create figure
    n_cols = 3 if image is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    col = 0
    if image is not None:
        img = denormalize_image(image) if isinstance(image, torch.Tensor) else image
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        axes[col].imshow(img)
        axes[col].set_title("Input Image")
        axes[col].axis('off')
        col += 1

    # Error map
    axes[col].imshow(error_map, cmap=cmap, vmin=0, vmax=3)
    axes[col].set_title("Error Map")
    axes[col].axis('off')
    col += 1

    # Legend
    axes[col].axis('off')
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                   markersize=15, label='True Positive'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=15, label='False Positive'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=15, label='False Negative'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=15, label='True Negative'),
    ]
    axes[col].legend(handles=legend_elements, loc='center', fontsize=12)
    axes[col].set_title("Legend")

    # Compute error stats
    tp = (error_map == 1).sum()
    fp = (error_map == 2).sum()
    fn = (error_map == 3).sum()
    tn = (error_map == 0).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    fig.suptitle(f"IoU: {iou:.4f} | Dice: {dice:.4f} | FP: {fp} | FN: {fn}", fontsize=12)

    plt.tight_layout()
    return fig


def create_grid_visualization(
    images: List[Union[torch.Tensor, np.ndarray]],
    pred_masks: List[Union[torch.Tensor, np.ndarray]],
    gt_masks: List[Union[torch.Tensor, np.ndarray]],
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize_per_image: Tuple[float, float] = (4, 4)
) -> plt.Figure:
    """
    Create grid visualization of multiple predictions.

    Args:
        images: List of input images
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        titles: Optional titles for each row
        n_cols: Number of columns (image, gt, pred, overlay)
        figsize_per_image: Size per subplot

    Returns:
        Matplotlib figure
    """
    n_samples = len(images)
    n_cols = 4  # image, gt, pred, overlay

    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_samples)
    )

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        img = denormalize_image(images[i])
        pred = tensor_to_numpy(pred_masks[i])
        gt = tensor_to_numpy(gt_masks[i])

        # Image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Image")

        # Ground truth
        axes[i, 1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("Ground Truth")

        # Prediction
        axes[i, 2].imshow(pred, cmap=MASK_CMAP, vmin=0, vmax=1)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Prediction")

        # Overlay
        overlay = img.copy()
        pred_bool = pred > 0.5
        overlay[pred_bool] = overlay[pred_bool] * (1 - OVERLAY_ALPHA) + \
                             np.array([1, 0, 0]) * OVERLAY_ALPHA
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title("Overlay")

        # Row title
        if titles and i < len(titles):
            axes[i, 0].set_ylabel(titles[i], fontsize=10, rotation=0, ha='right')

    plt.tight_layout()
    return fig


def save_prediction_visualization(
    image: Union[torch.Tensor, np.ndarray],
    pred_mask: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    output_path: Path,
    title: Optional[str] = None,
    metrics: Optional[dict] = None
):
    """
    Save prediction visualization to file.

    Args:
        image: Input image
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        output_path: Path to save figure
        title: Optional title
        metrics: Optional metrics dict to include in title
    """
    if metrics:
        title = f"{title or ''} | IoU: {metrics.get('iou', 0):.4f} | Dice: {metrics.get('dice', 0):.4f}"

    fig = visualize_prediction(image, pred_mask, gt_mask, title=title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
