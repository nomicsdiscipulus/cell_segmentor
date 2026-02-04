"""
Evaluation script for trained cell segmentation model.

Runs inference on test set with full tiling and computes metrics.

Usage:
    python Codes/evaluate.py --checkpoint outputs/training/exp/best.pth
    python Codes/evaluate.py --checkpoint outputs/training/exp/best.pth --visualize 10

Arguments:
    --checkpoint: Path to model checkpoint
    --mode: Inference mode ("padding" or "tiling")
    --visualize: Number of samples to visualize (0 to disable)
    --output_dir: Output directory for results
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Path setup
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import CellSegmentor
from data import (
    create_test_dataset,
    get_collate_fn,
    stitch_tiles,
    create_gaussian_weight_map,
    unpad,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
)
from evaluation import (
    SegmentationMetrics,
    compute_per_image_metrics,
    visualize_prediction,
    visualize_error_map,
    create_grid_visualization,
    save_prediction_visualization,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate cell segmentation model on test set"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (default: same directory as checkpoint)"
    )
    parser.add_argument(
        "--mode", type=str, default="tiling",
        choices=["padding", "tiling"],
        help="Inference mode"
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to BCCD data root"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize", type=int, default=10,
        help="Number of samples to visualize (0 to disable)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Prediction threshold for binary mask"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to evaluate (default: all). Use for quick testing."
    )

    args = parser.parse_args()

    # Defaults
    if args.data_root is None:
        args.data_root = PROJECT_ROOT / "Data" / "BCCD"
    else:
        args.data_root = Path(args.data_root)

    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output_dir = checkpoint_dir / "evaluation"
    else:
        args.output_dir = Path(args.output_dir)

    if args.config is None:
        args.config = Path(args.checkpoint).parent / "config.json"
    else:
        args.config = Path(args.config)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def load_model(checkpoint_path: Path, config_path: Path, device: str):
    """Load model from checkpoint."""
    # Load config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        backbone_size = config.get("backbone", "small")
        head_type = config.get("head", "linear")
    else:
        print(f"Config not found at {config_path}, using defaults")
        backbone_size = "small"
        head_type = "linear"

    # Create model
    print(f"Creating model: DinoBloom-{backbone_size} + {head_type} head")
    model = CellSegmentor(
        backbone_size=backbone_size,
        head_type=head_type,
        num_classes=2,
        freeze_backbone=True,
        device=device
    )

    # Load weights
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Print checkpoint info
    epoch = checkpoint.get("epoch", "unknown")
    metrics = checkpoint.get("metrics", {})
    print(f"  Epoch: {epoch}")
    if metrics:
        print(f"  Val loss: {metrics.get('val_loss', 'N/A'):.4f}")
        print(f"  Val dice: {metrics.get('val_dice', 'N/A'):.4f}")

    return model, config


@torch.no_grad()
def predict_padding(model, image_tensor, device):
    """
    Run inference with padding approach.

    Args:
        model: Segmentation model
        image_tensor: Padded image tensor (C, H, W)
        device: Device

    Returns:
        Predicted mask (H, W)
    """
    # Add batch dimension
    x = image_tensor.unsqueeze(0).to(device)

    # Forward pass
    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    # Get cell probability
    cell_prob = probs[0, 1].cpu()  # (H, W)

    return cell_prob


@torch.no_grad()
def predict_tiling(
    model,
    tiles: torch.Tensor,
    positions: list,
    original_size: tuple,
    device: str,
    tile_size: int = DEFAULT_TILE_SIZE
):
    """
    Run inference with tiling approach.

    Args:
        model: Segmentation model
        tiles: Tile tensors (N, C, H, W)
        positions: List of (x, y) positions for each tile
        original_size: Original image size (W, H)
        device: Device
        tile_size: Tile size

    Returns:
        Predicted probability map (H, W)
    """
    # Process tiles in batches
    batch_size = 16
    all_probs = []

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size].to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        cell_probs = probs[:, 1].cpu()  # (B, H, W)
        all_probs.append(cell_probs)

    all_probs = torch.cat(all_probs, dim=0)  # (N, H, W)

    # Stitch tiles
    W, H = original_size
    stitched = stitch_tiles(
        tiles=all_probs.numpy(),
        positions=positions,
        output_size=(W, H),  # (width, height)
        tile_size=tile_size
    )

    return torch.from_numpy(stitched)


def evaluate_test_set(model, args):
    """
    Evaluate model on test set.

    Args:
        model: Segmentation model
        args: Command line arguments

    Returns:
        Dict with evaluation results
    """
    print(f"\nLoading test dataset...")
    test_dataset = create_test_dataset(
        data_root=args.data_root,
        mode=args.mode,
        tile_size=DEFAULT_TILE_SIZE,
        overlap=DEFAULT_OVERLAP
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Determine number of samples to evaluate
    total_samples = len(test_dataset)
    if args.num_samples is not None:
        total_samples = min(args.num_samples, len(test_dataset))
        print(f"  Evaluating: {total_samples} samples (quick mode)")

    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=2)
    per_image_results = []

    # Output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Process each image
    print(f"\nRunning inference (mode={args.mode})...")
    for idx in tqdm(range(total_samples)):
        item = test_dataset[idx]
        filename = item["filename"]
        gt_mask = item["mask"]

        # Run inference based on mode
        if args.mode == "padding":
            image_tensor = item["image"]
            padding_info = item["padding_info"]

            pred_prob = predict_padding(model, image_tensor, args.device)

            # Remove padding
            pred_prob = unpad(pred_prob.numpy(), padding_info)
            pred_prob = torch.from_numpy(pred_prob)

            # Also unpad ground truth for comparison
            gt_mask = unpad(gt_mask.numpy(), padding_info)
            gt_mask = torch.from_numpy(gt_mask)

        else:  # tiling
            tiles = item["tiles"]
            positions = item["positions"]
            original_size = item["original_size"]

            pred_prob = predict_tiling(
                model, tiles, positions, original_size,
                args.device, DEFAULT_TILE_SIZE
            )

        # Threshold to binary mask
        pred_mask = (pred_prob > args.threshold).long()
        gt_mask_int = (gt_mask > 0.5).long()

        # Update metrics
        metrics.update(pred_mask, gt_mask_int)

        # Per-image metrics
        img_metrics = compute_per_image_metrics(pred_mask, gt_mask_int, num_classes=2)
        img_metrics["filename"] = filename
        per_image_results.append(img_metrics)

        # Visualize selected samples
        if args.visualize > 0 and idx < args.visualize:
            # Load original image for visualization
            img_path = args.data_root / "test" / "original" / filename
            original_img = np.array(Image.open(img_path).convert('RGB')) / 255.0

            # Save visualization
            save_prediction_visualization(
                image=original_img,
                pred_mask=pred_mask.numpy(),
                gt_mask=gt_mask_int.numpy(),
                output_path=vis_dir / f"{Path(filename).stem}_pred.png",
                title=filename,
                metrics=img_metrics
            )

            # Save error map
            fig = visualize_error_map(
                pred_mask=pred_mask.numpy(),
                gt_mask=gt_mask_int.numpy(),
                image=original_img
            )
            fig.savefig(vis_dir / f"{Path(filename).stem}_error.png", dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Compute final metrics
    results = metrics.compute()

    return results, per_image_results


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("Cell Segmentation Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    if args.num_samples:
        print(f"Samples: {args.num_samples} (quick mode)")
    print()

    # Load model
    model, config = load_model(
        Path(args.checkpoint),
        args.config,
        args.device
    )

    # Run evaluation
    results, per_image_results = evaluate_test_set(model, args)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics (Cell Class):")
    print(f"  IoU:       {results['iou']:.4f}")
    print(f"  Dice:      {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"  Background IoU:  {results['iou_background']:.4f}")
    print(f"  Background Dice: {results['dice_background']:.4f}")
    print(f"  Cell IoU:        {results['iou_cell']:.4f}")
    print(f"  Cell Dice:       {results['dice_cell']:.4f}")

    print(f"\nAggregate Metrics:")
    print(f"  Mean IoU:  {results['miou']:.4f}")
    print(f"  Mean Dice: {results['mdice']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")

    # Save results
    results_path = args.output_dir / "metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {results_path}")

    # Save per-image results
    per_image_path = args.output_dir / "per_image_metrics.json"
    with open(per_image_path, 'w') as f:
        json.dump(per_image_results, f, indent=2)
    print(f"Per-image metrics saved to: {per_image_path}")

    # Find best and worst samples
    sorted_by_dice = sorted(per_image_results, key=lambda x: x['dice'])
    print(f"\nWorst 5 samples (by Cell Dice):")
    for item in sorted_by_dice[:5]:
        print(f"  {item['filename']}: Dice={item['dice']:.4f}, IoU={item['iou']:.4f}")

    print(f"\nBest 5 samples (by Cell Dice):")
    for item in sorted_by_dice[-5:][::-1]:
        print(f"  {item['filename']}: Dice={item['dice']:.4f}, IoU={item['iou']:.4f}")

    if args.visualize > 0:
        print(f"\nVisualizations saved to: {args.output_dir / 'visualizations'}")

    return results


if __name__ == "__main__":
    main()
