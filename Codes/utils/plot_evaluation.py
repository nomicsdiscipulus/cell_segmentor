"""
Visualization script for evaluation metrics.

Creates:
1. Bar chart of aggregate metrics
2. Histogram of per-image Dice scores
3. Scatter plot of IoU vs Dice
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_aggregate_metrics(metrics: dict, output_path: Path):
    """Bar chart of aggregate evaluation metrics."""
    # Select key metrics
    metric_names = ['Cell IoU', 'Cell Dice', 'Precision', 'Recall', 'Accuracy']
    metric_keys = ['iou_cell', 'dice_cell', 'precision', 'recall', 'accuracy']
    values = [metrics[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cell Segmentation Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'aggregate_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'aggregate_metrics.png'}")


def plot_dice_distribution(per_image_metrics: list, output_path: Path):
    """Histogram of per-image Dice scores."""
    dice_scores = [m['dice'] for m in per_image_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins, patches = ax.hist(dice_scores, bins=20, color='#3498db',
                                edgecolor='black', alpha=0.7)

    # Color bars by value
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0.9:
            patch.set_facecolor('#e74c3c')
        elif left_edge < 0.95:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#2ecc71')

    # Add statistics
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    ax.axvline(mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    ax.axvline(mean_dice - std_dice, color='red', linestyle=':', alpha=0.5)
    ax.axvline(mean_dice + std_dice, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Dice Score', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title(f'Distribution of Per-Image Dice Scores (n={len(dice_scores)})',
                 fontsize=14, fontweight='bold')
    ax.legend()

    # Add text box with stats
    stats_text = f'Mean: {mean_dice:.4f}\nStd: {std_dice:.4f}\nMin: {min(dice_scores):.4f}\nMax: {max(dice_scores):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / 'dice_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'dice_distribution.png'}")


def plot_iou_vs_dice(per_image_metrics: list, output_path: Path):
    """Scatter plot of IoU vs Dice scores."""
    iou_scores = [m['iou'] for m in per_image_metrics]
    dice_scores = [m['dice'] for m in per_image_metrics]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(iou_scores, dice_scores, alpha=0.6, s=50, c='#3498db', edgecolor='black')

    # Add diagonal reference line
    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', alpha=0.3, label='y=x')

    ax.set_xlabel('IoU Score', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('IoU vs Dice Score per Image', fontsize=14, fontweight='bold')
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.7, 1.0)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'iou_vs_dice.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'iou_vs_dice.png'}")


def plot_class_comparison(metrics: dict, output_path: Path):
    """Compare background vs cell class metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['IoU', 'Dice', 'Precision', 'Recall']
    bg_values = [metrics['iou_background'], metrics['dice_background'],
                 metrics['precision_background'], metrics['recall_background']]
    cell_values = [metrics['iou_cell'], metrics['dice_cell'],
                   metrics['precision_cell'], metrics['recall_cell']]

    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, bg_values, width, label='Background', color='#95a5a6', edgecolor='black')
    bars2 = ax.bar(x + width/2, cell_values, width, label='Cell', color='#e74c3c', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics by Class: Background vs Cell', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / 'class_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'class_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation metrics")
    parser.add_argument(
        "--experiment",
        type=str,
        default="tiling_transposed_conv_20260203_203221",
        help="Experiment folder name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: evaluation folder)"
    )
    args = parser.parse_args()

    # Find paths
    script_dir = Path(__file__).parent
    codes_dir = script_dir.parent
    experiment_dir = codes_dir / "outputs" / "training" / args.experiment
    eval_dir = experiment_dir / "evaluation"

    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = eval_dir / "metric_plots"

    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics_file = eval_dir / "metrics.json"
    per_image_file = eval_dir / "per_image_metrics.json"

    if not metrics_file.exists():
        print(f"Error: metrics.json not found at {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print(f"Loaded aggregate metrics from {metrics_file}")

    # Plot aggregate metrics
    plot_aggregate_metrics(metrics, output_path)
    plot_class_comparison(metrics, output_path)

    # Plot per-image metrics if available
    if per_image_file.exists():
        with open(per_image_file, 'r') as f:
            per_image_metrics = json.load(f)
        print(f"Loaded per-image metrics ({len(per_image_metrics)} images)")

        plot_dice_distribution(per_image_metrics, output_path)
        plot_iou_vs_dice(per_image_metrics, output_path)
    else:
        print(f"Note: per_image_metrics.json not found, skipping distribution plots")

    print(f"\nAll plots saved to: {output_path}")


if __name__ == "__main__":
    main()
