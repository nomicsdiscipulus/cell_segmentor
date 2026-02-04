"""
Plot training history from history.json.

Usage:
    python Codes/utils/plot_history.py --history outputs/training/exp/history.json
    python Codes/utils/plot_history.py --history outputs/training/exp/history.json --metrics loss dice

Arguments:
    --history: Path to history.json
    --metrics: Which metrics to plot (loss, dice, iou, lr, all)
    --output: Output path for figure (default: same folder as history.json)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history")

    parser.add_argument(
        "--history", type=str, required=True,
        help="Path to history.json"
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=["loss", "dice"],
        choices=["loss", "dice", "iou", "lr", "all"],
        help="Metrics to plot (default: loss dice)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for figure (default: training_curves.png in same folder)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively"
    )

    return parser.parse_args()


def load_history(path: Path) -> dict:
    """Load history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_metric(ax, history, train_key, val_key, ylabel, title):
    """Plot a single metric with train and val curves."""
    epochs = range(1, len(history[train_key]) + 1)

    ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2)
    if val_key in history and len(history[val_key]) > 0:
        ax.plot(epochs, history[val_key], 'r-', label='Val', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch (lowest val_loss or highest val_dice/iou)
    if val_key in history and len(history[val_key]) > 0:
        if 'loss' in val_key:
            best_idx = np.argmin(history[val_key])
            best_val = history[val_key][best_idx]
        else:
            best_idx = np.argmax(history[val_key])
            best_val = history[val_key][best_idx]
        ax.axvline(x=best_idx + 1, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_idx + 1}')
        ax.scatter([best_idx + 1], [best_val], color='g', s=100, zorder=5)


def plot_lr(ax, history):
    """Plot learning rate schedule."""
    epochs = range(1, len(history['lr']) + 1)

    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')


def create_figure(history, metrics):
    """Create figure with specified metrics."""
    # Determine which plots to show
    if "all" in metrics:
        metrics = ["loss", "dice", "iou", "lr"]

    n_plots = len(metrics)

    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    elif n_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

    plot_idx = 0

    if "loss" in metrics:
        plot_metric(axes[plot_idx], history, 'train_loss', 'val_loss', 'Loss', 'Training & Validation Loss')
        plot_idx += 1

    if "dice" in metrics:
        plot_metric(axes[plot_idx], history, 'train_dice', 'val_dice', 'Dice', 'Training & Validation Dice')
        plot_idx += 1

    if "iou" in metrics:
        plot_metric(axes[plot_idx], history, 'train_iou', 'val_iou', 'IoU', 'Training & Validation IoU')
        plot_idx += 1

    if "lr" in metrics:
        plot_lr(axes[plot_idx], history)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def print_summary(history):
    """Print training summary."""
    n_epochs = len(history['train_loss'])

    print(f"\nTraining Summary ({n_epochs} epochs)")
    print("=" * 40)

    # Best epoch by val_loss
    if 'val_loss' in history and len(history['val_loss']) > 0:
        best_loss_idx = np.argmin(history['val_loss'])
        print(f"Best val_loss: {history['val_loss'][best_loss_idx]:.4f} (epoch {best_loss_idx + 1})")

    # Best epoch by val_dice
    if 'val_dice' in history and len(history['val_dice']) > 0:
        best_dice_idx = np.argmax(history['val_dice'])
        print(f"Best val_dice: {history['val_dice'][best_dice_idx]:.4f} (epoch {best_dice_idx + 1})")

    # Final metrics
    print(f"\nFinal metrics (epoch {n_epochs}):")
    print(f"  Train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train dice: {history['train_dice'][-1]:.4f}")
    if 'val_loss' in history and len(history['val_loss']) > 0:
        print(f"  Val loss:   {history['val_loss'][-1]:.4f}")
        print(f"  Val dice:   {history['val_dice'][-1]:.4f}")


def main():
    args = parse_args()

    # Load history
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        sys.exit(1)

    print(f"Loading: {history_path}")
    history = load_history(history_path)

    # Print summary
    print_summary(history)

    # Create figure
    fig = create_figure(history, args.metrics)

    # Save figure
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = history_path.parent / "training_curves.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")

    # Show if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
