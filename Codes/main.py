"""
Main training script for BCCD cell segmentation.

This script trains a CellSegmentor model on the BCCD dataset using
DinoBloom as the backbone and a configurable segmentation head.

Usage:
    python Codes/main.py --mode tiling --head linear --epochs 50
    python Codes/main.py --mode padding --head transposed_conv --epochs 100

Arguments:
    --mode: Data mode ("padding" or "tiling")
    --head: Segmentation head type ("linear" or "transposed_conv")
    --backbone: DinoBloom size ("small", "base", "large")
    --epochs: Number of training epochs
    --batch_size: Batch size for training
    --lr: Learning rate
    --device: Device to train on ("cuda" or "cpu")
    --output_dir: Directory to save outputs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Path setup
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader

# Local imports
from data import (
    create_datasets,
    create_train_val_split,
    get_collate_fn,
)
from models import CellSegmentor
from training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    get_loss_function,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train cell segmentation model on BCCD dataset"
    )

    # Data arguments
    parser.add_argument(
        "--mode", type=str, default="tiling",
        choices=["padding", "tiling"],
        help="Data mode: padding (full image) or tiling (random crops)"
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to BCCD data root (default: Data/BCCD)"
    )

    # Model arguments
    parser.add_argument(
        "--backbone", type=str, default="small",
        choices=["small", "base", "large"],
        help="DinoBloom backbone size"
    )
    parser.add_argument(
        "--head", type=str, default="linear",
        choices=["linear", "transposed_conv"],
        help="Segmentation head type"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", default=True,
        help="Freeze DinoBloom backbone weights"
    )
    parser.add_argument(
        "--unfreeze_backbone", action="store_true",
        help="Unfreeze DinoBloom backbone (fine-tune)"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--loss", type=str, default="bce_dice",
        choices=["ce", "dice", "focal", "bce_dice"],
        help="Loss function"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["none", "cosine", "step", "plateau"],
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=10,
        help="Early stopping patience (0 to disable)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Experiment name (default: auto-generated)"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on (default: cuda if available)"
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="Log every N batches (0 to disable)"
    )

    args = parser.parse_args()

    # Handle freeze/unfreeze
    if args.unfreeze_backbone:
        args.freeze_backbone = False

    return args


def setup_paths(args):
    """Setup data and output paths."""
    # Data root
    if args.data_root is None:
        args.data_root = PROJECT_ROOT / "Data" / "BCCD"
    else:
        args.data_root = Path(args.data_root)

    # Split path
    args.split_path = args.data_root / "splits.json"

    # Output directory
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "outputs" / "training"

    args.output_dir = Path(args.output_dir)

    # Experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.mode}_{args.head}_{timestamp}"

    args.checkpoint_dir = args.output_dir / args.experiment_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    # Ensure split exists
    if not args.split_path.exists():
        print(f"Creating train/val split at {args.split_path}...")
        create_train_val_split(
            image_dir=args.data_root / "train" / "original",
            output_path=args.split_path,
            val_ratio=0.2,
            seed=args.seed
        )

    # Create datasets
    print(f"Loading datasets (mode={args.mode})...")
    train_dataset, val_dataset = create_datasets(
        data_root=args.data_root,
        split_path=args.split_path,
        mode=args.mode
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=get_collate_fn(args.mode, train=True),
        pin_memory=torch.cuda.is_available()
    )

    # Note: For validation during training, we use train=True collate
    # to get random crops (same as training). This gives a quick estimate.
    # Full tiling inference (all tiles + stitching) is for final evaluation.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=get_collate_fn(args.mode, train=True),
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def create_model(args):
    """Create the segmentation model."""
    print(f"Creating model...")
    print(f"  Backbone: DinoBloom-{args.backbone}")
    print(f"  Head: {args.head}")
    print(f"  Freeze backbone: {args.freeze_backbone}")

    model = CellSegmentor(
        backbone_size=args.backbone,
        head_type=args.head,
        num_classes=2,
        freeze_backbone=args.freeze_backbone,
        device=args.device
    )

    # Print parameter counts
    params = model.count_parameters(trainable_only=True)
    print(f"  Trainable parameters: {params['total']:,}")
    print(f"    Backbone: {params['backbone']:,}")
    print(f"    Head: {params['head']:,}")

    return model


def save_config(args):
    """Save training configuration."""
    import json

    config = vars(args).copy()
    # Convert Path objects to strings
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)

    config_path = args.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {config_path}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup paths
    args = setup_paths(args)

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("BCCD Cell Segmentation Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Output: {args.checkpoint_dir}")
    print()

    # Set seed
    set_seed(args.seed)

    # Save config
    save_config(args)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    model = create_model(args)

    # Create optimizer
    print(f"\nOptimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    optimizer = create_optimizer(
        model,
        name="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    print(f"Scheduler: {args.scheduler}")
    scheduler = create_scheduler(
        optimizer,
        name=args.scheduler,
        num_epochs=args.epochs
    )

    # Create loss function
    print(f"Loss: {args.loss}")
    loss_fn = get_loss_function(args.loss)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        num_classes=2,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.early_stopping}")
    print()

    history = trainer.fit(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        save_last=True
    )

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints: {args.checkpoint_dir}")

    return history


if __name__ == "__main__":
    main()
