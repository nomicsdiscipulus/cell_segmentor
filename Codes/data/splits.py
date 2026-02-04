"""
Train/validation split utilities for BCCD dataset.

Creates reproducible 80/20 split and saves to JSON for consistency.
"""

import json
import random
from pathlib import Path
from typing import Tuple, List


def create_train_val_split(
    image_dir: Path,
    output_path: Path,
    val_ratio: float = 0.2,
    seed: int = 2026
) -> Tuple[List[str], List[str]]:
    """
    Create train/val split from image directory.

    Args:
        image_dir: Directory containing training images
        output_path: Path to save split JSON
        val_ratio: Fraction for validation (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files) lists
    """
    # Get all image files
    image_files = sorted([f.name for f in image_dir.glob("*.png")] +
                         [f.name for f in image_dir.glob("*.jpg")])

    # Shuffle with seed
    random.seed(seed)
    shuffled = image_files.copy()
    random.shuffle(shuffled)

    # Split
    val_size = int(len(shuffled) * val_ratio)
    val_files = shuffled[:val_size]
    train_files = shuffled[val_size:]

    # Save to JSON
    split_data = {
        "seed": seed,
        "val_ratio": val_ratio,
        "total": len(image_files),
        "train": train_files,
        "val": val_files
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)

    print(f"Split created: {len(train_files)} train, {len(val_files)} val")
    print(f"Saved to: {output_path}")

    return train_files, val_files


def load_split(split_path: Path) -> Tuple[List[str], List[str]]:
    """Load existing split from JSON."""
    with open(split_path, 'r') as f:
        data = json.load(f)
    return data["train"], data["val"]
