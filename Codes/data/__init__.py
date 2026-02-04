"""
Data pipeline for BCCD cell segmentation.

Supports two inference approaches:
1. Padding: Process entire image at once (pad to multiple of 14)
2. Tiling: Extract overlapping tiles with Gaussian blending

Usage:
    from data import create_datasets, create_test_dataset, get_collate_fn

    # Create train/val datasets
    train_ds, val_ds = create_datasets(
        data_root=Path("Data/BCCD"),
        split_path=Path("Data/BCCD/splits.json"),
        mode="tiling"  # or "padding"
    )

    # Create dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        collate_fn=get_collate_fn(mode="tiling", train=True)
    )
"""

from .dataset import (
    BCCDDataset,
    create_datasets,
    create_test_dataset,
    padding_collate_fn,
    tiling_train_collate_fn,
    tiling_inference_collate_fn,
    get_collate_fn
)

from .splits import (
    create_train_val_split,
    load_split
)

from .tiling import (
    pad_to_multiple,
    unpad,
    calculate_tile_positions,
    extract_tiles,
    random_crop,
    create_gaussian_weight_map,
    stitch_tiles,
    resize_mask_to_patches,
    extract_tile_masks,
    PATCH_SIZE,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_GAUSSIAN_SIGMA
)

from .transforms import (
    get_transform,
    get_normalize_transform,
    get_denormalize_transform,
    PaddingTransform,
    TileTransform,
    RandomCropTransform,
    IMAGENET_MEAN,
    IMAGENET_STD
)

__all__ = [
    # Dataset
    "BCCDDataset",
    "create_datasets",
    "create_test_dataset",
    "padding_collate_fn",
    "tiling_train_collate_fn",
    "tiling_inference_collate_fn",
    "get_collate_fn",
    # Splits
    "create_train_val_split",
    "load_split",
    # Tiling
    "pad_to_multiple",
    "unpad",
    "calculate_tile_positions",
    "extract_tiles",
    "random_crop",
    "create_gaussian_weight_map",
    "stitch_tiles",
    "resize_mask_to_patches",
    "extract_tile_masks",
    "PATCH_SIZE",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_OVERLAP",
    "DEFAULT_GAUSSIAN_SIGMA",
    # Transforms
    "get_transform",
    "get_normalize_transform",
    "get_denormalize_transform",
    "PaddingTransform",
    "TileTransform",
    "RandomCropTransform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
