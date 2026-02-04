"""
DinoBloom model wrapper for cell segmentation.

DinoBloom is a foundation model for hematology cell embeddings, built on
DINOv2 and fine-tuned on 13 diverse blood cell datasets (MICCAI 2024).

This wrapper provides:
- Easy model loading with size selection (small/base/large/giant)
- Feature extraction utilities
- Configuration management

Reference:
    Koch et al., "DinoBloom: A Foundation Model for Generalizable Cell
    Embeddings in Hematology", MICCAI 2024.
"""

from pathlib import Path
from typing import Union, Literal

import torch
import torch.nn as nn

# =============================================================================
# Configuration
# =============================================================================

# Path to DinoBloom checkpoints (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # Codes/backbone -> Fd-To-Sg
_CHECKPOINT_DIR = _PROJECT_ROOT / "DinoBloom" / "checkpoints"

# Model configurations
DINOBLOOM_CONFIGS = {
    "small": {
        "checkpoint": "DinoBloom-S.pth",
        "arch": "dinov2_vits14",
        "embed_dim": 384,
        "params": "22M",
    },
    "base": {
        "checkpoint": "DinoBloom-B.pth",
        "arch": "dinov2_vitb14",
        "embed_dim": 768,
        "params": "86M",
    },
    "large": {
        "checkpoint": "DinoBloom-L.pth",
        "arch": "dinov2_vitl14",
        "embed_dim": 1024,
        "params": "304M",
    },
    "giant": {
        "checkpoint": "DinoBloom-G.pth",
        "arch": "dinov2_vitg14",
        "embed_dim": 1536,
        "params": "1.1B",
    },
}

# ImageNet normalization (required for DINOv2-based models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model input constraints
PATCH_SIZE = 14  # DinoBloom uses 14x14 patches
DEFAULT_IMG_SIZE = 224  # Default input size (must be divisible by PATCH_SIZE)

# =============================================================================
# Model Loading
# =============================================================================

def load_dinobloom(
    size: Literal["small", "base", "large", "giant"] = "small",
    checkpoint_path: Union[str, Path, None] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Load DinoBloom model with pre-trained weights.

    Args:
        size: Model size - "small" (22M), "base" (86M), "large" (304M), or "giant" (1.1B)
        checkpoint_path: Custom path to checkpoint file. If None, uses default.
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Loaded DinoBloom model

    Example:
        >>> model = load_dinobloom("small")
        >>> features = model.forward_features(images)
        >>> patch_tokens = features["x_norm_patchtokens"]  # (B, 256, 384)
    """
    if size not in DINOBLOOM_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(DINOBLOOM_CONFIGS.keys())}")

    config = DINOBLOOM_CONFIGS[size]

    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = _CHECKPOINT_DIR / config["checkpoint"]
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Download from: https://zenodo.org/records/10908163"
        )

    # Load base DINOv2 architecture from torch hub
    print(f"Loading DinoBloom-{size.upper()} ({config['params']} params)...")
    model = torch.hub.load('facebookresearch/dinov2', config["arch"])

    # Load fine-tuned DinoBloom weights
    pretrained = torch.load(checkpoint_path, map_location=device)

    # Extract backbone weights (remove training heads)
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or 'ibot_head' in key:
            continue
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value

    # Adjust positional embeddings for 224x224 input
    # 224/14 = 16 patches per side -> 16*16 = 256 patches + 1 CLS = 257 tokens
    num_tokens = (DEFAULT_IMG_SIZE // PATCH_SIZE) ** 2 + 1  # 257
    model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, config["embed_dim"]))

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Embed dim: {config['embed_dim']}")
    return model


def get_model_info(size: Literal["small", "base", "large", "giant"]) -> dict:
    """Get configuration info for a model size."""
    if size not in DINOBLOOM_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(DINOBLOOM_CONFIGS.keys())}")
    return DINOBLOOM_CONFIGS[size].copy()


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    return_cls: bool = False,
) -> torch.Tensor:
    """
    Extract features from images using DinoBloom.

    Args:
        model: Loaded DinoBloom model
        images: Input tensor of shape (B, 3, H, W), normalized with ImageNet stats
        return_cls: If True, return CLS token; otherwise return patch tokens

    Returns:
        If return_cls=False: Patch tokens of shape (B, num_patches, embed_dim)
        If return_cls=True: CLS token of shape (B, embed_dim), for image level representation

    Note:
        For 224x224 input: num_patches = 256 (16x16 grid)
        For 518x518 input: num_patches = 1369 (37x37 grid)
    """
    model.eval()
    with torch.no_grad():
        features_dict = model.forward_features(images)

        if return_cls:
            return features_dict['x_norm_clstoken']
        else:
            return features_dict['x_norm_patchtokens']


def get_patch_grid_size(img_size: int) -> int:
    """
    Calculate the number of patches per side for a given image size.

    Args:
        img_size: Input image size (must be divisible by 14)

    Returns:
        Number of patches per side (e.g., 16 for 224x224 input)
    """
    if img_size % PATCH_SIZE != 0:
        raise ValueError(f"Image size {img_size} must be divisible by patch size {PATCH_SIZE}")
    return img_size // PATCH_SIZE


# =============================================================================
# Utilities
# =============================================================================

def list_available_models() -> None:
    """Print available DinoBloom model configurations."""
    print("Available DinoBloom Models:")
    print("-" * 50)
    for name, config in DINOBLOOM_CONFIGS.items():
        checkpoint_path = _CHECKPOINT_DIR / config["checkpoint"]
        status = "Found" if checkpoint_path.exists() else "Not found (not downloaded)"
        print(f"  {name:8s} | {config['params']:>5s} params | {config['embed_dim']} dim | {status}")


if __name__ == "__main__":
    # Quick test
    list_available_models()