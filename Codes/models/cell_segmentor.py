"""
Full cell segmentation model combining DinoBloom backbone with segmentation head.

This module provides the complete model for cell segmentation:
- DinoBloom backbone for feature extraction (frozen or fine-tunable)
- Segmentation head for pixel-level predictions

Usage:
    from models import CellSegmentor

    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True
    )

    # Forward pass
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)  # (2, 2, 224, 224)

    # Get predictions
    probs = model.predict(images)  # Softmax probabilities
    masks = model.predict_mask(images)  # Binary masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Any

from backbone import load_dinobloom, extract_features, DINOBLOOM_CONFIGS
from .segmentation_head import get_segmentation_head, SegmentationHead, count_parameters


class CellSegmentor(nn.Module):
    """
    Full cell segmentation model.

    Combines DinoBloom backbone with a segmentation head for
    end-to-end cell segmentation.

    Args:
        backbone_size: DinoBloom model size ("small", "base", "large")
        head_type: Segmentation head type ("linear", "transposed_conv")
        num_classes: Number of output classes (default: 2 for binary)
        freeze_backbone: If True, freeze backbone weights (default: True)
        img_size: Input image size (default: 224)
        pretrained: If True, load pretrained DinoBloom weights (default: True)
        device: Device to load model on (default: "cpu")
        head_kwargs: Additional kwargs for segmentation head
    """

    def __init__(
        self,
        backbone_size: Literal["small", "base", "large"] = "small",
        head_type: Literal["linear", "transposed_conv"] = "linear",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        img_size: int = 224,
        pretrained: bool = True,
        device: str = "cpu",
        **head_kwargs
    ):
        super().__init__()

        self.backbone_size = backbone_size
        self.head_type = head_type
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.img_size = img_size
        self.device = device

        # Get backbone config
        self.backbone_config = DINOBLOOM_CONFIGS[backbone_size]
        self.embed_dim = self.backbone_config["embed_dim"]

        # Load backbone
        if pretrained:
            self.backbone = load_dinobloom(size=backbone_size, device=device)
        else:
            # For testing without pretrained weights
            self.backbone = None

        # Freeze backbone if requested
        if self.backbone is not None and freeze_backbone:
            self._freeze_backbone()

        # Create segmentation head
        self.head = get_segmentation_head(
            name=head_type,
            in_channels=self.embed_dim,
            num_classes=num_classes,
            img_size=img_size,
            **head_kwargs
        )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def train(self, mode: bool = True):
        """
        Set training mode.

        Note: If backbone is frozen, it stays in eval mode.
        """
        super().train(mode)
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images, shape (B, 3, H, W)
                Should be normalized with ImageNet mean/std

        Returns:
            logits: Raw predictions, shape (B, num_classes, H, W)
        """
        # Extract features from backbone
        # features shape: (B, num_patches, embed_dim) e.g., (B, 256, 384)
        features = extract_features(self.backbone, images, return_cls=False)

        # Pass through segmentation head
        # logits shape: (B, num_classes, H, W)
        logits = self.head(features)

        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get softmax probabilities.

        Args:
            images: Input images, shape (B, 3, H, W)

        Returns:
            probs: Softmax probabilities, shape (B, num_classes, H, W)
        """
        logits = self.forward(images)
        return F.softmax(logits, dim=1)

    def predict_mask(
        self,
        images: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary segmentation masks.

        Args:
            images: Input images, shape (B, 3, H, W)
            threshold: Probability threshold for positive class

        Returns:
            masks: Binary masks, shape (B, H, W)
                1 = cell, 0 = background
        """
        probs = self.predict(images)

        if self.num_classes == 2:
            # Use class 1 (cell) probability
            masks = (probs[:, 1] > threshold).long()
        else:
            # For multi-class, use argmax
            masks = probs.argmax(dim=1)

        return masks

    def get_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features without passing through head.

        Useful for visualization and debugging.

        Args:
            images: Input images, shape (B, 3, H, W)

        Returns:
            features: Patch tokens, shape (B, num_patches, embed_dim)
        """
        return extract_features(self.backbone, images, return_cls=False)

    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Dict with parameter counts for backbone, head, and total
        """
        def count(module, trainable):
            if trainable:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        backbone_params = count(self.backbone, trainable_only) if self.backbone else 0
        head_params = count(self.head, trainable_only)

        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params
        }

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "backbone_size": self.backbone_size,
            "head_type": self.head_type,
            "num_classes": self.num_classes,
            "freeze_backbone": self.freeze_backbone,
            "img_size": self.img_size,
            "embed_dim": self.embed_dim,
        }

    def __repr__(self) -> str:
        params = self.count_parameters(trainable_only=True)
        return (
            f"CellSegmentor(\n"
            f"  backbone: DinoBloom-{self.backbone_size} (embed_dim={self.embed_dim}, frozen={self.freeze_backbone})\n"
            f"  head: {self.head_type}\n"
            f"  num_classes: {self.num_classes}\n"
            f"  trainable_params: {params['total']:,}\n"
            f")"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_cell_segmentor(
    backbone_size: str = "small",
    head_type: str = "linear",
    num_classes: int = 2,
    freeze_backbone: bool = True,
    device: str = "cpu",
    **kwargs
) -> CellSegmentor:
    """
    Factory function to create CellSegmentor.

    Args:
        backbone_size: "small", "base", or "large"
        head_type: "linear" or "transposed_conv"
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone
        device: Device to load model on
        **kwargs: Additional arguments

    Returns:
        CellSegmentor instance
    """
    return CellSegmentor(
        backbone_size=backbone_size,
        head_type=head_type,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=device,
        **kwargs
    )
