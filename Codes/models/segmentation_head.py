"""
Segmentation heads for DinoBloom-based cell segmentation.

This module provides modular segmentation heads that can be attached to
DinoBloom backbone for pixel-level predictions.

Available heads:
- LinearDecoder: Simple linear projection + bilinear upsample (baseline)
- TransposedConvDecoder: Learnable upsampling with transposed convolutions

Usage:
    from models.segmentation_head import get_segmentation_head

    head = get_segmentation_head("linear", in_channels=384, num_classes=2)
    head = get_segmentation_head("transposed_conv", in_channels=384, num_classes=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# =============================================================================
# Base Class
# =============================================================================

class SegmentationHead(nn.Module):
    """
    Base class for segmentation heads.

    All segmentation heads take DinoBloom patch tokens as input and
    produce pixel-level class predictions.

    Args:
        in_channels: Input feature dimension (DinoBloom embed_dim)
        num_classes: Number of output classes
        img_size: Output image size (default: 224)
        patch_size: DinoBloom patch size (default: 14)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 14
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 16 for 224/14

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Patch tokens from DinoBloom
                Shape: (B, num_patches, embed_dim) e.g., (B, 256, 384)

        Returns:
            logits: Pixel-level class predictions
                Shape: (B, num_classes, img_size, img_size)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def _reshape_to_spatial(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat patch tokens to spatial feature map.

        Args:
            features: (B, num_patches, C) e.g., (B, 256, 384)

        Returns:
            spatial: (B, C, H, W) e.g., (B, 384, 16, 16)
        """
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Cannot reshape {N} patches to square grid"
        return features.reshape(B, H, W, C).permute(0, 3, 1, 2)


# =============================================================================
# Linear Decoder (Baseline)
# =============================================================================

class LinearDecoder(SegmentationHead):
    """
    Simple linear projection + bilinear upsampling.

    This is the simplest learnable decoder:
    1. Linear projection: embed_dim -> num_classes
    2. Bilinear upsample: 16x16 -> 224x224

    Learnable parameters: Only the linear layer (~embed_dim * num_classes)

    Args:
        in_channels: Input feature dimension (DinoBloom embed_dim)
        num_classes: Number of output classes
        img_size: Output image size (default: 224)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 224,
        **kwargs
    ):
        super().__init__(in_channels, num_classes, img_size)

        # Single linear projection
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (B, 256, embed_dim) patch tokens

        Returns:
            logits: (B, num_classes, 224, 224)
        """
        B = features.shape[0]

        # Project to num_classes: (B, 256, embed_dim) -> (B, 256, num_classes)
        x = self.linear(features)

        # Reshape to spatial: (B, 256, num_classes) -> (B, num_classes, 16, 16)
        x = x.reshape(B, self.grid_size, self.grid_size, self.num_classes)
        x = x.permute(0, 3, 1, 2)  # (B, num_classes, 16, 16)

        # Bilinear upsample to output size
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        return x


# =============================================================================
# Transposed Convolution Decoder
# =============================================================================

class TransposedConvDecoder(SegmentationHead):
    """
    Learnable upsampling with transposed convolutions.

    Architecture:
    1. Project embed_dim to decoder channels
    2. Series of ConvTranspose2d + Conv + BN + ReLU blocks (upsample 2x each)
    3. Final conv to num_classes

    For 224x224 output from 16x16 input, needs ~4 upsample stages:
    16 -> 32 -> 64 -> 128 -> 256 (then crop or interpolate to 224)

    Args:
        in_channels: Input feature dimension (DinoBloom embed_dim)
        num_classes: Number of output classes
        img_size: Output image size (default: 224)
        decoder_channels: List of channel sizes for decoder stages
            Default: [256, 128, 64, 32] for 4 upsample stages
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: int = 224,
        decoder_channels: list = None,
        **kwargs
    ):
        super().__init__(in_channels, num_classes, img_size)

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        self.decoder_channels = decoder_channels

        # Initial projection from embed_dim to first decoder channel
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels[0], kernel_size=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Upsample blocks
        self.upsample_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]

        for out_ch in decoder_channels[1:]:
            block = nn.Sequential(
                # Transposed conv for 2x upsampling
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # Additional conv for refinement
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.upsample_blocks.append(block)
            in_ch = out_ch

        # Final classification head
        self.classifier = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (B, 256, embed_dim) patch tokens

        Returns:
            logits: (B, num_classes, 224, 224)
        """
        # Reshape to spatial: (B, 256, embed_dim) -> (B, embed_dim, 16, 16)
        x = self._reshape_to_spatial(features)

        # Initial projection
        x = self.input_proj(x)

        # Upsample stages
        # 16 -> 32 -> 64 -> 128 -> 256
        for block in self.upsample_blocks:
            x = block(x)

        # Final interpolation to exact output size (256 -> 224)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

        # Classification
        x = self.classifier(x)

        return x


# =============================================================================
# Factory Function
# =============================================================================

HEAD_REGISTRY = {
    "linear": LinearDecoder,
    "transposed_conv": TransposedConvDecoder,
}


def get_segmentation_head(
    name: Literal["linear", "transposed_conv"],
    in_channels: int,
    num_classes: int,
    img_size: int = 224,
    **kwargs
) -> SegmentationHead:
    """
    Factory function to create segmentation head by name.

    Args:
        name: Head type - "linear" or "transposed_conv"
        in_channels: Input feature dimension (DinoBloom embed_dim)
        num_classes: Number of output classes
        img_size: Output image size (default: 224)
        **kwargs: Additional arguments passed to head constructor

    Returns:
        SegmentationHead instance

    Example:
        >>> head = get_segmentation_head("linear", in_channels=384, num_classes=2)
        >>> head = get_segmentation_head("transposed_conv", in_channels=384, num_classes=2)
    """
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head '{name}'. Available: {list(HEAD_REGISTRY.keys())}"
        )

    return HEAD_REGISTRY[name](
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs
    )


def list_available_heads() -> list:
    """List available segmentation head types."""
    return list(HEAD_REGISTRY.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing segmentation heads...")
    print("-" * 50)

    # Test input (simulating DinoBloom output)
    batch_size = 2
    num_patches = 256  # 16x16
    embed_dim = 384    # DinoBloom-S
    num_classes = 2    # Binary segmentation

    dummy_features = torch.randn(batch_size, num_patches, embed_dim)

    for head_name in list_available_heads():
        print(f"\n{head_name}:")
        head = get_segmentation_head(head_name, in_channels=embed_dim, num_classes=num_classes)

        # Forward pass
        output = head(dummy_features)

        print(f"  Input:  {dummy_features.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Params: {count_parameters(head):,}")

        # Verify output shape
        assert output.shape == (batch_size, num_classes, 224, 224), \
            f"Wrong output shape: {output.shape}"
        print("  [PASS]")

    print("\n" + "-" * 50)
    print("All tests passed!")
