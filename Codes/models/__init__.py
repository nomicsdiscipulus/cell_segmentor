"""
Model architecture module for cell segmentation.

Provides segmentation heads and full models for DinoBloom-based cell segmentation.

Usage:
    from models import CellSegmentor, get_segmentation_head

    # Full model (backbone + head)
    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True
    )
    logits = model(images)  # (B, 3, 224, 224) -> (B, 2, 224, 224)

    # Just segmentation head
    head = get_segmentation_head("linear", in_channels=384, num_classes=2)
"""

from .segmentation_head import (
    SegmentationHead,
    LinearDecoder,
    TransposedConvDecoder,
    get_segmentation_head,
    list_available_heads,
    count_parameters,
)

from .cell_segmentor import (
    CellSegmentor,
    create_cell_segmentor,
)

__all__ = [
    # Full model
    "CellSegmentor",
    "create_cell_segmentor",
    # Segmentation heads
    "SegmentationHead",
    "LinearDecoder",
    "TransposedConvDecoder",
    "get_segmentation_head",
    "list_available_heads",
    # Utilities
    "count_parameters",
]
