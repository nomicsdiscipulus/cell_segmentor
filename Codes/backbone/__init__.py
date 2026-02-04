"""
Backbone module for DinoBloom foundation model.

This module provides wrapper functions to load and use the DinoBloom model
for cell segmentation tasks. DinoBloom is a pathology-specific foundation
model based on DINOv2, fine-tuned on hematology cell images.

Usage:
    from backbone import load_dinobloom, DINOBLOOM_CONFIGS

    model = load_dinobloom("small")  # or "base", "large", "Giant"
    features = model.forward_features(images)
"""

from .dinobloom import (
    load_dinobloom,
    extract_features,
    DINOBLOOM_CONFIGS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "load_dinobloom",
    "extract_features",
    "DINOBLOOM_CONFIGS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
