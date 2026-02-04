"""
Test script for models/segmentation_head.py module.

Tests:
1. LinearDecoder forward pass and output shape
2. TransposedConvDecoder forward pass and output shape
3. Factory function get_segmentation_head()
4. Parameter count verification
5. Gradient flow (backpropagation works)
6. Different input sizes (variable grid sizes)
7. Different num_classes configurations

Usage:
    python Codes/unitTest/04_model_architecture/01_test_segmentation_head.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import torch
import torch.nn as nn

from models.segmentation_head import (
    SegmentationHead,
    LinearDecoder,
    TransposedConvDecoder,
    get_segmentation_head,
    list_available_heads,
    count_parameters,
)

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


# =============================================================================
# Test Functions
# =============================================================================

def test_linear_decoder():
    """Test 1: LinearDecoder forward pass and output shape."""
    print("\n[Test 1] LinearDecoder forward pass...")

    # Config
    batch_size = 2
    num_patches = 256  # 16x16
    embed_dim = 384    # DinoBloom-S
    num_classes = 2
    img_size = 224

    # Create decoder
    decoder = LinearDecoder(
        in_channels=embed_dim,
        num_classes=num_classes,
        img_size=img_size
    )

    # Dummy input (simulating DinoBloom output)
    features = torch.randn(batch_size, num_patches, embed_dim)

    # Forward pass
    output = decoder(features)

    # Verify output shape
    expected_shape = (batch_size, num_classes, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Wrong output shape: {output.shape} != {expected_shape}"

    print(f"  Input shape:  {features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters:   {count_parameters(decoder):,}")
    print("  PASS")
    return True


def test_transposed_conv_decoder():
    """Test 2: TransposedConvDecoder forward pass and output shape."""
    print("\n[Test 2] TransposedConvDecoder forward pass...")

    # Config
    batch_size = 2
    num_patches = 256
    embed_dim = 384
    num_classes = 2
    img_size = 224

    # Create decoder
    decoder = TransposedConvDecoder(
        in_channels=embed_dim,
        num_classes=num_classes,
        img_size=img_size
    )

    # Dummy input
    features = torch.randn(batch_size, num_patches, embed_dim)

    # Forward pass
    output = decoder(features)

    # Verify output shape
    expected_shape = (batch_size, num_classes, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Wrong output shape: {output.shape} != {expected_shape}"

    print(f"  Input shape:  {features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters:   {count_parameters(decoder):,}")
    print("  PASS")
    return True


def test_factory_function():
    """Test 3: Factory function get_segmentation_head()."""
    print("\n[Test 3] Factory function get_segmentation_head()...")

    embed_dim = 384
    num_classes = 2

    # Test all available heads
    available = list_available_heads()
    print(f"  Available heads: {available}")

    for head_name in available:
        head = get_segmentation_head(
            head_name,
            in_channels=embed_dim,
            num_classes=num_classes
        )
        assert isinstance(head, SegmentationHead), \
            f"{head_name} is not a SegmentationHead instance"
        print(f"  Created {head_name}: {type(head).__name__}")

    # Test invalid name
    try:
        get_segmentation_head("invalid_head", in_channels=384, num_classes=2)
        assert False, "Should raise ValueError for invalid head name"
    except ValueError as e:
        print(f"  Invalid name correctly raises: {str(e)[:50]}...")

    print("  PASS")
    return True


def test_parameter_count():
    """Test 4: Parameter count verification."""
    print("\n[Test 4] Parameter count verification...")

    embed_dim = 384
    num_classes = 2

    # LinearDecoder: embed_dim * num_classes + num_classes (bias)
    linear = LinearDecoder(in_channels=embed_dim, num_classes=num_classes)
    linear_params = count_parameters(linear)
    expected_linear = embed_dim * num_classes + num_classes  # 384*2 + 2 = 770
    assert linear_params == expected_linear, \
        f"LinearDecoder params: {linear_params} != {expected_linear}"
    print(f"  LinearDecoder: {linear_params:,} params (expected {expected_linear})")

    # TransposedConvDecoder: Should have significantly more params
    transposed = TransposedConvDecoder(in_channels=embed_dim, num_classes=num_classes)
    transposed_params = count_parameters(transposed)
    assert transposed_params > 100000, \
        f"TransposedConvDecoder should have >100K params, got {transposed_params}"
    print(f"  TransposedConvDecoder: {transposed_params:,} params")

    print("  PASS")
    return True


def test_gradient_flow():
    """Test 5: Gradient flow (backpropagation works)."""
    print("\n[Test 5] Gradient flow...")

    embed_dim = 384
    num_classes = 2
    batch_size = 2
    num_patches = 256

    for head_name in list_available_heads():
        head = get_segmentation_head(head_name, in_channels=embed_dim, num_classes=num_classes)
        head.train()

        # Forward pass
        features = torch.randn(batch_size, num_patches, embed_dim, requires_grad=True)
        output = head(features)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None, f"No gradient for input in {head_name}"

        # Check model parameters have gradients
        has_grad = False
        for param in head.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, f"No gradient for parameters in {head_name}"

        print(f"  {head_name}: gradients flow correctly")

    print("  PASS")
    return True


def test_different_embed_dims():
    """Test 6: Different DinoBloom model sizes."""
    print("\n[Test 6] Different embed dimensions (DinoBloom sizes)...")

    batch_size = 2
    num_patches = 256
    num_classes = 2

    # DinoBloom model sizes
    embed_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
    }

    for model_name, embed_dim in embed_dims.items():
        features = torch.randn(batch_size, num_patches, embed_dim)

        for head_name in list_available_heads():
            head = get_segmentation_head(head_name, in_channels=embed_dim, num_classes=num_classes)
            output = head(features)

            assert output.shape == (batch_size, num_classes, 224, 224), \
                f"Wrong shape for {head_name} with embed_dim={embed_dim}"

        print(f"  DinoBloom-{model_name} (embed_dim={embed_dim}): OK")

    print("  PASS")
    return True


def test_different_num_classes():
    """Test 7: Different num_classes configurations."""
    print("\n[Test 7] Different num_classes configurations...")

    batch_size = 2
    num_patches = 256
    embed_dim = 384

    # Different segmentation scenarios
    class_configs = {
        "binary": 1,           # Single channel (sigmoid)
        "binary_2ch": 2,       # Two channels (softmax)
        "multi_class": 4,      # WBC, RBC, Platelet, Background
    }

    features = torch.randn(batch_size, num_patches, embed_dim)

    for config_name, num_classes in class_configs.items():
        for head_name in list_available_heads():
            head = get_segmentation_head(head_name, in_channels=embed_dim, num_classes=num_classes)
            output = head(features)

            expected_shape = (batch_size, num_classes, 224, 224)
            assert output.shape == expected_shape, \
                f"Wrong shape for {head_name} with num_classes={num_classes}"

        print(f"  {config_name} (num_classes={num_classes}): OK")

    print("  PASS")
    return True


def test_custom_decoder_channels():
    """Test 8: TransposedConvDecoder with custom channel configuration."""
    print("\n[Test 8] Custom decoder channels...")

    batch_size = 2
    num_patches = 256
    embed_dim = 384
    num_classes = 2

    features = torch.randn(batch_size, num_patches, embed_dim)

    # Test different channel configurations
    channel_configs = [
        [256, 128, 64, 32],      # Default
        [128, 64, 32, 16],       # Lighter
        [512, 256, 128, 64, 32], # Deeper (5 stages)
    ]

    for channels in channel_configs:
        decoder = TransposedConvDecoder(
            in_channels=embed_dim,
            num_classes=num_classes,
            decoder_channels=channels
        )
        output = decoder(features)

        assert output.shape == (batch_size, num_classes, 224, 224), \
            f"Wrong shape with channels={channels}"

        print(f"  channels={channels}: {count_parameters(decoder):,} params")

    print("  PASS")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing models/segmentation_head.py")
    print("=" * 60)

    results = []

    tests = [
        ("LinearDecoder forward", test_linear_decoder),
        ("TransposedConvDecoder forward", test_transposed_conv_decoder),
        ("Factory function", test_factory_function),
        ("Parameter count", test_parameter_count),
        ("Gradient flow", test_gradient_flow),
        ("Different embed dims", test_different_embed_dims),
        ("Different num_classes", test_different_num_classes),
        ("Custom decoder channels", test_custom_decoder_channels),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except AssertionError as e:
            results.append((test_name, f"FAIL: {e}"))
            print(f"  FAIL: {e}")
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for test_name, status in results:
        symbol = "+" if status == "PASS" else "-"
        print(f"  [{symbol}] {test_name}: {status}")

    print(f"\n  {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
