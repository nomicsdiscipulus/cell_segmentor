"""
Test script for models/cell_segmentor.py module.

Tests:
1. CellSegmentor initialization
2. Forward pass with dummy input
3. Forward pass with real BCCD image
4. Predict (softmax probabilities)
5. Predict mask (binary output)
6. Backbone freezing/unfreezing
7. Parameter counting
8. Different head types
9. Visualization of predictions

Usage:
    python Codes/unitTest/04_model_architecture/02_test_cell_segmentor.py
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).parent
CODES_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODES_DIR.parent
sys.path.insert(0, str(CODES_DIR))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import CellSegmentor, create_cell_segmentor
from backbone import IMAGENET_MEAN, IMAGENET_STD

# Paths
DATA_ROOT = PROJECT_ROOT / "Data" / "BCCD"
TEST_IMAGE_DIR = DATA_ROOT / "test" / "original"
TEST_MASK_DIR = DATA_ROOT / "test" / "mask"

# Output directory
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = SCRIPT_DIR / "outputs" / SCRIPT_NAME


# =============================================================================
# Test Functions
# =============================================================================

def test_initialization():
    """Test 1: CellSegmentor initialization."""
    print("\n[Test 1] CellSegmentor initialization...")

    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )

    assert model.backbone is not None, "Backbone should be loaded"
    assert model.head is not None, "Head should be created"
    assert model.embed_dim == 384, f"Wrong embed_dim: {model.embed_dim}"
    assert model.num_classes == 2, f"Wrong num_classes: {model.num_classes}"

    print(f"  Backbone: DinoBloom-{model.backbone_size}")
    print(f"  Head: {model.head_type}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Num classes: {model.num_classes}")
    print(f"  Backbone frozen: {model.freeze_backbone}")
    print("  PASS")
    return True


def test_forward_dummy():
    """Test 2: Forward pass with dummy input."""
    print("\n[Test 2] Forward pass with dummy input...")

    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )
    model.eval()

    # Dummy input (normalized)
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        logits = model(images)

    # Verify output shape
    expected_shape = (batch_size, 2, 224, 224)
    assert logits.shape == expected_shape, \
        f"Wrong output shape: {logits.shape} != {expected_shape}"

    print(f"  Input shape:  {images.shape}")
    print(f"  Output shape: {logits.shape}")
    print("  PASS")
    return True


def test_forward_real_image():
    """Test 3: Forward pass with real BCCD image."""
    print("\n[Test 3] Forward pass with real BCCD image...")

    # Find a test image
    test_images = list(TEST_IMAGE_DIR.glob("*.png")) + list(TEST_IMAGE_DIR.glob("*.jpg"))
    if not test_images:
        print("  SKIP: No test images found")
        return True

    img_path = test_images[0]
    print(f"  Image: {img_path.name}")

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    print(f"  Original size: {original_size}")

    # Resize to 224x224 and normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    # Create model and run forward pass
    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)

    # Verify output
    assert logits.shape == (1, 2, 224, 224), f"Wrong shape: {logits.shape}"

    print(f"  Input shape:  {img_tensor.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print("  PASS")
    return True


def test_predict():
    """Test 4: Predict (softmax probabilities)."""
    print("\n[Test 4] Predict (softmax probabilities)...")

    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        probs = model.predict(images)

    # Verify probabilities sum to 1
    prob_sum = probs.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "Probabilities should sum to 1"

    # Verify range [0, 1]
    assert probs.min() >= 0 and probs.max() <= 1, \
        f"Probabilities out of range: [{probs.min()}, {probs.max()}]"

    print(f"  Output shape: {probs.shape}")
    print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Sum check: {prob_sum[0, 0, 0].item():.6f} (should be 1.0)")
    print("  PASS")
    return True


def test_predict_mask():
    """Test 5: Predict mask (binary output)."""
    print("\n[Test 5] Predict mask (binary output)...")

    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        masks = model.predict_mask(images, threshold=0.5)

    # Verify output shape (no channel dim)
    assert masks.shape == (2, 224, 224), f"Wrong shape: {masks.shape}"

    # Verify binary values
    unique_values = torch.unique(masks)
    assert all(v in [0, 1] for v in unique_values), \
        f"Mask should be binary, got values: {unique_values}"

    print(f"  Output shape: {masks.shape}")
    print(f"  Unique values: {unique_values.tolist()}")
    print(f"  Cell pixels: {(masks == 1).sum().item()}")
    print("  PASS")
    return True


def test_backbone_freezing():
    """Test 6: Backbone freezing/unfreezing."""
    print("\n[Test 6] Backbone freezing/unfreezing...")

    # Create model with frozen backbone
    model = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )

    # Check backbone is frozen
    backbone_trainable = sum(p.requires_grad for p in model.backbone.parameters())
    assert backbone_trainable == 0, "Backbone should be frozen"
    print(f"  Frozen: backbone trainable params = {backbone_trainable}")

    # Head should still be trainable
    head_trainable = sum(p.requires_grad for p in model.head.parameters())
    assert head_trainable > 0, "Head should be trainable"
    print(f"  Head trainable params = {head_trainable}")

    # Unfreeze backbone
    model._unfreeze_backbone()
    backbone_trainable_after = sum(p.requires_grad for p in model.backbone.parameters())
    assert backbone_trainable_after > 0, "Backbone should be unfrozen"
    print(f"  Unfrozen: backbone trainable params = {backbone_trainable_after}")

    print("  PASS")
    return True


def test_parameter_counting():
    """Test 7: Parameter counting."""
    print("\n[Test 7] Parameter counting...")

    # With frozen backbone
    model_frozen = CellSegmentor(
        backbone_size="small",
        head_type="linear",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )

    params_frozen = model_frozen.count_parameters(trainable_only=True)
    print(f"  Frozen backbone:")
    print(f"    Backbone: {params_frozen['backbone']:,}")
    print(f"    Head: {params_frozen['head']:,}")
    print(f"    Total trainable: {params_frozen['total']:,}")

    assert params_frozen['backbone'] == 0, "Frozen backbone should have 0 trainable params"
    assert params_frozen['head'] > 0, "Head should have trainable params"

    # Total params (including frozen)
    params_all = model_frozen.count_parameters(trainable_only=False)
    print(f"  Total params (including frozen):")
    print(f"    Backbone: {params_all['backbone']:,}")
    print(f"    Head: {params_all['head']:,}")
    print(f"    Total: {params_all['total']:,}")

    print("  PASS")
    return True


def test_different_heads():
    """Test 8: Different head types."""
    print("\n[Test 8] Different head types...")

    images = torch.randn(1, 3, 224, 224)
    head_types = ["linear", "transposed_conv"]

    for head_type in head_types:
        model = CellSegmentor(
            backbone_size="small",
            head_type=head_type,
            num_classes=2,
            freeze_backbone=True,
            device="cpu"
        )
        model.eval()

        with torch.no_grad():
            logits = model(images)

        assert logits.shape == (1, 2, 224, 224), \
            f"Wrong shape for {head_type}: {logits.shape}"

        params = model.count_parameters(trainable_only=True)
        print(f"  {head_type}: output shape {logits.shape}, trainable params {params['total']:,}")

    print("  PASS")
    return True


def test_visualization():
    """Test 9: Visualization of predictions."""
    print("\n[Test 9] Visualization of predictions...")

    import matplotlib.pyplot as plt

    # Find a test image
    test_images = list(TEST_IMAGE_DIR.glob("*.png")) + list(TEST_IMAGE_DIR.glob("*.jpg"))
    if not test_images:
        print("  SKIP: No test images found")
        return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = test_images[0]
    mask_path = TEST_MASK_DIR / f"{img_path.stem}.png"

    # Load image and mask
    img = Image.open(img_path).convert('RGB')
    mask_gt = Image.open(mask_path).convert('L') if mask_path.exists() else None

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Create model and predict
    model = CellSegmentor(
        backbone_size="small",
        head_type="transposed_conv",
        num_classes=2,
        freeze_backbone=True,
        device="cpu"
    )
    model.eval()

    with torch.no_grad():
        probs = model.predict(img_tensor)
        mask_pred = model.predict_mask(img_tensor)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Original image (resized)
    img_resized = img.resize((224, 224))
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("Input Image (224x224)")
    axes[0, 0].axis('off')

    # Ground truth mask (if available)
    if mask_gt:
        mask_gt_resized = mask_gt.resize((224, 224))
        axes[0, 1].imshow(mask_gt_resized, cmap='gray')
        axes[0, 1].set_title("Ground Truth Mask")
    else:
        axes[0, 1].text(0.5, 0.5, "No GT mask", ha='center', va='center')
        axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis('off')

    # Predicted mask
    axes[0, 2].imshow(mask_pred[0].numpy(), cmap='gray')
    axes[0, 2].set_title("Predicted Mask")
    axes[0, 2].axis('off')

    # Background probability
    axes[1, 0].imshow(probs[0, 0].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title("P(Background)")
    axes[1, 0].axis('off')

    # Cell probability
    axes[1, 1].imshow(probs[0, 1].numpy(), cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title("P(Cell)")
    axes[1, 1].axis('off')

    # Overlay
    img_np = np.array(img_resized) / 255.0
    mask_np = mask_pred[0].numpy()
    overlay = img_np.copy()
    overlay[mask_np == 1] = overlay[mask_np == 1] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay (red=predicted)")
    axes[1, 2].axis('off')

    plt.suptitle(f"CellSegmentor Prediction\n{img_path.name}", fontsize=12)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "cell_segmentor_prediction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    print("  PASS")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing models/cell_segmentor.py")
    print("=" * 60)

    results = []

    tests = [
        ("Initialization", test_initialization),
        ("Forward (dummy)", test_forward_dummy),
        ("Forward (real image)", test_forward_real_image),
        ("Predict (probs)", test_predict),
        ("Predict mask", test_predict_mask),
        ("Backbone freezing", test_backbone_freezing),
        ("Parameter counting", test_parameter_counting),
        ("Different heads", test_different_heads),
        ("Visualization", test_visualization),
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
            import traceback
            traceback.print_exc()

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

    if OUTPUT_DIR.exists():
        print(f"  Outputs: {OUTPUT_DIR}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
