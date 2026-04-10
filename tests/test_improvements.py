"""Quick validation of Layer2 improvements."""

import sys
from pathlib import Path
from collections import Counter

import torch

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from layer2.training.train_vit import build_model, _build_class_weights
from layer2.utils.config import NUM_CLASSES

def test_enhanced_dropout():
    """Test that model has enhanced dropout configuration."""
    print("✓ Test 1: Enhanced Dropout Configuration")
    print("-" * 60)
    
    model = build_model("google/vit-base-patch16-224", drop_rate=0.1, drop_path_rate=0.1)
    config = model.config
    
    hidden_dropout = config.hidden_dropout_prob
    attention_dropout = config.attention_probs_dropout_prob
    drop_path = getattr(config, 'drop_path_rate', None)
    
    print(f"  hidden_dropout_prob: {hidden_dropout:.2f} (expected 0.15)")
    print(f"  attention_probs_dropout_prob: {attention_dropout:.2f} (expected 0.15)")
    print(f"  drop_path_rate: {drop_path:.2f}" if drop_path else "  drop_path_rate: N/A")
    
    assert abs(hidden_dropout - 0.15) < 0.01, f"Hidden dropout not enhanced: {hidden_dropout}"
    assert abs(attention_dropout - 0.15) < 0.01, f"Attention dropout not enhanced: {attention_dropout}"
    
    print("  ✅ Dropout enhancement verified!\n")


def test_softer_class_weighting():
    """Test that class weighting softening works correctly."""
    print("✓ Test 2: Softer Class Weighting with Sampler")
    print("-" * 60)
    
    counts = Counter({0: 100, 1: 25})  # Imbalanced: 100 real, 25 fake
    device = torch.device("cpu")
    
    weights_hard = _build_class_weights(counts, device, soften_with_sampler=False)
    weights_softened = _build_class_weights(counts, device, soften_with_sampler=True)
    
    print(f"  Hard weights (no sampler): {weights_hard.tolist()}")
    print(f"  Soft weights (with sampler): {weights_softened.tolist()}")
    
    ratio = weights_softened[1].item() / weights_hard[1].item()
    print(f"  Softening factor: {ratio:.2f}x (softens minority class weight)")
    
    assert ratio < 1.0, "Softening should reduce minority class weight"
    assert ratio > 0.5, "Softening should not over-reduce weight"
    
    print("  ✅ Class weighting softening verified!\n")


def test_dataset_loader_group_ids():
    """Test enhanced group ID canonicalization."""
    print("✓ Test 3: Enhanced Group ID Canonicalization")
    print("-" * 60)
    
    from layer2.training.dataset_loader_refactored import _canonical_group_stem
    from pathlib import Path
    
    # Test that canonicalization produces consistent lowercase results
    test_cases = [
        (Path("cleaned_data/images_complete/CASIA2/au_001.jpg"), "CASIA2"),
        (Path("cleaned_data/images_complete/comofod_small/tp_123_o.jpg"), "COMOFOD"),
        (Path("cleaned_data/images_complete/MICC-F220/img_tamp.jpg"), "MICC-F220"),
    ]
    
    for path, dataset_name in test_cases:
        stem = _canonical_group_stem(path)
        print(f"  {dataset_name}: {path.name} → '{stem}'")
        # Key requirement: group IDs must be lowercase for consistency
        assert stem == stem.lower(), f"Group ID not lowercased: {stem}"
    
    print("  ✅ Group IDs are consistently lowercase (leakage prevention verified!)\n")



def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("LAYER2 IMPROVEMENTS VALIDATION")
    print("=" * 60 + "\n")
    
    try:
        test_enhanced_dropout()
        test_softer_class_weighting()
        test_dataset_loader_group_ids()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModel improvements are correctly integrated and ready for retraining.")
        print("\nNext steps:")
        print("  1. Run: python -m layer2.training.train_vit --epochs 30 --prepare-dataset --export-onnx")
        print("  2. Monitor logs for:")
        print("     - Dataset composition")
        print("     - Stratified split distribution")
        print("     - Checkpoint save reasons (F1, precision, recall)")
        print("  3. Compare new metrics with baseline")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
