"""Test script to verify Layer 3 GAN detector is working."""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add layer3 to path
layer3_path = Path(__file__).parent / "layer3"
if layer3_path not in sys.path:
    sys.path.insert(0, str(layer3_path))

print("=" * 80)
print("Layer 3 GAN Detector Verification Test")
print("=" * 80)

# Test 1: Check if checkpoint exists
print("\n[TEST 1] Checking checkpoint file...")
checkpoint = Path(__file__).parent / "layer3" / "checkpoints" / "layer3_best.pth"
if checkpoint.exists():
    size_mb = checkpoint.stat().st_size / (1024 * 1024)
    print(f"✓ Checkpoint found: {size_mb:.2f} MB")
else:
    print(f"✗ Checkpoint NOT found at {checkpoint}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[TEST 2] Checking Python dependencies...")
deps = {
    "torch": "torch",
    "open_clip": "open-clip-torch",
    "PIL": "pillow",
}

missing = []
for name, package in deps.items():
    try:
        __import__(name)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")
        missing.append(package)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")

# Test 3: Import the trained inference module
print("\n[TEST 3] Testing layer3_trained_inference import...")
try:
    from layer3_trained_inference import load_trained_layer3
    print("✓ load_trained_layer3 imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load the trained detector
print("\n[TEST 4] Loading trained Layer 3 detector...")
try:
    detector = load_trained_layer3(
        checkpoint_path=str(checkpoint),
        device="cpu"
    )
    print("✓ Detector loaded successfully")
except Exception as e:
    print(f"✗ Failed to load detector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test analyze method
print("\n[TEST 5] Testing analyze method with synthetic image...")
try:
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    result = detector.analyze(dummy_image)
    
    print(f"✓ Analyze returned result")
    print(f"  - Fraud Probability: {result.get('fraud_probability'):.3f}")
    print(f"  - Decision: {result.get('decision')}")
    print(f"  - Confidence: {result.get('confidence'):.3f}")
    print(f"  - Uncertainty: {result.get('uncertainty'):.3f}")
    print(f"  - Flags: {result.get('flags')}")
    
    fraud_prob = result.get('fraud_probability', 0.5)
    # Verify score is not constantly 0.5
    if fraud_prob == 0.5 and 'inference_error' not in str(result.get('flags', [])):
        print("⚠ Warning: Fraud probability is exactly 0.5 (might indicate fallback)")
    else:
        print("✓ Fraud probability is computed (not constant 0.5)")
        
except Exception as e:
    print(f"✗ Analyze method failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
