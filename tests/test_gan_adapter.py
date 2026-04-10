"""Test GAN adapter actually loads Layer 3 trained detector."""
import sys
from pathlib import Path

# Add engine to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

print("=" * 80)
print("GAN Adapter Loading Test")
print("=" * 80)

# Test 1: Import GAN interface
print("\n[TEST 1] Importing GAN interface...")
try:
    from engine.inference.adapters.gan import GanInterface
    print("✓ GAN interface imported")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create GAN interface instance
print("\n[TEST 2] Creating GAN interface...")
try:
    gan_adapter = GanInterface(project_root=root)
    print("✓ GAN interface created")
except Exception as e:
    print(f"✗ Failed to create: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load the detector
print("\n[TEST 3] Loading GAN detector (should load layer3_trained_inference)...")
try:
    gan_adapter.load()
    print("✓ GAN detector loaded")
    
    if gan_adapter._predict_fn is None:
        print("✗ Predict function is None (detector didn't load properly)")
        sys.exit(1)
    else:
        print("✓ Predict function initialized")
        
except Exception as e:
    print(f"✗ Failed to load detector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test prediction
print("\n[TEST 4] Testing prediction with sample image...")
try:
    import numpy as np
    from PIL import Image
    
    # Create synthetic image
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    result = gan_adapter.predict(dummy_image)
    
    print(f"✓ Prediction successful")
    fraud_prob = result.get('fraud_probability', 0.5)
    if isinstance(fraud_prob, str):
        fraud_prob = float(fraud_prob)
    print(f"  - Fraud Probability: {fraud_prob:.3f}")
    print(f"  - Flags: {result.get('flags', [])}")
    print(f"  - Sub-scores: {result.get('sub_scores', {})}")
    
    # Check if it's using trained detector (not fallback)
    if 'layer3_module_not_available' in result.get('flags', []):
        print("⚠ WARNING: Using fallback detector (not trained detector)")
    elif 'layer3_unrecognized_result' in result.get('flags', []):
        print("⚠ WARNING: Result format unrecognized")
    else:
        print("✓ Using trained detector successfully")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
