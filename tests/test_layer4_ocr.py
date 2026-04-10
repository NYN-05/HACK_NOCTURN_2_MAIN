"""Test script to verify Layer 4 OCR components are working."""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add layer4 to path
layer4_path = Path(__file__).parent / "layer4"
if layer4_path not in sys.path:
    sys.path.insert(0, str(layer4_path))

print("=" * 80)
print("Layer 4 OCR Component Verification Test")
print("=" * 80)

# Test 1: Check if best.pt exists
print("\n[TEST 1] Checking best.pt file...")
best_pt = layer4_path / "best.pt"
if best_pt.exists():
    size_mb = best_pt.stat().st_size / (1024 * 1024)
    print(f"✓ best.pt found: {size_mb:.2f} MB")
else:
    print(f"✗ best.pt NOT found at {best_pt}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[TEST 2] Checking Python dependencies...")
deps = {
    "torch": "torch",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "easyocr": "easyocr",
    "paddleocr": "paddleocr"
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

# Test 3: Try importing OCRVerificationModule
print("\n[TEST 3] Testing OCRVerificationModule import...")
try:
    from inference.ocr_verification import OCRVerificationModule
    print("✓ OCRVerificationModule imported successfully")
except Exception as e:
    print(f"✗ Failed to import OCRVerificationModule: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try initializing OCRVerificationModule
print("\n[TEST 4] Initializing OCRVerificationModule...")
try:
    module = OCRVerificationModule(
        text_detector_model=str(best_pt),
        use_gpu=False,
        enable_yolo=True,
        enable_easyocr=True,
        enable_paddle_fallback=True,
        confidence_threshold=0.3
    )
    print("✓ OCRVerificationModule initialized successfully")
    
    # Check if YOLO detector loaded
    if module.detector is None:
        print("✗ YOLO detector is None after initialization!")
    else:
        print(f"✓ YOLO detector loaded: {type(module.detector)}")
    
    # Check OCR engines
    if module.ocr_engines:
        print(f"✓ OCR engines initialized: {list(module.ocr_engines.keys())}")
    else:
        print("✗ No OCR engines initialized")
        
except Exception as e:
    print(f"✗ Failed to initialize OCRVerificationModule: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test analyze method
print("\n[TEST 5] Testing analyze method with a sample image...")
try:
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    result = module.analyze(dummy_image)
    
    print(f"✓ Analyze returned result")
    print(f"  - Score: {result.get('score')}")
    print(f"  - Available: {result.get('available')}")
    print(f"  - Uncertainty: {result.get('uncertainty')}")
    print(f"  - Flags: {result.get('flags')}")
    print(f"  - Regions detected: {result.get('details', {}).get('regions_detected')}")
    
    # Verify score is not constantly 50.0
    if result.get('score') == 50.0:
        print("⚠ Warning: Score is 50.0 (might indicate fallback)")
    else:
        print("✓ Score is varying (not constant 50.0)")
        
except Exception as e:
    print(f"✗ Analyze method failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
