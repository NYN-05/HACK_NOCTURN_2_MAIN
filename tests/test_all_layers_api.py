"""Test /api/v1/verify endpoint with all 4 layers."""
import requests
import json
import numpy as np
from PIL import Image
from io import BytesIO

# Server endpoint
BASE_URL = "http://127.0.0.1:8000"
VERIFY_ENDPOINT = f"{BASE_URL}/api/v1/verify"

print("=" * 80)
print("Testing /api/v1/verify Endpoint with All 4 Layers")
print("=" * 80)

# Test 1: Create synthetic test image
print("\n[TEST 1] Creating synthetic test image...")
try:
    # Create a simple colored image
    img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # Add some pattern to make it less trivial
    for i in range(256):
        img_array[i, :, 0] = (img_array[i, :, 0] + i) % 256
        img_array[i, :, 1] = (img_array[i, :, 1] + (i * 2 % 256)) % 256
    
    img = Image.fromarray(img_array, 'RGB')
    
    # Convert to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    print(f"✓ Test image created (256x256 RGB)")
except Exception as e:
    print(f"✗ Failed to create image: {e}")
    import sys
    sys.exit(1)

# Test 2: Send request to verify endpoint
print("\n[TEST 2] Sending request to /api/v1/verify endpoint...")
try:
    files = {'image': ('test_image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'order_date': '2026-04-09',
        'delivery_date': '2026-04-09',
        'mfg_date_claimed': '2026-04-01'
    }
    
    response = requests.post(VERIFY_ENDPOINT, files=files, data=data, timeout=30)
    
    if response.status_code != 200:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"Response: {response.text}")
        import sys
        sys.exit(1)
    
    result = response.json()
    print("✓ Request successful")
    
except Exception as e:
    print(f"✗ Request failed: {e}")
    import sys
    sys.exit(1)

# Test 3: Validate response structure
print("\n[TEST 3] Validating response structure...")
required_keys = [
    'authenticity_score',
    'decision',
    'layer_scores',
    'confidence',
    'schema_version'
]

missing_keys = [k for k in required_keys if k not in result]
if missing_keys:
    print(f"✗ Missing keys in response: {missing_keys}")
else:
    print(f"✓ All required keys present")

# Test 4: Check layer scores
print("\n[TEST 4] Analyzing layer scores...")
layer_scores = result.get('layer_scores', {})
print(f"  Layer Scores: {json.dumps(layer_scores, indent=2)}")

# Check if all layers are present
expected_layers = ['layer1_score', 'layer2_score', 'layer3_score', 'layer4_score']
found_layers = [k for k in expected_layers if k in layer_scores]
missing_layers = [k for k in expected_layers if k not in layer_scores]

print(f"\n  Found layers: {len(found_layers)}/4")
for layer in found_layers:
    score = layer_scores[layer]
    print(f"    ✓ {layer}: {score:.3f}")

if missing_layers:
    print(f"\n  Missing layers: {missing_layers}")

# Test 5: Check for dynamic scores (not constants)
print("\n[TEST 5] Verifying dynamic scores (not constants)...")
dynamic_check = True

for layer, score in layer_scores.items():
    if score == 0.5 or score == 50.0:
        print(f"  ⚠ {layer} returning constant {score} (might be fallback)")
        dynamic_check = False
    elif score == 0.0 or score == 1.0:
        print(f"  ⚠ {layer} returning boundary value {score} (unusual)")
    else:
        print(f"  ✓ {layer} returning computed value: {score:.3f}")

if dynamic_check:
    print("\n✓ All layers returning dynamic scores")

# Test 6: Check ensemble result
print("\n[TEST 6] Checking ensemble result...")
authenticity_score = result.get('authenticity_score')
decision = result.get('decision')
confidence = result.get('confidence')

print(f"  Authenticity Score: {authenticity_score:.1f}")
print(f"  Decision: {decision}")
print(f"  Confidence: {confidence:.3f}")

# Test 7: Check availability and status
print("\n[TEST 7] Checking layer availability and status...")
layer_status = result.get('layer_status', {})
available_layers = result.get('available_layers', [])

print(f"  Available Layers: {available_layers}")
print(f"  Layer Status:")
for layer, status in layer_status.items():
    print(f"    - {layer}: {status}")

# Test 8: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✓ Server responding to /api/v1/verify endpoint")
print(f"✓ Response schema version: {result.get('schema_version')}")
print(f"✓ All 4 layers processed: {len(found_layers) == 4}")
print(f"✓ Dynamic scores verified: {dynamic_check}")
print(f"\nFinal Result:")
print(f"  - Authenticity Score: {authenticity_score:.1f}/100")
print(f"  - Decision: {decision}")
print(f"  - Confidence: {confidence:.3f}")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
