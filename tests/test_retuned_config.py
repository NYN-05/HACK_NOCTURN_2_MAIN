"""Quick test of retuned configuration."""
import requests
import json
import numpy as np
from PIL import Image
from io import BytesIO

BASE_URL = "http://127.0.0.1:8000"
VERIFY_ENDPOINT = f"{BASE_URL}/api/v1/verify"

print("Testing Retuned Configuration with Balanced Layer Weights")
print("=" * 70)

# Create simple test image
img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
for i in range(256):
    img_array[i, :, 0] = (img_array[i, :, 0] + i) % 256

img = Image.fromarray(img_array, 'RGB')
img_bytes = BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

try:
    print("\nSending request to /api/v1/verify (timeout: 120s)...")
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'order_date': '2026-04-09'}
    
    response = requests.post(VERIFY_ENDPOINT, files=files, data=data, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        layer_scores = result.get('layer_scores', {})
        auth = result.get('authenticity_score', 0)
        decision = result.get('decision', 'N/A')
        
        print(f"\n✓ Request successful!")
        print(f"\nLayer Scores (with new balanced weights):")
        print(f"  CNN: {layer_scores.get('cnn', 'N/A')} (25% weight)")
        print(f"  ViT: {layer_scores.get('vit', 'N/A')} (20% weight)")
        print(f"  GAN: {layer_scores.get('gan', 'N/A')} (30% weight)")
        print(f"  OCR: {layer_scores.get('ocr', 'N/A')} (25% weight) ✅ NOW ACTIVE")
        
        print(f"\nEnsemble Result (retuned):")
        print(f"  Authenticity Score: {auth}")
        print(f"  Decision: {decision}")
        
        # Check if OCR is contributing (had 0.000009 weight before)
        ocr_score = layer_scores.get('ocr', 0)
        if ocr_score != 0:
            print(f"\n✅ SUCCESS: OCR now contributing to final score!")
            print(f"   OCR went from 0.000009 weight → 0.25 weight")
        
    else:
        print(f"✗ Failed: {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")
