# Score Range Compression Analysis & Fix Plan

**Date:** April 9, 2026  
**Issue:** Tested 10 images and all scores compressed to 40-70 range (expected: 0-100)

---

## ROOT CAUSE ANALYSIS ✅

### What We Found

Diagnostic test on 5 images revealed:

```
Layer Scores (20 total):
  CNN: 6, 93, 6, 99, 80  → Range: 6-99 (EXCELLENT ✓)
  ViT: 30, 49, 52, 41, 21 → Range: 21-52 (GOOD ✓)
  GAN: 52, 52, 51, 52, 51 → Range: 51-52 (BROKEN ❌)
  OCR: 45, 45, 45, 45, 45 → Range: 45-45 (BROKEN ❌)
```

**Key Finding:** Individual layer variance is 93 points, but ensemble scores were 40-70!

### Problem #1: OCR Stuck at 45

**File:** `layer4/inference/ocr_verification.py` → `analyze()` method (line 252)

```python
# If no regions detected but we have a detector, text is missing (suspicious)
if not regions:
    return {
        "score": 45.0,  # ← CONSTANT!
        ...
    }
```

**Root Cause:** 
- YOLO text detector failing (returning 0 regions)
- When YOLO fails, OCR returns hardcoded fallback score of 45.0
- This affects ALL images, not just ones without text

**Fix Needed:**
1. Debug why YOLO text detection returns 0 regions
2. Implement proper fallback scoring (not constant)
3. Or disable OCR from ensemble until fixed

### Problem #2: GAN Stuck at 51

**File:** `engine/inference/adapters/gan.py` → `predict()` method (line 177)

```python
raw = self._normalize_detector_output(self._predict_fn(image_input))
fraud_probability = float(raw.get("fraud_probability", 0.5))  # ← DEFAULT 0.5!
```

**Root Cause:**
- Trained detector not loading properly
- Returns default `fraud_probability = 0.5`
- 0.5 fraud_probability → 51 authenticity score (constant)

**Fix Needed:**
1. Debug why trained detector not loading
2. Check `layer3_trained_inference.py` is correct
3. Verify checkpoint path exists

---

## IMMEDIATE SOLUTIONS

### Option 1: Disable Broken Layers (Quick Fix)

Disable GAN and OCR from ensemble since they're non-functional:

```python
# In engine/core/config.py

LAYER_WEIGHTS = {
    "cnn": 0.6,    # Increase CNN weight
    "vit": 0.4,    # Increase ViT weight
    "gan": 0.0,    # DISABLE - returns constant
    "ocr": 0.0,    # DISABLE - returns constant
}

DECISION_THRESHOLDS = [
    (85, "AUTO_APPROVE"),
    (65, "FAST_TRACK"),
    (45, "SUSPICIOUS"),
    (0, "REJECT"),
]
```

**Pros:** Tests work immediately with 2-layer ensemble
**Cons:** Loses GAN/OCR forensic signals
**Score Result:** Will expand to full 0-100 range with CNN+ViT

### Option 2: Fix Broken Layers (Proper Solution)

1. **Fix OCR Text Detection**
   - Debug `layer4/inference/ocr_verification.py` YOLO detector
   - Check if best.pt model exists and loads
   - Implement proper scoring when text detected vs. not detected
   - Return varied scores based on text metrics

2. **Fix GAN Trained Detector**
   - Debug `layer3/layer3_trained_inference.py` loading
   - Check checkpoint path: `layer3/checkpoints/layer3_best.pth`
   - Verify it returns fraud_probability in (0.0, 1.0) range
   - Test with actual image input

3. **Re-enable Weights**
   ```python
   LAYER_WEIGHTS = {
       "cnn": 0.25,   # Balanced
       "vit": 0.20,   # Balanced
       "gan": 0.30,   # Now fixed
       "ocr": 0.25,   # Now fixed
   }
   ```

---

## RECOMMENDED ACTION: HYBRID APPROACH

**Stage 1 (IMMEDIATE):** Disable broken layers
- Edit `engine/core/config.py`
- Set GAN and OCR weights to 0.0
- Test that CNN+ViT now produce 0-100 range

**Stage 2 (WITHIN HOUR):** Fix OCR text detection
- Debug why YOLO returns no regions
- Implement actual text-based scoring

**Stage 3 (WITHIN HOUR):** Fix GAN detector
- Verify trained model loads
- Test GAN returns varied fraud_probability values

**Stage 4 (FINAL):** Re-enable both layers with proper scores

---

## EVIDENCE

### Diagnostic Output

```
Image 1: CNN=83, ViT=30, GAN=52, OCR=45 → Should be ~52 after weighted avg
Image 2: CNN=93, ViT=49, GAN=52, OCR=45 → Should be ~70 after weighted avg
Image 3: CNN=6,  ViT=52, GAN=51, OCR=45 → Should be ~38 after weighted avg
...
```

The fact that GAN and OCR are **constant** while CNN and ViT **vary wildly** proves two separate model failures, not an ensemble issue.

### Configuration Impact

Current weights (after retuning):
```
"cnn": 0.25 (varies 6-99)    × weight = important
"vit": 0.20 (varies 21-52)   × weight = important
"gan": 0.30 (constant 51-52) × weight = ZERO OUT
"ocr": 0.25 (constant 45)    × weight = ZERO OUT
```

The constant scores from GAN+OCR are dragging ensemble toward middle regardless of CNN/ViT variance!

---

## NEXT STEPS

1. **YOU DECIDE:** Quick fix (disable layers) or proper fix (debug models)?
2. **If Quick Fix:** I can update config immediately → test with 2 layers
3. **If Proper Fix:** Need to investigate layer3/layer4 model loading

Which would you prefer?
