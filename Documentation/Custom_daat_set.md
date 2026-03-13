# VeriSight-PFD: Product Fraud Detection Dataset

**Version:** 1.0  
**Team:** return0  
**Hackathon:** Hack-Nocturne '26 | Cybersecurity Domain  
**Total Images:** 150  

---

## Overview

VeriSight-PFD is a custom-built dataset created specifically for the task of detecting digitally manipulated product complaint images in e-commerce and quick-commerce refund fraud scenarios. No publicly available dataset addresses this specific use case — existing benchmarks (CASIA 2.0, CoMoFoD) target generic image forgery, not product packaging manipulation.

This dataset was constructed from scratch by Team return0 to train, validate, and demonstrate the VeriSight detection pipeline — particularly the OCR-assisted expiry date forensics module and the CNN-based ELA forgery detection module.

---

## Dataset Composition

| Class | Count | Description |
|---|---|---|
| `genuine` | 50 | Real product photographs, unmodified |
| `manually_edited` | 50 | Products with human-edited manipulations using Photoshop / Canva |
| `ai_edited` | 50 | Products with AI-generated or AI-inpainted manipulations |
| **Total** | **150** | **Across 3 classes** |

---

## Class 1 — Genuine (50 images)

### Description
Unmodified photographs of real consumer products taken under varied real-world conditions. These represent legitimate complaint images that a genuine customer might submit.

### Product Categories Covered
- Packaged snacks (chips, biscuits, namkeen)
- Beverages (bottled water, soft drinks, juices)
- Dairy products (milk cartons, yogurt, paneer)
- Grocery staples (rice, flour, pulses)
- Medicines and supplements (OTC tablets, vitamin bottles)
- Cosmetics and personal care (creams, shampoos)

### Capture Conditions
- Varied lighting: natural daylight, indoor fluorescent, low-light
- Varied angles: straight-on, slight tilt, top-down
- Varied backgrounds: table surface, floor, hand-held
- Varied focus: sharp, slightly blurred (simulating phone camera)
- JPEG compression at varied quality levels (60–95%)

### Label Fields
```
image_id, class, product_category, capture_condition, 
expiry_date_visible (bool), expiry_date_value, ocr_ground_truth
```

---

## Class 2 — Manually Edited (50 images)

### Description
Product images from the genuine set that have been deliberately manipulated by a human operator using standard consumer-grade editing tools — replicating the exact methods used by real-world refund fraudsters.

### Manipulation Types

| Manipulation Type | Count | Tools Used |
|---|---|---|
| Expiry date alteration (future → past) | 20 | Photoshop, Canva, PicsArt |
| Brand / product name modification | 10 | Photoshop, Adobe Firefly inpaint |
| MRP or weight tampering | 8 | Canva, Snapseed |
| Batch number modification | 7 | Photoshop |
| Damage insertion (tear, stain, mold) | 5 | Photoshop clone stamp |

### Manipulation Protocol
1. Start from a genuine image in Class 1
2. Identify the target region (expiry date, label, MRP)
3. Apply manipulation using the listed tool
4. Re-export at matched JPEG quality to reduce compression artifacts
5. Verify the manipulation is visually convincing to the naked eye

### Label Fields
```
image_id, class, source_genuine_id, manipulation_type, 
manipulation_tool, manipulated_region_bbox, 
ground_truth_original_text, ground_truth_modified_text
```

---

## Class 3 — AI Edited (50 images)

### Description
Product images where specific regions have been modified using AI-based generation or inpainting tools — replicating the next generation of fraud where generative AI is weaponised to create photorealistic manipulations indistinguishable from real photos.

### AI Manipulation Types

| Manipulation Type | Count | Tools Used |
|---|---|---|
| Expiry date inpainting via AI | 20 | Adobe Firefly inpaint, Stable Diffusion inpaint |
| Full label replacement (AI-generated label) | 12 | DALL-E 3, Stable Diffusion |
| Damage/mold insertion via AI inpaint | 10 | Adobe Firefly, RunwayML |
| Fully synthetic product complaint image | 8 | DALL-E 3, Midjourney |

### Generation Protocol
1. For inpainting: mask the target region → prompt the model to generate a plausible but fraudulent replacement
2. For full generation: prompt model with detailed product description + damage/expiry scenario
3. All outputs verified for photorealism before inclusion
4. Images re-compressed to JPEG to match real-world submission conditions

### Label Fields
```
image_id, class, manipulation_type, generation_tool, 
generation_prompt (summarised), inpaint_mask_path,
is_fully_synthetic (bool)
```

---

## Full Label Schema

Every image in the dataset carries the following unified label record:

```json
{
  "image_id": "vpfd_0042",
  "filename": "vpfd_0042.jpg",
  "class": "manually_edited",
  "product_category": "dairy",
  "manipulation_type": "expiry_date_alteration",
  "manipulation_tool": "Canva",
  "source_genuine_id": "vpfd_0012",
  "expiry_date_visible": true,
  "ground_truth_original_date": "2025-11-30",
  "ground_truth_modified_date": "2024-08-15",
  "manipulated_region_bbox": [142, 88, 210, 112],
  "ocr_ground_truth": "USE BY 15/08/24",
  "capture_condition": "indoor_fluorescent",
  "jpeg_quality": 78,
  "is_fully_synthetic": false,
  "verified": true
}
```

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total images | 150 |
| Genuine images | 50 (33.3%) |
| Manipulated images | 100 (66.7%) |
| Images with expiry date visible | ~110 |
| Product categories | 6 |
| Manipulation tools used | 8+ |
| Images with pixel-level region annotations | 100 |
| Average image resolution | ~1080 × 1440px |
| File format | JPEG |

---

## Train / Validation / Test Split

| Split | Count | Genuine | Manipulated |
|---|---|---|---|
| Train | 105 (70%) | 35 | 70 |
| Validation | 23 (15%) | 8 | 15 |
| Test | 22 (15%) | 7 | 15 |

Stratified split — class proportions maintained across all three sets.

---

## How VeriSight Uses This Dataset

| Module | Usage |
|---|---|
| CNN Image Forensics (EfficientNet-B4 + ELA) | Fine-tuning on manually_edited class for packaging-specific forgery patterns |
| Transformer AI Detection (ViT + CLIP) | Fine-tuning on ai_edited class for product-context AI generation detection |
| OCR Text Verification | Ground truth expiry dates used to validate OCR extraction accuracy |
| Score Engine Calibration | All 150 images used to calibrate ACS thresholds (genuine vs manipulated boundaries) |

---

## Limitations

- Dataset size (150 images) is intentionally constrained for the hackathon scope. Production deployment would require 10,000+ images across broader product categories.
- Current dataset covers Indian consumer product packaging primarily — international packaging styles are underrepresented.
- Manipulation difficulty is not uniformly graded — future versions will include a `manipulation_difficulty` label (easy / medium / hard).
- Video complaint submissions (increasingly common on platforms) are not covered in v1.0.

---

## Ethical Statement

All product images were photographed by Team return0 members from products they personally own. No proprietary platform data was used. Manipulated images were created solely for the purpose of training a fraud detection system and are not intended for any deceptive use. The dataset will not be publicly released without removal of any personally identifiable information.

---

## Citation

If referencing this dataset in academic or competition contexts:

```
Team return0 (2026). VeriSight-PFD: Product Fraud Detection Dataset v1.0.
Hack-Nocturne '26, Cybersecurity Domain.
Custom dataset for packaging image manipulation detection in e-commerce refund fraud.
```

---

*VeriSight-PFD v1.0 — Hack-Nocturne '26 — Team return0 — Confidential*