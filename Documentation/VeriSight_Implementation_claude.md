# VeriSight — Implementation Guide
> AI-Powered Deepfake & Image Manipulation Detection for Refund Fraud Prevention  
> Team: **return0** | Domain: Cybersecurity

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Detection Pipeline (4-Layer)](#3-detection-pipeline-4-layer)
4. [Tech Stack](#4-tech-stack)
5. [Dataset Strategy](#5-dataset-strategy)
6. [Scoring & Decision Engine](#6-scoring--decision-engine)
7. [API Design](#7-api-design)
8. [Frontend Dashboard](#8-frontend-dashboard)
9. [XAI & Explainability](#9-xai--explainability)
10. [Deployment](#10-deployment)
11. [Challenges & Mitigations](#11-challenges--mitigations)
12. [References & Research Papers](#12-references--research-papers)

---

## 1. Project Overview

### Problem Statement
E-commerce and quick-commerce platforms (Zomato, Swiggy, Blinkit, Amazon, Flipkart) approve refunds based **solely on user-uploaded images**. Fraudsters exploit this by:
- Manipulating expiry dates and product labels using AI tools
- Generating synthetic complaint images via generative models
- Operating organized fraud rings (e.g., the Rs 5.5 Cr Meesho case)

**Scale of the problem:** Returns fraud cost merchants **USD 103 Billion globally (2023)**, with 45% of Indian consumers affected. No automated, scalable system currently exists to verify the authenticity of complaint images.

### Solution — VeriSight
An AI-powered visual verification API that:
- Auto-analyzes every complaint image submitted to a platform
- Runs a **4-layer parallel detection pipeline** in under 5 seconds
- Returns an **Authenticity Confidence Score (0–100)**
- Provides fully explainable decisions via heatmaps and SHAP reports

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VERISIGHT FLOW                           │
│                                                                 │
│  STEP 1                STEP 2              STEP 3               │
│  Customer         →    API Gateway    →    Parallel Analysis    │
│  Submits               FastAPI · Auth       Modules             │
│  Complaint             · Validation                             │
│  (image +                                                       │
│   order details)                                                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            PARALLEL ANALYSIS MODULES                     │   │
│  │                                                          │   │
│  │  CNN Image Forensics        (35% weight)                 │   │
│  │  EfficientNet-B4 · ELA Maps · Copy-Move · Splicing       │   │
│  │                                                          │   │
│  │  Transformer AI Detection   (30% weight)                 │   │
│  │  ViT-B/16 · CLIP-ResNet50 · Frequency Analysis           │   │
│  │                                                          │   │
│  │  GAN Artifact Analysis      (20% weight)                 │   │
│  │  AutoGAN · Spectrum · Texture · Boundary Blending        │   │
│  │                                                          │   │
│  │  OCR Text Verification      (15% weight)                 │   │
│  │  EasyOCR · YOLOv8 · Expiry Date · Font Forensics         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                    SCORE ENGINE (0–100)                         │
│                    SHAP · Grad-CAM · Heatmap                    │
│                            │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DECISION THRESHOLDS                         │   │
│  │  85–100  ✅  Genuine      → Auto-approve refund (<60s)   │   │
│  │  60–84   🔍  Likely       → Fast-track human spot-check  │   │
│  │  35–59   ⚠️  Suspicious   → Hold · Request more proof    │   │
│  │  0–34    ❌  Manipulated  → Reject · Flag evidence       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│              OUTPUT: Score · Heatmap · Audit Log                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detection Pipeline (4-Layer)

### Layer 1 — CNN Image Forensics (Weight: 35%)
**Model:** EfficientNet-B4

| Technique | Purpose |
|---|---|
| Error Level Analysis (ELA) | Detects re-saved/edited regions by compression artifact differences |
| Copy-Move Detection | Identifies copy-pasted regions within the same image |
| Splicing Detection | Detects regions composited from different source images |
| Noise Pattern Analysis | Reveals inconsistencies in sensor noise across image regions |

**Implementation:**
```python
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b4')
# Fine-tune on CASIA 2.0 + custom dataset
# ELA preprocessing before inference
```

---

### Layer 2 — Transformer AI Detection (Weight: 30%)
**Models:** ViT-B/16 + CLIP-ResNet50

| Technique | Purpose |
|---|---|
| Vision Transformer (ViT-B/16) | Global patch-level feature analysis for AI-generation artifacts |
| CLIP-ResNet50 | GAN fingerprint detection (CVPR 2023 — 99.4% accuracy) |
| Frequency Domain Analysis | FFT-based detection of generative model artifacts |

**Implementation:**
```python
from transformers import ViTForImageClassification, ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# Fine-tune on ImagiNet (200K real vs AI-generated images)
```

---

### Layer 3 — GAN Artifact Analysis (Weight: 20%)
**Approach:** AutoGAN-based spectrum and texture analysis

| Technique | Purpose |
|---|---|
| Spectrum Analysis | Detects GAN-specific frequency artifacts (checkerboard patterns) |
| Texture Inconsistency | Surface texture mismatches between regions |
| Boundary Blending Detection | Unnatural transitions at edit boundaries |
| LGrad Gradient Analysis | Learning on gradients for generalized GAN detection |

**Implementation:**
```python
# AutoGAN detection via spectrum analysis
import numpy as np
from PIL import Image

def analyze_spectrum(image_path):
    img = np.array(Image.open(image_path).convert('L'))
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    return magnitude  # Feed to classifier
```

---

### Layer 4 — OCR Text Verification (Weight: 15%)
**Models:** EasyOCR + PaddleOCR + YOLOv8

| Technique | Purpose |
|---|---|
| Text Localization (YOLOv8) | Detect and crop text regions (expiry dates, labels) |
| OCR Extraction (EasyOCR) | Extract text from packaging |
| Font Forensics | Detect inconsistent fonts, kerning, or typeface mismatches |
| Expiry Date Validation | DBNet + CBAM for improved date field recognition |
| Cross-field Consistency | Compare MFG date, expiry date, and batch number logic |

**Implementation:**
```python
import easyocr
from ultralytics import YOLO

yolo_model = YOLO('yolov8n.pt')  # Text detection
reader = easyocr.Reader(['en'])

def verify_expiry(image_path):
    text_regions = yolo_model(image_path)
    extracted_text = reader.readtext(image_path)
    # Font forensics + date logic validation
    return extracted_text
```

---

## 4. Tech Stack

### AI / ML
| Component | Technology |
|---|---|
| CNN Forensics | EfficientNet-B4 (PyTorch) |
| Transformer Detection | ViT-B/16, CLIP-ResNet50 (HuggingFace) |
| GAN Analysis | AutoGAN, LGrad |
| OCR | EasyOCR + PaddleOCR |
| Text Detection | YOLOv8 |
| Inference Runtime | ONNX Runtime (with quantization) |
| XAI | SHAP, Grad-CAM, GCA-Net |
| CV Utilities | OpenCV, Pillow |

### Backend
| Component | Technology |
|---|---|
| API Framework | FastAPI (async REST) |
| Task Queue | Celery + Redis |
| Database | PostgreSQL |
| Auth | FastAPI Auth middleware |

### Frontend
| Component | Technology |
|---|---|
| Dashboard | React.js |
| Containerization | Docker + Kubernetes |
| CI/CD | GitHub Actions |

### Cloud / DevOps
| Component | Technology |
|---|---|
| Cloud Deployment | AWS / GCP |
| Inference Optimization | ONNX Runtime + model quantization |

---

## 5. Dataset Strategy

### Public Datasets
| Dataset | Size | Purpose |
|---|---|---|
| CASIA 2.0 | 12,614 images | Image forgery (splicing + copy-move) |
| DF2023 | 1M+ images | Robust generalization across manipulation types |
| ImagiNet (Boychev et al.) | 200,000 images | Real vs AI-generated image classification |

### Custom Dataset
- **4,000+ labeled product complaint images** built from scratch
- Covers **5 manipulation classes:**
  1. Expiry date manipulation
  2. Label/text editing
  3. Synthetic image generation (GAN)
  4. Copy-move fraud
  5. Splicing / compositing

### Data Augmentation Strategy
```python
from torchvision import transforms

augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 6. Scoring & Decision Engine

### Weighted Score Computation
```python
def compute_authenticity_score(cnn_score, vit_score, gan_score, ocr_score):
    weights = {
        'cnn': 0.35,
        'vit': 0.30,
        'gan': 0.20,
        'ocr': 0.15
    }
    score = (
        cnn_score * weights['cnn'] +
        vit_score * weights['vit'] +
        gan_score * weights['gan'] +
        ocr_score * weights['ocr']
    )
    return round(score, 2)

def get_decision(score):
    if score >= 85:
        return "AUTO_APPROVE", "Genuine — refund processed in <60s"
    elif score >= 60:
        return "FAST_TRACK", "Likely genuine — human spot-check"
    elif score >= 35:
        return "SUSPICIOUS", "Hold — request additional proof"
    else:
        return "REJECT", "Manipulated — flag and generate evidence report"
```

---

## 7. API Design

### Endpoint: POST `/api/v1/verify`
```json
// Request
{
  "order_id": "ORD-20240312-001",
  "platform": "zomato",
  "complaint_type": "expired_product",
  "image_base64": "<base64_encoded_image>",
  "order_metadata": {
    "product_name": "Amul Milk 500ml",
    "order_date": "2024-03-10",
    "mfg_date_claimed": "2024-02-01"
  }
}

// Response
{
  "verification_id": "VST-20240312-XYZ",
  "authenticity_score": 23,
  "decision": "REJECT",
  "decision_label": "Manipulated",
  "layer_scores": {
    "cnn_forensics": 18,
    "transformer_detection": 21,
    "gan_analysis": 30,
    "ocr_verification": 27
  },
  "flags": [
    "expiry_date_font_inconsistency",
    "ela_compression_artifact",
    "frequency_domain_anomaly"
  ],
  "processing_time_ms": 3240,
  "heatmap_url": "https://cdn.verisight.ai/heatmaps/VST-XYZ.png",
  "shap_report_url": "https://cdn.verisight.ai/shap/VST-XYZ.pdf",
  "audit_log_id": "AUDIT-20240312-XYZ"
}
```

### Endpoint: GET `/api/v1/status/{verification_id}`
Returns async processing status for queued jobs (Celery).

### Endpoint: GET `/api/v1/audit/{audit_log_id}`
Returns the full SHAP + Grad-CAM audit report (human-reviewer access).

---

## 8. Frontend Dashboard

### Reviewer Dashboard Features
- **Live feed** of incoming complaint image submissions
- **Score distribution** charts (genuine / suspicious / rejected)
- **Heatmap overlay** viewer for flagged images
- **Manual override** panel for 35–84 score range cases
- **Fraud pattern analytics** — trending manipulation types
- **Exportable audit reports** per complaint (SHAP PDF)

### Platform Integration
```javascript
// Example: Zomato complaint webhook integration
const verifyComplaintImage = async (complaintData) => {
  const response = await fetch('https://api.verisight.ai/v1/verify', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(complaintData)
  });
  const result = await response.json();
  return result; // { score, decision, heatmap_url, ... }
};
```

---

## 9. XAI & Explainability

### Tools Used
| Tool | Role |
|---|---|
| **SHAP** | Feature attribution — which image regions most influenced the score |
| **Grad-CAM** | Gradient-weighted class activation maps — visual explanation of CNN decisions |
| **GCA-Net** | Gated Context Attention — precise forgery region localization |

### Why Explainability Matters
- Human reviewers can **audit** the AI's reasoning, not just accept its verdict
- SHAP reports serve as **legally defensible digital evidence** in fraud cases
- Enables **threshold calibration** over time using reviewer feedback

### Example SHAP Output
```python
import shap

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(complaint_image_tensor)
shap.image_plot(shap_values, complaint_image_np)
# Highlights regions: expiry date zone, label area, spliced region
```

---

## 10. Deployment

### Infrastructure
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Platform   │────▶│  API Gateway │────▶│  FastAPI App    │
│  (Zomato /  │     │  (Auth +     │     │  (Docker Pod)   │
│   Swiggy /  │     │   Rate Limit)│     │                 │
│   Blinkit)  │     └──────────────┘     └────────┬────────┘
└─────────────┘                                   │
                                          ┌───────▼────────┐
                                          │  Celery Queue  │
                                          │  (Redis)       │
                                          └───────┬────────┘
                                                  │
                              ┌───────────────────┼──────────────────┐
                              │                   │                  │
                    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌───────▼───────┐
                    │  CNN Worker    │  │  ViT Worker    │  │  OCR Worker   │
                    │  (GPU Pod)     │  │  (GPU Pod)     │  │  (CPU Pod)    │
                    └────────────────┘  └────────────────┘  └───────────────┘
```

### Docker Setup
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Performance Optimization
- **ONNX Runtime** + INT8 quantization for CPU inference during early deployment
- **Celery** parallel task workers for sub-5-second response at scale
- **Redis caching** for repeated similar image hash lookups
- GPU-backed pods for ViT and EfficientNet inference at scale

---

## 11. Challenges & Mitigations

| Challenge | Risk Level | Mitigation Strategy |
|---|---|---|
| Adversarial attacks — fraudsters adapting | High | Continuous model retraining on newly detected fraud patterns |
| OCR on poor lighting / damaged packaging | Medium | Image preprocessing (contrast enhancement, denoising) + PaddleOCR fallback |
| New generative AI models outpacing detection | High | Regular benchmarking on DF2023; CLIP fingerprint updates |
| False positives — genuine customers flagged | Medium | Human reviewer override for 35–84 range; XAI audit trail |
| No public platform-specific complaint dataset | High | Custom dataset of 4,000+ labeled images across 5 manipulation classes |
| GPU requirement for sub-5s latency at scale | Medium | ONNX quantization for CPU inference; GPU pods for production scale |
| Threshold miscalibration across platforms | Medium | Platform-specific threshold tuning via A/B testing and reviewer feedback |

---

## 12. References & Research Papers

### Key Research Papers
1. *Image Forgery Detection via ELA-CNN, VGG19, EfficientNet-B2*
2. *GCA-Net: Gated Context Attention for Image Forgery Localization*
3. *LGrad: Learning on Gradients for Generalized GAN Image Detection*
4. *AutoGAN: Detecting GAN-Fake Images via Spectrum Analysis*
5. *Advanced Detection of AI-Generated Images via Vision Transformers*
6. *Expiry Date Recognition with Improved DBNet + CBAM*
7. *Integration of LLM in Expiration Date Scanning (R-CNN: 91.1% precision)*
8. *Generalized FCN Framework for Expiration Date Recognition*

### Industry Reports
- Loop Returns 2024
- McAfee Global Survey 2024
- IEEE 2024

### Datasets & Pretrained Models
- [CASIA 2.0](https://github.com/namtpham/casia2groundtruth) — 12,614 image forgery samples
- [ImagiNet](https://github.com/boychev/imaginet) — 200,000 real vs AI-generated images
- [HuggingFace ViT-B/16](https://huggingface.co/google/vit-base-patch16-224)
- [OpenAI CLIP](https://github.com/openai/CLIP) — GAN fingerprint detection

---

*VeriSight — Team return0 | Built for AiFi National Level AI Hackathon, REVA University & Hack-Nocturne '26*
