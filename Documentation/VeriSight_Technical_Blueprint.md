# VeriSight — Complete Professional Technical Blueprint
### AI-Powered Visual Verification System for Delivery Platform Fraud Detection

**Team:** return0 | **Hackathon:** Hack-Nocturne '26 | **Domain:** Cybersecurity  
**Venues:** Hack-Nocturne '26 & AiFi National Level AI Hackathon — REVA University  
**Document Type:** Senior Architect Technical Blueprint | **Version:** 1.0

---

> **Note on Document Scope:** This blueprint is derived from the three provided project documents. All assumptions or inferred details that go beyond what is explicitly stated in those documents are clearly marked with the tag **`[INFERRED]`**.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Conceptual and Theoretical Foundations](#3-conceptual-and-theoretical-foundations)
4. [System Architecture](#4-system-architecture)
5. [Algorithmic and Model Design](#5-algorithmic-and-model-design)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Technology Stack](#7-technology-stack)
8. [Environment Setup](#8-environment-setup)
9. [Step-by-Step Implementation Plan](#9-step-by-step-implementation-plan)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Testing Strategy](#11-testing-strategy)
12. [Limitations](#12-limitations)
13. [Risk Analysis](#13-risk-analysis)
14. [Production-Grade System Design](#14-production-grade-system-design)
15. [Deployment Architecture](#15-deployment-architecture)
16. [Experiment Tracking and Versioning](#16-experiment-tracking-and-versioning)
17. [Recommended Project Directory Structure](#17-recommended-project-directory-structure)
18. [Reproducibility Guide](#18-reproducibility-guide)
19. [Future Improvements](#19-future-improvements)

---

## 1. Executive Summary

### Short Overview

VeriSight is a multi-modal, AI-powered **Visual Verification Microservice** designed specifically for delivery and e-commerce platforms. It intercepts every user-submitted product complaint image and subjects it to four parallel forensic analysis layers, returning a single **Authenticity Confidence Score (0–100)** with a full explainability report — all within five seconds.

### Problem the Project Solves

Delivery platforms such as Zomato, Swiggy, Blinkit, BigBasket, Amazon, and Flipkart process millions of refund and complaint requests daily, approving them almost entirely on the basis of user-uploaded photographs. Fraudsters exploit this trust-first system by digitally manipulating product images — altering expiry dates, editing packaging labels, generating fully synthetic complaint photos using generative AI, or using GAN-based inpainting to insert realistic-looking damage into genuine product photographs. The financial cost of this fraud globally exceeds **USD 103 billion annually** in returns abuse alone.

### Core Innovation

VeriSight's innovation lies in three dimensions:

1. **Domain specificity.** It is the first system designed explicitly for the refund fraud image-verification use case, with a dedicated OCR module for expiry date forensics that no general-purpose forensics system provides.
2. **Four-layer parallel multi-modal analysis.** Each layer targets a distinct manipulation vector (pixel forensics, AI generation, GAN artifacts, and OCR text plausibility), eliminating any single point of failure.
3. **Explainable, auditable decisions.** Every score is accompanied by a SHAP feature-attribution breakdown and Grad-CAM + GCA-Net heatmap overlays, making outputs usable as legally defensible digital evidence.

### Expected System Outcome

For every submitted complaint image, VeriSight produces a numeric Authenticity Confidence Score (0–100), a decision label (GENUINE / LIKELY_GENUINE / SUSPICIOUS / MANIPULATED), a recommended platform action, a visual heatmap overlay on the complaint image, a SHAP-based evidence report, and OCR-extracted text with expiry date plausibility analysis.

---

## 2. Problem Statement

### 2.1 The Refund Complaint Ecosystem

Modern quick-commerce and food delivery platforms have made generous, fast refund policies a cornerstone of their customer experience. When a customer reports an expired product, wrong item, or contaminated food, platforms typically process the refund with minimal verification — relying almost entirely on the user-uploaded image as proof. This design served platforms well when image manipulation required professional expertise. It has become a critical vulnerability in the era of freely available generative AI tools.

### 2.2 Fraud Patterns Targeted

VeriSight is designed to detect six distinct fraud patterns:

| Fraud Type | Description |
|---|---|
| **Expiry Date Manipulation** | A valid future expiry date is digitally altered to a past date to falsely claim an expired delivery |
| **Packaging Label Manipulation** | Brand name, MRP, weight, batch number, or ingredient list is digitally edited to simulate wrong product delivery |
| **AI Image Generation** | A fully synthetic product complaint image is created using Stable Diffusion, DALL-E, Adobe Firefly, or Midjourney |
| **GAN-Manipulated Images** | GAN inpainting tools are used to realistically insert damage, mold, or incorrect labeling into a genuine product photo |
| **Image Splicing / Compositing** | Content from a different product image is composited into the complaint photo |
| **Recycled Complaint Images** | Previously submitted complaint images are reused across orders or accounts |

### 2.3 Why Existing Systems Fail

- **Human-only review** is slow (24–48 hour turnaround), inconsistent, and does not scale to platform volumes.
- **Rule-based filters** operate at the account level and do not examine image authenticity at all.
- **Hash-based duplicate detection** is trivially bypassed by applying minor edits to a recycled image.
- **Generative AI tools** produce photorealistic images visually indistinguishable from genuine photographs without algorithmic analysis.
- **No dedicated product-image forensics system** exists for the refund fraud context.

### 2.4 Market Scale and Financial Impact

| Metric | Data |
|---|---|
| Global e-commerce fraud losses (2024) | USD 48 billion annually |
| Return and refund fraud globally (2023) | USD 103 billion |
| India deepfake shopping scams (2024) | 45% of Indian consumers affected (McAfee Global Survey) |
| India — Meesho fraud case | Rs 5.5 crore defrauded by a single organized group |
| India consumer complaints growth | 7,482 grievances against food delivery apps in 2024–25 — a 10× increase from 2020–21 |

---

## 3. Conceptual and Theoretical Foundations

### 3.1 Error Level Analysis — The Foundation of CNN Forensics

**Core idea:** When a JPEG image is saved, the compression algorithm applies lossy encoding uniformly. If any region is digitally edited and the image is re-saved, that region is re-encoded from a different compression baseline than the surrounding pixels. This asymmetry is measurable.

**How it works:** VeriSight re-saves the complaint image at a controlled JPEG quality level, then computes the pixel-wise absolute difference between the original and the re-saved version. Regions that have been manipulated exhibit disproportionately higher error levels. This ELA map is fed as additional input channels alongside the original RGB image, giving the CNN a two-perspective view.

**Mathematical intuition:** If `I` is the original image and `I'` is the re-compressed version at quality `q`, the ELA map `E` is computed as `E = |I - I'|`, scaled for visual clarity. Manipulated regions cluster toward the high end of this difference distribution.

### 3.2 Copy-Move and Splicing Forensics

**Copy-move forgery** occurs when a region is duplicated and repositioned within the same image. The CNN detects this through noise pattern analysis: genuine images have consistent sensor-level noise distributions, whereas copied regions exhibit identical noise patterns to their source, creating detectable statistical correlations between spatially separated regions.

**Image splicing** inserts content from a different photograph. The boundary between genuine and inserted content creates measurable statistical discontinuities in pixel-level distributions, lighting gradients, and shadow directions that the CNN is trained to identify.

### 3.3 Vision Transformers and Self-Attention for AI Image Detection

**Why CNNs are insufficient:** CNNs process images through local receptive fields and cannot easily detect semantic inconsistencies spanning non-adjacent image regions — for example, a barcode that does not match the brand name shown elsewhere on the same packaging.

**How Vision Transformers solve this:** ViTs divide the image into a grid of 16×16 pixel patches and process them as a sequence using self-attention. The attention mechanism computes relationships between every patch and every other patch simultaneously, enabling detection of long-range semantic inconsistencies. The attention formula is:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

where Q, K, V are query, key, and value projections of each patch embedding.

**CLIP embeddings:** CLIP trains a vision encoder and a text encoder jointly on aligned image-text pairs. The resulting visual embeddings encode semantic meaning at a higher abstraction level and preserve GAN-specific statistical fingerprints that persist even after post-processing. CLIP-ResNet50 achieves 99.4% accuracy on StyleGAN detection (CVPR 2023).

### 3.4 Frequency Domain GAN Artifact Analysis

**Core idea:** GAN generators rely on transposed convolution upsampling, which introduces characteristic periodic artifacts in the frequency domain — specifically, spectral replication patterns appearing as repeating peaks in the 2D Fourier spectrum.

**Checkerboard artifacts:** Transposed convolution upsampling produces a characteristic checkerboard pattern at the pixel level. While often invisible spatially, these patterns are clearly identifiable in the frequency domain and serve as reliable indicators of GAN synthesis.

**Re-synthesis detection `[INFERRED]`:** The input image is passed through auxiliary networks (super-resolution, denoising, colorization) and the outputs are compared with the original. Genuinely photographed regions reconstruct differently from GAN-synthesized regions because their underlying statistics differ.

### 3.5 OCR-Assisted Semantic Verification

A digitally manipulated expiry date introduces a change in the visual texture of the text region. Physically printed text has natural ink variation and substrate interaction; digitally inserted text has pixel-perfect edges. VeriSight combines two analyses:

1. **Content plausibility:** The extracted date is compared against the platform's order and delivery timestamps. Dates in the distant past relative to delivery are flagged.
2. **Texture forensics:** The CNN analyzes the text region's visual texture to determine whether text is physically printed or digitally inserted.

This dual design means OCR failures do not drive incorrect decisions — text plausibility is only one input in the weighted score.

### 3.6 Weighted Evidence Fusion

The core scoring principle is that different manipulation types leave different forensic traces, and no single detector can catch all of them reliably. Four independent detectors run in parallel, and their outputs are fused through a weighted linear combination:

```
Score = (CNN_score × 0.35) + (ViT_score × 0.30) + (GAN_score × 0.20) + (OCR_score × 0.15)
```

Each sub-score is in [0, 100] representing the probability the image is genuine (100 = certainly genuine, 0 = certainly manipulated). CNN forensics receives the highest weight (35%) as the most established forensic technique.

---

## 4. System Architecture

### 4.1 High-Level Architecture Overview

VeriSight is structured as a **cloud-ready, API-first microservice** with a clear separation between the intake gateway, the parallel analysis layer, the score engine, and the storage and presentation layers.

```mermaid
graph TD
    A[Customer Submits Complaint<br/>Product Image + Order Metadata] --> B[Delivery Platform Backend]
    B --> C[VeriSight REST API Gateway<br/>FastAPI + JWT Auth + Pydantic]
    C --> D[Celery Task Queue<br/>Redis Broker]

    D --> E[Module 1: CNN Image Forensics<br/>EfficientNet-B4 + GCA-Net + ELA<br/>Weight: 35%]
    D --> F[Module 2: Transformer AI Detection<br/>ViT-B/16 + CLIP-ResNet50 + FatFormer<br/>Weight: 30%]
    D --> G[Module 3: GAN Artifact Analysis<br/>AutoGAN Spectrum + FFT<br/>Weight: 20%]
    D --> H[Module 4: OCR Text Verification<br/>YOLOv8 + EasyOCR + Regex + CNN Texture<br/>Weight: 15%]

    E --> I[Score Engine]
    F --> I
    G --> I
    H --> I

    I --> J[XAI Layer<br/>SHAP + Grad-CAM + GCA-Net Overlay]
    J --> K[Final JSON Response<br/>Score + Decision + Heatmap + Report]

    K --> L[Platform API — Automated Action]
    K --> M[PostgreSQL Audit Log]
    K --> N[React.js Admin Dashboard]
```

### 4.2 Processing Pipeline — Step by Step

1. Customer submits a complaint with a product image via the platform's app.
2. The platform backend forwards the image and complaint metadata (order ID, account ID, order date, delivery timestamp) to VeriSight via HTTP POST.
3. The API Gateway authenticates via JWT, validates the payload with Pydantic, and dispatches a Celery task to the Redis broker.
4. Celery workers route the image simultaneously to all four analysis modules, executing them in parallel.
5. Module 1 (CNN) returns: forgery probability + ELA heatmap + GCA-Net pixel manipulation map.
6. Module 2 (Transformer) returns: AI-generation probability + Grad-CAM heatmap + CLIP score.
7. Module 3 (GAN) returns: GAN artifact probability + spectral analysis result + boundary score.
8. Module 4 (OCR) returns: extracted text + date plausibility flag + text texture forensics score.
9. Score Engine aggregates the four sub-scores into the Authenticity Confidence Score.
10. XAI layer generates SHAP attribution values and composite heatmap overlay.
11. Final JSON response is returned, stored in PostgreSQL, and reflected in the admin dashboard.
12. Platform executes the recommended action.

### 4.3 Module Responsibilities

| Module | Primary Responsibility | Fraud Vectors Targeted |
|---|---|---|
| **CNN Image Forensics** | Pixel-level forgery detection | Copy-move, splicing, compositing, ELA anomalies |
| **Transformer AI Detection** | Whole-image and patch-level synthesis detection | Fully AI-generated images, diffusion model outputs |
| **GAN Artifact Analysis** | Frequency-domain localized GAN edit detection | GAN inpainting, checkerboard artifacts, spectral replication |
| **OCR Text Verification** | Semantic content plausibility and text texture forensics | Expiry date manipulation, label text editing |
| **Score Engine** | Weighted evidence fusion + decision mapping | All types (aggregation) |
| **XAI Layer** | Human-interpretable evidence generation | All types (explainability overlay) |

### 4.4 Data Flow Diagram

```mermaid
graph LR
    IMG[Raw Complaint Image] --> PRE[Image Preprocessor<br/>OpenCV + Pillow]
    PRE --> RGB[RGB Tensor 224x224x3]
    PRE --> ELA[ELA Map 224x224x3]
    PRE --> GRAY[Grayscale Enhanced for OCR]
    PRE --> FFT[FFT Magnitude Spectrum]

    RGB --> M1[CNN Module<br/>EfficientNet-B4 + GCA-Net]
    ELA --> M1
    RGB --> M2[ViT Module<br/>ViT-B/16 + CLIP]
    RGB --> M3[GAN Module<br/>AutoGAN Classifier]
    FFT --> M3
    GRAY --> M4[OCR Pipeline<br/>YOLOv8 + EasyOCR]

    M1 --> SE[Score Engine]
    M2 --> SE
    M3 --> SE
    M4 --> SE

    SE --> SHAP[SHAP Attribution]
    SE --> GCAM[Grad-CAM + GCA-Net]
    SHAP --> OUT[JSON Response]
    GCAM --> OUT
```

### 4.5 Decision Threshold Mapping

| Score Range | Classification | Platform Action | SLA |
|---|---|---|---|
| 85–100 | GENUINE | Auto-approve refund | < 60 seconds |
| 60–84 | LIKELY_GENUINE | Fast-track human spot-check | < 2 hours |
| 35–59 | SUSPICIOUS | Hold — request additional evidence | Customer notified |
| 0–34 | MANIPULATED | Reject refund + flag account | Evidence report generated |

---

## 5. Algorithmic and Model Design

### 5.1 Module 1 — CNN Image Forensics

**Architecture:** EfficientNet-B4 is selected as the backbone over ResNet-152 and VGG-19 due to its superior accuracy-to-computation ratio: 7.6× smaller and 5.7× faster than ResNet-152 while achieving higher accuracy on forgery benchmarks. The model is modified to accept **six input channels** — three RGB channels concatenated with three ELA map channels — giving the model simultaneous spatial and compression-artifact views.

**GCA-Net** (Gated Context Attention Network) is attached as a segmentation head, producing a pixel-level manipulation probability map that forms the visual heatmap overlay shown to human reviewers.

**Training approach:** Pre-trained on ImageNet, then fine-tuned in two stages: first on CASIA 2.0 (general forgery detection), then on the custom VeriSight product complaint dataset. Uses cross-entropy loss for the binary classification head and pixel-level binary cross-entropy for the GCA-Net localization head.

**Inference output:** Forgery probability in [0, 1] (converted to sub-score), GCA-Net pixel mask, and ELA heatmap.

### 5.2 Module 2 — Transformer-Based AI-Generated Image Detection

**ViT-B/16:** Divides the image into 196 non-overlapping 16×16 patches. Each patch is linearly projected into an embedding vector. A learnable classification token is prepended, and the sequence is processed through 12 transformer encoder layers with multi-head self-attention. This allows every patch to attend to every other patch, detecting cross-region semantic inconsistencies characteristic of AI-generated images.

**CLIP-ResNet50:** The CLIP visual encoder extracts a semantic embedding. A lightweight classification head trained on top of frozen embeddings distinguishes genuine from GAN-generated images, leveraging GAN-specific statistical fingerprints preserved in the embedding space.

**FatFormer:** The Forgery-Aware Adapter integrated into the ViT backbone, combined with Language-Guided Alignment from CLIP text embeddings, enables generalization to unseen generative models — critical because new AI tools emerge continuously.

**Inference output:** AI-generation probability in [0, 1], Grad-CAM heatmap, and CLIP embedding distance metric.

### 5.3 Module 3 — GAN Artifact and Texture Inconsistency Analysis

**AutoGAN spectrum classification:** The complaint image is converted to grayscale and a 2D Fast Fourier Transform is applied. The log-magnitude spectrum is passed through a compact convolutional classifier. GAN-generated or inpainted images exhibit characteristic spectral replication patterns — peaks at regular intervals — absent from genuine photographs.

**Color channel statistical analysis:** Per-channel mean, variance, and cross-channel correlation are computed for localized regions. GAN-generated regions exhibit subtle but measurable statistical differences compared to surrounding genuine content.

**Boundary blending analysis:** Even sophisticated GAN inpainting leaves statistical traces at the boundary between genuine and generated regions, detectable through local pixel distribution analysis.

**Inference output:** GAN artifact probability in [0, 1], spectral replication score, and boundary blending anomaly score.

### 5.4 Module 4 — OCR-Assisted Text and Expiry Date Verification

The OCR pipeline proceeds through six sequential stages:

**Stage 1 — Preprocessing:** Grayscale conversion → CLAHE contrast enhancement → adaptive thresholding → denoising. This maximizes OCR accuracy on varied real-world packaging surfaces.

**Stage 2 — Text Region Detection:** YOLOv8 (or DBNet with CBAM attention module) localizes all text regions. This is critical because targeted OCR of the relevant region is far more accurate than full-image OCR on complex packaging layouts.

**Stage 3 — Text Recognition:** EasyOCR (CRNN + CRAFT architecture) serves as the primary engine. PaddleOCR is the fallback for dot-matrix printed expiry dates, which are among the most challenging text types in this domain.

**Stage 4 — Date Extraction:** Regex pattern matching extracts all date-format strings across multiple formats (DD/MM/YY, DD.MM.YYYY, MM-YYYY, text-based formats like "JAN 2025"). The most likely expiry date candidate is selected.

**Stage 5 — Plausibility Analysis:** The extracted date is cross-referenced against the platform's order and delivery timestamps. Dates implausibly in the past relative to delivery are flagged as suspicious.

**Stage 6 — Text Texture Forensics:** The localized text region is passed to the CNN for texture analysis. Physically printed text exhibits natural ink variation and substrate interaction; digitally inserted text exhibits pixel-perfect edge regularity detectable at the feature level.

**Critical design constraint:** OCR contributes only 15% to the final score. OCR failures do not drive incorrect decisions.

### 5.5 Score Engine — Weighted Fusion and XAI

**Weighted score computation:** Each module returns a probability of genuineness in [0, 1], multiplied by 100 and by its module weight, then summed and clamped to [0, 100].

**SHAP attribution:** SHAP decomposes the final score into contributions from each module's features, producing a human-readable explanation of which signals drove the authenticity decision.

**Grad-CAM + GCA-Net heatmap overlay:** Gradient-weighted Class Activation Maps from CNN and ViT modules, combined with GCA-Net pixel masks, are composited onto the original complaint image, showing reviewers exactly which regions triggered the detection.

---

## 6. Dataset Strategy

### 6.1 Public Benchmark Datasets

| Dataset | Size | Primary Use | Source |
|---|---|---|---|
| **CASIA 2.0** | 12,614 images | CNN Module — primary training (splicing + copy-move) | IEEE Dataport |
| **CoMoFoD** | 260 image sets | CNN robustness — copy-move with post-processing | vcl.fer.hr |
| **MICC-F220** | 220 images + masks | GCA-Net pixel-level localization ground truth | micc.unifi.it |
| **DF2023** | 1M+ images | CNN large-scale generalization training | arXiv / GitHub |
| **ImagiNet** | 200,000 images | ViT Module — real vs AI-generated classification | GitHub (Boychev et al.) |
| **FaceForensics++** | 1,000 videos | GAN Module — deepfake and GAN detection pre-training | FaceForensics GitHub |

### 6.2 Custom VeriSight Product Complaint Dataset

No public dataset exists for product packaging image manipulation in the refund fraud context. The custom dataset is constructed as follows:

**Physical data collection:** Photograph 150+ real product packages across six categories (snacks, beverages, dairy, medicines, cosmetics, grocery) at multiple angles, distances, and lighting conditions.

**Manipulation class construction:**

| Class | Construction Method | Tools Used |
|---|---|---|
| Genuine | Photograph at slightly poor angle / soft lighting | Camera |
| Expiry-Manipulated | Digitally alter printed expiry date to a past date | Photoshop, Canva, SD Inpaint |
| Label-Manipulated | Alter brand name, MRP, weight, batch number | Photoshop, Canva |
| AI-Generated | Generate entirely synthetic complaint photos | Stable Diffusion, DALL-E 3, Adobe Firefly |
| GAN-Inpainted | Insert realistic damage / wrong labeling into genuine photos | SD Inpaint, GAN tools |

**Augmentation:** Rotation (±15°), brightness variation, JPEG re-compression (quality 60–95%), Gaussian noise.

**Labeling:** Each image is labeled with its manipulation type (class label), a pixel-level manipulation mask for the altered region (required for GCA-Net training), the tool used, and a label confidence level.

**Target scale:** 4,000+ labeled product complaint images across all five classes.

### 6.3 Dataset Directory Structure

```
data/
├── public/
│   ├── casia2/
│   │   ├── Au/               # Authentic images
│   │   ├── Tp/               # Tampered images
│   │   └── Gt/               # Ground truth masks
│   ├── imaginet/
│   │   ├── real/
│   │   └── ai_generated/
│   └── df2023/               # Large-scale forgery subset
│
├── custom/
│   ├── genuine/
│   ├── expiry_manipulated/
│   ├── label_manipulated/
│   ├── ai_generated/
│   ├── gan_inpainted/
│   └── masks/                # Pixel-level manipulation masks
│
└── splits/
    ├── train.csv             # 70% stratified
    ├── val.csv               # 15% stratified
    └── test.csv              # 15% stratified
```

**Split CSV format:** `image_path, label, manipulation_type, manipulation_tool, mask_path, split`

### 6.4 Data Quality Considerations

- Maintain **class balance** — imbalance toward genuine images causes under-prediction of fraud.
- Ensure **diversity of packaging types** — different font styles, label layouts, and printing methods (laser-etched, dot-matrix, inkjet) across categories.
- Apply **stratified splitting** — all manipulation tools and product categories must be represented in all three splits.
- Do **not** apply JPEG re-compression augmentation to images used for ELA training, as it alters the ELA signal.

---

## 7. Technology Stack

### 7.1 Programming Languages

| Language | Purpose |
|---|---|
| **Python 3.11** | All AI/ML models, backend API, preprocessing, OCR pipeline |
| **JavaScript (ES2023)** | React.js admin dashboard frontend |
| **SQL** | PostgreSQL schema and queries |
| **Bash / Shell** | Deployment scripts, CI/CD automation |

### 7.2 AI and ML Frameworks

| Library | Recommended Version | Role |
|---|---|---|
| **PyTorch** | 2.2+ | Primary deep learning framework (all models) |
| **torchvision** | 0.17+ | EfficientNet-B4, image transforms |
| **HuggingFace Transformers** | 4.40+ | ViT-B/16 pretrained model and fine-tuning |
| **OpenAI CLIP** | Latest | CLIP-ResNet50 for GAN fingerprinting |
| **EasyOCR** | 1.7+ | Primary OCR engine (CRNN + CRAFT) |
| **PaddleOCR** | 2.7+ | Fallback OCR for dot-matrix text |
| **Ultralytics YOLOv8** | 8.2+ | Text region detection on packaging |
| **OpenCV** | 4.9+ | ELA map generation, image preprocessing |
| **Pillow** | 10.3+ | Image I/O, heatmap overlay rendering |
| **NumPy + SciPy** | 1.26+ / 1.13+ | FFT spectrum analysis, array operations |
| **SHAP** | 0.45+ | Feature-level score attribution |
| **grad-cam** | 1.5+ | Grad-CAM and GCA-Net heatmap generation |
| **ONNX Runtime** | 1.18+ | Optimized production model inference |
| **scikit-learn** | 1.4+ | Evaluation metrics, train/val splits |
| **PyWavelets** | 1.5+ | Discrete Wavelet Transform for frequency analysis |

### 7.3 Backend and Infrastructure

| Technology | Role |
|---|---|
| **FastAPI** | Async REST API framework |
| **Pydantic v2** | Request and response schema validation |
| **Celery** | Distributed async task queue for parallel module execution |
| **Redis** | Celery message broker + model and feature caching |
| **PostgreSQL** | Persistent storage — complaints, scores, fraud flags, audit log |
| **SQLAlchemy** | ORM for database model management |
| **Alembic** | Database schema migration management |
| **Uvicorn** | ASGI server for FastAPI |

### 7.4 Frontend and DevOps

| Technology | Role |
|---|---|
| **React.js** | Admin monitoring dashboard |
| **Recharts** | Analytics charts and real-time score feeds |
| **Leaflet.js** | Geographic fraud pattern visualization |
| **Docker** | Containerization of all services |
| **Kubernetes** | Container orchestration and horizontal scaling |
| **GitHub Actions** | CI/CD pipeline — test, build, deploy |
| **AWS EC2 / GCP** | Cloud compute for training and inference |

### 7.5 Hardware Requirements

**Development:** 8-core CPU, 32 GB RAM, NVIDIA GPU ≥8 GB VRAM (RTX 3070 or better), 500 GB SSD.

**Model Training:** NVIDIA A100 (40 GB) or RTX 4090 (24 GB), 64 GB RAM, 2 TB NVMe SSD, CUDA 12.1+.

**Production Inference (per API node):** NVIDIA T4 (16 GB) for sub-5-second latency. Alternatively, ONNX Runtime with INT8 quantization on CPU for early deployment phases.

---

## 8. Environment Setup

### 8.1 System-Level Dependencies

```bash
# Ubuntu 22.04 LTS recommended
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential git wget curl \
    libopencv-dev libpq-dev redis-server \
    postgresql postgresql-contrib docker.io docker-compose
```

### 8.2 Python Environment

```bash
# Create isolated virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade package management tools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install OpenAI CLIP (no PyPI package — install from source)
pip install git+https://github.com/openai/CLIP.git

# Install all remaining dependencies
pip install -r requirements.txt
```

**Core `requirements.txt`:**
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
python-jose[cryptography]==3.3.0
celery[redis]==5.4.0
redis==5.0.4
sqlalchemy==2.0.30
psycopg2-binary==2.9.9
alembic==1.13.1
transformers==4.40.0
easyocr==1.7.1
paddleocr==2.7.3
ultralytics==8.2.0
opencv-python==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
scipy==1.13.0
shap==0.45.1
grad-cam==1.5.3
pywavelets==1.5.0
onnxruntime-gpu==1.18.0
scikit-learn==1.4.2
python-multipart==0.0.9
aiofiles==23.2.1
```

### 8.3 Database Configuration

```bash
# Initialize PostgreSQL database
sudo -u postgres psql -c "CREATE DATABASE verisight;"
sudo -u postgres psql -c "CREATE USER verisight_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE verisight TO verisight_user;"

# Run Alembic migrations
alembic upgrade head
```

### 8.4 Environment Variables

Create a `.env` file at the project root:

```ini
DATABASE_URL=postgresql://verisight_user:your_password@localhost:5432/verisight
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=replace-with-strong-secret
JWT_ALGORITHM=HS256
MODEL_CACHE_DIR=/models
MAX_IMAGE_SIZE_MB=10
PROCESSING_TIMEOUT_SECONDS=30
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 8.5 Service Startup

```bash
# Start Redis
sudo systemctl start redis-server

# Start Celery worker (with venv active)
celery -A verisight.celery_app worker --loglevel=info --concurrency=4

# Start FastAPI development server
uvicorn verisight.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. Step-by-Step Implementation Plan

### Phase 1 — Data Preparation

**Objective:** Acquire, construct, and validate all training datasets before any model work begins.

Begin by downloading all public benchmark datasets (CASIA 2.0, ImagiNet, CoMoFoD, MICC-F220, DF2023 subset) and organizing into the directory structure from Section 6.3. Verify integrity via checksums.

Construct the custom product complaint dataset by photographing real product packages across all six categories, then systematically creating all five manipulation classes using the specified tools. Each manipulated image must be accompanied by a pixel-level mask indicating the exact altered region — this ground truth is required for GCA-Net localization training.

Apply augmentation and generate ELA maps for all images at this stage, caching them alongside the originals. Generate stratified train/val/test split manifests (70/15/15 ratio).

**Deliverable:** Validated dataset splits with 4,000+ custom images, all public benchmark data organized, ELA maps pre-generated, and split CSV manifests complete.

---

### Phase 2 — Feature Extraction Pipeline

**Objective:** Implement all preprocessing and feature extraction utilities consumed by each module.

Implement the **ELA pipeline** as a standalone utility — load image, re-save at target quality, compute pixel-wise absolute difference, scale and return as a three-channel array. Validate visually on known manipulated images.

Implement the **FFT feature extractor** for the GAN module — convert to grayscale, apply 2D FFT, shift zero frequency to center, compute log-magnitude spectrum, normalize to [0, 1].

Implement the **OCR preprocessing pipeline** — grayscale conversion, CLAHE, adaptive thresholding, denoising. Validate OCR accuracy on a sample of product packaging images with known text ground truth.

Implement a **unified preprocessing function** that takes a raw complaint image and returns all derived representations (RGB tensor, ELA tensor, FFT spectrum, OCR-enhanced grayscale) ready for routing to each module.

**Deliverable:** A `preprocessing/` module with tested, modular functions for each feature type.

---

### Phase 3 — Model Development

**Objective:** Implement the four detection module architectures and verify they accept correct inputs and produce correct output shapes.

**Module 1 (CNN Forensics):** Load EfficientNet-B4 with pretrained ImageNet weights. Modify the first convolutional layer to accept six input channels (RGB + ELA). Attach the GCA-Net segmentation head. Verify forward pass produces classification logits and pixel mask.

**Module 2 (ViT AI Detection):** Load ViT-B/16 from HuggingFace. Replace the classification head with a two-class head. Load CLIP-ResNet50 separately with a lightweight two-class head on its frozen visual encoder. Verify both forward passes independently.

**Module 3 (GAN Spectrum):** Build the AutoGAN-style spectrum classifier — a compact CNN operating on the FFT magnitude spectrum. Implement color channel statistical analysis as a non-parametric feature extractor.

**Module 4 (OCR Pipeline):** Integrate YOLOv8 for text region detection. Set up EasyOCR and PaddleOCR. Implement regex date extraction across all date format variants. Implement date plausibility checker against order/delivery dates. Implement text texture forensics sub-module routing the localized text region to the CNN.

**Score Engine:** Implement the weighted aggregation formula, threshold mapping, SHAP computation, and Grad-CAM/GCA-Net overlay generation.

**Deliverable:** All model classes and the score engine implemented, passing unit tests with dummy inputs.

---

### Phase 4 — Training Pipeline

**Objective:** Train and fine-tune all models to achieve accuracy targets above baseline thresholds.

Implement a shared training harness with: dataset loading, data augmentation, training loop with AdamW optimizer and cosine annealing LR schedule, validation loop, model checkpointing on best validation accuracy, and training progress logging.

**Training sequence:**
1. Pre-train CNN Module on CASIA 2.0 + CoMoFoD (general forgery) → fine-tune on custom product dataset.
2. Fine-tune ViT Module on ImagiNet (real vs AI-generated) → further fine-tune on custom AI-generated and GAN-inpainted product images.
3. Train CLIP classification head on ImagiNet GAN subset + FaceForensics++ frames.
4. Train AutoGAN spectrum classifier on GAN vs genuine pairs.
5. Fine-tune EasyOCR on custom product packaging OCR dataset with expiry date ground truth.

**Deliverable:** Saved model checkpoints for all modules, training curves, and benchmarking report showing per-module accuracy on held-out test sets.

---

### Phase 5 — API and Integration Layer

**Objective:** Build the FastAPI backend, Celery parallel task queue, and complete end-to-end inference pipeline.

Implement the FastAPI application with the complaint intake endpoint (`POST /api/v1/verify`), accepting a multipart form with the image file and JSON complaint metadata.

Implement Celery task definitions — one task per module — each loading their respective ONNX-exported model and returning a structured result dictionary.

Implement the Score Engine aggregation task collecting results from all four module tasks and computing the final score, decision, and XAI outputs.

Export all trained PyTorch models to ONNX format and validate that ONNX Runtime produces identical results to PyTorch inference.

**Deliverable:** Running FastAPI application accepting complaint images, executing all four modules in parallel, returning complete JSON response in under 5 seconds on GPU.

---

### Phase 6 — Dashboard and Demo Preparation

**Objective:** Build the admin monitoring dashboard and validate end-to-end demonstration scenarios.

Build the React.js admin dashboard with: real-time score feed, complaint review queue highlighting SUSPICIOUS and MANIPULATED cases, heatmap overlay viewer displaying the composite Grad-CAM + GCA-Net visualization, per-module sub-score breakdown, and account-level fraud analytics charts.

Prepare a demonstration dataset of paired genuine and manipulated images across all five fraud types. Validate that genuine images consistently score 85+ and manipulated images score below 35.

**Deliverable:** Running admin dashboard connected to the live API, demonstration dataset ready, end-to-end test passing.

---

## 10. Evaluation Methodology

### 10.1 Per-Module Evaluation Targets

| Module | Primary Metric | Target | Baseline Reference |
|---|---|---|---|
| CNN Forensics | Binary accuracy (genuine/manipulated) | > 93% | ResNet-101: 93.46% on CASIA 2.0 (IEEE 2024) |
| ViT AI Detection | Binary accuracy (real/AI-generated) | > 95% | ViT literature: up to 96% |
| CLIP GAN Fingerprint | Binary accuracy (genuine/GAN-generated) | > 99% | CLIP-ResNet50: 99.4% on StyleGAN (CVPR 2023) |
| OCR Date Extraction | F1 Score | > 88% | R-CNN: 80.9% F1 on pharmaceutical packaging |

### 10.2 System-Level Evaluation Metrics

| Metric | Target | Rationale |
|---|---|---|
| **Overall Precision** | > 91% | Minimizes false positives — protects genuine customers from unjust rejections |
| **Overall Recall** | > 89% | Maximizes fraud detection — ensures manipulated images are not approved |
| **End-to-End Latency** | < 5 seconds | Real-time usability — enables automated platform decisions |
| **F1 Score (macro)** | > 90% | Balanced view across all five manipulation classes |

### 10.3 Validation Strategy

- **Train/validation/test split:** 70/15/15 stratified across all classes and dataset sources.
- **Cross-dataset evaluation `[INFERRED]`:** Test the CNN module trained on CASIA 2.0 on the custom product dataset without fine-tuning to quantify the domain gap and justify domain-specific fine-tuning.
- **Cross-model generalization:** Test the ViT module on AI-generated images from generative models not seen during training to measure out-of-distribution generalization.

### 10.4 Baseline Comparison

VeriSight must be compared against:
1. **Random baseline** — score distribution from a uniformly random classifier.
2. **Hash-based duplicate detection** — current industry-standard approach.
3. **Single-module ablations** — each of the four modules run independently, validating that four-layer fusion outperforms any individual module.
4. **Published benchmarks** — ResNet-101 at 93.46% (CNN baseline), ViT literature at 96% (transformer baseline).

### 10.5 Experiment Design

For each evaluation: report mean and standard deviation across three random seeds; use macro-averaged precision, recall, and F1; report confusion matrices to identify which manipulation types are most commonly confused with genuine images; report latency distribution (p50, p90, p99) under simulated concurrent load.

---

## 11. Testing Strategy

### 11.1 Unit Testing

Each component requires independent unit tests:

- **Preprocessing functions:** Validate ELA maps are statistically distinguishable on known manipulated vs genuine images. Validate FFT spectrum shapes and value ranges. Validate CLAHE preserves text readability.
- **OCR pipeline:** Validate date extraction against ground-truth packaging images. Validate all regex date format patterns. Validate plausibility checker correctly flags past dates relative to a known order date.
- **Score engine:** Validate weighted aggregation produces correct numerical outputs. Validate threshold mapping produces correct decision labels for boundary values (exactly 35, 60, 85).
- **Individual model forward passes:** Validate all models accept correctly shaped inputs and produce correctly shaped outputs.

### 11.2 Integration Testing

- **API endpoint testing:** Submit genuine and manipulated test images via `/verify` and validate the response JSON contains all required fields with correct types and value ranges.
- **Parallel execution testing:** Verify all four Celery tasks execute concurrently and results are correctly aggregated.
- **Database write testing:** Verify every API call results in a correct record written to PostgreSQL.
- **Latency testing:** Submit batches of 10, 50, and 100 concurrent requests and measure end-to-end latency.

### 11.3 Model Validation

- **Regression testing:** After any model retraining, run the full held-out test set and confirm no accuracy degradation relative to the recorded baseline.
- **Adversarial robustness `[INFERRED]`:** Apply minor perturbations (JPEG re-compression, rotation, brightness shift) to known manipulated images and verify the system continues to classify them correctly. These are realistic attacks a fraudster might use.

### 11.4 Performance Testing

- Measure GPU utilization and memory consumption during peak concurrent load.
- Validate ONNX Runtime inference matches PyTorch inference within a tolerance of ±0.001 on sub-scores.
- Confirm p99 end-to-end latency remains below 5 seconds under realistic request rates.

---

## 12. Limitations

### 12.1 Dataset Limitations

- **No public product-specific dataset exists.** The custom dataset must be constructed from scratch. Its quality, diversity, and scale directly determine real-world performance. 4,000 images is small relative to production-scale forgery benchmarks.
- **Custom dataset bias:** Images photographed by the team may not represent the full diversity of real complaint images submitted across millions of orders on live platforms.

### 12.2 OCR Limitations

- OCR accuracy degrades significantly on poorly lit, damaged, or low-resolution packaging photographs — which are common in real-world complaint submissions.
- Thermally printed dot-matrix expiry dates on dairy and pharmaceutical products are among the most challenging text types for general-purpose OCR engines.
- OCR failures cause the OCR sub-score to default to a neutral value, reducing but not eliminating its contribution.

### 12.3 Model Limitations

- **Adversarial robustness:** A sophisticated adversary understanding VeriSight's detection mechanisms could craft manipulations designed to evade frequency-domain analysis or ELA detection.
- **New generative models:** New AI tools emerge continuously. While FatFormer provides generalization, models trained on today's generators may not perfectly detect future architectures.
- **False positive risk:** Genuine complaint images taken under poor conditions (heavy compression, extreme lighting, damaged packaging) may exhibit statistical properties resembling manipulated images.

### 12.4 Computational Limitations

- Sub-5-second latency requires GPU inference. CPU-only deployment with ONNX Runtime and INT8 quantization increases latency beyond 5 seconds at scale.
- Running four models in parallel requires approximately 8–12 GB VRAM for all models loaded simultaneously.

### 12.5 Calibration Limitations

- Score thresholds (35, 60, 85) are initial defaults. Optimal thresholds differ across platforms depending on their specific fraud rates, product categories, and acceptable false positive tolerances.

---

## 13. Risk Analysis

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **Adversarial adaptation** — fraudsters learn to evade specific detectors | High | Medium | Four-layer architecture increases the attack surface; continuous retraining on new fraud patterns; monitor score distribution shifts |
| **OCR failure on challenging packaging** | Medium | High | PaddleOCR fallback; CLAHE preprocessing; OCR weight limited to 15% |
| **New generative AI models** outpacing detection benchmarks | High | High | FatFormer generalization module; scheduled retraining cadence; monitor false negative rates |
| **False positives on genuine customers** | High | Low–Medium | Human reviewer override for scores 35–84; SHAP report enables transparent auditing and appeals |
| **GPU dependency for latency target** | Medium | High | ONNX Runtime INT8 quantization for CPU fallback; Kubernetes horizontal GPU scaling |
| **Dataset class imbalance** | Medium | Medium | Stratified sampling; augmentation; class-weighted loss functions during training |
| **Training instability** | Low | Low | Cosine annealing LR schedule; gradient clipping; early stopping; checkpoint on validation improvement |
| **Score threshold miscalibration** | High | Medium | Per-platform threshold tuning; monitoring dashboards for false positive/negative rates |
| **Model serving cold-start latency** | Medium | Medium | Pre-load all models into GPU memory at service startup; Redis model cache |

---

## 14. Production-Grade System Design

### 14.1 Modular Code Architecture

The codebase must be strictly modular with independent encapsulation of each detection module. Each module must implement a standardized interface — `__init__(model_path)`, `preprocess(image)`, `infer(preprocessed)`, `postprocess(raw_output)` — allowing modules to be swapped or upgraded without affecting the Score Engine or API layer.

The Score Engine must be isolated from all model-specific logic, consuming only normalized sub-score outputs. This ensures weight adjustments or threshold changes require modifications to only one well-defined component.

### 14.2 API Layer Design

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/verify` | POST | Primary complaint image intake — accepts image + metadata, returns full score report |
| `/api/v1/score/{complaint_id}` | GET | Retrieve previously computed score and evidence report |
| `/api/v1/health` | GET | Service health check for load balancer probes |
| `/api/v1/metrics` | GET | Prometheus-format metrics endpoint `[INFERRED]` |

All endpoints require JWT authentication. Rate limiting is applied at the API Gateway level. Pydantic models enforce strict input validation.

### 14.3 Inference Pipeline Optimization

All PyTorch models must be exported to ONNX format and served via ONNX Runtime, which offers lower inference latency than native PyTorch on both GPU and CPU. Export must be validated — ONNX Runtime outputs must match PyTorch outputs within numerical tolerance.

Model quantization (INT8 dynamic quantization) should be applied to ONNX models intended for CPU-only deployment, reducing model size approximately 4× and inference time 2–3× with marginal accuracy loss.

All models must be loaded into GPU memory at service startup and kept resident — not loaded per-request — to eliminate cold-start latency from the request path.

### 14.4 Logging and Monitoring

**Structured logging:** All API requests and responses must be logged in structured JSON format, including complaint ID, account ID, per-module processing latency, final score, decision label, and error conditions.

**Metrics `[INFERRED]`:** Expose Prometheus metrics for request rate, per-module inference latency (histogram), score distribution, decision label distribution, error rate, GPU utilization, and queue depth.

**Audit trail:** Every complaint evaluation must be written to PostgreSQL with the full evidence report — scores, decision, OCR text, extracted date, heatmap path, SHAP report path. This constitutes the legal evidence record.

### 14.5 Scalability Considerations

VeriSight is designed for horizontal scaling. The API Gateway (FastAPI) is stateless and scales horizontally behind a load balancer. Celery Workers scale by adding nodes. Redis Cluster provides high-availability brokering. PostgreSQL read replicas offload dashboard queries from the primary write node. At very high volume, each of the four modules can be deployed as separate microservices `[INFERRED]`, enabling independent scaling.

---

## 15. Deployment Architecture

### 15.1 Containerization Strategy

Every service is containerized with Docker. The project ships a `docker-compose.yml` for local development and Kubernetes manifests for production.

**Service containers:** `verisight-api` (FastAPI), `verisight-worker` (Celery, one or more replicas on GPU nodes), `verisight-redis`, `verisight-db` (PostgreSQL), `verisight-dashboard` (React + Nginx).

### 15.2 Kubernetes Deployment Topology

```mermaid
graph TD
    LB[Cloud Load Balancer] --> API1[API Pod 1]
    LB --> API2[API Pod 2]
    LB --> APIN[API Pod N]

    API1 --> RQ[Redis Queue]
    API2 --> RQ
    APIN --> RQ

    RQ --> W1[Celery Worker Pod 1 — GPU Node]
    RQ --> W2[Celery Worker Pod 2 — GPU Node]
    RQ --> WN[Celery Worker Pod N — GPU Node]

    W1 --> DB[(PostgreSQL Primary)]
    W2 --> DB
    WN --> DB

    DB --> DASH[Admin Dashboard]
```

GPU nodes must be labeled in Kubernetes with the NVIDIA device plugin enabled. Celery worker pods are scheduled exclusively on GPU nodes via node selectors.

### 15.3 Model Storage and Serving

Trained ONNX models are stored in a cloud object store (AWS S3 or GCP Cloud Storage). At worker pod startup, models are downloaded and loaded into GPU memory. Model versions are referenced by version-tagged paths. Kubernetes ConfigMaps reference the current active model version for zero-downtime model updates.

### 15.4 CI/CD Pipeline

GitHub Actions implements the following pipeline on every push to the main branch:
1. Run unit and integration tests.
2. Build Docker images for all services.
3. Push images to container registry (tagged with commit SHA).
4. Deploy to staging and run smoke tests.
5. On manual approval, deploy to production via Kubernetes rolling update.

---

## 16. Experiment Tracking and Versioning

### 16.1 Model Versioning

Each model checkpoint is versioned using a naming convention including module name, training dataset version, training date, and validation accuracy. Example: `cnn_forensics_casia2v2_custom_v1_2026-03-15_acc94.2.pth`. The corresponding ONNX export carries the same version identifier.

### 16.2 Dataset Versioning

The custom product complaint dataset is version-controlled via a hash of the complete image manifest (file paths + labels). Any addition or label change creates a new dataset version. Public benchmark datasets are versioned by their official release identifiers.

### 16.3 Experiment Tracking Tools

**`[INFERRED]`** MLflow or Weights & Biases (W&B) are recommended. Every training run should log: hyperparameters, dataset version, per-epoch metrics, final test set performance, model checkpoint path, and hardware configuration. This enables complete reproducibility of any model version from its training run log alone.

### 16.4 Score Threshold Versioning

Threshold configurations (the 35/60/85 boundaries) are version-controlled separately from model weights in `configs/thresholds/`, as they may be adjusted for per-platform calibration without requiring model retraining.

---

## 17. Recommended Project Directory Structure

```
verisight/
│
├── data/                          # All dataset storage and manifests
│   ├── public/                    # Public benchmark datasets
│   ├── custom/                    # Custom product complaint dataset (5 classes + masks)
│   └── splits/                    # Train/val/test CSV manifests
│
├── preprocessing/                 # Feature extraction and preprocessing utilities
│   ├── ela.py                     # Error Level Analysis map generation
│   ├── fft.py                     # FFT spectrum feature extraction
│   ├── ocr_preprocess.py          # CLAHE, thresholding, denoising for OCR
│   └── transforms.py              # Unified preprocessing + augmentation pipeline
│
├── models/                        # Model architecture definitions only
│   ├── cnn_forensics.py           # EfficientNet-B4 + GCA-Net (Module 1)
│   ├── vit_detector.py            # ViT-B/16 + CLIP-ResNet50 (Module 2)
│   ├── gan_spectrum.py            # AutoGAN spectrum classifier (Module 3)
│   ├── ocr_pipeline.py            # YOLOv8 + EasyOCR + plausibility logic (Module 4)
│   └── score_engine.py            # Weighted fusion + SHAP + Grad-CAM
│
├── training/                      # Training scripts and hyperparameter configs
│   ├── train_cnn.py
│   ├── train_vit.py
│   ├── train_gan_classifier.py
│   ├── finetune_ocr.py
│   └── configs/                   # YAML training hyperparameter configs per module
│
├── evaluation/                    # Evaluation and benchmarking scripts
│   ├── evaluate_module.py         # Per-module evaluation (accuracy, F1, etc.)
│   ├── evaluate_system.py         # End-to-end system evaluation
│   ├── ablation_study.py          # Single-module vs. full-system ablations
│   └── benchmark_latency.py       # Latency benchmarking under concurrent load
│
├── api/                           # FastAPI application and Celery tasks
│   ├── main.py                    # FastAPI app initialization
│   ├── routers/
│   │   └── verify.py              # /api/v1/verify endpoint definition
│   ├── schemas/                   # Pydantic request/response models
│   ├── celery_app.py              # Celery application configuration
│   ├── tasks/                     # One task file per detection module
│   │   ├── task_cnn.py
│   │   ├── task_vit.py
│   │   ├── task_gan.py
│   │   └── task_ocr.py
│   └── db/                        # SQLAlchemy models and Alembic migrations
│       ├── models.py
│       └── migrations/
│
├── frontend/                      # React.js admin dashboard
│   ├── src/
│   │   ├── components/            # Dashboard, HeatmapViewer, ScoreFeed, ReviewQueue
│   │   └── App.js
│   └── package.json
│
├── deployment/                    # Docker and Kubernetes infrastructure configs
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── docker-compose.yml
│   └── k8s/
│       ├── api-deployment.yaml
│       ├── worker-deployment.yaml
│       ├── redis-deployment.yaml
│       └── postgres-deployment.yaml
│
├── tests/                         # All test suites
│   ├── unit/                      # Preprocessing, OCR, score engine unit tests
│   ├── integration/               # API endpoint and database integration tests
│   └── fixtures/                  # Test images — genuine and manipulated pairs
│
├── exports/                       # ONNX-exported model files (versioned)
│   ├── cnn_forensics_v1.onnx
│   ├── vit_detector_v1.onnx
│   ├── gan_spectrum_v1.onnx
│   └── ocr_text_cnn_v1.onnx
│
├── configs/                       # Non-model system configuration
│   ├── thresholds/                # Score threshold configs per platform
│   └── logging.yaml
│
├── docs/                          # Documentation
│   └── VeriSight_Technical_Blueprint.md
│
├── .env                           # Environment variables (not version controlled)
├── .gitignore
├── requirements.txt
└── README.md
```

**Directory purposes:**

- **`data/`** — All training, validation, and test data. Not committed to version control due to size; managed via DVC or cloud storage references.
- **`preprocessing/`** — Standalone, testable feature extraction utilities with no model dependencies. Imported by both training scripts and the API task layer.
- **`models/`** — Pure model architecture definitions. No training logic, no data loading. One file per module.
- **`training/`** — Training scripts that import from `models/` and `preprocessing/`. Each script is independently runnable for a single module.
- **`evaluation/`** — Isolated scripts loading checkpoints and running inference on test sets.
- **`api/`** — Complete FastAPI application including Celery task definitions and database ORM models.
- **`frontend/`** — Self-contained React.js application communicating with the FastAPI backend.
- **`deployment/`** — All infrastructure-as-code for containerized deployment.
- **`tests/`** — Comprehensive test suite organized by testing level.
- **`exports/`** — Versioned ONNX model files ready for production serving.
- **`configs/`** — Non-code configuration version-controlled separately from model weights.

---

## 18. Reproducibility Guide

The following checklist enables a developer or researcher to rebuild the entire VeriSight system from scratch:

### Environment Setup
- [ ] Clone the repository and create a Python 3.11 virtual environment.
- [ ] Install all dependencies from `requirements.txt`, including the CUDA-specific PyTorch wheel and CLIP from source.
- [ ] Configure all environment variables in `.env` per Section 8.4.
- [ ] Start PostgreSQL, run Alembic migrations (`alembic upgrade head`).
- [ ] Start Redis (`sudo systemctl start redis-server`).

### Dataset Acquisition
- [ ] Download CASIA 2.0, CoMoFoD, MICC-F220, ImagiNet, and a DF2023 subset from their respective sources.
- [ ] Organize into the directory structure specified in Section 6.3.
- [ ] Construct the custom product complaint dataset: photograph 150+ products, create all five manipulation classes with pixel-level masks.
- [ ] Apply augmentation, generate and cache ELA maps for all images.
- [ ] Produce stratified train/val/test split CSV manifests (70/15/15).

### Model Training
- [ ] Run `training/train_cnn.py` with CASIA 2.0 configuration → save checkpoint.
- [ ] Fine-tune CNN checkpoint on custom product dataset → save fine-tuned checkpoint.
- [ ] Run `training/train_vit.py` with ImagiNet configuration → save checkpoint.
- [ ] Fine-tune ViT on custom AI-generated product images → save checkpoint.
- [ ] Run `training/train_gan_classifier.py` → save checkpoint.
- [ ] Run `training/finetune_ocr.py` on custom packaging OCR dataset → save checkpoint.

### Model Export and Validation
- [ ] Export all trained models to ONNX format.
- [ ] Validate ONNX Runtime inference matches PyTorch inference (tolerance ≤ 0.001 on sub-scores).
- [ ] Place versioned ONNX files in the `exports/` directory.

### Evaluation
- [ ] Run `evaluation/evaluate_module.py` for each module on the held-out test set.
- [ ] Confirm all module accuracy targets from Section 10.1 are met.
- [ ] Run `evaluation/evaluate_system.py` for end-to-end precision, recall, F1.
- [ ] Run `evaluation/benchmark_latency.py` and confirm p99 < 5 seconds on GPU.

### API Deployment
- [ ] Start Celery workers pointing to the correct Redis URL and ONNX model paths.
- [ ] Start the FastAPI application with Uvicorn.
- [ ] Verify the `/api/v1/health` endpoint responds.
- [ ] Submit a test image to `/api/v1/verify` and confirm a well-formed JSON response.

### Dashboard
- [ ] Install frontend dependencies (`npm install`).
- [ ] Configure the dashboard API base URL to point to the FastAPI backend.
- [ ] Start the development server (`npm start`) and confirm live complaint scores display.

### Production Deployment
- [ ] Build Docker images for all services.
- [ ] Push images to the container registry.
- [ ] Apply Kubernetes manifests and verify all pods reach Running state.
- [ ] Run end-to-end smoke tests against the production cluster.
- [ ] Confirm monitoring metrics are flowing to the observability stack.

---

## 19. Future Improvements

### 19.1 Model and Detection Improvements

- **Video complaint evidence:** Extend VeriSight to analyze short video clips submitted as complaint evidence, detecting temporal inconsistencies (discontinuous motion, inconsistent lighting across frames) characteristic of AI-generated video.
- **Multimodal cross-referencing:** Cross-reference the complaint image against the platform's own product catalog database. Genuine products should have packaging matching catalog reference images; significant deviations would be flagged.
- **Continual learning pipeline `[INFERRED]`:** Implement an active learning system where human reviewer overrides on borderline cases (35–84 score range) create new training examples, continuously improving model accuracy on emerging fraud patterns.
- **Adversarial training:** Augment training with adversarially crafted examples — images specifically designed to evade the current detector — to harden the system against adaptive adversaries.

### 19.2 OCR Improvements

- **Multi-script support:** Extend the OCR pipeline to handle regional Indian scripts (Devanagari, Tamil, Telugu, Kannada) for locally manufactured products whose packaging contains expiry dates in regional languages.
- **LLM-assisted OCR correction `[INFERRED]`:** Use a small language model to post-process OCR output and correct common character-level recognition errors in date strings, improving date extraction F1 on degraded images.

### 19.3 Architecture and Scalability

- **Dedicated per-module microservices `[INFERRED]`:** At very high scale, decompose the four detection modules into independent microservices with their own REST APIs, enabling independent horizontal scaling of the most compute-intensive module.
- **Edge deployment:** Compress and quantize models for deployment on mobile or edge devices, enabling on-device preliminary screening before full server-side analysis.

### 19.4 Domain Extensions

- **Insurance claims fraud:** The same four-layer approach applies directly to fraudulent insurance claim images (vehicle damage photos, medical receipt images).
- **Warranty claim fraud:** Product photos submitted for warranty claims on electronics and appliances exhibit the same manipulation patterns.
- **B2B commerce fraud:** Enterprise purchasing platforms face similar fraudulent invoice image and goods-receipt manipulation.
- **Legal evidence authentication:** VeriSight's forensic methodology and SHAP-based evidence reports are directly applicable to authenticating digital photographs submitted as legal evidence in civil and criminal proceedings.

### 19.5 Research Directions

- **Cryptographic image provenance watermarking:** Investigate whether platforms can embed cryptographic watermarks in product images at the point of delivery, allowing VeriSight to verify image provenance in addition to analyzing manipulation.
- **Unified foundation model:** Rather than four separate models, investigate training a single large vision foundation model fine-tuned for multi-task forgery detection — jointly optimizing for ELA anomaly detection, AI generation detection, GAN artifact analysis, and OCR text forensics.
- **Causal attribution:** Move beyond correlational XAI (SHAP, Grad-CAM) toward causal attribution methods that identify which specific editing operations were applied to the image, supporting more precise legal evidence reports.

---

*End of VeriSight Technical Blueprint — Team return0 — Hack-Nocturne '26 / AiFi REVA National Level AI Hackathon*

---

> **Assumptions Summary:** Sections marked `[INFERRED]` contain technically consistent details implied by the documents but not explicitly stated. These include the re-synthesis detection implementation detail (Section 3.4), Prometheus metrics specifics (Section 14.4), the active learning pipeline (Section 19.1), LLM OCR correction (Section 19.2), per-module microservices architecture (Section 19.3), and federated calibration concepts. All other content is directly grounded in the three provided source documents.
