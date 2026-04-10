# VeriSight API Reference
### Frontend Developer Documentation · v2.0.0

---

## Overview

VeriSight is an image authenticity verification API powered by a multi-layer AI fusion engine (CNN + ViT + GAN + OCR). This document covers everything a frontend developer needs to integrate with the backend.

**Base URL:** `http://<your-server>/api/v1`  
**Content-Type for uploads:** `multipart/form-data`  
**Response format:** `application/json`

---

## ⚠️ Current Implementation Status

| Endpoint | Status | Notes |
|---|---|---|
| `POST /api/v1/verify` | ✅ Implemented | Core verification — use this |
| `GET /health` | ❌ Not implemented | Plan for fallback/polling |
| `GET /api/v1/status` | ❌ Not implemented | Stub in UI |
| `POST /api/v1/verify-batch` | ❌ Not implemented | Disable batch UI for now |
| `GET /api/v1/results/<id>` | ❌ Not implemented | No history feature yet |
| `GET /api/v1/config` | ❌ Not implemented | Hardcode thresholds in frontend |

> **Frontend Strategy:** Build around the single working endpoint. Gracefully disable or hide UI for unimplemented endpoints. Use loading states, not errors, when polling for readiness.

---

## Endpoint 1 — Image Verification

### `POST /api/v1/verify`

The **only active endpoint**. Submits an image for AI-powered authenticity analysis and returns a multi-layer verdict.

---

### Request

**Headers:**
```
Content-Type: multipart/form-data
```

**Form Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | `File` | ✅ Yes | Image to verify. Accepted: JPEG, PNG, WebP, BMP |
| `order_date` | `string` | ❌ Optional | Date the order was placed (any format) |
| `delivery_date` | `string` | ❌ Optional | Date of delivery |
| `mfg_date_claimed` | `string` | ❌ Optional | Manufacturer's claimed production date |

**Example — JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);
formData.append("order_date", "2024-11-01");
formData.append("delivery_date", "2024-11-15");
formData.append("mfg_date_claimed", "2024-09-20");

const response = await fetch("http://<server>/api/v1/verify", {
  method: "POST",
  body: formData,
  // Do NOT set Content-Type manually — browser sets boundary automatically
});

const result = await response.json();
```

**Example — Axios:**
```javascript
import axios from "axios";

const formData = new FormData();
formData.append("image", file);

const { data } = await axios.post("/api/v1/verify", formData, {
  headers: { "Content-Type": "multipart/form-data" },
  onUploadProgress: (e) => setProgress(Math.round((e.loaded / e.total) * 100)),
});
```

---

### Response

**HTTP 200 — Success**

```json
{
  "schema_version": "2.0.0",
  "authenticity_score": 82,
  "decision": "FAST_TRACK",
  "confidence": 0.91,
  "abstained": false,
  "fusion_strategy": "weighted_average",
  "meta_model_used": false,
  "early_exit_triggered": false,
  "processing_time_ms": 1340,

  "layer_scores": {
    "cnn": 85.2,
    "vit": 80.1,
    "gan": 79.4,
    "ocr": 84.0
  },

  "layer_reliabilities": {
    "cnn": 0.93,
    "vit": 0.88,
    "gan": 0.76,
    "ocr": 0.91
  },

  "effective_weights": {
    "cnn": 0.35,
    "vit": 0.30,
    "gan": 0.20,
    "ocr": 0.15
  },

  "layer_status": {
    "cnn": "loaded",
    "vit": "loaded",
    "gan": "loaded",
    "ocr": "loaded"
  },

  "layer_outputs": {
    "cnn": { ... },
    "vit": { ... },
    "gan": { ... },
    "ocr": { ... }
  },

  "available_layers": ["cnn", "vit", "gan", "ocr"]
}
```

---

### Response Field Reference

#### Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | `string` | API schema version (`"2.0.0"`) |
| `authenticity_score` | `number` | Overall score **0–100**. Higher = more authentic |
| `decision` | `string` | Final verdict — see Decision Types below |
| `confidence` | `number` | Model confidence **0.0–1.0** |
| `abstained` | `boolean` | `true` if model could not make a confident decision |
| `fusion_strategy` | `string` | How layers were combined (`"weighted_average"`) |
| `meta_model_used` | `boolean` | Whether a meta-model overrode the fusion |
| `early_exit_triggered` | `boolean` | Whether a fast decision path was taken |
| `processing_time_ms` | `number` | Server-side processing duration in milliseconds |
| `available_layers` | `string[]` | Layers that were active during this verification |

#### `layer_scores`

Individual AI layer predictions, each **0–100**.

| Key | Layer | What It Detects |
|---|---|---|
| `cnn` | Convolutional Neural Net | Low-level visual artifacts, pixel manipulation |
| `vit` | Vision Transformer | High-level semantic inconsistencies |
| `gan` | GAN Detector | AI-generated / deepfake patterns |
| `ocr` | OCR Analyzer | Text authenticity, date/label consistency |

#### `layer_reliabilities`

How much to trust each layer's output for this specific image (**0.0–1.0**).  
Use this to show confidence bars per layer.

#### `effective_weights`

The dynamic weight applied to each layer during fusion (sums to ~1.0).  
Show as a weight distribution visualization if desired.

#### `layer_status`

Whether each layer was active for this request.

| Value | Meaning |
|---|---|
| `"loaded"` | Layer ran successfully |
| `"unavailable"` | Layer not loaded — score may be absent |
| `"error"` | Layer crashed — treat score as unreliable |

#### `layer_outputs`

Detailed internal output per layer. Structure varies by layer. Use carefully — parse defensively.

---

### Decision Types

The `decision` field drives your primary UI state. Map it as follows:

| `decision` Value | Score Range (approx) | UI Treatment | Color |
|---|---|---|---|
| `AUTO_APPROVE` | 85–100 | ✅ Green badge, approve CTA | `#22c55e` |
| `FAST_TRACK` | 65–84 | 🟡 Yellow badge, minor review | `#f59e0b` |
| `SUSPICIOUS` | 40–64 | 🟠 Orange badge, flag for review | `#f97316` |
| `REJECT` | 0–39 | 🔴 Red badge, block / reject | `#ef4444` |

> **Note:** Thresholds are approximate. The backend determines the final decision — do not recalculate it on the frontend. Display `decision` as-is.

**Special case — `abstained: true`:**  
The model could not decide. Show a neutral grey "Inconclusive" state regardless of `decision`. Do not display a score — show `"—"` instead.

---

### Error Responses

| HTTP Status | When It Happens | Frontend Action |
|---|---|---|
| `400 Bad Request` | Missing image, wrong file type, malformed form | Show field-level validation error |
| `413 Payload Too Large` | File too big | Show "File too large" message |
| `422 Unprocessable Entity` | Image corrupted or unreadable | Show "Could not process image" |
| `500 Internal Server Error` | Model crash or unexpected error | Show generic error with retry button |
| `503 Service Unavailable` | Models still loading on startup | Show "System warming up" with polling |

**Error response shape (when available):**
```json
{
  "error": "string describing what went wrong",
  "detail": "optional additional info"
}
```

---

## Frontend Integration Patterns

### Upload Flow (Recommended UX)

```
User selects image
       ↓
Client-side validation (type, size < 10MB)
       ↓
Show upload progress bar (use axios onUploadProgress)
       ↓
POST /api/v1/verify
       ↓
Show "Analyzing..." skeleton/spinner (avg ~1–3s)
       ↓
Render result card with decision + score + layer breakdown
```

### Handling `abstained: true`

```javascript
if (result.abstained) {
  showState("INCONCLUSIVE");  // Grey, neutral UI
  hideScore();
  showMessage("The system could not make a confident determination.");
} else {
  showDecision(result.decision);
  showScore(result.authenticity_score);
}
```

### Displaying Layer Scores

```javascript
const layers = ["cnn", "vit", "gan", "ocr"];

layers.forEach(layer => {
  const score = result.layer_scores[layer];
  const reliability = result.layer_reliabilities[layer];
  const status = result.layer_status[layer];

  if (status !== "loaded") {
    renderLayerUnavailable(layer);
  } else {
    renderLayerBar(layer, score, reliability);
  }
});
```

### Confidence Display

```javascript
// Convert 0-1 confidence to percentage
const confidencePercent = Math.round(result.confidence * 100);
// e.g., 0.91 → "91% confident"
```

---

## File Constraints (Enforce on Frontend)

| Constraint | Value |
|---|---|
| Accepted types | `image/jpeg`, `image/png`, `image/webp`, `image/bmp` |
| Recommended max size | `10 MB` (confirm with backend team) |
| Min dimensions | No restriction (but tiny images reduce accuracy) |

**Client-side MIME check:**
```javascript
const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

function validateFile(file) {
  if (!ACCEPTED.includes(file.type)) {
    throw new Error("Unsupported file type. Use JPEG, PNG, WebP, or BMP.");
  }
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("File too large. Maximum size is 10MB.");
  }
}
```

---

## Planned Endpoints (Not Yet Available)

These should be **stubbed in the UI** but not actively called. Disable related features gracefully.

### `GET /health`
**Purpose:** Check server readiness before upload  
**Frontend stub:**
```javascript
// Call this before showing upload UI
// If endpoint doesn't exist, assume healthy and proceed
async function checkHealth() {
  try {
    const res = await fetch("/health");
    return res.ok;
  } catch {
    return true; // Fail open — let the verify call surface errors
  }
}
```

### `GET /api/v1/status`
**Purpose:** See which AI layers are currently loaded  
**Frontend stub:** Show all layers as "Loading..." until verify response confirms status

### `POST /api/v1/verify-batch`
**Purpose:** Upload multiple images at once  
**Frontend stub:** Disable batch upload button with tooltip: `"Batch processing coming soon"`

### `GET /api/v1/results/<image_id>`
**Purpose:** Retrieve cached results by ID  
**Frontend stub:** No history tab until implemented

### `GET /api/v1/config`
**Purpose:** Fetch decision thresholds dynamically  
**Frontend stub:** Hardcode decision labels and colors from the table above

---

## Hardcoded Config (Until `/api/v1/config` is ready)

```javascript
export const DECISION_CONFIG = {
  AUTO_APPROVE: {
    label: "Auto Approved",
    color: "#22c55e",
    bg: "#f0fdf4",
    icon: "✅",
    description: "Image appears authentic. No review needed.",
  },
  FAST_TRACK: {
    label: "Fast Track",
    color: "#f59e0b",
    bg: "#fffbeb",
    icon: "⚡",
    description: "Likely authentic. Minor review recommended.",
  },
  SUSPICIOUS: {
    label: "Suspicious",
    color: "#f97316",
    bg: "#fff7ed",
    icon: "⚠️",
    description: "Potential issues detected. Requires manual review.",
  },
  REJECT: {
    label: "Rejected",
    color: "#ef4444",
    bg: "#fef2f2",
    icon: "🚫",
    description: "Image flagged as inauthentic. Do not approve.",
  },
};

export const LAYER_LABELS = {
  cnn: "CNN · Visual Artifacts",
  vit: "ViT · Semantic Analysis",
  gan: "GAN · Deepfake Detection",
  ocr: "OCR · Text Verification",
};
```

---

## Quick Reference Cheatsheet

```
POST /api/v1/verify
  body: FormData { image: File, order_date?, delivery_date?, mfg_date_claimed? }

Response key fields:
  .authenticity_score   → 0–100 gauge
  .decision             → AUTO_APPROVE | FAST_TRACK | SUSPICIOUS | REJECT
  .confidence           → 0–1 (multiply by 100 for %)
  .abstained            → if true, show "Inconclusive", hide score
  .layer_scores         → { cnn, vit, gan, ocr } each 0–100
  .layer_reliabilities  → { cnn, vit, gan, ocr } each 0–1
  .layer_status         → check for "loaded" before rendering a layer
  .processing_time_ms   → show as "Processed in Xs"
```

---

*Last updated: April 2026 · VeriSight Backend Team*  
*Questions? Ping the backend team before assuming endpoint behavior.*
