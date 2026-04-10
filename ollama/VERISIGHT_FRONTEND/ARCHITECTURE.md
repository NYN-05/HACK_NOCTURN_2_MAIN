# VeriSight Frontend - Architecture & Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 VeriSight Frontend                       │
│              (React + TypeScript + Vite)                │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│           React Components (10 Total)                    │
│  ┌─ Header                ┌─ DecisionBadge              │
│  ├─ UploadSection         ├─ LayerAnalysis             │
│  ├─ ImagePreview          ├─ ConfidenceIndicator       │
│  ├─ FormFields            ├─ ErrorMessage              │
│  ├─ Footer                └─ LoadingSkeleton           │
│  └─ App (Orchestrator)                                  │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              Service Layer (Axios)                       │
│          API Client - VerificationAPI                   │
│  - HTTP header management                               │
│  - Request/response transformation                      │
│  - Error handling and mapping                           │
│  - Upload progress tracking                             │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│          VeriSight Backend API (v2.0.0)                 │
│        ┌─ POST /api/v1/verify                          │
│        ├─ GET /health (Stub)                           │
│        ├─ GET /api/v1/status (Stub)                    │
│        └─ Additional endpoints (Future)                │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### User Journey: Verification Flow

```
1. USER INTERACTION
   ↓
   User drags/clicks image
   ↓
2. FILE SELECTION
   ├─ UploadSection component triggered
   ├─ validateFile() checks type & size
   ├─ generatePreview() creates data URL
   └─ onFileSelect callback fired
   ↓
3. STATE UPDATE (App.tsx)
   ├─ uploadedFile = File object
   ├─ preview = Data URL string
   └─ Components re-render
   ↓
4. IMAGE DISPLAY
   ├─ ImagePreview component shows image
   ├─ FormFields component shows date inputs
   └─ User fills optional fields
   ↓
5. FORM SUBMISSION
   ├─ handleFormSubmit() called
   ├─ isAnalyzing = true (loading state)
   └─ API request sent
   ↓
6. API REQUEST
   ├─ verificationAPI.verifyImage() executes
   ├─ FormData created with image + dates
   ├─ POST /api/v1/verify sent
   ├─ Upload progress tracked
   └─ onUploadProgress callback updates UI
   ↓
7. LOADING STATE
   ├─ LoadingSkeleton displayed
   ├─ "Analyzing image..." message shown
   ├─ User waits 1-3 seconds
   └─ Server processes image
   ↓
8. RESPONSE RECEIVED
   ├─ Success: result state updated
   ├─ Error: error state updated
   └─ isAnalyzing = false
   ↓
9. RESULTS DISPLAY
   ├─ if error:
   │  └─ ErrorMessage component shown
   ├─ if success:
      ├─ DecisionBadge displays verdict
      ├─ ConfidenceIndicator shows confidence
      ├─ LayerAnalysis shows layer breakdown
      └─ "Analyze Another Image" button shown
   ↓
10. NEXT ACTION
    └─ User can retry (if error) or analyze another image
```

## Component Data Flow

### App.tsx (Orchestrator)

```
App Component State:
├─ uploadedFile: File | null
├─ preview: string | null
├─ uploadProgress: number
├─ isAnalyzing: boolean
├─ result: VerificationResponse | null
└─ error: string | null

Event Handlers:
├─ handleFileSelect(file, preview)
│  └─ Updates uploadedFile, preview
├─ handleFormSubmit(dates)
│  └─ Calls API, updates result/error
├─ handleRemoveImage()
│  └─ Resets all state
└─ handleRetry()
   └─ Resets error state
```

### Component Communication

```
App
├─ manages all state
├─ passes props down to children
├─ receives callbacks from children
│
├─ Header
│  └─ (no props needed)
│
├─ UploadSection
│  ├─ receives: onFileSelect, isLoading, uploadProgress
│  └─ calls: onFileSelect(file, preview)
│
├─ ImagePreview
│  ├─ receives: src, fileName, onRemove
│  └─ calls: onRemove()
│
├─ FormFields
│  ├─ receives: onSubmit, isLoading, previewImage
│  └─ calls: onSubmit({orderDate, deliveryDate, mfgDateClaimed})
│
├─ DecisionBadge
│  ├─ receives: decision, score, abstained
│  └─ no callbacks
│
├─ ConfidenceIndicator
│  ├─ receives: confidence
│  └─ no callbacks
│
├─ LayerAnalysis
│  ├─ receives: response
│  └─ no callbacks
│
├─ ErrorMessage
│  ├─ receives: message, onRetry
│  └─ calls: onRetry()
│
├─ LoadingSkeleton
│  └─ no props
│
└─ Footer
   └─ (no props needed)
```

## State Management Pattern

### Lifting State Up

All state is in `App.tsx`. This ensures:

1. Single source of truth
2. Easy debugging
3. Simple data flow
4. Predictable state changes

### State Updates Flow

```
User Action
    ↓
Event Handler (in App)
    ↓
Update State (setState)
    ↓
Component Re-render
    ↓
Props Updated
    ↓
Child Components Re-render
    ↓
UI Updates
```

## API Integration Pattern

### Request Flow

```
Component calls handleFormSubmit()
    ↓
App calls verificationAPI.verifyImage()
    ↓
Axios client:
├─ Creates FormData
├─ Sets headers
├─ Attaches interceptors
└─ Makes POST request
    ↓
Backend processes:
├─ Receives FormData
├─ Validates image
├─ Runs AI models (CNN+ViT+GAN+OCR)
├─ Calculates scores
└─ Returns response
    ↓
Axios receives response:
├─ Checks status (200 = success)
├─ Parses JSON
└─ Returns data
    ↓
App updates result state
    ↓
Components re-render with results
```

### Error Handling Pattern

```
API Request Made
    ↓
Network/Server Error Occurs
    ↓
Axios interceptor catches error:
├─ Check error type
├─ Check HTTP status
├─ Check response.data.error
└─ Generate user-friendly message
    ↓
Error thrown with message
    ↓
Catch block in handleFormSubmit:
├─ Extract error message
├─ Update error state
└─ Stop loading indicator
    ↓
App renders ErrorMessage component
    ↓
User sees error with suggestions
    ↓
User clicks retry
    ↓
handleRetry clears error and resets upload
```

## Configuration System

### Config File Structure

```
config.ts
├─ API_CONFIG
│  ├─ BASE_URL
│  ├─ VERIFY_ENDPOINT
│  ├─ TIMEOUT
│  └─ HEALTH_ENDPOINT
│
├─ FILE_CONFIG
│  ├─ MAX_SIZE
│  ├─ ACCEPTED_TYPES
│  └─ ACCEPTED_EXTENSIONS
│
├─ DECISION_CONFIG
│  ├─ AUTO_APPROVE: { label, color, bg, etc }
│  ├─ FAST_TRACK: { ... }
│  ├─ SUSPICIOUS: { ... }
│  └─ REJECT: { ... }
│
├─ LAYER_CONFIG
│  ├─ cnn: { name, description, icon }
│  ├─ vit: { ... }
│  ├─ gan: { ... }
│  └─ ocr: { ... }
│
└─ UI_CONFIG
   ├─ SKELETON_LOADING_TIME
   ├─ ANIMATION_DURATION
   └─ TOAST_DURATION
```

### Configuration Usage

```typescript
// In components:
import { DECISION_CONFIG, LAYER_CONFIG } from '../config';

// Access configuration:
const config = DECISION_CONFIG[decision];
console.log(config.label);     // "Auto Approved"
console.log(config.color);     // "#22c55e"

// Update: Edit config.ts directly
// Changes apply on next component mount
```

## Type System

### Type Hierarchy

```
VerificationResponse (from backend)
├─ schema_version: string
├─ authenticity_score: number
├─ decision: 'AUTO_APPROVE' | ...
├─ confidence: number (0-1)
├─ abstained: boolean
│
├─ layer_scores: LayerScores
│  └─ { cnn, vit, gan, ocr }: number
│
├─ layer_reliabilities: LayerReliabilities
│  └─ { cnn, vit, gan, ocr }: number
│
├─ effective_weights: EffectiveWeights
│  └─ { cnn, vit, gan, ocr }: number
│
├─ layer_status: LayerStatus
│  └─ { cnn, vit, gan, ocr }: 'loaded' | 'unavailable' | 'error'
│
└─ Additional fields...

FormData (to backend)
├─ image: File
├─ order_date?: string
├─ delivery_date?: string
└─ mfg_date_claimed?: string
```

## Request-Response Cycle

### Complete Example

```javascript
// 1. PREPARE
const formData = new FormData();
formData.append('image', file);
formData.append('order_date', '2024-11-01');

// 2. REQUEST
POST /api/v1/verify
Content-Type: multipart/form-data
Body: formData

// 3. UPLOAD
Upload Progress: 0% → 50% → 100%
(Tracked with onUploadProgress)

// 4. SERVER PROCESSES
Backend:
1. Receives FormData
2. Extracts image file
3. Runs CNN model → 85.2 score
4. Runs ViT model → 80.1 score
5. Runs GAN model → 79.4 score
6. Runs OCR model → 84.0 score
7. Fuses results → 82.0 final score
8. Determines decision → FAST_TRACK

// 5. RESPONSE
HTTP 200 OK
Content-Type: application/json
Body: {
  "authenticity_score": 82,
  "decision": "FAST_TRACK",
  "confidence": 0.91,
  "layer_scores": { ... },
  ...
}

// 6. DISPLAY
Components render with:
├─ DecisionBadge: "FAST_TRACK" in yellow
├─ Score: 82
├─ Confidence: 91%
└─ Layers: All 4 scores displayed
```

## Performance Characteristics

### Component Lifecycle

```
1. Initial Render
   └─ App mounts
   └─ State initialized
   └─ Header, Footer, UploadSection render

2. File Selected
   └─ handleFileSelect() called
   └─ preview generated (async)
   └─ State updated
   └─ ImagePreview, FormFields mount

3. Form Submitted
   └─ Loading state = true
   └─ LoadingSkeleton renders
   └─ API request made
   └─ Upload progress updated

4. Results Received
   └─ Loading state = false
   └─ LoadingSkeleton unmounts
   └─ Result state updated
   └─ DecisionBadge, LayerAnalysis render

5. Retry Clicked
   └─ State reset
   └─ Back to step 1 or 2
```

### Re-render Optimization

```
App re-renders when:
├─ uploadedFile changes
├─ preview changes
├─ uploadProgress changes
├─ isAnalyzing changes
├─ result changes
└─ error changes

Children only re-render if props change:
├─ UploadSection: when isAnalyzing or uploadProgress changes
├─ ImagePreview: when src or fileName changes
├─ FormFields: when isLoading or previewImage changes
├─ DecisionBadge: when result changes
└─ etc.

This ensures efficient rendering without unnecessary updates.
```

## Error Recovery Flow

```
Error Occurs
    ↓
Error Caught
    ↓
Error Message Generated
    ↓
Error State Updated
    ↓
App Re-renders
    ↓
ErrorMessage Component Shown
    ↓
User Sees:
├─ What went wrong
├─ Possible causes
├─ Suggested solutions
└─ Retry button
    ↓
User Clicks Retry
    ↓
handleRetry() Called:
├─ Clears error state
├─ Resets file state
├─ Clears results
└─ Back to initial state
    ↓
User Can Upload Again
```

---

**Document Version**: 1.0  
**Last Updated**: April 2026  
**For**: Developers and architects
