# VeriSight Frontend - API Testing Guide

This guide helps you test the VeriSight frontend with the backend API.

## Prerequisites

1. Backend API running (v2.0.0)
2. API URL configured in `.env.local`
3. Frontend running with `npm run dev`
4. Sample image file (JPEG, PNG, WebP, or BMP)

## API Endpoint Reference

### Main Verification Endpoint

**POST** `/api/v1/verify`

#### Request Format

Using `curl`:
```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "image=@/path/to/image.jpg" \
  -F "order_date=2024-11-01" \
  -F "delivery_date=2024-11-15" \
  -F "mfg_date_claimed=2024-09-20"
```

Using JavaScript fetch:
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('order_date', '2024-11-01');
formData.append('delivery_date', '2024-11-15');
formData.append('mfg_date_claimed', '2024-09-20');

const response = await fetch('http://localhost:8000/api/v1/verify', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result);
```

#### Expected Response (200 OK)

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

## Testing Scenarios

### Scenario 1: Basic Image Upload

1. Start frontend: `npm run dev`
2. Open http://localhost:5173
3. Drag-drop or click to upload an image
4. Fill optional date fields
5. Click "Verify Image"
6. Check results display

### Scenario 2: Error Handling

#### Invalid File Type
- Upload a text file (.txt)
- Expected: "Unsupported file type" error

#### File Too Large
- Try uploading file > 10MB
- Expected: "File too large" error  

#### Network Error
- Stop backend server
- Try uploading image
- Expected: "Failed to verify image" error with retry button

#### Corrupted Image
- Create empty image file
- Upload it
- Expected: Processing error message

### Scenario 3: Decision Display

Upload images that should trigger different decisions:

**AUTO_APPROVE** (85-100):
- Genuine, unmodified product image
- Clear, authentic labels
- No signs of manipulation

**FAST_TRACK** (65-84):
- Mostly authentic with minor variations
- Some slight artifacts
- Overall good authenticity

**SUSPICIOUS** (40-64):
- Signs of manipulation detected
- Lab for review recommended
- Mixed signals from layers

**REJECT** (0-39):
- Clear signs of deepfake or fabrication
- Multiple layer alerts
- Do not approve

### Scenario 4: Layer Analysis

Verify layer scores display correctly:

1. Check all 4 layers show scores if status is "loaded"
2. Verify reliability percentages (0-100%)
3. Check weights sum to approximately 1.0
4. Confirm processing time displays

### Scenario 5: Abstained Response

If backend returns `abstained: true`:

1. Score display should show "--"
2. Decision should show "Inconclusive"
3. Description should indicate uncertainty
4. No layer scores should be highlighted

## Testing Tools

### Browser DevTools

1. Open DevTools: F12
2. Go to Network tab
3. Filter by XHR (XMLHttpRequest)
4. Upload image
5. Click on `/verify` request
6. Check:
   - Request headers (Content-Type should be multipart/form-data)
   - Request body (FormData contents)
   - Response status (200 for success)
   - Response body (JSON result)

### Advanced Testing with curl

Test upload progress:
```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "image=@/path/to/image.jpg" \
  -w "\nStatus: %{http_code}\n" \
  -v
```

### Testing with Postman

1. Open Postman
2. Create new POST request
3. URL: `http://localhost:8000/api/v1/verify`
4. Go to Body tab
5. Select "form-data"
6. Add key `image` with value (File type)
7. Add optional keys: `order_date`, `delivery_date`, `mfg_date_claimed`
8. Send request
9. Review response

## Debugging API Issues

### Issue: CORS Error

**Error Message**: "Access to XMLHttpRequest blocked by CORS policy"

**Solutions**:
1. Check backend has CORS enabled
2. Verify frontend URL is in CORS whitelist
3. Ensure backend and frontend are on same network

### Issue: 404 Not Found

**Error Message**: "POST /api/v1/verify 404"

**Solutions**:
1. Verify API URL in `.env.local`
2. Check backend is running on correct port
3. Ensure endpoint path is `/api/v1/verify`

### Issue: 413 Payload Too Large

**Error Message**: "413 Payload Too Large"

**Solutions**:
1. Image file is > 10MB
2. Compress image before uploading
3. Use smaller resolution image

### Issue: 503 Service Unavailable

**Error Message**: "503 Service Unavailable"

**Solutions**:
1. Backend is warming up (AI models loading)
2. Wait 30-60 seconds and retry
3. Check backend logs for startup progress

### Issue: Timeout

**Error Message**: "Request timeout"

**Solutions**:
1. Check network connectivity
2. Increase API timeout in `src/services/api.ts`
3. Verify backend is responsive

## Response Validation

### Verify Response Structure

Create test file `test-api.js`:

```javascript
const fs = require('fs');

const validResponse = {
  schema_version: 'string',
  authenticity_score: 'number (0-100)',
  decision: 'string (AUTO_APPROVE|FAST_TRACK|SUSPICIOUS|REJECT)',
  confidence: 'number (0-1)',
  abstained: 'boolean',
  layer_scores: 'object with cnn,vit,gan,ocr',
  layer_reliabilities: 'object with cnn,vit,gan,ocr',
  effective_weights: 'object with cnn,vit,gan,ocr',
  processing_time_ms: 'number',
};

console.log('Expected Response Structure:');
console.log(JSON.stringify(validResponse, null, 2));
```

Run: `node test-api.js`

## Performance Testing

### Measure Response Time

In browser console:

```javascript
const start = performance.now();

fetch('/api/v1/verify', {
  method: 'POST',
  body: formData,
}).then(r => r.json()).then(() => {
  const end = performance.now();
  console.log(`Total time: ${end - start}ms`);
});
```

### Load Testing

Test with multiple images:

```javascript
async function testMultiple() {
  const times = [];
  
  for (let i = 0; i < 5; i++) {
    const start = performance.now();
    const response = await fetch('/api/v1/verify', {
      method: 'POST',
      body: formData,
    });
    const end = performance.now();
    times.push(end - start);
    console.log(`Request ${i+1}: ${end-start}ms`);
  }
  
  const avg = times.reduce((a,b) => a+b) / times.length;
  console.log(`Average: ${avg}ms`);
}

testMultiple();
```

## Integration Checklist

- [ ] Backend API running and accessible
- [ ] API URL configured in `.env.local`
- [ ] Frontend can reach API (no CORS errors)
- [ ] Image upload works
- [ ] Results display with correct decision
- [ ] All 4 AI layers show scores
- [ ] Confidence indicator works
- [ ] Error messages display for failures
- [ ] Retry works after error
- [ ] "Analyze Another Image" button resets state
- [ ] Mobile responsiveness verified

## Example Test Cases

### Test Case 1: Valid Image Upload

**Input**: Valid JPEG image
**Expected**: Full verification result with decision and layer scores

### Test Case 2: Invalid File Type

**Input**: PDF file
**Expected**: Error message "Unsupported file type"

### Test Case 3: Large File

**Input**: Image file 15MB+
**Expected**: Error message "File too large"

### Test Case 4: Optional Dates

**Input**: Image with all optional dates filled
**Expected**: Dates passed to backend, result displayed

### Test Case 5: No Optional Dates

**Input**: Image with no optional dates
**Expected**: Works without dates (optional fields)

### Test Case 6: Backend Unavailable

**Input**: Stop backend, try upload
**Expected**: Connection error with retry option

### Test Case 7: API Timeout

**Input**: Simulate delay > 30s
**Expected**: Timeout error message

### Test Case 8: Abstained Response

**Input**: Image that returns abstained: true
**Expected**: Inconclusive display, "--" score

## Monitoring API Calls

### Real-time Monitoring

1. Open DevTools Network tab
2. Start filtering: Type "verify"
3. Upload image
4. Watch request/response in real-time
5. Check timing breakdown

### API Analytics (if backend provides)

1. Monitor endpoint response times
2. Track error rates
3. Measure layer processing times
4. Identify slow requests

## Troubleshooting Checklist

| Issue | Check |
|-------|-------|
| API not responding | Backend running? URL correct? Network OK? |
| CORS error | Backend CORS enabled? |
| File upload stuck | File size under 10MB? File format correct? |
| Results not showing | Response received? Check console. |
| Layers missing | All layers loaded? Check layer_status. |

---

**Last Updated**: April 2026  
**API Version**: v2.0.0
