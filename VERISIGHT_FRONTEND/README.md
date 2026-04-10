# VeriSight Frontend

Professional AI-powered image authenticity verification frontend built with React, TypeScript, and TailwindCSS.

## Overview

VeriSight is a modern web application that enables users to verify image authenticity using a multi-layer AI fusion engine combining CNN, Vision Transformer (ViT), GAN detection, and OCR analysis.

## Features

### Core Functionality
- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Real-time Progress**: Visual upload progress indicator
- **AI Verification**: Multi-layer analysis using 4 AI models
- **Detailed Results**: Comprehensive authenticity scoring and layer breakdown

### User Experience
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Beautiful UI**: Modern gradient design with intuitive controls
- **Error Handling**: Graceful error messages with helpful suggestions
- **Loading States**: Informative skeleton loaders during processing
- **Fast Analysis**: Typical verification completes in 1-3 seconds

### Decision Types
- **AUTO_APPROVE** (85-100): Image appears authentic, no review needed
- **FAST_TRACK** (65-84): Likely authentic, minor review recommended
- **SUSPICIOUS** (40-64): Potential issues detected, requires review
- **REJECT** (0-39): Image flagged as inauthentic, do not approve
- **INCONCLUSIVE**: Model could not make confident determination

### AI Layer Analysis
- **CNN**: Detects low-level visual artifacts and pixel manipulation
- **ViT**: Analyzes high-level semantic inconsistencies
- **GAN**: Identifies AI-generated and deepfake patterns
- **OCR**: Verifies text authenticity and date consistency

## Tech Stack

- **React 18**: Modern UI library with hooks and concurrent rendering
- **TypeScript**: Type-safe JavaScript for better developer experience
- **Vite**: Lightning-fast build tool with HMR
- **TailwindCSS**: Utility-first CSS framework for responsive design
- **Axios**: Promise-based HTTP client for API calls

## Project Structure

```
src/
├── components/           # React components
│   ├── Header.tsx       # App header with branding
│   ├── Footer.tsx       # App footer
│   ├── UploadSection.tsx    # Drag-drop upload interface
│   ├── FormFields.tsx       # Optional date fields
│   ├── ImagePreview.tsx     # Image display
│   ├── DecisionBadge.tsx    # Decision display
│   ├── LayerAnalysis.tsx    # AI layer breakdown
│   ├── ConfidenceIndicator.tsx  # Confidence visualization
│   ├── ErrorMessage.tsx     # Error display
│   └── LoadingSkeleton.tsx  # Loading state
├── services/            # API integration
│   └── api.ts          # Verification API client
├── utils/              # Helper functions
│   └── helpers.ts      # Utility functions
├── config.ts           # Application configuration
├── types.ts            # TypeScript definitions
├── App.tsx             # Main application component
├── main.tsx            # React entry point
└── index.css           # Global styles

```

## Installation

### Prerequisites
- Node.js 18+ 
- npm or yarn package manager

### Setup

1. Clone the repository:
```bash
cd VERISIGHT_FRONTEND
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment:
```bash
cp .env.example .env.local
```

Update `.env.local` with your API server URL:
```env
VITE_API_URL=http://your-backend-server:8000/api/v1
```

## Development

### Start Development Server
```bash
npm run dev
```

The application will open at `http://localhost:5173` with hot module replacement enabled.

### Build for Production
```bash
npm run build
```

Output files will be in the `dist/` directory.

### Preview Production Build
```bash
npm run preview
```

### Type Checking
```bash
npm run type-check
```

## API Integration

### Verification Endpoint

**POST** `/api/v1/verify`

#### Request
```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);
formData.append("order_date", "2024-11-01");  // Optional
formData.append("delivery_date", "2024-11-15");  // Optional
formData.append("mfg_date_claimed", "2024-09-20");  // Optional

const response = await fetch("http://your-server/api/v1/verify", {
  method: "POST",
  body: formData,
});
```

#### Response
```json
{
  "authenticity_score": 82,
  "decision": "FAST_TRACK",
  "confidence": 0.91,
  "abstained": false,
  "layer_scores": { "cnn": 85.2, "vit": 80.1, "gan": 79.4, "ocr": 84.0 },
  "layer_reliabilities": { "cnn": 0.93, "vit": 0.88, "gan": 0.76, "ocr": 0.91 },
  "effective_weights": { "cnn": 0.35, "vit": 0.30, "gan": 0.20, "ocr": 0.15 },
  "processing_time_ms": 1340
}
```

## Configuration

### API Configuration
Edit `src/config.ts` to customize:
- API base URL
- File upload constraints
- Decision thresholds
- AI layer names and descriptions

### File Constraints
- **Max Size**: 10 MB
- **Accepted Types**: JPEG, PNG, WebP, BMP
- **Min Dimensions**: No restriction (larger images improve accuracy)

## Component Documentation

### UploadSection
Handles image upload with drag-and-drop support.
- Validates file type and size
- Shows upload progress
- Generates image preview

### FormFields
Optional date input fields for enhanced verification.
- Order date
- Delivery date
- Manufacturing date claim

### DecisionBadge
Displays verification decision with color-coded indicator.
- Shows authenticity score (0-100)
- Displays decision label
- Shows confidence level
- Includes actionable description

### LayerAnalysis
Detailed breakdown of AI layer contributions.
- Individual layer scores (0-100)
- Reliability indicators
- Weight distribution
- Processing metadata

### ConfidenceIndicator
Visual confidence level display.
- Percentage-based confidence
- Color graduation from red to green
- Confidence level label

## Error Handling

The application gracefully handles various error scenarios:

| Error | Handling |
|-------|----------|
| Invalid file type | Shows format requirements |
| File too large | Displays size limit |
| Network timeout | Suggests retry with connection check |
| Server error | Shows generic error with retry button |
| Corrupted image | Suggests file validation |

## Performance Optimizations

- **Code Splitting**: Components lazy-loaded where applicable
- **Asset Optimization**: Minimized CSS and JavaScript
- **Image Optimization**: Efficient preview generation
- **Request Optimization**: Single API call per verification

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## Accessibility

- Semantic HTML structure
- ARIA labels where needed
- Keyboard navigation support
- Color-blind friendly design
- Sufficient contrast ratios

## Security Considerations

- No sensitive data stored locally
- Secure HTTPS recommended for production
- File validation on client side
- No credentials exposed in frontend code

## Troubleshooting

### "Cannot connect to API"
- Verify backend server is running
- Check API URL in `.env.local`
- Ensure CORS is configured on backend

### "File rejected"
- Check file format (JPEG, PNG, WebP, BMP only)
- Verify file size under 10MB
- Ensure image is not corrupted

### "Analysis taking too long"
- Check network connection
- Verify backend processing time (typically 1-3s)
- Try with a smaller image file

## Future Enhancements

- [ ] Batch image verification
- [ ] Result history and comparison
- [ ] Advanced filtering and search
- [ ] Export verification reports
- [ ] Real-time layer visualization
- [ ] Dark mode support
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Follow existing code style
4. Add tests for new features
5. Submit a pull request

## License

Copyright 2026 VeriSight. All rights reserved.

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation
- Contact the backend team for endpoint issues

---

**Version**: 2.0.0  
**Last Updated**: April 2026  
**Built with**: React, TypeScript, TailwindCSS, Vite
