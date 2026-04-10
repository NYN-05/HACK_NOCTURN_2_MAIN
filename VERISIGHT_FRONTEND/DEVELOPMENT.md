# VeriSight Frontend - Developer Guide

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
```bash
cp .env.example .env.local
```

Update the API URL:
```env
VITE_API_URL=http://localhost:8000/api/v1
```

### 3. Start Development Server
```bash
npm run dev
```

The app will open at `http://localhost:5173`

## Architecture Overview

### Component Hierarchy

```
App (Main orchestrator)
├── Header
├── Main Content
│   ├── UploadSection (or)
│   ├── ImagePreview
│   ├── FormFields
│   ├── LoadingSkeleton
│   ├── DecisionBadge
│   ├── ConfidenceIndicator
│   └── LayerAnalysis
├── ErrorMessage
└── Footer
```

### State Management

The app uses React hooks for state management:

- `uploadedFile`: Currently selected file
- `preview`: Image preview data URL
- `isAnalyzing`: API request in progress
- `result`: Verification response from backend
- `error`: Error message if request fails
- `uploadProgress`: File upload progress (0-100)

### Data Flow

```
User selects file
    ↓
generatePreview() creates data URL
    ↓
File + preview stored in state
    ↓
User fills optional dates and clicks "Verify"
    ↓
handleFormSubmit() called
    ↓
verificationAPI.verifyImage() sends FormData
    ↓
Response received and stored in result state
    ↓
Components re-render with results
```

## Key Components

### Header.tsx
- Static header with branding
- Displays version information
- No state management needed

### UploadSection.tsx
- Handles file selection (click or drag-drop)
- Validates file type and size
- Generates preview
- Shows upload progress bar
- Props:
  - `onFileSelect`: Callback when file is selected
  - `isLoading`: Disable during API request
  - `uploadProgress`: Current upload percentage

### FormFields.tsx
- Optional date input fields
- Only displays when image is selected
- Props:
  - `onSubmit`: Called when user clicks verify
  - `isLoading`: Disable form during API request
  - `previewImage`: Required to show form

### ImagePreview.tsx
- Displays selected image
- Shows file name
- Remove button (if not loading)
- Props:
  - `src`: Image data URL
  - `fileName`: Original file name
  - `onRemove`: Reset app state

### DecisionBadge.tsx
- Displays main decision with color coding
- Shows authenticity score or "--" if abstained
- Includes decision description
- Props:
  - `decision`: Decision type
  - `score`: Authenticity score (0-100)
  - `abstained`: If true, show inconclusive state

### LayerAnalysis.tsx
- Shows individual AI layer scores
- Displays reliability and weight for each layer
- Shows processing metadata
- Props:
  - `response`: Full verification response

### ConfidenceIndicator.tsx
- Confidence level visualization
- Progress bar with gradient
- Percentage display
- Props:
  - `confidence`: Confidence value (0.0-1.0)

### ErrorMessage.tsx
- Displays error with helpful suggestions
- Retry button available
- Props:
  - `message`: Error message
  - `onRetry`: Callback for retry action

### LoadingSkeleton.tsx
- Skeleton loader during API request
- Matches result layout
- Pure component, no props

## Working with Types

All TypeScript types are in `src/types.ts`:

```typescript
// API Response
interface VerificationResponse {
  authenticity_score: number;
  decision: 'AUTO_APPROVE' | 'FAST_TRACK' | 'SUSPICIOUS' | 'REJECT';
  confidence: number;
  abstained: boolean;
  // ... other fields
}

// Form data
interface FormData {
  order_date: string;
  delivery_date: string;
  mfg_date_claimed: string;
}
```

## API Integration

The API client is in `src/services/api.ts`:

```typescript
// Single method for verification
await verificationAPI.verifyImage(
  file,
  orderDate,
  deliveryDate,
  mfgDateClaimed,
  onUploadProgress
);
```

Error handling is built-in:
- 400: Invalid file or format
- 413: File too large
- 422: Image corrupted
- 503: Server warming up
- Other: Generic error message

## Configuration

### Customizing Decision Colors

Edit `src/config.ts`:

```typescript
export const DECISION_CONFIG = {
  AUTO_APPROVE: {
    label: 'Custom Label',
    color: '#custom-hex',
    bg: '#custom-bg',
    // ...
  },
};
```

### Customizing AI Layer Names

Edit `src/config.ts`:

```typescript
export const LAYER_CONFIG = {
  cnn: {
    name: 'Convolutional NN',
    fullName: 'Full Name',
    description: 'What this layer does',
    icon: 'CNN',
  },
};
```

## Common Tasks

### Add a New Component

1. Create `src/components/NewComponent.tsx`:
```typescript
interface NewComponentProps {
  // Define props
}

export function NewComponent(props: NewComponentProps) {
  return (
    // JSX
  );
}
```

2. Import and use in `App.tsx`:
```typescript
import { NewComponent } from './components/NewComponent';

// In JSX:
<NewComponent {...props} />
```

### Add a Utility Function

1. Add to `src/utils/helpers.ts`:
```typescript
export function myUtilityFunction(input: string): string {
  // Implementation
  return result;
}
```

2. Import where needed:
```typescript
import { myUtilityFunction } from '../utils/helpers';
```

### Customize Styling

Use TailwindCSS utility classes in components:

```tsx
<div className="flex items-center justify-between p-4 bg-slate-100 rounded-lg">
  {/* Content */}
</div>
```

For complex styles, add to `src/index.css`:

```css
@layer components {
  .custom-class {
    @apply px-4 py-2 bg-blue-600 rounded-lg;
  }
}
```

## Testing

While specific test files aren't included, here are guidelines:

### Manual Testing Checklist

- [ ] File upload with drag-and-drop
- [ ] File upload with click
- [ ] File validation (wrong type, too large)
- [ ] Image preview displays
- [ ] Form submits with optional dates
- [ ] Loading state appears
- [ ] Results display correctly
- [ ] Error handling works
- [ ] Retry button resets state
- [ ] Analyze another image works
- [ ] Mobile responsive

### Test API URLs

For development, use:
- `http://localhost:8000/api/v1` (local backend)
- Update in `.env.local` for production

## Performance Tips

1. **Lazy Load Components** (if needed):
```typescript
const DecisionBadge = lazy(() => 
  import('./components/DecisionBadge').then(m => ({ default: m.DecisionBadge }))
);
```

2. **Optimize Image Preview**:
- Limit preview image dimensions
- Use WebP format when possible
- Compress before sending

3. **Cache Policy**:
- Results are not cached (intentional for accuracy)
- Each verification is fresh

## Debugging

### Enable Debug Logging

Add to `src/services/api.ts`:

```typescript
client.interceptors.response.use(
  response => {
    console.log('API Response:', response.data);
    return response;
  },
  error => {
    console.error('API Error:', error);
    throw error;
  }
);
```

### Browser DevTools

- Use React DevTools to inspect component state
- Check Network tab for API requests
- Use Console for error messages

## Deployment

### Build Production Bundle

```bash
npm run build
```

### Environment Variables for Production

Create `.env.production`:
```env
VITE_API_URL=https://your-production-api.com/api/v1
```

### Deploy to Hosting

```bash
# Build
npm run build

# Upload dist/ folder to your hosting service
# Examples: Vercel, Netlify, GitHub Pages, etc.
```

## Code Style Guidelines

### Component Structure
```typescript
// 1. Imports
import { useState } from 'react';
import { SomeComponent } from './components/SomeComponent';

// 2. Interface definitions
interface ComponentProps {
  prop1: string;
  prop2: number;
}

// 3. Component definition
export function MyComponent({ prop1, prop2 }: ComponentProps) {
  // 3a. State hooks
  const [state, setState] = useState(false);

  // 3b. Effect hooks (if needed)
  
  // 3c. Event handlers
  const handleClick = () => {
    setState(!state);
  };

  // 3d. Render
  return (
    <div className="...">
      {/* JSX */}
    </div>
  );
}
```

### Naming Conventions
- Components: PascalCase (e.g., `DecisionBadge`)
- Functions: camelCase (e.g., `formatConfidence`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_FILE_SIZE`)
- Files: Match component name (e.g., `DecisionBadge.tsx`)

## Resources

- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TailwindCSS Docs](https://tailwindcss.com/docs)
- [Vite Guide](https://vitejs.dev/guide/)
- [Axios Documentation](https://axios-http.com/)

## Support

For issues:
1. Check error messages in browser console
2. Verify API server is running
3. Confirm API URL in environment config
4. Check network connectivity
5. Review component props are correct

---

**Last Updated**: April 2026
