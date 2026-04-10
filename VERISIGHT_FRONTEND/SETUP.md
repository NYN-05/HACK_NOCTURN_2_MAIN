# VeriSight Frontend - Setup & Installation Guide

## System Requirements

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher (or yarn/pnpm)
- **Modern Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Backend**: VeriSight API v2.0.0 running

## Step-by-Step Installation

### Step 1: Verify Node.js Installation

```bash
node --version
npm --version
```

Expected output:
- Node.js v18+ 
- npm 9+

If not installed, download from [nodejs.org](https://nodejs.org/)

### Step 2: Navigate to Project Directory

```bash
cd /path/to/VERISIGHT_FRONTEND
```

### Step 3: Install Dependencies

```bash
npm install
```

This will install all required packages:
- react
- react-dom
- axios (HTTP client)
- tailwindcss (CSS framework)
- typescript
- vite (build tool)

Installation time: 2-5 minutes depending on internet speed

### Step 4: Configure Environment

Copy example configuration:
```bash
cp .env.example .env.local
```

Edit `.env.local` with your backend API URL:
```env
VITE_API_URL=http://localhost:8000/api/v1
```

### Step 5: Start Development Server

```bash
npm run dev
```

Expected output:
```
VITE v5.0.0  running at:

  > Local:    http://localhost:5173/
  > press h to show help
```

Your default browser will open automatically to `http://localhost:5173`

## Verifying Installation

### In Browser

1. Open http://localhost:5173
2. Should see VeriSight header and upload interface
3. Try uploading a test image
4. Verify connection to backend (should show error if backend is not running)

### In Terminal

- Vite will show "ready in XXms"
- Any errors will be displayed with helpful messages
- Check console (Ctrl+Shift+K) for debugging

## Quick Test

### Without Backend

1. Upload an image
2. You should see upload progress
3. When analyzing starts, API will fail (expected if backend not running)
4. Error message should display

### With Backend

1. Ensure backend is running at configured URL
2. Upload an image  
3. See verification result within 1-3 seconds
4. Review all components display correctly

## Project Commands

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build locally
npm run preview

# Type checking
npm run type-check

# Run linting (if configured)
npm run lint
```

## Troubleshooting Installation

### Issue: "npm command not found"
**Solution**: Install Node.js from nodejs.org

### Issue: Port 5173 already in use
**Solution**: 
```bash
# Kill process on port 5173, or edit vite.config.ts:
npm run dev -- --port 3000  # Use different port
```

### Issue: "Cannot find module 'react'"
**Solution**:
```bash
rm -rf node_modules
npm install
```

### Issue: Blank page or error on startup
**Solution**:
1. Open DevTools (F12)
2. Check Console tab for errors
3. Clear browser cache (Ctrl+Shift+Delete)
4. Restart dev server

## File Structure After Setup

```
VERISIGHT_FRONTEND/
├── node_modules/          # Dependencies (created by npm install)
├── src/                   # Source code
│   ├── components/        # React components
│   ├── services/          # API integration
│   ├── utils/             # Helper functions
│   ├── App.tsx            # Main app component
│   ├── main.tsx           # React entry point
│   ├── index.css          # Global styles
│   ├── config.ts          # Configuration
│   └── types.ts           # TypeScript types
├── index.html             # HTML template
├── package.json           # Project metadata
├── tsconfig.json          # TypeScript config
├── vite.config.ts         # Vite config
├── tailwind.config.js     # Tailwind config
├── .env.local             # Environment variables (local)
└── dist/                  # Build output (after npm run build)
```

## Configuring Backend Connection

### Development Environment

Update `.env.local`:
```env
VITE_API_URL=http://localhost:8000/api/v1
```

### Production Environment

1. Create `.env.production`:
```env
VITE_API_URL=https://your-production-api.com/api/v1
```

2. Build for production:
```bash
npm run build
```

## Testing API Connection

### Method 1: Using Browser Console

```javascript
fetch('http://localhost:8000/api/v1/verify', {
  method: 'POST',
  body: new FormData()
}).catch(e => console.log('API Error:', e));
```

### Method 2: Using curl

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -F "image=@test-image.jpg"
```

### Method 3: In-App Testing

1. Start frontend with `npm run dev`
2. Upload an image
3. Click "Verify Image"
4. Check browser Network tab for `/verify` request
5. Look for response or error messages

## Stopping the Development Server

Press `Ctrl+C` in terminal where `npm run dev` is running

## Next Steps

1. Read [README.md](./README.md) for feature overview
2. Check [DEVELOPMENT.md](./DEVELOPMENT.md) for dev guidelines
3. Review [API Documentation](./DOCUMENTATION/VERISIGHT_API_REFERENCE.md)
4. Customize colors in `src/config.ts`
5. Deploy when ready

## Getting Help

### Check These First

1. **Browser Console**: F12 > Console tab for errors
2. **Terminal Output**: Dev server output shows build errors
3. **Network Tab**: F12 > Network tab to see API requests
4. **README files**: Comprehensive documentation included

### Common Solutions

| Problem | Solution |
|---------|----------|
| Black/blank page | Clear cache, restart server |
| API not responding | Check backend is running, verify URL |
| Hot reload not working | Kill process, restart `npm run dev` |
| TypeScript errors | Run `npm run type-check` |
| Build fails | Delete `node_modules`, run `npm install` again |

## System Performance

### Development Server Resource Usage

- Memory: 100-200 MB
- CPU: Normal idle, spikes during save/rebuild
- Disk: ~350 MB for node_modules

### Build Output Size

- Minified JS: ~60 KB
- CSS: ~15 KB
- Total gzipped: ~25 KB

## Browser DevTools Setup

### Recommended Extensions

1. **React DevTools**
   - Chrome: [React DevTools](https://chrome.google.com/webstore)
   - Firefox: [React DevTools for Firefox](https://addons.mozilla.org/addon/react-devtools/)

2. **TypeScript Debugging**
   - Built-in source maps for debugging

### Using React DevTools

1. Install extension
2. Open DevTools (F12)
3. Find "Components" tab
4. Inspect component hierarchy
5. View props and state

## Additional Resources

- [Node.js Documentation](https://nodejs.org/docs/)
- [npm Docs](https://docs.npmjs.com/)
- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)

## Support Checklist

Before seeking help, verify:

- [ ] Node.js 18+ installed (`node --version`)
- [ ] Dependencies installed (`npm install` succeeded)
- [ ] .env.local configured with correct API URL
- [ ] Dev server running (`npm run dev`)
- [ ] Browser showing page (not blank/error)
- [ ] Browser console has no red errors
- [ ] Backend API is running and accessible

---

**Version**: 2.0.0  
**Last Updated**: April 2026  
**For Issues**: Check documentation or contact development team
