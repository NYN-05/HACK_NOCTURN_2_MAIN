# Quick Start Guide - VeriSight Frontend

Get up and running in less than 5 minutes.

## Prerequisites

Have these ready:
- Node.js 18+ ([Download](https://nodejs.org/))
- A code editor (VS Code recommended)
- A terminal/command prompt
- Backend API URL or local backend running

## Step 1: Install Dependencies (2 minutes)

```bash
cd VERISIGHT_FRONTEND
npm install
```

You'll see installation progress. Once complete, you have all dependencies.

## Step 2: Configure Backend Connection (1 minute)

```bash
cp .env.example .env.local
```

Then edit `.env.local`:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

Replace `localhost:8000` with your actual backend URL.

## Step 3: Start Development Server (1 minute)

```bash
npm run dev
```

You should see:
```
VITE v5.0.0 running at:
  > Local:    http://localhost:5173/
```

Your browser will automatically open. If not, go to http://localhost:5173

## Step 4: Test It Out (1 minute)

1. You should see the VeriSight interface
2. Click the upload area or drag-drop an image (JPG, PNG, WebP, BMP)
3. Fill optional date fields
4. Click "Verify Image"
5. See the results!

## That's It!

You now have VeriSight Frontend running.

## Common Next Steps

### See the Code

Open `src/App.tsx` to see the main component structure.

### Customize Colors

Edit `src/config.ts` to customize decision colors and descriptions.

### Change API URL

Update `.env.local` anytime - saves and hot-reloads automatically.

### Build for Production

```bash
npm run build
```

Creates optimized `dist/` folder for deployment.

## Troubleshooting

### Port Already in Use

Use a different port:
```bash
npm run dev -- --port 3000
```

### API Connection Issues

1. Check backend is running on the configured URL
2. Open browser console (F12) for error messages
3. Try with `http://localhost:8000` as backend URL

### Blank Page

1. Check browser console (F12 > Console tab)
2. Hard refresh: Ctrl+Shift+R (Cmd+Shift+R on Mac)
3. Clear cache: Settings > Clear browsing data

### "npm command not found"

Install Node.js from nodejs.org

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/App.tsx` | Main application component |
| `src/config.ts` | Configuration (colors, labels, etc) |
| `.env.local` | Environment variables (API URL) |
| `src/components/` | React components |
| `src/services/api.ts` | Backend API integration |

## Useful Commands

```bash
npm run dev              # Start development server
npm run build            # Build for production
npm run preview          # Preview production build
npm run type-check       # Check TypeScript types
npm run lint             # Run ESLint (if configured)
```

## Documentation

| Document | Read When |
|----------|-----------|
| [README.md](./README.md) | Want full feature overview |
| [SETUP.md](./SETUP.md) | Need detailed installation help |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Want to develop new features |
| [API_TESTING.md](./API_TESTING.md) | Testing API integration |
| [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) | Want project overview |

## Enable Hot Reload

During development:
- Save any file
- Browser automatically reloads
- See changes instantly

## Debug Tips

Press F12 to open Developer Tools:

- **Console Tab**: See errors and logs
- **Elements Tab**: Inspect HTML structure
- **Network Tab**: See API requests
- **Sources Tab**: Set breakpoints in code

## Next: Time to Explore!

1. Look at the components in `src/components/`
2. Try uploading different images
3. Review the configuration options
4. Test different API responses
5. Customize the design

## Need Help?

1. Check the browser console for error messages
2. Read the relevant documentation file
3. Review inline component comments
4. Check your `.env.local` configuration

## You're All Set!

VeriSight Frontend is ready to use. Start uploading images and verifying authenticity!

---

**Frontend Version**: 2.0.0  
**Setup Time**: ~5 minutes  
**Next**: Read other documentation as needed
