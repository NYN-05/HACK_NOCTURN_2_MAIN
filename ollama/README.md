# Ollama Image Matrix

This folder contains a local image classification workflow that uses an Ollama vision model and maps the result into the same matrix:

- `AUTO_APPROVE`
- `FAST_TRACK`
- `SUSPICIOUS`
- `REJECT`

## Quick Start - Full Stack Integration

To run the **complete application** (frontend + backend):

### Terminal 1: Start the Backend API

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a vision model if needed (in another terminal)
ollama pull gemma4:e4b

# 4. Start the FastAPI backend (in yet another terminal)
cd c:\Users\JHASHANK\Downloads\gem_verisight_v3\ollama
uvicorn ollama_image_matrix:app --reload --port 8001
```

### Terminal 2: Start the Frontend UI

```bash
# 1. Install dependencies
cd VERISIGHT_FRONTEND
npm install

# 2. Start the development server
npm run dev
```

The frontend will open at `http://localhost:5173` and automatically proxy API requests to the backend at `http://localhost:8001`.

---

## Backend

The backend is in [ollama_image_matrix.py](ollama_image_matrix.py).

### API Endpoints

- **`POST /api/v1/verify`** - Main endpoint for image verification
  - Accepts: `image` (file), `order_date`, `delivery_date`, `mfg_date_claimed` (optional)
  - Returns: Complete verification response with confidence score, decision, and layer analysis

- **`POST /analyze-image`** - Legacy endpoint for direct image analysis
  - Accepts: `image` (file)
  - Returns: Simple analysis response

- **`GET /health`** - Health check endpoint
  - Returns: Status and model information

### Requirements

- Ollama installed and running locally
- A vision model pulled in Ollama, such as `gemma4:e4b`
- `OLLAMA_KEEP_ALIVE=1h` recommended for lower repeated-request latency

### Environment Variables

```bash
OLLAMA_MODEL=gemma4:e4b          # Vision model to use
OLLAMA_HOST=http://localhost:11434  # Ollama server URL
OLLAMA_KEEP_ALIVE=1h              # Keep model loaded for faster requests
```

### Run the API

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure Ollama is running:

   ```bash
   ollama serve
   ```

3. Pull a vision model if needed:

   ```bash
   ollama pull gemma4:e4b
   ```

4. Start the API:

   ```bash
   uvicorn ollama_image_matrix:app --reload --port 8001
   ```

### CLI

Run the script directly against an image:

```bash
python ollama_image_matrix.py path/to/image.jpg
```

## Frontend

The React UI lives in `VERISIGHT_FRONTEND/` and communicates with the FastAPI backend.

### Setup & Development

1. Install frontend dependencies:

   ```bash
   cd VERISIGHT_FRONTEND
   npm install
   ```

2. Configure the backend URL (optional):

   - Copy `.env.example` to `.env`
   - Update `VITE_API_URL` if your backend is running on a different address
   - Default: `http://localhost:8001` (matches backend port)

3. Start the UI:

   ```bash
   npm run dev
   ```

4. Open in browser: `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

### Features

- **Image Upload**: Drag-and-drop or click to select image
- **Optional Metadata**: Add order date, delivery date, manufacturing date
- **Real-time Analysis**: Image verification with progress tracking
- **Results Display**: 
  - Decision badge (AUTO_APPROVE, FAST_TRACK, SUSPICIOUS, REJECT)
  - Confidence score and authenticity assessment
  - Detailed layer analysis
  - Image preview

## Development Notes

- The backend enables CORS for local development (all origins allowed)
- The default Ollama host is `http://localhost:11434`
- The default model is `gemma4:e4b`, override with `OLLAMA_MODEL` env var
- Set `OLLAMA_KEEP_ALIVE` to keep the model loaded longer between requests
- Vite proxy configuration handles `/api` requests during development
- In production, update the frontend `VITE_API_URL` to point to your backend deployment

## Project Structure

```
ollama/
├── ollama_image_matrix.py      # FastAPI backend
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── VERISIGHT_FRONTEND/         # React frontend
    ├── src/
    │   ├── components/         # React components
    │   ├── services/           # API client
    │   ├── config.ts           # Configuration
    │   ├── types.ts            # TypeScript interfaces
    │   └── App.tsx             # Main app component
    ├── package.json            # NPM dependencies
    ├── vite.config.ts          # Vite config (includes API proxy)
    ├── tsconfig.json           # TypeScript config
    └── .env.example            # Environment template
```

## Troubleshooting

### Backend won't start
- Ensure Ollama is running: `ollama serve`
- Check that port 8001 is available: `netstat -ano | find "8001"` (Windows)
- Verify Python dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Check backend is running on port 8001
- Verify VITE_API_URL in `.env` matches your backend address
- Check browser console for CORS errors (should not occur with local setup)
- If using proxy, ensure vite.config.ts has the correct proxy configuration

### Ollama model not found
- Pull the model: `ollama pull gemma4:e4b`
- Or set a different model: `OLLAMA_MODEL=llava ollama serve`
- Check available models: `ollama list`

