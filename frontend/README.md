# VeriSight Frontend (React + Vite)

This frontend allows you to upload an image and optional metadata, then call the backend verification API that powers `generate_final_score.py`.

## What It Connects To

- Backend endpoint: `/api/v1/verify`
- Backend implementation: `engine/pipeline/api_router.py`
- Orchestrator used: `engine/pipeline/orchestrator.py` (same flow used by `generate_final_score.py`)

## Local Setup

1. Start backend API from project root:

	```bash
	uvicorn engine.pipeline.app:app --reload --host 127.0.0.1 --port 8000
	```

2. Start frontend:

	```bash
	cd frontend
	npm install
	npm run dev
	```

3. Open the URL shown by Vite (normally `http://localhost:5173`).

## API Base URL (Optional)

By default, the frontend uses relative `/api/v1/verify`, and Vite proxies `/api` to `http://127.0.0.1:8000` in development.

To point at another backend, create `.env` in `frontend`:

```bash
VITE_API_BASE_URL=http://your-host:your-port
```

Then restart the frontend dev server.
