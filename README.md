# VeriSight Local Run Guide

This document explains how to run the backend API and frontend UI for this project.

## Prerequisites

- Windows PowerShell
- Python virtual environment at `.venv`
- Node.js and npm installed

## 1) Run Backend (FastAPI)

Open a new terminal at project root.

```powershell
cd C:\Users\JHASHANK\Downloads\VERISIGHT_V1
.\.venv\Scripts\Activate.ps1
python -m uvicorn engine.pipeline.app:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

- API base: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Verify endpoint used by frontend: http://127.0.0.1:8000/api/v1/verify

## 2) Run Frontend (React + Vite)

Open another terminal.

```powershell
cd C:\Users\JHASHANK\Downloads\VERISIGHT_V1\frontend
npm install
npm run dev
```

Open the Vite URL shown in terminal (usually http://localhost:5173).

## 3) How the UI Works

- Input: product image only
- The UI auto-sends today date for `order_date` and `delivery_date`
- `mfg_date_claimed` is not sent, so OCR can infer manufacturing date from the image

## Quick Troubleshooting

### Error: ECONNREFUSED 127.0.0.1:8000

Cause: backend is not running.

Fix:

1. Start backend using the command in section 1.
2. Keep backend terminal open while using frontend.

### Error: No module named uvicorn

Install API dependencies inside `.venv`:

```powershell
python -m pip install uvicorn fastapi python-multipart
```

### Frontend started from wrong folder

If you run npm commands from root accidentally, move to frontend folder first:

```powershell
cd C:\Users\JHASHANK\Downloads\VERISIGHT_V1\frontend
```

## 4) Canonical End-to-End Benchmark

Run a benchmark report with metrics + latency + calibration:

```powershell
python evaluation/evaluate_system.py --output-json evaluation/latest_benchmark.json --append-history
```

Key outputs:

- `evaluation/latest_benchmark.json`
- `evaluation/benchmark_history.json`

Check regression status from saved benchmark history:

```powershell
python evaluation/check_regression.py --history-json evaluation/benchmark_history.json
```

Calibrate decision thresholds from a benchmark report that includes per-sample data:

```powershell
python evaluation/calibrate_thresholds.py --input-json evaluation/latest_benchmark.json --output-json evaluation/calibrated_thresholds.json
```

## 5) Runtime Tuning (Optional)

Environment variables:

- `VERISIGHT_MAX_CONCURRENT_REQUESTS` (default `4`)
- `VERISIGHT_REQUEST_TIMEOUT_MS` (default `15000`)
- `VERISIGHT_OCR_ENGINE` (`auto`, `easy`, `paddle`, `yolo`)
- `VERISIGHT_OCR_GPU` (`true` / `false`)
