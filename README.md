# VeriSight Local Run Guide

This document explains how to run the backend API and frontend UI for this project.

## Prerequisites

- Windows PowerShell
- Python virtual environment at `.venv`
- Node.js and npm installed

## 1) Run Backend (FastAPI)

Open a new terminal at project root.

```powershell
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26
.\.venv\Scripts\Activate.ps1
python -m uvicorn pipeline.app:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

- API base: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Verify endpoint used by frontend: http://127.0.0.1:8000/api/v1/verify

## 2) Run Frontend (React + Vite)

Open another terminal.

```powershell
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26\frontend
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
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26\frontend
```
