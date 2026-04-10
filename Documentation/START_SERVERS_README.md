# VeriSight - Start Backend & Frontend

This directory contains scripts to launch both the VeriSight backend and frontend servers simultaneously.

## Quick Start

### Option 1: Batch File (CMD) - Recommended for Windows
**Double-click this file:**
```
start-app.cmd
```

This will:
1. Open a terminal window for the Backend API server
2. Open another terminal window for the Frontend dev server
3. Both servers run simultaneously

**Servers will be available at:**
- Backend API: `http://127.0.0.1:8000`
- Frontend UI: `http://localhost:5173`

### Option 2: PowerShell Script
**Run in PowerShell:**
```powershell
.\start-app.ps1
```

Or if you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\start-app.ps1
```

### Option 3: Manual - Start Servers Separately

**Terminal 1 - Backend:**
```bash
python -m uvicorn engine.pipeline.app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## Prerequisites

### Backend Requirements
- Python 3.10+
- All dependencies installed: `pip install -r requirements.lock.txt`

### Frontend Requirements
- Node.js 16+ and npm installed
- Dependencies installed: `cd frontend && npm install`

---

## Troubleshooting

### Backend server fails to start
- Ensure Python is in your PATH
- Run: `pip install -r requirements.lock.txt`
- Check if port 8000 is already in use

### Frontend server fails to start
- Ensure Node.js and npm are installed
- Run: `cd frontend && npm install`
- Check if port 5173 is already in use

### Both windows close immediately
- Open a terminal manually and run the commands separately to see error messages
- Check for missing dependencies or Python/Node.js installation issues

---

## Features

✅ Backend API running with auto-reload  
✅ Frontend dev server with hot module replacement  
✅ Both servers in separate terminal windows  
✅ Easy to monitor and debug  
✅ Press CTRL+C in either window to stop that server  

---

## API Endpoints

Once running, the backend API documentation is available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

Main API endpoint:
- **POST** `/api/v1/verify` - Image verification endpoint

See backend code for more details.
