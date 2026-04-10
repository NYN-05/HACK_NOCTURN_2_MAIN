@echo off
REM Script to run VeriSight Backend and Frontend simultaneously
REM This script launches both servers in separate windows

echo ========================================
echo   VeriSight - Backend and Frontend Launcher
echo ========================================
echo.
echo Starting backend server on http://127.0.0.1:8000
echo Starting frontend server on http://localhost:5173
echo.

REM Start Backend Server
echo [1/2] Launching Backend API Server...
start "VeriSight Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn engine.pipeline.app:app --host 127.0.0.1 --port 8000 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Start Frontend Server
echo [2/2] Launching Frontend Dev Server...
start "VeriSight Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ========================================
echo   Servers are starting...
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:5173
echo ========================================
echo.
echo Press any key to close this window (servers will continue running)...
pause >nul
