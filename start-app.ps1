# VeriSight - Backend & Frontend Launcher (PowerShell)
# This script launches both servers simultaneously

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VeriSight - Backend & Frontend Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting backend server on http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Starting frontend server on http://localhost:5173" -ForegroundColor Yellow
Write-Host ""

# Get the root directory
$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend Server
Write-Host "[1/2] Launching Backend API Server..." -ForegroundColor Green
$backendJob = Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "cd '$rootDir'; python -m uvicorn engine.pipeline.app:app --host 127.0.0.1 --port 8000 --reload" -PassThru -WindowStyle Normal

# Wait for backend to initialize
Start-Sleep -Seconds 3

# Start Frontend Server
Write-Host "[2/2] Launching Frontend Dev Server..." -ForegroundColor Green
$frontendJob = Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "cd '$rootDir\frontend'; npm run dev" -PassThru -WindowStyle Normal

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Servers are starting..." -ForegroundColor Cyan
Write-Host "  Backend:  http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend Process ID: $($backendJob.Id)" -ForegroundColor Gray
Write-Host "Frontend Process ID: $($frontendJob.Id)" -ForegroundColor Gray
Write-Host ""
Write-Host "Both servers are running in separate windows." -ForegroundColor Gray
Write-Host "Close individual windows to stop each server." -ForegroundColor Gray
