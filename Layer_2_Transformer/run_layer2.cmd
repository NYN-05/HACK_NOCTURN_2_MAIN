@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

if /I "%MODE%"=="help" goto :help
if /I "%MODE%"=="--help" goto :help
if /I "%MODE%"=="-h" goto :help

echo [Layer-2] Working directory: %CD%
echo [Layer-2] Mode: %MODE%

echo [1/6] Creating virtual environment if missing...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :fail
)

echo [2/6] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo [3/6] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
pip install -r requirements.txt
if errorlevel 1 goto :fail

if /I "%MODE%"=="api" goto :api_only
if /I "%MODE%"=="train" goto :train_only
if /I "%MODE%"=="full" goto :full

echo [ERROR] Unknown mode: %MODE%
goto :help

:full
echo [4/6] Preparing merged dataset layout...
python -m training.dataset_loader
if errorlevel 1 goto :fail

echo [5/6] Training ViT and exporting ONNX...
python -m training.train_vit --epochs 8 --batch-size 16 --export-onnx
if errorlevel 1 goto :fail

goto :start_api

:train_only
echo [4/5] Preparing merged dataset layout...
python -m training.dataset_loader
if errorlevel 1 goto :fail

echo [5/5] Training ViT and exporting ONNX...
python -m training.train_vit --epochs 8 --batch-size 16 --export-onnx
if errorlevel 1 goto :fail

echo [DONE] Training completed.
goto :success

:api_only
echo [4/4] Starting API only mode...
if not exist "models\vit_layer2_detector.onnx" (
    echo [ERROR] ONNX model not found at models\vit_layer2_detector.onnx
    echo Run: run_layer2.cmd full
    goto :fail
)

goto :start_api

:start_api
echo [6/6] Starting FastAPI service on http://0.0.0.0:8000 ...
uvicorn api.main:app --host 0.0.0.0 --port 8000
if errorlevel 1 goto :fail

goto :success

:help
echo.
echo Usage:
echo   run_layer2.cmd full   ^(default^) - setup + prepare data + train + export onnx + start api
echo   run_layer2.cmd train             - setup + prepare data + train + export onnx
echo   run_layer2.cmd api               - setup + start api only ^(requires models\vit_layer2_detector.onnx^)
echo.
goto :eof

:success
echo [DONE] Layer-2 workflow completed successfully.
endlocal
exit /b 0

:fail
echo [FAILED] Layer-2 workflow failed.
endlocal
exit /b 1
