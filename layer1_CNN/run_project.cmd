@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==============================================
echo VeriSight Layer 1 - CUDA Run-All Pipeline
echo ==============================================
echo.

set DATASET_ROOT=dataset
set OUTPUT_DIR=artifacts
set EPOCHS=30
set BATCH_SIZE=16
set IMAGE_SIZE=224
set CHECKPOINT=artifacts\best_model.pth
set OUTPUT_ONNX=artifacts\verisight_layer1.onnx
set OUTPUT_IMAGE=artifacts\gradcam_overlay.png
set DEVICE=cuda

for /f %%i in ('python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2^>nul') do set HAS_CUDA=%%i
if not "%HAS_CUDA%"=="1" goto no_cuda

echo.
echo Running complete pipeline:
echo 1^) Install dependencies
echo 2^) Train model
echo 3^) Evaluate model
echo 4^) Export ONNX
echo 5^) Inference and Grad-CAM
echo.

echo [1/5] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo [2/5] Training model...
python -m training.train --dataset-root "%DATASET_ROOT%" --output-dir "%OUTPUT_DIR%" --epochs %EPOCHS% --batch-size %BATCH_SIZE% --image-size %IMAGE_SIZE% --device %DEVICE% --amp
if errorlevel 1 goto failed

echo [3/5] Evaluating model...
python -m evaluation.evaluate --dataset-root "%DATASET_ROOT%" --checkpoint "%CHECKPOINT%" --device %DEVICE%
if errorlevel 1 goto failed

echo [4/5] Exporting ONNX...
python scripts\export_onnx.py --checkpoint "%CHECKPOINT%" --output "%OUTPUT_ONNX%"
if errorlevel 1 goto failed
echo ONNX model saved to %OUTPUT_ONNX%

echo.
set /p IMAGE_PATH=Enter image path for inference and Grad-CAM: 
if "%IMAGE_PATH%"=="" goto failed

echo [5/5] Running inference...
python inference.py --checkpoint "%CHECKPOINT%" --image "%IMAGE_PATH%" --device %DEVICE%
if errorlevel 1 goto failed

echo Generating Grad-CAM...
python scripts\gradcam_demo.py --checkpoint "%CHECKPOINT%" --image "%IMAGE_PATH%" --output "%OUTPUT_IMAGE%" --device %DEVICE%
if errorlevel 1 goto failed
echo Grad-CAM saved to %OUTPUT_IMAGE%

echo.
echo CUDA pipeline completed successfully.
goto done

:no_cuda
echo ERROR: CUDA device not available. This script is CUDA-only.
goto done

:failed
echo.
echo Pipeline stopped because one step failed.
goto done

:done
echo.
echo Done.
endlocal
