@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set PYTHON=python
if exist ".venv\Scripts\python.exe" (
	for /f %%i in ('.venv\Scripts\python.exe -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2^>nul') do set VENV_HAS_CUDA=%%i
	if "!VENV_HAS_CUDA!"=="1" (
		set PYTHON=.venv\Scripts\python.exe
	) else (
		for /f %%i in ('python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2^>nul') do set SYS_HAS_CUDA=%%i
		if "!SYS_HAS_CUDA!"=="1" set PYTHON=python
	)
)

echo ==============================================
echo VeriSight Layer 1 - CUDA Run-All Pipeline
echo ==============================================
echo.

set DATASET_ROOT=dataset
set OUTPUT_DIR=artifacts
set EPOCHS=30
set BATCH_SIZE=16
set IMAGE_SIZE=224
set NUM_WORKERS=8
set PREFETCH_FACTOR=4
set NUM_CPU_THREADS=8
set EVAL_BATCH_SIZE=64
set CHECKPOINT=artifacts\best_model.pth
set OUTPUT_ONNX=artifacts\verisight_layer1.onnx
set OUTPUT_IMAGE=artifacts\gradcam_overlay.png
set DEVICE=cuda
set TORCHDYNAMO_DISABLE=1
set TORCH_COMPILE_DISABLE=1

for /f %%i in ('%PYTHON% -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2^>nul') do set HAS_CUDA=%%i
if not "%HAS_CUDA%"=="1" goto no_cuda

set TRAIN_ACCEL_FLAGS=
for /f %%i in ('%PYTHON% -c "import torch; print(torch.cuda.device_count())" 2^>nul') do set GPU_COUNT=%%i
if not "%GPU_COUNT%"=="" (
	if %GPU_COUNT% GTR 1 set TRAIN_ACCEL_FLAGS=--data-parallel
)

set COMPILE_FLAGS=
set HAS_TRITON=0
if "%TRAIN_ACCEL_FLAGS%"=="--data-parallel" set COMPILE_FLAGS=

echo Auto-tuning runtime parameters from CPU/GPU resources...
for /f "tokens=1,2,3,4" %%a in ('%PYTHON% -c "import os,torch; cpu=os.cpu_count() or 8; mem=torch.cuda.get_device_properties(0).total_memory/(1024**3); nw=min(16,max(4,cpu-2)); pf=4 if nw>=8 else 2; eb=128 if mem>=20 else (96 if mem>=12 else (64 if mem>=8 else 32)); nct=max(4,min(16,cpu)); print(f'{nw} {pf} {eb} {nct}')" 2^>nul') do (
	set NUM_WORKERS=%%a
	set PREFETCH_FACTOR=%%b
	set EVAL_BATCH_SIZE=%%c
	set NUM_CPU_THREADS=%%d
)

if "%OS%"=="Windows_NT" (
	set NUM_WORKERS=0
	set PREFETCH_FACTOR=2
)

echo Tuned NUM_WORKERS=!NUM_WORKERS! PREFETCH_FACTOR=!PREFETCH_FACTOR! EVAL_BATCH_SIZE=!EVAL_BATCH_SIZE! NUM_CPU_THREADS=!NUM_CPU_THREADS!
echo Training acceleration mode: !TRAIN_ACCEL_FLAGS! !COMPILE_FLAGS! (GPUs detected: !GPU_COUNT!, Triton: !HAS_TRITON!)

echo.
echo Running complete pipeline:
echo 1^) Install dependencies
echo 2^) Train model
echo 3^) Evaluate model
echo 4^) Export ONNX
echo 5^) Inference and Grad-CAM
echo.

echo [1/5] Installing dependencies...
%PYTHON% -c "import torch, torchvision, numpy, PIL, sklearn, tqdm, matplotlib, cv2, onnx" 1>nul 2>nul
if errorlevel 1 (
	echo Dependencies missing. Installing...
	%PYTHON% -m pip install --upgrade pip
	if errorlevel 1 goto failed
	%PYTHON% -m pip install -r requirements.txt
	if errorlevel 1 goto failed
) else (
	echo Dependencies already installed. Skipping install.
)

echo [2/5] Training model...
%PYTHON% -m training.train --dataset-root "%DATASET_ROOT%" --output-dir "%OUTPUT_DIR%" --epochs %EPOCHS% --batch-size %BATCH_SIZE% --image-size %IMAGE_SIZE% --device %DEVICE% --amp !TRAIN_ACCEL_FLAGS! !COMPILE_FLAGS! --channels-last --num-workers %NUM_WORKERS% --prefetch-factor %PREFETCH_FACTOR% --num-cpu-threads %NUM_CPU_THREADS%
if errorlevel 1 (
	echo Initial training launch failed. Retrying with num_workers=0 for Windows multiprocessing stability...
	set NUM_WORKERS=0
	set PREFETCH_FACTOR=2
	%PYTHON% -m training.train --dataset-root "%DATASET_ROOT%" --output-dir "%OUTPUT_DIR%" --epochs %EPOCHS% --batch-size %BATCH_SIZE% --image-size %IMAGE_SIZE% --device %DEVICE% --amp !TRAIN_ACCEL_FLAGS! !COMPILE_FLAGS! --channels-last --num-workers %NUM_WORKERS% --prefetch-factor %PREFETCH_FACTOR% --num-cpu-threads %NUM_CPU_THREADS%
	if errorlevel 1 goto failed
)

echo [3/5] Evaluating model...
%PYTHON% -m evaluation.evaluate --dataset-root "%DATASET_ROOT%" --checkpoint "%CHECKPOINT%" --device %DEVICE% --batch-size %EVAL_BATCH_SIZE% --num-workers %NUM_WORKERS% --prefetch-factor %PREFETCH_FACTOR% --num-cpu-threads %NUM_CPU_THREADS% !COMPILE_FLAGS! --channels-last
if errorlevel 1 goto failed

echo [4/5] Exporting ONNX...
%PYTHON% -m scripts.export_onnx --checkpoint "%CHECKPOINT%" --output "%OUTPUT_ONNX%" --num-cpu-threads %NUM_CPU_THREADS%
if errorlevel 1 goto failed
echo ONNX model saved to %OUTPUT_ONNX%

echo.
set /p IMAGE_PATH=Enter image path for inference and Grad-CAM: 
if "%IMAGE_PATH%"=="" goto failed

echo [5/5] Running inference...
%PYTHON% inference.py --checkpoint "%CHECKPOINT%" --image "%IMAGE_PATH%" --device %DEVICE% !COMPILE_FLAGS! --channels-last
if errorlevel 1 goto failed

echo Generating Grad-CAM...
%PYTHON% -m scripts.gradcam_demo --checkpoint "%CHECKPOINT%" --image "%IMAGE_PATH%" --output "%OUTPUT_IMAGE%" --device %DEVICE%
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
