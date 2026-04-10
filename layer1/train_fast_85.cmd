@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=py"
set "PYTHON_ARGS=-3"
set "DATASET_ROOT=%REPO_ROOT%\Data"
set "OUTPUT_DIR=%~dp0artifacts_full_data"
set "PYTHONUNBUFFERED=1"
if defined PYTHONPATH (
	set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
) else (
	set "PYTHONPATH=%REPO_ROOT%"
)

if not exist "%REPO_ROOT%\requirements.lock.txt" (
	echo [ERROR] Missing dependency lockfile: %REPO_ROOT%\requirements.lock.txt
	exit /b 1
)

echo [SETUP] Verifying CUDA-enabled PyTorch...
for /f "usebackq tokens=1-5 delims=|" %%i in (`py -3 -c "import torch; cuda_available = int(torch.cuda.is_available()); torch_version = torch.__version__; cuda_version = torch.version.cuda or ''; device_count = torch.cuda.device_count() if cuda_available else 0; device_name = torch.cuda.get_device_name(0) if cuda_available and device_count else ''; print(f'{cuda_available}|{torch_version}|{cuda_version}|{device_count}|{device_name}')" 2^>nul`) do (
	set "HAS_CUDA=%%i"
	set "TORCH_VERSION=%%j"
	set "TORCH_CUDA_VERSION=%%k"
	set "GPU_COUNT=%%l"
	set "CUDA_DEVICE_NAME=%%m"
)

if not "%HAS_CUDA%"=="1" (
	echo [ERROR] CUDA is not available in the current Python environment.
	echo         Install the CUDA build from requirements.lock.txt and rerun this script.
	exit /b 1
)

if not defined TORCH_VERSION set "TORCH_VERSION=unknown"
if not defined TORCH_CUDA_VERSION set "TORCH_CUDA_VERSION=unknown"
echo [CUDA] PyTorch !TORCH_VERSION! built with CUDA !TORCH_CUDA_VERSION!.
if defined CUDA_DEVICE_NAME echo [CUDA] Using GPU !GPU_COUNT!: !CUDA_DEVICE_NAME!

echo [TRAIN] Starting fast Layer 1 training with early stopping...
echo [TRAIN] Output directory: %OUTPUT_DIR%
echo [TRAIN] Dataset root: %DATASET_ROOT%

"%PYTHON_EXE%" %PYTHON_ARGS% -m training.train ^
	--dataset-root "%DATASET_ROOT%" ^
	--output-dir "%OUTPUT_DIR%" ^
	--device cuda ^
	--epochs 20 ^
	--batch-size 16 ^
	--image-size 224 ^
	--num-workers 4 ^
	--prefetch-factor 2 ^
	--num-cpu-threads 8 ^
	--lr 1e-4 ^
	--weight-decay 1e-4 ^
	--patience 4 ^
	--min-delta 0.002 ^
	--grad-clip 1.0 ^
	--label-smoothing 0.05 ^
	--freeze-backbone-epochs 3 ^
	--balanced-sampling ^
	--class-weighted-loss ^
	--ela-cache-size 1024 ^
	--seed 42 ^
	--amp ^
	--channels-last ^
	--resume

if errorlevel 1 (
	echo [FAILED] Training run failed.
	exit /b 1
)

echo.
echo [DONE] Fast training finished. Check %OUTPUT_DIR%\best_model.pth and %OUTPUT_DIR%\test_metrics.json
exit /b 0