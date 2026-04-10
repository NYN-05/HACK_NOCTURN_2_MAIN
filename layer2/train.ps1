# Layer 2 Training Script for VeriSight
# Trains a ViT-B/16 model with optimized hyperparameters for REAL vs AI_GENERATED classification

$ErrorActionPreference = "Stop"

# Path configuration
$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$LAYER2_DIR = $PSScriptRoot
$TRAINING_DIR = Join-Path $LAYER2_DIR "training"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "VeriSight Layer 2 - Training Script" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# Parse command-line arguments
$epochs = 30                          # Default: 30 epochs (was 12)
$batch_size = 8                       # Batch size
$warmup_epochs = 2                    # Head-only warmup phase
$patience = 4                         # Early stopping patience
$use_balanced_sampling = $true        # Use weighted sampling for class balance
$export_onnx = $false                 # Optional: export to ONNX
$prepare_dataset = $false             # Optional: rebuild dataset

# Parse arguments from command line
for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]
    
    if ($arg -eq "--epochs" -and $i + 1 -lt $args.Count) {
        $epochs = [int]$args[$i + 1]
        $i++
    }
    elseif ($arg -eq "--batch-size" -and $i + 1 -lt $args.Count) {
        $batch_size = [int]$args[$i + 1]
        $i++
    }
    elseif ($arg -eq "--patience" -and $i + 1 -lt $args.Count) {
        $patience = [int]$args[$i + 1]
        $i++
    }
    elseif ($arg -eq "--export-onnx") {
        $export_onnx = $true
    }
    elseif ($arg -eq "--prepare-dataset") {
        $prepare_dataset = $true
    }
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Epochs:                $epochs"
Write-Host "  Batch Size:            $batch_size"
Write-Host "  Warmup Epochs:         $warmup_epochs"
Write-Host "  Early Stopping Patience: $patience"
Write-Host "  Balanced Sampling:     $use_balanced_sampling"
Write-Host "  Export to ONNX:        $export_onnx"
Write-Host "  Prepare Dataset:       $prepare_dataset"
Write-Host ""

# Dataset preparation (optional)
if ($prepare_dataset) {
    Write-Host "Preparing dataset..." -ForegroundColor Yellow
    Push-Location $TRAINING_DIR
    python -c "from dataset_loader_refactored import prepare_dataset; prepare_dataset()"
    Pop-Location
    Write-Host "Dataset prepared." -ForegroundColor Green
    Write-Host ""
}

# Build command
$cmd = @(
    "python",
    "$TRAINING_DIR\train_vit.py",
    "--epochs", $epochs.ToString(),
    "--batch-size", $batch_size.ToString(),
    "--warmup-epochs", $warmup_epochs.ToString(),
    "--patience", $patience.ToString()
)

# Add optional flags
if (-not $use_balanced_sampling) {
    $cmd += "--no-balanced-sampling"
}

# Export to ONNX after training
if ($export_onnx) {
    $cmd += "--export-onnx"
}

Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host "Command: $($cmd -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run training
Invoke-Expression ($cmd -join ' ')
$trainExitCode = $LASTEXITCODE

if ($trainExitCode -eq 0) {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "[OK] Training completed successfully!" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output files:" -ForegroundColor Yellow
    Write-Host "  Model:              layer2/models/vit_layer2_detector.pth"
    Write-Host "  Metrics:            layer2/models/vit_layer2_training_metrics.json"
    if ($export_onnx) {
        Write-Host "  ONNX Model:         layer2/models/vit_layer2_detector.onnx"
    }
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review metrics in: layer2/models/vit_layer2_training_metrics.json"
    Write-Host "  2. Run inference:     python layer2/inference/onnx_inference.py [image_path]"
    Write-Host "  3. Start API server:  python layer2/api/main.py"
}
else {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "[FAIL] Training failed with exit code $trainExitCode" -ForegroundColor Red
    Write-Host "=================================================================" -ForegroundColor Red
    exit 1
}
