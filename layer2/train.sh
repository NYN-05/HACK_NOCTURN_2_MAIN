#!/bin/bash
# Layer 2 Training Script for VeriSight (Linux/macOS)
# Trains a ViT-B/16 model with optimized hyperparameters for REAL vs AI_GENERATED classification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Path configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
LAYER2_DIR="$REPO_ROOT/layer2"
TRAINING_DIR="$LAYER2_DIR/training"

# Default parameters
EPOCHS=30
BATCH_SIZE=8
WARMUP_EPOCHS=2
PATIENCE=4
BALANCED_SAMPLING=true
EXPORT_ONNX=false
PREPARE_DATASET=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --export-onnx)
            EXPORT_ONNX=true
            shift
            ;;
        --prepare-dataset)
            PREPARE_DATASET=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display header
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}VeriSight Layer 2 - Training Script${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Epochs:                $EPOCHS"
echo "  Batch Size:            $BATCH_SIZE"
echo "  Warmup Epochs:         $WARMUP_EPOCHS"
echo "  Early Stopping Patience: $PATIENCE"
echo "  Balanced Sampling:     $BALANCED_SAMPLING"
echo "  Export to ONNX:        $EXPORT_ONNX"
echo "  Prepare Dataset:       $PREPARE_DATASET"
echo ""

# Prepare dataset if requested
if [ "$PREPARE_DATASET" = true ]; then
    echo -e "${YELLOW}Preparing dataset...${NC}"
    cd "$TRAINING_DIR"
    python -c "from dataset_loader_refactored import prepare_dataset; prepare_dataset()"
    cd - > /dev/null
    echo -e "${GREEN}Dataset prepared.${NC}"
    echo ""
fi

# Build and run training command
echo -e "${CYAN}Starting training...${NC}"

CMD="python $TRAINING_DIR/train_vit.py"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --warmup-epochs $WARMUP_EPOCHS"
CMD="$CMD --patience $PATIENCE"

if [ "$BALANCED_SAMPLING" = false ]; then
    CMD="$CMD --no-balanced-sampling"
fi

if [ "$EXPORT_ONNX" = true ]; then
    CMD="$CMD --export-onnx"
fi

echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Run training
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Output files:${NC}"
    echo "  Model:              layer2/models/vit_layer2_detector.pth"
    echo "  Metrics:            layer2/models/vit_layer2_training_metrics.json"
    if [ "$EXPORT_ONNX" = true ]; then
        echo "  ONNX Model:         layer2/models/vit_layer2_detector.onnx"
    fi
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  1. Review metrics in: layer2/models/vit_layer2_training_metrics.json"
    echo "  2. Run inference:     python layer2/inference/onnx_inference.py <image_path>"
    echo "  3. Start API server:  python layer2/api/main.py"
else
    echo ""
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi
