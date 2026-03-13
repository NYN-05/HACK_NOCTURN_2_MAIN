# VeriSight Layer 1 - CNN Image Forensics Module

This repository contains Layer 1 of VeriSight: a PyTorch CNN module for detecting manipulated images used in refund fraud.

## SECTION 1 - System Architecture Explanation

### End-to-end pipeline

1. Image preprocessing
   - Load RGB image.
   - Resize to 224x224.
   - Apply geometric augmentation during training (flip/rotation) to improve robustness.

2. ELA (Error Level Analysis)
   - Recompress image as JPEG at fixed quality (default 90).
   - Compute absolute pixel difference between original and recompressed image.
   - Normalize and scale the difference map to highlight local compression inconsistency.

3. RGB + ELA channel fusion
   - Convert RGB image to tensor (3 channels).
   - Convert ELA map to tensor (3 channels).
   - Concatenate tensors to produce a 6-channel tensor.

4. CNN backbone
   - EfficientNet-B4 pretrained on ImageNet.
   - First convolution modified from 3 input channels to 6 input channels.
   - New channels initialized from pretrained stem statistics.

5. Classification head
   - Dropout + fully connected layer with 2 output logits:
     - class 0: authentic
     - class 1: manipulated

6. Manipulation probability output
   - Apply softmax over logits.
   - Use probability of class 1 as forgery probability.

7. Score conversion
   - cnn_score = forgery_probability * 100
   - prediction = manipulated if probability >= 0.5 else authentic

## SECTION 2 - Project Folder Structure

- data/
- models/
- preprocessing/
- training/
- evaluation/
- utils/
- scripts/
- configs/
- artifacts/

## SECTION 3 - Dataset Loader

Implemented in data/dataset.py.

Features:
- Supports CASIA, CoMoFoD, CG-1050 (and generic fallback scanning).
- Auto-labels authentic/manipulated from folder/file naming patterns.
- Excludes masks/ground-truth files.
- Performs stratified train/val/test split.
- Resizes to 224x224.
- Applies data augmentation for train split.
- Generates ELA maps per sample.
- Produces fused 6-channel RGB+ELA tensors.

## SECTION 4 - ELA Preprocessing Module

Implemented in preprocessing/ela.py.

Features:
- JPEG recompression.
- Absolute difference image generation.
- Normalized ELA map scaling.
- Utility for aligned RGB and ELA production.

## SECTION 5 - CNN Model Architecture

Implemented in models/efficientnet_forensics.py.

Model details:
- EfficientNet-B4 backbone with ImageNet pretrained weights.
- First conv adapted to 6 channels.
- Binary classification head.
- Softmax-based manipulation probability through helper method.

## SECTION 6 - Training Pipeline

Implemented in training/train.py.

Includes:
- Full PyTorch training loop.
- AdamW optimizer.
- CrossEntropyLoss.
- ReduceLROnPlateau scheduler.
- Validation every epoch.
- Best-checkpoint saving.
- Early stopping.
- Training history and test metrics export to JSON.

## SECTION 7 - Evaluation

Implemented in evaluation/metrics.py and evaluation/evaluate.py.

Metrics:
- accuracy
- precision
- recall
- F1 score
- confusion matrix

## SECTION 8 - Explainability

Implemented in evaluation/gradcam.py and scripts/gradcam_demo.py.

Features:
- Grad-CAM generation from deep EfficientNet feature layer.
- Heatmap overlay generation for manipulated-region visualization.

## SECTION 9 - Inference Module

Implemented in inference.py.

Output format:

{
  "cnn_score": 0-100,
  "forgery_probability": float,
  "prediction": "authentic" | "manipulated"
}

## SECTION 10 - Optimization Recommendations

1. ONNX export
   - Use scripts/export_onnx.py for runtime portability.
   - Prefer opset 17+.

2. Quantization
   - For CPU deployment, use post-training dynamic quantization or static quantization after calibration.
   - Validate F1 drop before production rollout.

3. GPU vs CPU inference
   - GPU: high throughput, lower latency for batch scoring.
   - CPU: easier deployment and lower cost for low QPS.
   - Use batch inference and pinned memory for GPU pipelines.

## Installation

1. Create and activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Training Instructions

Example training command:

python -m training.train \
  --dataset-root dataset \
  --output-dir artifacts \
  --epochs 30 \
  --batch-size 16 \
  --image-size 224

## Evaluation Instructions

python -m evaluation.evaluate \
  --dataset-root dataset \
  --checkpoint artifacts/best_model.pth

## Inference Example

python inference.py \
  --checkpoint artifacts/best_model.pth \
  --image path/to/sample.jpg \
  --device cpu

## Grad-CAM Example

python scripts/gradcam_demo.py \
  --checkpoint artifacts/best_model.pth \
  --image path/to/sample.jpg \
  --output artifacts/gradcam_overlay.png

## ONNX Export Example

python scripts/export_onnx.py \
  --checkpoint artifacts/best_model.pth \
  --output artifacts/verisight_layer1.onnx
