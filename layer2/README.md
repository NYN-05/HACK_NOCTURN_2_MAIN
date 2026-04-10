# VeriSight Layer-2 Transformer Module

Layer-2 microservice for REAL vs AI_GENERATED image detection using a pretrained Hugging Face ViT-B/16 model (`google/vit-base-patch16-224`) that is fine-tuned on the VeriSight dataset.

## Project Tree

```text
Layer_2_Transformer/
|-- dataset/
|-- models/
|-- training/
|   |-- train_vit.py
|   `-- dataset_loader.py
|-- inference/
|   |-- onnx_inference.py
|   `-- preprocessing.py
|-- api/
|   |-- main.py
|   `-- router.py
|-- utils/
|   |-- metrics.py
|   `-- config.py
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Dataset Input

Accepted source layouts:

- Full cleaned data corpus: `cleaned_data/images_complete/` recursively
- Metadata-backed labeled subset: `cleaned_data/metadata/unified_groundtruth.csv` with files under `cleaned_data/images/`
- Legacy `Data/` layouts are still accepted as a fallback, but the cleaned_data tree is the default source of truth

Prepared output layout (if rebuild is needed):

- `dataset/train/real`, `dataset/train/fake`
- `dataset/val/real`, `dataset/val/fake`
- `dataset/test/real`, `dataset/test/fake`

Label mapping:

- `REAL = 0`
- `AI_GENERATED = 1`

## Install

```bash
pip install -r requirements.txt
```

## Prepare Dataset

```bash
python -c "from layer2.training.dataset_loader_refactored import prepare_dataset; prepare_dataset('cleaned_data')"
```

## Train + Export ONNX

```bash
python -m layer2.training.train_vit --prepare-dataset --export-onnx
```

By default the trainer loads `google/vit-base-patch16-224`, freezes the encoder briefly for head warmup, then progressively unfreezes the top transformer blocks.

The dataset loader now:
- Keeps duplicate and source groups together during train/val/test splitting (prevents data leakage)
- Uses robust per-dataset group ID derivation to handle CASIA2/COMOFOD/MICC-F220 naming schemes
- Logs class and dataset distribution across splits for validation transparency

Training improvements:
- **MixUp/CutMix enabled by default** (α=0.3/0.5) for blended forgery robustness
- **Enhanced dropout** (0.15 attention, 0.15 hidden) to reduce overfitting to synthetic artifacts
- **Softer class weighting** to prevent gradient oscillation when paired with balanced sampling
- **F1-first validation** with precision tie-breaking (not recall-first) to prevent false "positive" detection rate
- **Improved LR warmup** (0.1 → 1.0) for smoother unfreezing of backbone blocks
- **Better threshold calibration** logged with uncertainty bounds

Useful overrides:

- `--no-mixup` to disable MixUp/CutMix (enabled by default)
- `--batch-size 4` if memory constrained
- `--no-balanced-sampling` for plain shuffled sampling
- `--use-labeled-subset` for smaller metadata-backed subset
- `--drop-rate 0.1 --drop-path-rate 0.1` to adjust regularization
- `--patience 4` to control early stopping

By default, layer 2 walks the entire `cleaned_data/images_complete/` tree and splits the full corpus for training, validation, and testing, but the split is now group-aware rather than image-wise.

Outputs:

- `models/vit_layer2_detector.pth`
- `models/vit_layer2_detector.onnx`

## Inference Function

```python
from inference.onnx_inference import detect_ai_generated
print(detect_ai_generated("sample.jpg"))
```

Expected output format:

```json
{
  "vit_score": 82,
  "label": "AI_GENERATED"
}
```

## Run API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoint:

- `POST /api/v1/transformer-detect` with `multipart/form-data` field `image`

Response:

```json
{
  "vit_score": 82,
  "label": "AI_GENERATED",
  "processing_time_ms": 120
}
```

## Docker

```bash
docker build -t verisight-layer2 .
docker run -p 8000:8000 verisight-layer2
```
