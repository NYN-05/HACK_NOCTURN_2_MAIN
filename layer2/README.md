# VeriSight Layer-2 Transformer Module

Layer-2 microservice for REAL vs AI_GENERATED image detection using `google/vit-base-patch16-224`.

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

- Direct processed splits: `Data/train/{real,fake}`, `Data/val/{real,fake}`, `Data/test/{real,fake}`
- CIFAKE-style splits anywhere under Data: `*/train/{REAL|real}/{FAKE|fake}` and `*/test/{REAL|real}/{FAKE|fake}`
- Optional ImageNet Mini folder (`imagenet_mini` or `imagenet-mini`) for extra REAL samples

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
python -m training.dataset_loader
```

## Train + Export ONNX

```bash
python -m training.train_vit --prepare-dataset --epochs 8 --batch-size 16 --export-onnx
```

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
