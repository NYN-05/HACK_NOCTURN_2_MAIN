# VeriSight GAN Layer

GAN-only training and inference pipeline for Layer 3 artifact detection.

## Scope

- This repository now contains only Layer 3 GAN components.
- Layers 1, 2, and 4, API orchestration, XAI modules, and frontend have been removed.
- Core outputs are:
	- `checkpoints/layer3_best.pth`
	- `checkpoints/clip_real_centroid.pt`

## Project Layout

- `layer3_train_gpu.py`: primary Layer 3 trainer.
- `layer3_gan/layer3_train_rtx4060.py`: RTX 4060 optimized trainer.
- `layer3_trained_inference.py`: inference loader and smoke test runner.
- `layer3_dataset_builder.py`: dataset build/augmentation utility.
- `layer3_gan/verisight_layer3_gan.py`: GAN detector implementation.

## Quick Start

1. Create environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

2. Optional env file:

```bash
copy .env.example .env
```

3. Build or refresh dataset:

```bash
python layer3_dataset_builder.py
```

4. Train Layer 3:

```bash
python layer3_train_gpu.py
```

5. Run inference smoke test:

```bash
python layer3_trained_inference.py
```

## Dataset Requirements

- Folder structure:
	- `dataset/real`
	- `dataset/gan_fake`
- Class mapping is fixed:
	- `real = 0`
	- `gan_fake = 1`

## Notes

- CLIP RN50 backbone is frozen and only the classifier head is trained.
- Mixed precision and cosine warm restarts are used in training.
- Inference uses checkpoint plus centroid calibration when both files exist.
