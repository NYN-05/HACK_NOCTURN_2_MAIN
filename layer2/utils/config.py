from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Raw source dataset locations
CIFAKE_DIR = BASE_DIR / "dataset" / "cifake"
IMAGENET_MINI_DIR = BASE_DIR / "dataset" / "imagenet_mini"

# Final processed dataset root expected by the training pipeline
PROCESSED_DATASET_DIR = BASE_DIR / "dataset"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "vit_layer2_detector.pth"
ONNX_MODEL_PATH = MODELS_DIR / "vit_layer2_detector.onnx"

PRETRAINED_MODEL_NAME = "google/vit-base-patch16-224"

LABEL_TO_ID = {
    "real": 0,
    "fake": 1,
}
ID_TO_LABEL = {
    0: "REAL",
    1: "AI_GENERATED",
}

RANDOM_SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 2
