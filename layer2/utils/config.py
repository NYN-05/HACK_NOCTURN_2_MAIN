from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_data_root() -> Path:
    repo_root = BASE_DIR.parent
    for folder_name in ("DATA", "Data", "data"):
        candidate = repo_root / folder_name
        if candidate.exists():
            return candidate
    return repo_root / "DATA"


DATA_ROOT = _resolve_data_root()


def _resolve_cifake_dir() -> Path:
    candidate = DATA_ROOT / "cifake"
    if candidate.exists():
        return candidate
    return DATA_ROOT


def _resolve_imagenet_mini_dir() -> Path:
    for folder_name in ("imagenet_mini", "imagenet-mini"):
        candidate = DATA_ROOT / folder_name
        if candidate.exists():
            return candidate
    return DATA_ROOT / "imagenet_mini"


# Raw source dataset locations
CIFAKE_DIR = _resolve_cifake_dir()
IMAGENET_MINI_DIR = _resolve_imagenet_mini_dir()

# Final processed dataset root expected by the training pipeline
PROCESSED_DATASET_DIR = DATA_ROOT

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
