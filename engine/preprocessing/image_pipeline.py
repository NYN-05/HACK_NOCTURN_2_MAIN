from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    from layer1.preprocessing.ela import ELAGenerator
    HAS_ELA = True
except ImportError:
    ELAGenerator = None  # type: ignore[assignment]
    HAS_ELA = False

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")

    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as opened:
            return opened.convert("RGB")

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] == 3:
            return Image.fromarray(array.astype(np.uint8), mode="RGB")
        if array.ndim == 3 and array.shape[2] == 4:
            return Image.fromarray(array.astype(np.uint8), mode="RGBA").convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _generate_ela(rgb_image: Image.Image, image_size: int, generate_ela: bool) -> Image.Image:
    if generate_ela and HAS_ELA and ELAGenerator is not None:
        try:
            ela_generator = ELAGenerator(jpeg_quality=90, ela_scale=10.0)
            return ela_generator.generate(rgb_image)
        except Exception:
            pass

    return Image.new("RGB", (image_size, image_size), color=(128, 128, 128))


def preprocess_all(image: Any, image_size: int = 224, generate_ela: bool = True) -> Dict[str, Any]:
    rgb_image = load_image(image).resize((image_size, image_size))
    ela_image = _generate_ela(rgb_image, image_size, generate_ela)

    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    ela_uint8 = np.asarray(ela_image, dtype=np.uint8)
    bgr_uint8 = rgb_uint8[:, :, ::-1].copy()

    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    ela_float = ela_uint8.astype(np.float32) / 255.0

    rgb_normalized = (rgb_float - RGB_MEAN) / RGB_STD
    ela_normalized = (ela_float - 0.5) / 0.5
    clip_normalized = (rgb_float - CLIP_MEAN) / CLIP_STD

    cnn_input_np = np.expand_dims(
        np.transpose(np.concatenate([rgb_normalized, ela_normalized], axis=2), (2, 0, 1)),
        axis=0,
    ).astype(np.float32)
    clip_input = np.expand_dims(np.transpose(clip_normalized, (2, 0, 1)), axis=0).astype(np.float32)

    try:
        import torch

        rgb_tensor = torch.from_numpy(np.transpose(rgb_normalized, (2, 0, 1)).copy())
        ela_tensor = torch.from_numpy(np.transpose(ela_normalized, (2, 0, 1)).copy())
        normalized = torch.cat([rgb_tensor, ela_tensor], dim=0).unsqueeze(0).contiguous()
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        rgb_tensor = None
        ela_tensor = None
        normalized = None
        normalized_np = cnn_input_np

    return {
        "rgb": rgb_image,
        "ela": ela_image,
        "rgb_array": rgb_uint8,
        "ela_array": ela_uint8,
        "bgr_array": bgr_uint8,
        "bgr": bgr_uint8,
        "ocr_input": bgr_uint8,
        "rgb_tensor": rgb_tensor,
        "ela_tensor": ela_tensor,
        "cnn_input": cnn_input_np,
        "cnn_input_np": cnn_input_np,
        "clip_input": clip_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
        "image_size": image_size,
    }


def preprocess_cnn(image: Any, image_size: int = 224, generate_ela: bool = True) -> Dict[str, Any]:
    bundle = preprocess_all(image, image_size=image_size, generate_ela=generate_ela)
    return {
        "rgb": bundle["rgb"],
        "ela": bundle["ela"],
        "rgb_array": bundle["rgb_array"],
        "ela_array": bundle["ela_array"],
        "rgb_tensor": bundle["rgb_tensor"],
        "ela_tensor": bundle["ela_tensor"],
        "cnn_input": bundle["cnn_input"],
        "cnn_input_np": bundle["cnn_input_np"],
        "normalized": bundle["normalized"],
        "normalized_np": bundle["normalized_np"],
    }


def preprocess_vit(image: Any, image_size: int = 224) -> Dict[str, Any]:
    bundle = preprocess_all(image, image_size=image_size)
    return {
        "rgb": bundle["rgb"],
        "rgb_array": bundle["rgb_array"],
        "rgb_tensor": bundle["rgb_tensor"],
        "clip_input": bundle["clip_input"],
        "normalized": bundle["normalized"],
        "normalized_np": bundle["normalized_np"],
    }


def preprocess_clip(image: Any, image_size: int = 224) -> Dict[str, Any]:
    return preprocess_vit(image, image_size=image_size)


def preprocess_yolo(image: Any, image_size: int = 640) -> Dict[str, Any]:
    bundle = preprocess_all(image, image_size=image_size)
    return {
        "rgb": bundle["rgb"],
        "rgb_array": bundle["rgb_array"],
        "rgb_tensor": bundle["rgb_tensor"],
        "yolo_input": np.expand_dims(np.transpose(np.asarray(bundle["rgb_array"], dtype=np.float32) / 255.0, (2, 0, 1)), axis=0).astype(np.float32),
        "normalized": bundle["normalized"],
        "normalized_np": bundle["normalized_np"],
    }