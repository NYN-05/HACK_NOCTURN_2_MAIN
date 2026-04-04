from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from layer1.preprocessing.ela import ELAGenerator

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_pil_image(image: Any) -> Image.Image:
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

    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def preprocess_all(image: Any, image_size: int = 224) -> Dict[str, Any]:
    rgb_image = _to_pil_image(image).resize((image_size, image_size))
    ela_image = ELAGenerator(jpeg_quality=90, ela_scale=10.0).generate(rgb_image)

    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    ela_uint8 = np.asarray(ela_image, dtype=np.uint8)

    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    ela_float = ela_uint8.astype(np.float32) / 255.0

    rgb_normalized = (rgb_float - RGB_MEAN) / RGB_STD
    ela_normalized = (ela_float - 0.5) / 0.5
    cnn_input_np = np.expand_dims(
        np.transpose(np.concatenate([rgb_normalized, ela_normalized], axis=2), (2, 0, 1)),
        axis=0,
    ).astype(np.float32)

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        torch = None

    if torch is not None:
        rgb_tensor = torch.from_numpy(np.transpose(rgb_normalized, (2, 0, 1)).copy())
        ela_tensor = torch.from_numpy(np.transpose(ela_normalized, (2, 0, 1)).copy())
        normalized = torch.cat([rgb_tensor, ela_tensor], dim=0).unsqueeze(0).contiguous()
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
    else:
        rgb_tensor = None
        ela_tensor = None
        normalized = None
        normalized_np = cnn_input_np

    clip_input = np.expand_dims(np.transpose((rgb_float - 0.5) / 0.5, (2, 0, 1)), axis=0).astype(np.float32)
    bgr_uint8 = rgb_uint8[:, :, ::-1].copy()

    return {
        "rgb": rgb_image,
        "ela": ela_image,
        "rgb_array": rgb_uint8,
        "ela_array": ela_uint8,
        "rgb_tensor": rgb_tensor,
        "ela_tensor": ela_tensor,
        "normalized": normalized,
        "cnn_input_np": normalized_np,
        "clip_input": clip_input,
        "bgr": bgr_uint8,
        "ocr_input": bgr_uint8,
        "image_size": image_size,
    }