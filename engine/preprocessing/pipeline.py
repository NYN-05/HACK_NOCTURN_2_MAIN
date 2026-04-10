from .image_pipeline import load_image, preprocess_all

__all__ = ["load_image", "preprocess_all"] 

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

# Try to import ELA generator, gracefully degrade if not available
try:
    from layer1.preprocessing.ela import ELAGenerator
    HAS_ELA = True
except ImportError:
    HAS_ELA = False


# Standard normalization values
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_image(image: Any) -> Image.Image:
    """
    Load image from various input formats.
    Supports: PIL Image, file path (str/Path), bytes, numpy array.
    """
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


def preprocess_cnn(
    image: Any,
    image_size: int = 224,
    generate_ela: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess image for CNN+ELA layer.
    Returns: RGB array, ELA array, normalized tensors.
    """
    rgb_image = load_image(image).resize((image_size, image_size))

    # Generate ELA if available
    ela_image = None
    if generate_ela and HAS_ELA:
        try:
            ela_gen = ELAGenerator(jpeg_quality=90, ela_scale=10.0)
            ela_image = ela_gen.generate(rgb_image)
        except Exception:
            ela_image = None

    if ela_image is None:
        # Fallback: create placeholder ELA
        ela_image = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))

    # Convert to arrays
    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    ela_uint8 = np.asarray(ela_image, dtype=np.uint8)

    # Normalize
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    ela_float = ela_uint8.astype(np.float32) / 255.0

    rgb_normalized = (rgb_float - RGB_MEAN) / RGB_STD
    ela_normalized = (ela_float - 0.5) / 0.5

    # Create CNN input (6 channels: RGB + ELA)
    cnn_input = np.concatenate([rgb_normalized, ela_normalized], axis=2)
    cnn_input = np.transpose(cnn_input, (2, 0, 1))  # CHW
    cnn_input = np.expand_dims(cnn_input, axis=0).astype(np.float32)  # BCHW

    # Try to create torch tensors
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
        normalized_np = cnn_input

    return {
        "rgb": rgb_image,
        "ela": ela_image,
        "rgb_array": rgb_uint8,
        "ela_array": ela_uint8,
        "rgb_tensor": rgb_tensor,
        "ela_tensor": ela_tensor,
        "cnn_input": cnn_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
    }


def preprocess_vit(
    image: Any,
    image_size: int = 224,
) -> Dict[str, Any]:
    """
    Preprocess image for Vision Transformer layer.
    Returns: RGB tensor with ImageNet normalization.
    """
    rgb_image = load_image(image).resize((image_size, image_size))
    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    rgb_normalized = (rgb_float - RGB_MEAN) / RGB_STD

    vit_input = np.transpose(rgb_normalized, (2, 0, 1))  # CHW
    vit_input = np.expand_dims(vit_input, axis=0).astype(np.float32)  # BCHW

    try:
        import torch
        tensor = torch.from_numpy(np.transpose(rgb_normalized, (2, 0, 1)).copy())
        normalized = tensor.unsqueeze(0).contiguous()
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        tensor = None
        normalized = None
        normalized_np = vit_input

    return {
        "rgb": rgb_image,
        "rgb_array": rgb_uint8,
        "rgb_tensor": tensor,
        "vit_input": vit_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
    }


def preprocess_clip(
    image: Any,
    image_size: int = 224,
) -> Dict[str, Any]:
    """
    Preprocess image for CLIP model.
    Returns: RGB tensor with CLIP normalization.
    """
    rgb_image = load_image(image).resize((image_size, image_size))
    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0

    # CLIP uses specific normalization
    rgb_normalized = (rgb_float - CLIP_MEAN) / CLIP_STD

    clip_input = np.transpose(rgb_normalized, (2, 0, 1))  # CHW
    clip_input = np.expand_dims(clip_input, axis=0).astype(np.float32)  # BCHW

    try:
        import torch
        tensor = torch.from_numpy(np.transpose(rgb_normalized, (2, 0, 1)).copy())
        normalized = tensor.unsqueeze(0).contiguous()
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        tensor = None
        normalized = None
        normalized_np = clip_input

    return {
        "rgb": rgb_image,
        "rgb_array": rgb_uint8,
        "rgb_tensor": tensor,
        "clip_input": clip_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
    }


def preprocess_yolo(
    image: Any,
    image_size: int = 640,
) -> Dict[str, Any]:
    """
    Preprocess image for YOLO detection layer.
    Returns: RGB array normalized for object detection.
    """
    rgb_image = load_image(image).resize((image_size, image_size))
    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0

    yolo_input = np.transpose(rgb_float, (2, 0, 1))  # CHW
    yolo_input = np.expand_dims(yolo_input, axis=0).astype(np.float32)  # BCHW

    try:
        import torch
        tensor = torch.from_numpy(np.transpose(rgb_float, (2, 0, 1)).copy())
        normalized = tensor.unsqueeze(0).contiguous()
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        tensor = None
        normalized = None
        normalized_np = yolo_input

    return {
        "rgb": rgb_image,
        "rgb_array": rgb_uint8,
        "rgb_tensor": tensor,
        "yolo_input": yolo_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
    }


def preprocess_all(
    image: Any,
    image_size: int = 224,
    generate_ela: bool = True,
) -> Dict[str, Any]:
    """
    Complete preprocessing for all layers.
    Returns consolidated dict with all preprocessing formats.
    """
    rgb_image = load_image(image).resize((image_size, image_size))

    # Generate ELA if available
    ela_image = None
    if generate_ela and HAS_ELA:
        try:
            ela_gen = ELAGenerator(jpeg_quality=90, ela_scale=10.0)
            ela_image = ela_gen.generate(rgb_image)
        except Exception:
            ela_image = None

    if ela_image is None:
        ela_image = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))

    rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
    ela_uint8 = np.asarray(ela_image, dtype=np.uint8)
    bgr_uint8 = rgb_uint8[:, :, ::-1].copy()

    # Float conversions
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    ela_float = ela_uint8.astype(np.float32) / 255.0

    # Normalizations
    rgb_normalized = (rgb_float - RGB_MEAN) / RGB_STD
    ela_normalized = (ela_float - 0.5) / 0.5

    # CNN input (6 channels)
    cnn_input = np.expand_dims(
        np.transpose(np.concatenate([rgb_normalized, ela_normalized], axis=2), (2, 0, 1)),
        axis=0,
    ).astype(np.float32)

    # CLIP input (normalized)
    clip_normalized = (rgb_float - CLIP_MEAN) / CLIP_STD
    clip_input = np.expand_dims(np.transpose(clip_normalized, (2, 0, 1)), axis=0).astype(np.float32)

    # Try torch conversion
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
        normalized_np = cnn_input

    return {
        "rgb": rgb_image,
        "ela": ela_image,
        "rgb_array": rgb_uint8,
        "ela_array": ela_uint8,
        "bgr_array": bgr_uint8,
        "rgb_tensor": rgb_tensor,
        "ela_tensor": ela_tensor,
        "cnn_input": cnn_input,
        "clip_input": clip_input,
        "normalized": normalized,
        "normalized_np": normalized_np,
    }
