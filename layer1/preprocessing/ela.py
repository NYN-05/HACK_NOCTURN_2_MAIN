from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops


class ELAGenerator:
    """Generate Error Level Analysis maps for forensic feature extraction."""

    def __init__(self, jpeg_quality: int = 90, ela_scale: float = 10.0) -> None:
        self.jpeg_quality = jpeg_quality
        self.ela_scale = ela_scale

    def generate(self, image: Image.Image) -> Image.Image:
        """Create an RGB ELA image from the provided PIL image."""
        rgb = image.convert("RGB")

        buffer = BytesIO()
        rgb.save(buffer, "JPEG", quality=self.jpeg_quality)
        buffer.seek(0)

        recompressed = Image.open(buffer).convert("RGB")
        diff = ImageChops.difference(rgb, recompressed)

        ela_array = np.asarray(diff).astype(np.float32)
        max_val = float(ela_array.max())

        if max_val > 0:
            ela_array = ela_array / max_val

        ela_array = np.clip(ela_array * self.ela_scale, 0.0, 1.0)
        ela_uint8 = (ela_array * 255.0).astype(np.uint8)
        return Image.fromarray(ela_uint8, mode="RGB")


def generate_ela_map(
    image: Image.Image,
    jpeg_quality: int = 90,
    ela_scale: float = 10.0,
) -> Image.Image:
    """Convenience function for one-off ELA map generation."""
    return ELAGenerator(jpeg_quality=jpeg_quality, ela_scale=ela_scale).generate(image)


def rgb_ela_fusion(
    image: Image.Image,
    ela_generator: ELAGenerator,
    size: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """Resize image and produce aligned RGB and ELA images."""
    resized = image.convert("RGB").resize(size, Image.BILINEAR)
    ela = ela_generator.generate(resized)
    return resized, ela
