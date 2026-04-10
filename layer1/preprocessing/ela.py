from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops


class ELAGenerator:
    """Generate Error Level Analysis maps for forensic feature extraction."""

    def __init__(self, jpeg_quality: int = 90, ela_scale: float = 10.0, cache_size: int = 1024) -> None:
        self.jpeg_quality = jpeg_quality
        self.ela_scale = ela_scale
        self.cache_size = cache_size
        self._generate_from_path_cached = self._build_cached_path_generator(cache_size)

    def _build_cached_path_generator(self, cache_size: int):
        if cache_size <= 0:
            return self._generate_from_path_uncached
        return lru_cache(maxsize=cache_size)(self._generate_from_path_uncached)

    def __getstate__(self) -> dict[str, object]:
        return {
            "jpeg_quality": self.jpeg_quality,
            "ela_scale": self.ela_scale,
            "cache_size": self.cache_size,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.jpeg_quality = int(state.get("jpeg_quality", 90))
        self.ela_scale = float(state.get("ela_scale", 10.0))
        self.cache_size = int(state.get("cache_size", 1024))
        self._generate_from_path_cached = self._build_cached_path_generator(self.cache_size)

    def generate(self, image: Image.Image) -> Image.Image:
        """Create an RGB ELA image from the provided PIL image."""
        rgb = image.convert("RGB")

        buffer = BytesIO()
        rgb.save(buffer, "JPEG", quality=self.jpeg_quality)
        buffer.seek(0)

        with Image.open(buffer) as recompressed_file:
            recompressed = recompressed_file.convert("RGB")
        diff = ImageChops.difference(rgb, recompressed)

        ela_array = np.asarray(diff).astype(np.float32)
        max_val = float(ela_array.max())

        if max_val > 0:
            ela_array = ela_array / max_val

        ela_array = np.clip(ela_array * self.ela_scale, 0.0, 1.0)
        ela_uint8 = (ela_array * 255.0).astype(np.uint8)
        return Image.fromarray(ela_uint8, mode="RGB")

    def generate_from_path(self, image_path: str | Path, size: Tuple[int, int] | None = None) -> Image.Image:
        path = Path(image_path)
        stat = path.stat()
        cached = self._generate_from_path_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size, size)
        return cached.copy()

    def _generate_from_path_uncached(
        self,
        resolved_path: str,
        modified_time_ns: int,
        file_size: int,
        size: Tuple[int, int] | None,
    ) -> Image.Image:
        del modified_time_ns, file_size
        with Image.open(resolved_path) as image_file:
            rgb = image_file.convert("RGB")
        if size is not None:
            rgb = rgb.resize(size, Image.BILINEAR)
        return self.generate(rgb)


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
