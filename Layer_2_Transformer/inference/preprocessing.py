from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))

    array = np.asarray(image).astype("float32") / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return array
