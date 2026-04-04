from typing import Any

import numpy as np

from engine.preprocessing.shared_pipeline import preprocess_all


def preprocess_image(image: Any, image_size: int = 224) -> np.ndarray:
    return np.asarray(preprocess_all(image, image_size=image_size)["clip_input"], dtype=np.float32)
