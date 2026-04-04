from __future__ import annotations

import argparse
from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageFilter


def augment_image(img_path: Path, output_dir: Path, n: int = 5) -> None:
    img = Image.open(img_path).convert("RGB")

    for i in range(n):
        aug = img.copy()

        if random.random() > 0.5:
            aug = aug.rotate(random.uniform(-5, 5), expand=False)
        if random.random() > 0.5:
            aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

        quality = random.choice([60, 75, 85, 95])
        out_name = f"aug_{i}_{img_path.stem}.jpg"
        aug.save(output_dir / out_name, "JPEG", quality=quality)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset augmenter")
    parser.add_argument("--input", required=True, help="Directory with source images")
    parser.add_argument("--output", required=True, help="Directory for augmented images")
    parser.add_argument("--n", type=int, default=5, help="Augmentations per image")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in input_dir.glob("*.*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            augment_image(path, output_dir, n=args.n)

    print("Augmentation complete")


if __name__ == "__main__":
    main()
