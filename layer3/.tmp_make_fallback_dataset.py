from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

real_dir = Path('dataset/real')
fake_dir = Path('dataset/gan_fake')
real_dir.mkdir(parents=True, exist_ok=True)
fake_dir.mkdir(parents=True, exist_ok=True)

for p in real_dir.glob('*.jpg'):
    p.unlink()
for p in fake_dir.glob('*.jpg'):
    p.unlink()

rng = np.random.default_rng(42)
H = W = 224
N = 300

# Real-like: smooth gradients + mild natural noise.
for i in range(N):
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    base = np.stack([
        0.55 * xv + 0.45 * yv,
        0.50 * np.sin(2 * np.pi * xv) * 0.2 + 0.5,
        0.55 * yv + 0.25 * xv,
    ], axis=-1)
    noise = rng.normal(0, 0.05, size=(H, W, 3)).astype(np.float32)
    arr = np.clip(base + noise, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB').filter(ImageFilter.GaussianBlur(radius=0.6))
    img.save(real_dir / f'r_{i:06d}.jpg', quality=92)

# GAN-like: high-frequency artifacts + blocky/checker textures.
for i in range(N):
    grid = ((np.indices((H, W)).sum(axis=0) // rng.integers(2, 6)) % 2).astype(np.float32)
    ch1 = np.roll(grid, rng.integers(0, 20), axis=0)
    ch2 = np.roll(grid, rng.integers(0, 20), axis=1)
    ch3 = (ch1 * 0.6 + ch2 * 0.4)
    arr = np.stack([ch1, ch2, ch3], axis=-1)
    arr += rng.normal(0, 0.10, size=(H, W, 3)).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    # Simulate compression artifacts by re-encoding at lower quality occasionally.
    img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')
    q = 65 if (i % 3 == 0) else 80
    img.save(fake_dir / f'g_{i:06d}.jpg', quality=q)

print(f'Generated fallback dataset. Real={len(list(real_dir.glob("*.jpg")))} GAN={len(list(fake_dir.glob("*.jpg")))}')
