from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

Path('dataset/real').mkdir(parents=True, exist_ok=True)
Path('dataset/gan_fake').mkdir(parents=True, exist_ok=True)

print('Downloading batgre/CIFAKE ...')
ds = load_dataset('batgre/CIFAKE', split='train', streaming=True)

n_real, n_gan, MAX = 0, 0, 4000
for r in tqdm(ds, total=MAX*2, desc='Saving'):
    if n_real >= MAX and n_gan >= MAX:
        break
    try:
        img = r['image'].convert('RGB').resize((224,224))
        lbl = int(r['label'])
        if lbl == 1 and n_real < MAX:
            img.save(f'dataset/real/r_{n_real:06d}.jpg', quality=92)
            n_real += 1
        elif lbl == 0 and n_gan < MAX:
            img.save(f'dataset/gan_fake/g_{n_gan:06d}.jpg', quality=92)
            n_gan += 1
    except Exception:
        continue

print(f'Done. Real: {n_real} | GAN: {n_gan}')
