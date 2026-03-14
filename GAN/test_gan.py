"""
VeriSight Layer 3 — Sanity & Smoke Test Suite
Run BEFORE training: python test_gan.py
Run AFTER training:  python test_gan.py --post
All tests must PASS before proceeding to train_gan.py
"""

import os
import sys
import time
import tempfile
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from torch.amp import GradScaler as TorchGradScaler
    AMP_API_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler
    AMP_API_NEW = False

POST_MODE = "--post" in sys.argv
FAILED = False


def _fail(name, reason):
    """Mark test as failed and print reason."""
    global FAILED
    FAILED = True
    print(f"  [FAIL] {name}: {reason}")


def _pass(name):
    """Print successful test result."""
    print(f"  [PASS] {name}")


def test_torch_cuda():
    """Verify PyTorch finds the GPU or reports CPU gracefully."""
    name = "test_torch_cuda"
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            print(f"    GPU: {props.name} | VRAM: {vram_gb:.1f} GB")
            if vram_gb < 2.0:
                print("    VRAM < 2 GB detected, but CPU fallback remains supported.")
        else:
            print("    No GPU found — CPU fallback will be used.")
        _pass(name)
    except Exception as e:
        print(f"    CUDA query error handled gracefully: {e}")
        _pass(name)


def test_clip_import():
    """Verify open_clip is installed and CLIP-RN50 can be loaded."""
    name = "test_clip_import"
    try:
        import open_clip

        m, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _pass(name)
    except ImportError:
        _fail(name, "pip install open-clip-torch")
    except Exception as e:
        _fail(name, str(e))


def test_clip_normalization():
    """Verify CLIP normalization constants are correct (not ImageNet values)."""
    name = "test_clip_normalization"
    try:
        expected_mean = [0.48145466, 0.4578275, 0.40821073]
        expected_std = [0.26862954, 0.26130258, 0.27577711]
        wrong_mean = [0.485, 0.456, 0.406]

        if expected_mean == wrong_mean:
            _fail(name, "CLIP mean unexpectedly equals ImageNet mean")
            return

        x = torch.rand(1, 3, 224, 224, dtype=torch.float32)
        norm = transforms.Normalize(expected_mean, expected_std)
        y = norm(x)

        if y.shape != (1, 3, 224, 224):
            _fail(name, f"Bad shape after normalize: {tuple(y.shape)}")
            return
        if y.dtype != torch.float32:
            _fail(name, f"Bad dtype after normalize: {y.dtype}")
            return

        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_focal_loss():
    """Verify FocalLoss produces valid loss values and gradients."""
    name = "test_focal_loss"
    try:
        mod = importlib.import_module("train_gan")
        criterion = mod.FocalLoss()

        pred = torch.tensor([[0.8], [0.2], [0.9], [0.1]], requires_grad=True)
        target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss = criterion(pred, target)
        if torch.isnan(loss).item():
            _fail(name, "Loss is NaN")
            return

        val = float(loss.item())
        if not (0.0 < val < 10.0):
            _fail(name, f"Unexpected loss value: {val}")
            return

        loss.backward()
        if pred.grad is None:
            _fail(name, "No gradients on prediction tensor")
            return

        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_gan_head_shape():
    """Verify GANHead output shape is (B,1) for batch of any size."""
    name = "test_gan_head_shape"
    try:
        mod = importlib.import_module("train_gan")
        head = mod.GANHead()
        head.eval()

        with torch.no_grad():
            for b in [1, 8, 64]:
                x = torch.randn(b, 1024)
                out = head(x)
                if out.shape != (b, 1):
                    _fail(name, f"Bad output shape for B={b}: {tuple(out.shape)}")
                    return
                if float(out.min()) < 0.0 or float(out.max()) > 1.0:
                    _fail(name, f"Output out of [0,1] range for B={b}")
                    return

        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_dataset_folders():
    """Verify dataset/real/ and dataset/gan_fake/ exist and contain JPEG files."""
    name = "test_dataset_folders"
    try:
        real_dir = Path("dataset/real")
        gan_dir = Path("dataset/gan_fake")

        if not real_dir.exists() or not real_dir.is_dir():
            _fail(name, "Run build_dataset() first")
            return
        if not gan_dir.exists() or not gan_dir.is_dir():
            _fail(name, "Run build_dataset() first")
            return

        n_real = len(list(real_dir.glob("*.jpg")))
        n_gan = len(list(gan_dir.glob("*.jpg")))

        if n_real < 50:
            _fail(name, f"Only {n_real} real images. Need >= 50.")
            return
        if n_gan < 50:
            _fail(name, f"Only {n_gan} GAN images. Need >= 50.")
            return

        print(f"    Real: {n_real} | GAN: {n_gan}")
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_imagefolder_labels():
    """Verify ImageFolder assigns and label_flip logic handles class ordering."""
    name = "test_imagefolder_labels"
    try:
        mod = importlib.import_module("train_gan")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, _, meta = mod.make_loaders("dataset", device)
        _, labels = next(iter(train_loader))
        uniq = sorted(set(int(x) for x in labels.tolist()))

        if not set(uniq).issubset({0, 1}):
            _fail(name, f"Non-binary labels from loader: {uniq}")
            return

        print(f"    label_set={uniq} | label_flip={meta.get('label_flip')}")
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_amp_scaler():
    """Verify GradScaler initializes correctly on GPU, skips on CPU."""
    name = "test_amp_scaler"
    try:
        if AMP_API_NEW:
            scaler = TorchGradScaler("cuda", enabled=torch.cuda.is_available())
        else:
            scaler = TorchGradScaler(enabled=torch.cuda.is_available())
        if scaler is None:
            _fail(name, "GradScaler is None")
            return
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_single_forward_pass():
    """Run one full forward pass: image → encoder → head → loss."""
    name = "test_single_forward_pass"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mod = importlib.import_module("train_gan")
        encoder = mod.load_encoder(device)
        head = mod.GANHead().to(device)
        criterion = mod.FocalLoss()

        dummy_img = torch.randn(2, 3, 224, 224).to(device)
        dummy_lbl = torch.tensor([[1.0], [0.0]]).to(device)

        with torch.no_grad():
            feats = encoder(dummy_img)
            feats = feats.view(feats.size(0), -1)
            feats = F.normalize(feats, dim=-1)
            preds = head(feats)
            loss = criterion(preds, dummy_lbl)

        if preds.shape != (2, 1):
            _fail(name, f"Bad prediction shape: {tuple(preds.shape)}")
            return
        if torch.isnan(loss).item() or float(loss.item()) <= 0.0:
            _fail(name, f"Invalid loss value: {float(loss.item())}")
            return

        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_dataloader_speed():
    """Load one batch from train DataLoader and time it."""
    name = "test_dataloader_speed"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mod = importlib.import_module("train_gan")
        train_loader, _, meta = mod.make_loaders("dataset", device)

        t0 = time.time()
        imgs, labels = next(iter(train_loader))
        elapsed = time.time() - t0

        if tuple(imgs.shape[1:]) != (3, 224, 224):
            _fail(name, f"Bad image shape: {tuple(imgs.shape)}")
            return

        uniq = sorted(set(int(x) for x in labels.tolist()))
        if not set(uniq).issubset({0, 1}):
            _fail(name, f"Labels not binary: {uniq}")
            return

        label_preview = labels.tolist()[:4]
        print(f"    Batch: {tuple(imgs.shape)} | Labels: {label_preview} | Time: {elapsed:.2f}s")
        print(f"    Meta: batch_size={meta.get('batch_size')} | label_flip={meta.get('label_flip')}")
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_checkpoint_exists():
    """Verify checkpoint file exists and has correct keys."""
    name = "test_checkpoint_exists"
    try:
        if not POST_MODE:
            _pass(name)
            return

        path = "checkpoints/layer3_best.pth"
        if not os.path.exists(path):
            _fail(name, "Checkpoint not found. Did training complete?")
            return

        ckpt = torch.load(path, map_location="cpu")
        required_keys = {"epoch", "encoder_state", "head_state", "val_auc", "val_acc"}
        ckpt_keys = set(ckpt.keys())

        missing = required_keys - ckpt_keys
        if missing:
            _fail(name, f"Missing keys: {missing}")
            return

        extra = ckpt_keys - required_keys
        if extra:
            _fail(name, f"Unexpected extra keys: {extra}")
            return

        print(
            f"    AUC={float(ckpt['val_auc']):.4f} | "
            f"Acc={float(ckpt['val_acc']):.4f} | Epoch={int(ckpt['epoch'])}"
        )
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_centroid_exists():
    """Verify CLIP centroid file exists, is 1-D tensor of dim 1024."""
    name = "test_centroid_exists"
    try:
        if not POST_MODE:
            _pass(name)
            return

        path = "checkpoints/clip_real_centroid.pt"
        if not os.path.exists(path):
            _fail(name, "Centroid not found. Run calibrate().")
            return

        c = torch.load(path, map_location="cpu")
        if c.dim() != 1:
            _fail(name, f"Expected 1-D tensor, got shape {tuple(c.shape)}")
            return
        if c.shape[0] != 1024:
            _fail(name, f"Expected dim 1024, got {c.shape[0]}")
            return

        norm = c.norm().item()
        if abs(norm - 1.0) > 0.01:
            _fail(name, f"Centroid not unit-normalized, norm={norm:.4f}")
            return

        print(f"    Shape: {tuple(c.shape)} | Norm: {norm:.6f}")
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


def test_integration_with_gan_detector():
    """Load trained model into GANDetector and run inference on synthetic images."""
    name = "test_integration_with_gan_detector"
    try:
        if not POST_MODE:
            _pass(name)
            return

        try:
            gan_mod = importlib.import_module("verisight_layer3_gan")
        except Exception:
            gan_mod = importlib.import_module("layer3_gan.verisight_layer3_gan")
        GANDetector = gan_mod.GANDetector
        Layer3Config = gan_mod.Layer3Config

        mod = importlib.import_module("train_gan")

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = Layer3Config(device=device_str)
        detector = GANDetector(cfg)

        ckpt = torch.load("checkpoints/layer3_best.pth", map_location=device_str)
        ckpt_epoch = int(ckpt.get("epoch", 0))
        centroid = torch.load("checkpoints/clip_real_centroid.pt", map_location="cpu")

        detector.clip_detector._real_centroid = centroid
        detector.clip_detector._calibrated = True
        detector.clip_detector.model = mod.GANHead().to(device_str)
        detector.clip_detector.model.load_state_dict(ckpt["head_state"])
        detector.clip_detector.model.eval()

        rng = np.random.default_rng(0)
        ii, jj = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
        r = (128 + 70 * np.sin(ii / 28.0) + rng.normal(0, 8, (224, 224))).clip(0, 255)
        g = (128 + 55 * np.cos(jj / 22.0) + rng.normal(0, 8, (224, 224))).clip(0, 255)
        b = (100 + 45 * np.sin((ii + jj) / 36.0) + rng.normal(0, 8, (224, 224))).clip(0, 255)
        real_img = np.stack([r, g, b], axis=2).astype(np.uint8)

        gan_img = real_img.astype(np.float64).copy()
        for i in range(112):
            for j in range(112):
                sign = 1 if (i // 8 + j // 8) % 2 == 0 else -1
                gan_img[i, j, 0] += sign * 30
        gan_img[:, :, 1] = gan_img[:, :, 0] * 0.975
        gan_img[:, :, 2] = gan_img[:, :, 0] * 0.960
        gan_img = gan_img.clip(0, 255).astype(np.uint8)

        import cv2

        with tempfile.TemporaryDirectory() as td:
            real_path = os.path.join(td, "tmp_real.jpg")
            gan_path = os.path.join(td, "tmp_gan.jpg")
            cv2.imwrite(real_path, cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(gan_path, cv2.cvtColor(gan_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            r_real = detector.analyze(real_path)
            r_gan = detector.analyze(gan_path)

        print(f"    Genuine fraud_prob: {r_real.fraud_probability:.4f}")
        print(f"    GAN     fraud_prob: {r_gan.fraud_probability:.4f}")
        print(f"    Sub-scores (genuine): {vars(r_real.sub_scores)}")
        print(f"    Sub-scores (GAN):     {vars(r_gan.sub_scores)}")

        if ckpt_epoch <= 2:
            if not (r_gan.fraud_probability > r_real.fraud_probability + 0.03):
                _fail(
                    name,
                    (
                        "Smoke checkpoint failed relative separation: "
                        f"gan={r_gan.fraud_probability:.4f}, real={r_real.fraud_probability:.4f}"
                    ),
                )
                return
        else:
            if not (r_real.fraud_probability < 0.98):
                _fail(name, f"Genuine score too high: {r_real.fraud_probability:.4f}")
                return
            if not (r_gan.fraud_probability > 0.35):
                _fail(name, f"GAN score too low: {r_gan.fraud_probability:.4f}")
                return

        _pass(name)
    except Exception as e:
        _fail(name, str(e))


if __name__ == "__main__":
    import time, numpy as np
    from pathlib import Path
    import torch, torch.nn.functional as F

    mode = "POST-TRAINING" if POST_MODE else "PRE-TRAINING"
    print(f"\n{'='*55}")
    print(f"  VeriSight Layer 3 — {mode} Test Suite")
    print(f"{'='*55}\n")

    test_torch_cuda()
    test_clip_import()
    test_clip_normalization()
    test_focal_loss()
    test_gan_head_shape()
    test_dataset_folders()
    test_imagefolder_labels()
    test_amp_scaler()
    test_single_forward_pass()
    test_dataloader_speed()

    if POST_MODE:
        print("\n  — Post-training verification —")
        test_checkpoint_exists()
        test_centroid_exists()
        test_integration_with_gan_detector()

    print(f"\n{'='*55}")
    if FAILED:
        print("  RESULT: ✗ SOME TESTS FAILED — fix before proceeding")
        sys.exit(1)
    else:
        n = 13 if POST_MODE else 10
        print(f"  RESULT: ✓ ALL {n} TESTS PASSED")
        if not POST_MODE:
            print("  → Ready to run: python train_gan.py")
        else:
            print("  → Layer 3 is trained and operational")
    print(f"{'='*55}\n")
