import os
import sys
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from verisight_layer3_gan import GANDetector, Layer3Config, CLIPGANDetector
except ImportError:
    from layer3_gan.verisight_layer3_gan import GANDetector, Layer3Config

    class CLIPGANDetector:  # Compatibility fallback type for local project layout
        pass


def _build_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def _normalize_head_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict

    if all(str(key).startswith("net.") for key in state_dict):
        return {str(key)[4:]: value for key, value in state_dict.items()}

    return state_dict


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


def load_trained_layer3(
    checkpoint_path: str = "checkpoints/layer3_best.pth",
    centroid_path: str = "checkpoints/clip_real_centroid.pt",
    device: str = "cuda",
) -> GANDetector:
    device = _resolve_device(device)

    cfg = Layer3Config(device=device)
    detector = GANDetector(cfg)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Layer 3 could not load trained weights. Returning base detector.")
        return detector

    if not os.path.exists(centroid_path):
        print(f"ERROR: Centroid not found: {centroid_path}")
        print("Layer 3 could not load centroid calibration. Returning base detector.")
        return detector

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Ensure clip detector exposes a trainable head object for patched inference.
        if not hasattr(detector.clip_detector, "model") or detector.clip_detector.model is None:
            detector.clip_detector.model = _build_head().to(device)
        else:
            detector.clip_detector.model = detector.clip_detector.model.to(device)

        head_state = _normalize_head_state_dict(ckpt["head_state"])
        detector.clip_detector.model.load_state_dict(head_state, strict=False)

        centroid = torch.load(centroid_path, map_location="cpu")
        detector.clip_detector._real_centroid = centroid
        detector.clip_detector._calibrated = True
        detector.clip_detector.model.eval()

        print("Layer 3 loaded with trained weights. CLIP centroid: calibrated")
        print(f"Checkpoint Val AUC: {float(ckpt.get('val_auc', 0.0)):.4f}")

        return detector
    except Exception as exc:
        print(f"ERROR: Failed to load trained Layer 3 artifacts: {exc}")
        print("Returning base detector without trained patch.")
        return detector


def run_inference_test(detector: GANDetector, image_path: str) -> Dict[str, Any]:
    result = detector.analyze(image_path)

    fraud = float(result.fraud_probability)
    classification = "GENUINE" if fraud < 0.5 else "GAN DETECTED"

    def signal(score: float) -> str:
        return "GAN SIGNAL" if score > 0.5 else "clean"

    print("=== VeriSight Layer 3 - GAN Analysis Report ===")
    print(f"Image: {image_path}")
    print(f"Fraud Probability: {fraud:.4f}")
    print(f"Classification: {classification}")
    print("Sub-scores:")
    print(f"  Spectrum  : {result.sub_scores.spectrum:.4f}  [{signal(result.sub_scores.spectrum)}]")
    print(f"  CLIP      : {result.sub_scores.clip:.4f}  [{signal(result.sub_scores.clip)}]")
    print(f"  Channel   : {result.sub_scores.channel:.4f}  [{signal(result.sub_scores.channel)}]")
    print(f"  Boundary  : {result.sub_scores.boundary:.4f}  [{signal(result.sub_scores.boundary)}]")
    print(f"  Texture   : {result.sub_scores.texture:.4f}  [{signal(result.sub_scores.texture)}]")
    print(f"  Re-synth  : {result.sub_scores.resynth:.4f}  [{signal(result.sub_scores.resynth)}]")
    print("Flags:")
    if result.flags:
        for flag in result.flags:
            print(f"  - {flag}")
    else:
        print("  None")

    return {
        "fraud_probability": fraud,
        "classification": classification,
        "sub_scores": {
            "spectrum": float(result.sub_scores.spectrum),
            "clip": float(result.sub_scores.clip),
            "channel": float(result.sub_scores.channel),
            "boundary": float(result.sub_scores.boundary),
            "texture": float(result.sub_scores.texture),
            "resynth": float(result.sub_scores.resynth),
        },
        "flags": list(result.flags),
    }


def _make_test_images() -> tuple[str, str]:
    h, w = 256, 256

    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    smooth = np.zeros((h, w, 3), dtype=np.float32)
    smooth[..., 0] = 0.45 + 0.25 * xv
    smooth[..., 1] = 0.50 + 0.20 * yv
    smooth[..., 2] = 0.55 + 0.15 * (1.0 - xv)
    smooth = np.clip(smooth * 255.0, 0, 255).astype(np.uint8)

    gan_like = smooth.copy()
    tile = 8
    checker = ((np.indices((80, 80)).sum(axis=0) // tile) % 2).astype(np.float32)
    checker = (checker * 2.0 - 1.0) * 35.0

    y1, x1 = 90, 90
    patch = gan_like[y1 : y1 + 80, x1 : x1 + 80].astype(np.float32)
    patch[..., 0] = np.clip(patch[..., 0] + checker, 0, 255)
    patch[..., 1] = np.clip(patch[..., 0] * 0.98, 0, 255)
    patch[..., 2] = np.clip(patch[..., 0] * 0.97, 0, 255)
    patch = cv2.GaussianBlur(patch.astype(np.uint8), (3, 3), sigmaX=0.5)
    gan_like[y1 : y1 + 80, x1 : x1 + 80] = patch

    genuine_path = "test_genuine.jpg"
    gan_path = "test_gan.jpg"

    cv2.imwrite(genuine_path, cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite(gan_path, cv2.cvtColor(gan_like, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return genuine_path, gan_path


if __name__ == "__main__":
    detector = load_trained_layer3()

    genuine_path, gan_path = _make_test_images()

    genuine_result = run_inference_test(detector, genuine_path)
    gan_result = run_inference_test(detector, gan_path)

    print("Expected: test_genuine -> fraud < 0.5, test_gan -> fraud > 0.5")

    genuine_ok = genuine_result["fraud_probability"] < 0.5
    gan_ok = gan_result["fraud_probability"] > 0.5

    if genuine_ok and gan_ok:
        print("PASS")
    else:
        print("FAIL")
        if not genuine_ok:
            print(
                f"  Genuine classification mismatch: {genuine_result['fraud_probability']:.4f}"
            )
        if not gan_ok:
            print(f"  GAN classification mismatch: {gan_result['fraud_probability']:.4f}")

    # Keep symbol referenced to satisfy explicit import contract.
    _ = CLIPGANDetector
