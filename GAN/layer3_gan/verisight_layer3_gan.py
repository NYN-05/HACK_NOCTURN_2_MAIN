from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np

try:
    import open_clip
    import torch
    import torch.nn as nn
    from PIL import Image
except Exception:
    open_clip = None
    torch = None
    nn = None
    Image = None


@dataclass
class Layer3Config:
    device: str = "cpu"
    lbp_uniformity_threshold: float = 0.55


@dataclass
class SubScores:
    spectrum: float
    clip: float
    channel: float
    boundary: float
    texture: float
    resynth: float


@dataclass
class Layer3Result:
    fraud_probability: float
    sub_scores: SubScores
    flags: List[str]
    heatmap: np.ndarray


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class _ClipDetector:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._calibrated = False
        self._clip_model = None
        self._preprocess = None
        self.model = None
        self._real_centroid = None

    def calibrate(self, real_image_tensors: list) -> None:
        if torch is None:
            self._calibrated = False
            return

        tensors = [t for t in real_image_tensors if isinstance(t, torch.Tensor)]
        if not tensors:
            self._calibrated = False
            return

        stacked = torch.cat([t.reshape(1, -1) for t in tensors], dim=0)
        centroid = stacked.mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-8)
        self._real_centroid = centroid
        self._calibrated = True

    @property
    def has_trained_head(self) -> bool:
        return self.model is not None and torch is not None

    def _ensure_backbone(self) -> bool:
        if self._clip_model is not None and self._preprocess is not None:
            return True
        if open_clip is None or torch is None:
            return False

        use_device = self.device
        if use_device == "cuda" and not torch.cuda.is_available():
            use_device = "cpu"
        self.device = use_device

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai"
        )
        self._clip_model = clip_model.to(self.device)
        self._clip_model.eval()
        self._preprocess = preprocess
        return True

    def _heuristic_fallback(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(edges.mean() / 255.0)
        score = 1.0 - edge_density
        if not self._calibrated:
            score = min(1.0, score + 0.08)
        return float(np.clip(score, 0.0, 1.0))

    def score(self, img: np.ndarray) -> float:
        if not self.has_trained_head:
            return self._heuristic_fallback(img)

        if not self._ensure_backbone() or Image is None:
            return self._heuristic_fallback(img)

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = self._preprocess(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self._clip_model.encode_image(x)
                head_out = self.model(emb)
                head_score = float(
                    torch.clamp(head_out.reshape(-1)[0], min=0.0, max=1.0).item()
                )

                if self._calibrated and isinstance(self._real_centroid, torch.Tensor):
                    norm_emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    centroid = self._real_centroid.to(self.device).reshape(1, -1)
                    centroid = centroid / centroid.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    cos_sim = float((norm_emb * centroid).sum(dim=-1).item())
                    distance = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
                    dist_score = float(np.clip((distance - 0.05) / 0.35, 0.0, 1.0))
                    return float(np.clip((0.8 * head_score) + (0.2 * dist_score), 0.0, 1.0))

                return head_score
        except Exception:
            return self._heuristic_fallback(img)


class GANDetector:
    def __init__(self, cfg: Layer3Config | None = None):
        self.cfg = cfg or Layer3Config()
        self.clip_detector = _ClipDetector(device=self.cfg.device)

    def _spectrum_score(self, gray: np.ndarray) -> float:
        size = max(gray.shape)
        padded = np.zeros((size, size), dtype=np.float32)
        padded[: gray.shape[0], : gray.shape[1]] = gray
        fft = np.abs(np.fft.fft2(padded))
        fft = np.fft.fftshift(fft)
        center = fft[size // 4 : (3 * size) // 4, size // 4 : (3 * size) // 4].mean()
        outer = np.concatenate(
            [
                fft[: size // 8, :].ravel(),
                fft[-size // 8 :, :].ravel(),
                fft[:, : size // 8].ravel(),
                fft[:, -size // 8 :].ravel(),
            ]
        ).mean()
        ratio = outer / (center + 1e-8)
        return float(np.clip(ratio / 2.0, 0.0, 1.0))

    def _clip_fingerprint_score(self, img: np.ndarray) -> float:
        return self.clip_detector.score(img)

    def _channel_score(self, img: np.ndarray) -> float:
        b, g, r = cv2.split(img)
        stats = np.array([b.std(), g.std(), r.std()], dtype=np.float32)
        spread = float(np.std(stats))
        return float(np.clip(spread / 25.0, 0.0, 1.0))

    def _boundary_score(self, gray: np.ndarray) -> tuple[float, np.ndarray]:
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        mag = np.abs(lap)
        heatmap = (255 * mag / (mag.max() + 1e-8)).astype(np.uint8)
        # Use raw Laplacian magnitude percentile to avoid suppression by a single strong edge.
        score = float(np.clip(np.percentile(mag, 95) / 32.0, 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0)), heatmap

    def _texture_score(self, gray: np.ndarray) -> float:
        # Combine blur residual and Laplacian spread to capture synthetic patch texture drift.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        residual = float(diff.mean() / 255.0)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_spread = float(np.clip(lap.std() / 24.0, 0.0, 1.0))

        anomaly = (0.35 * np.clip(residual * 10.0, 0.0, 1.0)) + (0.65 * lap_spread)
        return float(np.clip(anomaly, 0.0, 1.0))

    def _resynth_score(self, img: np.ndarray) -> float:
        encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])[1]
        rec = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, rec)
        return float(np.clip(diff.mean() / 32.0, 0.0, 1.0))

    def analyze(self, image_path: str) -> Layer3Result:
        img = cv2.imread(image_path)
        if img is None:
            zeros = np.zeros((224, 224), dtype=np.uint8)
            scores = SubScores(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
            return Layer3Result(0.5, scores, ["UNREADABLE_IMAGE"], zeros)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        spectrum = self._spectrum_score(gray)
        clip = self._clip_fingerprint_score(img)
        channel = self._channel_score(img)
        boundary, heatmap = self._boundary_score(gray)
        texture = self._texture_score(gray)
        resynth = self._resynth_score(img)

        heuristic = (
            (0.30 * spectrum)
            + (0.20 * channel)
            + (0.20 * boundary)
            + (0.20 * texture)
            + (0.10 * resynth)
        )

        if self.clip_detector.has_trained_head:
            # Prefer trained CLIP head when available, but retain heuristic guardrails.
            raw = (0.68 * clip) + (0.32 * heuristic)
            # Trained head outputs are conservative in this stack, so shift center lower.
            fraud_probability = _sigmoid(9.0 * (raw - 0.30))
        else:
            raw = (0.30 * clip) + (0.70 * heuristic)
            fraud_probability = _sigmoid(9.0 * (raw - 0.46))

        flags: List[str] = []
        if spectrum > 0.6:
            flags.append("SPECTRAL_REPLICATION_PATTERN")
        if clip > 0.65:
            flags.append("GAN_FINGERPRINT_SIGNAL")
        if boundary > 0.6:
            flags.append("BOUNDARY_BLEND_INCONSISTENCY")
        if texture > 0.55:
            flags.append("TEXTURE_UNIFORMITY_ANOMALY")
        if not flags:
            flags.append("NO_STRONG_GAN_SIGNAL")

        scores = SubScores(spectrum, clip, channel, boundary, texture, resynth)
        return Layer3Result(float(np.clip(fraud_probability, 0.0, 1.0)), scores, flags, heatmap)


ImagePreprocessor = SimpleNamespace  # Compatibility alias for calibration snippets.
CLIPGANDetector = _ClipDetector
