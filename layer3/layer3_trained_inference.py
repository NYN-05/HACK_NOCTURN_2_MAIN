"""Layer 3 Trained GAN Detector - Inference module for trained CLIP-based detector."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import open_clip
except ImportError:
    open_clip = None

LOGGER = logging.getLogger(__name__)

# CLIP normalization from training
CLIP_NORMALIZE = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


class TrainedLayer3Detector:
    """Trained CLIP-based GAN detector for inference."""
    
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        real_centroid: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
        decision_threshold: float = 0.5,
        img_size: int = 224,
    ):
        """
        Initialize the trained detector.
        
        Args:
            encoder: CLIP visual encoder
            head: Classification head
            device: Device for inference (cpu/cuda)
            decision_threshold: Threshold for binary decision (0-1)
            img_size: Input image size
        """
        self.encoder = encoder.to(device)
        self.head = head.to(device)
        self.real_centroid = real_centroid.to(device) if real_centroid is not None else None
        self.device = device
        self.decision_threshold = decision_threshold
        self.img_size = img_size
        
        self.encoder.eval()
        self.head.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            CLIP_NORMALIZE,
        ])

    def _centroid_fraud_probability(self, embeddings: torch.Tensor) -> float | None:
        if self.real_centroid is None:
            return None

        centroid = self.real_centroid
        if centroid.ndim == 1:
            centroid = centroid.unsqueeze(0)

        similarity = F.cosine_similarity(embeddings, centroid, dim=-1).item()
        similarity = float(np.clip(similarity, -1.0, 1.0))
        # Map centroid similarity to a fraud probability proxy.
        return float(np.clip(1.0 - ((similarity + 1.0) / 2.0), 0.0, 1.0))
    
    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode image using CLIP encoder."""
        if isinstance(image, str):
            with Image.open(image) as img_file:
                image = img_file.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image format: {type(image)}")
        
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(x)
            if features.ndim > 2:
                features = features.view(features.size(0), -1)
            embeddings = F.normalize(features, dim=-1)
        
        return embeddings
    
    def analyze(self, image: Any) -> Dict[str, Any]:
        """
        Analyze image for GAN artifacts using trained detector.
        
        Args:
            image: Image path (str) or numpy array
        
        Returns:
            Dictionary with fraud_probability and other metadata
        """
        try:
            embeddings = self._encode_image(image)
            
            with torch.no_grad():
                logits = self.head(embeddings)
                fraud_prob = float(np.clip(logits.item(), 0.0, 1.0))

            centroid_prob = self._centroid_fraud_probability(embeddings)
            if centroid_prob is not None:
                fraud_prob = float((fraud_prob * 0.45) + (centroid_prob * 0.55))
            
            # Ensure probability is in [0, 1]
            fraud_prob = float(np.clip(fraud_prob, 0.0, 1.0))
            
            # Generate flags based on confidence
            flags = []
            if fraud_prob > 0.95:
                flags.append("high_gan_confidence")
            elif fraud_prob > 0.75:
                flags.append("moderate_gan_indicators")
            elif fraud_prob < 0.05:
                flags.append("high_authenticity_confidence")
            elif fraud_prob < 0.25:
                flags.append("likely_authentic")
            else:
                flags.append("uncertain_classification")
            
            # Calculate uncertainty (how far from decision boundary)
            distance_to_threshold = abs(fraud_prob - self.decision_threshold)
            uncertainty = 1.0 - (distance_to_threshold * 2.0)  # Closer to threshold = higher uncertainty
            uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
            
            return {
                "fraud_probability": fraud_prob,
                "decision": "FAKE" if fraud_prob >= self.decision_threshold else "AUTHENTIC",
                "uncertainty": uncertainty,
                "confidence": 1.0 - uncertainty,
                "flags": flags,
                "sub_scores": {
                    "embedding_norm": float(torch.norm(embeddings).item()),
                    "decision_threshold": self.decision_threshold,
                    "margin": float(abs(fraud_prob - self.decision_threshold)),
                    "centroid_fraud_probability": centroid_prob,
                }
            }
        
        except Exception as e:
            LOGGER.error(f"Layer 3 inference failed: {e}", exc_info=True)
            return {
                "fraud_probability": 0.5,
                "decision": "UNKNOWN",
                "uncertainty": 1.0,
                "confidence": 0.0,
                "flags": [f"inference_error: {str(e)}"],
                "sub_scores": {}
            }


def build_models(device: torch.device):
    """Build CLIP encoder and classification head (matching training architecture)."""
    if open_clip is None:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")
    
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    clip_model = clip_model.visual.to(device)
    clip_model.eval()
    
    # Freeze CLIP encoder
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # Build head (must match training architecture exactly)
    head = nn.Sequential(
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
    ).to(device)
    
    return clip_model, head


def _load_real_centroid(
    encoder: nn.Module,
    repo_root: Path,
    device: torch.device,
    img_size: int = 224,
    max_images: int = 48,
) -> torch.Tensor | None:
    real_dir = repo_root / "cleaned_data" / "images_complete" / "real"
    if not real_dir.exists():
        return None

    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(sorted(real_dir.glob(pattern)))
    image_paths = image_paths[:max_images]

    if not image_paths:
        return None

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        CLIP_NORMALIZE,
    ])

    embeddings: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                features = encoder(tensor)
                if features.ndim > 2:
                    features = features.view(features.size(0), -1)
                normalized = F.normalize(features, dim=-1)
                embeddings.append(normalized.detach().cpu())
            except Exception as exc:
                LOGGER.debug("Skipping centroid sample %s: %s", image_path.name, exc)

    if not embeddings:
        return None

    centroid = torch.cat(embeddings, dim=0).mean(dim=0)
    centroid = centroid / centroid.norm()
    return centroid


def load_trained_layer3(
    checkpoint_path: str,
    centroid_path: str = None,
    device: str = "cpu",
) -> TrainedLayer3Detector:
    """
    Load trained Layer 3 detector from checkpoint.
    
    Args:
        checkpoint_path: Path to layer3_best.pth checkpoint
        centroid_path: Path to real image centroid (optional)
        device: Device for inference ("cpu" or "cuda")
    
    Returns:
        Initialized TrainedLayer3Detector instance
    """
    device_obj = torch.device(device)
    
    # Build models
    LOGGER.info("Building CLIP model and classification head...")
    encoder, head = build_models(device_obj)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        LOGGER.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    def _load_state_dict(module: nn.Module, state_dict: Dict[str, Any], module_name: str) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"{module_name} state_dict must be a mapping")

        candidate_states = [state_dict]
        stripped_state = {
            k.replace("module.", "").replace("encoder.", "").replace("head.", "").replace("net.", ""): v
            for k, v in state_dict.items()
        }
        if stripped_state != state_dict:
            candidate_states.append(stripped_state)

        last_error: Exception | None = None
        for candidate in candidate_states:
            try:
                module.load_state_dict(candidate, strict=True)
                LOGGER.info("Loaded %s state dict with %d tensors", module_name, len(candidate))
                return
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            LOGGER.warning("Strict %s load failed; falling back to partial load: %s", module_name, last_error)
        module.load_state_dict(candidate_states[-1], strict=False)
    
    # Extract head weights from checkpoint
    if "head_state" in checkpoint:
        head_state = checkpoint["head_state"]
    elif "model_state" in checkpoint:
        # Filter out encoder weights, keep only head weights
        head_state = {k.replace("head.", ""): v for k, v in checkpoint["model_state"].items() if k.startswith("head.")}
    elif "state_dict" in checkpoint:
        head_state = {k.replace("head.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("head.")}
    else:
        head_state = checkpoint
    
    # Handle "net." prefix in state_dict keys (from training script)
    if "net.0.weight" in head_state:
        head_state = {k.replace("net.", ""): v for k, v in head_state.items()}
    
    if "encoder_state" in checkpoint:
        _load_state_dict(encoder, checkpoint["encoder_state"], "encoder")
    else:
        LOGGER.warning("Checkpoint does not contain encoder_state; using default CLIP encoder weights")

    _load_state_dict(head, head_state, "head")
    LOGGER.info("Checkpoint loaded successfully")

    centroid_tensor = None
    if centroid_path:
        centroid_file = Path(centroid_path)
        if centroid_file.exists():
            try:
                centroid_tensor = torch.load(centroid_file, map_location=device_obj)
                LOGGER.info("Loaded Layer 3 centroid from %s", centroid_file)
            except Exception as exc:
                LOGGER.warning("Failed to load centroid from %s: %s", centroid_file, exc)

    if centroid_tensor is None:
        repo_root = checkpoint_path.parents[2]
        centroid_tensor = _load_real_centroid(encoder, repo_root, device_obj)
        if centroid_tensor is not None:
            LOGGER.info("Calibrated Layer 3 centroid from cleaned_data/real images")
    
    # Extract decision threshold if available
    decision_threshold = checkpoint.get("best_threshold", 0.5)
    
    detector = TrainedLayer3Detector(
        encoder=encoder,
        head=head,
        real_centroid=centroid_tensor,
        device=device_obj,
        decision_threshold=float(decision_threshold),
        img_size=224,
    )
    
    LOGGER.info(f"Layer 3 detector initialized (threshold={decision_threshold:.3f})")
    return detector
