from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM generator for manipulation explainability."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self.model.backbone.features[-1]

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad(set_to_none=True)
        target_score = output[:, class_idx]
        target_score.backward(retain_graph=True)

        pooled_grads = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted = self.activations * pooled_grads
        cam = torch.sum(weighted, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_heatmap(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay Grad-CAM heatmap over an RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap, alpha, 0)
    return overlay
