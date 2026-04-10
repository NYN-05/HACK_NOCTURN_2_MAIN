"""
Layer 2 Inference Module - Image to Forgery Detection Score
Loads the trained ViT model and generates confidence scores for REAL vs AI_GENERATED images.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


class Layer2Detector:
    """Loads and uses the trained ViT Layer 2 model for forgery detection."""
    
    def __init__(
        self, 
        model_path: str = r"C:\Users\JHASHANK\Downloads\VERISIGHT_V1\layer2\models\vit_layer2_detector.pth"
    ):
        """
        Initialize the Layer 2 detector with the trained model.
        
        Args:
            model_path: Path to the saved model weights (.pth file)
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.label_mapping = {0: "REAL", 1: "AI_GENERATED"}
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pretrained ViT model and processor."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        # Initialize the pretrained ViT model from Hugging Face
        model_name = "google/vit-base-patch16-224"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "REAL", 1: "AI_GENERATED"},
            label2id={"REAL": 0, "AI_GENERATED": 1},
            ignore_mismatched_sizes=True,
        )
        
        # Load the trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully")
    
    def generate_score(self, image_path: str) -> Dict[str, float]:
        """
        Generate forgery detection scores for an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with scores:
            {
                'REAL': float (0.0-1.0),
                'AI_GENERATED': float (0.0-1.0),
                'predicted_label': str ('REAL' or 'AI_GENERATED'),
                'confidence': float (0.0-1.0)
            }
        """
        image_path = Path(image_path)
        # Try relative path from repo root if it doesn't exist
        if not image_path.exists():
            repo_root = Path(__file__).resolve().parent.parent
            image_path = repo_root / image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Extract scores
        scores = probabilities[0].cpu().numpy()
        real_score = float(scores[0])
        ai_generated_score = float(scores[1])
        predicted_idx = int(torch.argmax(logits, dim=-1).item())
        predicted_label = self.label_mapping[predicted_idx]
        confidence = float(probabilities[0, predicted_idx].item())
        
        return {
            'REAL': real_score,
            'AI_GENERATED': ai_generated_score,
            'predicted_label': predicted_label,
            'confidence': confidence
        }
    
    def batch_score(self, image_paths: list) -> list:
        """
        Generate scores for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of score dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                score = self.generate_score(image_path)
                results.append({
                    'image_path': str(image_path),
                    'scores': score,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'scores': None,
                    'error': str(e)
                })
        return results


def main():
    """Example usage of the Layer 2 detector."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Layer 2 Forgery Detection - Generate scores for images"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to analyze"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"C:\Users\JHASHANK\Downloads\VERISIGHT_V1\layer2\models\vit_layer2_detector.pth",
        help="Path to the trained model weights"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = Layer2Detector(model_path=args.model_path)
    
    # Generate score
    result = detector.generate_score(args.image_path)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"Image: {args.image_path}")
    print("=" * 60)
    print(f"Predicted Label:  {result['predicted_label']}")
    print(f"Confidence:       {result['confidence']:.4f}")
    print(f"\nDetailed Scores:")
    print(f"  REAL:           {result['REAL']:.4f}")
    print(f"  AI_GENERATED:   {result['AI_GENERATED']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
