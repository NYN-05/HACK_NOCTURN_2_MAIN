#!/usr/bin/env python
"""
Layer 2 Training Script for VeriSight
Trains a ViT-B/16 model with optimized hyperparameters for REAL vs AI_GENERATED classification

Usage:
    python train_layer2.py
    python train_layer2.py --epochs 30 --batch-size 8 --patience 4
    python train_layer2.py --export-onnx --prepare-dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="VeriSight Layer 2 Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (30 epochs)
  %(prog)s

  # Custom epochs and batch size
  %(prog)s --epochs 50 --batch-size 16

  # With ONNX export and dataset preparation
  %(prog)s --export-onnx --prepare-dataset

  # Reduced patience for faster early stopping
  %(prog)s --patience 2
        """
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of warmup epochs (head-only training, default: 2)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience (default: 4)"
    )
    parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable weighted sampling for class balance"
    )

    # Optional features
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export trained model to ONNX format after training"
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Prepare dataset before training"
    )

    # Additional training parameters for advanced users
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Head learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Backbone learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default: 0.1)"
    )

    args = parser.parse_args()

    # Get script directory and construct paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    training_script = script_dir / "training" / "train_vit.py"

    print("\n" + "=" * 72)
    print(" VeriSight Layer 2 - Training Script".ljust(72))
    print("=" * 72)
    print()

    # Display configuration
    print("Configuration:")
    print(f"  Epochs:                {args.epochs}")
    print(f"  Batch Size:            {args.batch_size}")
    print(f"  Warmup Epochs:         {args.warmup_epochs}")
    print(f"  Early Stopping Patience: {args.patience}")
    print(f"  Head LR:               {args.lr}")
    print(f"  Backbone LR:           {args.backbone_lr}")
    print(f"  Weight Decay:          {args.weight_decay}")
    print(f"  Label Smoothing:       {args.label_smoothing}")
    print(f"  Balanced Sampling:     {not args.no_balanced_sampling}")
    print(f"  Export to ONNX:        {args.export_onnx}")
    print(f"  Prepare Dataset:       {args.prepare_dataset}")
    print()

    # Prepare dataset if requested
    if args.prepare_dataset:
        print("Preparing dataset...")
        try:
            from layer2.training.dataset_loader_refactored import prepare_dataset
            prepare_dataset()
            print("✓ Dataset prepared.\n")
        except Exception as e:
            print(f"✗ Dataset preparation failed: {e}\n")
            return 1

    # Build training command
    cmd = [
        sys.executable,
        str(training_script),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--warmup-epochs", str(args.warmup_epochs),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--backbone-lr", str(args.backbone_lr),
        "--weight-decay", str(args.weight_decay),
        "--label-smoothing", str(args.label_smoothing),
    ]

    if args.no_balanced_sampling:
        cmd.append("--no-balanced-sampling")

    if args.export_onnx:
        cmd.append("--export-onnx")

    print("Starting training...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run training
    try:
        result = subprocess.run(cmd, check=False)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n✗ Error running training: {e}")
        return 1

    # Report results
    if exit_code == 0:
        print("\n" + "=" * 72)
        print(" ✓ Training completed successfully!".ljust(72))
        print("=" * 72)
        print()
        print("Output files:")
        print("  Model:              layer2/models/vit_layer2_detector.pth")
        print("  Metrics:            layer2/models/vit_layer2_training_metrics.json")
        if args.export_onnx:
            print("  ONNX Model:         layer2/models/vit_layer2_detector.onnx")
        print()
        print("Next steps:")
        print("  1. Review metrics in: layer2/models/vit_layer2_training_metrics.json")
        print("  2. Run inference:     python layer2/inference/onnx_inference.py <image_path>")
        print("  3. Start API server:  python layer2/api/main.py")
        print()
        return 0
    else:
        print("\n" + "=" * 72)
        print(f" ✗ Training failed with exit code {exit_code}".ljust(72))
        print("=" * 72)
        print()
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
