#!/usr/bin/env python3
"""YOLOv5 Classification Training for SignBridge.

Trains a YOLOv5s-cls model on the prepared sign language dataset
using transfer learning from pretrained ImageNet weights.

Based on the approach from Sign_language_Generation_Using_YOLO_v5.ipynb,
adapted for classification (since dataset has no bounding box annotations).

Usage:
    python scripts/train_yolov5.py
    python scripts/train_yolov5.py --epochs 50 --batch-size 32 --img-size 224
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5')


def setup_yolov5():
    """Clone YOLOv5 repository if not already present."""
    if os.path.isdir(YOLOV5_DIR) and os.path.isfile(os.path.join(YOLOV5_DIR, 'classify', 'train.py')):
        print(f"YOLOv5 already cloned at: {YOLOV5_DIR}")
        return True
    
    print("Cloning YOLOv5 repository...")
    try:
        subprocess.run(
            ['git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOV5_DIR],
            check=True,
            cwd=PROJECT_ROOT,
        )
        print(f"YOLOv5 cloned to: {YOLOV5_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone YOLOv5: {e}")
        return False
    
    # Install YOLOv5 requirements
    req_file = os.path.join(YOLOV5_DIR, 'requirements.txt')
    if os.path.isfile(req_file):
        print("Installing YOLOv5 dependencies...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', req_file],
            check=False,
        )
    
    return True


def train_classifier(
    dataset_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    img_size: int = 224,
    model: str = 'yolov5s-cls.pt',
    project: str = None,
    name: str = 'signbridge_cls',
    device: str = '',
    workers: int = 4,
    exist_ok: bool = True,
):
    """Train YOLOv5 classification model.
    
    Args:
        dataset_path: Path to dataset with train/val/test subdirs.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        img_size: Input image size.
        model: Pretrained model (yolov5s-cls.pt, yolov5m-cls.pt, etc.).
        project: Project output directory.
        name: Experiment name.
        device: Device ('' for auto, 'cpu', '0' for GPU 0).
        workers: Number of dataloader workers.
        exist_ok: Whether to overwrite existing experiment.
    """
    if project is None:
        project = os.path.join(PROJECT_ROOT, 'runs', 'classify')
    
    train_script = os.path.join(YOLOV5_DIR, 'classify', 'train.py')
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"YOLOv5 classify/train.py not found at {train_script}")
    
    cmd = [
        sys.executable, train_script,
        '--model', model,
        '--data', dataset_path,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--imgsz', str(img_size),
        '--project', project,
        '--name', name,
        '--workers', str(workers),
    ]
    
    if device:
        cmd.extend(['--device', device])
    if exist_ok:
        cmd.append('--exist-ok')
    
    print(f"\n{'='*60}")
    print("STARTING YOLOv5 CLASSIFICATION TRAINING")
    print(f"{'='*60}")
    print(f"  Dataset:    {dataset_path}")
    print(f"  Model:      {model}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Project:    {project}")
    print(f"  Name:       {name}")
    print(f"  Device:     {device or 'auto'}")
    print(f"{'='*60}\n")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=YOLOV5_DIR)
    
    if result.returncode != 0:
        print(f"\nWARNING: Training exited with code {result.returncode}")
        return None
    
    # Find best weights
    exp_dir = os.path.join(project, name)
    weights_dir = os.path.join(exp_dir, 'weights')
    best_pt = os.path.join(weights_dir, 'best.pt')
    last_pt = os.path.join(weights_dir, 'last.pt')
    
    if os.path.isfile(best_pt):
        print(f"\nBest weights saved at: {best_pt}")
        # Copy to models directory for the app
        dst = os.path.join(PROJECT_ROOT, 'models', 'yolov5_sign_cls.pt')
        import shutil
        shutil.copy2(best_pt, dst)
        print(f"Copied best weights to: {dst}")
        return best_pt
    elif os.path.isfile(last_pt):
        print(f"\nLast weights saved at: {last_pt}")
        dst = os.path.join(PROJECT_ROOT, 'models', 'yolov5_sign_cls.pt')
        import shutil
        shutil.copy2(last_pt, dst)
        print(f"Copied last weights to: {dst}")
        return last_pt
    else:
        print("\nWARNING: No trained weights found!")
        return None


def validate_model(
    weights_path: str,
    dataset_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    device: str = '',
):
    """Validate trained model on test set."""
    val_script = os.path.join(YOLOV5_DIR, 'classify', 'val.py')
    if not os.path.isfile(val_script):
        print("WARNING: classify/val.py not found, skipping validation")
        return
    
    cmd = [
        sys.executable, val_script,
        '--weights', weights_path,
        '--data', dataset_path,
        '--imgsz', str(img_size),
        '--batch-size', str(batch_size),
    ]
    if device:
        cmd.extend(['--device', device])
    
    print(f"\nRunning validation: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=YOLOV5_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 classification model for SignBridge"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'dataset', 'yolov5_cls'),
        help='Path to prepared YOLOv5 classification dataset',
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument(
        '--model', type=str, default='yolov5s-cls.pt',
        help='Pretrained model to start from',
    )
    parser.add_argument('--device', type=str, default='', help='Device (auto, cpu, 0)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', type=str, default='signbridge_cls')
    parser.add_argument('--skip-setup', action='store_true', help='Skip YOLOv5 clone')
    parser.add_argument('--validate', action='store_true', default=True, help='Run validation after training')
    
    args = parser.parse_args()
    
    # Step 1: Setup YOLOv5
    if not args.skip_setup:
        if not setup_yolov5():
            print("Failed to setup YOLOv5. Exiting.")
            sys.exit(1)
    
    # Step 2: Verify dataset exists
    if not os.path.isdir(args.dataset):
        print(f"ERROR: Dataset not found at {args.dataset}")
        print("Please run: python scripts/prepare_yolov5_dataset.py first")
        sys.exit(1)
    
    train_dir = os.path.join(args.dataset, 'train')
    if not os.path.isdir(train_dir):
        print(f"ERROR: train/ directory not found in {args.dataset}")
        sys.exit(1)
    
    # Step 3: Train
    best_weights = train_classifier(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model=args.model,
        device=args.device,
        workers=args.workers,
        name=args.name,
    )
    
    # Step 4: Validate
    if args.validate and best_weights:
        validate_model(
            weights_path=best_weights,
            dataset_path=args.dataset,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
