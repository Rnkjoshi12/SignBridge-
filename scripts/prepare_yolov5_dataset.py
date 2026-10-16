#!/usr/bin/env python3
r"""Prepare CollectedImages dataset for YOLOv5 classification training.

Converts the flat class-folder structure into train/val/test splits
compatible with YOLOv5 classify mode.

Usage:
    python scripts/prepare_yolov5_dataset.py \
        --source "e:\IVP\SignBridgeDataset\CollectedImages" \
        --output "dataset/yolov5_cls"
"""

import argparse
import os
import random
import shutil
import sys
import yaml
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def prepare_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True,
) -> dict:
    """Prepare YOLOv5 classification dataset from class-folder images.
    
    Args:
        source_dir: Path to CollectedImages folder with class subfolders.
        output_dir: Path to output YOLOv5 classification dataset.
        train_ratio: Fraction of images for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.
        copy_files: If True, copy files; otherwise create symlinks.
    
    Returns:
        Dictionary with dataset statistics.
    """
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Discover classes (subdirectories, skip hidden/dot files)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    classes = sorted([
        d.name for d in source_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {source_dir}")
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Collect images per class
    class_images = defaultdict(list)
    skipped_files = []
    
    for cls_name in classes:
        cls_dir = source_path / cls_name
        for f in sorted(cls_dir.iterdir()):
            if f.name.startswith('.'):
                skipped_files.append(str(f))
                continue
            if f.is_file() and f.suffix.lower() in valid_extensions:
                class_images[cls_name].append(f)
            elif f.is_file():
                skipped_files.append(str(f))
    
    # Validate
    total_images = sum(len(v) for v in class_images.values())
    if total_images == 0:
        raise RuntimeError("No valid images found in source directory")
    
    print(f"Total valid images: {total_images}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} non-image/hidden files")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls_name in classes:
            (output_path / split / cls_name).mkdir(parents=True, exist_ok=True)
    
    # Split per class (stratified)
    stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = {}
    
    for cls_name in classes:
        images = list(class_images[cls_name])
        random.shuffle(images)
        
        n = len(images)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, n - n_train - n_val)
        
        # Adjust if rounding leaves remainder
        if n_train + n_val + n_test > n:
            n_train = n - n_val - n_test
        if n_train + n_val + n_test < n:
            n_train += (n - n_train - n_val - n_test)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_path in split_imgs:
                dst = output_path / split_name / cls_name / img_path.name
                if not dst.exists():
                    if copy_files:
                        shutil.copy2(str(img_path), str(dst))
                    else:
                        try:
                            os.symlink(str(img_path), str(dst))
                        except OSError:
                            shutil.copy2(str(img_path), str(dst))
                stats[split_name] += 1
        
        class_stats[cls_name] = {
            'total': n,
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs),
        }
    
    # Generate dataset_config.yaml
    config = {
        'path': str(output_path.resolve()),
        'train': str((output_path / 'train').resolve()),
        'val': str((output_path / 'val').resolve()),
        'test': str((output_path / 'test').resolve()),
        'nc': len(classes),
        'names': classes,
    }
    
    config_path = output_path.parent / 'dataset_config.yaml'
    with open(str(config_path), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Also save class_names.json for the app
    import json
    class_labels = {str(i): name for i, name in enumerate(classes)}
    labels_path = Path(PROJECT_ROOT) / 'models' / 'class_labels.json'
    with open(str(labels_path), 'w', encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATASET PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Source:       {source_dir}")
    print(f"  Output:       {output_path.resolve()}")
    print(f"  Config:       {config_path.resolve()}")
    print(f"  Classes:      {len(classes)}")
    print(f"  Total Images: {total_images}")
    print(f"  Train:        {stats['train']} ({stats['train']/total_images*100:.1f}%)")
    print(f"  Validation:   {stats['val']} ({stats['val']/total_images*100:.1f}%)")
    print(f"  Test:         {stats['test']} ({stats['test']/total_images*100:.1f}%)")
    print(f"\nPer-class breakdown:")
    for cls_name, cs in sorted(class_stats.items()):
        print(f"  {cls_name:25s} total={cs['total']:3d}  train={cs['train']:3d}  val={cs['val']:3d}  test={cs['test']:3d}")
    print(f"{'='*60}")
    
    return {
        'classes': classes,
        'num_classes': len(classes),
        'total_images': total_images,
        'splits': stats,
        'class_stats': class_stats,
        'config_path': str(config_path),
        'output_dir': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CollectedImages for YOLOv5 classification"
    )
    parser.add_argument(
        '--source',
        type=str,
        default=os.environ.get(
            'SIGNBRIDGE_DATASET_SOURCE',
            r'e:\IVP\SignBridgeDataset\CollectedImages'
        ),
        help='Path to CollectedImages folder',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'dataset', 'yolov5_cls'),
        help='Output path for prepared dataset',
    )
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--copy', action='store_true', default=True,
        help='Copy files instead of symlinking'
    )
    
    args = parser.parse_args()
    
    try:
        prepare_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_files=args.copy,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
