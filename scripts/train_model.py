#!/usr/bin/env python3
"""SignBridge Model Training Pipeline.

Trains a fine-tuned MobileNetV2 classifier on sign language phrase images.
Features:
    - torchvision.datasets.ImageFolder loader
    - Stratified train/val/test splits (70% / 15% / 15%)
    - Domain-tailored data augmentations (no horizontal flipping)
    - Pretrained MobileNetV2 ImageNet backbone with frozen early layers
    - Adam optimizer with weight decay
    - ReduceLROnPlateau learning rate scheduler
    - Early stopping on validation accuracy
    - Class labels JSON export and model checkpointing
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")

# Fallback constants
DEFAULT_DATASET_PATH = "/Users/aryansoni/Desktop/Sign Language Judge/images for phrases"
DEFAULT_OUTPUT_MODEL = "/Users/aryansoni/Desktop/Sign Language Judge/SignBridge/models/sign_model.pth"
DEFAULT_LABELS_PATH = "/Users/aryansoni/Desktop/Sign Language Judge/SignBridge/models/class_labels.json"
DEFAULT_SPLITS_PATH = "/Users/aryansoni/Desktop/Sign Language Judge/SignBridge/models/dataset_splits.json"


# -----------------------------------------------------------------------------
# Configuration Resolution
# -----------------------------------------------------------------------------

def resolve_default_dataset_path() -> str:
    """Attempts to discover default dataset path from project settings or env."""
    cleaned_dir = os.path.join(PROJECT_ROOT, "dataset", "cleaned_images")
    if os.path.isdir(cleaned_dir):
        return cleaned_dir

    try:
        from app.core.config import settings
        for attr in ("DATASET_PATH", "dataset_path", "DATASET_DIR", "dataset_dir"):
            val = getattr(settings, attr, None)
            if val and isinstance(val, (str, os.PathLike)):
                return str(val)
    except Exception:
        pass

    try:
        from app.config import settings
        for attr in ("DATASET_PATH", "dataset_path", "DATASET_DIR", "dataset_dir"):
            val = getattr(settings, attr, None)
            if val and isinstance(val, (str, os.PathLike)):
                return str(val)
    except Exception:
        pass

    env_val = os.environ.get("SIGNBRIDGE_DATASET_PATH")
    if env_val:
        return env_val

    return DEFAULT_DATASET_PATH


def resolve_default_model_path() -> str:
    """Attempts to discover default model output path from project settings or env."""
    try:
        from app.core.config import settings
        for attr in ("MODEL_PATH", "model_path", "SIGN_MODEL_PATH"):
            val = getattr(settings, attr, None)
            if val and isinstance(val, (str, os.PathLike)):
                return str(val)
    except Exception:
        pass

    try:
        from app.config import settings
        for attr in ("MODEL_PATH", "model_path", "SIGN_MODEL_PATH"):
            val = getattr(settings, attr, None)
            if val and isinstance(val, (str, os.PathLike)):
                return str(val)
    except Exception:
        pass

    env_val = os.environ.get("SIGNBRIDGE_MODEL_PATH")
    if env_val:
        return env_val

    return DEFAULT_OUTPUT_MODEL


# -----------------------------------------------------------------------------
# Hardware Acceleration
# -----------------------------------------------------------------------------

def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolves optimal PyTorch computation device (CUDA, MPS, or CPU)."""
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seeds random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Dataset and Transforms
# -----------------------------------------------------------------------------

def get_train_transforms() -> transforms.Compose:
    """Builds training image transformation and augmentation pipeline.

    Crucially excludes horizontal flips because hand laterality and direction
    alter the semantic meaning of sign language gestures.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_test_transforms() -> transforms.Compose:
    """Builds deterministic validation and test image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TransformedDataset(Dataset):
    """Wraps an ImageFolder dataset with specific indices and transforms.

    Allows train and val/test splits to draw from the same underlying directory
    structure while applying distinct data augmentation pipelines.
    """

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices: Sequence[int],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        path, target = self.base_dataset.samples[real_idx]
        image = self.base_dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_stratified_splits(
    dataset: datasets.ImageFolder,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Partitions dataset samples into stratified train, val, and test splits.

    Guarantees balanced representation across all classes in each subset.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    targets = [target for _, target in dataset.samples]
    indices = np.arange(len(targets))

    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=temp_ratio,
        stratify=targets,
        random_state=random_state,
    )

    # Second split: val vs test
    val_ratio_of_temp = val_ratio / temp_ratio
    temp_targets = [targets[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1.0 - val_ratio_of_temp),
        stratify=temp_targets,
        random_state=random_state,
    )

    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

def build_model(
    num_classes: int,
    pretrained: bool = True,
    fine_tune_start_layer: int = 14,
) -> nn.Module:
    """Builds MobileNetV2 with ImageNet pretrained weights and fine-tuning setup.

    Freezes early feature blocks (0 up to `fine_tune_start_layer`) to preserve
    generic edge and texture filters, while unfreezing higher-level semantic
    feature blocks and replacing the classifier with a custom Linear projection.

    Args:
        num_classes: Number of target sign gesture classes.
        pretrained: Whether to load ImageNet pretrained backbone weights.
        fine_tune_start_layer: Layer index in `model.features` where fine-tuning starts.

    Returns:
        Configured PyTorch neural network module.
    """
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
        except (AttributeError, TypeError):
            model = models.mobilenet_v2(pretrained=True)
    else:
        try:
            model = models.mobilenet_v2(weights=None)
        except (AttributeError, TypeError):
            model = models.mobilenet_v2(pretrained=False)

    # Freeze all feature extractor blocks initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the last few blocks of the feature extractor for fine-tuning
    if fine_tune_start_layer is not None and fine_tune_start_layer < len(model.features):
        for block in model.features[fine_tune_start_layer:]:
            for param in block.parameters():
                param.requires_grad = True

    # Replace the ImageNet 1000-class classifier with custom projection
    in_features = model.last_channel  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# -----------------------------------------------------------------------------
# Training and Validation Routines
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single forward-backward optimization epoch over training data."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model loss and top-1 accuracy on validation or test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    eval_loss = running_loss / total if total > 0 else 0.0
    eval_acc = correct / total if total > 0 else 0.0
    return eval_loss, eval_acc


# -----------------------------------------------------------------------------
# Main Training Pipeline
# -----------------------------------------------------------------------------

def train_pipeline(
    dataset_path: str,
    output_model: str,
    labels_path: str,
    splits_path: Optional[str] = None,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    fine_tune_start_layer: int = 14,
    scheduler_patience: int = 3,
    early_stopping_patience: int = 5,
    num_workers: int = 2,
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes the complete end-to-end training and checkpointing pipeline."""
    set_seed(seed)
    device = get_device(device_name)
    logger.info("Using device: %s", device)

    # 1. Dataset Verification and Loading
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset path does not exist: '{dataset_path}'. "
            f"Please verify the directory location."
        )

    logger.info("Scanning dataset directory: %s", dataset_path)
    base_dataset = datasets.ImageFolder(dataset_path)
    num_samples = len(base_dataset)
    num_classes = len(base_dataset.classes)

    if num_samples == 0:
        raise RuntimeError(f"No images discovered in dataset path '{dataset_path}'")

    logger.info("Found %d images across %d classes.", num_samples, num_classes)

    # 2. Data Splitting (check for pre-computed splits to prevent temporal leakage)
    train_idx, val_idx, test_idx = None, None, None
    if splits_path and os.path.exists(splits_path):
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                saved_splits = json.load(f)
            t_idx = saved_splits.get("train_indices")
            v_idx = saved_splits.get("val_indices")
            te_idx = saved_splits.get("test_indices")
            if t_idx and v_idx and te_idx and len(t_idx) + len(v_idx) + len(te_idx) == num_samples:
                train_idx, val_idx, test_idx = t_idx, v_idx, te_idx
                logger.info("Loaded pre-computed split indices from: %s", splits_path)
        except Exception as err:
            logger.warning("Could not read splits from %s: %s", splits_path, err)

    if train_idx is None:
        logger.info("Performing stratified splits (70%% train, 15%% val, 15%% test)...")
        train_idx, val_idx, test_idx = get_stratified_splits(
            base_dataset,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=seed,
        )

    logger.info(
        "Splits active: %d train, %d val, %d test samples",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    # 3. Create Datasets with Dedicated Augmentations
    train_transforms = get_train_transforms()
    val_test_transforms = get_val_test_transforms()

    train_dataset = TransformedDataset(base_dataset, train_idx, transform=train_transforms)
    val_dataset = TransformedDataset(base_dataset, val_idx, transform=val_test_transforms)
    test_dataset = TransformedDataset(base_dataset, test_idx, transform=val_test_transforms)

    # Use 0 workers on non-Linux or small batches to prevent multiprocessing issues
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 4. Save Class Labels & Split Information
    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)), exist_ok=True)

    # idx_to_class mapping
    idx_to_class = {i: cls_name for i, cls_name in enumerate(base_dataset.classes)}
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    logger.info("Saved class labels (idx_to_class) to: %s", labels_path)

    if splits_path:
        os.makedirs(os.path.dirname(os.path.abspath(splits_path)), exist_ok=True)
        splits_record = {}
        if os.path.exists(splits_path):
            try:
                with open(splits_path, "r", encoding="utf-8") as f:
                    splits_record = json.load(f)
            except Exception:
                splits_record = {}
        splits_record.update({
            "random_state": seed,
            "num_classes": num_classes,
            "classes": base_dataset.classes,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        })
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits_record, f, indent=2)
        logger.info("Saved dataset splits to: %s", splits_path)

    # 5. Model Architecture & Optimization Setup
    logger.info(
        "Initializing MobileNetV2 (pretrained ImageNet weights, fine-tune layer %s+)...",
        fine_tune_start_layer,
    )
    model = build_model(
        num_classes=num_classes,
        pretrained=True,
        fine_tune_start_layer=fine_tune_start_layer,
    )
    model.to(device)

    # Only optimize parameters with requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters: %d / %d total",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in model.parameters()),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=scheduler_patience,
    )

    # 6. Training Loop with Early Stopping
    logger.info("Starting training loop (%d epochs, batch size %d, initial lr %f)...", epochs, batch_size, lr)
    print("\n" + "=" * 80)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^12} | {'Val Acc':^11} | {'LR':^10}")
    print("=" * 80)

    best_val_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    training_history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        print(
            f"{epoch + 1:^8d} | "
            f"{train_loss:^12.4f} | "
            f"{train_acc * 100:^10.2f}% | "
            f"{val_loss:^12.4f} | "
            f"{val_acc * 100:^10.2f}% | "
            f"{current_lr:^10.6f}"
        )

        # Validation Improvement & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())

            # Save checkpoint immediately
            torch.save(best_model_state, output_model)
            print(f"  [+] Validation accuracy improved to {val_acc * 100:.2f}%. Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"  [-] No improvement for {epochs_no_improve} epoch(s). Best: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n[!] Early stopping triggered: No val accuracy improvement for {early_stopping_patience} consecutive epochs.")
                break

        # Step LR Scheduler based on validation accuracy
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            print(f"  [*] Learning rate stepped down: {current_lr:.6f} -> {new_lr:.6f}")

    total_duration = time.time() - start_time

    # 7. Final Model Verification & Test Evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # Fallback if no improvement occurred (unlikely)
        best_model_state = model.state_dict()
        torch.save(best_model_state, output_model)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    model_size_mb = os.path.getsize(output_model) / (1024 * 1024) if os.path.exists(output_model) else 0.0

    # 8. Print Training Summary
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 80)
    print(f"  Classes Count:          {num_classes}")
    print(f"  Total Images:           {num_samples} (Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)})")
    print(f"  Epochs Completed:       {len(training_history)} / {epochs}")
    print(f"  Best Validation Epoch:  {best_epoch}")
    print(f"  Best Validation Acc:    {best_val_acc * 100:.2f}%")
    print(f"  Final Test Accuracy:    {test_acc * 100:.2f}% (Test Loss: {test_loss:.4f})")
    print(f"  Total Training Time:    {total_duration:.1f}s")
    print(f"  Saved Model Checkpoint: {output_model} ({model_size_mb:.2f} MB)")
    print(f"  Saved Class Labels:     {labels_path}")
    if splits_path:
        print(f"  Saved Splits Metadata:  {splits_path}")
    print("=" * 80 + "\n")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": training_history,
        "output_model": output_model,
        "labels_path": labels_path,
        "splits_path": splits_path,
    }


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="SignBridge Gesture Recognition Model Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=resolve_default_dataset_path(),
        help="Path to folder containing per-phrase image subdirectories",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for DataLoader",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial Adam learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization factor for Adam",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=resolve_default_model_path(),
        help="Output destination path for trained PyTorch model weights (.pth)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Output destination path for class labels JSON mapping",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default=None,
        help="Output destination path for dataset split indices JSON",
    )
    parser.add_argument(
        "--fine-tune-start-layer",
        type=int,
        default=14,
        help="Layer index in MobileNetV2 features from which fine-tuning is enabled",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit compute device ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (0 recommended on macOS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and weight initialization",
    )
    return parser


def main() -> None:
    """Main CLI execution wrapper with error handling."""
    parser = build_parser()
    args = parser.parse_args()

    # Determine paths
    output_model = os.path.abspath(args.output_model)
    model_dir = os.path.dirname(output_model)

    labels_path = (
        os.path.abspath(args.labels_path)
        if args.labels_path
        else os.path.join(model_dir, "class_labels.json")
    )
    splits_path = (
        os.path.abspath(args.splits_path)
        if args.splits_path
        else os.path.join(model_dir, "dataset_splits.json")
    )

    try:
        train_pipeline(
            dataset_path=args.dataset_path,
            output_model=output_model,
            labels_path=labels_path,
            splits_path=splits_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fine_tune_start_layer=args.fine_tune_start_layer,
            num_workers=args.num_workers,
            device_name=args.device,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Exiting cleanly.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Fatal error during model training: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
