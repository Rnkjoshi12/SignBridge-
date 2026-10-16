#!/usr/bin/env python3
"""SignBridge Model Evaluation Script.

Evaluates a trained MobileNetV2 sign language classification model on the
held-out test split.
Features:
    - Loads trained weights and class label mapping (idx_to_class)
    - Evaluates on identical stratified test split
    - Reports overall top-1 accuracy
    - Generates full classification report (precision, recall, F1 per class)
    - Generates and saves high-resolution confusion matrix plot (matplotlib)
    - Identifies and ranks top confused gesture pairs
    - Prints engineering summary of model limitations and operational considerations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Enable importing from the parent SignBridge project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

# Import shared pipeline utilities from train_model
from scripts.train_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_OUTPUT_MODEL,
    DEFAULT_SPLITS_PATH,
    TransformedDataset,
    build_model,
    get_device,
    get_stratified_splits,
    get_val_test_transforms,
    resolve_default_dataset_path,
    resolve_default_model_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_model")


# -----------------------------------------------------------------------------
# Checkpoint and Label Loading
# -----------------------------------------------------------------------------

def load_class_labels(labels_path: str) -> Tuple[Dict[int, str], List[str]]:
    """Loads class labels mapping from JSON file.

    Supports both dictionary mapping {"0": "phrase", ...} and list ["phrase", ...].

    Returns:
        Tuple of (idx_to_class mapping, ordered list of class names).
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels file not found at: '{labels_path}'")

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        idx_to_class = {int(k): str(v) for k, v in data.items()}
    elif isinstance(data, list):
        idx_to_class = {i: str(v) for i, v in enumerate(data)}
    else:
        raise ValueError(f"Unrecognized class labels format in '{labels_path}'")

    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    return idx_to_class, class_names


def load_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Instantiates MobileNetV2 architecture and loads trained weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: '{model_path}'")

    logger.info("Instantiating MobileNetV2 architecture (classes: %d)...", num_classes)
    model = build_model(num_classes=num_classes, pretrained=False)

    logger.info("Loading model weights from: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device)

    # Support both raw state_dict and nested checkpoint dicts
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise TypeError(f"Unexpected checkpoint format loaded from '{model_path}'")

    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Test Evaluation Core
# -----------------------------------------------------------------------------

def run_test_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward inference across test set and aggregates predictions.

    Returns:
        Tuple of (y_true, y_pred, y_probabilities).
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_targets, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=float),
    )


# -----------------------------------------------------------------------------
# Visualization & Reporting
# -----------------------------------------------------------------------------

def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "SignBridge Gesture Recognition - Confusion Matrix (Test Set)",
) -> None:
    """Generates and saves a high-resolution confusion matrix heatmap."""
    num_classes = len(class_names)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(22, 20), dpi=300)

    # Render confusion matrix heatmap
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Configure axes
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8, ha="center")
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel("Predicted Sign Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20, fontweight="bold")

    # Annotate counts inside cells if non-zero
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > (max_val / 2.0) else "black"
                ax.text(
                    j,
                    i,
                    str(int(val)),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix visualization to: %s", output_path)


def get_top_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
) -> List[Tuple[str, str, int]]:
    """Identifies and ranks the most frequent off-diagonal confusion errors."""
    num_classes = len(class_names)
    confusions: List[Tuple[str, str, int]] = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], int(cm[i, j])))

    # Sort descending by error count
    confusions.sort(key=lambda item: item[2], reverse=True)
    return confusions[:top_k]


def print_limitations_summary() -> None:
    """Prints a structured engineering summary of model assumptions and limitations."""
    summary = """
================================================================================
MODEL LIMITATIONS & OPERATIONAL CONSIDERATIONS
================================================================================

1. Static 2D Representation vs. Dynamic Spatio-temporal Signs:
   - Natural sign language involves dynamic movements, trajectory paths, speed,
     and non-manual markers (e.g., facial expressions, lip patterns, head tilts).
   - This MobileNetV2 architecture classifies isolated static 2D frames (224x224).
     Signs that share similar stationary poses but differ primarily in movement
     direction or velocity cannot be fully resolved without temporal modeling.

2. Dataset Volume and Signer Generalization:
   - The current dataset contains 40 images per category across 44 categories
     (1,760 total frames).
   - Real-world deployment will encounter diverse signers with variations in hand
     anatomy, skin pigmentation, signing speed, clothing contrast, and ambient
     lighting that were not fully represented in the training distribution.

3. Morphological Similarity & Self-Occlusion:
   - Gestures requiring nuanced finger positioning (e.g. thumb resting on index vs.
     tucked inward, or subtle knuckle bending) have narrow visual decision boundaries.
   - 2D camera angles suffer from self-occlusion when the palm or forearm blocks
     salient finger configurations from the camera line of sight.

4. Streaming Pipeline Sensitivity:
   - For real-time webcam inference, isolated single-frame predictions can flicker.
   - Upstream hand detection / bounding box stability and downstream temporal
     stabilization (such as SignBridge's PredictionStabilizer) are mandatory to
     suppress transition noise between active signs.

5. Edge Hardware Inference Latency:
   - MobileNetV2 is lightweight (~3.5M parameters), making it suitable for edge CPUs
     and Apple Silicon (MPS). However, overall system latency includes camera grab,
     preprocessing, neural inference, and TTS audio synthesis.
================================================================================
"""
    print(summary)


# -----------------------------------------------------------------------------
# Evaluation Pipeline
# -----------------------------------------------------------------------------

def evaluate_pipeline(
    model_path: str,
    labels_path: str,
    dataset_path: str,
    splits_path: Optional[str] = None,
    output_plot: Optional[str] = None,
    batch_size: int = 32,
    top_k: int = 10,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes complete evaluation on the test split and produces reports."""
    device = get_device(device_name)
    logger.info("Using evaluation device: %s", device)

    # 1. Load Class Labels
    logger.info("Loading class labels from: %s", labels_path)
    idx_to_class, class_names = load_class_labels(labels_path)
    num_classes = len(class_names)
    logger.info("Loaded %d class categories.", num_classes)

    # 2. Load Trained Model
    model = load_model(model_path, num_classes=num_classes, device=device)

    # 3. Load Dataset & Test Split
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found at: '{dataset_path}'")

    logger.info("Loading dataset from: %s", dataset_path)
    base_dataset = datasets.ImageFolder(dataset_path)

    # Retrieve test split indices
    test_idx: List[int] = []
    if splits_path and os.path.exists(splits_path):
        logger.info("Loading pre-computed split indices from: %s", splits_path)
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        test_idx = splits_data.get("test_indices", [])
        logger.info("Loaded %d test indices from file.", len(test_idx))

    if not test_idx:
        logger.info("No split file found; reproducing stratified split (15%% test, seed=42)...")
        _, _, test_idx = get_stratified_splits(
            base_dataset,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        logger.info("Generated %d stratified test indices.", len(test_idx))

    val_test_transforms = get_val_test_transforms()
    test_dataset = TransformedDataset(base_dataset, test_idx, transform=val_test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Run Test Inference
    logger.info("Running inference on test split (%d samples)...", len(test_dataset))
    y_true, y_pred, y_probs = run_test_inference(model, test_loader, device)

    # 5. Overall Accuracy
    accuracy = float(accuracy_score(y_true, y_pred))
    correct_count = int(np.sum(y_true == y_pred))
    total_count = len(y_true)

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - OVERALL TEST ACCURACY: {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    print("=" * 80)

    # 6. Per-class Classification Report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nPER-CLASS CLASSIFICATION REPORT:")
    print("-" * 80)
    print(report)

    # 7. Confusion Matrix & Heatmap Plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    default_plot_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "confusion_matrix.png")
    plot_file = output_plot or default_plot_path
    plot_and_save_confusion_matrix(cm, class_names, plot_file)
    print(f"\n[+] Confusion matrix plot saved to: {plot_file}")

    # 8. Identify Top Confused Class Pairs
    top_confused = get_top_confused_pairs(cm, class_names, top_k=top_k)
    print("\n" + "=" * 80)
    print(f"TOP {top_k} CONFUSED CLASS PAIRS:")
    print("-" * 80)
    if not top_confused:
        print("  No confusion pairs detected! 100% test accuracy on held-out split.")
    else:
        for rank, (true_cls, pred_cls, err_count) in enumerate(top_confused, start=1):
            s_plural = "s" if err_count > 1 else ""
            print(f"  {rank:2d}. True: {true_cls:<20} --> Predicted: {pred_cls:<20} ({err_count} error{s_plural})")
    print("=" * 80)

    # 9. Summary of Limitations
    print_limitations_summary()

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "classification_report": report,
        "confusion_matrix": cm,
        "top_confused_pairs": top_confused,
        "plot_path": plot_file,
    }


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for model evaluation."""
    parser = argparse.ArgumentParser(
        description="SignBridge Model Evaluation and Performance Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=resolve_default_model_path(),
        help="Path to trained PyTorch model checkpoint (.pth)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Path to class labels JSON mapping file",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=resolve_default_dataset_path(),
        help="Path to root folder containing sign phrase images",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default=None,
        help="Path to dataset_splits.json containing test split indices",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Output destination path for confusion matrix plot (.png)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size for inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top confused class pairs to display",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cpu', 'cuda', 'mps')",
    )
    return parser


def main() -> None:
    """CLI execution wrapper with error handling."""
    parser = build_parser()
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    model_dir = os.path.dirname(model_path)

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
    output_plot = (
        os.path.abspath(args.output_plot)
        if args.output_plot
        else os.path.join(model_dir, "confusion_matrix.png")
    )

    try:
        evaluate_pipeline(
            model_path=model_path,
            labels_path=labels_path,
            dataset_path=args.dataset_path,
            splits_path=splits_path,
            output_plot=output_plot,
            batch_size=args.batch_size,
            top_k=args.top_k,
            device_name=args.device,
        )
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
