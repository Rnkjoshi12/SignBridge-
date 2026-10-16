"""Sign language predictor module for SignBridge.

Performs neural network inference using a custom-trained YOLOv5 classification
model to classify sign language gestures from camera frames.

Replaces the previous MobileNetV2-based predictor with YOLOv5 classify
inference, using the same interface so the rest of the app remains unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SignPredictor:
    """Classifies sign language gestures from video frames using YOLOv5-cls.

    Loads a custom-trained YOLOv5 classification checkpoint and class label
    mapping, and provides predictions with confidence scores.

    Falls back to direct torchvision-based loading if YOLOv5 hub loading
    is unavailable.

    Args:
        model_path: Path to the saved YOLOv5 classification weights (.pt).
        labels_path: Path to the JSON file containing label names.
        img_size: Input resolution for the network (default: 224).
        device: PyTorch device. Auto-detects if None.
        conf_threshold: Minimum confidence to consider a prediction valid.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        labels_path: Union[str, Path],
        img_size: int = 224,
        device: Optional[Union[str, torch.device]] = None,
        conf_threshold: float = 0.15,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self._is_loaded: bool = False
        self.model = None
        self.labels: List[str] = []
        self.num_classes: int = 0
        self._model_type: str = "unknown"  # "yolov5_cls" or "mobilenet"

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device.")

        # Preprocessing transform for YOLOv5 classification
        self._setup_transform()

        # Try loading labels and model
        try:
            self.labels = self._load_labels(self.labels_path)
            self.num_classes = len(self.labels)
            self._load_model(self.model_path)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.warning("Could not load model: %s", e)
            self._is_loaded = False

    def _setup_transform(self) -> None:
        """Setup image preprocessing transform."""
        try:
            from torchvision import transforms
            from app.vision.preprocessing import ConvertBGRToPIL

            resize_dim = 256 if self.img_size == 224 else int(self.img_size * 1.143)
            self.transform = transforms.Compose([
                ConvertBGRToPIL(),
                transforms.Resize(resize_dim),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        except ImportError:
            self.transform = None
            logger.warning("Could not setup transforms (missing torchvision)")

    def _load_labels(self, path: Path) -> List[str]:
        """Load class labels from a JSON file.

        Supports formats:
        - dict: {"0": "label0", "1": "label1", ...}
        - list: ["label0", "label1", ...]
        """
        if not path.is_file():
            raise FileNotFoundError(f"Labels file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Keys are string indices: {"0": "again", "1": "agree", ...}
            try:
                int_keys = {int(k): str(v) for k, v in data.items()}
                labels = [int_keys[i] for i in range(len(int_keys))]
            except (ValueError, KeyError):
                labels = list(data.values())
        elif isinstance(data, list):
            labels = [str(x) for x in data]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")

        if not labels:
            raise ValueError(f"No labels found in {path}")

        return labels

    def _load_model(self, path: Path) -> None:
        """Load model weights — supports YOLOv5 classify .pt and legacy MobileNetV2 .pth."""
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_ext = path.suffix.lower()
        model_name = path.name.lower()

        # Determine model type and load accordingly
        if 'yolov5' in model_name or model_ext == '.pt':
            self._try_load_yolov5(path)
        else:
            self._try_load_mobilenet(path)

        if not self._is_loaded:
            # Fallback: try the other loader
            if self._model_type != "yolov5_cls":
                try:
                    self._try_load_yolov5(path)
                except Exception:
                    pass
            if not self._is_loaded and self._model_type != "mobilenet":
                try:
                    self._try_load_mobilenet(path)
                except Exception:
                    pass

    def _try_load_yolov5(self, path: Path) -> None:
        """Load a YOLOv5 classification model checkpoint."""
        try:
            # Try loading via torch.hub (works with the cloned YOLOv5 repo)
            project_root = Path(__file__).parent.parent.parent
            yolov5_dir = project_root / 'yolov5'

            if yolov5_dir.is_dir():
                import sys
                yolov5_str = str(yolov5_dir)
                if yolov5_str not in sys.path:
                    sys.path.insert(0, yolov5_str)

                # Load the checkpoint directly
                ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

                if isinstance(ckpt, dict) and 'model' in ckpt:
                    model = ckpt['model']
                    if hasattr(model, 'float'):
                        model = model.float()
                    model.to(self.device)
                    model.eval()
                    self.model = model
                    self._model_type = "yolov5_cls"
                    self._is_loaded = True

                    # Try to extract class names from checkpoint
                    if 'names' in ckpt and isinstance(ckpt['names'], (dict, list)):
                        names = ckpt['names']
                        if isinstance(names, dict):
                            ckpt_labels = [names[i] for i in range(len(names))]
                        else:
                            ckpt_labels = list(names)
                        # Only override if our labels file doesn't match
                        if len(ckpt_labels) == self.num_classes:
                            logger.info("Using class names from checkpoint")

                    logger.info(
                        "YOLOv5 classify model loaded on %s (%d classes).",
                        self.device, self.num_classes
                    )
                    return
            
            # Alternative: try torch.hub.load
            try:
                model = torch.hub.load(
                    str(yolov5_dir) if yolov5_dir.is_dir() else 'ultralytics/yolov5',
                    'custom',
                    path=str(path),
                    source='local' if yolov5_dir.is_dir() else 'github',
                    force_reload=False,
                )
                model.to(self.device)
                model.eval()
                self.model = model
                self._model_type = "yolov5_cls"
                self._is_loaded = True
                logger.info("YOLOv5 model loaded via hub on %s.", self.device)
                return
            except Exception as hub_err:
                logger.debug("Hub load failed: %s", hub_err)

        except Exception as e:
            logger.debug("YOLOv5 load attempt failed: %s", e)
            self._is_loaded = False

    def _try_load_mobilenet(self, path: Path) -> None:
        """Load legacy MobileNetV2 model (backward compatibility)."""
        try:
            from torchvision import models

            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)

            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get("state_dict", checkpoint)
                )
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix from DataParallel checkpoints
            cleaned = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                cleaned[key] = v

            model.load_state_dict(cleaned)
            model.to(self.device)
            model.eval()

            self.model = model
            self._model_type = "mobilenet"
            self._is_loaded = True
            logger.info("MobileNetV2 model loaded on %s (%d classes).", self.device, self.num_classes)

        except Exception as e:
            logger.debug("MobileNetV2 load attempt failed: %s", e)
            self._is_loaded = False

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """Predict the sign from a BGR OpenCV frame.

        Args:
            frame: BGR image numpy array from OpenCV.

        Returns:
            (predicted_label, confidence) where confidence is in [0, 1].
        """
        if not self.is_loaded() or self.model is None:
            raise RuntimeError("Model is not loaded.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame.")

        if self._model_type == "yolov5_cls":
            return self._predict_yolov5(frame)
        else:
            return self._predict_standard(frame)

    def _predict_yolov5(self, frame: np.ndarray) -> Tuple[str, float]:
        """Run YOLOv5 classification inference."""
        try:
            # Check if model supports direct image inference (hub-loaded)
            if hasattr(self.model, '__call__') and not isinstance(self.model, nn.Module):
                # Hub-loaded model can take images directly
                results = self.model(frame)
                # Parse results
                if hasattr(results, 'probs'):
                    probs = results.probs
                    conf = float(probs.max())
                    idx = int(probs.argmax())
                    label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
                    return label, conf
        except Exception:
            pass

        # Standard forward pass with transform
        if self.transform is None:
            raise RuntimeError("Transform not available")

        tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            # Handle different output shapes
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        idx = pred_idx.item()
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return label, float(confidence.item())

    def _predict_standard(self, frame: np.ndarray) -> Tuple[str, float]:
        """Run standard torchvision model inference (MobileNetV2, etc.)."""
        if self.transform is None:
            raise RuntimeError("Transform not available")

        tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        label = self.labels[pred_idx.item()]
        return label, float(confidence.item())

    def predict_top_k(self, frame: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k predictions with confidence scores.

        Args:
            frame: BGR image numpy array from OpenCV.
            k: Number of top predictions to return.

        Returns:
            List of (label, confidence) tuples sorted by confidence.
        """
        if not self.is_loaded() or self.model is None:
            raise RuntimeError("Model is not loaded.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame.")

        k = min(k, self.num_classes)

        if self.transform is None:
            raise RuntimeError("Transform not available")

        tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)

        return [
            (self.labels[idx] if idx < len(self.labels) else f"class_{idx}", float(prob))
            for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist())
        ]

    def get_labels(self) -> List[str]:
        """Return a copy of all class labels."""
        return list(self.labels)

    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return self.num_classes

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return bool(self._is_loaded and self.model is not None)

    def get_model_type(self) -> str:
        """Return the type of loaded model ('yolov5_cls', 'mobilenet', or 'unknown')."""
        return self._model_type
