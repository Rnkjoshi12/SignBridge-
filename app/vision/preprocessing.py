"""Image Preprocessing and Augmentation Utilities for SignBridge.

Provides torchvision transform pipelines for:
- Training (with sign-safe augmentation: rotation ±10°, color jitter, resized crop, NO horizontal flip)
- Validation / Testing (resize, center crop, normalize)
- Live inference from OpenCV BGR frames
- Batch tensor preparation (adds batch dimension)

Uses ImageNet normalization statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Standard ImageNet normalization statistics
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Graceful optional imports for environments where dependencies are installed later
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    torch = None  # type: ignore
    transforms = None  # type: ignore


class ConvertBGRToPIL:
    """Transform callable that converts an OpenCV BGR numpy array to a PIL RGB Image."""

    def __call__(self, img: Any) -> Any:
        """Converts input BGR image to PIL RGB Image.

        Args:
            img: OpenCV BGR frame (numpy.ndarray) or PIL Image.

        Returns:
            PIL.Image.Image in RGB mode.
        """
        if Image is None:
            raise ImportError("Pillow is required for image transforms. Install with 'pip install pillow'.")

        if np is not None and isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                if cv2 is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # BGR to RGB reverse channel indexing fallback
                    rgb = img[:, :, ::-1]
                return Image.fromarray(rgb)
            return Image.fromarray(img)

        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img

        raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(img)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Alias for backward compatibility
BGRToPILImage = ConvertBGRToPIL


def _check_torchvision() -> None:
    """Verifies that torch and torchvision are available."""
    if torch is None or transforms is None:
        raise ImportError(
            "torch and torchvision are required for preprocessing transforms. "
            "Please install them via 'pip install torch torchvision'."
        )


def get_train_transform(
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Any:
    """Returns torchvision transforms for training with sign-safe augmentations.

    Augmentations applied:
        - RandomResizedCrop (scale 0.8-1.0) for scale variations
        - RandomRotation (±10°) for realistic hand orientation variations
        - ColorJitter (brightness, contrast, saturation) for lighting robustness
        - ToTensor: converts image to tensor in [0.0, 1.0]
        - Normalize: normalizes using ImageNet mean and std

    Note:
        RandomHorizontalFlip is intentionally EXCLUDED because horizontal
        flipping changes hand chirality (e.g., distinguishing left vs right hand)
        and alters the semantic meaning of sign language gestures.

    Args:
        img_size: Target square image dimension (default: 224).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    _check_torchvision()

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(0.8, 1.0),
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def get_val_transform(
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Any:
    """Returns transforms for validation and testing (no random augmentations).

    Standard ImageNet evaluation pipeline:
        - Resize slightly larger (256 for 224 target)
        - CenterCrop to exact target dimension
        - ToTensor: converts image to tensor in [0.0, 1.0]
        - Normalize: normalizes with ImageNet statistics

    Args:
        img_size: Target square image dimension (default: 224).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    _check_torchvision()

    resize_dim = 256 if img_size == 224 else int(img_size * 1.143)

    return transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def get_inference_transform(
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Any:
    """Returns inference transform for OpenCV BGR frames or PIL Images.

    Converts BGR numpy array to PIL RGB, resizes, center crops, converts to
    tensor, and applies ImageNet normalization.

    Args:
        img_size: Target square image dimension (default: 224).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    _check_torchvision()

    resize_dim = 256 if img_size == 224 else int(img_size * 1.143)

    return transforms.Compose(
        [
            ConvertBGRToPIL(),
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def preprocess_frame(
    frame: Any,
    transform: Optional[Any] = None,
    img_size: int = 224,
) -> Any:
    """Preprocesses an OpenCV BGR frame and returns a batch-ready tensor.

    Converts BGR frame to RGB/PIL, applies the transform pipeline,
    and adds a leading batch dimension: (1, 3, H, W).

    Args:
        frame: OpenCV BGR frame (numpy.ndarray) or PIL Image.
        transform: Optional transform pipeline. If None, defaults to
                   get_inference_transform(img_size).
        img_size: Target image size if transform is not explicitly passed.

    Returns:
        torch.Tensor of shape (1, 3, H, W) ready for model input.
    """
    _check_torchvision()

    if transform is None:
        transform = get_inference_transform(img_size=img_size)

    # If the frame is a numpy ndarray and the transform does not contain
    # ConvertBGRToPIL, convert it to PIL Image first
    input_data = frame
    if np is not None and isinstance(frame, np.ndarray):
        # Check if transform already has BGR conversion
        has_bgr_converter = False
        if hasattr(transform, "transforms"):
            for step in transform.transforms:
                if isinstance(step, (ConvertBGRToPIL, BGRToPILImage)):
                    has_bgr_converter = True
                    break
        if not has_bgr_converter:
            converter = ConvertBGRToPIL()
            input_data = converter(frame)

    tensor = transform(input_data)

    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    if hasattr(tensor, "dim") and tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor
