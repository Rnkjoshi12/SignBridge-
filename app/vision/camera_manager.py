"""Camera manager module for SignBridge.

Provides thread-safe video capture and frame acquisition using OpenCV,
with robust error handling for camera availability, permissions, and disconnections.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Thread-safe camera manager wrapping OpenCV VideoCapture.

    Handles camera acquisition, resolution management, frame retrieval,
    and error states such as missing cameras, permission issues, or
    unexpected disconnections.

    Attributes:
        camera_index: Device index of the camera (default: 0).
        target_fps: Requested frame rate for capture (default: 30).
    """

    def __init__(self, camera_index: int = 0, target_fps: int = 30) -> None:
        self.camera_index = camera_index
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running: bool = False
        self._lock: threading.RLock = threading.RLock()

        self._target_width: Optional[int] = None
        self._target_height: Optional[int] = None

    def start(self) -> bool:
        """Open the camera and prepare for frame acquisition.

        Returns:
            True if the camera opened successfully, False otherwise.
        """
        with self._lock:
            if self.is_running():
                return True

            try:
                logger.info("Opening camera index %d...", self.camera_index)
                cap = cv2.VideoCapture(self.camera_index)

                if not cap.isOpened():
                    logger.error(
                        "Failed to open camera index %d. Check camera connection "
                        "and permissions.", self.camera_index
                    )
                    cap.release()
                    self._cap = None
                    self._is_running = False
                    return False

                # Configure target frame rate
                if self.target_fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

                # Apply configured resolution if set before start()
                if self._target_width is not None and self._target_height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._target_width))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._target_height))

                # Verify capture with an initial test frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None or test_frame.size == 0:
                    logger.error(
                        "Camera %d opened but failed to capture initial frame. "
                        "Permission may be denied or device is busy.",
                        self.camera_index
                    )
                    cap.release()
                    self._cap = None
                    self._is_running = False
                    return False

                self._cap = cap
                self._is_running = True
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(
                    "Camera %d started (%dx%d @ %.1f fps).",
                    self.camera_index, actual_w, actual_h, actual_fps
                )
                return True

            except Exception as exc:
                logger.exception("Error starting camera %d: %s", self.camera_index, exc)
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                self._is_running = False
                return False

    def stop(self) -> None:
        """Release the camera and clean up resources."""
        with self._lock:
            self._is_running = False
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    logger.warning("Error releasing camera %d: %s", self.camera_index, exc)
                finally:
                    self._cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.

        Returns:
            (success, frame) tuple. If reading fails, returns (False, None).
        """
        with self._lock:
            if not self._is_running or self._cap is None:
                return False, None

            if not self._cap.isOpened():
                self._is_running = False
                return False, None

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None or frame.size == 0:
                    return False, None
                return True, frame
            except Exception as exc:
                logger.error("Error reading frame from camera %d: %s", self.camera_index, exc)
                return False, None

    def is_running(self) -> bool:
        """Check if the camera is actively capturing frames."""
        with self._lock:
            return bool(
                self._is_running
                and self._cap is not None
                and self._cap.isOpened()
            )

    def set_resolution(self, width: int, height: int) -> bool:
        """Set the camera resolution.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            True if resolution was accepted by the driver.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Resolution must be positive, got {width}x{height}")

        with self._lock:
            self._target_width = width
            self._target_height = height

            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return actual_w == width and actual_h == height

            return True

    def get_resolution(self) -> Tuple[int, int]:
        """Get the current camera resolution.

        Returns:
            (width, height) in pixels. Returns (0, 0) if unknown.
        """
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return (
                    int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            if self._target_width and self._target_height:
                return self._target_width, self._target_height
            return 0, 0

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
