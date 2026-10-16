"""Tests for YOLOv5 model integration into SignBridge.

Validates that:
- The trained YOLOv5s classification model loads successfully
- SignPredictor correctly identifies model type as 'yolov5_cls'
- Class count matches the 31 signs in CollectedImages
- Predictions on real test images yield valid labels and high confidence
- Top-k predictions are ordered and formatted properly
- Error handling on invalid/empty inputs works as expected
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.vision.predictor import SignPredictor
from app.core.settings import MODEL_PATH, CLASS_LABELS_PATH, DATASET_PATH


class TestYOLOv5Integration(unittest.TestCase):
    """Test suite for YOLOv5 model integration and prediction workflow."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = MODEL_PATH
        cls.labels_path = CLASS_LABELS_PATH
        cls.predictor = SignPredictor(
            model_path=cls.model_path,
            labels_path=cls.labels_path,
        )

    def test_model_loaded_successfully(self):
        """Model should load and report ready."""
        self.assertTrue(
            self.predictor.is_loaded(),
            f"YOLOv5 model failed to load from {self.model_path}",
        )

    def test_model_type_is_yolov5(self):
        """Predictor should recognize model as YOLOv5 classification."""
        self.assertEqual(
            self.predictor.get_model_type(),
            "yolov5_cls",
            "Model type should be 'yolov5_cls'",
        )

    def test_class_count_and_labels(self):
        """Model should have exactly 31 classes matching the CollectedImages dataset."""
        labels = self.predictor.get_labels()
        self.assertEqual(len(labels), 31)
        self.assertEqual(self.predictor.get_num_classes(), 31)
        self.assertIn("again", labels)
        self.assertIn("book", labels)
        self.assertIn("stop", labels)
        self.assertIn("write", labels)

    def test_prediction_dummy_frame(self):
        """Predict on a synthetic frame should return valid label and confidence in [0, 1]."""
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        label, conf = self.predictor.predict(dummy)
        self.assertIsInstance(label, str)
        self.assertIn(label, self.predictor.get_labels())
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_predict_top_k(self):
        """predict_top_k should return sorted (label, prob) pairs."""
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        top5 = self.predictor.predict_top_k(dummy, k=5)
        self.assertEqual(len(top5), 5)
        # Check ordering
        probs = [p for _, p in top5]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for lbl, p in top5:
            self.assertIn(lbl, self.predictor.get_labels())
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_invalid_frame_handling(self):
        """Invalid or empty frames should raise ValueError."""
        with self.assertRaises(ValueError):
            self.predictor.predict(None)

        with self.assertRaises(ValueError):
            self.predictor.predict(np.array([]))

        with self.assertRaises(ValueError):
            self.predictor.predict("not an image")

    def test_real_image_prediction(self):
        """Test inference on a real test set image from the YOLOv5 dataset."""
        test_dir = Path(PROJECT_ROOT) / "dataset" / "yolov5_cls" / "test"
        if not test_dir.is_dir():
            self.skipTest("yolov5_cls test directory not present")

        import cv2
        for cls_name in ["again", "book", "write"]:
            cls_folder = test_dir / cls_name
            if cls_folder.is_dir():
                images = list(cls_folder.glob("*.jpg"))
                if images:
                    img_path = str(images[0])
                    frame = cv2.imread(img_path)
                    self.assertIsNotNone(frame, f"Failed to read {img_path}")
                    pred_label, conf = self.predictor.predict(frame)
                    self.assertEqual(
                        pred_label,
                        cls_name,
                        f"Expected {cls_name}, got {pred_label} (conf: {conf:.2f})",
                    )
                    self.assertGreater(
                        conf,
                        0.50,
                        f"Confidence for {cls_name} should be > 50%, got {conf:.2f}",
                    )


if __name__ == "__main__":
    unittest.main()
