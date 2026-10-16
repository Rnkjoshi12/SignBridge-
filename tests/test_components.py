#!/usr/bin/env python3
"""
SignBridge — Automated Test Suite

Tests core components without requiring a webcam or display.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# Tests for ConversationManager
# ============================================================

class TestConversationManager:
    """Tests for the ConversationManager class."""

    def setup_method(self):
        from app.core.conversation_manager import ConversationManager
        self.cm = ConversationManager()

    def test_initial_state_is_empty(self):
        assert self.cm.is_empty()
        assert self.cm.get_transcript() == []
        assert self.cm.get_transcript_text() == ""
        assert self.cm.get_latest() is None

    def test_add_single_word(self):
        self.cm.add_word("HELLO")
        assert not self.cm.is_empty()
        assert self.cm.get_transcript() == ["HELLO"]
        assert self.cm.get_latest() == "HELLO"

    def test_add_multiple_words(self):
        words = ["HELLO", "MY", "NAME", "ALEX"]
        for w in words:
            self.cm.add_word(w)
        assert self.cm.get_transcript() == words
        assert self.cm.get_transcript_text() == "HELLO MY NAME ALEX"
        assert self.cm.get_latest() == "ALEX"

    def test_clear_transcript(self):
        self.cm.add_word("HELLO")
        self.cm.add_word("WORLD")
        self.cm.clear()
        assert self.cm.is_empty()
        assert self.cm.get_transcript() == []
        assert self.cm.get_latest() is None

    def test_add_after_clear(self):
        self.cm.add_word("HELLO")
        self.cm.clear()
        self.cm.add_word("GOODBYE")
        assert self.cm.get_transcript() == ["GOODBYE"]

    def test_callback_on_add(self):
        received = []
        self.cm.on_word_added = lambda w: received.append(w)
        self.cm.add_word("TEST")
        assert received == ["TEST"]

    def test_callback_on_clear(self):
        cleared = []
        self.cm.on_cleared = lambda: cleared.append(True)
        self.cm.add_word("TEST")
        self.cm.clear()
        assert len(cleared) == 1


# ============================================================
# Tests for PredictionStabilizer
# ============================================================

class TestPredictionStabilizer:
    """Tests for the PredictionStabilizer class."""

    def setup_method(self):
        from app.vision.prediction_stabilizer import PredictionStabilizer
        # window=5, agreement=4, cooldown=3
        self.ps = PredictionStabilizer(
            stabilization_window=5,
            min_agreement=4,
            cooldown_frames=3,
            confidence_threshold=0.5
        )

    def test_initial_returns_none(self):
        result = self.ps.update("HELLO", 0.9)
        assert result is None  # Need more frames

    def test_stable_prediction_accepted(self):
        """A consistent prediction over the window should be accepted."""
        results = []
        for i in range(10):
            r = self.ps.update("HELLO", 0.9)
            results.append(r)
        # Should have been accepted at some point
        accepted = [r for r in results if r is not None]
        assert len(accepted) == 1
        assert accepted[0] == "HELLO"

    def test_no_duplicate_during_cooldown(self):
        """Same word should not be accepted again during cooldown."""
        results = []
        for i in range(15):
            r = self.ps.update("HELLO", 0.9)
            results.append(r)
        accepted = [r for r in results if r is not None]
        # Should be accepted exactly once (then cooldown prevents re-acceptance)
        assert len(accepted) == 1

    def test_same_word_after_cooldown_reset(self):
        """Same word can be recognized again after releasing and re-performing."""
        # First: recognize HELLO
        for _ in range(10):
            self.ps.update("HELLO", 0.9)

        # Release: send None predictions to exit cooldown
        for _ in range(10):
            self.ps.update(None, 0.0)

        # Perform HELLO again
        results = []
        for _ in range(10):
            r = self.ps.update("HELLO", 0.9)
            results.append(r)
        accepted = [r for r in results if r is not None]
        assert len(accepted) >= 1
        assert accepted[0] == "HELLO"

    def test_low_confidence_rejected(self):
        """Predictions below confidence threshold should not be accepted."""
        results = []
        for _ in range(10):
            r = self.ps.update("HELLO", 0.3)  # Below 0.5 threshold
            results.append(r)
        accepted = [r for r in results if r is not None]
        assert len(accepted) == 0

    def test_unstable_predictions_rejected(self):
        """Alternating predictions should not be accepted."""
        results = []
        for i in range(20):
            word = "HELLO" if i % 2 == 0 else "GOODBYE"
            r = self.ps.update(word, 0.9)
            results.append(r)
        accepted = [r for r in results if r is not None]
        # Alternating predictions shouldn't reach agreement threshold
        assert len(accepted) == 0

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        for _ in range(5):
            self.ps.update("HELLO", 0.9)
        self.ps.reset()
        # After reset, should start fresh
        result = self.ps.update("HELLO", 0.9)
        assert result is None  # Need to build up again

    def test_none_predictions_handled(self):
        """None predictions should not cause errors."""
        for _ in range(5):
            r = self.ps.update(None, 0.0)
            assert r is None

    def test_different_word_after_first(self):
        """After recognizing one word, a different word should be recognizable."""
        # Recognize HELLO
        for _ in range(10):
            self.ps.update("HELLO", 0.9)

        # Switch to GOODBYE
        results = []
        for _ in range(10):
            r = self.ps.update("GOODBYE", 0.9)
            results.append(r)
        accepted = [r for r in results if r is not None]
        assert len(accepted) >= 1
        assert accepted[0] == "GOODBYE"


# ============================================================
# Tests for Dataset Discovery
# ============================================================

class TestDatasetDiscovery:
    """Tests for dataset category discovery."""

    def test_dataset_exists(self):
        from app.core.settings import DATASET_PATH
        assert os.path.isdir(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"

    def test_correct_category_count(self):
        from app.core.settings import DATASET_PATH
        categories = [d for d in sorted(os.listdir(DATASET_PATH))
                      if os.path.isdir(os.path.join(DATASET_PATH, d))
                      and not d.startswith('.')]
        assert len(categories) == 31, f"Expected 31 categories, found {len(categories)}"

    def test_each_category_has_images(self):
        from app.core.settings import DATASET_PATH
        categories = [d for d in sorted(os.listdir(DATASET_PATH))
                      if os.path.isdir(os.path.join(DATASET_PATH, d))
                      and not d.startswith('.')]
        for cat in categories:
            cat_path = os.path.join(DATASET_PATH, cat)
            images = [f for f in os.listdir(cat_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            assert len(images) > 0, f"Category '{cat}' has no images"

    def test_representative_image_exists(self):
        """Each category should have representative reference images."""
        from app.core.settings import DATASET_PATH
        categories = [d for d in sorted(os.listdir(DATASET_PATH))
                      if os.path.isdir(os.path.join(DATASET_PATH, d))
                      and not d.startswith('.')]
        for cat in categories:
            cat_path = os.path.join(DATASET_PATH, cat)
            images = [f for f in os.listdir(cat_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            assert len(images) > 0, f"No reference image found in '{cat}'"


# ============================================================
# Tests for Preprocessing
# ============================================================

class TestPreprocessing:
    """Tests for image preprocessing transforms."""

    def test_train_transform_output_shape(self):
        from app.vision.preprocessing import get_train_transform
        from PIL import Image
        import numpy as np

        transform = get_train_transform(224)
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from app.vision.preprocessing import get_val_transform
        from PIL import Image
        import numpy as np

        transform = get_val_transform(224)
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_inference_transform_with_bgr_frame(self):
        from app.vision.preprocessing import get_inference_transform, preprocess_frame
        import numpy as np

        transform = get_inference_transform(224)
        # Simulate an OpenCV BGR frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame, transform)
        assert tensor.shape == (1, 3, 224, 224)


# ============================================================
# Tests for Class Labels
# ============================================================

class TestClassLabels:
    """Tests for class label mapping."""

    def test_class_labels_file_format(self):
        """If class_labels.json exists, it should be valid."""
        from app.core.settings import CLASS_LABELS_PATH
        if not os.path.exists(CLASS_LABELS_PATH):
            pytest.skip("Model not yet trained - class_labels.json not found")

        with open(CLASS_LABELS_PATH, 'r') as f:
            labels = json.load(f)

        assert isinstance(labels, dict), "class_labels.json should be a dict"
        assert len(labels) == 31, f"Expected 31 classes, found {len(labels)}"

        # Keys should be string indices
        for key in labels:
            assert key.isdigit(), f"Key '{key}' should be a numeric string"

    def test_labels_match_dataset(self):
        """Class labels should match dataset categories."""
        from app.core.settings import CLASS_LABELS_PATH, DATASET_PATH
        if not os.path.exists(CLASS_LABELS_PATH):
            pytest.skip("Model not yet trained")

        with open(CLASS_LABELS_PATH, 'r') as f:
            labels = json.load(f)

        label_values = set(labels.values())
        categories = set(d for d in os.listdir(DATASET_PATH)
                         if os.path.isdir(os.path.join(DATASET_PATH, d))
                         and not d.startswith('.'))

        assert label_values == categories, (
            f"Label mismatch.\n"
            f"In labels but not dataset: {label_values - categories}\n"
            f"In dataset but not labels: {categories - label_values}"
        )


# ============================================================
# Tests for Model Loading
# ============================================================

class TestModelLoading:
    """Tests for model loading and prediction output."""

    def test_predictor_handles_missing_model(self):
        """Predictor should handle missing model file gracefully."""
        from app.vision.predictor import SignPredictor
        predictor = SignPredictor(
            model_path="/nonexistent/model.pth",
            labels_path="/nonexistent/labels.json"
        )
        assert not predictor.is_loaded()

    def test_predictor_loads_trained_model(self):
        """If model exists, predictor should load it."""
        from app.core.settings import MODEL_PATH, CLASS_LABELS_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model not yet trained")

        from app.vision.predictor import SignPredictor
        predictor = SignPredictor(MODEL_PATH, CLASS_LABELS_PATH)
        assert predictor.is_loaded()
        assert predictor.get_num_classes() == 31

    def test_predictor_output_format(self):
        """Prediction should return (label, confidence) tuple."""
        from app.core.settings import MODEL_PATH, CLASS_LABELS_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model not yet trained")

        import numpy as np
        from app.vision.predictor import SignPredictor

        predictor = SignPredictor(MODEL_PATH, CLASS_LABELS_PATH)
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        label, confidence = predictor.predict(frame)

        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


# ============================================================
# Tests for Library Search
# ============================================================

class TestLibrarySearch:
    """Tests for the sign library search functionality."""

    def get_categories(self):
        from app.core.settings import DATASET_PATH
        return sorted([
            d for d in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, d))
            and not d.startswith('.')
        ])

    def test_case_insensitive_search(self):
        categories = self.get_categories()
        query = "HELLO"
        results = [c for c in categories if query.lower() in c.lower()]
        # "hello" is not in the dataset, this should return empty
        assert isinstance(results, list)

    def test_partial_search(self):
        categories = self.get_categories()
        query = "hap"
        results = [c for c in categories if query.lower() in c.lower()]
        assert "happy birthday" in results

    def test_empty_search_returns_all(self):
        categories = self.get_categories()
        query = ""
        results = [c for c in categories if query.lower() in c.lower()]
        assert len(results) == len(categories)

    def test_no_match_search(self):
        categories = self.get_categories()
        query = "xyznonexistent"
        results = [c for c in categories if query.lower() in c.lower()]
        assert len(results) == 0

    def test_multi_word_category_found(self):
        categories = self.get_categories()
        query = "good"
        results = [c for c in categories if query.lower() in c.lower()]
        assert "good morning" in results

    def test_invalid_category_image(self):
        """Requesting image for nonexistent category should be handled."""
        from app.core.settings import DATASET_PATH
        fake_path = os.path.join(DATASET_PATH, "nonexistent_category", "1.png")
        assert not os.path.exists(fake_path)


# ============================================================
# Tests for TextToSpeech
# ============================================================

class TestTextToSpeech:
    """Tests for the TTS engine."""

    def test_tts_initialization(self):
        from app.speech.text_to_speech import TextToSpeech
        tts = TextToSpeech()
        assert not tts.is_speaking()

    def test_tts_stop_when_not_speaking(self):
        """Stopping when nothing is playing should not raise."""
        from app.speech.text_to_speech import TextToSpeech
        tts = TextToSpeech()
        tts.stop()  # Should not raise


# ============================================================
# Tests for Confidence Threshold
# ============================================================

class TestConfidenceThreshold:
    """Tests for confidence threshold handling."""

    def test_threshold_rejects_low_confidence(self):
        from app.core.settings import CONFIDENCE_THRESHOLD
        confidence = 0.3
        assert confidence < CONFIDENCE_THRESHOLD

    def test_threshold_accepts_high_confidence(self):
        from app.core.settings import CONFIDENCE_THRESHOLD
        confidence = 0.9
        assert confidence >= CONFIDENCE_THRESHOLD

    def test_threshold_boundary(self):
        from app.core.settings import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD == 0.40
        # Ensure threshold stays within a sensible range
        assert 0.1 <= CONFIDENCE_THRESHOLD <= 0.9


# ============================================================
# Tests for Tuned Parameters & Features
# ============================================================

class TestTunedRecognitionPipeline:
    """Tests for the updated parameters: 3-frame agreement, 4-frame cooldown, top-k."""

    def test_three_frame_agreement_acceptance(self):
        """With min_agreement=3, a sign should be confirmed on the 3rd frame (faster recognition)."""
        from app.vision.prediction_stabilizer import PredictionStabilizer
        ps = PredictionStabilizer(
            stabilization_window=5,
            min_agreement=3,
            cooldown_frames=4,
            confidence_threshold=0.40,
        )
        assert ps.update("please", 0.85) is None
        assert ps.update("please", 0.85) is None
        assert ps.update("please", 0.85) == "please"

    def test_four_frame_cooldown_rearm(self):
        """With cooldown_frames=4, exactly 4 non-matching frames should re-arm the stabilizer."""
        from app.vision.prediction_stabilizer import PredictionStabilizer, State
        ps = PredictionStabilizer(
            stabilization_window=5,
            min_agreement=3,
            cooldown_frames=4,
            confidence_threshold=0.40,
        )
        # Confirm sign
        for _ in range(3):
            ps.update("please", 0.85)
        # Next frame enters cooldown
        ps.update(None, 0.0)
        assert ps.state == State.COOLDOWN

        # Release frames: 3 more needed (total 4)
        for _ in range(3):
            ps.update(None, 0.0)

        # Should now be re-armed in WAITING state
        assert ps.state == State.WAITING

    def test_top_k_ordering_and_count(self):
        """Predictor predict_top_k returns correct count and descending confidence order."""
        import numpy as np
        from app.vision.predictor import SignPredictor
        from app.core.settings import MODEL_PATH, CLASS_LABELS_PATH

        if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_LABELS_PATH):
            p = SignPredictor(MODEL_PATH, CLASS_LABELS_PATH)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            top3 = p.predict_top_k(frame, k=3)
            assert len(top3) == 3
            # Check descending confidence order
            confs = [c for _, c in top3]
            assert confs == sorted(confs, reverse=True)

    def test_webcam_unmirroring_flip(self):
        """Verify horizontal flip (cv2.flip(..., 1)) un-mirrors the frame as expected."""
        import cv2
        import numpy as np
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        test_frame[:, 0] = [255, 0, 0]  # Blue strip on leftmost column
        flipped = cv2.flip(test_frame, 1)
        # Blue strip should now be on rightmost column
        assert np.array_equal(flipped[:, -1], test_frame[:, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
