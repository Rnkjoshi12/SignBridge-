"""Unit tests for PredictionStabilizer."""

import unittest

from app.vision.prediction_stabilizer import PredictionStabilizer, StabilizerState


class TestPredictionStabilizer(unittest.TestCase):
    """Test suite for PredictionStabilizer."""

    def test_initial_state(self):
        """Stabilizer should start in WAITING state with empty window."""
        s = PredictionStabilizer(
            stabilization_window=10,
            min_agreement=7,
            cooldown_frames=15,
            confidence_threshold=0.6,
        )
        self.assertEqual(s.state, StabilizerState.WAITING)
        self.assertEqual(s.cooldown_remaining, 0)
        self.assertIsNone(s.last_confirmed)
        self.assertEqual(len(s.window), 0)

    def test_invalid_params(self):
        """Constructor should validate non-positive window or agreement."""
        with self.assertRaises(ValueError):
            PredictionStabilizer(stabilization_window=0)
        with self.assertRaises(ValueError):
            PredictionStabilizer(min_agreement=0)
        with self.assertRaises(ValueError):
            PredictionStabilizer(cooldown_frames=-1)

    def test_detecting_state_transition(self):
        """Single prediction should transition state from WAITING to DETECTING."""
        s = PredictionStabilizer(stabilization_window=5, min_agreement=3)
        self.assertEqual(s.state, StabilizerState.WAITING)

        result = s.update("hello", 0.9)
        self.assertIsNone(result)
        self.assertEqual(s.state, StabilizerState.DETECTING)

    def test_confirmed_prediction(self):
        """Prediction should be confirmed when agreement and confidence thresholds are reached."""
        s = PredictionStabilizer(
            stabilization_window=5,
            min_agreement=3,
            cooldown_frames=4,
            confidence_threshold=0.7,
        )

        self.assertIsNone(s.update("hello", 0.8))  # 1/3 agreement -> DETECTING
        self.assertEqual(s.state, StabilizerState.DETECTING)

        self.assertIsNone(s.update("hello", 0.85))  # 2/3 agreement -> DETECTING
        self.assertEqual(s.state, StabilizerState.DETECTING)

        confirmed = s.update("hello", 0.9)  # 3/3 agreement -> CONFIRMED!
        self.assertEqual(confirmed, "hello")
        self.assertEqual(s.state, StabilizerState.CONFIRMED)
        self.assertEqual(s.last_confirmed, "hello")

    def test_confidence_threshold_rejection(self):
        """Even with agreement, low confidence should prevent confirmation."""
        s = PredictionStabilizer(
            stabilization_window=3,
            min_agreement=3,
            cooldown_frames=5,
            confidence_threshold=0.8,
        )

        s.update("hello", 0.5)
        s.update("hello", 0.6)
        result = s.update("hello", 0.5)  # Avg conf = 0.533 < 0.8

        self.assertIsNone(result)
        self.assertEqual(s.state, StabilizerState.DETECTING)

    def test_cooldown_lifecycle_and_same_word_repeat(self):
        """After CONFIRMED, enters COOLDOWN for cooldown_frames updates, then returns to WAITING."""
        cooldown = 3
        s = PredictionStabilizer(
            stabilization_window=3,
            min_agreement=2,
            cooldown_frames=cooldown,
            confidence_threshold=0.5,
        )

        # Trigger confirmation
        s.update("thank_you", 0.9)
        res = s.update("thank_you", 0.9)
        self.assertEqual(res, "thank_you")
        self.assertEqual(s.state, StabilizerState.CONFIRMED)

        # Release sign: update 1 in cooldown
        self.assertIsNone(s.update(None, 0.0))
        self.assertEqual(s.state, StabilizerState.COOLDOWN)
        self.assertEqual(s.cooldown_remaining, 2)

        # Update 2 in cooldown
        self.assertIsNone(s.update(None, 0.0))
        self.assertEqual(s.state, StabilizerState.COOLDOWN)
        self.assertEqual(s.cooldown_remaining, 1)

        # Update 3 in cooldown - finishes cooldown
        self.assertIsNone(s.update(None, 0.0))
        self.assertEqual(s.state, StabilizerState.WAITING)
        self.assertEqual(s.cooldown_remaining, 0)

        # Now outside cooldown: can confirm the SAME word again once agreement is met
        s.update("thank_you", 0.9)
        res2 = s.update("thank_you", 0.9)
        self.assertEqual(res2, "thank_you")
        self.assertEqual(s.state, StabilizerState.CONFIRMED)

    def test_no_duplicate_while_holding_sign(self):
        """Continuous frames of the same sign should not emit duplicates."""
        s = PredictionStabilizer(
            stabilization_window=3,
            min_agreement=2,
            cooldown_frames=3,
            confidence_threshold=0.5,
        )

        results = []
        for _ in range(10):
            r = s.update("hello", 0.9)
            results.append(r)

        accepted = [r for r in results if r is not None]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0], "hello")

    def test_reset(self):
        """reset() should return stabilizer to pure WAITING state and clear history."""
        s = PredictionStabilizer(stabilization_window=5, min_agreement=2, cooldown_frames=5)
        s.update("hello", 0.9)
        s.update("hello", 0.9)
        self.assertEqual(s.state, StabilizerState.CONFIRMED)

        s.reset()
        self.assertEqual(s.state, StabilizerState.WAITING)
        self.assertEqual(s.cooldown_remaining, 0)
        self.assertIsNone(s.last_confirmed)
        self.assertEqual(len(s.window), 0)

    def test_cooldown_zero(self):
        """If cooldown_frames is 0, should transition directly back to WAITING without lingering."""
        s = PredictionStabilizer(stabilization_window=2, min_agreement=2, cooldown_frames=0)
        s.update("test", 0.9)
        res = s.update("test", 0.9)
        self.assertEqual(res, "test")
        self.assertEqual(s.state, StabilizerState.CONFIRMED)

        # Next update transitions immediately to WAITING and can evaluate candidate
        s.update(None, 0.0)
        s.update("test", 0.9)
        s.update("test", 0.9)
        self.assertEqual(s.state, StabilizerState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
