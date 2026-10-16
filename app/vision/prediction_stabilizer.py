"""PredictionStabilizer — Temporal smoothing for sign recognition.

Implements a state machine that requires multiple consistent predictions before
accepting a sign, preventing duplicate announcements while allowing the same
word to be re-recognized after release and cooldown.

States:
    WAITING   — No consistent prediction yet / hands absent
    DETECTING — Building confidence in a candidate prediction
    CONFIRMED — A prediction has been accepted (emitted once)
    COOLDOWN  — Waiting before allowing re-recognition
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import logging
import threading
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


class State(str, Enum):
    """States for PredictionStabilizer."""

    WAITING = "WAITING"
    DETECTING = "DETECTING"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


# Alias for compatibility
StabilizerState = State


class PredictionStabilizer:
    """Smooths frame-by-frame predictions into stable accepted signs.

    Args:
        stabilization_window: Number of recent frames to consider in rolling window.
        min_agreement: Minimum frames that must agree on a prediction.
        cooldown_frames: Frames to wait after acceptance before re-arming.
        confidence_threshold: Minimum average confidence for acceptance.
    """

    def __init__(
        self,
        stabilization_window: int = 5,
        min_agreement: int = 4,
        cooldown_frames: int = 8,
        confidence_threshold: float = 0.55,
    ) -> None:
        if stabilization_window <= 0:
            raise ValueError("stabilization_window must be > 0")
        if min_agreement <= 0:
            raise ValueError("min_agreement must be > 0")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be >= 0")

        self.stabilization_window = stabilization_window
        self.min_agreement = min(min_agreement, stabilization_window)
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = float(confidence_threshold)

        # Internal state
        self._lock = threading.Lock()
        self._state: State = State.WAITING
        self._window: Deque[Tuple[Optional[str], float]] = deque(maxlen=stabilization_window)
        self._cooldown_counter: int = 0
        self._last_accepted: Optional[str] = None

    def update(self, prediction: Optional[str], confidence: float = 0.0) -> Optional[str]:
        """Process a new frame prediction.

        Args:
            prediction: The predicted label, or None if no sign detected.
            confidence: The confidence score for this prediction.

        Returns:
            The accepted label string if a sign is confirmed this frame,
            or None if no new sign should be announced.
        """
        with self._lock:
            cleaned_pred: Optional[str] = None
            if prediction is not None and isinstance(prediction, str):
                s = prediction.strip()
                if s:
                    cleaned_pred = s

            conf = float(confidence) if cleaned_pred is not None else 0.0

            # 1. Add to rolling window
            self._window.append((cleaned_pred, conf))

            # 2. Transition from CONFIRMED -> COOLDOWN or WAITING
            if self._state == State.CONFIRMED:
                if self.cooldown_frames > 0:
                    self._state = State.COOLDOWN
                    self._cooldown_counter = 0
                else:
                    self._state = State.WAITING
                    self._last_accepted = None

            # 3. Handle COOLDOWN state
            if self._state == State.COOLDOWN:
                # If sign is released (None) or changed, advance cooldown counter
                if cleaned_pred is None or cleaned_pred != self._last_accepted:
                    self._cooldown_counter += 1

                if self._cooldown_counter >= self.cooldown_frames:
                    self._state = State.WAITING
                    self._last_accepted = None
                    self._cooldown_counter = 0
                    self._window.clear()

                return None

            # 4. Handle WAITING / DETECTING state
            if cleaned_pred is None:
                # None counts as 'no sign'
                valid_in_window = [p for p, _ in self._window if p is not None]
                if not valid_in_window:
                    self._state = State.WAITING
                return None

            # Count occurrences of the current candidate in the window
            matching = [(p, c) for p, c in self._window if p == cleaned_pred]
            count = len(matching)

            if count >= self.min_agreement:
                avg_confidence = sum(c for _, c in matching) / count
                if avg_confidence >= self.confidence_threshold:
                    self._state = State.CONFIRMED
                    self._last_accepted = cleaned_pred
                    return cleaned_pred

            # Detections exist but agreement or confidence not yet reached
            self._state = State.DETECTING if count >= 1 else State.WAITING
            return None

    def reset(self) -> None:
        """Reset all stabilization state."""
        with self._lock:
            self._state = State.WAITING
            self._window.clear()
            self._cooldown_counter = 0
            self._last_accepted = None

    @property
    def state(self) -> State:
        """Return the current state."""
        with self._lock:
            return self._state

    def get_state(self) -> str:
        """Return the current state name."""
        with self._lock:
            return self._state.name

    @property
    def last_accepted(self) -> Optional[str]:
        """Return the last accepted prediction."""
        with self._lock:
            return self._last_accepted

    @property
    def last_confirmed(self) -> Optional[str]:
        """Alias for last_accepted."""
        with self._lock:
            return self._last_accepted

    @property
    def cooldown_remaining(self) -> int:
        """Remaining frames in cooldown."""
        with self._lock:
            if self._state != State.COOLDOWN:
                return 0
            return max(0, self.cooldown_frames - self._cooldown_counter)

    @property
    def window(self) -> List[Tuple[Optional[str], float]]:
        """Return snapshot of rolling window."""
        with self._lock:
            return list(self._window)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"PredictionStabilizer(state={self._state.name}, "
                f"window_size={len(self._window)}, "
                f"last_confirmed={self._last_accepted})"
            )
