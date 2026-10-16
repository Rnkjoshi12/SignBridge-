"""Conversation Manager for SignBridge.

Maintains an ordered transcript of recognized words/phrases, notifies listeners
via simple callbacks/signals when words are added or the transcript is cleared,
and provides full thread safety.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages the running transcript of recognized sign language signs.

    Provides methods to add words, retrieve the transcript,
    clear it, and notify listeners via callbacks.
    """

    def __init__(
        self,
        on_word_added: Optional[Callable[[str], Any]] = None,
        on_cleared: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Initialize the ConversationManager.

        Args:
            on_word_added: Optional callable invoked with (word: str) when a word is added.
            on_cleared: Optional callable invoked with () when the transcript is cleared.
        """
        self._lock = threading.RLock()
        self._transcript: List[str] = []

        # Simple callback attributes for direct assignment or constructor passing
        self.on_word_added: Optional[Callable[[str], Any]] = on_word_added
        self.on_cleared: Optional[Callable[[], Any]] = on_cleared

        # Subscription list for multiple listeners
        self._word_added_subscribers: List[Callable[[str], Any]] = []
        self._cleared_subscribers: List[Callable[[], Any]] = []

        if on_word_added is not None:
            self._word_added_subscribers.append(on_word_added)
        if on_cleared is not None:
            self._cleared_subscribers.append(on_cleared)

    def add_word(self, word: str) -> None:
        """Appends a recognized word or phrase to the transcript.

        Args:
            word: The recognized sign label to append.
        """
        if not word or not isinstance(word, str):
            return

        cleaned = word.strip()
        if not cleaned:
            return

        with self._lock:
            self._transcript.append(cleaned)

        # Notify direct callback if set
        if self.on_word_added is not None:
            try:
                self.on_word_added(cleaned)
            except Exception as e:
                logger.exception("Error in on_word_added callback: %s", e)

        # Notify registered subscribers
        with self._lock:
            subscribers = list(self._word_added_subscribers)

        for cb in subscribers:
            if cb is not self.on_word_added:
                try:
                    cb(cleaned)
                except Exception as e:
                    logger.exception("Error in word_added subscriber: %s", e)

    def get_transcript(self) -> List[str]:
        """Returns a copy of the ordered list of recognized words."""
        with self._lock:
            return list(self._transcript)

    def get_transcript_text(self) -> str:
        """Returns the recognized words joined by spaces."""
        with self._lock:
            return " ".join(self._transcript)

    def get_latest(self) -> Optional[str]:
        """Returns the most recently added word, or None if empty."""
        with self._lock:
            if not self._transcript:
                return None
            return self._transcript[-1]

    def clear(self) -> None:
        """Clears all words from the transcript and notifies listeners."""
        with self._lock:
            self._transcript.clear()

        # Notify direct callback if set
        if self.on_cleared is not None:
            try:
                self.on_cleared()
            except Exception as e:
                logger.exception("Error in on_cleared callback: %s", e)

        # Notify registered subscribers
        with self._lock:
            subscribers = list(self._cleared_subscribers)

        for cb in subscribers:
            if cb is not self.on_cleared:
                try:
                    cb()
                except Exception as e:
                    logger.exception("Error in cleared subscriber: %s", e)

    def is_empty(self) -> bool:
        """Returns True if the transcript is empty, False otherwise."""
        with self._lock:
            return len(self._transcript) == 0

    def subscribe_word_added(self, callback: Callable[[str], Any]) -> None:
        """Registers an additional callback for word additions."""
        with self._lock:
            if callable(callback) and callback not in self._word_added_subscribers:
                self._word_added_subscribers.append(callback)

    def unsubscribe_word_added(self, callback: Callable[[str], Any]) -> None:
        """Removes a previously registered word addition callback."""
        with self._lock:
            if callback in self._word_added_subscribers:
                self._word_added_subscribers.remove(callback)

    def subscribe_cleared(self, callback: Callable[[], Any]) -> None:
        """Registers an additional callback for transcript clears."""
        with self._lock:
            if callable(callback) and callback not in self._cleared_subscribers:
                self._cleared_subscribers.append(callback)

    def unsubscribe_cleared(self, callback: Callable[[], Any]) -> None:
        """Removes a previously registered clear callback."""
        with self._lock:
            if callback in self._cleared_subscribers:
                self._cleared_subscribers.remove(callback)

    def __len__(self) -> int:
        with self._lock:
            return len(self._transcript)

    def __iter__(self):
        return iter(self.get_transcript())

    def __repr__(self) -> str:
        with self._lock:
            return f"ConversationManager(words={len(self._transcript)})"

    def __str__(self) -> str:
        return self.get_transcript_text()
