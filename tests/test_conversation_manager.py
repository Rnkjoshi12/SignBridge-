"""Unit tests for ConversationManager."""

import threading
import unittest

from app.core.conversation_manager import ConversationManager


class TestConversationManager(unittest.TestCase):
    """Test suite for ConversationManager."""

    def setUp(self):
        self.manager = ConversationManager()

    def test_initial_state(self):
        """Newly created manager should be empty."""
        self.assertTrue(self.manager.is_empty())
        self.assertEqual(self.manager.get_transcript(), [])
        self.assertEqual(self.manager.get_transcript_text(), "")
        self.assertIsNone(self.manager.get_latest())
        self.assertEqual(len(self.manager), 0)

    def test_add_words_and_order(self):
        """Words should be maintained in insertion order."""
        self.manager.add_word("hello")
        self.assertFalse(self.manager.is_empty())
        self.assertEqual(self.manager.get_transcript(), ["hello"])
        self.assertEqual(self.manager.get_transcript_text(), "hello")
        self.assertEqual(self.manager.get_latest(), "hello")

        self.manager.add_word("world")
        self.manager.add_word("how are you")
        self.assertEqual(
            self.manager.get_transcript(),
            ["hello", "world", "how are you"],
        )
        self.assertEqual(
            self.manager.get_transcript_text(),
            "hello world how are you",
        )
        self.assertEqual(self.manager.get_latest(), "how are you")
        self.assertEqual(len(self.manager), 3)

    def test_get_transcript_is_copy(self):
        """Modifying the returned transcript list should not affect internal state."""
        self.manager.add_word("apple")
        transcript = self.manager.get_transcript()
        transcript.append("banana")
        self.assertEqual(self.manager.get_transcript(), ["apple"])

    def test_clear_transcript(self):
        """clear() should remove all words and reset latest."""
        self.manager.add_word("apple")
        self.manager.add_word("banana")
        self.assertEqual(len(self.manager), 2)

        self.manager.clear()
        self.assertTrue(self.manager.is_empty())
        self.assertEqual(self.manager.get_transcript(), [])
        self.assertEqual(self.manager.get_transcript_text(), "")
        self.assertIsNone(self.manager.get_latest())

    def test_ignore_empty_or_whitespace(self):
        """Empty or whitespace-only inputs should not be added."""
        self.manager.add_word("")
        self.manager.add_word("   ")
        self.manager.add_word(None)  # type: ignore
        self.assertTrue(self.manager.is_empty())

        # Whitespace surrounding word should be stripped
        self.manager.add_word("  hello  ")
        self.assertEqual(self.manager.get_transcript(), ["hello"])

    def test_callback_word_added(self):
        """word_added subscribers should be notified on new words."""
        received = []

        def on_add(word):
            received.append(word)

        self.manager.subscribe_word_added(on_add)
        self.manager.add_word("first")
        self.manager.add_word("second")

        self.assertEqual(received, ["first", "second"])

        # Test unsubscribe
        self.manager.unsubscribe_word_added(on_add)
        self.manager.add_word("third")
        self.assertEqual(received, ["first", "second"])

    def test_callback_cleared(self):
        """cleared subscribers should be notified when transcript is cleared."""
        cleared_count = [0]

        def on_clear():
            cleared_count[0] += 1

        self.manager.subscribe_cleared(on_clear)
        self.manager.add_word("word")
        self.manager.clear()
        self.manager.clear()

        self.assertEqual(cleared_count[0], 2)

        self.manager.unsubscribe_cleared(on_clear)
        self.manager.clear()
        self.assertEqual(cleared_count[0], 2)

    def test_constructor_callbacks(self):
        """Callbacks passed to constructor should be registered."""
        added = []
        cleared = [False]

        cm = ConversationManager(
            on_word_added=lambda w: added.append(w),
            on_cleared=lambda: cleared.__setitem__(0, True),
        )
        cm.add_word("test")
        self.assertEqual(added, ["test"])
        cm.clear()
        self.assertTrue(cleared[0])

    def test_callback_error_isolation(self):
        """An error in one callback should not crash other callbacks or add_word."""
        working_received = []

        def faulty_callback(word):
            raise RuntimeError("Boom!")

        def working_callback(word):
            working_received.append(word)

        self.manager.subscribe_word_added(faulty_callback)
        self.manager.subscribe_word_added(working_callback)

        self.manager.add_word("resilient")
        self.assertEqual(working_received, ["resilient"])
        self.assertEqual(self.manager.get_latest(), "resilient")

    def test_thread_safety(self):
        """Concurrent additions from multiple threads should maintain integrity."""
        threads = []
        words_per_thread = 50
        num_threads = 4

        def worker(thread_id):
            for i in range(words_per_thread):
                self.manager.add_word(f"t{thread_id}-w{i}")

        for t in range(num_threads):
            th = threading.Thread(target=worker, args=(t,))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        self.assertEqual(len(self.manager), num_threads * words_per_thread)
        self.assertEqual(len(self.manager.get_transcript()), num_threads * words_per_thread)


if __name__ == "__main__":
    unittest.main()
