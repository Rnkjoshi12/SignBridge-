"""Unit tests for TextToSpeech."""

import unittest
from unittest.mock import MagicMock, patch

from app.speech.text_to_speech import TextToSpeech


class TestTextToSpeech(unittest.TestCase):
    """Test suite for TextToSpeech engine."""

    def setUp(self):
        self.tts = TextToSpeech(rate=180, voice="Alex")
        if not self.tts._say_binary:
            self.tts._say_binary = "say"

    def tearDown(self):
        self.tts.close()

    def test_initial_settings(self):
        """Rate and voice should be initialized properly."""
        self.assertEqual(self.tts.get_rate(), 180)
        self.assertEqual(self.tts.get_voice(), "Alex")
        self.assertFalse(self.tts.is_speaking())

    def test_set_rate(self):
        """set_rate should update rate and reject invalid values."""
        self.tts.set_rate(200)
        self.assertEqual(self.tts.get_rate(), 200)

        with self.assertRaises(ValueError):
            self.tts.set_rate(0)
        with self.assertRaises(ValueError):
            self.tts.set_rate(-10)

    def test_set_voice(self):
        """set_voice should update voice name."""
        self.tts.set_voice("Samantha")
        self.assertEqual(self.tts.get_voice(), "Samantha")

        self.tts.set_voice(None)
        self.assertIsNone(self.tts.get_voice())

        self.tts.set_voice("   ")
        self.assertIsNone(self.tts.get_voice())

    def test_ignore_empty_input(self):
        """Empty or whitespace input should be ignored."""
        self.tts.speak("")
        self.tts.speak("   ")
        self.tts.speak(None)  # type: ignore
        self.assertFalse(self.tts.is_speaking())

        self.tts.speak_sync("")
        self.tts.speak_sync("   ")
        self.tts.speak_sync(None)  # type: ignore
        self.assertFalse(self.tts.is_speaking())

    @patch("subprocess.Popen")
    def test_speak_sync_execution(self, mock_popen):
        """speak_sync should invoke 'say' with correct arguments and block."""
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        self.tts.speak_sync("Hello world")

        mock_popen.assert_called_once()
        cmd_args = mock_popen.call_args[0][0]
        self.assertIn("-r", cmd_args)
        self.assertIn("180", cmd_args)
        self.assertIn("-v", cmd_args)
        self.assertIn("Alex", cmd_args)
        self.assertIn("Hello world", cmd_args)

    @patch("subprocess.Popen")
    def test_stop_terminates_process(self, mock_popen):
        """stop() should terminate any active process."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # running
        mock_popen.return_value = mock_proc

        self.tts._process = mock_proc
        self.tts._speaking = True

        self.tts.stop()

        mock_proc.terminate.assert_called_once()
        self.assertFalse(self.tts.is_speaking())
        self.assertIsNone(self.tts._process)

    def test_rapid_calls_cancel_previous(self):
        """Rapid calls should cancel previous ongoing speech to prevent overlap."""
        with patch.object(self.tts, "stop") as mock_stop:
            with patch.object(self.tts, "_speak_impl"):
                self.tts.speak("First")
                self.tts.speak("Second")
                self.assertEqual(mock_stop.call_count, 2)

    @patch("shutil.which", return_value=None)
    def test_graceful_missing_binary(self, _):
        """Should handle missing 'say' binary gracefully without raising."""
        tts_no_say = TextToSpeech()
        tts_no_say.speak_sync("Hello")
        tts_no_say.speak("Hello")
        tts_no_say.close()


if __name__ == "__main__":
    unittest.main()
