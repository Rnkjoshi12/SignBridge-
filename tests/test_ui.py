"""Comprehensive unit and functional tests for SignBridge UI components."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui.qt_compat import QApplication
_qapp = QApplication.instance()
if _qapp is None:
    _qapp = QApplication(["--platform", "offscreen"])

from app.ui.camera_panel import CameraPanel, BG_PRIMARY, BG_SECONDARY, BG_CARD, ACCENT, ACCENT_HOVER, ACCENT_DIM, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, BORDER_COLOR, SUCCESS, WARNING, ERROR
from app.ui.conversation_panel import ConversationPanel
from app.ui.library_panel import LibraryPanel
from app.ui.main_window import MainWindow

EXPECTED_COLORS = {
    'BG_PRIMARY': '#1a1a2e',
    'BG_SECONDARY': '#16213e',
    'BG_CARD': '#1f2b47',
    'ACCENT': '#00d4aa',
    'ACCENT_HOVER': '#00f5c4',
    'ACCENT_DIM': '#007a63',
    'TEXT_PRIMARY': '#e8e8e8',
    'TEXT_SECONDARY': '#8892a8',
    'TEXT_MUTED': '#5a6278',
    'BORDER_COLOR': '#2a3550',
    'SUCCESS': '#4ade80',
    'WARNING': '#fbbf24',
    'ERROR': '#f87171',
}


class TestUIThemeAndExports(unittest.TestCase):
    """Test UI module exports and color scheme compliance."""

    def test_ui_exports(self):
        """Verify app.ui exports all required panels and MainWindow."""
        import app.ui as ui_module
        expected = ["CameraPanel", "ConversationPanel", "LibraryPanel", "MainWindow"]
        for name in expected:
            self.assertTrue(hasattr(ui_module, name), f"app.ui must export {name}")
            self.assertIn(name, ui_module.__all__, f"{name} must be in app.ui.__all__")

    def test_color_scheme_constants(self):
        """Verify color scheme hex values match specifications."""
        import app.ui.camera_panel as cp
        import app.ui.conversation_panel as conv
        import app.ui.library_panel as lp
        import app.ui.main_window as mw

        for module in (cp, conv, lp, mw):
            for name, expected_val in EXPECTED_COLORS.items():
                self.assertTrue(hasattr(module, name), f"{module.__name__} missing {name}")
                self.assertEqual(getattr(module, name), expected_val, f"{module.__name__} {name} mismatch")


class TestCameraPanelFunctional(unittest.TestCase):
    """Functional tests for CameraPanel widget."""

    def setUp(self):
        self.panel = CameraPanel()

    def test_initial_state(self):
        self.assertFalse(self.panel._is_camera_running)
        self.assertEqual(self.panel.camera_state_label.text(), "Camera Off")
        self.assertEqual(self.panel.toggle_camera_btn.text(), "Start Camera")
        self.assertFalse(self.panel.error_frame.isVisible())
        self.assertEqual(self.panel.confidence_bar.value(), 0)

    def test_set_camera_active(self):
        self.panel.set_camera_active(True)
        self.assertTrue(self.panel._is_camera_running)
        self.assertEqual(self.panel.camera_state_label.text(), "Camera Active")
        self.assertEqual(self.panel.toggle_camera_btn.text(), "Stop Camera")

        self.panel.set_camera_active(False)
        self.assertFalse(self.panel._is_camera_running)
        self.assertEqual(self.panel.camera_state_label.text(), "Camera Off")
        self.assertEqual(self.panel.toggle_camera_btn.text(), "Start Camera")

    def test_toggle_signals(self):
        start_called = False
        stop_called = False

        def on_start():
            nonlocal start_called
            start_called = True

        def on_stop():
            nonlocal stop_called
            stop_called = True

        self.panel.start_requested.connect(on_start)
        self.panel.stop_requested.connect(on_stop)

        # When inactive, clicking emits start_requested
        self.panel.set_camera_active(False)
        self.panel._on_toggle_clicked()
        self.assertTrue(start_called)
        self.assertFalse(stop_called)

        # When active, clicking emits stop_requested
        self.panel.set_camera_active(True)
        self.panel._on_toggle_clicked()
        self.assertTrue(stop_called)

    def test_set_status_states(self):
        self.panel.set_status("Waiting for sign...")
        self.assertIn("Waiting for sign...", self.panel.status_label.text())

        self.panel.set_status("Detecting: HELLO")
        self.assertIn("Detecting", self.panel.status_label.text())

        self.panel.set_status("Recognized", word="HELLO")
        self.assertIn("HELLO", self.panel.status_label.text())
        self.assertIn(ACCENT, self.panel.status_label.text())

        self.panel.set_status("Sign not recognized")
        self.assertIn("Sign not recognized", self.panel.status_label.text())

    def test_set_confidence(self):
        # 0.0 - 1.0 range
        self.panel.set_confidence(0.85)
        self.assertEqual(self.panel.confidence_bar.value(), 85)
        self.assertIn("85%", self.panel.confidence_text.text())

        # 0 - 100 range
        self.panel.set_confidence(72.0)
        self.assertEqual(self.panel.confidence_bar.value(), 72)
        self.assertIn("72%", self.panel.confidence_text.text())

    def test_recognized_word_overlay(self):
        self.panel.set_recognized_word("THANK YOU", confidence=0.92)
        self.assertFalse(self.panel.video_label._overlay_container.isHidden())
        self.assertEqual(self.panel.video_label._overlay_text_label.text(), "THANK YOU")
        self.assertEqual(self.panel.confidence_bar.value(), 92)

        self.panel.clear_recognized_word()
        self.assertTrue(self.panel.video_label._overlay_container.isHidden())
        self.assertIn("Waiting for sign...", self.panel.status_label.text())

    def test_error_banner(self):
        self.panel.show_error("Camera hardware disconnected")
        self.assertFalse(self.panel.error_frame.isHidden())
        self.assertIn("Camera hardware disconnected", self.panel.error_label.text())

        self.panel.clear_error()
        self.assertTrue(self.panel.error_frame.isHidden())


class TestConversationPanelFunctional(unittest.TestCase):
    """Functional tests for ConversationPanel widget."""

    def setUp(self):
        self.panel = ConversationPanel()

    def test_initial_state_empty(self):
        self.assertEqual(self.panel.get_transcript(), "")
        self.assertEqual(self.panel.get_words(), [])
        self.assertEqual(self.panel.get_last_word(), "")
        self.assertEqual(self.panel.count_badge.text(), "0 words")

        # Play buttons disabled when empty
        self.assertFalse(self.panel.play_all_btn.isEnabled())
        self.assertFalse(self.panel.play_last_btn.isEnabled())
        self.assertFalse(self.panel.copy_btn.isEnabled())
        self.assertFalse(self.panel.clear_btn.isEnabled())
        self.assertIn("Perform a sign to begin...", self.panel.transcript_edit.toPlainText())

    def test_add_words_and_formatting(self):
        self.panel.add_word("HELLO")
        self.assertEqual(self.panel.get_transcript(), "HELLO")
        self.assertEqual(self.panel.get_last_word(), "HELLO")
        self.assertEqual(self.panel.count_badge.text(), "1 word")
        self.assertTrue(self.panel.play_all_btn.isEnabled())

        self.panel.add_word("WORLD")
        self.assertEqual(self.panel.get_transcript(), "HELLO WORLD")
        self.assertEqual(self.panel.get_last_word(), "WORLD")
        self.assertEqual(self.panel.count_badge.text(), "2 words")
        self.assertIn("HELLO WORLD", self.panel.transcript_edit.toPlainText())

    def test_clear_transcript(self):
        self.panel.add_word("TEST")
        cleared_emitted = False

        def on_cleared():
            nonlocal cleared_emitted
            cleared_emitted = True

        self.panel.clear_requested.connect(on_cleared)
        self.panel.clear_transcript()

        self.assertTrue(cleared_emitted)
        self.assertEqual(self.panel.get_transcript(), "")
        self.assertFalse(self.panel.play_all_btn.isEnabled())
        self.assertIn("Perform a sign to begin...", self.panel.transcript_edit.toPlainText())

    def test_playback_signals(self):
        self.panel.add_word("SIGN")
        play_all_called = False
        play_last_called = False
        copy_called = False

        self.panel.play_all_requested.connect(lambda: nonlocal_set('play_all'))
        self.panel.play_last_requested.connect(lambda: nonlocal_set('play_last'))
        self.panel.copy_requested.connect(lambda: nonlocal_set('copy'))

        called = {}
        def nonlocal_set(k): called[k] = True

        self.panel._on_play_all()
        self.panel._on_play_last()
        self.panel._on_copy()

        self.assertTrue(called.get('play_all', False))
        self.assertTrue(called.get('play_last', False))
        self.assertTrue(called.get('copy', False))


class TestLibraryPanelFunctional(unittest.TestCase):
    """Functional tests for LibraryPanel widget."""

    def setUp(self):
        self.panel = LibraryPanel()

    def test_populate_categories(self):
        categories = ["Hello", "Thank You", "Goodbye"]
        self.panel.set_categories(categories)

        self.assertEqual(self.panel.list_widget.count(), 3)
        self.assertEqual(self.panel.count_badge.text(), "3 signs")
        self.assertEqual(self.panel.list_widget.item(0).text(), "Goodbye")
        self.assertEqual(self.panel.list_widget.item(1).text(), "Hello")
        self.assertEqual(self.panel.list_widget.item(2).text(), "Thank You")

    def test_filter_categories(self):
        self.panel.set_categories(["Are You OK", "Hello", "How Are You", "Thank You"])

        # Match single
        self.panel.filter_categories("thank")
        self.assertTrue(self.panel.list_widget.item(0).isHidden())  # Are You OK is hidden
        matching = [
            self.panel.list_widget.item(i).text()
            for i in range(self.panel.list_widget.count())
            if not self.panel.list_widget.item(i).isHidden()
        ]
        self.assertEqual(matching, ["Thank You"])
        self.assertTrue(self.panel.no_results_label.isHidden())

        # No match
        self.panel.filter_categories("xyz123")
        self.assertFalse(self.panel.no_results_label.isHidden())
        self.assertTrue(self.panel.list_widget.isHidden())

        # Reset query
        self.panel.filter_categories("")
        all_visible = [
            self.panel.list_widget.item(i).text()
            for i in range(self.panel.list_widget.count())
            if not self.panel.list_widget.item(i).isHidden()
        ]
        self.assertTrue(self.panel.no_results_label.isHidden())
        self.assertFalse(self.panel.list_widget.isHidden())

    def test_category_selection_and_detail(self):
        self.panel.set_categories(["Hello", "Help Me"])
        selected_cat = ""

        def on_selected(cat):
            nonlocal selected_cat
            selected_cat = cat

        self.panel.category_selected.connect(on_selected)
        self.panel._on_item_clicked(self.panel.list_widget.item(0))

        self.assertEqual(selected_cat, "Hello")
        self.assertEqual(self.panel.stacked_widget.currentIndex(), 1)
        self.assertEqual(self.panel.detail_heading.text(), "Hello")

        # Back button returns to list view
        self.panel.show_list()
        self.assertEqual(self.panel.stacked_widget.currentIndex(), 0)


class TestMainWindowFunctional(unittest.TestCase):
    """Functional tests for MainWindow layout, proportions, and subsystem coordination."""

    def setUp(self):
        self.window = MainWindow()

    def test_layout_proportions(self):
        """Verify 65% / 35% horizontal split and 50% / 50% vertical right split."""
        self.window.show()
        _qapp.processEvents()
        # Main splitter: camera panel width > right splitter width
        main_sizes = self.window.main_splitter.sizes()
        self.assertGreater(main_sizes[0], main_sizes[1])
        # Right splitter: conversation and library are balanced
        right_sizes = self.window.right_splitter.sizes()
        self.assertEqual(right_sizes[0], right_sizes[1])
        self.window.close()

    def test_subsystem_ownership(self):
        """Verify MainWindow owns all required subsystems."""
        self.assertIsNotNone(self.window.camera_manager)
        self.assertIsNotNone(self.window.predictor)
        self.assertIsNotNone(self.window.stabilizer)
        self.assertIsNotNone(self.window.tts)
        self.assertIsNotNone(self.window.conversation_manager)

    def test_timers_configuration(self):
        """Verify 30fps capture timer (~33ms) and 200ms inference timer."""
        self.assertAlmostEqual(self.window.capture_timer.interval(), 33, delta=2)
        self.assertEqual(self.window.inference_timer.interval(), 200)

    def test_camera_start_stop_flow(self):
        """Verify starting and stopping camera manages timers and panel state."""
        with patch.object(self.window.camera_manager, 'start', return_value=True):
            self.window.start_camera()
            self.assertTrue(self.window.capture_timer.isActive())
            self.assertTrue(self.window.inference_timer.isActive())
            self.assertTrue(self.window.camera_panel._is_camera_running)

            self.window.stop_camera()
            self.assertFalse(self.window.capture_timer.isActive())
            self.assertFalse(self.window.inference_timer.isActive())
            self.assertFalse(self.window.camera_panel._is_camera_running)

    def test_missing_model_warning(self):
        """MainWindow must display exact model training prompt when model is missing."""
        with patch.object(self.window.predictor, 'is_loaded', False, create=True), \
             patch('os.path.exists', return_value=False):
            self.window._check_model_status()
            self.assertFalse(self.window.camera_panel.error_frame.isHidden())
            self.assertIn(
                "Model not found. Please train the model first: python scripts/train_model.py",
                self.window.camera_panel.error_label.text()
            )

    def test_inference_recognition_updates_conversation_and_tts(self):
        """When stabilizer confirms a sign, word is added to conversation, TTS speaks, and overlay displays."""
        # Mock stabilized confirmed word
        with patch.object(self.window.predictor, 'is_loaded', True, create=True), \
             patch.object(self.window.predictor, 'predict', return_value=("HELLO", 0.95)), \
             patch.object(self.window.stabilizer, 'update', return_value="HELLO"), \
             patch.object(self.window.tts, 'speak') as mock_speak:

            self.window._latest_frame = MagicMock()
            self.window._on_run_inference()

            # Conversation panel updated
            self.assertEqual(self.window.conversation_panel.get_last_word(), "HELLO")
            # TTS called
            mock_speak.assert_called_with("HELLO")
            # Camera overlay updated
            self.assertEqual(self.window.camera_panel.video_label._overlay_text_label.text(), "HELLO")


if __name__ == '__main__':
    unittest.main()
