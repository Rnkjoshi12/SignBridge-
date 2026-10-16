"""Main application window tying together camera, recognition, conversation, and library panels."""

import os
import sys
from typing import List, Optional

import cv2

from app.ui.qt_compat import (
    Qt,
    QTimer,
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QSplitter,
)

from app.ui.camera_panel import CameraPanel
from app.ui.conversation_panel import ConversationPanel
from app.ui.library_panel import LibraryPanel

# COLOR SCHEME (Exact specifications)
BG_PRIMARY = '#1a1a2e'
BG_SECONDARY = '#16213e'
BG_CARD = '#1f2b47'
ACCENT = '#00d4aa'
ACCENT_HOVER = '#00f5c4'
ACCENT_DIM = '#007a63'
TEXT_PRIMARY = '#e8e8e8'
TEXT_SECONDARY = '#8892a8'
TEXT_MUTED = '#5a6278'
BORDER_COLOR = '#2a3550'
SUCCESS = '#4ade80'
WARNING = '#fbbf24'
ERROR = '#f87171'

# Global Stylesheet
GLOBAL_STYLESHEET = f"""
QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
}}
QScrollBar:vertical {{
    background: {BG_SECONDARY};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COLOR};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
}}
QToolTip {{
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    padding: 4px 8px;
    border-radius: 4px;
}}
"""

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_PATH = os.path.join(_ROOT, "dataset", "cleaned_images")
DEFAULT_MODEL_PATH = os.path.join(_ROOT, "models", "yolov5_sign_cls.pt")
DEFAULT_LABELS_PATH = os.path.join(_ROOT, "models", "class_labels.json")

# Fallback 31 phrase categories matching the custom trained dataset
FALLBACK_PHRASE_CATEGORIES = [
    "again", "agree", "answer", "attendance", "book", "break", "careful",
    "congratulations", "good morning", "happy birthday", "home",
    "how are you", "hungry", "i need help", "keepsmile", "meet", "mistake",
    "please", "practice", "pressure", "sick", "stop", "sun", "team",
    "thirsty", "this", "together", "understand", "wait", "where", "write",
]

# Robust imports for vision, speech, and core modules with graceful fallback shims
try:
    from app.vision.camera_manager import CameraManager
    from app.vision.predictor import SignPredictor  
    from app.vision.prediction_stabilizer import PredictionStabilizer
    from app.vision.preprocessing import get_inference_transform
    from app.speech.text_to_speech import TextToSpeech
    from app.core.conversation_manager import ConversationManager
    from app.core.settings import *
except ImportError:
    # Graceful fallbacks for testing or prior to background module assembly
    class CameraManager:
        def __init__(self, camera_index: int = 0, target_fps: int = 30):
            self.camera_index = camera_index
            self.target_fps = target_fps
            self._running = False

        def start(self) -> bool:
            self._running = True
            return True

        def stop(self) -> None:
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def read_frame(self):
            return False, None

    class SignPredictor:
        def __init__(self, model_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH, **kwargs):
            self.model_path = model_path
            self.labels_path = labels_path
            self._is_loaded = os.path.exists(model_path)
            self.labels = list(FALLBACK_PHRASE_CATEGORIES)

        def predict(self, frame):
            return None, 0.0

    class PredictionStabilizer:
        def __init__(self, stabilization_window: int = 5, min_agreement: int = 4, cooldown_frames: int = 8, confidence_threshold: float = 0.55):
            self.stabilization_window = stabilization_window
            self.min_agreement = min_agreement
            self.cooldown_frames = cooldown_frames
            self.confidence_threshold = confidence_threshold

        def update(self, prediction: Optional[str], confidence: float = 0.0) -> Optional[str]:
            return None

        def reset(self) -> None:
            pass

    def get_inference_transform():
        return lambda x: x

    class TextToSpeech:
        def __init__(self, rate: int = 180, voice: Optional[str] = None):
            self._busy = False

        def speak(self, text: str) -> None:
            pass

        def stop(self) -> None:
            self._busy = False

        def is_busy(self) -> bool:
            return self._busy

    class ConversationManager:
        def __init__(self):
            self._transcript: List[str] = []

        def add_word(self, word: str) -> None:
            self._transcript.append(word)

        def get_transcript(self) -> List[str]:
            return list(self._transcript)

        def get_transcript_text(self) -> str:
            return " ".join(self._transcript)

        def get_latest(self) -> Optional[str]:
            return self._transcript[-1] if self._transcript else None

        def get_words(self) -> List[str]:
            return list(self._transcript)

        def get_last_word(self) -> str:
            return self._transcript[-1] if self._transcript else ""

        def clear(self) -> None:
            self._transcript.clear()


class MainWindow(QMainWindow):
    """
    Main application window orchestrating the CameraPanel, ConversationPanel,
    and LibraryPanel with live capture and recognition timers.

    Layout Proportions:
    - LEFT: CameraPanel (65% width)
    - RIGHT: Right Splitter (35% width)
        - TOP: ConversationPanel (50% height)
        - BOTTOM: LibraryPanel (50% height)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._latest_frame = None

        # 1. Initialize Backend Subsystems
        self.camera_manager = CameraManager()

        model_file = globals().get('MODEL_PATH', DEFAULT_MODEL_PATH)
        labels_file = globals().get('CLASS_LABELS_PATH', DEFAULT_LABELS_PATH)
        try:
            self.predictor = SignPredictor(model_path=model_file, labels_path=labels_file)
        except TypeError:
            try:
                self.predictor = SignPredictor(model_path=model_file)
            except Exception:
                self.predictor = SignPredictor()
        except Exception:
            self.predictor = None

        self.stabilizer = PredictionStabilizer(
            stabilization_window=globals().get('STABILIZATION_WINDOW', 5),
            min_agreement=globals().get('STABILIZATION_AGREEMENT', 3),
            cooldown_frames=globals().get('COOLDOWN_FRAMES', 4),
            confidence_threshold=globals().get('CONFIDENCE_THRESHOLD', 0.40),
        )
        self.tts = TextToSpeech()
        self.conversation_manager = ConversationManager()

        # 2. Window Setup
        self.setWindowTitle("SignBridge — Sign Language Recognition Assistant")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 3. Setup Layout and Panels
        self._setup_ui()

        # 4. Connect Signals
        self._connect_signals()

        # 5. Setup Timers
        self._setup_timers()

        # 6. Populate Library and Check System Status
        self._load_categories()
        self._check_model_status()

    def _setup_ui(self) -> None:
        """Create and configure the main split layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        # Main Horizontal Splitter (Left: 65%, Right: 35%)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {BORDER_COLOR};
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {ACCENT};
            }}
        """)

        # LEFT: CameraPanel (occupies 65% width)
        self.camera_panel = CameraPanel(self)
        self.main_splitter.addWidget(self.camera_panel)

        # RIGHT: Vertical Splitter (Conversation 50%, Library 50%)
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setHandleWidth(4)
        self.right_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {BORDER_COLOR};
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {ACCENT};
            }}
        """)

        self.conversation_panel = ConversationPanel(self)
        self.library_panel = LibraryPanel(self)

        self.right_splitter.addWidget(self.conversation_panel)
        self.right_splitter.addWidget(self.library_panel)

        # Right column split equally (50% conversation, 50% library)
        self.right_splitter.setStretchFactor(0, 50)
        self.right_splitter.setStretchFactor(1, 50)
        self.right_splitter.setSizes([380, 380])

        self.main_splitter.addWidget(self.right_splitter)

        # Left 65%, Right 35%
        self.main_splitter.setStretchFactor(0, 65)
        self.main_splitter.setStretchFactor(1, 35)
        # Default size is 1400: 65% is 910, 35% is 490
        self.main_splitter.setSizes([910, 490])

        root_layout.addWidget(self.main_splitter)

    def _connect_signals(self) -> None:
        """Connect panel signals to handlers."""
        # Camera controls
        self.camera_panel.start_requested.connect(self.start_camera)
        self.camera_panel.stop_requested.connect(self.stop_camera)

        # Conversation actions
        self.conversation_panel.play_all_requested.connect(self.play_all_transcript)
        self.conversation_panel.play_last_requested.connect(self.play_last_word)
        self.conversation_panel.copy_requested.connect(self.copy_transcript)
        self.conversation_panel.clear_requested.connect(self.clear_transcript)

        # Library navigation
        self.library_panel.category_selected.connect(self.on_category_selected)

    def _setup_timers(self) -> None:
        """Initialize frame capture timer (30fps) and inference timer (200ms)."""
        # Frame capture timer (~30 FPS -> 33ms interval)
        fps = globals().get('CAMERA_FPS', 30)
        interval_ms = int(1000 / fps) if fps > 0 else 33
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(interval_ms)
        self.capture_timer.timeout.connect(self._on_capture_frame)

        # Inference timer (every 200ms)
        infer_interval = globals().get('INFERENCE_INTERVAL_MS', 200)
        self.inference_timer = QTimer(self)
        self.inference_timer.setInterval(infer_interval)
        self.inference_timer.timeout.connect(self._on_run_inference)

    def _load_categories(self) -> None:
        """Load category names from predictor labels, dataset directory, or fallbacks."""
        dataset_dir = globals().get('DATASET_PATH', DEFAULT_DATASET_PATH)
        self.library_panel.set_dataset_path(dataset_dir)

        categories: List[str] = []

        # 1. Check predictor labels
        if self.predictor is not None:
            if hasattr(self.predictor, 'labels') and self.predictor.labels:
                categories = list(self.predictor.labels)
            elif hasattr(self.predictor, 'class_names') and self.predictor.class_names:
                categories = list(self.predictor.class_names)
            elif hasattr(self.predictor, 'classes') and self.predictor.classes:
                categories = list(self.predictor.classes)

        # 2. If not available from predictor, scan dataset directory
        if not categories and os.path.exists(dataset_dir):
            try:
                entries = sorted([
                    d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
                ])
                if entries:
                    categories = entries
            except Exception:
                pass

        # 3. Fallback to default 44 categories
        if not categories:
            categories = list(FALLBACK_PHRASE_CATEGORIES)

        self.library_panel.set_categories(categories)

    def _check_model_status(self) -> None:
        """
        Check if model is loaded and ready. If not trained/found, display guidance banner:
        'Model not found. Please train the model first: python scripts/train_model.py'
        """
        model_file = globals().get('MODEL_PATH', DEFAULT_MODEL_PATH)
        model_loaded = False
        if self.predictor is not None:
            if hasattr(self.predictor, 'is_loaded') and callable(self.predictor.is_loaded):
                model_loaded = self.predictor.is_loaded()
            elif hasattr(self.predictor, '_is_loaded'):
                model_loaded = bool(self.predictor._is_loaded)

        if not model_loaded or not os.path.exists(model_file):
            self.camera_panel.show_error(
                "Model not found. Please train the model first: python scripts/train_model.py"
            )
        else:
            self.camera_panel.clear_error()

    # --- Camera and Recognition Control ---

    def start_camera(self) -> None:
        """Start camera video stream and inference loops."""
        self.camera_panel.clear_error()
        self._check_model_status()

        success = False
        try:
            if hasattr(self.camera_manager, 'start'):
                success = bool(self.camera_manager.start())
            elif hasattr(self.camera_manager, 'open'):
                success = bool(self.camera_manager.open())
            else:
                success = True
        except Exception as e:
            self.camera_panel.show_error(f"Camera initialization failed: {str(e)}")
            success = False

        if success:
            self.camera_panel.set_camera_active(True)
            self.capture_timer.start()
            self.inference_timer.start()
        else:
            self.camera_panel.show_error("Could not access camera. Please check webcam permissions.")
            self.camera_panel.set_camera_active(False)

    def stop_camera(self) -> None:
        """Stop video stream, timers, and inference loops."""
        self.capture_timer.stop()
        self.inference_timer.stop()

        try:
            if hasattr(self.camera_manager, 'stop'):
                self.camera_manager.stop()
            elif hasattr(self.camera_manager, 'release'):
                self.camera_manager.release()
        except Exception:
            pass

        self._latest_frame = None
        self.camera_panel.set_camera_active(False)
        self.camera_panel.set_frame(None)
        if hasattr(self.stabilizer, 'reset'):
            self.stabilizer.reset()

    def _on_capture_frame(self) -> None:
        """Timer callback (30fps) to capture a video frame and send to CameraPanel."""
        frame = None
        try:
            res = None
            if hasattr(self.camera_manager, 'read_frame'):
                res = self.camera_manager.read_frame()
            elif hasattr(self.camera_manager, 'get_frame'):
                res = self.camera_manager.get_frame()
            elif hasattr(self.camera_manager, 'read'):
                res = self.camera_manager.read()

            if isinstance(res, tuple):
                ret, f = res
                frame = f if ret else None
            else:
                frame = res
        except Exception as e:
            self.camera_panel.show_error(f"Frame capture error: {str(e)}")
            return

        if frame is not None:
            # Un-mirror macOS selfie-mode webcam (horizontal flip).
            # Training images are NOT mirrored, so model input must match.
            frame = cv2.flip(frame, 1)
            self._latest_frame = frame
            self.camera_panel.set_frame(frame)

    def _on_run_inference(self) -> None:
        """
        Timer callback (every 200ms) to run sign recognition on the latest frame,
        stabilize predictions, update conversation transcript, and speak words.
        """
        if self._latest_frame is None or self.predictor is None:
            return

        is_loaded = False
        if self.predictor is not None:
            if hasattr(self.predictor, 'is_loaded'):
                prop = getattr(self.predictor, 'is_loaded')
                is_loaded = prop() if callable(prop) else bool(prop)
            elif hasattr(self.predictor, '_is_loaded'):
                is_loaded = bool(self.predictor._is_loaded)
            else:
                is_loaded = True
        if not is_loaded:
            return

        pred_class = None
        confidence = 0.0

        try:
            result = self.predictor.predict(self._latest_frame)
            if isinstance(result, tuple):
                pred_class, confidence = result
            elif isinstance(result, dict):
                pred_class = result.get('label') or result.get('class')
                confidence = float(result.get('confidence', 0.0))
            elif isinstance(result, str):
                pred_class = result
                confidence = 1.0
        except Exception:
            return

        # Update confidence bar on camera panel
        self.camera_panel.set_confidence(confidence)

        # --- Tentative display: always show what the model sees ---
        # This makes the app feel responsive even before acceptance.
        threshold = globals().get('CONFIDENCE_THRESHOLD', 0.40)
        display_floor = 0.15  # Below this, prediction is noise — show waiting

        if pred_class and confidence >= display_floor:
            conf_pct = int(round(confidence * 100))
            if confidence >= threshold:
                self.camera_panel.set_status(
                    f"Detecting: {pred_class} ({conf_pct}%)"
                )
            else:
                # Uncertain zone: show tentative with top alternatives
                alt_text = f"Detecting: {pred_class} ({conf_pct}%)"
                try:
                    if hasattr(self.predictor, 'predict_top_k'):
                        top3 = self.predictor.predict_top_k(self._latest_frame, k=3)
                        alts = " | ".join(
                            f"{lbl} {int(round(p*100))}%"
                            for lbl, p in top3
                        )
                        alt_text = f"Maybe: {alts}"
                except Exception:
                    pass
                self.camera_panel.set_status(alt_text)
        else:
            self.camera_panel.set_status("Waiting for sign...")

        # --- Feed ALL predictions to stabilizer (it has its own threshold) ---
        confirmed_word = None
        if pred_class:
            if hasattr(self.stabilizer, 'update'):
                confirmed_word = self.stabilizer.update(pred_class, confidence)
            elif hasattr(self.stabilizer, 'process'):
                confirmed_word = self.stabilizer.process(pred_class, confidence)
        else:
            # Feed None to advance cooldown when no sign is visible
            if hasattr(self.stabilizer, 'update'):
                self.stabilizer.update(None, 0.0)

        # --- Accepted recognition: add to transcript and speak ---
        if confirmed_word:
            self.conversation_manager.add_word(confirmed_word)
            self.conversation_panel.add_word(confirmed_word)

            if hasattr(self.conversation_panel, 'set_dictation_status'):
                self.conversation_panel.set_dictation_status(speaking=True)
                QTimer.singleShot(1500, lambda: self.conversation_panel.set_dictation_status(active=True))

            if hasattr(self.tts, 'speak'):
                self.tts.speak(confirmed_word)

            self.camera_panel.set_recognized_word(confirmed_word, confidence)
            self.camera_panel.set_status("Recognized", word=confirmed_word)

    # --- Conversation Panel Signal Handlers ---

    def play_all_transcript(self) -> None:
        """Speak the entire transcript via TTS."""
        text = ""
        if hasattr(self.conversation_manager, 'get_transcript_text'):
            text = self.conversation_manager.get_transcript_text()
        elif hasattr(self.conversation_manager, 'get_transcript'):
            res = self.conversation_manager.get_transcript()
            text = " ".join(res) if isinstance(res, list) else str(res)

        if not text:
            text = self.conversation_panel.get_transcript()

        if text and hasattr(self.tts, 'speak'):
            if hasattr(self.conversation_panel, 'set_dictation_status'):
                self.conversation_panel.set_dictation_status(speaking=True)
                QTimer.singleShot(2500, lambda: self.conversation_panel.set_dictation_status(active=True))
            self.tts.speak(text)

    def play_last_word(self) -> None:
        """Speak only the latest recognized word via TTS."""
        word = ""
        if hasattr(self.conversation_manager, 'get_latest'):
            word = self.conversation_manager.get_latest() or ""
        elif hasattr(self.conversation_manager, 'get_last_word'):
            word = self.conversation_manager.get_last_word() or ""

        if not word:
            word = self.conversation_panel.get_last_word()

        if word and hasattr(self.tts, 'speak'):
            if hasattr(self.conversation_panel, 'set_dictation_status'):
                self.conversation_panel.set_dictation_status(speaking=True)
                QTimer.singleShot(1500, lambda: self.conversation_panel.set_dictation_status(active=True))
            self.tts.speak(word)

    def copy_transcript(self) -> None:
        """Copy the current transcript text to the system clipboard."""
        text = ""
        if hasattr(self.conversation_manager, 'get_transcript_text'):
            text = self.conversation_manager.get_transcript_text()
        if not text:
            text = self.conversation_panel.get_transcript()
        if text:
            from app.ui.qt_compat import QGuiApplication
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)

    def clear_transcript(self) -> None:
        """Reset conversation transcript and stop ongoing speech."""
        self.conversation_manager.clear()
        # Directly clear panel words without re-emitting clear_requested signal
        self.conversation_panel._words.clear()
        self.conversation_panel._update_display()
        if hasattr(self.tts, 'stop'):
            self.tts.stop()
        if hasattr(self.stabilizer, 'reset'):
            self.stabilizer.reset()
        self.camera_panel.clear_recognized_word()

    # --- Library Panel Signal Handlers ---

    def on_category_selected(self, category: str) -> None:
        """Handle user clicking a category in the Sign Library."""
        if category and hasattr(self.tts, 'speak'):
            self.tts.speak(category)

    # --- Window Lifecycle ---

    def closeEvent(self, event) -> None:
        """Cleanly release camera, stop timers, and shut down speech on window close."""
        self.stop_camera()
        if hasattr(self.tts, 'stop'):
            try:
                self.tts.stop()
            except Exception:
                pass
        event.accept()


def main():
    """Application entrypoint."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
