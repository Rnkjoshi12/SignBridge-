"""Camera panel for displaying the live video feed, recognition overlay, and status."""

from typing import Any, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from app.ui.qt_compat import (
    Qt,
    Signal,
    QRect,
    QSize,
    QTimer,
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QFont,
    QPaintEvent,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QFrame,
)

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


class VideoFeedLabel(QLabel):
    """
    Video display label that expands to fill available space while maintaining
    a strict 4:3 aspect ratio and rendering a 1px border.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)

        self._current_pixmap: Optional[QPixmap] = None
        self._is_camera_active: bool = False

        # Recognition overlay widget embedded at bottom of video area
        self._overlay_container = QWidget(self)
        self._overlay_container.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(22, 33, 62, 0.88);
                border: 2px solid {ACCENT};
                border-radius: 10px;
            }}
        """)
        overlay_layout = QVBoxLayout(self._overlay_container)
        overlay_layout.setContentsMargins(20, 8, 20, 8)

        self._overlay_text_label = QLabel("", self._overlay_container)
        self._overlay_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_text_label.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT};
                font-size: 26px;
                font-weight: 800;
                background: transparent;
                border: none;
                letter-spacing: 1px;
            }}
        """)
        overlay_layout.addWidget(self._overlay_text_label)
        self._overlay_container.setVisible(False)

        # Timer to auto-hide recognized overlay if needed
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.timeout.connect(self.hide_overlay)

    def set_camera_active(self, active: bool) -> None:
        """Update active status of the camera."""
        self._is_camera_active = active
        if not active:
            self._current_pixmap = None
            self.hide_overlay()
        self.update()

    def set_frame_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        """Set the frame pixmap to display."""
        self._current_pixmap = pixmap
        self.update()

    def show_recognized_word(self, word: str) -> None:
        """Display recognized word in the semi-transparent bottom overlay."""
        if not word:
            self.hide_overlay()
            return
        self._overlay_text_label.setText(word.upper())
        self._position_overlay()
        self._overlay_container.setVisible(True)
        self._overlay_container.raise_()
        # Keep overlay visible for 3 seconds before auto-clearing
        self._overlay_timer.start(3000)

    def hide_overlay(self) -> None:
        """Hide the recognized word overlay."""
        self._overlay_container.setVisible(False)
        self._overlay_text_label.setText("")

    def _get_video_rect(self) -> QRect:
        """Calculate the largest 4:3 rectangle centered within the widget."""
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return QRect(0, 0, 0, 0)

        aspect = 4.0 / 3.0
        # Leave a small margin (4px)
        margin = 4
        avail_w = max(1, w - (margin * 2))
        avail_h = max(1, h - (margin * 2))

        if avail_w / avail_h > aspect:
            target_h = avail_h
            target_w = int(avail_h * aspect)
        else:
            target_w = avail_w
            target_h = int(avail_w / aspect)

        x = (w - target_w) // 2
        y = (h - target_h) // 2
        return QRect(x, y, target_w, target_h)

    def _position_overlay(self) -> None:
        """Position the overlay near the bottom-center of the 4:3 video area."""
        rect = self._get_video_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        self._overlay_container.adjustSize()
        ow = self._overlay_container.width()
        oh = self._overlay_container.height()

        # Place 24px above bottom of the video rect
        ox = rect.x() + (rect.width() - ow) // 2
        oy = rect.y() + rect.height() - oh - 24
        self._overlay_container.move(max(ox, rect.x() + 10), max(oy, rect.y() + 10))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_overlay()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self._get_video_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        if self._current_pixmap and not self._current_pixmap.isNull():
            # Scale pixmap smoothly to fit 4:3 target rect
            scaled = self._current_pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            draw_x = rect.x() + (rect.width() - scaled.width()) // 2
            draw_y = rect.y() + (rect.height() - scaled.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled)

            # 1px border around video feed in BORDER_COLOR
            pen = QPen(QColor(BORDER_COLOR), 1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 8, 8)
        else:
            # Placeholder when camera is inactive or stream is starting
            painter.setPen(QPen(QColor(BORDER_COLOR), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(BG_CARD))
            painter.drawRoundedRect(rect, 8, 8)

            # Central placeholder icon and text
            painter.setPen(QPen(QColor(TEXT_SECONDARY)))
            font = painter.font()
            font.setPointSize(16)
            font.setBold(True)
            painter.setFont(font)

            status_header = "Camera Inactive" if not self._is_camera_active else "Starting Video Stream..."
            painter.drawText(
                rect.adjusted(0, -20, 0, -20),
                Qt.AlignmentFlag.AlignCenter,
                status_header,
            )

            font.setPointSize(12)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QPen(QColor(TEXT_MUTED)))

            status_sub = (
                "Click 'Start Camera' below to begin recognizing signs"
                if not self._is_camera_active
                else "Waiting for video input from device..."
            )
            painter.drawText(
                rect.adjusted(0, 30, 0, 30),
                Qt.AlignmentFlag.AlignCenter,
                status_sub,
            )


class CameraPanel(QWidget):
    """
    CameraPanel occupies 65% of the window width.

    Features:
    - Live video feed maintaining 4:3 aspect ratio
    - Start/Stop camera toggle button
    - Green/Red camera activity indicator dot
    - Error message display area for camera/model failures
    - Recognized word overlay at bottom of video
    - Status bar: 'Waiting for sign...', 'Detecting...', 'Recognized: WORD', 'Sign not recognized'
    - Thin horizontal confidence bar
    """

    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_camera_running: bool = False
        self._init_ui()

    def _init_ui(self) -> None:
        # Panel container styling: dark panel with rounded corners (12px)
        self.setStyleSheet(f"""
            CameraPanel {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # --- Top Header Bar ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)

        # Title
        title_label = QLabel("Live Recognition")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 17px;
                font-weight: 700;
                background: transparent;
            }}
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Camera Indicator Dot (Green when active, Red when stopped)
        self.indicator_dot = QLabel()
        self.indicator_dot.setFixedSize(10, 10)
        self._set_indicator_active(False)
        header_layout.addWidget(self.indicator_dot)

        # Camera state text
        self.camera_state_label = QLabel("Camera Off")
        self.camera_state_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_MUTED};
                font-size: 13px;
                font-weight: 500;
                background: transparent;
            }}
        """)
        header_layout.addWidget(self.camera_state_label)

        header_layout.addSpacing(6)

        # Start / Stop Toggle Button
        self.toggle_camera_btn = QPushButton("Start Camera")
        self.toggle_camera_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_camera_btn.clicked.connect(self._on_toggle_clicked)
        self._update_button_style(active=False)
        header_layout.addWidget(self.toggle_camera_btn)

        main_layout.addLayout(header_layout)

        # --- Error Banner Area (Initially Hidden) ---
        self.error_frame = QFrame()
        self.error_frame.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(248, 113, 113, 0.12);
                border: 1px solid {ERROR};
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        error_layout = QHBoxLayout(self.error_frame)
        error_layout.setContentsMargins(12, 8, 12, 8)
        error_layout.setSpacing(8)

        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"""
            QLabel {{
                color: {ERROR};
                font-size: 13px;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
        """)
        self.error_label.setWordWrap(True)
        error_layout.addWidget(self.error_label, 1)

        self.error_frame.setVisible(False)
        main_layout.addWidget(self.error_frame)

        # --- Video Feed Display (Expands, maintains 4:3 aspect ratio) ---
        self.video_label = VideoFeedLabel(self)
        main_layout.addWidget(self.video_label, 1)

        # --- Status & Recognition Footer Bar ---
        footer_layout = QVBoxLayout()
        footer_layout.setSpacing(6)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)

        # Status text: 'Waiting for sign...', 'Detecting...', etc.
        self.status_label = QLabel("Waiting for sign...")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 14px;
                font-weight: 500;
                background: transparent;
            }}
        """)
        status_row.addWidget(self.status_label, 1)

        # Confidence percentage text
        self.confidence_text = QLabel("Confidence: 0%")
        self.confidence_text.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_MUTED};
                font-size: 12px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        status_row.addWidget(self.confidence_text)

        footer_layout.addLayout(status_row)

        # Thin Horizontal Confidence Progress Bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(5)
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_CARD};
                border: none;
                border-radius: 2px;
                height: 5px;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT};
                border-radius: 2px;
            }}
        """)
        footer_layout.addWidget(self.confidence_bar)

        main_layout.addLayout(footer_layout)

    def _set_indicator_active(self, active: bool) -> None:
        """Update green/red dot indicator."""
        color = SUCCESS if active else ERROR
        self.indicator_dot.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                border-radius: 5px;
                min-width: 10px;
                max-width: 10px;
                min-height: 10px;
                max-height: 10px;
                border: none;
            }}
        """)

    def _update_button_style(self, active: bool) -> None:
        """Update start/stop button styling."""
        if active:
            self.toggle_camera_btn.setText("Stop Camera")
            self.toggle_camera_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_CARD};
                    color: {ERROR};
                    font-size: 13px;
                    font-weight: 700;
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                    padding: 6px 16px;
                    min-height: 32px;
                }}
                QPushButton:hover {{
                    background-color: {ERROR};
                    color: #1a1a2e;
                    border-color: {ERROR};
                }}
                QPushButton:pressed {{
                    background-color: #dc2626;
                    color: #ffffff;
                }}
            """)
        else:
            self.toggle_camera_btn.setText("Start Camera")
            self.toggle_camera_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT};
                    color: #1a1a2e;
                    font-size: 13px;
                    font-weight: 700;
                    border: none;
                    border-radius: 8px;
                    padding: 6px 16px;
                    min-height: 32px;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT_HOVER};
                }}
                QPushButton:pressed {{
                    background-color: {ACCENT_DIM};
                }}
            """)

    def _on_toggle_clicked(self) -> None:
        """Handle button click and emit appropriate signal."""
        if self._is_camera_running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    # --- Public Control API ---

    def set_camera_active(self, active: bool) -> None:
        """Update UI state when camera starts or stops."""
        self._is_camera_running = active
        self._set_indicator_active(active)
        self._update_button_style(active)
        self.video_label.set_camera_active(active)

        if active:
            self.camera_state_label.setText("Camera Active")
            self.camera_state_label.setStyleSheet(f"color: {SUCCESS}; font-size: 13px; font-weight: 600; background: transparent;")
            self.set_status("Waiting for sign...")
        else:
            self.camera_state_label.setText("Camera Off")
            self.camera_state_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px; font-weight: 500; background: transparent;")
            self.set_status("Camera stopped")
            self.set_confidence(0.0)

    def set_frame(self, frame: Union[Any, QImage, QPixmap, None]) -> None:
        """
        Accept and render a frame on the video feed.
        Supports numpy ndarray (RGB/BGR), QImage, QPixmap, or None.
        """
        if frame is None:
            self.video_label.set_frame_pixmap(None)
            return

        if isinstance(frame, QPixmap):
            self.video_label.set_frame_pixmap(frame)
            return

        if isinstance(frame, QImage):
            pixmap = QPixmap.fromImage(frame)
            self.video_label.set_frame_pixmap(pixmap)
            return

        # Handle numpy ndarray
        if hasattr(frame, 'shape'):
            h, w = frame.shape[:2]
            if len(frame.shape) == 2:
                # Grayscale
                bytes_per_line = w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()
            elif frame.shape[2] == 3:
                bytes_per_line = 3 * w
                # PySide6 supports Format_BGR888 for standard OpenCV frames
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
            elif frame.shape[2] == 4:
                bytes_per_line = 4 * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32).copy()
            else:
                return

            pixmap = QPixmap.fromImage(q_img)
            self.video_label.set_frame_pixmap(pixmap)

    def set_status(self, state: str, word: str = "") -> None:
        """
        Update the recognition status bar.
        Recognized word displayed in ACCENT, status text in TEXT_SECONDARY.
        """
        state_str = str(state)
        if word or "Recognized" in state_str:
            display_word = word if word else state_str.split(":", 1)[-1].strip()
            self.status_label.setText(
                f'Status: Recognized: <span style="color: {ACCENT}; font-weight: 700;">{display_word}</span>'
            )
        elif "Detecting" in state_str:
            self.status_label.setText(
                f'Status: <span style="color: {TEXT_PRIMARY}; font-weight: 600;">{state_str}</span>'
            )
        elif "not recognized" in state_str.lower():
            self.status_label.setText(
                f'Status: <span style="color: {WARNING}; font-weight: 600;">Sign not recognized</span>'
            )
        elif "waiting" in state_str.lower():
            self.status_label.setText(
                f'Status: <span style="color: {TEXT_SECONDARY};">Waiting for sign...</span>'
            )
        else:
            self.status_label.setText(
                f'Status: <span style="color: {TEXT_SECONDARY};">{state_str}</span>'
            )

    def set_confidence(self, value: float) -> None:
        """
        Update thin horizontal progress bar and percentage label.
        Accepts float between 0.0 and 1.0, or 0 and 100.
        """
        if 0.0 <= value <= 1.0:
            val_int = int(round(value * 100))
        else:
            val_int = int(round(min(max(value, 0.0), 100.0)))

        self.confidence_bar.setValue(val_int)
        self.confidence_text.setText(f"Confidence: {val_int}%")

    def set_recognized_word(self, word: str, confidence: Optional[float] = None) -> None:
        """Show recognized word in video overlay and update status."""
        self.video_label.show_recognized_word(word)
        self.set_status("Recognized", word=word)
        if confidence is not None:
            self.set_confidence(confidence)

    def clear_recognized_word(self) -> None:
        """Clear recognized word overlay and reset status."""
        self.video_label.hide_overlay()
        self.set_status("Waiting for sign...")

    def show_error(self, message: str) -> None:
        """Display an error message banner in the camera panel."""
        self.error_label.setText(f"⚠ {message}")
        self.error_frame.setVisible(True)

    def clear_error(self) -> None:
        """Hide the error banner."""
        self.error_label.setText("")
        self.error_frame.setVisible(False)
