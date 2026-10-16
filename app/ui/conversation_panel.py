"""Conversation panel for tracking, displaying, and replaying recognized sign transcript."""

from typing import List, Optional

from app.ui.qt_compat import (
    Qt,
    Signal,
    QTimer,
    QTextCursor,
    QGuiApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QSizePolicy,
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


class ConversationPanel(QWidget):
    """
    ConversationPanel occupies the top-right section (17.5% width, 50% height).

    Features:
    - Title 'Conversation' in TEXT_PRIMARY
    - Read-only scrollable QTextEdit displaying space-separated words
    - Auto-scroll to bottom on new content
    - Centered italic empty state: 'Perform a sign to begin...' in TEXT_MUTED
    - Compact action button row (height 28px):
        - ▶ Play All
        - ▶ Play Last
        - Copy
        - Clear
    - Action buttons disabled when transcript is empty
    - Signals: play_all_requested, play_last_requested, copy_requested, clear_requested
    """

    play_all_requested = Signal()
    play_last_requested = Signal()
    copy_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._words: List[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        # Panel styling: BG_SECONDARY with rounded corners (12px), 1px BORDER_COLOR, subtle padding
        self.setStyleSheet(f"""
            ConversationPanel {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Header ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # Title: 'Conversation' in TEXT_PRIMARY, medium font
        self.title_label = QLabel("Conversation")
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 15px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Dictation status badge
        self.dictation_badge = QLabel("🔊 Dictation Active")
        self.dictation_badge.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT};
                font-size: 11px;
                font-weight: 600;
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 2px 7px;
            }}
        """)
        header_layout.addWidget(self.dictation_badge)

        # Word count badge
        self.count_badge = QLabel("0 words")
        self.count_badge.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_MUTED};
                font-size: 12px;
                font-weight: 500;
                background: transparent;
            }}
        """)
        header_layout.addWidget(self.count_badge)

        layout.addLayout(header_layout)

        # --- Transcript Text Edit ---
        # Scrollable QTextEdit (read-only) with BG_CARD background and rounded 8px
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setAcceptRichText(True)
        self.transcript_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.transcript_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-family: inherit;
            }}
        """)
        layout.addWidget(self.transcript_edit, 1)

        # --- Compact Action Button Row (28px height) ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        btn_style = f"""
            QPushButton {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                height: 28px;
                min-height: 28px;
                max-height: 28px;
                padding: 0 10px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {ACCENT};
                color: {ACCENT};
                background-color: {BG_SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {ACCENT_DIM};
                color: {BG_PRIMARY};
            }}
            QPushButton:disabled {{
                background-color: {BG_CARD};
                color: {TEXT_MUTED};
                border-color: {BORDER_COLOR};
            }}
        """

        # ▶ Play All
        self.play_all_btn = QPushButton("▶ Play All")
        self.play_all_btn.setStyleSheet(btn_style)
        self.play_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_all_btn.setToolTip("Replay full conversation transcript")
        self.play_all_btn.clicked.connect(self._on_play_all)
        btn_layout.addWidget(self.play_all_btn)

        # ▶ Play Last
        self.play_last_btn = QPushButton("▶ Play Last")
        self.play_last_btn.setStyleSheet(btn_style)
        self.play_last_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_last_btn.setToolTip("Replay the latest recognized word")
        self.play_last_btn.clicked.connect(self._on_play_last)
        btn_layout.addWidget(self.play_last_btn)

        # Copy
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setStyleSheet(btn_style)
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setToolTip("Copy transcript to clipboard")
        self.copy_btn.clicked.connect(self._on_copy)
        btn_layout.addWidget(self.copy_btn)

        # Clear
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(btn_style)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setToolTip("Clear conversation transcript")
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)

        layout.addLayout(btn_layout)

        # Initialize empty state
        self._update_display()

    def _update_display(self) -> None:
        """Render transcript or empty state, and update button states."""
        has_content = len(self._words) > 0

        # Enable/disable buttons based on whether transcript has words
        self.play_all_btn.setEnabled(has_content)
        self.play_last_btn.setEnabled(has_content)
        self.copy_btn.setEnabled(has_content)
        self.clear_btn.setEnabled(has_content)

        # Update word count badge
        count = len(self._words)
        self.count_badge.setText(f"{count} {'word' if count == 1 else 'words'}")

        if not has_content:
            # Empty state: 'Perform a sign to begin...' in TEXT_MUTED, italic, centered
            self.transcript_edit.setHtml(f"""
                <div align="center" style="
                    color: {TEXT_MUTED};
                    font-style: italic;
                    margin-top: 45px;
                    font-size: 13px;
                ">
                    Perform a sign to begin...
                </div>
            """)
        else:
            # Each word on the same line, space-separated
            transcript_text = " ".join(self._words)
            self.transcript_edit.setHtml(f"""
                <div style="
                    color: {TEXT_PRIMARY};
                    font-size: 14px;
                    font-weight: 500;
                    line-height: 1.6;
                ">
                    {transcript_text}
                </div>
            """)
            # Auto-scrolls to bottom on new content
            cursor = self.transcript_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.transcript_edit.setTextCursor(cursor)
            self.transcript_edit.ensureCursorVisible()

    # --- Button Signal Handlers ---

    def _on_play_all(self) -> None:
        if self._words:
            self.play_all_requested.emit()

    def _on_play_last(self) -> None:
        if self._words:
            self.play_last_requested.emit()

    def _on_copy(self) -> None:
        if not self._words:
            return
        transcript = self.get_transcript()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(transcript)
        self.copy_requested.emit()

        # Temporary visual feedback on button
        orig_text = self.copy_btn.text()
        self.copy_btn.setText("Copied!")
        QTimer.singleShot(1200, lambda: self.copy_btn.setText(orig_text))

    def _on_clear(self) -> None:
        self._words.clear()
        self._update_display()
        self.clear_requested.emit()

    # --- Public Control API ---

    def add_word(self, word: str) -> None:
        """
        Add a newly recognized word to the transcript.
        Auto-scrolls to bottom and updates buttons.
        """
        cleaned = word.strip()
        if not cleaned:
            return
        self._words.append(cleaned)
        self._update_display()

    def set_words(self, words: List[str]) -> None:
        """Replace all words in the transcript with a new list."""
        self._words = [w.strip() for w in words if w.strip()]
        self._update_display()

    def get_transcript(self) -> str:
        """Return the complete transcript as a single space-separated string."""
        return " ".join(self._words)

    def get_words(self) -> List[str]:
        """Return a copy of the list of recognized words."""
        return list(self._words)

    def get_last_word(self) -> str:
        """Return the most recently recognized word, or an empty string."""
        return self._words[-1] if self._words else ""

    def clear_transcript(self) -> None:
        """Clear all words, restore empty state, and emit clear_requested."""
        self._words.clear()
        self._update_display()
        self.clear_requested.emit()

    def set_dictation_status(self, active: bool = True, speaking: bool = False) -> None:
        """Update dictation indicator in panel header."""
        if speaking:
            self.dictation_badge.setText("🔊 Dictating...")
            self.dictation_badge.setStyleSheet(f"""
                QLabel {{
                    color: {SUCCESS};
                    font-size: 11px;
                    font-weight: 600;
                    background-color: {BG_CARD};
                    border: 1px solid {SUCCESS};
                    border-radius: 4px;
                    padding: 2px 7px;
                }}
            """)
        elif active:
            self.dictation_badge.setText("🔊 Dictation Active")
            self.dictation_badge.setStyleSheet(f"""
                QLabel {{
                    color: {ACCENT};
                    font-size: 11px;
                    font-weight: 600;
                    background-color: {BG_CARD};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 4px;
                    padding: 2px 7px;
                }}
            """)
        else:
            self.dictation_badge.setText("🔇 Dictation Off")
            self.dictation_badge.setStyleSheet(f"""
                QLabel {{
                    color: {TEXT_MUTED};
                    font-size: 11px;
                    font-weight: 500;
                    background-color: {BG_CARD};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 4px;
                    padding: 2px 7px;
                }}
            """)

