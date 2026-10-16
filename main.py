#!/usr/bin/env python3
"""
SignBridge — Real-Time Sign Language Recognition and Communication Assistant

Main entry point for the SignBridge desktop application.
Launches the PySide6 GUI with webcam-based sign recognition,
live conversation transcript, text-to-speech, and a searchable sign library.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    """Launch the SignBridge application."""
    # Import PySide6 here so we get a clear error if it's not installed
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
    except ImportError:
        print("=" * 60)
        print("ERROR: PySide6 is not installed.")
        print()
        print("Please install dependencies first:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(1)

    from app.ui.main_window import MainWindow

    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("SignBridge")
    app.setOrganizationName("SignBridge")

    # Set default font — use system font to avoid alias lookup warning
    import platform
    if platform.system() == "Darwin":
        font = QFont(".AppleSystemUIFont", 11)
    else:
        font = QFont("Segoe UI", 11)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Apply global dark stylesheet
    app.setStyleSheet("""
        QWidget {
            background-color: #1a1a2e;
            color: #e8e8e8;
            font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
        }
        QScrollBar:vertical {
            background: #16213e;
            width: 8px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #2a3550;
            border-radius: 4px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #00d4aa;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #16213e;
            height: 8px;
            border-radius: 4px;
        }
        QScrollBar::handle:horizontal {
            background: #2a3550;
            border-radius: 4px;
            min-width: 30px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #00d4aa;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        QToolTip {
            background: #1f2b47;
            color: #e8e8e8;
            border: 1px solid #2a3550;
            padding: 4px 8px;
            border-radius: 4px;
        }
    """)

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Run the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
