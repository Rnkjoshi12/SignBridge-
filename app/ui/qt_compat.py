"""Qt compatibility shim for PySide6.

Allows SignBridge UI modules to run with genuine PySide6 when installed,
or with functional mock interfaces for headless test environments.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, Signal, QRect, QSize, QTimer
    from PySide6.QtGui import (
        QColor,
        QFont,
        QGuiApplication,
        QImage,
        QPainter,
        QPaintEvent,
        QPen,
        QPixmap,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE6_AVAILABLE = True

except ImportError:
    logger.info("PySide6 not available in current environment; initializing fallback mock Qt layer.")
    PYSIDE6_AVAILABLE = False

    class _MockQtNamespace:
        class AlignmentFlag:
            AlignCenter = 0x0004
            AlignHCenter = 0x0004
            AlignVCenter = 0x0080
            AlignLeft = 0x0001
            AlignRight = 0x0002
            AlignTop = 0x0020
            AlignBottom = 0x0040

        class AspectRatioMode:
            KeepAspectRatio = 1
            KeepAspectRatioByExpanding = 2
            IgnoreAspectRatio = 0

        class TransformationMode:
            FastTransformation = 0
            SmoothTransformation = 1

        class Orientation:
            Horizontal = 1
            Vertical = 2

        class CursorShape:
            PointingHandCursor = 13
            ArrowCursor = 0

        class PenStyle:
            SolidLine = 1
            DashLine = 2

        class BrushStyle:
            NoBrush = 0
            SolidPattern = 1

    Qt = _MockQtNamespace()

    class _SignalInstance:
        def __init__(self):
            self._callbacks: List[Callable] = []

        def connect(self, callback: Callable) -> None:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

        def disconnect(self, callback: Callable) -> None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        def emit(self, *args, **kwargs) -> None:
            for cb in self._callbacks:
                cb(*args, **kwargs)

    class Signal:
        def __init__(self, *types):
            self._types = types

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            attr = f"_sig_{id(self)}"
            if not hasattr(instance, attr):
                setattr(instance, attr, _SignalInstance())
            return getattr(instance, attr)

    class QRect:
        def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0):
            self._x = x
            self._y = y
            self._w = width
            self._h = height

        def x(self) -> int: return self._x
        def y(self) -> int: return self._y
        def width(self) -> int: return self._w
        def height(self) -> int: return self._h
        def size(self) -> QSize: return QSize(self._w, self._h)

        def adjusted(self, dx1: int, dy1: int, dx2: int, dy2: int) -> QRect:
            return QRect(self._x + dx1, self._y + dy1, self._w + dx2 - dx1, self._h + dy2 - dy1)

    class QSize:
        def __init__(self, width: int = 0, height: int = 0):
            self._w = width
            self._h = height

        def width(self) -> int: return self._w
        def height(self) -> int: return self._h

    class QTimer:
        timeout = Signal()

        def __init__(self, parent: Optional[Any] = None):
            self._interval = 0
            self._single_shot = False
            self._active = False

        def setInterval(self, ms: int) -> None: self._interval = ms
        def setSingleShot(self, single: bool) -> None: self._single_shot = single
        def start(self, ms: Optional[int] = None) -> None: self._active = True
        def stop(self) -> None: self._active = False
        def isActive(self) -> bool: return self._active

        @staticmethod
        def singleShot(ms: int, callback: Callable) -> None:
            pass

    class QColor:
        def __init__(self, *args): pass

    class QFont:
        def __init__(self, *args):
            self._size = 12
            self._bold = False

        def setPointSize(self, s: int) -> None: self._size = s
        def setBold(self, b: bool) -> None: self._bold = b

    class QPen:
        def __init__(self, *args): pass

    class QPainter:
        class RenderHint:
            Antialiasing = 1
            SmoothPixmapTransform = 2

        def __init__(self, *args): pass
        def setRenderHint(self, *args): pass
        def setPen(self, *args): pass
        def setBrush(self, *args): pass
        def setFont(self, *args): pass
        def font(self) -> QFont: return QFont()
        def drawPixmap(self, *args): pass
        def drawRoundedRect(self, *args): pass
        def drawText(self, *args): pass

    class QPaintEvent:
        pass

    class QPixmap:
        def __init__(self, *args):
            self._path = args[0] if args and isinstance(args[0], str) else ""

        def isNull(self) -> bool:
            return False

        def size(self) -> QSize:
            return QSize(640, 480)

        def width(self) -> int:
            return 640

        def height(self) -> int:
            return 480

        def scaled(self, *args, **kwargs) -> QPixmap:
            return self

        @staticmethod
        def fromImage(img: Any) -> QPixmap:
            return QPixmap()

    class QImage:
        class Format:
            Format_Grayscale8 = 1
            Format_BGR888 = 2
            Format_RGB888 = 3
            Format_ARGB32 = 4

        def __init__(self, *args): pass
        def copy(self) -> QImage: return self

    class QTextCursor:
        class MoveOperation:
            End = 1
        def movePosition(self, *args): pass

    class _MockClipboard:
        def __init__(self):
            self._text = ""

        def setText(self, text: str) -> None:
            self._text = text

        def text(self) -> str:
            return self._text

    class QGuiApplication:
        _clipboard = _MockClipboard()

        @classmethod
        def clipboard(cls) -> _MockClipboard:
            return cls._clipboard

    class QWidget:
        def __init__(self, parent: Optional[QWidget] = None):
            self._parent = parent
            self._layout = None
            self._visible = True
            self._style = ""
            self._w = 640
            self._h = 480

        def setLayout(self, layout: Any) -> None: self._layout = layout
        def setStyleSheet(self, style: str) -> None: self._style = style
        def setVisible(self, visible: bool) -> None: self._visible = visible
        def isVisible(self) -> bool: return self._visible
        def setFixedSize(self, w: int, h: int) -> None: self._w, self._h = w, h
        def setMinimumSize(self, w: int, h: int) -> None: pass
        def setSizePolicy(self, *args) -> None: pass
        def width(self) -> int: return self._w
        def height(self) -> int: return self._h
        def rect(self) -> QRect: return QRect(0, 0, self._w, self._h)
        def update(self) -> None: pass
        def adjustSize(self) -> None: pass
        def move(self, x: int, y: int) -> None: pass
        def raise_(self) -> None: pass
        def setCursor(self, *args) -> None: pass
        def setToolTip(self, text: str) -> None: pass
        def resizeEvent(self, event: Any) -> None: pass
        def paintEvent(self, event: Any) -> None: pass

    class QFrame(QWidget):
        pass

    class QMainWindow(QWidget):
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._central_widget = None
            self._title = ""

        def setCentralWidget(self, widget: QWidget) -> None:
            self._central_widget = widget

        def setWindowTitle(self, title: str) -> None:
            self._title = title

        def resize(self, w: int, h: int) -> None:
            self._w, self._h = w, h

        def show(self) -> None:
            self._visible = True

        def closeEvent(self, event: Any) -> None:
            pass

    class QLayout:
        def __init__(self, parent: Optional[QWidget] = None):
            self._parent = parent
            self._items: List[Any] = []

        def setContentsMargins(self, *args) -> None: pass
        def setSpacing(self, s: int) -> None: pass
        def addWidget(self, widget: Any, *args, **kwargs) -> None: self._items.append(widget)
        def addLayout(self, layout: Any, *args, **kwargs) -> None: self._items.append(layout)
        def addStretch(self, *args) -> None: pass
        def addSpacing(self, *args) -> None: pass

    class QVBoxLayout(QLayout):
        pass

    class QHBoxLayout(QLayout):
        pass

    class QLabel(QWidget):
        def __init__(self, text: str = "", parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._text = text

        def setText(self, text: str) -> None: self._text = text
        def text(self) -> str: return self._text
        def setAlignment(self, *args) -> None: pass
        def setWordWrap(self, wrap: bool) -> None: pass

    class QPushButton(QWidget):
        clicked = Signal()

        def __init__(self, text: str = "", parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._text = text
            self._enabled = True

        def setText(self, text: str) -> None: self._text = text
        def text(self) -> str: return self._text
        def setEnabled(self, enabled: bool) -> None: self._enabled = enabled
        def isEnabled(self) -> bool: return self._enabled

    class QProgressBar(QWidget):
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._min = 0
            self._max = 100
            self._val = 0

        def setRange(self, minimum: int, maximum: int) -> None:
            self._min, self._max = minimum, maximum

        def setValue(self, val: int) -> None: self._val = val
        def value(self) -> int: return self._val
        def setTextVisible(self, visible: bool) -> None: pass
        def setFixedHeight(self, h: int) -> None: self._h = h

    class QSizePolicy:
        class Policy:
            Expanding = 7
            Preferred = 5
        def __init__(self, *args): pass

    class QTextEdit(QWidget):
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._text = ""
            self._html = ""
            self._readonly = False

        def setReadOnly(self, ro: bool) -> None: self._readonly = ro
        def setAcceptRichText(self, rt: bool) -> None: pass
        def setHtml(self, html: str) -> None: self._html = html; self._text = html
        def setText(self, text: str) -> None: self._text = text
        def toPlainText(self) -> str: return self._text
        def textCursor(self) -> QTextCursor: return QTextCursor()
        def setTextCursor(self, cursor: QTextCursor) -> None: pass
        def ensureCursorVisible(self) -> None: pass

    class QLineEdit(QWidget):
        textChanged = Signal(str)

        def __init__(self, text: str = "", parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._text = text

        def setPlaceholderText(self, text: str) -> None: pass
        def setText(self, text: str) -> None: self._text = text; self.textChanged.emit(text)
        def text(self) -> str: return self._text

    class QListWidgetItem:
        def __init__(self, text: str = ""):
            self._text = text
            self._hidden = False

        def text(self) -> str: return self._text
        def setText(self, text: str) -> None: self._text = text
        def setHidden(self, hidden: bool) -> None: self._hidden = hidden
        def isHidden(self) -> bool: return self._hidden

    class QListWidget(QWidget):
        itemClicked = Signal(QListWidgetItem)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._items: List[QListWidgetItem] = []

        def addItem(self, item: QListWidgetItem) -> None: self._items.append(item)
        def count(self) -> int: return len(self._items)
        def item(self, idx: int) -> QListWidgetItem: return self._items[idx]
        def clear(self) -> None: self._items.clear()
        def clearSelection(self) -> None: pass

    class QStackedWidget(QWidget):
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._widgets: List[QWidget] = []
            self._curr_idx = 0

        def addWidget(self, widget: QWidget) -> int:
            self._widgets.append(widget)
            return len(self._widgets) - 1

        def setCurrentIndex(self, idx: int) -> None: self._curr_idx = idx
        def currentIndex(self) -> int: return self._curr_idx

    class QSplitter(QWidget):
        def __init__(self, orientation: Any = Qt.Orientation.Horizontal, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._widgets: List[QWidget] = []
            self._stretch: Dict[int, int] = {}
            self._sizes: List[int] = []

        def setHandleWidth(self, w: int) -> None: pass
        def addWidget(self, widget: QWidget) -> None: self._widgets.append(widget)
        def setStretchFactor(self, index: int, factor: int) -> None: self._stretch[index] = factor
        def setSizes(self, sizes: List[int]) -> None: self._sizes = list(sizes)

    class QApplication:
        def __init__(self, args: List[str]): pass
        def exec(self) -> int: return 0
        def exec_(self) -> int: return 0
