"""Library panel for browsing, searching, and viewing reference sign category images."""

import os
from typing import List, Optional

from app.ui.qt_compat import (
    Qt,
    Signal,
    QRect,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QPaintEvent,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
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

DEFAULT_DATASET_PATH = "/Users/aryansoni/Desktop/Sign Language Judge/SignBridge/dataset/cleaned_images"


class AspectImageLabel(QLabel):
    """Image display label that scales pixmaps to fit while maintaining aspect ratio."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 140)
        self._pixmap: Optional[QPixmap] = None
        self._placeholder_text: str = "No Preview Image"

    def set_image(self, pixmap: Optional[QPixmap], placeholder: str = "No Preview Image") -> None:
        self._pixmap = pixmap
        self._placeholder_text = placeholder
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        margin = 4
        avail_w = max(1, w - (margin * 2))
        avail_h = max(1, h - (margin * 2))

        if self._pixmap and not self._pixmap.isNull():
            orig_size = self._pixmap.size()
            scaled = self._pixmap.scaled(
                avail_w,
                avail_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            draw_x = (w - scaled.width()) // 2
            draw_y = (h - scaled.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled)

            # 1px border in BORDER_COLOR around image
            border_rect = QRect(draw_x, draw_y, scaled.width(), scaled.height())
            painter.setPen(QPen(QColor(BORDER_COLOR), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(border_rect, 6, 6)
        else:
            # Placeholder box
            rect = QRect(margin, margin, avail_w, avail_h)
            painter.setPen(QPen(QColor(BORDER_COLOR), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(BG_CARD))
            painter.drawRoundedRect(rect, 8, 8)

            painter.setPen(QPen(QColor(TEXT_MUTED)))
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_text)


class LibraryPanel(QWidget):
    """
    LibraryPanel occupies the bottom-right section (17.5% width, 50% height).

    Features:
    - Title 'Sign Library' in TEXT_PRIMARY
    - Search field (QLineEdit) with placeholder 'Search signs...'
    - Scrollable QListWidget showing category names with hover and selection styling
    - When a category is selected:
        - Category name heading
        - Representative image from dataset scaled maintaining aspect ratio
        - Back button to return to the list
    - 'No results found' message when search yields nothing
    - Signal: category_selected(str)
    - Methods: set_categories(list[str]), set_dataset_path(str)
    """

    category_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._categories: List[str] = []
        self._dataset_path: str = DEFAULT_DATASET_PATH
        self._current_selected_category: str = ""
        self._init_ui()

    def _init_ui(self) -> None:
        # Panel styling: BG_SECONDARY, 12px rounded corners, 1px BORDER_COLOR, 12px padding
        self.setStyleSheet(f"""
            LibraryPanel {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # --- Stacked Widget: Page 0 = List View, Page 1 = Detail View ---
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background: transparent;")

        # Page 0: Category List View
        self.list_page = self._create_list_page()
        self.stacked_widget.addWidget(self.list_page)

        # Page 1: Category Detail View
        self.detail_page = self._create_detail_page()
        self.stacked_widget.addWidget(self.detail_page)

        main_layout.addWidget(self.stacked_widget, 1)

    def _create_list_page(self) -> QWidget:
        """Create the browse and search category list page."""
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header Row
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("Sign Library")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 15px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.count_badge = QLabel("0 signs")
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

        # Search field (QLineEdit)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search signs...")
        self.search_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                selection-background-color: {ACCENT};
                selection-color: {BG_PRIMARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT};
            }}
            QLineEdit::placeholder {{
                color: {TEXT_MUTED};
            }}
        """)
        self.search_edit.textChanged.connect(self.filter_categories)
        layout.addWidget(self.search_edit)

        # Scrollable QListWidget showing category names
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 4px;
                outline: none;
            }}
            QListWidget::item {{
                color: {TEXT_PRIMARY};
                font-size: 13px;
                padding: 8px 12px;
                border-radius: 6px;
                margin-bottom: 2px;
            }}
            QListWidget::item:hover {{
                background-color: {ACCENT_DIM};
                color: #ffffff;
            }}
            QListWidget::item:selected {{
                background-color: {ACCENT};
                color: {BG_PRIMARY};
                font-weight: 700;
            }}
        """)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget, 1)

        # 'No results found' message when search yields nothing
        self.no_results_label = QLabel("No signs found matching your search")
        self.no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_results_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_MUTED};
                font-size: 13px;
                font-style: italic;
                padding: 24px;
                background: transparent;
            }}
        """)
        self.no_results_label.setVisible(False)
        layout.addWidget(self.no_results_label)

        return page

    def _create_detail_page(self) -> QWidget:
        """Create the category detail view page with representative image and Back button."""
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Top Navigation Bar with Back button and Heading
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)

        # Back button
        self.back_btn = QPushButton("← Back")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setStyleSheet(f"""
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
        """)
        self.back_btn.clicked.connect(self.show_list)
        top_bar.addWidget(self.back_btn)

        # Category name as heading
        self.detail_heading = QLabel("")
        self.detail_heading.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 15px;
                font-weight: 700;
                background: transparent;
            }}
        """)
        self.detail_heading.setWordWrap(True)
        top_bar.addWidget(self.detail_heading, 1)

        layout.addLayout(top_bar)

        # Representative Image Label (maintains aspect ratio, fills available width)
        self.image_container = QFrame()
        self.image_container.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        img_layout = QVBoxLayout(self.image_container)
        img_layout.setContentsMargins(4, 4, 4, 4)

        self.detail_image_label = AspectImageLabel(self)
        img_layout.addWidget(self.detail_image_label)

        layout.addWidget(self.image_container, 1)

        # Detail footer info
        self.detail_info_label = QLabel("")
        self.detail_info_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 12px;
                background: transparent;
            }}
        """)
        layout.addWidget(self.detail_info_label)

        return page

    # --- Category Selection and Navigation ---

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle category list item click."""
        category = item.text()
        self.show_category(category)
        self.category_selected.emit(category)

    def show_category(self, category: str) -> None:
        """Display representative image and category information in detail view."""
        self._current_selected_category = category
        self.detail_heading.setText(category)
        self.detail_info_label.setText(f"Sign reference for: {category}")

        # Load representative image from dataset
        # Path: /Users/aryansoni/Desktop/Sign Language Judge/images for phrases/{category}/1.png
        image_path = os.path.join(self._dataset_path, category, "1.png")

        pixmap = None
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
        else:
            # Check other possible image files in directory as fallback
            cat_dir = os.path.join(self._dataset_path, category)
            if os.path.isdir(cat_dir):
                try:
                    files = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        pixmap = QPixmap(os.path.join(cat_dir, files[0]))
                except Exception:
                    pixmap = None

        if pixmap and not pixmap.isNull():
            self.detail_image_label.set_image(pixmap)
        else:
            self.detail_image_label.set_image(
                None,
                placeholder=f"Image not found at:\n{category}/1.png"
            )

        self.stacked_widget.setCurrentIndex(1)

    def show_list(self) -> None:
        """Switch back to category list view and clear selection."""
        self.list_widget.clearSelection()
        self.stacked_widget.setCurrentIndex(0)

    # --- Search and Filter API ---

    def filter_categories(self, query: str) -> None:
        """Filter category list in real-time (case-insensitive)."""
        term = query.strip().lower()
        visible_count = 0

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            matches = term in item.text().lower()
            item.setHidden(not matches)
            if matches:
                visible_count += 1

        if visible_count == 0 and self.list_widget.count() > 0:
            self.list_widget.setVisible(False)
            self.no_results_label.setVisible(True)
        else:
            self.list_widget.setVisible(True)
            self.no_results_label.setVisible(False)

    # --- Public Data API ---

    def set_categories(self, categories: List[str]) -> None:
        """Populate the list with category names."""
        self._categories = sorted(list(set(categories)))
        self.list_widget.clear()

        for cat in self._categories:
            item = QListWidgetItem(cat)
            self.list_widget.addItem(item)

        total = len(self._categories)
        self.count_badge.setText(f"{total} {'sign' if total == 1 else 'signs'}")
        self.filter_categories(self.search_edit.text())

    def set_dataset_path(self, path: str) -> None:
        """Configure the root dataset path used for loading category images."""
        self._dataset_path = path
        # Refresh current image if detail view is active
        if self._current_selected_category:
            self.show_category(self._current_selected_category)
