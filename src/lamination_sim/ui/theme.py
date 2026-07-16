"""Visual system shared by the QWidget desktop application."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QFont


COLORS = {
    "window": "#0A0D12",
    "surface": "#11161E",
    "surface_raised": "#171D27",
    "surface_hover": "#1C2430",
    "border": "#283241",
    "border_soft": "#202936",
    "text": "#F4F7FB",
    "text_muted": "#99A6B8",
    "text_dim": "#6E7B8D",
    "accent": "#55D6BE",
    "accent_hover": "#6BE2CC",
    "accent_dark": "#173D39",
    "a": "#60A5FA",
    "a_dark": "#172C48",
    "b": "#F59E6A",
    "b_dark": "#44281D",
    "warning": "#F3C969",
    "danger": "#F07178",
    "success": "#55D6BE",
    "grid": "#293443",
}


def color(name: str, alpha: int | None = None) -> QColor:
    value = QColor(COLORS[name])
    if alpha is not None:
        value.setAlpha(alpha)
    return value


def apply_theme(app) -> None:
    """Apply the app font and a restrained, high-contrast dark theme."""

    # Malgun Gothic is the native Windows Korean UI font and prevents Hangul
    # labels from falling back to tofu glyphs in packaged builds.
    primary_family = "Malgun Gothic" if sys.platform == "win32" else "Noto Sans CJK KR"
    font = QFont(primary_family, 10)
    if not font.exactMatch():
        font = QFont("Segoe UI" if sys.platform == "win32" else "Arial", 10)
    app.setFont(font)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)


STYLE_SHEET = f"""
QWidget {{
    color: {COLORS['text']};
    background: transparent;
}}
QMainWindow, QDialog {{
    background: {COLORS['window']};
}}
QToolTip {{
    color: {COLORS['text']};
    background: {COLORS['surface_raised']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 5px 7px;
}}
QFrame[card="true"], QGroupBox {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border_soft']};
    border-radius: 10px;
}}
QGroupBox {{
    margin-top: 14px;
    padding: 15px 12px 11px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 11px;
    padding: 0 5px;
    color: {COLORS['text_muted']};
}}
QLabel[muted="true"] {{ color: {COLORS['text_muted']}; }}
QLabel[dim="true"] {{ color: {COLORS['text_dim']}; }}
QLabel[heading="true"] {{ font-size: 20px; font-weight: 650; }}
QLabel[subheading="true"] {{ font-size: 13px; font-weight: 600; }}
QPushButton, QToolButton {{
    min-height: 31px;
    padding: 0 12px;
    background: {COLORS['surface_raised']};
    border: 1px solid {COLORS['border']};
    border-radius: 7px;
    font-weight: 550;
}}
QPushButton:hover, QToolButton:hover {{
    background: {COLORS['surface_hover']};
    border-color: #3B485A;
}}
QPushButton:pressed, QToolButton:pressed {{ background: #0E131A; }}
QPushButton:disabled, QToolButton:disabled {{
    color: {COLORS['text_dim']};
    background: {COLORS['surface']};
    border-color: {COLORS['border_soft']};
}}
QPushButton[primary="true"] {{
    color: #071513;
    background: {COLORS['accent']};
    border-color: {COLORS['accent']};
    font-weight: 700;
}}
QPushButton[primary="true"]:hover {{
    background: {COLORS['accent_hover']};
    border-color: {COLORS['accent_hover']};
}}
QPushButton[primary="true"]:pressed {{ background: #42BEA8; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    min-height: 30px;
    padding: 0 8px;
    background: #0D1218;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    selection-background-color: {COLORS['accent_dark']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {COLORS['surface_raised']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_dark']};
    outline: 0;
}}
QTableWidget {{
    background: #0D1218;
    alternate-background-color: #10161E;
    border: 1px solid {COLORS['border_soft']};
    border-radius: 7px;
    gridline-color: {COLORS['border_soft']};
    selection-background-color: {COLORS['accent_dark']};
}}
QHeaderView::section {{
    color: {COLORS['text_muted']};
    background: {COLORS['surface_raised']};
    border: 0;
    border-right: 1px solid {COLORS['border_soft']};
    border-bottom: 1px solid {COLORS['border_soft']};
    padding: 6px;
    font-weight: 600;
}}
QTabWidget::pane {{ border: 0; }}
QTabBar::tab {{
    color: {COLORS['text_muted']};
    padding: 8px 14px;
    margin-right: 3px;
    background: transparent;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {COLORS['text']}; }}
QTabBar::tab:selected {{
    color: {COLORS['text']};
    border-bottom-color: {COLORS['accent']};
}}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:vertical {{
    min-height: 32px;
    background: #354153;
    border-radius: 4px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSlider::groove:horizontal {{
    height: 4px;
    border-radius: 2px;
    background: {COLORS['border']};
}}
QSlider::sub-page:horizontal {{ background: {COLORS['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {COLORS['text']};
    border: 2px solid {COLORS['accent']};
}}
QProgressBar {{
    min-height: 4px;
    max-height: 4px;
    background: {COLORS['border_soft']};
    border: 0;
    border-radius: 2px;
}}
QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 2px; }}
QCheckBox {{ spacing: 7px; color: {COLORS['text_muted']}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    background: #0D1218;
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}
QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}
QStatusBar {{ color: {COLORS['text_muted']}; background: {COLORS['surface']}; }}
"""
