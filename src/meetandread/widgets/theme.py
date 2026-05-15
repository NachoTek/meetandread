"""
Adaptive light/dark theme for floating panels and widget context menus.

Provides:
- ThemePalette dataclass with all named color tokens
- DARK_PALETTE / LIGHT_PALETTE presets
- current_palette() — auto-detects Windows desktop theme via Qt
- is_dark_mode() — quick boolean check
- Stylesheet generator functions that accept a ThemePalette and return QSS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arrow image paths for combo box / spinbox (SVG, crisp at any DPI)
# ---------------------------------------------------------------------------
_WIDGETS_DIR = Path(__file__).resolve().parent
ARROW_DOWN_SVG = str(_WIDGETS_DIR / "arrow-down.svg").replace("\\", "/")
ARROW_UP_SVG = str(_WIDGETS_DIR / "arrow-up.svg").replace("\\", "/")
CHECKMARK_SVG = str(_WIDGETS_DIR / "checkmark.svg").replace("\\", "/")


# ---------------------------------------------------------------------------
# ThemePalette — all named colour tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThemePalette:
    """Named colour tokens for a single theme variant.

    Every token is a hex colour string (e.g. "#1a1a1a") that can be
    interpolated directly into QSS templates.
    """

    # Backgrounds
    bg: str
    surface: str
    surface_alt: str
    surface_hover: str
    dialog_bg: str

    # Borders
    border: str
    border_light: str
    border_strong: str

    # Text
    text: str
    text_secondary: str
    text_tertiary: str
    text_disabled: str

    # Semantic colours
    accent: str
    accent_text: str
    danger: str
    info: str

    # Resize grip
    grip_bg: str
    grip_hover: str

    # Badge / overlay
    badge_bg: str
    separator: str


# ---------------------------------------------------------------------------
# Preset palettes — dark values extracted from current floating_panels.py
# ---------------------------------------------------------------------------

DARK_PALETTE = ThemePalette(
    bg="#1a1a1a",
    surface="#2a2a2a",
    surface_alt="#333333",
    surface_hover="#3a3a3a",
    dialog_bg="#1a1a1a",
    border="#444444",
    border_light="#555555",
    border_strong="#666666",
    text="#ffffff",
    text_secondary="#dddddd",
    text_tertiary="#aaaaaa",
    text_disabled="#555555",
    accent="#ff5545",
    accent_text="#ffffff",
    danger="#F44336",
    info="#4FC3F7",
    grip_bg="rgba(255, 255, 255, 60)",
    grip_hover="rgba(255, 255, 255, 120)",
    badge_bg="rgba(30, 30, 30, 210)",
    separator="#444444",
)

LIGHT_PALETTE = ThemePalette(
    bg="#ffffff",
    surface="#f5f5f5",
    surface_alt="#e0e0e0",
    surface_hover="#d6d6d6",
    dialog_bg="#ffffff",
    border="#bdbdbd",
    border_light="#cccccc",
    border_strong="#999999",
    text="#212121",
    text_secondary="#424242",
    text_tertiary="#757575",
    text_disabled="#9e9e9e",
    accent="#2E7D32",
    accent_text="#ffffff",
    danger="#C62828",
    info="#0277BD",
    grip_bg="rgba(0, 0, 0, 50)",
    grip_hover="rgba(0, 0, 0, 100)",
    badge_bg="rgba(255, 255, 255, 230)",
    separator="#bdbdbd",
)


# ---------------------------------------------------------------------------
# Theme detection helpers
# ---------------------------------------------------------------------------

def current_palette() -> ThemePalette:
    """Return the active palette based on the desktop colour scheme.

    Uses ``QGuiApplication.styleHints().colorScheme`` to detect the
    Windows light/dark theme.  Falls back to ``DARK_PALETTE`` when the
    scheme is ``Unknown`` or when Qt is unavailable (e.g. during tests
    without a QApplication).

    Returns:
        ThemePalette — either LIGHT_PALETTE or DARK_PALETTE.
    """
    try:
        from PyQt6.QtGui import QGuiApplication
        hints = QGuiApplication.styleHints()
        if hints is None:
            logger.info("Theme detection: no styleHints, falling back to dark")
            return DARK_PALETTE
        scheme = hints.colorScheme()
        if scheme is None:
            logger.info("Theme detection: scheme is None, falling back to dark")
            return DARK_PALETTE
        # Import the enum for comparison
        from PyQt6.QtGui import QtColorScheme
        if scheme == QtColorScheme.Dark:
            logger.debug("Theme detected: Dark")
            return DARK_PALETTE
        elif scheme == QtColorScheme.Light:
            logger.info("Theme detected: Light")
            return LIGHT_PALETTE
        else:
            logger.info("Theme detected: Unknown, falling back to dark")
            return DARK_PALETTE
    except (ImportError, RuntimeError) as exc:
        logger.info("Theme detection unavailable (%s), falling back to dark", exc)
        return DARK_PALETTE


def is_dark_mode() -> bool:
    """Return True when the active theme is the dark palette."""
    return current_palette() is DARK_PALETTE


# ---------------------------------------------------------------------------
# Stylesheet generators — each accepts a ThemePalette and returns QSS
# ---------------------------------------------------------------------------

def panel_base_css(p: ThemePalette, class_name: str = "QWidget") -> str:
    """Panel background, border, and border-radius.

    Args:
        p: Active theme palette.
        class_name: QWidget subclass name for the QSS selector.

    Returns:
        QSS string for the panel base.
    """
    return f"""
        {class_name} {{
            background-color: {p.bg};
            border: 1px solid {p.border};
            border-radius: 10px;
        }}
    """


def glass_panel_css(p: ThemePalette, class_name: str = "QWidget") -> str:
    """Glass-panel style matching the widget's translucent aesthetic.

    Uses semi-transparent background so desktop shows through at the
    panel's windowOpacity level (0.87 idle, 1.0 active). The border
    is subtle and semi-transparent to avoid a harsh rectangular frame.

    Args:
        p: Active theme palette.
        class_name: QWidget subclass name for the QSS selector.

    Returns:
        QSS string for the glass panel base.
    """
    # Extract RGB from hex bg color for rgba()
    bg_hex = p.bg.lstrip("#")
    bg_r, bg_g, bg_b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    border_hex = p.border.lstrip("#")
    br_r, br_g, br_b = int(border_hex[0:2], 16), int(border_hex[2:4], 16), int(border_hex[4:6], 16)
    return f"""
        {class_name} {{
            background-color: rgba({bg_r}, {bg_g}, {bg_b}, 230);
            border: 1px solid rgba({br_r}, {br_g}, {br_b}, 80);
            border-radius: 10px;
        }}
    """


def title_css(p: ThemePalette) -> str:
    """Panel title label — accent-coloured, bold.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the title QLabel.
    """
    return f"""
        QLabel {{
            color: {p.accent};
            font-weight: bold;
            font-size: 14px;
            padding: 5px;
        }}
    """


def header_button_css(p: ThemePalette, variant: str = "close") -> str:
    """Header button (close or legend toggle).

    Args:
        p: Active theme palette.
        variant: 'close' or 'legend'.

    Returns:
        QSS string for the button.
    """
    hover_bg = p.danger if variant == "close" else p.surface_hover
    hover_border = p.danger if variant == "close" else p.accent
    checked_bg = p.danger if variant == "close" else p.accent
    checked_text = p.text if variant == "close" else p.accent_text

    return f"""
        QPushButton {{
            background-color: {p.surface_alt};
            color: {p.text};
            border: 1px solid {p.border_light};
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            padding: 0;
        }}
        QPushButton:hover {{
            background-color: {hover_bg};
            border-color: {hover_border};
        }}
        QPushButton:checked {{
            background-color: {checked_bg};
            color: {checked_text};
            border-color: {checked_bg};
        }}
    """


def tab_widget_css(p: ThemePalette) -> str:
    """QTabWidget::pane and QTabBar::tab styles.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for tab widget and tab bar.
    """
    return f"""
        QTabWidget::pane {{
            border: 1px solid {p.border};
            border-radius: 5px;
            background-color: {p.bg};
        }}
        QTabBar::tab {{
            background-color: {p.surface};
            color: {p.text_tertiary};
            padding: 6px 14px;
            border: 1px solid {p.border};
            border-bottom: none;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background-color: {p.surface_alt};
            color: {p.accent};
            font-weight: bold;
        }}
        QTabBar::tab:hover {{
            background-color: {p.surface_hover};
        }}
    """


def text_area_css(p: ThemePalette) -> str:
    """QTextEdit / QTextBrowser styling.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for text areas.
    """
    return f"""
        QTextEdit {{
            background-color: {p.surface};
            color: {p.text};
            border: none;
            border-radius: 5px;
            padding: 8px;
            font-size: 13px;
            line-height: 1.4;
        }}
        QTextBrowser {{
            background-color: {p.surface};
            color: {p.text};
            border: none;
            border-radius: 5px;
            padding: 8px;
            font-size: 13px;
            line-height: 1.4;
        }}
    """


def combo_box_css(p: ThemePalette, accent_color: str | None = None) -> str:
    """QComboBox with optional accent colour override.

    Args:
        p: Active theme palette.
        accent_color: Override for hover border (defaults to palette accent).

    Returns:
        QSS string for combo boxes.
    """
    accent = accent_color or p.accent
    return f"""
        QComboBox {{
            background-color: {p.surface};
            color: {p.text_secondary};
            border: 1px solid {p.border_light};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            min-height: 22px;
        }}
        QComboBox:hover {{
            border-color: {accent};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {p.text_tertiary};
        }}
        QComboBox QAbstractItemView {{
            background-color: {p.surface};
            color: {p.text_secondary};
            border: 1px solid {p.border_light};
            selection-background-color: {p.surface_alt};
            selection-color: {p.text};
        }}
    """


def action_button_css(p: ThemePalette, variant: str = "scrub") -> str:
    """Action button with semantic variants.

    Variants:
        'scrub'     — info-tinted background, info text
        'delete'    — danger-tinted background, danger text
        'benchmark' — neutral background, accent text
        'dialog'    — neutral background for dialog buttons

    Args:
        p: Active theme palette.
        variant: One of 'scrub', 'delete', 'benchmark', 'dialog'.

    Returns:
        QSS string for the button.
    """
    if variant == "scrub":
        return f"""
            QPushButton {{
                background-color: {p.surface};
                color: {p.info};
                border: 1px solid {p.border_light};
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {p.surface_hover};
                border-color: {p.info};
            }}
            QPushButton:pressed {{
                background-color: {p.surface};
            }}
            QPushButton:disabled {{
                background-color: {p.surface_alt};
                color: {p.text_disabled};
                border-color: {p.border};
            }}
        """
    elif variant == "delete":
        return f"""
            QPushButton {{
                background-color: {p.surface};
                color: {p.danger};
                border: 1px solid {p.border_light};
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {p.surface_hover};
                border-color: {p.danger};
            }}
            QPushButton:pressed {{
                background-color: {p.surface};
            }}
        """
    elif variant == "benchmark":
        return f"""
            QPushButton {{
                background-color: {p.surface_alt};
                color: {p.accent};
                border: 1px solid {p.border_light};
                border-radius: 5px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {p.surface_hover};
                border-color: {p.accent};
            }}
            QPushButton:pressed {{
                background-color: {p.surface};
            }}
            QPushButton:disabled {{
                color: {p.text_disabled};
                border-color: {p.border};
            }}
        """
    else:  # 'dialog'
        return f"""
            QPushButton {{
                background-color: {p.surface_alt};
                color: {p.text_secondary};
                border: 1px solid {p.border_light};
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background-color: {p.surface_hover};
                border-color: {p.accent};
            }}
            QPushButton:pressed {{
                background-color: {p.surface};
            }}
        """


def list_widget_css(p: ThemePalette) -> str:
    """QListWidget styling.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for list widgets.
    """
    return f"""
        QListWidget {{
            background-color: {p.surface};
            color: {p.text_secondary};
            border: none;
            border-radius: 5px;
            font-size: 12px;
            padding: 4px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 6px 8px;
            border-bottom: 1px solid {p.surface_alt};
        }}
        QListWidget::item:selected {{
            background-color: {p.surface_alt};
            color: {p.text};
        }}
        QListWidget::item:hover {{
            background-color: {p.surface_hover};
        }}
    """


def progress_bar_css(p: ThemePalette, chunk_color: str | None = None) -> str:
    """QProgressBar template with customisable chunk colour.

    Args:
        p: Active theme palette.
        chunk_color: Colour for the progress chunk (defaults to accent).

    Returns:
        QSS string for progress bars.
    """
    color = chunk_color or p.accent
    return f"""
        QProgressBar {{
            border: 1px solid {p.border_light};
            border-radius: 4px;
            background-color: {p.surface};
            text-align: center;
            color: {p.text_secondary};
            font-size: 11px;
            height: 16px;
        }}
        QProgressBar::chunk {{
            background-color: {color};
            border-radius: 3px;
        }}
    """


def context_menu_css(p: ThemePalette, accent_color: str | None = None) -> str:
    """QMenu / context menu styling — Aetheric glass design.

    Matches the Aetheric settings shell aesthetic: dark translucent
    background, rounded corners, AETHERIC_RED accent for selection.

    Args:
        p: Active theme palette.
        accent_color: Override for selected-item background (defaults to
            AETHERIC_RED in dark mode, p.accent otherwise).

    Returns:
        QSS string for menus.
    """
    accent = accent_color or AETHERIC_RED
    return f"""
        QMenu {{
            background-color: {AETHERIC_SETTINGS_BG};
            color: {p.text_secondary};
            border: 1px solid {AETHERIC_BORDER_LIGHT};
            border-radius: {AETHERIC_RADIUS};
            padding: 6px 4px;
        }}
        QMenu::item {{
            padding: 8px 24px;
            border-radius: 8px;
            color: {p.text_secondary};
        }}
        QMenu::item:selected {{
            background-color: rgba(255, 85, 69, 0.2);
            color: {accent};
        }}
        QMenu::separator {{
            height: 1px;
            background: {AETHERIC_BORDER_LIGHT};
            margin: 4px 8px;
        }}
    """


def dialog_css(p: ThemePalette) -> str:
    """QDialog base styling.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for dialogs.
    """
    return f"""
        QDialog {{
            background-color: {p.dialog_bg};
        }}
        QLabel {{
            color: {p.text_secondary};
            font-size: 12px;
        }}
    """


def badge_css(p: ThemePalette) -> str:
    """New-content badge (auto-scroll pause indicator).

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the badge QPushButton.
    """
    return f"""
        QPushButton {{
            background-color: {p.badge_bg};
            color: {p.text};
            border: 1px solid {p.border_strong};
            border-radius: 12px;
            padding: 4px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {p.surface_hover};
            border: 1px solid {p.border_strong};
        }}
        QPushButton:pressed {{
            background-color: {p.surface_alt};
        }}
    """


def resize_grip_css(p: ThemePalette) -> str:
    """QSizeGrip styling.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the resize grip.
    """
    return f"""
        QSizeGrip {{
            background-color: {p.grip_bg};
            border-radius: 3px;
        }}
        QSizeGrip:hover {{
            background-color: {p.grip_hover};
        }}
    """


def legend_overlay_css(p: ThemePalette) -> Dict[str, str]:
    """Legend overlay styles — returns a dict of named QSS strings.

    The legend overlay is a QFrame with child QLabels and a separator
    QFrame.  Returns individual QSS blocks so the caller can apply them
    to the correct child widgets.

    Args:
        p: Active theme palette.

    Returns:
        Dict with keys: 'overlay', 'title', 'separator', 'range_label',
        'desc_label'.  Each value is a QSS string.
    """
    return {
        "overlay": f"""
            QFrame {{
                background-color: {p.surface};
                border: 1px solid {p.border_light};
                border-radius: 8px;
                padding: 8px;
            }}
        """,
        "title": f"color: {p.text}; font-weight: bold; font-size: 12px; border: none;",
        "separator": f"background-color: {p.border_light}; border: none;",
        "range_label": f"color: {p.text}; font-size: 11px; border: none;",
        "desc_label": f"color: {p.text_tertiary}; font-size: 11px; border: none;",
    }


def detail_header_css(p: ThemePalette) -> str:
    """Detail header frame (above history viewer).

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the detail header QFrame.
    """
    return f"""
        QFrame {{
            background-color: {p.surface};
            border-bottom: 1px solid {p.border};
            border-radius: 0px;
        }}
    """


def separator_css(p: ThemePalette) -> str:
    """QFrame horizontal separator.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for a horizontal separator line.
    """
    return f"QFrame {{ background-color: {p.separator}; max-height: 1px; border: none; }}"


def info_label_css(p: ThemePalette) -> str:
    """Small metric / info labels.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for info QLabels.
    """
    return f"QLabel {{ color: {p.text_tertiary}; font-size: 11px; }}"


def status_label_css(p: ThemePalette) -> str:
    """Status text label.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the status QLabel.
    """
    return f"""
        QLabel {{
            color: {p.text_tertiary};
            font-size: 11px;
            padding: 3px;
        }}
    """


def splitter_css(p: ThemePalette) -> str:
    """QSplitter handle styling.

    Args:
        p: Active theme palette.

    Returns:
        QSS string for splitter handles.
    """
    return f"""
        QSplitter::handle {{
            background-color: {p.border};
            height: 3px;
        }}
    """


# ---------------------------------------------------------------------------
# Aetheric Glass theme helpers — translucent dark settings shell
# ---------------------------------------------------------------------------

# Design tokens for the Aetheric Glass design system.
# See docs/AETHERIC-GLASS-DESIGN-SYSTEM.md for full reference.
AETHERIC_GLASS_BG = "rgba(30, 29, 30, 200)"
AETHERIC_SETTINGS_BG = "rgb(30, 29, 30)"  # solid — no translucency for the settings shell
AETHERIC_SIDEBAR_BG = "#101018"  # dark blue-tinted — visual separation for nav rail
AETHERIC_GLASS_ROW_BG = "rgba(53, 52, 54, 0.2)"
AETHERIC_SIDEBAR_WIDTH = "256px"
AETHERIC_RADIUS = "12px"

# Directional border cues: light on top-left, dark on bottom-right
AETHERIC_BORDER_LIGHT = "rgba(255, 255, 255, 30)"
AETHERIC_BORDER_DARK = "rgba(0, 0, 0, 80)"

# Navigation pill colours
AETHERIC_NAV_ACTIVE_BG = "rgba(255, 85, 69, 0.2)"  # red-500/20
AETHERIC_NAV_ACTIVE_GLOW = "rgba(255, 85, 69, 0.4)"  # red glow
AETHERIC_NAV_INACTIVE_TEXT = "rgba(255, 255, 255, 0.9)"  # white/90
AETHERIC_NAV_HOVER_BG = "rgba(255, 255, 255, 0.05)"  # white/5

# Accent colours for the Aetheric theme
AETHERIC_RED = "#ff5545"  # primary-container
AETHERIC_PURPLE = "#c9bfff"  # secondary
AETHERIC_CYAN = "#00dbe9"  # tertiary


# ---------------------------------------------------------------------------
# Aetheric CC overlay tokens — compact closed-caption live transcript
# ---------------------------------------------------------------------------

AETHERIC_CC_BG = "rgba(30, 29, 30, 179)"  # 70% opacity
AETHERIC_CC_TEXT = "rgba(180, 180, 180, 230)"  # Grey mono text
AETHERIC_CC_RADIUS = AETHERIC_RADIUS  # 12px — matches shell radius
AETHERIC_CC_PADDING = "10px 14px"
AETHERIC_CC_FONT_SIZE = "48px"
AETHERIC_CC_FONT_FAMILY = "'Consolas', 'Courier New', monospace"  # CC-style monospace
AETHERIC_CC_LINE_COUNT = 2


def aetheric_settings_shell_css(p: ThemePalette) -> str:
    """Aetheric Glass settings shell — solid dark panel base.

    The shell is a frameless top-level window with directional borders
    (light on top-left, dark on bottom-right) and a 12px corner radius.
    Uses a solid background (no translucency) for readability.

    Object name selector: ``AethericSettingsShell``

    Args:
        p: Active theme palette (used for fallback text colours).

    Returns:
        QSS string for the settings shell container.
    """
    return f"""
        QWidget#AethericSettingsShell {{
            background-color: transparent;
            border: 1px solid {AETHERIC_BORDER_LIGHT};
            border-top: 1px solid {AETHERIC_BORDER_LIGHT};
            border-left: 1px solid {AETHERIC_BORDER_LIGHT};
            border-bottom: 1px solid {AETHERIC_BORDER_DARK};
            border-right: 1px solid {AETHERIC_BORDER_DARK};
            border-radius: {AETHERIC_RADIUS};
        }}
    """


def aetheric_sidebar_css(p: ThemePalette) -> str:
    """Aetheric Glass sidebar — dark vertical navigation rail.

    Fixed 256px width with a solid dark background.

    Object name selector: ``AethericSidebar``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the sidebar container.
    """
    return f"""
        QWidget#AethericSidebar {{
            background-color: {AETHERIC_SIDEBAR_BG};
            border: none;
            border-right: 2px solid rgba(255, 85, 69, 60);
            border-top-left-radius: 0px;
            border-bottom-left-radius: {AETHERIC_RADIUS};
            min-width: {AETHERIC_SIDEBAR_WIDTH};
            max-width: {AETHERIC_SIDEBAR_WIDTH};
        }}
    """


def aetheric_title_bar_css(p: ThemePalette) -> str:
    """Aetheric Glass title bar — 25px drag handle at top of settings panel.

    Uses the sidebar dark background for visual cohesion with the nav rail.
    Label is centered-left with muted text. No window controls.

    Object name selectors: ``AethericTitleBar``, ``AethericTitleLabel``
    """
    return f"""
        QWidget#AethericTitleBar {{
            background-color: {AETHERIC_SETTINGS_BG};
            border-top-left-radius: {AETHERIC_RADIUS};
            border-top-right-radius: {AETHERIC_RADIUS};
            min-height: 25px;
            max-height: 25px;
        }}
        QLabel#AethericTitleLabel {{
            color: rgba(255, 255, 255, 120);
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 1px;
        }}
    """


def aetheric_nav_button_css(p: ThemePalette) -> str:
    """Aetheric Glass navigation pill buttons for sidebar nav.

    Active state uses red-500/20 background with red glow border.
    Inactive state uses white/40 text with white/5 hover background.

    Object name selector: ``AethericNavButton``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for navigation buttons.
    """
    return f"""
        QPushButton#AethericNavButton {{
            background-color: transparent;
            color: {AETHERIC_NAV_INACTIVE_TEXT};
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 8px 16px;
            text-align: left;
            font-size: 13px;
            font-weight: normal;
        }}
        QPushButton#AethericNavButton:hover {{
            background-color: {AETHERIC_NAV_HOVER_BG};
        }}
        QPushButton#AethericNavButton:checked {{
            background-color: {AETHERIC_NAV_ACTIVE_BG};
            color: {AETHERIC_RED};
            border: 1px solid {AETHERIC_NAV_ACTIVE_GLOW};
            font-weight: bold;
        }}
    """


def aetheric_placeholder_css(p: ThemePalette) -> str:
    """Aetheric Glass placeholder rows — glass translucent item rows.

    Used for placeholder/empty-state rows in the settings content area.

    Object name selector: ``AethericPlaceholderRow``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for placeholder rows.
    """
    return f"""
        QWidget#AethericPlaceholderRow {{
            background-color: {AETHERIC_GLASS_ROW_BG};
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 12px 16px;
        }}
        QWidget#AethericPlaceholderRow:hover {{
            border: 1px solid {AETHERIC_RED};
        }}
    """


# ---------------------------------------------------------------------------
# Aetheric CC overlay helper — compact closed-caption live transcript
# ---------------------------------------------------------------------------

def aetheric_cc_overlay_css(p: ThemePalette) -> str:
    """CC overlay styling — monospace grey text, transparent background.

    Background is painted manually in paintEvent() because QSS
    background-color with alpha does not render on Windows
    frameless translucent windows.

    Object name selector: ``QWidget#AethericCCOverlay``
    Child text selector: ``QTextEdit#AethericCCText``
    """
    return f"""
        QWidget#AethericCCOverlay {{
            background-color: transparent;
            border: none;
            padding: {AETHERIC_CC_PADDING};
        }}
        QTextEdit#AethericCCText {{
            color: {AETHERIC_CC_TEXT};
            font-family: {AETHERIC_CC_FONT_FAMILY};
            font-size: {AETHERIC_CC_FONT_SIZE};
            background-color: transparent;
            border: none;
        }}
    """


def aetheric_combo_box_css(p: ThemePalette) -> str:
    """Aetheric Glass combo box — modern dropdown styling.

    Solid readable text, visible dropdown button area with separator,
    and a clean popup list with hover/selection states.

    Object name selector: ``AethericComboBox``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for Aetheric-styled combo boxes.
    """
    return f"""
        QComboBox#AethericComboBox {{
            background-color: {AETHERIC_SETTINGS_BG};
            color: {p.text_secondary};
            border: 1px solid {AETHERIC_BORDER_LIGHT};
            border-radius: 8px;
            padding: 6px 32px 6px 12px;
            font-size: 12px;
            min-height: 24px;
        }}
        QComboBox#AethericComboBox:hover {{
            border-color: {AETHERIC_RED};
        }}
        QComboBox#AethericComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border: none;
            border-left: 1px solid {AETHERIC_BORDER_LIGHT};
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }}
        QComboBox#AethericComboBox::down-arrow {{
            image: url({ARROW_DOWN_SVG});
            width: 12px;
            height: 12px;
        }}
        QComboBox#AethericComboBox QAbstractItemView {{
            background-color: {AETHERIC_SETTINGS_BG};
            color: {p.text_secondary};
            border: 1px solid {AETHERIC_BORDER_LIGHT};
            border-radius: 8px;
            padding: 4px;
            outline: none;
            selection-background-color: {AETHERIC_NAV_ACTIVE_BG};
            selection-color: {AETHERIC_RED};
        }}
        QComboBox#AethericComboBox QAbstractItemView::item {{
            padding: 6px 12px;
            min-height: 24px;
            border-radius: 6px;
        }}
        QComboBox#AethericComboBox QAbstractItemView::item:hover {{
            background-color: {AETHERIC_NAV_HOVER_BG};
        }}
    """


# ---------------------------------------------------------------------------
# Aetheric Checkbox — SVG checkmark indicator
# ---------------------------------------------------------------------------

def aetheric_checkbox_css(p: ThemePalette) -> str:
    """Aetheric checkbox with SVG checkmark indicator.

    Uses AETHERIC_RED for the checked background and a white SVG
    checkmark for clear on/off state visibility.

    Object name selector: ``AethericCheckBox``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for Aetheric-styled checkboxes.
    """
    return f"""
        QCheckBox#AethericCheckBox {{
            color: {p.text_secondary};
            spacing: 8px;
            font-size: 12px;
        }}
        QCheckBox#AethericCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {AETHERIC_BORDER_LIGHT};
            border-radius: 4px;
            background-color: {AETHERIC_SETTINGS_BG};
        }}
        QCheckBox#AethericCheckBox::indicator:checked {{
            background-color: {AETHERIC_RED};
            border-color: {AETHERIC_RED};
            image: url({CHECKMARK_SVG});
        }}
        QCheckBox#AethericCheckBox::indicator:hover {{
            border-color: {AETHERIC_RED};
        }}
    """


# ---------------------------------------------------------------------------
# Aetheric History page helpers — scoped selectors for History widgets
# ---------------------------------------------------------------------------

def aetheric_history_list_css(p: ThemePalette) -> str:
    """Aetheric Glass history recording list.

    Glass-row items with translucent backgrounds, red hover accent,
    and directional border cues. Selected items use the red active bg.

    Object name selector: ``QListWidget#AethericHistoryList``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the history recording list.
    """
    return f"""
        QListWidget#AethericHistoryList {{
            background-color: transparent;
            color: {AETHERIC_NAV_INACTIVE_TEXT};
            border: none;
            border-radius: 8px;
            font-size: 12px;
            padding: 4px;
            outline: none;
        }}
        QListWidget#AethericHistoryList::item {{
            background-color: {AETHERIC_GLASS_ROW_BG};
            border: 1px solid transparent;
            border-top: 1px solid {AETHERIC_BORDER_LIGHT};
            border-left: 1px solid {AETHERIC_BORDER_LIGHT};
            border-bottom: 1px solid {AETHERIC_BORDER_DARK};
            border-right: 1px solid {AETHERIC_BORDER_DARK};
            border-radius: 8px;
            padding: 8px 12px;
            margin: 2px 4px;
        }}
        QListWidget#AethericHistoryList::item:selected {{
            background-color: {AETHERIC_NAV_ACTIVE_BG};
            color: {AETHERIC_RED};
            border: 1px solid {AETHERIC_NAV_ACTIVE_GLOW};
        }}
        QListWidget#AethericHistoryList::item:hover {{
            background-color: {AETHERIC_NAV_HOVER_BG};
            border: 1px solid {AETHERIC_RED};
        }}
    """


def aetheric_history_viewer_css(p: ThemePalette) -> str:
    """Aetheric Glass history transcript viewer.

    Transparent text browser with Aetheric text colour and subtle
    directional border for visual separation from the list.

    Object name selector: ``QTextBrowser#AethericHistoryViewer``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the transcript viewer.
    """
    return f"""
        QTextBrowser#AethericHistoryViewer {{
            background-color: transparent;
            color: {AETHERIC_NAV_INACTIVE_TEXT};
            border: none;
            border-radius: 8px;
            padding: 8px;
            font-size: 13px;
        }}
    """


def aetheric_history_splitter_css(p: ThemePalette) -> str:
    """Aetheric Glass splitter handle between history list and viewer.

    Uses the Aetheric border-dark token for a subtle separator that
    matches the directional border convention.

    Object name selector: ``QSplitter#AethericHistorySplitter``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the history splitter handle.
    """
    return f"""
        QSplitter#AethericHistorySplitter::handle {{
            background-color: {AETHERIC_BORDER_DARK};
            width: 2px;
            margin: 4px 2px;
        }}
    """


def aetheric_history_header_css(p: ThemePalette) -> str:
    """Aetheric Glass history detail header frame.

    Transparent background with a bottom directional border that
    visually separates the header actions from the content below.

    Object name selector: ``QFrame#AethericHistoryHeader``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for the detail header frame.
    """
    return f"""
        QFrame#AethericHistoryHeader {{
            background-color: transparent;
            border: none;
            border-bottom: 1px solid {AETHERIC_BORDER_DARK};
            padding: 4px 8px;
        }}
    """


def aetheric_history_action_button_css(p: ThemePalette) -> str:
    """Aetheric Glass action buttons for the History page.

    Supports four action variants driven by a Qt dynamic property
    ``action`` on the button:

    - ``action="scrub"``   — cyan/info tint
    - ``action="delete"``  — red/danger tint
    - ``action="accept"``  — green/accent tint
    - ``action="reject"``  — purple/secondary tint

    The base style uses glass-row background with red hover accent.
    Variants are applied via ``QPushButton[action="..."]`` selectors
    so each button's action property controls its colour scheme.

    Object name selector: ``QPushButton#AethericHistoryActionButton``

    Args:
        p: Active theme palette.

    Returns:
        QSS string for all action button variants.
    """
    return f"""
        QPushButton#AethericHistoryActionButton {{
            background-color: {AETHERIC_GLASS_ROW_BG};
            color: {AETHERIC_NAV_INACTIVE_TEXT};
            border: 1px solid transparent;
            border-top: 1px solid {AETHERIC_BORDER_LIGHT};
            border-left: 1px solid {AETHERIC_BORDER_LIGHT};
            border-bottom: 1px solid {AETHERIC_BORDER_DARK};
            border-right: 1px solid {AETHERIC_BORDER_DARK};
            border-radius: 8px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        QPushButton#AethericHistoryActionButton:hover {{
            background-color: {AETHERIC_NAV_HOVER_BG};
            border: 1px solid {AETHERIC_RED};
            color: {AETHERIC_RED};
        }}
        QPushButton#AethericHistoryActionButton:pressed {{
            background-color: {AETHERIC_NAV_ACTIVE_BG};
        }}
        QPushButton#AethericHistoryActionButton:disabled {{
            color: {AETHERIC_BORDER_DARK};
            border-color: {AETHERIC_BORDER_DARK};
        }}
        QPushButton#AethericHistoryActionButton[action="scrub"] {{
            color: {AETHERIC_CYAN};
        }}
        QPushButton#AethericHistoryActionButton[action="scrub"]:hover {{
            border-color: {AETHERIC_CYAN};
            color: {AETHERIC_CYAN};
        }}
        QPushButton#AethericHistoryActionButton[action="delete"] {{
            color: {AETHERIC_RED};
        }}
        QPushButton#AethericHistoryActionButton[action="delete"]:hover {{
            border-color: {AETHERIC_RED};
            color: {AETHERIC_RED};
        }}
        QPushButton#AethericHistoryActionButton[action="accept"] {{
            color: {AETHERIC_RED};
        }}
        QPushButton#AethericHistoryActionButton[action="accept"]:hover {{
            border-color: {AETHERIC_RED};
            color: {AETHERIC_RED};
        }}
        QPushButton#AethericHistoryActionButton[action="reject"] {{
            color: {AETHERIC_PURPLE};
        }}
        QPushButton#AethericHistoryActionButton[action="reject"]:hover {{
            border-color: {AETHERIC_PURPLE};
            color: {AETHERIC_PURPLE};
        }}
    """


def aetheric_playback_toolbar_css(p: ThemePalette) -> Dict[str, str]:
    """Aetheric Glass playback toolbar controls for the History header.

    Returns a dict of scoped QSS strings keyed by widget role:
    'play_button', 'skip_button', 'progress_slider', 'speed_combo',
    'volume_slider', 'volume_icon', 'status_label', 'status_label_error',
    'bookmark_button', 'bookmark_combo', 'bookmark_delete_button'.

    All selectors are scoped to object names (e.g.
    ``QPushButton#AethericHistoryPlaybackButton``) so they cannot
    leak to unrelated widgets.

    Object name selectors used:
    - ``QPushButton#AethericHistoryPlaybackButton``
    - ``QPushButton#AethericHistoryPlaybackSkipBackButton``
    - ``QPushButton#AethericHistoryPlaybackSkipFwdButton``
    - ``QSlider#AethericHistoryPlaybackProgressSlider``
    - ``QComboBox#AethericHistoryPlaybackSpeedCombo``
    - ``QSlider#AethericHistoryPlaybackVolumeSlider``
    - ``QLabel#AethericHistoryPlaybackVolumeIcon``
    - ``QLabel#AethericHistoryPlaybackStatusLabel``
    - ``QPushButton#AethericHistoryBookmarkButton``
    - ``QComboBox#AethericHistoryBookmarkCombo``
    - ``QPushButton#AethericHistoryBookmarkDeleteButton``

    Args:
        p: Active theme palette.

    Returns:
        Dict of named QSS strings, one per playback widget role.
    """
    return {
        "play_button": f"""
            QPushButton#AethericHistoryPlaybackButton {{
                background-color: {AETHERIC_GLASS_ROW_BG};
                color: {p.text_secondary};
                border: 1px solid transparent;
                border-top: 1px solid {AETHERIC_BORDER_LIGHT};
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-bottom: 1px solid {AETHERIC_BORDER_DARK};
                border-right: 1px solid {AETHERIC_BORDER_DARK};
                border-radius: 6px;
                padding: 2px 4px;
                font-size: 13px;
            }}
            QPushButton#AethericHistoryPlaybackButton:hover {{
                background-color: {AETHERIC_NAV_HOVER_BG};
                border: 1px solid {AETHERIC_RED};
                color: {AETHERIC_RED};
            }}
            QPushButton#AethericHistoryPlaybackButton:pressed {{
                background-color: {AETHERIC_NAV_ACTIVE_BG};
            }}
            QPushButton#AethericHistoryPlaybackButton:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
        "skip_button": f"""
            QPushButton#AethericHistoryPlaybackSkipBackButton,
            QPushButton#AethericHistoryPlaybackSkipFwdButton {{
                background-color: {AETHERIC_GLASS_ROW_BG};
                color: {p.text_secondary};
                border: 1px solid transparent;
                border-top: 1px solid {AETHERIC_BORDER_LIGHT};
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-bottom: 1px solid {AETHERIC_BORDER_DARK};
                border-right: 1px solid {AETHERIC_BORDER_DARK};
                border-radius: 6px;
                padding: 2px 4px;
                font-size: 13px;
            }}
            QPushButton#AethericHistoryPlaybackSkipBackButton:hover,
            QPushButton#AethericHistoryPlaybackSkipFwdButton:hover {{
                background-color: {AETHERIC_NAV_HOVER_BG};
                border: 1px solid {AETHERIC_RED};
                color: {AETHERIC_RED};
            }}
            QPushButton#AethericHistoryPlaybackSkipBackButton:pressed,
            QPushButton#AethericHistoryPlaybackSkipFwdButton:pressed {{
                background-color: {AETHERIC_NAV_ACTIVE_BG};
            }}
            QPushButton#AethericHistoryPlaybackSkipBackButton:disabled,
            QPushButton#AethericHistoryPlaybackSkipFwdButton:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
        "progress_slider": f"""
            QSlider#AethericHistoryPlaybackProgressSlider {{
                background: transparent;
                height: 6px;
            }}
            QSlider#AethericHistoryPlaybackProgressSlider::groove:horizontal {{
                background: {AETHERIC_BORDER_DARK};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider#AethericHistoryPlaybackProgressSlider::handle:horizontal {{
                background: {p.text_secondary};
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }}
            QSlider#AethericHistoryPlaybackProgressSlider::handle:horizontal:hover {{
                background: {AETHERIC_RED};
            }}
            QSlider#AethericHistoryPlaybackProgressSlider:disabled {{
                background: transparent;
            }}
            QSlider#AethericHistoryPlaybackProgressSlider::groove:horizontal:disabled {{
                background: transparent;
            }}
            QSlider#AethericHistoryPlaybackProgressSlider::handle:horizontal:disabled {{
                background: {AETHERIC_BORDER_DARK};
            }}
        """,
        "speed_combo": f"""
            QComboBox#AethericHistoryPlaybackSpeedCombo {{
                background-color: {AETHERIC_SETTINGS_BG};
                color: {p.text_tertiary};
                border: 1px solid {AETHERIC_BORDER_LIGHT};
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 11px;
                min-height: 20px;
            }}
            QComboBox#AethericHistoryPlaybackSpeedCombo:hover {{
                border-color: {AETHERIC_RED};
                color: {p.text_secondary};
            }}
            QComboBox#AethericHistoryPlaybackSpeedCombo::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border: none;
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox#AethericHistoryPlaybackSpeedCombo::down-arrow {{
                image: url({ARROW_DOWN_SVG});
                width: 10px;
                height: 10px;
            }}
            QComboBox#AethericHistoryPlaybackSpeedCombo QAbstractItemView {{
                background-color: {AETHERIC_SETTINGS_BG};
                color: {p.text_secondary};
                border: 1px solid {AETHERIC_BORDER_LIGHT};
                border-radius: 8px;
                padding: 2px;
                outline: none;
                selection-background-color: {AETHERIC_NAV_ACTIVE_BG};
                selection-color: {AETHERIC_RED};
            }}
            QComboBox#AethericHistoryPlaybackSpeedCombo:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
        "volume_slider": f"""
            QSlider#AethericHistoryPlaybackVolumeSlider {{
                background: transparent;
                height: 6px;
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider::groove:horizontal {{
                background: {AETHERIC_BORDER_DARK};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider::handle:horizontal {{
                background: {p.text_secondary};
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider::handle:horizontal:hover {{
                background: {AETHERIC_RED};
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider:disabled {{
                background: transparent;
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider::groove:horizontal:disabled {{
                background: transparent;
            }}
            QSlider#AethericHistoryPlaybackVolumeSlider::handle:horizontal:disabled {{
                background: {AETHERIC_BORDER_DARK};
            }}
        """,
        "volume_icon": f"""
            QLabel#AethericHistoryPlaybackVolumeIcon {{
                color: {p.text_secondary};
                font-size: 12px;
            }}
        """,
        "status_label": f"""
            QLabel#AethericHistoryPlaybackStatusLabel {{
                color: {p.text_tertiary};
                font-size: 10px;
                padding: 0 4px;
            }}
        """,
        "status_label_error": f"""
            QLabel#AethericHistoryPlaybackStatusLabel {{
                color: {p.text_disabled};
                font-size: 10px;
                padding: 0 4px;
            }}
        """,
        "bookmark_button": f"""
            QPushButton#AethericHistoryBookmarkButton {{
                background-color: {AETHERIC_GLASS_ROW_BG};
                color: {p.text_secondary};
                border: 1px solid transparent;
                border-top: 1px solid {AETHERIC_BORDER_LIGHT};
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-bottom: 1px solid {AETHERIC_BORDER_DARK};
                border-right: 1px solid {AETHERIC_BORDER_DARK};
                border-radius: 6px;
                padding: 2px 4px;
                font-size: 13px;
            }}
            QPushButton#AethericHistoryBookmarkButton:hover {{
                background-color: {AETHERIC_NAV_HOVER_BG};
                border: 1px solid {AETHERIC_RED};
                color: {AETHERIC_RED};
            }}
            QPushButton#AethericHistoryBookmarkButton:pressed {{
                background-color: {AETHERIC_NAV_ACTIVE_BG};
            }}
            QPushButton#AethericHistoryBookmarkButton:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
        "bookmark_combo": f"""
            QComboBox#AethericHistoryBookmarkCombo {{
                background-color: {AETHERIC_SETTINGS_BG};
                color: {p.text_tertiary};
                border: 1px solid {AETHERIC_BORDER_LIGHT};
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 11px;
                min-height: 20px;
            }}
            QComboBox#AethericHistoryBookmarkCombo:hover {{
                border-color: {AETHERIC_RED};
                color: {p.text_secondary};
            }}
            QComboBox#AethericHistoryBookmarkCombo::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border: none;
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox#AethericHistoryBookmarkCombo::down-arrow {{
                image: url({ARROW_DOWN_SVG});
                width: 10px;
                height: 10px;
            }}
            QComboBox#AethericHistoryBookmarkCombo QAbstractItemView {{
                background-color: {AETHERIC_SETTINGS_BG};
                color: {p.text_secondary};
                border: 1px solid {AETHERIC_BORDER_LIGHT};
                border-radius: 8px;
                padding: 2px;
                outline: none;
                selection-background-color: {AETHERIC_NAV_ACTIVE_BG};
                selection-color: {AETHERIC_RED};
            }}
            QComboBox#AethericHistoryBookmarkCombo:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
        "bookmark_delete_button": f"""
            QPushButton#AethericHistoryBookmarkDeleteButton {{
                background-color: {AETHERIC_GLASS_ROW_BG};
                color: {p.text_tertiary};
                border: 1px solid transparent;
                border-top: 1px solid {AETHERIC_BORDER_LIGHT};
                border-left: 1px solid {AETHERIC_BORDER_LIGHT};
                border-bottom: 1px solid {AETHERIC_BORDER_DARK};
                border-right: 1px solid {AETHERIC_BORDER_DARK};
                border-radius: 6px;
                padding: 2px;
                font-size: 11px;
            }}
            QPushButton#AethericHistoryBookmarkDeleteButton:hover {{
                background-color: {AETHERIC_NAV_HOVER_BG};
                border: 1px solid {AETHERIC_RED};
                color: {AETHERIC_RED};
            }}
            QPushButton#AethericHistoryBookmarkDeleteButton:pressed {{
                background-color: {AETHERIC_NAV_ACTIVE_BG};
            }}
            QPushButton#AethericHistoryBookmarkDeleteButton:disabled {{
                color: {AETHERIC_BORDER_DARK};
                border-color: transparent;
                background-color: transparent;
            }}
        """,
    }
