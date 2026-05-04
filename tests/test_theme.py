"""Tests for the adaptive theme module (theme.py).

Covers:
- ThemePalette field existence and types
- DARK_PALETTE matches hardcoded colours from floating_panels.py
- LIGHT_PALETTE contrast (WCAG AA)
- current_palette() returns DARK_PALETTE when Unknown
- is_dark_mode() correctness
- Each stylesheet generator produces valid QSS with expected tokens
- Contrast ratio >= 4.5:1 for text vs bg in both palettes
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from meetandread.widgets.theme import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    ThemePalette,
    action_button_css,
    badge_css,
    combo_box_css,
    context_menu_css,
    current_palette,
    dialog_css,
    detail_header_css,
    header_button_css,
    is_dark_mode,
    legend_overlay_css,
    list_widget_css,
    panel_base_css,
    progress_bar_css,
    resize_grip_css,
    separator_css,
    info_label_css,
    splitter_css,
    status_label_css,
    tab_widget_css,
    text_area_css,
    title_css,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert a hex colour string to (R, G, B) tuple (0–255).

    Handles both '#RRGGBB' and 'rgba(...)' formats — rgba returns (0,0,0)
    since luminance isn't meaningful for semi-transparent tokens.
    """
    hex_str = hex_str.strip()
    if hex_str.startswith("rgba") or hex_str.startswith("rgb"):
        return (0, 0, 0)  # cannot compute luminance for rgba
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
    if len(hex_str) != 6:
        return (0, 0, 0)
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def _relative_luminance(hex_str: str) -> float:
    """Compute WCAG relative luminance for a hex colour."""
    r, g, b = _hex_to_rgb(hex_str)
    # Linearise
    def linearise(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearise(r) + 0.7152 * linearise(g) + 0.0722 * linearise(b)


def _contrast_ratio(hex1: str, hex2: str) -> float:
    """Compute WCAG contrast ratio between two hex colours."""
    l1 = _relative_luminance(hex1)
    l2 = _relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# ThemePalette structure
# ---------------------------------------------------------------------------

class TestThemePaletteStructure:
    """ThemePalette fields exist with correct types."""

    EXPECTED_FIELDS = [
        "bg", "surface", "surface_alt", "surface_hover", "dialog_bg",
        "border", "border_light", "border_strong",
        "text", "text_secondary", "text_tertiary", "text_disabled",
        "accent", "accent_text", "danger", "info",
        "grip_bg", "grip_hover",
        "badge_bg", "separator",
    ]

    def test_all_fields_exist(self):
        field_names = {f.name for f in fields(ThemePalette)}
        for name in self.EXPECTED_FIELDS:
            assert name in field_names, f"Missing field: {name}"

    def test_field_count(self):
        assert len(fields(ThemePalette)) == len(self.EXPECTED_FIELDS)

    def test_all_fields_are_str(self):
        for f in fields(ThemePalette):
            val = getattr(DARK_PALETTE, f.name)
            assert isinstance(val, str), f"DARK_PALETTE.{f.name} should be str, got {type(val)}"


# ---------------------------------------------------------------------------
# DARK_PALETTE — matches hardcoded colours in floating_panels.py
# ---------------------------------------------------------------------------

class TestDarkPalette:
    """DARK_PALETTE values match the original hardcoded colours."""

    def test_bg(self):
        assert DARK_PALETTE.bg == "#1a1a1a"

    def test_surface(self):
        assert DARK_PALETTE.surface == "#2a2a2a"

    def test_surface_alt(self):
        assert DARK_PALETTE.surface_alt == "#333333"

    def test_surface_hover(self):
        assert DARK_PALETTE.surface_hover == "#3a3a3a"

    def test_dialog_bg(self):
        assert DARK_PALETTE.dialog_bg == "#1a1a1a"

    def test_border(self):
        assert DARK_PALETTE.border == "#444444"

    def test_border_light(self):
        assert DARK_PALETTE.border_light == "#555555"

    def test_text(self):
        assert DARK_PALETTE.text == "#ffffff"

    def test_text_secondary(self):
        assert DARK_PALETTE.text_secondary == "#dddddd"

    def test_text_tertiary(self):
        assert DARK_PALETTE.text_tertiary == "#aaaaaa"

    def test_accent(self):
        assert DARK_PALETTE.accent == "#ff5545"

    def test_danger(self):
        assert DARK_PALETTE.danger == "#F44336"

    def test_info(self):
        assert DARK_PALETTE.info == "#4FC3F7"

    def test_separator(self):
        assert DARK_PALETTE.separator == "#444444"


# ---------------------------------------------------------------------------
# LIGHT_PALETTE — contrast checks (WCAG AA >= 4.5:1)
# ---------------------------------------------------------------------------

class TestLightPaletteContrast:
    """LIGHT_PALETTE text vs background luminance meets WCAG AA."""

    def test_text_vs_bg_contrast(self):
        ratio = _contrast_ratio(LIGHT_PALETTE.text, LIGHT_PALETTE.bg)
        assert ratio >= 4.5, f"text:bg contrast {ratio:.2f} < 4.5"

    def test_text_vs_surface_contrast(self):
        ratio = _contrast_ratio(LIGHT_PALETTE.text, LIGHT_PALETTE.surface)
        assert ratio >= 4.5, f"text:surface contrast {ratio:.2f} < 4.5"

    def test_text_secondary_vs_bg_contrast(self):
        ratio = _contrast_ratio(LIGHT_PALETTE.text_secondary, LIGHT_PALETTE.bg)
        assert ratio >= 4.5, f"text_secondary:bg contrast {ratio:.2f} < 4.5"

    def test_danger_vs_bg_contrast(self):
        ratio = _contrast_ratio(LIGHT_PALETTE.danger, LIGHT_PALETTE.bg)
        assert ratio >= 3.0, f"danger:bg contrast {ratio:.2f} < 3.0"

    def test_accent_vs_bg_contrast(self):
        ratio = _contrast_ratio(LIGHT_PALETTE.accent, LIGHT_PALETTE.bg)
        assert ratio >= 3.0, f"accent:bg contrast {ratio:.2f} < 3.0"


# ---------------------------------------------------------------------------
# DARK_PALETTE — contrast checks
# ---------------------------------------------------------------------------

class TestDarkPaletteContrast:
    """DARK_PALETTE text vs background luminance meets WCAG AA."""

    def test_text_vs_bg_contrast(self):
        ratio = _contrast_ratio(DARK_PALETTE.text, DARK_PALETTE.bg)
        assert ratio >= 4.5, f"text:bg contrast {ratio:.2f} < 4.5"

    def test_text_vs_surface_contrast(self):
        ratio = _contrast_ratio(DARK_PALETTE.text, DARK_PALETTE.surface)
        assert ratio >= 4.5, f"text:surface contrast {ratio:.2f} < 4.5"

    def test_text_secondary_vs_surface_contrast(self):
        ratio = _contrast_ratio(DARK_PALETTE.text_secondary, DARK_PALETTE.surface)
        assert ratio >= 4.5, f"text_secondary:surface contrast {ratio:.2f} < 4.5"


# ---------------------------------------------------------------------------
# Theme detection
# ---------------------------------------------------------------------------

class TestCurrentPalette:
    """current_palette() returns DARK_PALETTE when detection unavailable."""

    def test_returns_dark_without_qapp(self):
        """Without a running QApplication, detection falls back to dark."""
        result = current_palette()
        assert result is DARK_PALETTE

    def test_returns_theme_palette(self):
        result = current_palette()
        assert isinstance(result, ThemePalette)


class TestIsDarkMode:
    """is_dark_mode() correctness."""

    def test_true_without_qapp(self):
        """Without QApplication, falls back to dark (True)."""
        assert is_dark_mode() is True


# ---------------------------------------------------------------------------
# Stylesheet generators — each produces valid QSS with expected tokens
# ---------------------------------------------------------------------------

class TestPanelBaseCss:
    def test_contains_class_name(self):
        css = panel_base_css(DARK_PALETTE, "FloatingTranscriptPanel")
        assert "FloatingTranscriptPanel" in css

    def test_contains_bg(self):
        css = panel_base_css(DARK_PALETTE)
        assert DARK_PALETTE.bg in css

    def test_contains_border(self):
        css = panel_base_css(DARK_PALETTE)
        assert DARK_PALETTE.border in css

    def test_uses_light_palette(self):
        css = panel_base_css(LIGHT_PALETTE)
        assert LIGHT_PALETTE.bg in css


class TestTitleCss:
    def test_contains_accent(self):
        css = title_css(DARK_PALETTE)
        assert DARK_PALETTE.accent in css

    def test_contains_font_weight(self):
        css = title_css(DARK_PALETTE)
        assert "font-weight" in css


class TestHeaderButtonCss:
    @pytest.mark.parametrize("variant", ["close", "legend"])
    def test_variants_produce_qss(self, variant):
        css = header_button_css(DARK_PALETTE, variant)
        assert "QPushButton" in css
        assert "border-radius" in css

    def test_close_has_danger_hover(self):
        css = header_button_css(DARK_PALETTE, "close")
        assert DARK_PALETTE.danger in css

    def test_legend_has_accent(self):
        css = header_button_css(DARK_PALETTE, "legend")
        assert DARK_PALETTE.accent in css


class TestTabWidgetCss:
    def test_contains_pane_and_tab(self):
        css = tab_widget_css(DARK_PALETTE)
        assert "QTabWidget::pane" in css
        assert "QTabBar::tab" in css

    def test_uses_palette_tokens(self):
        css = tab_widget_css(LIGHT_PALETTE)
        assert LIGHT_PALETTE.accent in css
        assert LIGHT_PALETTE.surface in css


class TestTextAreaCss:
    def test_contains_qtextedit(self):
        css = text_area_css(DARK_PALETTE)
        assert "QTextEdit" in css
        assert "QTextBrowser" in css

    def test_uses_surface_and_text(self):
        css = text_area_css(DARK_PALETTE)
        assert DARK_PALETTE.surface in css
        assert DARK_PALETTE.text in css


class TestComboBoxCss:
    def test_default_accent(self):
        css = combo_box_css(DARK_PALETTE)
        assert DARK_PALETTE.accent in css

    def test_custom_accent(self):
        css = combo_box_css(DARK_PALETTE, "#FF0000")
        assert "#FF0000" in css


class TestActionButtonCss:
    @pytest.mark.parametrize("variant", ["scrub", "delete", "benchmark", "dialog"])
    def test_variants_produce_qss(self, variant):
        css = action_button_css(DARK_PALETTE, variant)
        assert "QPushButton" in css

    def test_scrub_has_info(self):
        css = action_button_css(DARK_PALETTE, "scrub")
        assert DARK_PALETTE.info in css

    def test_delete_has_danger(self):
        css = action_button_css(DARK_PALETTE, "delete")
        assert DARK_PALETTE.danger in css

    def test_benchmark_has_accent(self):
        css = action_button_css(DARK_PALETTE, "benchmark")
        assert DARK_PALETTE.accent in css


class TestListWidgetCss:
    def test_produces_qss(self):
        css = list_widget_css(DARK_PALETTE)
        assert "QListWidget" in css
        assert "QListWidget::item" in css


class TestProgressBarCss:
    def test_default_chunk_color(self):
        css = progress_bar_css(DARK_PALETTE)
        assert DARK_PALETTE.accent in css

    def test_custom_chunk_color(self):
        css = progress_bar_css(DARK_PALETTE, "#FF9800")
        assert "#FF9800" in css


class TestContextMenuCss:
    def test_produces_qss(self):
        css = context_menu_css(DARK_PALETTE)
        assert "QMenu" in css
        assert "QMenu::item:selected" in css

    def test_custom_accent(self):
        css = context_menu_css(DARK_PALETTE, "#FF0000")
        assert "#FF0000" in css


class TestDialogCss:
    def test_produces_qss(self):
        css = dialog_css(DARK_PALETTE)
        assert "QDialog" in css
        assert "QLabel" in css


class TestBadgeCss:
    def test_produces_qss(self):
        css = badge_css(DARK_PALETTE)
        assert "QPushButton" in css
        assert DARK_PALETTE.badge_bg in css


class TestResizeGripCss:
    def test_produces_qss(self):
        css = resize_grip_css(DARK_PALETTE)
        assert "QSizeGrip" in css
        assert DARK_PALETTE.grip_bg in css


class TestLegendOverlayCss:
    def test_returns_dict(self):
        result = legend_overlay_css(DARK_PALETTE)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = legend_overlay_css(DARK_PALETTE)
        expected = {"overlay", "title", "separator", "range_label", "desc_label"}
        assert set(result.keys()) == expected

    def test_overlay_contains_surface(self):
        result = legend_overlay_css(DARK_PALETTE)
        assert DARK_PALETTE.surface in result["overlay"]


class TestDetailHeaderCss:
    def test_produces_qss(self):
        css = detail_header_css(DARK_PALETTE)
        assert "QFrame" in css
        assert DARK_PALETTE.border in css


class TestSeparatorCss:
    def test_produces_qss(self):
        css = separator_css(DARK_PALETTE)
        assert DARK_PALETTE.separator in css


class TestInfoLabelCss:
    def test_produces_qss(self):
        css = info_label_css(DARK_PALETTE)
        assert DARK_PALETTE.text_tertiary in css


class TestStatusLabelCss:
    def test_produces_qss(self):
        css = status_label_css(DARK_PALETTE)
        assert "QLabel" in css
        assert DARK_PALETTE.text_tertiary in css


class TestSplitterCss:
    def test_produces_qss(self):
        css = splitter_css(DARK_PALETTE)
        assert "QSplitter::handle" in css
        assert DARK_PALETTE.border in css
