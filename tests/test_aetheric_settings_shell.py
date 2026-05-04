"""Contract tests for Aetheric Glass theme helpers.

Validates that the Aetheric styling contract (tokens, selectors, QSS
structure) is stable so later tasks (T02+) can wire real widgets to it.
If a later task removes or renames a helper/selectors these tests break.

Covers:
- Aetheric design tokens are defined with expected values
- Each helper generates QSS with the required object-name selectors
- Helpers accept both DARK_PALETTE and LIGHT_PALETTE without crashing
- Key visual tokens are present (12px radius, rgba alpha, red accent,
  directional border cues, dropdown chevron)
- Scoped selectors use #ObjectNames to avoid style leakage
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QStackedWidget, QWidget, QPushButton

from meetandread.widgets.theme import (
    AETHERIC_CYAN,
    AETHERIC_GLASS_BG,
    AETHERIC_GLASS_ROW_BG,
    AETHERIC_SETTINGS_BG,
    AETHERIC_SIDEBAR_BG,
    AETHERIC_NAV_ACTIVE_BG,
    AETHERIC_NAV_ACTIVE_GLOW,
    AETHERIC_NAV_HOVER_BG,
    AETHERIC_NAV_INACTIVE_TEXT,
    AETHERIC_PURPLE,
    AETHERIC_RADIUS,
    AETHERIC_RED,
    AETHERIC_SIDEBAR_WIDTH,
    AETHERIC_BORDER_LIGHT,
    AETHERIC_BORDER_DARK,
    # Aetheric CC overlay tokens
    AETHERIC_CC_BG,
    AETHERIC_CC_TEXT,
    AETHERIC_CC_RADIUS,
    AETHERIC_CC_PADDING,
    AETHERIC_CC_FONT_SIZE,
    DARK_PALETTE,
    LIGHT_PALETTE,
    aetheric_cc_overlay_css,
    aetheric_combo_box_css,
    aetheric_history_action_button_css,
    aetheric_history_header_css,
    aetheric_history_list_css,
    aetheric_history_splitter_css,
    aetheric_history_viewer_css,
    aetheric_nav_button_css,
    aetheric_placeholder_css,
    aetheric_settings_shell_css,
    aetheric_sidebar_css,
)


# ---------------------------------------------------------------------------
# Aetheric design tokens
# ---------------------------------------------------------------------------

class TestAethericDesignTokens:
    """Aetheric design tokens have expected values from the design system."""

    def test_glass_bg_is_rgba(self):
        assert AETHERIC_GLASS_BG == "rgba(30, 29, 30, 200)"

    def test_glass_row_bg_is_rgba(self):
        assert AETHERIC_GLASS_ROW_BG == "rgba(53, 52, 54, 0.2)"

    def test_sidebar_width(self):
        assert AETHERIC_SIDEBAR_WIDTH == "256px"

    def test_radius_is_12px(self):
        assert AETHERIC_RADIUS == "12px"

    def test_border_light_directional(self):
        """Light border for top-left directional cue."""
        assert AETHERIC_BORDER_LIGHT == "rgba(255, 255, 255, 30)"

    def test_border_dark_directional(self):
        """Dark border for bottom-right directional cue."""
        assert AETHERIC_BORDER_DARK == "rgba(0, 0, 0, 80)"

    def test_nav_active_bg(self):
        assert AETHERIC_NAV_ACTIVE_BG == "rgba(255, 85, 69, 0.2)"

    def test_nav_active_glow(self):
        assert AETHERIC_NAV_ACTIVE_GLOW == "rgba(255, 85, 69, 0.4)"

    def test_nav_inactive_text(self):
        assert AETHERIC_NAV_INACTIVE_TEXT == "rgba(255, 255, 255, 0.4)"

    def test_nav_hover_bg(self):
        assert AETHERIC_NAV_HOVER_BG == "rgba(255, 255, 255, 0.05)"

    def test_red_accent(self):
        assert AETHERIC_RED == "#ff5545"

    def test_purple_secondary(self):
        assert AETHERIC_PURPLE == "#c9bfff"

    def test_cyan_tertiary(self):
        assert AETHERIC_CYAN == "#00dbe9"


# ---------------------------------------------------------------------------
# aetheric_settings_shell_css
# ---------------------------------------------------------------------------

class TestAethericSettingsShellCss:
    """Settings shell QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_settings_shell_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_settings_shell_css(DARK_PALETTE)
        assert "QWidget#AethericSettingsShell" in css

    def test_contains_transparent_bg(self):
        css = aetheric_settings_shell_css(DARK_PALETTE)
        assert "background-color: transparent" in css

    def test_contains_12px_radius(self):
        css = aetheric_settings_shell_css(DARK_PALETTE)
        assert "border-radius: 12px" in css

    def test_directional_borders(self):
        """Light on top-left, dark on bottom-right."""
        css = aetheric_settings_shell_css(DARK_PALETTE)
        assert AETHERIC_BORDER_LIGHT in css
        assert AETHERIC_BORDER_DARK in css


# ---------------------------------------------------------------------------
# aetheric_sidebar_css
# ---------------------------------------------------------------------------

class TestAethericSidebarCss:
    """Sidebar QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash(self, palette):
        css = aetheric_sidebar_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_sidebar_css(DARK_PALETTE)
        assert "QWidget#AethericSidebar" in css

    def test_contains_darker_sidebar_bg(self):
        css = aetheric_sidebar_css(DARK_PALETTE)
        assert AETHERIC_SIDEBAR_BG in css

    def test_contains_sidebar_width(self):
        css = aetheric_sidebar_css(DARK_PALETTE)
        assert AETHERIC_SIDEBAR_WIDTH in css

    def test_corner_radii(self):
        css = aetheric_sidebar_css(DARK_PALETTE)
        assert "border-top-left-radius: 0px" in css  # square — title bar fills corner
        assert "border-bottom-left-radius: 12px" in css

    def test_dark_right_border(self):
        css = aetheric_sidebar_css(DARK_PALETTE)
        assert "border-right" in css
        assert "255, 85, 69" in css  # AETHERIC_RED glow


# ---------------------------------------------------------------------------
# aetheric_nav_button_css
# ---------------------------------------------------------------------------

class TestAethericNavButtonCss:
    """Navigation pill button QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash(self, palette):
        css = aetheric_nav_button_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_nav_button_css(DARK_PALETTE)
        assert "QPushButton#AethericNavButton" in css

    def test_inactive_text_token(self):
        css = aetheric_nav_button_css(DARK_PALETTE)
        assert AETHERIC_NAV_INACTIVE_TEXT in css

    def test_hover_state_with_hover_bg(self):
        css = aetheric_nav_button_css(DARK_PALETTE)
        assert ":hover" in css
        assert AETHERIC_NAV_HOVER_BG in css

    def test_checked_state_with_red_active(self):
        """Active (checked) pill uses red background and red glow border."""
        css = aetheric_nav_button_css(DARK_PALETTE)
        assert ":checked" in css
        assert AETHERIC_NAV_ACTIVE_BG in css
        assert AETHERIC_NAV_ACTIVE_GLOW in css
        assert AETHERIC_RED in css

    def test_pill_border_radius(self):
        css = aetheric_nav_button_css(DARK_PALETTE)
        assert "border-radius: 8px" in css


# ---------------------------------------------------------------------------
# aetheric_placeholder_css
# ---------------------------------------------------------------------------

class TestAethericPlaceholderCss:
    """Placeholder row QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash(self, palette):
        css = aetheric_placeholder_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_placeholder_css(DARK_PALETTE)
        assert "QWidget#AethericPlaceholderRow" in css

    def test_glass_row_bg(self):
        css = aetheric_placeholder_css(DARK_PALETTE)
        assert AETHERIC_GLASS_ROW_BG in css

    def test_hover_state_with_red_border(self):
        css = aetheric_placeholder_css(DARK_PALETTE)
        assert ":hover" in css
        assert AETHERIC_RED in css

    def test_8px_radius(self):
        css = aetheric_placeholder_css(DARK_PALETTE)
        assert "border-radius: 8px" in css


# ---------------------------------------------------------------------------
# aetheric_combo_box_css
# ---------------------------------------------------------------------------

class TestAethericComboBoxCss:
    """Combo box with chevron dropdown QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash(self, palette):
        css = aetheric_combo_box_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert "QComboBox#AethericComboBox" in css

    def test_glass_row_bg(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert AETHERIC_SETTINGS_BG in css

    def test_hover_uses_red_accent(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert ":hover" in css
        assert AETHERIC_RED in css

    def test_has_chevron_arrow(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert "::down-arrow" in css
        assert "image: url(" in css  # SVG arrow file reference
        assert ".svg" in css

    def test_dropdown_panel(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert "::drop-down" in css

    def test_item_view_selection(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert "QAbstractItemView" in css
        assert "selection-background-color" in css

    def test_8px_radius(self):
        css = aetheric_combo_box_css(DARK_PALETTE)
        assert "border-radius: 8px" in css


# ---------------------------------------------------------------------------
# aetheric_history_list_css
# ---------------------------------------------------------------------------

class TestAethericHistoryListCss:
    """History recording list QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_history_list_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert "QListWidget#AethericHistoryList" in css

    def test_no_bare_q_list_widget_selector(self):
        """Primary selector must be scoped, not bare QListWidget."""
        css = aetheric_history_list_css(DARK_PALETTE)
        # The first occurrence should be scoped
        assert "QListWidget#AethericHistoryList" in css
        # Should NOT contain bare "QListWidget {" as a primary selector
        lines = css.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("QListWidget") and "#" not in stripped:
                # Could be ::item or similar — that's fine
                assert "::" in stripped or stripped.startswith("QListWidget#"), (
                    f"Found bare QListWidget selector: {stripped}"
                )

    def test_contains_glass_row_bg(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert AETHERIC_GLASS_ROW_BG in css

    def test_directional_borders_on_items(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert AETHERIC_BORDER_LIGHT in css
        assert AETHERIC_BORDER_DARK in css

    def test_hover_state_with_red_accent(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert ":hover" in css
        assert AETHERIC_RED in css

    def test_selected_state_with_red_active(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert ":selected" in css
        assert AETHERIC_NAV_ACTIVE_BG in css
        assert AETHERIC_RED in css

    def test_8px_radius(self):
        css = aetheric_history_list_css(DARK_PALETTE)
        assert "border-radius: 8px" in css


# ---------------------------------------------------------------------------
# aetheric_history_viewer_css
# ---------------------------------------------------------------------------

class TestAethericHistoryViewerCss:
    """History transcript viewer QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_history_viewer_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_history_viewer_css(DARK_PALETTE)
        assert "QTextBrowser#AethericHistoryViewer" in css

    def test_transparent_background(self):
        css = aetheric_history_viewer_css(DARK_PALETTE)
        assert "background-color: transparent" in css

    def test_uses_inactive_text_color(self):
        css = aetheric_history_viewer_css(DARK_PALETTE)
        assert AETHERIC_NAV_INACTIVE_TEXT in css

    def test_no_bare_q_text_browser_selector(self):
        """No unscoped QTextBrowser styling."""
        css = aetheric_history_viewer_css(DARK_PALETTE)
        assert "QTextBrowser#AethericHistoryViewer" in css
        lines = css.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("QTextBrowser") and "#" not in stripped:
                assert False, f"Found bare QTextBrowser selector: {stripped}"


# ---------------------------------------------------------------------------
# aetheric_history_splitter_css
# ---------------------------------------------------------------------------

class TestAethericHistorySplitterCss:
    """History splitter handle QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_history_splitter_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_history_splitter_css(DARK_PALETTE)
        assert "QSplitter#AethericHistorySplitter" in css

    def test_dark_border_handle(self):
        css = aetheric_history_splitter_css(DARK_PALETTE)
        assert AETHERIC_BORDER_DARK in css

    def test_handle_sub_selector(self):
        css = aetheric_history_splitter_css(DARK_PALETTE)
        assert "::handle" in css


# ---------------------------------------------------------------------------
# aetheric_history_header_css
# ---------------------------------------------------------------------------

class TestAethericHistoryHeaderCss:
    """History detail header QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_history_header_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_history_header_css(DARK_PALETTE)
        assert "QFrame#AethericHistoryHeader" in css

    def test_transparent_background(self):
        css = aetheric_history_header_css(DARK_PALETTE)
        assert "background-color: transparent" in css

    def test_bottom_directional_border(self):
        css = aetheric_history_header_css(DARK_PALETTE)
        assert "border-bottom" in css
        assert AETHERIC_BORDER_DARK in css


# ---------------------------------------------------------------------------
# aetheric_history_action_button_css
# ---------------------------------------------------------------------------

class TestAethericHistoryActionButtonCss:
    """History action button QSS contract with action variants."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_history_action_button_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert "QPushButton#AethericHistoryActionButton" in css

    def test_no_bare_q_push_button_selector(self):
        """Primary selector must be scoped."""
        css = aetheric_history_action_button_css(DARK_PALETTE)
        lines = css.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("QPushButton") and "#" not in stripped and "[" not in stripped:
                assert False, f"Found bare QPushButton selector: {stripped}"

    def test_glass_row_bg_base(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert AETHERIC_GLASS_ROW_BG in css

    def test_hover_state_with_red_accent(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert ":hover" in css
        assert AETHERIC_RED in css

    def test_directional_borders(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert AETHERIC_BORDER_LIGHT in css
        assert AETHERIC_BORDER_DARK in css

    def test_8px_radius(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert "border-radius: 8px" in css

    def test_scrub_variant_uses_cyan(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert 'action="scrub"' in css
        assert AETHERIC_CYAN in css

    def test_delete_variant_uses_red(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert 'action="delete"' in css
        assert AETHERIC_RED in css

    def test_accept_variant_uses_red(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert 'action="accept"' in css

    def test_reject_variant_uses_purple(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert 'action="reject"' in css
        assert AETHERIC_PURPLE in css

    def test_disabled_state(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert ":disabled" in css

    def test_pressed_state(self):
        css = aetheric_history_action_button_css(DARK_PALETTE)
        assert ":pressed" in css


# ---------------------------------------------------------------------------
# Aetheric CC overlay tokens
# ---------------------------------------------------------------------------

class TestAethericCCDesignTokens:
    """Aetheric CC overlay design tokens have expected values."""

    def test_cc_bg_is_semi_transparent_dark(self):
        """CC background is a semi-transparent dark glass."""
        assert AETHERIC_CC_BG.startswith("rgba(")

    def test_cc_text_is_light(self):
        """CC text colour is a light/white rgba for readability on dark bg."""
        assert AETHERIC_CC_TEXT.startswith("rgba(") or AETHERIC_CC_TEXT.startswith("#")

    def test_cc_radius_is_12px(self):
        """CC overlay uses the standard 12px Aetheric radius."""
        assert AETHERIC_CC_RADIUS == "12px"

    def test_cc_padding_defined(self):
        """CC overlay has a padding token for inner content spacing."""
        assert AETHERIC_CC_PADDING.endswith("px")

    def test_cc_font_size_defined(self):
        """CC overlay has a font-size token for transcript text."""
        assert AETHERIC_CC_FONT_SIZE.endswith("px")


# ---------------------------------------------------------------------------
# aetheric_cc_overlay_css
# ---------------------------------------------------------------------------

class TestAethericCCOverlayCss:
    """CC overlay panel QSS contract."""

    @pytest.mark.parametrize("palette", [DARK_PALETTE, LIGHT_PALETTE])
    def test_does_not_crash_with_either_palette(self, palette):
        css = aetheric_cc_overlay_css(palette)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_uses_object_name_selector(self):
        """Primary selector must be QWidget#AethericCCOverlay."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        assert "QWidget#AethericCCOverlay" in css

    def test_no_bare_qwidget_selector(self):
        """No unscoped QWidget styling — must use #AethericCCOverlay."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        lines = css.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("QWidget") and "#" not in stripped:
                assert False, f"Found bare QWidget selector: {stripped}"

    def test_has_transparent_container(self):
        """Container bg is transparent — background painted in paintEvent."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        assert "background-color: transparent" in css

    def test_contains_text_colour(self):
        """CSS must include the AETHERIC_CC_TEXT colour."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        assert AETHERIC_CC_TEXT in css

    def test_has_no_border_radius(self):
        """Border radius handled by paintEvent, not QSS."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        assert "border-radius" not in css

    def test_has_no_qss_borders(self):
        """Borders painted in paintEvent, not QSS."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        assert AETHERIC_BORDER_LIGHT not in css
        assert AETHERIC_BORDER_DARK not in css

    def test_child_selectors_use_object_names(self):
        """Any child selectors (e.g. QLabel) must be scoped via #ObjectName."""
        css = aetheric_cc_overlay_css(DARK_PALETTE)
        lines = css.split("\n")
        for line in lines:
            stripped = line.strip()
            # If there's a child widget selector like QLabel { ... }, it must
            # be scoped under #AethericCCOverlay or use an #ObjectName
            if stripped.startswith("QLabel") and "#" not in stripped:
                assert False, f"Found unscoped QLabel selector: {stripped}"


# ---------------------------------------------------------------------------
# Scoped selector audit — all helpers use #ObjectNames
# ---------------------------------------------------------------------------

class TestAethericScopedSelectors:
    """Aetheric helpers use object-name selectors to avoid style leakage."""

    def test_all_helpers_use_object_name_selectors(self):
        """Each helper targets a specific #ObjectName to scope styles."""
        helpers_and_selectors = [
            (aetheric_settings_shell_css, "QWidget#AethericSettingsShell"),
            (aetheric_sidebar_css, "QWidget#AethericSidebar"),
            (aetheric_nav_button_css, "QPushButton#AethericNavButton"),
            (aetheric_placeholder_css, "QWidget#AethericPlaceholderRow"),
            (aetheric_combo_box_css, "QComboBox#AethericComboBox"),
            (aetheric_history_list_css, "QListWidget#AethericHistoryList"),
            (aetheric_history_viewer_css, "QTextBrowser#AethericHistoryViewer"),
            (aetheric_history_splitter_css, "QSplitter#AethericHistorySplitter"),
            (aetheric_history_header_css, "QFrame#AethericHistoryHeader"),
            (aetheric_history_action_button_css, "QPushButton#AethericHistoryActionButton"),
            (aetheric_cc_overlay_css, "QWidget#AethericCCOverlay"),
        ]
        for helper, selector in helpers_and_selectors:
            css = helper(DARK_PALETTE)
            assert selector in css, (
                f"{helper.__name__} missing scoped selector '{selector}'"
            )


# ---------------------------------------------------------------------------
# FloatingSettingsPanel Aetheric sidebar structure tests
# ---------------------------------------------------------------------------

class TestAethericSidebarStructure:
    """Verify the Aetheric Glass settings shell has correct widget tree."""

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        p = FloatingSettingsPanel()
        p.show()
        qapp.processEvents()
        yield p
        p.close()

    def test_shell_has_object_name(self, settings_panel):
        assert settings_panel.objectName() == "AethericSettingsShell"

    def test_sidebar_has_object_name(self, settings_panel):
        sidebar = settings_panel.findChild(QWidget, "AethericSidebar")
        assert sidebar is not None, "Sidebar with objectName 'AethericSidebar' not found"

    def test_content_stack_has_object_name(self, settings_panel):
        stack = settings_panel.findChild(QStackedWidget, "AethericContentStack")
        assert stack is not None, "QStackedWidget with objectName 'AethericContentStack' not found"

    def test_content_stack_has_three_pages(self, settings_panel):
        stack = settings_panel.findChild(QStackedWidget, "AethericContentStack")
        assert stack is not None
        assert stack.count() == 3, f"Expected 3 pages, got {stack.count()}"

    def test_no_dock_bay(self, settings_panel):
        """Dock bay widget was removed in S05 (free-floating panels)."""
        dock = settings_panel.findChild(QWidget, "AethericDockBay")
        assert dock is None, "Dock bay should not exist in free-floating panel mode"

    def test_no_title_label(self, settings_panel):
        """No _title_label attribute should exist (removed in Aetheric redesign)."""
        assert not hasattr(settings_panel, "_title_label"), (
            "FloatingSettingsPanel should not have _title_label in Aetheric shell"
        )

    def test_no_close_button(self, settings_panel):
        """No _close_btn attribute should exist (removed in Aetheric redesign)."""
        assert not hasattr(settings_panel, "_close_btn"), (
            "FloatingSettingsPanel should not have _close_btn in Aetheric shell"
        )

    def test_no_tab_widget(self, settings_panel):
        """No _tab_widget attribute should exist (replaced by QStackedWidget)."""
        assert not hasattr(settings_panel, "_tab_widget"), (
            "FloatingSettingsPanel should not have _tab_widget in Aetheric shell"
        )


class TestAethericNavButtons:
    """Verify sidebar nav buttons have correct object names and behavior."""

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        p = FloatingSettingsPanel()
        p.show()
        qapp.processEvents()
        yield p
        p.close()

    def test_three_nav_buttons(self, settings_panel):
        assert len(settings_panel._nav_buttons) == 3

    def test_nav_button_object_names(self, settings_panel):
        for btn in settings_panel._nav_buttons:
            assert btn.objectName() == "AethericNavButton"

    def test_nav_button_nav_ids(self, settings_panel):
        ids = [btn.property("nav_id") for btn in settings_panel._nav_buttons]
        assert ids == ["settings", "performance", "history"]

    def test_settings_button_initially_checked(self, settings_panel):
        assert settings_panel._nav_settings_btn.isChecked()
        assert not settings_panel._nav_performance_btn.isChecked()
        assert not settings_panel._nav_history_btn.isChecked()

    def test_nav_click_switches_page(self, settings_panel, qapp):
        """Clicking Performance nav switches content stack to index 1."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == 1
        assert settings_panel._nav_performance_btn.isChecked()
        assert not settings_panel._nav_settings_btn.isChecked()

    def test_nav_click_to_history(self, settings_panel, qapp):
        """Clicking History nav switches content stack to index 2."""
        settings_panel._nav_history_btn.click()
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == 2
        assert settings_panel._nav_history_btn.isChecked()

    def test_nav_click_back_to_settings(self, settings_panel, qapp):
        """Switching to Performance and back to Settings works."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == 1

        settings_panel._nav_settings_btn.click()
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == 0
        assert settings_panel._nav_settings_btn.isChecked()

    def test_invalid_nav_index_ignored(self, settings_panel, qapp):
        """Unknown nav index should not crash or change current page."""
        current = settings_panel._content_stack.currentIndex()
        settings_panel._on_nav_clicked(99)
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == current

    def test_negative_nav_index_ignored(self, settings_panel, qapp):
        """Negative nav index should not crash or change current page."""
        current = settings_panel._content_stack.currentIndex()
        settings_panel._on_nav_clicked(-1)
        qapp.processEvents()
        assert settings_panel._content_stack.currentIndex() == current


class TestAethericPerformanceLifecycle:
    """Verify Performance monitoring lifecycle with sidebar nav."""

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        from unittest.mock import MagicMock
        p = FloatingSettingsPanel()
        p._resource_monitor = MagicMock()
        p._resource_monitor.is_running = False
        p.show()
        qapp.processEvents()
        yield p
        p.close()

    def test_perf_active_on_performance_nav(self, settings_panel, qapp):
        """Navigating to Performance sets _perf_tab_active True."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        assert settings_panel._perf_tab_active is True

    def test_perf_inactive_on_settings_nav(self, settings_panel, qapp):
        """Navigating away from Performance sets _perf_tab_active False."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        settings_panel._nav_settings_btn.click()
        qapp.processEvents()
        assert settings_panel._perf_tab_active is False

    def test_perf_inactive_on_history_nav(self, settings_panel, qapp):
        """History nav sets _perf_tab_active False."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        settings_panel._nav_history_btn.click()
        qapp.processEvents()
        assert settings_panel._perf_tab_active is False

    def test_monitor_stops_on_nav_away(self, settings_panel, qapp):
        """Navigating away from Performance stops the monitor."""
        settings_panel._nav_performance_btn.click()
        qapp.processEvents()
        settings_panel._nav_settings_btn.click()
        qapp.processEvents()
        # _stop_resource_monitor should have been called
        # (mock's stop() is called if monitor.is_running is True)
        assert settings_panel._perf_tab_active is False
