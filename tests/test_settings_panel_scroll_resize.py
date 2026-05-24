"""Tests for scroll-area wrapping in FloatingSettingsPanel.

Validates that each settings tab page is wrapped in a QScrollArea with
the expected properties, and that existing widget references survive the
hierarchy change.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QScrollArea, QStackedWidget

from meetandread.widgets.floating_panels import FloatingSettingsPanel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    """Create a FloatingSettingsPanel for testing."""
    p = FloatingSettingsPanel()
    p.show()
    yield p
    p.close()


# ---------------------------------------------------------------------------
# Scroll-area contract tests
# ---------------------------------------------------------------------------

class TestScrollAreaWrapping:
    """Each tab page is wrapped in a styled QScrollArea."""

    def test_stack_has_four_pages(self, panel):
        """Content stack must still have exactly four pages."""
        assert panel._content_stack.count() == 4

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_each_page_is_scroll_area(self, panel, index):
        """Each content-stack entry must be a QScrollArea."""
        widget = panel._content_stack.widget(index)
        assert isinstance(widget, QScrollArea), (
            f"Stack widget at index {index} is {type(widget).__name__}, expected QScrollArea"
        )

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_widget_resizable_enabled(self, panel, index):
        """Each scroll area must have widgetResizable == True."""
        scroll = panel._content_stack.widget(index)
        assert isinstance(scroll, QScrollArea)
        assert scroll.widgetResizable() is True

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_scroll_area_has_non_null_content(self, panel, index):
        """Each scroll area must contain a non-null widget."""
        scroll = panel._content_stack.widget(index)
        assert isinstance(scroll, QScrollArea)
        inner = scroll.widget()
        assert inner is not None, f"Scroll area at index {index} has null widget"

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_scroll_area_object_name(self, panel, index):
        """Each scroll area must have the expected objectName."""
        scroll = panel._content_stack.widget(index)
        assert isinstance(scroll, QScrollArea)
        assert scroll.objectName() == "AethericSettingsScrollArea"

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_vertical_scrollbar_as_needed(self, panel, index):
        """Each scroll area should useScrollBarAsNeeded for vertical."""
        from PyQt6.QtCore import Qt
        scroll = panel._content_stack.widget(index)
        assert isinstance(scroll, QScrollArea)
        assert scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_horizontal_scrollbar_always_off(self, panel, index):
        """Horizontal scrollbar should be always off."""
        from PyQt6.QtCore import Qt
        scroll = panel._content_stack.widget(index)
        assert isinstance(scroll, QScrollArea)
        assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


# ---------------------------------------------------------------------------
# Existing widget reference survival tests
# ---------------------------------------------------------------------------

class TestExistingReferencesSurvive:
    """Widget references that existed before wrapping must still be valid."""

    def test_model_combo_exists(self, panel):
        assert hasattr(panel, "_model_combo") or hasattr(panel, "_live_model_combo")

    def test_benchmark_btn_exists(self, panel):
        assert hasattr(panel, "_benchmark_btn")

    def test_history_list_exists(self, panel):
        assert hasattr(panel, "_history_list")

    def test_identities_splitter_exists(self, panel):
        assert hasattr(panel, "_identities_splitter")

    def test_history_list_is_accessible(self, panel):
        """_history_list should still be a valid widget in the hierarchy."""
        assert panel._history_list is not None

    def test_identities_splitter_is_accessible(self, panel):
        """_identities_splitter should still be a valid widget in the hierarchy."""
        assert panel._identities_splitter is not None
