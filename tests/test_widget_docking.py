"""Tests for widget screen-boundary clamping and free-floating panel positioning.

Covers:
- Widget drag clamped to visible desktop area
- Settings panel opens at offset, stays free-floating
- CC overlay is free-floating
"""

from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Widget fixture — creates a minimal MeetAndReadWidget with mocked screen
# ---------------------------------------------------------------------------

class _FakeScreenGeometry:
    """Minimal stand-in for QScreen.geometry() — fixed 1920×1080 screen."""

    def __init__(self, x=0, y=0, width=1920, height=1080):
        self._x = x
        self._y = y
        self._w = width
        self._h = height

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h

    def contains(self, point):
        return self._x <= point.x() < self._x + self._w and self._y <= point.y() < self._y + self._h


@pytest.fixture
def widget(qapp):
    """Create a MeetAndReadWidget with mocked screen geometry."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    fake_screen = MagicMock()
    fake_screen.geometry.return_value = _FakeScreenGeometry(1920, 1080)
    fake_screen.availableGeometry.return_value = _FakeScreenGeometry(1920, 1080)

    with patch.object(QApplication, "primaryScreen", return_value=fake_screen), \
         patch.object(QApplication, "screens", return_value=[fake_screen]), \
         patch("meetandread.widgets.main_widget.get_config") as mock_get, \
         patch("meetandread.widgets.main_widget.save_config"):
        from meetandread.config.models import UISettings, AppSettings
        mock_get.return_value = AppSettings(ui=UISettings())

        w = MeetAndReadWidget()
        w.show()
        yield w
        w.hide()
        w.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCREEN_W = 1920
SCREEN_H = 1080


# ---------------------------------------------------------------------------
# Tests — desktop clamping
# ---------------------------------------------------------------------------

class TestDesktopClamping:
    """Widget drag should be clamped so the entire widget stays visible."""

    def test_clamp_returns_valid_position(self, widget):
        """Clamping a valid position should still produce a valid position."""
        widget.move(500, 500)
        clamped = widget._clamp_to_desktop(QPoint(500, 500))
        screens = QApplication.screens()
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        assert min_x <= clamped.x()
        assert min_y <= clamped.y()
        assert clamped.x() + widget.width() <= max_x
        assert clamped.y() + widget.height() <= max_y

    def test_clamp_prevents_left_overflow(self, widget):
        """Widget should not be allowed past the left screen edge."""
        clamped = widget._clamp_to_desktop(QPoint(-50, 500))
        assert clamped.x() >= 0

    def test_clamp_prevents_top_overflow(self, widget):
        """Widget should not be allowed past the top screen edge."""
        clamped = widget._clamp_to_desktop(QPoint(500, -50))
        assert clamped.y() >= 0

    def test_clamp_prevents_right_overflow(self, widget):
        """Widget right edge should not extend past screen right edge."""
        clamped = widget._clamp_to_desktop(QPoint(99999, 500))
        # Right edge (x + width) must not exceed screen
        screens = QApplication.screens()
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        assert clamped.x() + widget.width() <= max_x

    def test_clamp_prevents_bottom_overflow(self, widget):
        """Widget bottom edge should not extend past screen bottom edge."""
        clamped = widget._clamp_to_desktop(QPoint(500, 99999))
        screens = QApplication.screens()
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        assert clamped.y() + widget.height() <= max_y


# ---------------------------------------------------------------------------
# Tests — free-floating panels
# ---------------------------------------------------------------------------

class TestFreeFloatingPanelPositioning:
    """Panels should open at a simple offset from widget — no docking."""

    def test_settings_panel_opens_at_offset(self, widget, qapp):
        """Settings panel should appear to the right of the widget."""
        widget.move(300, 200)
        panel = widget._floating_settings_panel
        assert panel is not None

        # Toggle settings open
        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        # Panel should be visible
        assert panel.isVisible()

        # Panel should be positioned near the widget (offset or clamped to screen)
        expected_x = widget.x() + widget.width() + 10
        assert panel.x() >= widget.x() + widget.width()

        # Clean up
        panel.hide_panel()

    def test_settings_panel_not_synced_on_widget_move(self, widget, qapp):
        """After opening, moving the widget should NOT move the settings panel."""
        widget.move(300, 200)
        panel = widget._floating_settings_panel

        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        panel_pos = panel.pos()

        # Move widget
        widget.move(600, 400)
        for _ in range(5):
            qapp.processEvents()

        # Panel should NOT have moved (free-floating)
        assert panel.pos() == panel_pos

        # Clean up
        panel.hide_panel()

    def test_cc_overlay_free_floating(self, widget, qapp):
        """CC overlay should not follow widget moves."""
        overlay = widget._cc_overlay
        if overlay is None:
            pytest.skip("CC overlay not available")

        widget.move(300, 200)
        overlay.show_panel()
        for _ in range(20):
            qapp.processEvents()

        original_pos = overlay.pos()

        # Move widget
        widget.move(600, 400)
        for _ in range(5):
            qapp.processEvents()

        # Overlay should NOT have moved
        assert overlay.pos() == original_pos

        # Clean up
        overlay.hide_panel()
