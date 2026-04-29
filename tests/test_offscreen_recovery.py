"""Tests for off-screen widget recovery on startup.

Covers CAP-86e76926: widget can get lost off screen and cannot be recovered.
_verify that _recover_offscreen_position detects off-screen positions and
recovers to the center of the primary monitor.
"""

from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtCore import QPoint, QRect
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
# Helpers
# ---------------------------------------------------------------------------

class _FakeGeometry(QRect):
    """QRect-based geometry for deterministic screen testing."""

    def __init__(self, x=0, y=0, width=1920, height=1080):
        super().__init__(x, y, width, height)


@pytest.fixture
def widget(qapp):
    """Create a MeetAndReadWidget with mocked screen geometry."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    primary_screen = MagicMock()
    primary_screen.geometry.return_value = _FakeGeometry(0, 0, 1920, 1080)

    with patch.object(QApplication, "primaryScreen", return_value=primary_screen), \
         patch.object(QApplication, "screens", return_value=[primary_screen]), \
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
# Tests — on-screen positions should not trigger recovery
# ---------------------------------------------------------------------------

class TestOnScreenPositions:
    """Widget center is within a screen — no recovery needed."""

    def test_center_of_primary_screen(self, widget):
        """Widget at center of primary screen stays put."""
        widget.move(960, 540)
        widget._recover_offscreen_position()
        assert widget.pos().x() == 960
        assert widget.pos().y() == 540

    def test_top_left_corner(self, widget):
        """Widget at (0, 0) stays put."""
        widget.move(0, 0)
        widget._recover_offscreen_position()
        assert widget.pos().x() == 0
        assert widget.pos().y() == 0

    def test_near_right_edge(self, widget):
        """Widget just inside right edge stays put."""
        widget.move(1920 - widget.width() - 1, 500)
        widget._recover_offscreen_position()
        assert widget.pos().x() == 1920 - widget.width() - 1

    def test_near_bottom_edge(self, widget):
        """Widget just inside bottom edge stays put."""
        widget.move(500, 1080 - widget.height() - 1)
        widget._recover_offscreen_position()
        assert widget.pos().y() == 1080 - widget.height() - 1


# ---------------------------------------------------------------------------
# Tests — off-screen positions should trigger recovery
# ---------------------------------------------------------------------------

class TestOffScreenRecovery:
    """Widget center is outside all screens — recovers to primary center."""

    def test_far_off_right(self, widget):
        """Widget at x=5000 recovers to primary center."""
        widget.move(5000, 500)
        widget._recover_offscreen_position()
        primary = QApplication.primaryScreen().geometry()
        expected_x = primary.x() + (primary.width() - widget.width()) // 2
        expected_y = primary.y() + (primary.height() - widget.height()) // 2
        assert widget.pos().x() == expected_x
        assert widget.pos().y() == expected_y

    def test_far_off_bottom(self, widget):
        """Widget at y=5000 recovers to primary center."""
        widget.move(500, 5000)
        widget._recover_offscreen_position()
        primary = QApplication.primaryScreen().geometry()
        expected_x = primary.x() + (primary.width() - widget.width()) // 2
        expected_y = primary.y() + (primary.height() - widget.height()) // 2
        assert widget.pos().x() == expected_x
        assert widget.pos().y() == expected_y

    def test_negative_position(self, widget):
        """Widget at (-500, -500) recovers to primary center."""
        widget.move(-500, -500)
        widget._recover_offscreen_position()
        primary = QApplication.primaryScreen().geometry()
        expected_x = primary.x() + (primary.width() - widget.width()) // 2
        expected_y = primary.y() + (primary.height() - widget.height()) // 2
        assert widget.pos().x() == expected_x
        assert widget.pos().y() == expected_y

    def test_top_left_off_but_center_on(self, widget):
        """Widget at (-50, -50) with center still on-screen stays put.

        The recovery checks center, not top-left. A widget at (-50, -50)
        that's ~200x120 has its center at (~50, ~10) — still on screen.
        """
        widget.move(-50, -50)
        widget._recover_offscreen_position()
        # Center is at (-50 + width//2, -50 + height//2)
        # If center is on screen, no recovery
        center_x = -50 + widget.width() // 2
        center_y = -50 + widget.height() // 2
        primary = QApplication.primaryScreen().geometry()
        center_on_screen = primary.contains(QPoint(center_x, center_y))
        if center_on_screen:
            # Should NOT have moved
            assert widget.pos().x() == -50
            assert widget.pos().y() == -50
        else:
            # Would recover — but this shouldn't happen with typical widget sizes
            expected_x = primary.x() + (primary.width() - widget.width()) // 2
            assert widget.pos().x() == expected_x


# ---------------------------------------------------------------------------
# Tests — multi-monitor recovery
# ---------------------------------------------------------------------------

class TestMultiMonitorRecovery:
    """Widget on disconnected second monitor recovers to primary."""

    def test_disconnected_second_monitor(self, qapp):
        """Widget saved on second monitor that no longer exists recovers."""
        from meetandread.widgets.main_widget import MeetAndReadWidget

        primary_screen = MagicMock()
        primary_screen.geometry.return_value = _FakeGeometry(0, 0, 1920, 1080)

        # Simulate single-monitor setup (second monitor disconnected)
        with patch.object(QApplication, "primaryScreen", return_value=primary_screen), \
             patch.object(QApplication, "screens", return_value=[primary_screen]), \
             patch("meetandread.widgets.main_widget.get_config") as mock_get, \
             patch("meetandread.widgets.main_widget.save_config"):
            from meetandread.config.models import UISettings, AppSettings
            mock_get.return_value = AppSettings(ui=UISettings())

            w = MeetAndReadWidget()
            w.show()

            # Position where second monitor used to be
            w.move(2500, 500)
            w._recover_offscreen_position()

            # Should recover to primary center
            primary = QApplication.primaryScreen().geometry()
            expected_x = primary.x() + (primary.width() - w.width()) // 2
            expected_y = primary.y() + (primary.height() - w.height()) // 2
            assert w.pos().x() == expected_x
            assert w.pos().y() == expected_y

            w.hide()
            w.close()

    def test_dual_monitor_widget_stays_on_secondary(self, qapp):
        """Widget on an active second monitor is NOT recovered."""
        from meetandread.widgets.main_widget import MeetAndReadWidget

        primary_screen = MagicMock()
        primary_screen.geometry.return_value = _FakeGeometry(0, 0, 1920, 1080)

        secondary_screen = MagicMock()
        secondary_screen.geometry.return_value = _FakeGeometry(1920, 0, 1920, 1080)

        with patch.object(QApplication, "primaryScreen", return_value=primary_screen), \
             patch.object(QApplication, "screens", return_value=[primary_screen, secondary_screen]), \
             patch("meetandread.widgets.main_widget.get_config") as mock_get, \
             patch("meetandread.widgets.main_widget.save_config"):
            from meetandread.config.models import UISettings, AppSettings
            mock_get.return_value = AppSettings(ui=UISettings())

            w = MeetAndReadWidget()
            w.show()

            # Position on second monitor
            w.move(2400, 500)
            w._recover_offscreen_position()

            # Should NOT have moved — center is on secondary screen
            assert w.pos().x() == 2400
            assert w.pos().y() == 500

            w.hide()
            w.close()
