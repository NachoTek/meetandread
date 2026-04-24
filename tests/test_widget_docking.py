"""Tests for widget docking, animation, and config persistence.

Covers T03 must-haves:
- Left/right-only snap detection (top returns None)
- Peek position math (1/5th visible when docked)
- Slide animation convergence over ~9 ticks
- Config save/load round-trip for dock_edge
- Live magnet snap during drag
"""

import time as _time
from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped, same pattern as test_transcript_management.py)
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

    def __init__(self, width=1920, height=1080):
        self._w = width
        self._h = height

    def width(self):
        return self._w

    def height(self):
        return self._h

    def contains(self, point):
        return 0 <= point.x() < self._w and 0 <= point.y() < self._h


@pytest.fixture
def widget(qapp):
    """Create a MeetAndReadWidget with mocked screen geometry.

    QApplication.primaryScreen() is patched so tests are deterministic
    regardless of the host machine.  Config save/load is mocked to avoid
    cross-test contamination via real config file.
    """
    from metamemory.widgets.main_widget import MeetAndReadWidget

    fake_screen = MagicMock()
    fake_screen.geometry.return_value = _FakeScreenGeometry(1920, 1080)
    fake_screen.availableGeometry.return_value = _FakeScreenGeometry(1920, 1080)

    with patch.object(QApplication, "primaryScreen", return_value=fake_screen), \
         patch.object(QApplication, "screens", return_value=[fake_screen]), \
         patch("metamemory.widgets.main_widget.get_config") as mock_get, \
         patch("metamemory.widgets.main_widget.save_config"):
        # Return default settings so widget doesn't pick up stale state
        from metamemory.config.models import UISettings, AppSettings
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
WIDGET_W = 200
WIDGET_H = 120
SNAP_THRESHOLD = 20


def _move_to(widget, x, y):
    """Move widget and fully reset docking/animation state."""
    widget.move(x, y)
    widget.is_docked = False
    widget.dock_edge = None
    widget.is_dragging = False
    widget._slide_state.active = False


def _advance_animations(widget, ticks=15, interval_ms=33):
    """Advance _update_animations for *ticks* frames, simulating real time.

    Patches ``_time.monotonic`` so the animation engine sees elapsed time
    increasing by *interval_ms* per tick (matching the 33ms animation timer).
    """
    base_monotonic = _time.monotonic
    call_count = 0

    def fake_monotonic():
        nonlocal call_count
        call_count += 1
        # Each call to monotonic from _update_animations returns base + accumulated ticks
        return base_monotonic() + (call_count * interval_ms / 1000.0)

    with patch("metamemory.widgets.main_widget._time.monotonic", fake_monotonic):
        for _ in range(ticks):
            widget._update_animations()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSnapDetectionLeftRightOnly:
    """_check_snap_to_edge should only detect left and right edges."""

    def test_snap_left(self, widget):
        """Widget near left edge should dock to 'left'."""
        _move_to(widget, 10, 500)
        widget._check_snap_to_edge()
        assert widget.dock_edge == "left"
        assert widget.is_docked is True

    def test_snap_right(self, widget):
        """Widget near right edge should dock to 'right'."""
        _move_to(widget, SCREEN_W - WIDGET_W - 10, 500)
        widget._check_snap_to_edge()
        assert widget.dock_edge == "right"
        assert widget.is_docked is True

    def test_top_returns_none(self, widget):
        """Widget near top edge should NOT dock (top/bottom removed)."""
        _move_to(widget, 500, 5)
        widget._check_snap_to_edge()
        assert widget.dock_edge is None
        assert widget.is_docked is False

    def test_bottom_returns_none(self, widget):
        """Widget near bottom edge should NOT dock (top/bottom removed)."""
        _move_to(widget, 500, SCREEN_H - WIDGET_H - 5)
        widget._check_snap_to_edge()
        assert widget.dock_edge is None
        assert widget.is_docked is False

    def test_center_no_snap(self, widget):
        """Widget in center should not dock."""
        _move_to(widget, 500, 500)
        widget._check_snap_to_edge()
        assert widget.dock_edge is None
        assert widget.is_docked is False


class TestPeekPositionCalculation:
    """When docked, only 1/5th of widget width should be visible."""

    def test_peek_width_value(self, widget):
        """_peek_width should be 1/5th of widget width (40px for 200px widget)."""
        assert widget.width() == WIDGET_W
        assert widget._peek_width == 40  # int(200 * 0.2)

    def test_dock_left_peek_position(self, widget):
        """Docked left: x should be -(width - peek_width) = -160."""
        _move_to(widget, 10, 500)
        widget.dock_edge = "left"
        widget.is_docked = True
        widget._update_docked_state()

        # Animation is running — advance to completion with simulated time
        _advance_animations(widget, ticks=12)
        assert widget.pos().x() == -(WIDGET_W - 40)  # -160

    def test_dock_right_peek_position(self, widget):
        """Docked right: x should be screen_width - peek_width = 1880."""
        _move_to(widget, SCREEN_W - WIDGET_W - 10, 500)
        widget.dock_edge = "right"
        widget.is_docked = True
        widget._update_docked_state()

        _advance_animations(widget, ticks=12)
        assert widget.pos().x() == SCREEN_W - 40  # 1880


class TestSlideAnimationInterpolation:
    """Slide animation should converge to target with ease-out deceleration."""

    def test_converges_to_target(self, widget):
        """After ~9 ticks (~300ms), widget position should reach target."""
        _move_to(widget, 500, 500)
        target = QPoint(100, 500)
        widget._start_slide_to(target)

        # Advance 10 ticks (~330ms at 33ms/frame) with simulated time
        _advance_animations(widget, ticks=10)

        assert not widget._slide_state.active, "Animation should be finished"
        assert widget.pos().x() == target.x()
        assert widget.pos().y() == target.y()

    def test_ease_out_deceleration(self, widget):
        """Position should move more in early steps than later steps (ease-out)."""
        _move_to(widget, 500, 500)
        target = QPoint(100, 500)

        # Manually advance with simulated time, capturing positions per tick
        base_monotonic = _time.monotonic
        call_count = 0

        def fake_monotonic():
            nonlocal call_count
            call_count += 1
            return base_monotonic() + (call_count * 33 / 1000.0)

        widget._start_slide_to(target)
        positions = []
        with patch("metamemory.widgets.main_widget._time.monotonic", fake_monotonic):
            for _ in range(9):
                widget._update_animations()
                positions.append(widget.pos().x())

        # Calculate deltas between consecutive positions
        deltas = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]

        # Early deltas should be larger than later deltas (ease-out property)
        mid = len(deltas) // 2
        early_avg = sum(deltas[:mid]) / max(mid, 1)
        late_avg = sum(deltas[mid:]) / max(len(deltas) - mid, 1)

        assert early_avg > late_avg, (
            f"Ease-out expected: early avg {early_avg:.1f} should be > late avg {late_avg:.1f}"
        )


class TestConfigPersistenceRoundTrip:
    """dock_edge should survive config save/load cycle."""

    def test_save_and_restore_dock_left(self, widget):
        """Set dock_edge='left', save, reload config, assert dock_edge='left'."""
        from metamemory.config import get_config, set_config, save_config, AppSettings
        from metamemory.config.models import UISettings

        with patch("metamemory.widgets.main_widget.save_config"):
            widget.dock_edge = "left"
            widget._save_position()

        # Read back via get_config (uses real config backend)
        settings = get_config()
        assert settings.ui.widget_dock_edge == "left"

        # Clean up — restore None for subsequent tests
        widget.dock_edge = None
        with patch("metamemory.widgets.main_widget.save_config"):
            widget._save_position()

    def test_save_and_restore_undocked(self, widget):
        """Set dock_edge=None, save, reload config, assert dock_edge=None."""
        from metamemory.config import get_config

        with patch("metamemory.widgets.main_widget.save_config"):
            widget.dock_edge = None
            widget._save_position()

        settings = get_config()
        assert settings.ui.widget_dock_edge is None

    def test_save_and_restore_dock_right(self, widget):
        """Set dock_edge='right', save, reload, assert dock_edge='right'."""
        from metamemory.config import get_config

        with patch("metamemory.widgets.main_widget.save_config"):
            widget.dock_edge = "right"
            widget._save_position()

        settings = get_config()
        assert settings.ui.widget_dock_edge == "right"

        # Clean up
        widget.dock_edge = None
        with patch("metamemory.widgets.main_widget.save_config"):
            widget._save_position()


class TestLiveMagnetSnapDuringDrag:
    """Live magnet snap should dock/undock while dragging."""

    def test_magnet_snap_to_left_edge(self, widget):
        """Dragging near left edge should trigger magnet snap."""
        _move_to(widget, 500, 500)

        # Simulate drag state
        widget.is_dragging = True
        widget.drag_start_pos = QPoint(500, 500)
        widget.widget_start_pos = QPoint(500, 500)

        # Compute position near left edge
        new_pos = QPoint(10, 500)  # within 20px of left edge
        widget._apply_drag_position(new_pos)

        # Advance animation to let snap complete
        _advance_animations(widget, ticks=12)

        assert widget.is_docked is True
        assert widget.dock_edge == "left"

    def test_magnet_snap_to_right_edge(self, widget):
        """Dragging near right edge should trigger magnet snap."""
        _move_to(widget, 500, 500)

        widget.is_dragging = True
        widget.drag_start_pos = QPoint(500, 500)
        widget.widget_start_pos = QPoint(500, 500)

        new_pos = QPoint(SCREEN_W - WIDGET_W - 10, 500)
        widget._apply_drag_position(new_pos)

        _advance_animations(widget, ticks=12)

        assert widget.is_docked is True
        assert widget.dock_edge == "right"

    def test_magnet_unsnap_away_from_edge(self, widget):
        """Moving away from edge during drag should unsnap."""
        # First snap to left
        _move_to(widget, 10, 500)
        widget.is_dragging = True
        widget.drag_start_pos = QPoint(10, 500)
        widget.widget_start_pos = QPoint(10, 500)
        widget.is_docked = True
        widget.dock_edge = "left"

        # Now move to center — should unsnap immediately
        center_pos = QPoint(500, 500)
        widget._apply_drag_position(center_pos)

        assert widget.is_docked is False
        assert widget.dock_edge is None
        assert widget.pos().x() == 500

    def test_drag_edge_snap_check_method(self, widget):
        """_check_drag_edge_snap returns correct (should_snap, edge) tuples."""
        # Near left edge
        snap, edge = widget._check_drag_edge_snap(QPoint(5, 500))
        assert snap is True
        assert edge == "left"

        # Near right edge
        snap, edge = widget._check_drag_edge_snap(QPoint(SCREEN_W - WIDGET_W - 5, 500))
        assert snap is True
        assert edge == "right"

        # Center — no snap
        snap, edge = widget._check_drag_edge_snap(QPoint(500, 500))
        assert snap is False
        assert edge is None

        # Near top — no snap (top removed)
        snap, edge = widget._check_drag_edge_snap(QPoint(500, 5))
        assert snap is False
        assert edge is None
