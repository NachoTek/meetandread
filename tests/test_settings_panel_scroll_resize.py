"""Tests for scroll-area wrapping and edge resize in FloatingSettingsPanel.

Validates that each settings tab page is wrapped in a QScrollArea with
the expected properties, existing widget references survive the hierarchy
change, and the edge-resize mechanism works correctly with proper cursor
mapping, minimum-size clamping, state management, and bottom-right
square-corner paint geometry.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QScrollArea, QStackedWidget, QSizeGrip

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


# ---------------------------------------------------------------------------
# Edge-resize detection tests
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    """_get_edge_at_cursor returns the correct edge/corner string."""

    @pytest.mark.parametrize("edge_pos,expected", [
        # Four corners (corners before edges)
        (QPoint(1, 1), "top-left"),
        (QPoint(-1, -1), "top-left"),  # outside top-left still triggers
        (QPoint(7, 3), "top-left"),
    ])
    def test_corner_detection(self, panel, edge_pos, expected):
        result = panel._get_edge_at_cursor(edge_pos)
        assert result == expected

    def test_top_left_corner(self, panel):
        assert panel._get_edge_at_cursor(QPoint(0, 0)) == "top-left"

    def test_top_right_corner(self, panel):
        w = panel.width()
        assert panel._get_edge_at_cursor(QPoint(w - 1, 0)) == "top-right"

    def test_bottom_left_corner(self, panel):
        h = panel.height()
        assert panel._get_edge_at_cursor(QPoint(0, h - 1)) == "bottom-left"

    def test_bottom_right_corner(self, panel):
        w, h = panel.width(), panel.height()
        assert panel._get_edge_at_cursor(QPoint(w - 1, h - 1)) == "bottom-right"

    def test_left_edge(self, panel):
        assert panel._get_edge_at_cursor(QPoint(3, panel.height() // 2)) == "left"

    def test_right_edge(self, panel):
        w = panel.width()
        assert panel._get_edge_at_cursor(QPoint(w - 3, panel.height() // 2)) == "right"

    def test_top_edge(self, panel):
        assert panel._get_edge_at_cursor(QPoint(panel.width() // 2, 2)) == "top"

    def test_bottom_edge(self, panel):
        h = panel.height()
        assert panel._get_edge_at_cursor(QPoint(panel.width() // 2, h - 2)) == "bottom"

    def test_center_returns_none(self, panel):
        """Positions well inside the panel should return None."""
        cx, cy = panel.width() // 2, panel.height() // 2
        assert panel._get_edge_at_cursor(QPoint(cx, cy)) is None

    def test_just_outside_threshold_returns_none(self, panel):
        """Positions just beyond the threshold should return None."""
        t = panel._resize_edge_threshold
        cx = panel.width() // 2
        cy = panel.height() // 2
        # Beyond left edge
        assert panel._get_edge_at_cursor(QPoint(t + 2, cy)) is None
        # Beyond right edge
        assert panel._get_edge_at_cursor(QPoint(panel.width() - t - 2, cy)) is None
        # Beyond top edge
        assert panel._get_edge_at_cursor(QPoint(cx, t + 2)) is None
        # Beyond bottom edge
        assert panel._get_edge_at_cursor(QPoint(cx, panel.height() - t - 2)) is None


# ---------------------------------------------------------------------------
# Cursor mapping tests
# ---------------------------------------------------------------------------

class TestCursorMapping:
    """_cursor_for_resize_edge returns the correct Qt cursor shape."""

    @pytest.mark.parametrize("edge,expected_cursor", [
        ("left", Qt.CursorShape.SizeHorCursor),
        ("right", Qt.CursorShape.SizeHorCursor),
        ("top", Qt.CursorShape.SizeVerCursor),
        ("bottom", Qt.CursorShape.SizeVerCursor),
        ("top-left", Qt.CursorShape.SizeFDiagCursor),
        ("bottom-right", Qt.CursorShape.SizeFDiagCursor),
        ("top-right", Qt.CursorShape.SizeBDiagCursor),
        ("bottom-left", Qt.CursorShape.SizeBDiagCursor),
    ])
    def test_edge_cursor(self, panel, edge, expected_cursor):
        assert panel._cursor_for_resize_edge(edge) == expected_cursor

    def test_none_returns_arrow(self, panel):
        assert panel._cursor_for_resize_edge(None) == Qt.CursorShape.ArrowCursor

    def test_unknown_returns_arrow(self, panel):
        assert panel._cursor_for_resize_edge("unknown") == Qt.CursorShape.ArrowCursor


# ---------------------------------------------------------------------------
# Edge-resize state and clamping tests
# ---------------------------------------------------------------------------

def _make_mouse_event(local_pos: QPoint, button=Qt.MouseButton.LeftButton,
                      global_pos: QPoint = None) -> QMouseEvent:
    """Create a QMouseEvent for testing mouse interactions."""
    if global_pos is None:
        global_pos = local_pos
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(local_pos),
        QPointF(global_pos),
        button,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _send_edge_press(panel, local_pos: QPoint, global_pos: QPoint = None) -> None:
    """Send a mouse press via the panel's eventFilter (simulates child-widget event)."""
    if global_pos is None:
        global_pos = panel.mapToGlobal(local_pos)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(local_pos), QPointF(global_pos),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    # eventFilter intercepts events targeted at child widgets
    panel.eventFilter(panel, event)


def _send_edge_move(panel, local_pos: QPoint, global_pos: QPoint = None) -> None:
    """Send a mouse move via the panel's eventFilter."""
    if global_pos is None:
        global_pos = panel.mapToGlobal(local_pos)
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(local_pos), QPointF(global_pos),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.eventFilter(panel, event)


def _send_edge_release(panel, local_pos: QPoint, global_pos: QPoint = None) -> None:
    """Send a mouse release via the panel's eventFilter."""
    if global_pos is None:
        global_pos = panel.mapToGlobal(local_pos)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(local_pos), QPointF(global_pos),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.eventFilter(panel, event)


class TestEdgeResizeState:
    """Edge resize starts, runs, and clears state correctly."""

    def test_resize_state_initialized(self, panel):
        """Edge resize state must be False/None at init."""
        assert panel._edge_resizing is False
        assert panel._resize_edge is None
        assert panel._resize_start_pos is None
        assert panel._resize_start_geometry is None
        assert panel._resize_edge_threshold == 8

    def test_mouse_tracking_enabled(self, panel):
        """Mouse tracking must be enabled for hover cursor updates."""
        assert panel.hasMouseTracking() is True

    def test_resize_grip_still_present(self, panel):
        """TexturedSizeGrip must still exist as a child."""
        grip = panel.findChild(QSizeGrip)
        assert grip is not None
        assert grip.isVisible()

    def test_resize_grip_positioned_after_resize(self, panel, qapp):
        """Grip must be at bottom-right after programmatic resize."""
        panel.resize(600, 500)
        qapp.processEvents()
        grip = panel.findChild(QSizeGrip)
        assert grip is not None
        assert grip.x() == panel.width() - grip.width()
        assert grip.y() == panel.height() - grip.height()

    def test_mouse_press_starts_edge_resize(self, panel, qapp):
        """Left-button press on an edge should start resize state."""
        edge_pos = QPoint(2, panel.height() // 2)  # left edge
        global_pos = panel.mapToGlobal(edge_pos)
        _send_edge_press(panel, edge_pos, global_pos)
        assert panel._edge_resizing is True
        assert panel._resize_edge == "left"
        assert panel._resize_start_pos == global_pos
        assert panel._resize_start_geometry is not None
        # Clean up
        _send_edge_release(panel, edge_pos, global_pos)

    def test_mouse_press_center_does_not_start_resize(self, panel):
        """Left-button press at center should not start resize."""
        center = QPoint(panel.width() // 2, panel.height() // 2)
        global_center = panel.mapToGlobal(center)
        _send_edge_press(panel, center, global_center)
        assert panel._edge_resizing is False

    def test_release_clears_resize_state(self, panel, qapp):
        """Mouse release should clear all resize state."""
        edge_pos = QPoint(2, panel.height() // 2)
        global_pos = panel.mapToGlobal(edge_pos)
        _send_edge_press(panel, edge_pos, global_pos)
        assert panel._edge_resizing is True

        _send_edge_release(panel, edge_pos, global_pos)
        assert panel._edge_resizing is False
        assert panel._resize_edge is None
        assert panel._resize_start_pos is None
        assert panel._resize_start_geometry is None

    def test_release_clears_state_even_after_active_resize(self, panel, qapp):
        """Release should clear state even after a resize drag has happened."""
        # Start resize from right edge
        edge_pos = QPoint(panel.width() - 2, panel.height() // 2)
        global_pos = panel.mapToGlobal(edge_pos)
        _send_edge_press(panel, edge_pos, global_pos)
        assert panel._edge_resizing is True

        # Simulate drag
        move_global = global_pos + QPoint(20, 0)
        _send_edge_move(panel, panel.mapFromGlobal(move_global), move_global)

        # Release
        _send_edge_release(panel, panel.mapFromGlobal(move_global), move_global)
        assert panel._edge_resizing is False
        assert panel._resize_edge is None


class TestResizeClamping:
    """Resize from edges should clamp to minimum size."""

    def test_left_resize_clamps_to_min_width(self, panel, qapp):
        """Dragging left edge past minimum width should clamp."""
        original_w = panel.width()
        min_w = panel.minimumWidth()
        edge_pos = QPoint(2, panel.height() // 2)
        global_pos = panel.mapToGlobal(edge_pos)

        # Start resize
        _send_edge_press(panel, edge_pos, global_pos)

        # Drag far to the right (trying to shrink width below minimum)
        move_global = global_pos + QPoint(original_w, 0)
        _send_edge_move(panel, panel.mapFromGlobal(move_global), move_global)
        qapp.processEvents()

        assert panel.width() >= min_w

        # Release
        _send_edge_release(panel, panel.mapFromGlobal(move_global), move_global)

    def test_top_resize_clamps_to_min_height(self, panel, qapp):
        """Dragging top edge past minimum height should clamp."""
        original_h = panel.height()
        min_h = panel.minimumHeight()
        edge_pos = QPoint(panel.width() // 2, 2)
        global_pos = panel.mapToGlobal(edge_pos)

        _send_edge_press(panel, edge_pos, global_pos)

        # Drag far downward (trying to shrink height below minimum)
        move_global = global_pos + QPoint(0, original_h)
        _send_edge_move(panel, panel.mapFromGlobal(move_global), move_global)
        qapp.processEvents()

        assert panel.height() >= min_h

        _send_edge_release(panel, panel.mapFromGlobal(move_global), move_global)

    def test_right_resize_expands(self, panel, qapp):
        """Dragging right edge should expand width."""
        original_w = panel.width()
        edge_pos = QPoint(panel.width() - 2, panel.height() // 2)
        global_pos = panel.mapToGlobal(edge_pos)

        _send_edge_press(panel, edge_pos, global_pos)

        move_global = global_pos + QPoint(50, 0)
        _send_edge_move(panel, panel.mapFromGlobal(move_global), move_global)
        qapp.processEvents()

        assert panel.width() > original_w

        _send_edge_release(panel, panel.mapFromGlobal(move_global), move_global)

    def test_bottom_resize_expands(self, panel, qapp):
        """Dragging bottom edge should expand height."""
        original_h = panel.height()
        edge_pos = QPoint(panel.width() // 2, panel.height() - 2)
        global_pos = panel.mapToGlobal(edge_pos)

        _send_edge_press(panel, edge_pos, global_pos)

        move_global = global_pos + QPoint(0, 50)
        _send_edge_move(panel, panel.mapFromGlobal(move_global), move_global)
        qapp.processEvents()

        assert panel.height() > original_h

        _send_edge_release(panel, panel.mapFromGlobal(move_global), move_global)


# ---------------------------------------------------------------------------
# Bottom-right corner rect tests
# ---------------------------------------------------------------------------

class TestBottomRightCornerRect:
    """_bottom_right_corner_rect returns deterministic geometry."""

    def test_rect_at_initial_size(self, panel):
        """Corner rect should be at bottom-right of the panel."""
        rect = panel._bottom_right_corner_rect()
        assert isinstance(rect, QRect)
        from meetandread.widgets.theme import AETHERIC_RADIUS
        radius = int(AETHERIC_RADIUS.replace("px", ""))
        assert rect.x() == panel.width() - radius
        assert rect.y() == panel.height() - radius
        assert rect.width() == radius
        assert rect.height() == radius

    def test_rect_after_resize(self, panel, qapp):
        """Corner rect should update after panel resize."""
        panel.resize(700, 550)
        qapp.processEvents()
        rect = panel._bottom_right_corner_rect()
        from meetandread.widgets.theme import AETHERIC_RADIUS
        radius = int(AETHERIC_RADIUS.replace("px", ""))
        assert rect.x() == panel.width() - radius
        assert rect.y() == panel.height() - radius
        assert rect.width() == radius
        assert rect.height() == radius

    def test_rect_size_matches_radius_token(self, panel):
        """Rect size should match AETHERIC_RADIUS parsed to int."""
        from meetandread.widgets.theme import AETHERIC_RADIUS
        radius = int(AETHERIC_RADIUS.replace("px", ""))
        rect = panel._bottom_right_corner_rect()
        assert rect.width() == radius
        assert rect.height() == radius


# ---------------------------------------------------------------------------
# Title-bar drag still works
# ---------------------------------------------------------------------------

class TestTitleBarDragPreserved:
    """Title-bar drag must continue to work alongside edge resize."""

    def test_title_bar_press_starts_drag(self, panel):
        """Title bar press should still set _title_dragging."""
        assert panel._title_dragging is False
        title_pos = QPoint(panel._title_bar.width() // 2, 12)
        global_pos = panel._title_bar.mapToGlobal(title_pos)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(title_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mousePressEvent(event)
        assert panel._title_dragging is True

        # Release
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(title_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mouseReleaseEvent(release)
        assert panel._title_dragging is False

    def test_edge_resize_does_not_interfere_with_title_bar(self, panel):
        """Edge resize state and title-bar drag state are independent."""
        assert panel._edge_resizing is False
        assert panel._title_dragging is False

        title_pos = QPoint(panel._title_bar.width() // 2, 12)
        global_pos = panel._title_bar.mapToGlobal(title_pos)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(title_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mousePressEvent(event)
        assert panel._title_dragging is True
        assert panel._edge_resizing is False

        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(title_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mouseReleaseEvent(release)
