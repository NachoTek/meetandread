"""Interaction Trace Qt-widget-seam tests (issue #105).

The spec's testing decision: "Qt widget seam (prior art: the widget
test suite's pattern of driving real widgets): install a trace sink,
drive real widgets, assert semantic named events appear and the
no-keystroke-content invariant holds."

Real Qt widgets are driven (real panels, real lobes, real line edits,
real mouse/key events) with the trace sink installed — no mocked-away
seam. The privacy assertions grep BOTH the trace file AND a real
capture DEBUG log (configure_capture_logging into the same dir), since
the widget-layer DEBUG instrumentation (batch C, #120) just landed and
must be verified to carry no text content either.

These tests run in the fast lane (Qt is available there); the
authoritative pass is the Windows venv per ADR 0001, exercised by
tests/test_interaction_trace_subprocess.py.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QPointF, QPoint, Qt, QEvent
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton

import meetandread.interaction_trace as itrace
from meetandread.interaction_trace import (
    TRACE_FILE_NAME,
    install_interaction_trace,
    read_trace_events,
)

# Distinctive canary the trace and the capture log must never contain.
TEXT_CANARY = "SECREThunter2bluechair CANARY typed-content 8899"


@pytest.fixture()
def trace_sink(tmp_path: Path) -> Path:
    """A capture dir with the Interaction Trace installed and NO Qt
    filter — widget tests install the filter explicitly when testing
    it (the sink must work when events are emitted by instrumentation)."""
    capture_dir = tmp_path / "capture" / "qt-run"
    capture_dir.mkdir(parents=True)
    install_interaction_trace(capture_dir)
    return capture_dir


@pytest.fixture(autouse=True)
def _reset_trace_state():
    saved_writer = itrace._writer
    itrace._writer = None
    try:
        yield
    finally:
        remove_interaction_trace_qt_filter()
        if itrace._writer is not None:
            itrace._writer.close()
        itrace._writer = saved_writer


def itrace_qt_filter_installed() -> bool:
    from meetandread.interaction_trace_qt import (
        interaction_trace_qt_filter_installed,
    )

    return interaction_trace_qt_filter_installed()


def remove_interaction_trace_qt_filter() -> None:
    from meetandread.interaction_trace_qt import (
        remove_interaction_trace_qt_filter as _remove,
    )

    _remove()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Semantics: driving real widgets produces named events
# ---------------------------------------------------------------------------


class TestRealWidgetEvents:
    def test_record_button_click_produces_button_pressed(self, qapp, trace_sink):
        from meetandread.widgets.main_widget import RecordButtonItem

        parent = MagicMock()
        parent.is_dragging = False
        parent._click_consumed = False
        parent.toggle_recording = MagicMock()
        button = RecordButtonItem(parent)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(10, 10), QPointF(10, 10),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        button.mouseReleaseEvent(release)
        parent.toggle_recording.assert_called_once()
        events = read_trace_events(trace_sink)
        assert [(e["event"], e["target"]) for e in events] == [
            ("button_pressed", "record_button")
        ]

    def test_lobe_toggle_produces_device_selected(self, qapp, trace_sink):
        from meetandread.widgets.main_widget import ToggleLobeItem

        parent = MagicMock()
        parent.is_dragging = False
        parent._click_consumed = False
        parent._on_lobe_toggled = MagicMock()
        lobe = ToggleLobeItem("microphone", parent)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(10, 10), QPointF(10, 10),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        lobe.mouseReleaseEvent(release)
        parent._on_lobe_toggled.assert_called_once()
        events = read_trace_events(trace_sink)
        assert events[0]["event"] == "device_selected"
        assert events[0]["target"] == "microphone"
        assert events[0]["selected"] is True
        # Toggle back off
        lobe.mouseReleaseEvent(release)
        events = read_trace_events(trace_sink)
        assert events[1]["selected"] is False

    def test_tray_menu_selection_produces_menu_item_selected(
        self, qapp, trace_sink
    ):
        from meetandread.widgets.tray_icon import TrayIconManager

        tray = TrayIconManager(widget=None)
        tray._handle_toggle_recording()  # Start Recording menu action path
        events = read_trace_events(trace_sink)
        assert [(e["event"], e["target"]) for e in events] == [
            ("menu_item_selected", "tray.toggle_recording")
        ]

    def test_settings_panel_open_close_produces_panel_events(
        self, qapp, trace_sink
    ):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        panel = FloatingSettingsPanel()
        panel.show_panel()
        qapp.processEvents()
        panel.hide_panel()
        events = read_trace_events(trace_sink)
        pairs = [(e["event"], e["target"]) for e in events]
        assert ("panel_opened", "settings_panel") in pairs
        assert ("panel_closed", "settings_panel") in pairs
        assert pairs.index(("panel_opened", "settings_panel")) < pairs.index(
            ("panel_closed", "settings_panel")
        )

    def test_settings_panel_title_drag_produces_panel_moved(
        self, qapp, trace_sink
    ):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        panel = FloatingSettingsPanel()
        panel.show()
        title_pos = QPoint(panel._title_bar.width() // 2, 12)
        global_pos = panel._title_bar.mapToGlobal(title_pos)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(title_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mousePressEvent(press)
        moved_global = global_pos + QPoint(40, 25)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(panel._title_bar.mapFromGlobal(moved_global)),
            QPointF(moved_global),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mouseMoveEvent(move)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(panel._title_bar.mapFromGlobal(moved_global)),
            QPointF(moved_global),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel._title_bar.mouseReleaseEvent(release)
        events = read_trace_events(trace_sink)
        moved = [e for e in events if e["event"] == "panel_moved"]
        assert moved and moved[0]["target"] == "settings_panel"
        assert isinstance(moved[0]["x"], int) and isinstance(moved[0]["y"], int)

    def test_settings_panel_edge_resize_produces_panel_resized(
        self, qapp, trace_sink
    ):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        panel = FloatingSettingsPanel()
        panel.show()
        edge_pos = QPoint(panel.width() - 2, panel.height() // 2)
        global_pos = panel.mapToGlobal(edge_pos)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(edge_pos), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.mousePressEvent(press)
        moved_global = global_pos + QPoint(50, 0)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(panel.mapFromGlobal(moved_global)),
            QPointF(moved_global),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.mouseMoveEvent(move)
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(panel.mapFromGlobal(moved_global)),
            QPointF(moved_global),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.mouseReleaseEvent(release)
        qapp.processEvents()
        events = read_trace_events(trace_sink)
        resized = [e for e in events if e["event"] == "panel_resized"]
        assert resized and resized[0]["target"] == "settings_panel"
        assert isinstance(resized[0]["w"], int) and isinstance(resized[0]["h"], int)


class TestCCOverlayPanelTrace:
    """The CC overlay is draggable AND resizable (unlike the settings
    panel it also resizes through the grip eventFilter) — every one of
    those end-of-gesture paths must land in the trace sink."""

    def _cc_overlay(self, qapp):
        from meetandread.widgets.floating_panels import CCOverlayPanel

        panel = CCOverlayPanel()
        panel.show()
        panel.move(100, 100)
        qapp.processEvents()
        return panel

    def _mouse(self, etype, local, global_pos):
        return QMouseEvent(
            etype,
            QPointF(local), QPointF(global_pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

    def test_cc_overlay_drag_release_produces_panel_moved(
        self, qapp, trace_sink
    ):
        panel = self._cc_overlay(qapp)
        center = QPoint(panel.width() // 2, panel.height() // 2)
        press_global = panel.mapToGlobal(center)
        panel.mousePressEvent(
            self._mouse(
                QMouseEvent.Type.MouseButtonPress, center, press_global
            )
        )
        moved_global = press_global + QPoint(40, 25)
        move_local = panel.mapFromGlobal(moved_global)
        panel.mouseMoveEvent(
            self._mouse(
                QMouseEvent.Type.MouseMove, move_local, moved_global
            )
        )
        release_local = panel.mapFromGlobal(moved_global)
        panel.mouseReleaseEvent(
            self._mouse(
                QMouseEvent.Type.MouseButtonRelease,
                release_local,
                moved_global,
            )
        )
        events = read_trace_events(trace_sink)
        moved = [e for e in events if e["event"] == "panel_moved"]
        assert moved and moved[0]["target"] == "cc_overlay"
        assert isinstance(moved[0]["x"], int) and isinstance(moved[0]["y"], int)

    def test_cc_overlay_edge_resize_release_produces_panel_resized(
        self, qapp, trace_sink
    ):
        panel = self._cc_overlay(qapp)
        edge_pos = QPoint(panel.width() - 2, panel.height() // 2)
        press_global = panel.mapToGlobal(edge_pos)
        panel.mousePressEvent(
            self._mouse(
                QMouseEvent.Type.MouseButtonPress, edge_pos, press_global
            )
        )
        moved_global = press_global + QPoint(50, 0)
        move_local = panel.mapFromGlobal(moved_global)
        panel.mouseMoveEvent(
            self._mouse(
                QMouseEvent.Type.MouseMove, move_local, moved_global
            )
        )
        release_local = panel.mapFromGlobal(moved_global)
        panel.mouseReleaseEvent(
            self._mouse(
                QMouseEvent.Type.MouseButtonRelease,
                release_local,
                moved_global,
            )
        )
        events = read_trace_events(trace_sink)
        resized = [e for e in events if e["event"] == "panel_resized"]
        assert resized and resized[0]["target"] == "cc_overlay"
        assert isinstance(resized[0]["w"], int) and isinstance(resized[0]["h"], int)

    def test_cc_overlay_grip_resize_release_produces_panel_resized(
        self, qapp, trace_sink
    ):
        """The resize-grip eventFilter path (the CC overlay's second,
        settings-panel-absent resize funnel): real QMouseEvents sent
        through QApplication to the real grip, routed by the installed
        filter."""
        panel = self._cc_overlay(qapp)
        grip = panel._resize_grip
        grip_center = QPoint(grip.width() // 2, grip.height() // 2)
        press_global = grip.mapToGlobal(grip_center)
        QApplication.sendEvent(
            grip,
            self._mouse(
                QMouseEvent.Type.MouseButtonPress,
                grip_center,
                press_global,
            ),
        )
        moved_global = press_global + QPoint(50, 50)
        move_local = grip.mapFromGlobal(moved_global)
        QApplication.sendEvent(
            grip,
            self._mouse(
                QMouseEvent.Type.MouseMove, move_local, moved_global
            ),
        )
        release_local = grip.mapFromGlobal(moved_global)
        QApplication.sendEvent(
            grip,
            self._mouse(
                QMouseEvent.Type.MouseButtonRelease,
                release_local,
                moved_global,
            ),
        )
        qapp.processEvents()
        events = read_trace_events(trace_sink)
        resized = [e for e in events if e["event"] == "panel_resized"]
        assert resized and resized[0]["target"] == "cc_overlay"
        assert isinstance(resized[0]["w"], int) and isinstance(resized[0]["h"], int)


# ---------------------------------------------------------------------------
# The Qt application-level filter: focus changes, generic buttons,
# text editing through FocusOut — never content
# ---------------------------------------------------------------------------


class TestInteractionTraceFilter:
    def test_install_and_remove_filter(self, qapp, trace_sink):
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
            remove_interaction_trace_qt_filter,
        )

        install_interaction_trace_qt_filter()
        assert itrace_qt_filter_installed()
        remove_interaction_trace_qt_filter()
        assert not itrace_qt_filter_installed()

    def test_install_without_trace_is_refused(self, qapp):
        from meetandread.interaction_trace_qt import (
            InteractionTraceFilterError,
            install_interaction_trace_qt_filter,
        )

        with pytest.raises(InteractionTraceFilterError):
            install_interaction_trace_qt_filter()

    def test_focus_change_produces_window_focus_changed(self, qapp, trace_sink):
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
        )

        install_interaction_trace_qt_filter()
        try:
            top = QPushButton("focus target")
            top.show()
            top.setFocus()
            qapp.processEvents()
            itrace.emit_interaction_event  # sanity import
        finally:
            top.hide()
            top.deleteLater()
        events = read_trace_events(trace_sink)
        focus = [
            e for e in events if e["event"] == "window_focus_changed"
        ]
        assert focus, f"no focus events: {events}"
        assert all(isinstance(e["focused"], bool) for e in focus)

    def test_generic_button_release_produces_button_pressed(
        self, qapp, trace_sink
    ):
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
        )

        install_interaction_trace_qt_filter()
        button = QPushButton("Benchmark")
        button.show()
        try:
            release = QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(5, 5), QPointF(5, 5),
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            # Deliver the release to the real button — QApplication's
            # notify path routes it through the installed filter, the
            # exact path a real click takes.
            QApplication.sendEvent(button, release)
            qapp.processEvents()
        finally:
            button.hide()
            button.deleteLater()
        events = read_trace_events(trace_sink)
        pressed = [e for e in events if e["event"] == "button_pressed"]
        assert pressed and pressed[0]["target"] == "generic_button"

    def test_line_edit_editing_then_focus_out_records_char_count_only(
        self, qapp, trace_sink
    ):
        """THE privacy invariant at the widget seam: type a canary into
        a real QLineEdit, focus out, assert exactly one text_edited
        event carrying only the char count — and the canary is in
        neither the trace nor the capture DEBUG log."""
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
        )

        install_interaction_trace_qt_filter()
        edit = QLineEdit()
        edit.show()
        try:
            edit.setFocus()
            qapp.processEvents()
            edit.insert(TEXT_CANARY)
            qapp.processEvents()
            # Leave the field: FocusOut collapses the edit session.
            out = QEvent(QEvent.Type.FocusOut)
            QApplication.sendEvent(edit, out)
            qapp.processEvents()
        finally:
            edit.hide()
            edit.deleteLater()
        events = read_trace_events(trace_sink)
        text_events = [e for e in events if e["event"] == "text_edited"]
        assert len(text_events) == 1, (
            f"expected exactly one text_edited event, got: {events}"
        )
        assert text_events[0]["chars"] == len(TEXT_CANARY)
        assert set(text_events[0]) == {"ts", "event", "target", "chars"}
        raw_trace = (trace_sink / TRACE_FILE_NAME).read_text(encoding="utf-8")
        assert TEXT_CANARY not in raw_trace
        assert "SECREThunter2bluechair" not in raw_trace

    def test_line_edit_no_trailing_edits_no_spurious_event(
        self, qapp, trace_sink
    ):
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
        )

        install_interaction_trace_qt_filter()
        edit = QLineEdit()
        edit.show()
        try:
            edit.setFocus()
            qapp.processEvents()
            out = QEvent(QEvent.Type.FocusOut)
            QApplication.sendEvent(edit, out)
            qapp.processEvents()
        finally:
            edit.hide()
            edit.deleteLater()
        events = read_trace_events(trace_sink)
        assert [e for e in events if e["event"] == "text_edited"] == []


# ---------------------------------------------------------------------------
# The capture DEBUG log must carry no text content either (AC + brief:
# "including a grep of the capture DEBUG log")
# ---------------------------------------------------------------------------


class TestCaptureLogPrivacy:
    def test_typed_content_absent_from_capture_debug_log(
        self, qapp, tmp_path
    ):
        from meetandread.capture_mode import configure_capture_logging
        from meetandread.interaction_trace_qt import (
            install_interaction_trace_qt_filter,
        )

        # Snapshot logging state (configure_capture_logging replaces it).
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        capture_dir = tmp_path / "capture" / "log-run"
        capture_dir.mkdir(parents=True)
        try:
            log_file = configure_capture_logging(capture_dir)
            install_interaction_trace(capture_dir)
            install_interaction_trace_qt_filter()

            edit = QLineEdit()
            edit.show()
            try:
                edit.setFocus()
                qapp.processEvents()
                edit.insert(TEXT_CANARY)
                qapp.processEvents()
                QApplication.sendEvent(edit, QEvent(QEvent.Type.FocusOut))
                qapp.processEvents()
            finally:
                edit.hide()
                edit.deleteLater()

            for handler in list(root.handlers):
                handler.flush()
            trace_events = read_trace_events(capture_dir)
            assert [e["event"] for e in trace_events if e["event"] == "text_edited"]
            log_text = log_file.read_text(encoding="utf-8")
            assert TEXT_CANARY not in log_text
            assert "SECREThunter2bluechair" not in log_text
            assert "typed-content 8899" not in log_text
        finally:
            for handler in list(root.handlers):
                if handler not in saved_handlers:
                    root.removeHandler(handler)
                    try:
                        handler.close()
                    except OSError:
                        pass
            root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Normal run: filter never installed, nothing recorded
# ---------------------------------------------------------------------------


class TestNormalRunQuiet:
    def test_no_filter_and_no_events_without_capture(self, qapp, tmp_path):
        # No install: driving a real widget records nothing anywhere.
        assert not itrace.trace_installed()
        button = QPushButton("plain")
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(5, 5), QPointF(5, 5),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        from meetandread.widgets.main_widget import ToggleLobeItem

        parent = MagicMock()
        parent.is_dragging = False
        parent._click_consumed = False
        parent._on_lobe_toggled = MagicMock()
        lobe = ToggleLobeItem("microphone", parent)
        lobe.mouseReleaseEvent(release)
        parent._on_lobe_toggled.assert_called_once()
        assert not list(tmp_path.rglob("*.jsonl"))
