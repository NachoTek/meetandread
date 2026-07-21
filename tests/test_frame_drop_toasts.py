"""Tests for throttled frame-drop toast notifications.

Covers ToastManager replacement/dismiss behavior plus MeetAndReadWidget's
non-spammy frame-drop warning policy without requiring real audio hardware.
"""

from types import MethodType
from unittest.mock import MagicMock

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _MinimalWidget:
    pass


class _FakeToastManager:
    def __init__(self):
        self.shown = []
        self.dismissed = []

    def show(self, toast_id, title, message, *, duration_ms=0):
        self.shown.append(
            {
                "toast_id": toast_id,
                "title": title,
                "message": message,
                "duration_ms": duration_ms,
            }
        )
        return MagicMock()

    def dismiss(self, toast_id):
        self.dismissed.append(toast_id)


def _minimal_widget():
    from meetandread.widgets.main_widget import MeetAndReadWidget

    widget = _MinimalWidget()
    widget._reset_frame_drop_toast_state = MethodType(MeetAndReadWidget._reset_frame_drop_toast_state, widget)
    widget._maybe_show_frame_drop_toast = MethodType(MeetAndReadWidget._maybe_show_frame_drop_toast, widget)
    widget._on_frames_dropped = MethodType(MeetAndReadWidget._on_frames_dropped, widget)
    widget.record_button = MagicMock()
    widget.record_button.on_frames_dropped = MagicMock()
    widget.toast_manager = _FakeToastManager()
    widget._frame_drop_toast_id = "frame-drops"
    widget._frame_drop_toast_last_count = 0
    widget._frame_drop_toast_last_ts = 0.0
    widget._frame_drop_toast_reminder_seconds = 60.0
    return widget


class TestToastManager:
    def test_same_id_replaces_existing_toast(self, qapp):
        from meetandread.widgets.floating_panels import ToastManager

        anchor = QWidget()
        manager = ToastManager(anchor)
        try:
            first = manager.show("frame-drops", "Recording quality warning", "Dropped 5 frames", duration_ms=0)
            second = manager.show("frame-drops", "Recording quality warning", "Dropped 9 frames", duration_ms=0)

            assert first is second
            assert manager.active_ids() == ["frame-drops"]
            assert "9" in second.message_label.text()
        finally:
            manager.dismiss_all()
            anchor.deleteLater()

    def test_replacing_toast_replaces_and_clears_action(self, qapp):
        from meetandread.widgets.floating_panels import ToastManager

        anchor = QWidget()
        manager = ToastManager(anchor)
        first_callback = MagicMock()
        replacement_callback = MagicMock()
        try:
            toast = manager.show(
                "recovery",
                "Recording interrupted",
                "Reconnect the microphone.",
                duration_ms=0,
                action_label="Retry old",
                action_callback=first_callback,
            )
            action = toast.findChild(QPushButton, "toast-action")
            assert action is not None
            assert action.text() == "Retry old"
            action.click()
            assert first_callback.call_count == 1

            replacement = manager.show(
                "recovery",
                "Recording paused",
                "Resume when ready.",
                duration_ms=0,
                action_label="Resume Recording",
                action_callback=replacement_callback,
            )
            assert replacement is toast
            action = replacement.findChild(QPushButton, "toast-action")
            assert action is not None
            assert action.text() == "Resume Recording"
            action.click()
            assert first_callback.call_count == 1
            assert replacement_callback.call_count == 1

            manager.show(
                "recovery",
                "Recording resumed",
                "Your microphone is available again.",
                duration_ms=0,
            )
            assert toast.findChild(QPushButton, "toast-action") is None
        finally:
            manager.dismiss_all()
            anchor.deleteLater()

    @pytest.mark.parametrize(
        ("action_label", "action_callback"),
        [("Resume Recording", None), (None, MagicMock())],
    )
    def test_incomplete_action_is_not_rendered(
        self,
        qapp,
        action_label,
        action_callback,
    ):
        from meetandread.widgets.floating_panels import ToastManager

        anchor = QWidget()
        manager = ToastManager(anchor)
        try:
            toast = manager.show(
                "recovery",
                "Recording paused",
                "Resume when ready.",
                duration_ms=0,
                action_label=action_label,
                action_callback=action_callback,
            )
            assert toast.findChild(QPushButton, "toast-action") is None
        finally:
            manager.dismiss_all()
            anchor.deleteLater()

    def test_auto_dismiss_and_dismiss_all(self, qapp):
        from meetandread.widgets.floating_panels import ToastManager

        anchor = QWidget()
        manager = ToastManager(anchor)
        try:
            manager.show("short", "Title", "Auto dismiss", duration_ms=10)
            assert manager.active_ids() == ["short"]
            QTest.qWait(30)
            qapp.processEvents()
            assert manager.active_ids() == []

            manager.show("a", "Title", "A", duration_ms=0)
            manager.show("b", "Title", "B", duration_ms=0)
            assert set(manager.active_ids()) == {"a", "b"}
            manager.dismiss_all()
            assert manager.active_ids() == []
        finally:
            manager.dismiss_all()
            anchor.deleteLater()


class TestFrameDropToastThrottling:
    def test_first_positive_drop_shows_toast_and_preserves_record_button_signal(self, monkeypatch):
        from meetandread.widgets import main_widget

        widget = _minimal_widget()
        monkeypatch.setattr(main_widget._time, "monotonic", lambda: 100.0)

        widget._on_frames_dropped(5)

        widget.record_button.on_frames_dropped.assert_called_once_with(5)
        assert len(widget.toast_manager.shown) == 1
        toast = widget.toast_manager.shown[0]
        assert toast["toast_id"] == "frame-drops"
        assert "Recording quality warning" == toast["title"]
        assert "5" in toast["message"]
        assert toast["duration_ms"] == 10000

    def test_repeated_drops_within_one_minute_are_throttled(self, monkeypatch):
        from meetandread.widgets import main_widget

        widget = _minimal_widget()
        times = iter([100.0, 120.0, 159.9])
        monkeypatch.setattr(main_widget._time, "monotonic", lambda: next(times))

        widget._on_frames_dropped(5)
        widget._on_frames_dropped(6)
        widget._on_frames_dropped(7)

        assert [call.args[0] for call in widget.record_button.on_frames_dropped.call_args_list] == [5, 6, 7]
        assert len(widget.toast_manager.shown) == 1
        assert widget._frame_drop_toast_last_count == 7

    def test_reminder_after_one_minute_replaces_same_toast_id(self, monkeypatch):
        from meetandread.widgets import main_widget

        widget = _minimal_widget()
        times = iter([100.0, 161.0])
        monkeypatch.setattr(main_widget._time, "monotonic", lambda: next(times))

        widget._on_frames_dropped(5)
        widget._on_frames_dropped(12)

        assert [toast["toast_id"] for toast in widget.toast_manager.shown] == ["frame-drops", "frame-drops"]
        assert "12" in widget.toast_manager.shown[-1]["message"]

    def test_malformed_and_non_positive_counts_do_not_toast_or_signal(self):
        widget = _minimal_widget()

        for value in (None, "bad", -1, 0):
            widget._on_frames_dropped(value)

        widget.record_button.on_frames_dropped.assert_not_called()
        assert widget.toast_manager.shown == []

    def test_reset_clears_throttle_and_dismisses_stale_toast(self):
        widget = _minimal_widget()
        widget._frame_drop_toast_last_count = 99
        widget._frame_drop_toast_last_ts = 123.4

        widget._reset_frame_drop_toast_state()

        assert widget._frame_drop_toast_last_count == 0
        assert widget._frame_drop_toast_last_ts == 0.0
        assert widget.toast_manager.dismissed == ["frame-drops"]


class TestBridgeControllerCompatibility:
    def test_bridge_signal_reaches_toast_enabled_widget_slot(self, qapp, monkeypatch):
        from meetandread.widgets import main_widget
        from meetandread.widgets.main_widget import _ControllerBridge

        widget = _minimal_widget()
        monkeypatch.setattr(main_widget._time, "monotonic", lambda: 200.0)
        bridge = _ControllerBridge()
        bridge.frames_dropped.connect(widget._on_frames_dropped)

        bridge.frames_dropped.emit(8)
        qapp.processEvents()

        widget.record_button.on_frames_dropped.assert_called_once_with(8)
        assert len(widget.toast_manager.shown) == 1
        assert "8" in widget.toast_manager.shown[0]["message"]
