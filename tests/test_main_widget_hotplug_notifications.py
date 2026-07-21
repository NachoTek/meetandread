import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.recording.controller import RecoveryOutcome, RecoveryResult
from meetandread.widgets.main_widget import MeetAndReadWidget, _ControllerBridge


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeToastManager:
    def __init__(self):
        self.shown = []

    def show(
        self,
        toast_id,
        title,
        message,
        *,
        duration_ms=8000,
        action_label=None,
        action_callback=None,
    ):
        self.shown.append(
            {
                "toast_id": toast_id,
                "title": title,
                "message": message,
                "duration_ms": duration_ms,
                "action_label": action_label,
                "action_callback": action_callback,
            }
        )


def widget_shell():
    widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
    widget.toast_manager = FakeToastManager()
    widget._recovery_toast_id = "recording-device-recovery"
    widget._controller = None
    return widget


def test_bridge_exposes_hotplug_signals(qapp):
    bridge = _ControllerBridge()
    observed = []
    bridge.device_changed.connect(lambda event: observed.append(("device", event)))
    bridge.recovery_attempted.connect(lambda result: observed.append(("recovery", result)))

    event = DeviceEvent(DeviceEventType.REMOVED, "dev-1", friendly_name="USB Headset", state="inactive")
    result = RecoveryResult(RecoveryOutcome.AUTO_RECOVERED, source_type="mic", message="using replacement")

    bridge.device_changed.emit(event)
    bridge.recovery_attempted.emit(result)
    qapp.processEvents()

    assert observed == [("device", event), ("recovery", result)]


def test_device_disconnect_notification_is_persistent_stable_toast(qapp):
    widget = widget_shell()
    event = DeviceEvent(DeviceEventType.REMOVED, "dev-1", friendly_name="USB Headset", state="inactive")

    widget._on_device_changed(event)

    toast = widget.toast_manager.shown[-1]
    assert toast["toast_id"] == "recording-device-recovery"
    assert toast["title"] == "Recording device disconnected"
    assert "disconnected" in toast["message"]
    assert "USB Headset" in toast["message"]
    assert toast["duration_ms"] == 0


@pytest.mark.parametrize(
    "outcome,expected_fragment,title,persistent,actionable",
    [
        (RecoveryOutcome.LOST, "Attempting to recover", "Recording device disconnected", True, False),
        (RecoveryOutcome.DEGRADED, "continues with remaining sources", "Recording continued", False, False),
        (RecoveryOutcome.AUTO_RECOVERED, "recovered", "Recording recovered", False, False),
        (RecoveryOutcome.MANUAL_RECOVERED, "resumed", "Recording resumed", False, False),
        (RecoveryOutcome.MANUAL_RETRY_REQUIRED, "Resume recording manually", "Recording paused", True, True),
        (RecoveryOutcome.TOTAL_LOSS, "paused until an audio device returns", "Recording paused", True, True),
    ],
)
def test_recovery_outcome_notifications(
    outcome, expected_fragment, title, persistent, actionable, qapp
):
    widget = widget_shell()
    result = RecoveryResult(outcome, source_type="mic", message="sanitized detail")

    widget._on_recovery_attempted(result)

    toast = widget.toast_manager.shown[-1]
    assert toast["toast_id"] == "recording-device-recovery"
    assert toast["title"] == title
    assert expected_fragment in toast["message"]
    assert "sanitized detail" in toast["message"]
    assert (toast["duration_ms"] == 0) is persistent
    assert (toast["action_label"] == "Resume Recording") is actionable
    assert callable(toast["action_callback"]) is actionable


def test_ignored_recovery_outcome_is_silent(qapp):
    widget = widget_shell()
    result = RecoveryResult(RecoveryOutcome.IGNORED, message="unmatched device")

    widget._on_recovery_attempted(result)

    assert widget.toast_manager.shown == []
