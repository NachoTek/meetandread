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


class FakeIndicator:
    def __init__(self):
        self.messages = []

    def set_text(self, message, is_recoverable=True):
        self.messages.append((message, is_recoverable))


def widget_shell():
    widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
    widget._warning_indicator = FakeIndicator()
    widget._error_indicator = FakeIndicator()
    widget._show_resource_warning = lambda message: widget._warning_indicator.set_text(message)
    widget._show_error = lambda message, is_recoverable=True: widget._error_indicator.set_text(message, is_recoverable)
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


def test_device_disconnect_notification_is_non_blocking_warning(qapp):
    widget = widget_shell()
    event = DeviceEvent(DeviceEventType.REMOVED, "dev-1", friendly_name="USB Headset", state="inactive")

    widget._on_device_changed(event)

    message, _recoverable = widget._warning_indicator.messages[-1]
    assert "disconnected" in message
    assert "USB Headset" in message
    assert widget._error_indicator.messages == []


@pytest.mark.parametrize(
    "outcome,expected_fragment,error_expected",
    [
        (RecoveryOutcome.LOST, "Attempting to recover", False),
        (RecoveryOutcome.DEGRADED, "continues with remaining sources", False),
        (RecoveryOutcome.AUTO_RECOVERED, "recovered", False),
        (RecoveryOutcome.MANUAL_RECOVERED, "resumed", False),
        (RecoveryOutcome.MANUAL_RETRY_REQUIRED, "Resume recording manually", True),
        (RecoveryOutcome.TOTAL_LOSS, "paused until an audio device returns", True),
    ],
)
def test_recovery_outcome_notifications(outcome, expected_fragment, error_expected, qapp):
    widget = widget_shell()
    result = RecoveryResult(outcome, source_type="mic", message="sanitized detail", recoverable=not error_expected)

    widget._on_recovery_attempted(result)

    indicator = widget._error_indicator if error_expected else widget._warning_indicator
    message, recoverable = indicator.messages[-1]
    assert expected_fragment in message
    assert "sanitized detail" in message
    assert recoverable is (not error_expected)


def test_ignored_recovery_outcome_is_silent(qapp):
    widget = widget_shell()
    result = RecoveryResult(RecoveryOutcome.IGNORED, message="unmatched device")

    widget._on_recovery_attempted(result)

    assert widget._warning_indicator.messages == []
    assert widget._error_indicator.messages == []
