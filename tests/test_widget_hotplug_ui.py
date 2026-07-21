"""UI contract tests for recording-device recovery feedback and manual retry."""

from unittest.mock import Mock

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.recording.controller import RecoveryOutcome, RecoveryResult
from meetandread.widgets.main_widget import MeetAndReadWidget


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


class FakeController:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.retry_calls = 0
        self.start_recording = Mock()

    def retry_recovery(self):
        self.retry_calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def widget_shell(controller=None):
    widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
    widget.toast_manager = FakeToastManager()
    widget._controller = controller or FakeController()
    widget._recovery_toast_id = "recording-device-recovery"
    return widget


def test_device_loss_and_auto_recovery_replace_one_stable_toast():
    widget = widget_shell()
    event = DeviceEvent(
        DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="Desk Mic",
        flow="capture",
    )

    widget._on_device_changed(event)
    widget._on_recovery_attempted(
        RecoveryResult(RecoveryOutcome.AUTO_RECOVERED, "mic", "mic-1")
    )

    pending, recovered = widget.toast_manager.shown
    assert pending["toast_id"] == recovered["toast_id"] == "recording-device-recovery"
    assert pending["title"] == "Recording device disconnected"
    assert pending["duration_ms"] == 0
    assert recovered["title"] == "Recording recovered"
    assert recovered["duration_ms"] > 0
    assert recovered["action_label"] is None
    assert recovered["action_callback"] is None


def test_manual_retry_action_preserves_session_and_success_replaces_failure():
    result = RecoveryResult(RecoveryOutcome.MANUAL_RECOVERED, "mic", "mic-1")
    controller = FakeController(result=result)
    widget = widget_shell(controller)

    widget._on_recovery_attempted(
        RecoveryResult(RecoveryOutcome.MANUAL_RETRY_REQUIRED, "mic", "mic-1")
    )

    required = widget.toast_manager.shown[-1]
    assert required["toast_id"] == "recording-device-recovery"
    assert required["title"] == "Recording paused"
    assert required["duration_ms"] == 0
    assert required["action_label"] == "Resume Recording"

    required["action_callback"]()
    assert controller.retry_calls == 1
    controller.start_recording.assert_not_called()

    # The real controller emits this result through _ControllerBridge. Simulate
    # that signal delivery to prove it replaces the persistent action toast.
    widget._on_recovery_attempted(result)
    resumed = widget.toast_manager.shown[-1]
    assert resumed["toast_id"] == required["toast_id"]
    assert resumed["title"] == "Recording resumed"
    assert resumed["duration_ms"] > 0
    assert resumed["action_label"] is None
    assert resumed["action_callback"] is None


def test_total_loss_also_offers_non_expiring_resume_action():
    widget = widget_shell()

    widget._on_recovery_attempted(
        RecoveryResult(RecoveryOutcome.TOTAL_LOSS, "system", "speaker-1")
    )

    toast = widget.toast_manager.shown[-1]
    assert toast["title"] == "Recording paused"
    assert toast["duration_ms"] == 0
    assert toast["action_label"] == "Resume Recording"
    assert callable(toast["action_callback"])


def test_retry_failure_keeps_actionable_toast_visible():
    controller = FakeController(error=RuntimeError("device still unavailable"))
    widget = widget_shell(controller)
    widget._on_recovery_attempted(
        RecoveryResult(RecoveryOutcome.MANUAL_RETRY_REQUIRED, "mic", "mic-1")
    )

    widget.toast_manager.shown[-1]["action_callback"]()

    failed = widget.toast_manager.shown[-1]
    assert controller.retry_calls == 1
    assert failed["toast_id"] == "recording-device-recovery"
    assert failed["title"] == "Recording recovery failed"
    assert failed["duration_ms"] == 0
    assert failed["action_label"] == "Resume Recording"
    assert callable(failed["action_callback"])
