import time

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
    RecoveryOutcome,
)


class FakeMonitor:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.events = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def drain_events(self, max_events=100):
        events = self.events[:max_events]
        del self.events[:max_events]
        return events


class FakeSession:
    def __init__(self):
        self.started_config = None
        self.stopped = 0

    def start(self, config):
        self.started_config = config

    def stop(self):
        self.stopped += 1
        return "recording.wav"

    def get_state(self):
        return ControllerState.RECORDING

    def get_stats(self):
        raise RuntimeError("not needed")

    def get_error(self):
        return None


def _event(kind, device_id="mic-1", friendly_name="USB Mic", flow="capture", when=None):
    return DeviceEvent(
        event_type=kind,
        device_id=device_id,
        friendly_name=friendly_name,
        flow=flow,
        timestamp=when,
    )


def _recording_controller(monkeypatch, *, now=100.0):
    ctrl = RecordingController(enable_transcription=False)
    fake_session = FakeSession()
    fake_monitor = FakeMonitor()
    monkeypatch.setattr("meetandread.recording.controller.AudioSession", lambda: fake_session)
    monkeypatch.setattr("meetandread.recording.controller.WindowsDeviceMonitor", lambda: fake_monitor)
    monkeypatch.setattr("meetandread.recording.controller._time.time", lambda: now)
    return ctrl, fake_session, fake_monitor


def test_monitor_lifecycle_and_active_source_snapshot(monkeypatch):
    ctrl, fake_session, fake_monitor = _recording_controller(monkeypatch)

    assert ctrl.start({"mic", "system"}) is None

    assert fake_session.started_config is not None
    assert fake_monitor.started == 1
    diagnostics = ctrl.get_diagnostics()["hotplug"]
    assert diagnostics["monitor_active"] is True
    assert {source["type"] for source in diagnostics["active_sources"]} == {"mic", "system"}
    assert all("raw" not in str(source).lower() for source in diagnostics["active_sources"])

    ctrl.stop()

    assert fake_monitor.stopped == 1


def test_same_source_reappears_within_window_auto_recovers(monkeypatch):
    ctrl, _, _ = _recording_controller(monkeypatch, now=100.0)
    callbacks = []
    ctrl.on_device_change = lambda event: callbacks.append(("device", event.event_type.value))
    ctrl.on_recovery_attempt = lambda result: callbacks.append(("recovery", result.outcome.value, result.source_type))
    assert ctrl.start({"mic"}) is None

    ctrl.handle_device_event(_event(DeviceEventType.REMOVED, "mic-1", flow="capture"), now=100.0)
    result = ctrl.handle_device_event(_event(DeviceEventType.ADDED, "mic-1", flow="capture"), now=104.9)

    assert result.outcome is RecoveryOutcome.AUTO_RECOVERED
    assert result.source_type == "mic"
    assert callbacks == [
        ("device", "removed"),
        ("recovery", "total_loss", "mic"),
        ("device", "added"),
        ("recovery", "auto_recovered", "mic"),
    ]
    assert ctrl.get_state() is ControllerState.RECORDING


def test_recovery_window_expiry_requires_manual_retry(monkeypatch):
    ctrl, _, _ = _recording_controller(monkeypatch, now=100.0)
    assert ctrl.start({"mic"}) is None

    ctrl.handle_device_event(_event(DeviceEventType.REMOVED, "mic-1", flow="capture"), now=100.0)
    result = ctrl.handle_device_event(_event(DeviceEventType.ADDED, "mic-1", flow="capture"), now=106.0)

    assert result.outcome is RecoveryOutcome.MANUAL_RETRY_REQUIRED
    assert ctrl.get_state() is ControllerState.ERROR
    assert ctrl.get_error().is_recoverable is True

    retry_result = ctrl.retry_recovery(now=106.1)
    assert retry_result.outcome is RecoveryOutcome.MANUAL_RECOVERED
    assert ctrl.get_state() is ControllerState.RECORDING


def test_partial_source_loss_continues_with_remaining_source(monkeypatch):
    ctrl, _, _ = _recording_controller(monkeypatch, now=100.0)
    assert ctrl.start({"mic", "system"}) is None

    result = ctrl.handle_device_event(_event(DeviceEventType.REMOVED, "mic-1", flow="capture"), now=100.0)

    assert result.outcome is RecoveryOutcome.DEGRADED
    assert ctrl.get_state() is ControllerState.RECORDING
    diagnostics = ctrl.get_diagnostics()["hotplug"]
    assert diagnostics["active_source_count"] == 1
    assert diagnostics["lost_source_count"] == 1
    assert diagnostics["last_recovery_result"]["outcome"] == "degraded"


def test_total_source_loss_sets_recoverable_error(monkeypatch):
    ctrl, _, _ = _recording_controller(monkeypatch, now=100.0)
    assert ctrl.start({"mic"}) is None

    result = ctrl.handle_device_event(_event(DeviceEventType.REMOVED, "mic-1", flow="capture"), now=100.0)

    assert result.outcome is RecoveryOutcome.TOTAL_LOSS
    assert ctrl.get_state() is ControllerState.ERROR
    assert ctrl.get_error().is_recoverable is True
    assert "capture source lost" in ctrl.get_error().message.lower()


def test_handle_device_event_ignores_events_while_not_recording(monkeypatch):
    ctrl, _, fake_monitor = _recording_controller(monkeypatch, now=100.0)

    result = ctrl.handle_device_event(_event(DeviceEventType.REMOVED), now=100.0)

    assert result is None
    assert fake_monitor.started == 0
    assert ctrl.get_diagnostics()["hotplug"]["monitor_active"] is False
