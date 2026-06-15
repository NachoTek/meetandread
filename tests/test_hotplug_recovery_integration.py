import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

from PyQt6.QtWidgets import QApplication

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.audio.session import SourceConfig
from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
    RecoveryOutcome,
)
from meetandread.widgets.main_widget import MeetAndReadWidget


class FakeMonitor:
    def __init__(self, events=None):
        self.events = list(events or [])
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def drain_events(self, max_events=100):
        events = self.events[:max_events]
        del self.events[:max_events]
        return events


class RaisingMonitor(FakeMonitor):
    def drain_events(self, max_events=100):
        raise RuntimeError("backend drain failed")


class FakeIndicator:
    def __init__(self):
        self.messages = []

    def set_text(self, message, is_recoverable=True):
        self.messages.append((message, is_recoverable))


def _source(source_type, device_id=None, friendly_name=None):
    source = SourceConfig(type=source_type, device_id=device_id)
    source.friendly_name = friendly_name
    return source


def _recording_controller(*sources, monitor=None):
    controller = RecordingController(enable_transcription=False)
    controller._state = ControllerState.RECORDING
    controller._snapshot_active_sources(list(sources))
    if monitor is not None:
        controller._hotplug_monitor = monitor
        controller._hotplug_monitor_active = True
    return controller


def _widget_shell():
    QApplication.instance() or QApplication([])
    widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
    widget._warning_indicator = FakeIndicator()
    widget._error_indicator = FakeIndicator()
    return widget


def _emit_to_widget(widget, events, results):
    for event in events:
        MeetAndReadWidget._on_device_changed(widget, event)
    for result in results:
        MeetAndReadWidget._on_recovery_attempted(widget, result)


def test_monitor_controller_and_ui_emit_ordered_hotplug_recovery_messages():
    remove = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="USB Headset",
        flow="capture",
    )
    add = DeviceEvent(
        event_type=DeviceEventType.ADDED,
        device_id="mic-1",
        friendly_name="USB Headset",
        flow="capture",
    )
    monitor = FakeMonitor([remove, add])
    controller = _recording_controller(
        _source("mic", "mic-1", "USB Headset"),
        _source("system", "sys-1", "Laptop Speakers"),
        monitor=monitor,
    )
    device_callbacks = []
    recovery_callbacks = []
    controller.on_device_change = device_callbacks.append
    controller.on_recovery_attempt = recovery_callbacks.append

    results = controller.drain_hotplug_events()

    assert device_callbacks == [remove, add]
    assert [r.outcome for r in results] == [
        RecoveryOutcome.DEGRADED,
        RecoveryOutcome.AUTO_RECOVERED,
    ]
    assert recovery_callbacks == results

    widget = _widget_shell()
    _emit_to_widget(widget, device_callbacks, results)

    warning_messages = [message for message, _ in widget._warning_indicator.messages]
    assert "capture device changed" in warning_messages[0]
    assert "USB Headset" not in warning_messages[0]
    assert "Recording degraded" in warning_messages[1]
    assert "Recording source recovered" in warning_messages[2]
    assert widget._error_indicator.messages == []


def test_duplicate_remove_unknown_device_and_callback_exceptions_are_contained():
    known_remove = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="Private USB Microphone",
        flow="capture",
    )
    duplicate_remove = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="Private USB Microphone",
        flow="capture",
    )
    unknown_remove = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="unknown",
        friendly_name="Unknown Device",
        flow="capture",
    )
    controller = _recording_controller(
        _source("mic", "mic-1", "Private USB Microphone"),
        _source("system", "sys-1", "System Loopback"),
        monitor=FakeMonitor([known_remove, duplicate_remove, unknown_remove]),
    )
    device_callbacks = []
    controller.on_device_change = device_callbacks.append

    def failing_recovery_callback(result):
        raise RuntimeError("ui bridge unavailable")

    controller.on_recovery_attempt = failing_recovery_callback

    results = controller.drain_hotplug_events()

    assert device_callbacks == [known_remove, duplicate_remove, unknown_remove]
    assert [r.outcome for r in results] == [
        RecoveryOutcome.DEGRADED,
        RecoveryOutcome.IGNORED,
        RecoveryOutcome.IGNORED,
    ]
    assert controller.get_diagnostics()["hotplug"]["lost_sources"] == 1


def test_no_monitor_backend_and_monitor_exception_do_not_break_recording_state():
    no_monitor = _recording_controller(_source("mic", "mic-1", "USB Mic"), monitor=None)
    assert no_monitor.drain_hotplug_events() == []
    assert no_monitor.get_state() is ControllerState.RECORDING
    assert no_monitor.get_diagnostics()["hotplug"]["monitor_active"] is False

    raising = _recording_controller(
        _source("mic", "mic-1", "USB Mic"),
        monitor=RaisingMonitor(),
    )
    assert raising.drain_hotplug_events() == []
    assert raising.get_state() is ControllerState.RECORDING


def test_source_identity_falls_back_to_friendly_name_and_recovery_window_expires():
    controller = _recording_controller(_source("mic", None, "USB Conference Mic"), monitor=FakeMonitor())
    lost = controller.handle_device_event(
        DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="rotating-endpoint-id",
            friendly_name="USB Conference Mic",
            flow="capture",
        ),
        now=100.0,
    )
    assert lost.outcome is RecoveryOutcome.TOTAL_LOSS
    assert controller.get_state() is ControllerState.ERROR

    expired = controller.handle_device_event(
        DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="new-endpoint-id",
            friendly_name="USB Conference Mic",
            flow="capture",
        ),
        now=106.0,
    )
    assert expired.outcome is RecoveryOutcome.MANUAL_RETRY_REQUIRED
    assert controller.get_state() is ControllerState.ERROR


def test_diagnostics_are_sanitized_and_never_expose_content_payloads():
    controller = _recording_controller(
        _source("mic", "mic-secret-id", "Private USB Microphone"),
        monitor=FakeMonitor(),
    )
    result = controller.handle_device_event(
        DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="mic-secret-id",
            friendly_name="Private USB Microphone",
            flow="capture",
            source_metadata={
                "audio_samples": [1, 2, 3],
                "transcript_text": "confidential transcript",
                "embedding": [0.1, 0.2],
                "api_key": "sk-secret",
            },
        ),
        now=time.monotonic(),
    )
    assert result.outcome is RecoveryOutcome.TOTAL_LOSS

    diagnostics = controller.get_diagnostics()
    hotplug = diagnostics["hotplug"]
    assert set(hotplug) == {
        "monitor_active",
        "active_sources",
        "lost_sources",
        "last_device_event",
        "last_recovery_result",
        "recovery_window_seconds",
    }
    flattened = repr(hotplug)
    assert "confidential transcript" not in flattened
    assert "audio_samples" not in flattened
    assert "embedding" not in flattened
    assert "api_key" not in flattened
    assert "sk-secret" not in flattened


def test_burst_processing_preserves_order_with_bounded_work():
    events = [
        DeviceEvent(event_type=DeviceEventType.STATE_CHANGED, device_id=f"unknown-{i}", flow="capture")
        for i in range(12)
    ]
    monitor = FakeMonitor(events)
    controller = _recording_controller(_source("mic", "mic-1", "USB Mic"), monitor=monitor)
    seen = []
    controller.on_device_change = seen.append

    first_batch = controller.drain_hotplug_events(max_events=5)
    second_batch = controller.drain_hotplug_events(max_events=20)

    assert seen == events
    assert len(first_batch) == 5
    assert len(second_batch) == 7
    assert [result.outcome for result in first_batch + second_batch] == [RecoveryOutcome.IGNORED] * 12
