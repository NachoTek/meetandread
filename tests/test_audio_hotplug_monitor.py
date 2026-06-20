import datetime as dt
import logging

import pytest

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType, WindowsDeviceMonitor


def test_device_event_normalizes_string_fields_and_metadata():
    event = DeviceEvent(
        event_type="added",
        device_id=" {0.0.1.00000000}.{abc} ",
        timestamp=dt.datetime(2026, 1, 2, 3, 4, 5),
        friendly_name="  USB Headset  ",
        flow=" render ",
        role=" multimedia ",
        state=" active ",
        source=" mmdevice ",
        source_metadata={"endpoint_id": "endpoint-1", "sample_rate": 48000},
    )

    assert event.event_type is DeviceEventType.ADDED
    assert event.device_id == "{0.0.1.00000000}.{abc}"
    assert event.friendly_name == "USB Headset"
    assert event.flow == "render"
    assert event.role == "multimedia"
    assert event.state == "active"
    assert event.source == "mmdevice"
    assert event.source_metadata == {"endpoint_id": "endpoint-1", "sample_rate": 48000}
    assert event.sanitized() == {
        "event_type": "added",
        "device_id": "{0.0.1.00000000}.{abc}",
        "timestamp": "2026-01-02T03:04:05",
        "friendly_name": "USB Headset",
        "flow": "render",
        "role": "multimedia",
        "state": "active",
        "source": "mmdevice",
        "source_metadata": {"endpoint_id": "endpoint-1", "sample_rate": 48000},
    }


@pytest.mark.parametrize("event_type", ["", "unknown", object()])
def test_device_event_rejects_unknown_event_type(event_type):
    with pytest.raises(ValueError):
        DeviceEvent(event_type=event_type, device_id="dev-1")


def test_monitor_lifecycle_is_idempotent_with_unavailable_backend(caplog):
    monitor = WindowsDeviceMonitor(platform_name="Linux", comtypes_backend=None)

    with caplog.at_level(logging.INFO):
        monitor.start_monitoring(lambda event: None)
        monitor.start_monitoring(lambda event: None)
        monitor.stop_monitoring()
        monitor.stop_monitoring()

    assert monitor.is_monitoring is False
    assert monitor.available is False
    assert "unavailable" in caplog.text.lower()
    assert "stopped" in caplog.text.lower()


def test_windows_without_comtypes_degrades_to_inactive_monitor(caplog):
    monitor = WindowsDeviceMonitor(platform_name="Windows", comtypes_backend=None)

    with caplog.at_level(logging.INFO):
        monitor.start_monitoring(lambda event: None)

    assert monitor.available is False
    assert monitor.is_monitoring is True
    assert monitor.get_diagnostics()["unavailable_reason"] == "comtypes unavailable"
    assert "unavailable" in caplog.text.lower()


def test_injected_event_dispatches_to_consumer_callback():
    received = []
    monitor = WindowsDeviceMonitor(platform_name="Linux", comtypes_backend=None)
    monitor.start_monitoring(received.append)

    event = DeviceEvent(event_type=DeviceEventType.DEFAULT_CHANGED, device_id="dev-2")
    monitor.inject_event(event)
    monitor.drain_pending_events()

    assert received == [event]
    assert monitor.get_diagnostics()["queued_events"] == 0


def test_consumer_callback_error_is_logged_and_monitor_continues(caplog):
    received = []

    def callback(event):
        received.append(event)
        if len(received) == 1:
            raise RuntimeError("boom")

    monitor = WindowsDeviceMonitor(platform_name="Linux", comtypes_backend=None)
    monitor.start_monitoring(callback)

    first = DeviceEvent(event_type="removed", device_id="dev-1")
    second = DeviceEvent(event_type="added", device_id="dev-2")

    with caplog.at_level(logging.ERROR):
        monitor.inject_event(first)
        monitor.inject_event(second)
        monitor.drain_pending_events()

    assert received == [first, second]
    assert "Hot-plug consumer callback failed" in caplog.text
    assert monitor.is_monitoring is True


def test_backend_callback_queues_normalized_events_without_dispatching_immediately():
    received = []
    monitor = WindowsDeviceMonitor(platform_name="Windows", comtypes_backend=object())
    monitor.start_monitoring(received.append)

    monitor.queue_device_event(
        event_type="state_changed",
        device_id=" dev-3 ",
        friendly_name=" Speakers ",
        flow="render",
        role="console",
        state="disabled",
        source_metadata={"backend": "unit-test"},
    )

    assert received == []
    assert monitor.get_diagnostics()["queued_events"] == 1

    monitor.drain_pending_events()

    assert [event.device_id for event in received] == ["dev-3"]
    assert received[0].event_type is DeviceEventType.STATE_CHANGED
    assert received[0].source_metadata == {"backend": "unit-test"}


def test_queue_overflow_logs_and_drops_event(caplog):
    monitor = WindowsDeviceMonitor(
        platform_name="Windows",
        comtypes_backend=object(),
        max_queue_size=1,
    )
    monitor.start_monitoring(lambda event: None)
    monitor.inject_event(DeviceEvent(event_type="added", device_id="dev-1"))

    with caplog.at_level(logging.WARNING):
        monitor.inject_event(DeviceEvent(event_type="removed", device_id="dev-2"))

    assert "queue full" in caplog.text.lower()
    assert monitor.get_diagnostics()["dropped_events"] == 1
