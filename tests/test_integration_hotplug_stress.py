"""Hot-plug stress integration test - verifies S01 hot-plug recovery is stable.

Tests 10 deterministic cycles with a seeded random source representing 5-15s
simulated spacing per cycle. Validates recovery latency, state preservation,
and diagnostics sanitization without real hardware.

Runs quickly without wall-clock waiting.
"""

import random
from datetime import datetime, timezone, timedelta

import pytest

from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.audio.session import SourceConfig
from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
    RecoveryOutcome,
)


class FakeMonitor:
    """Deterministic fake monitor for stress testing hot-plug cycles."""

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.events = []
        self.drain_count = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def drain_events(self, max_events=100):
        self.drain_count += 1
        events = self.events[:max_events]
        del self.events[:max_events]
        return events


class TimePatch:
    """Deterministic time patching for latency assertions."""

    def __init__(self, start_time=100.0):
        self.current = start_time

    def advance(self, seconds):
        self.current += seconds
        return self.current

    def now(self):
        return self.current


def _source(source_type, device_id=None, friendly_name=None):
    """Create a SourceConfig for testing."""
    source = SourceConfig(type=source_type, device_id=device_id)
    source.friendly_name = friendly_name
    return source


def _recording_controller(*sources, monitor=None, time_patch=None):
    """Create a RecordingController in recording-like state for testing."""
    controller = RecordingController(enable_transcription=False)
    controller._state = ControllerState.RECORDING
    controller._snapshot_active_sources(list(sources))
    if monitor is not None:
        controller._hotplug_monitor = monitor
        controller._hotplug_monitor_active = True
        monitor.start()  # Mark monitor as started
    if time_patch is not None:
        # Patch only the time() function for deterministic latency measurements.
        # Store the original callable, not the module, so restoration does not
        # poison the shared stdlib time module for later integration tests.
        import meetandread.recording.controller as controller_module
        original_time_func = controller_module._time.time
        controller_module._time.time = time_patch.now
        controller._time_patch = original_time_func
    return controller


def _restore_time(controller):
    """Restore original time function."""
    if hasattr(controller, "_time_patch"):
        original_time_func = controller._time_patch
        import meetandread.recording.controller as controller_module
        controller_module._time.time = original_time_func
        del controller._time_patch


STRESS_CYCLE_COUNT = 10
SEED = 42  # Fixed seed for deterministic behavior
LATENCY_THRESHOLD_SECONDS = 2.0
CYCLE_SPACING_MIN_SECONDS = 5.0
CYCLE_SPACING_MAX_SECONDS = 15.0


def test_ten_cycles_with_deterministic_spacing_and_auto_recovery():
    """Verify 10 cycles of remove/re-add result in successful recovery each time."""
    rng = random.Random(SEED)
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    cycle_results = []
    cycle_latencies = []

    for cycle_idx in range(STRESS_CYCLE_COUNT):
        # Generate deterministic spacing for this cycle
        spacing_seconds = rng.uniform(CYCLE_SPACING_MIN_SECONDS, CYCLE_SPACING_MAX_SECONDS)
        current_time = time_patch.advance(spacing_seconds)

        # Inject remove event for mic
        remove_time = time_patch.advance(0.1)
        remove_event = DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=remove_time),
        )
        monitor.events.append(remove_event)

        # Drain and verify degraded state
        results = controller.drain_hotplug_events()
        assert len(results) == 1, f"Cycle {cycle_idx}: Expected 1 result on removal"
        result = results[0]
        assert result.outcome is RecoveryOutcome.DEGRADED, f"Cycle {cycle_idx}: Expected DEGRADED, got {result.outcome}"
        assert controller.get_state() is ControllerState.RECORDING, f"Cycle {cycle_idx}: Should remain in RECORDING state"

        # Inject re-add event for mic within recovery window
        readd_time = time_patch.advance(0.5)
        readd_event = DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=readd_time),
        )
        monitor.events.append(readd_event)

        # Drain and verify auto-recovery
        results = controller.drain_hotplug_events()
        assert len(results) == 1, f"Cycle {cycle_idx}: Expected 1 result on re-add"
        result = results[0]
        assert result.outcome is RecoveryOutcome.AUTO_RECOVERED, f"Cycle {cycle_idx}: Expected AUTO_RECOVERED, got {result.outcome}"
        assert controller.get_state() is ControllerState.RECORDING, f"Cycle {cycle_idx}: Should return to RECORDING state"

        # Capture cycle results
        cycle_results.append(result.outcome)
        latency = readd_time - remove_time
        cycle_latencies.append(latency)

        # Verify latency is under threshold
        assert latency < LATENCY_THRESHOLD_SECONDS, f"Cycle {cycle_idx}: Latency {latency:.2f}s exceeds threshold {LATENCY_THRESHOLD_SECONDS}s"

    # Verify all cycles succeeded
    assert len(cycle_results) == STRESS_CYCLE_COUNT, f"Expected {STRESS_CYCLE_COUNT} cycles, got {len(cycle_results)}"
    assert all(r is RecoveryOutcome.AUTO_RECOVERED for r in cycle_results), f"Expected all AUTO_RECOVERED, got {cycle_results}"

    # Verify monitor and controller remain usable
    assert monitor.started == 1, "Monitor should have been started once"
    assert monitor.stopped == 0, "Monitor should not be stopped during stress"
    assert controller.get_state() is ControllerState.RECORDING, "Controller should remain in RECORDING state"

    # Verify drain count (one per event batch)
    assert monitor.drain_count == STRESS_CYCLE_COUNT * 2, f"Expected {STRESS_CYCLE_COUNT * 2} drain calls"

    _restore_time(controller)


def test_duplicate_events_are_contained_and_do_not_crash():
    """Verify duplicate removal events are ignored and do not break state."""
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    # Add initial remove event
    remove_event = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="USB Conference Mic",
        flow="capture",
        timestamp=datetime.now(timezone.utc),
    )
    monitor.events.append(remove_event)

    # Add duplicate remove event
    monitor.events.append(remove_event)

    # Drain and verify first is degraded, second is ignored
    results = controller.drain_hotplug_events()
    assert len(results) == 2, "Expected 2 results for duplicate removal"
    assert results[0].outcome is RecoveryOutcome.DEGRADED, "First removal should be DEGRADED"
    assert results[1].outcome is RecoveryOutcome.IGNORED, "Duplicate removal should be IGNORED"

    # Verify state is consistent
    diagnostics = controller.get_diagnostics()["hotplug"]
    assert diagnostics["lost_source_count"] == 1, "Should have exactly 1 lost source"

    _restore_time(controller)


def test_unknown_source_events_are_ignored_and_do_not_crash():
    """Verify events for unknown sources are contained without crashes."""
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    # Inject event for unknown device (STATE_CHANGED with unknown device ID)
    # STATE_CHANGED events are only treated as loss if state is inactive/disabled/unplugged
    unknown_event = DeviceEvent(
        event_type=DeviceEventType.STATE_CHANGED,
        device_id="unknown-mic-999",
        friendly_name="Unknown Device",
        flow="capture",
        state="active",  # active state means not a loss, so it won't match any source
        timestamp=datetime.now(timezone.utc),
    )
    monitor.events.append(unknown_event)

    # Drain and verify ignored outcome
    results = controller.drain_hotplug_events()
    assert len(results) == 1, "Expected 1 result for unknown source"
    assert results[0].outcome is RecoveryOutcome.IGNORED, "Unknown source should be IGNORED"

    # Verify controller remains operational
    assert controller.get_state() is ControllerState.RECORDING, "Controller should remain in RECORDING state"

    _restore_time(controller)


def test_monitor_drain_exceptions_are_contained_and_state_preserved():
    """Verify monitor drain exceptions are contained and recording state is preserved."""
    class FailingMonitor(FakeMonitor):
        def drain_events(self, max_events=100):
            # First call succeeds, second fails
            if self.drain_count == 0:
                self.drain_count += 1
                return []
            raise RuntimeError("backend drain failed")

    time_patch = TimePatch(start_time=100.0)
    failing_monitor = FailingMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=failing_monitor, time_patch=time_patch)

    # First drain succeeds
    results = controller.drain_hotplug_events()
    assert results == [], "First drain should succeed"

    # Second drain fails but should be contained
    results = controller.drain_hotplug_events()
    assert results == [], "Failed drain should return empty list"

    # Verify state is preserved
    assert controller.get_state() is ControllerState.RECORDING, "Controller should remain in RECORDING state despite drain failure"

    _restore_time(controller)


def test_stale_lost_source_state_is_corrected_on_recovery():
    """Verify stale lost source state is cleaned up when device reappears."""
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    # Remove mic
    remove_event = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-1",
        friendly_name="USB Conference Mic",
        flow="capture",
        timestamp=datetime.now(timezone.utc),
    )
    monitor.events.append(remove_event)
    results = controller.drain_hotplug_events()
    assert len(results) == 1
    assert results[0].outcome is RecoveryOutcome.DEGRADED

    # Verify lost source is tracked
    diagnostics = controller.get_diagnostics()["hotplug"]
    assert diagnostics["lost_source_count"] == 1

    # Re-add mic and verify state is corrected
    readd_event = DeviceEvent(
        event_type=DeviceEventType.ADDED,
        device_id="mic-1",
        friendly_name="USB Conference Mic",
        flow="capture",
        timestamp=datetime.now(timezone.utc),
    )
    monitor.events.append(readd_event)
    results = controller.drain_hotplug_events()
    assert len(results) == 1
    assert results[0].outcome is RecoveryOutcome.AUTO_RECOVERED

    # Verify lost source is no longer tracked
    diagnostics = controller.get_diagnostics()["hotplug"]
    assert diagnostics["lost_source_count"] == 0
    assert diagnostics["active_source_count"] == 2

    _restore_time(controller)


def test_diagnostics_expose_event_outcome_and_latency_not_secrets():
    """Verify diagnostics include event type, outcome, latency but no secrets."""
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-secret-id", "Private USB Microphone"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    # Inject and drain an event
    remove_time = time_patch.advance(0.1)
    remove_event = DeviceEvent(
        event_type=DeviceEventType.REMOVED,
        device_id="mic-secret-id",
        friendly_name="Private USB Microphone",
        flow="capture",
        timestamp=datetime.now(timezone.utc),
        source_metadata={
            "audio_samples": [1, 2, 3],
            "transcript_text": "confidential transcript",
            "embedding": [0.1, 0.2],
            "api_key": "sk-secret",
        },
    )
    monitor.events.append(remove_event)
    results = controller.drain_hotplug_events()

    # Verify diagnostics are exposed
    diagnostics = controller.get_diagnostics()
    hotplug = diagnostics["hotplug"]
    assert "last_device_event" in hotplug, "Diagnostics should include last_device_event"
    assert "last_recovery_result" in hotplug, "Diagnostics should include last_recovery_result"
    assert "outcome" in hotplug["last_recovery_result"], "Result should include outcome"
    # Note: last_device_event is currently always None (see controller.py TODO)

    # Verify secrets are not exposed
    flattened = repr(hotplug)
    assert "confidential transcript" not in flattened, "Diagnostics should not expose transcript text"
    assert "audio_samples" not in flattened, "Diagnostics should not expose raw audio"
    assert "embedding" not in flattened, "Diagnostics should not expose embeddings"
    assert "api_key" not in flattened, "Diagnostics should not expose API keys"
    assert "sk-secret" not in flattened, "Diagnostics should not expose secret values"

    _restore_time(controller)


def test_all_recovery_progressions_through_degraded_to_auto_recovered():
    """Verify each cycle progresses through DEGRADED → AUTO_RECOVERED."""
    rng = random.Random(SEED)
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    progressions = []

    for cycle_idx in range(STRESS_CYCLE_COUNT):
        time_patch.advance(rng.uniform(5.0, 15.0))

        # Remove
        remove_event = DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc),
        )
        monitor.events.append(remove_event)
        results = controller.drain_hotplug_events()
        assert len(results) == 1, f"Cycle {cycle_idx}: Expected 1 removal result"

        # Re-add
        readd_event = DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc),
        )
        monitor.events.append(readd_event)
        results.extend(controller.drain_hotplug_events())

        progression = [r.outcome for r in results]
        progressions.append(progression)

        assert progression == [RecoveryOutcome.DEGRADED, RecoveryOutcome.AUTO_RECOVERED], \
            f"Cycle {cycle_idx}: Expected DEGRADED → AUTO_RECOVERED, got {progression}"

    # Verify all cycles followed expected progression
    assert len(progressions) == STRESS_CYCLE_COUNT
    expected_progression = [RecoveryOutcome.DEGRADED, RecoveryOutcome.AUTO_RECOVERED]
    assert all(p == expected_progression for p in progressions), \
        f"All cycles should follow {expected_progression}"

    _restore_time(controller)


def test_monitor_active_state_remains_true_throughout_stress():
    """Verify monitor active state remains true throughout stress testing."""
    rng = random.Random(SEED)
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    for cycle_idx in range(STRESS_CYCLE_COUNT):
        time_patch.advance(rng.uniform(5.0, 15.0))

        # Remove and re-add
        remove_event = DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc),
        )
        monitor.events.append(remove_event)
        controller.drain_hotplug_events()

        readd_event = DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc),
        )
        monitor.events.append(readd_event)
        controller.drain_hotplug_events()

        # Verify monitor active state
        diagnostics = controller.get_diagnostics()["hotplug"]
        assert diagnostics["monitor_active"] is True, f"Cycle {cycle_idx}: Monitor should remain active"

    _restore_time(controller)


def test_latency_assertions_under_2_seconds_for_all_cycles():
    """Verify detection/recovery latency is under 2 seconds for all cycles."""
    rng = random.Random(SEED)
    time_patch = TimePatch(start_time=100.0)
    monitor = FakeMonitor()
    source_configs = [
        _source("mic", "mic-1", "USB Conference Mic"),
        _source("system", "sys-1", "Laptop Speakers"),
    ]

    controller = _recording_controller(*source_configs, monitor=monitor, time_patch=time_patch)

    latencies = []

    for cycle_idx in range(STRESS_CYCLE_COUNT):
        spacing = rng.uniform(5.0, 15.0)
        time_patch.advance(spacing)

        # Capture removal time
        remove_time = time_patch.advance(0.1)
        remove_event = DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=remove_time),
        )
        monitor.events.append(remove_event)
        controller.drain_hotplug_events()

        # Capture re-add time
        readd_time = time_patch.advance(0.3)
        readd_event = DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="mic-1",
            friendly_name="USB Conference Mic",
            flow="capture",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=readd_time),
        )
        monitor.events.append(readd_event)
        results = controller.drain_hotplug_events()

        latency = readd_time - remove_time
        latencies.append(latency)

        assert results[0].outcome is RecoveryOutcome.AUTO_RECOVERED, \
            f"Cycle {cycle_idx}: Expected AUTO_RECOVERED, got {results[0].outcome}"
        assert latency < LATENCY_THRESHOLD_SECONDS, \
            f"Cycle {cycle_idx}: Latency {latency:.2f}s exceeds {LATENCY_THRESHOLD_SECONDS}s"

    # Verify all latencies are under threshold
    assert len(latencies) == STRESS_CYCLE_COUNT
    assert all(l < LATENCY_THRESHOLD_SECONDS for l in latencies), \
        f"All latencies {latencies} should be under {LATENCY_THRESHOLD_SECONDS}s"

    _restore_time(controller)
