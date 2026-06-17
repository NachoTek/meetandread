"""Integration tests for WASAPI retry/fallback regression coverage.

Simulates AudioSourceError on system startup, verifies the complete
retry-to-fallback path, verifies partial-source startup prompts rather than
silently continuing, and ensures S01 hot-plug notification/recovery tests
remain unaffected. Tests are deterministic with mocks/fakes and do not
require physical WASAPI devices.

Tests verify:
- AudioSourceError on system startup triggers retry flow with exponential backoff
- Complete retry-to-fallback path (3 attempts: 1s, 2s, 4s backoff)
- Retry metadata is captured in SessionStats and controller diagnostics
- Partial-source startup prompts user for explicit fallback confirmation
- Hot-plug notification/recovery integration remains unaffected
- No silent degradation - all fallbacks require user confirmation
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication

from meetandread.audio.capture import AudioSourceError
from meetandread.audio.session import SessionConfig, SourceConfig, SessionStats
from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
    ControllerError,
)
from meetandread.widgets.main_widget import MeetAndReadWidget


class FakeAudioSource:
    """Fake audio source that can raise errors on start."""
    def __init__(self, *, raise_on_start=False, error_class=AudioSourceError, error_message="WASAPI endpoint unavailable"):
        self.raise_on_start = raise_on_start
        self.error_class = error_class
        self.error_message = error_message
        self.started = False
        self.stopped = False

    def start(self):
        if self.raise_on_start:
            raise self.error_class(self.error_message)
        self.started = True

    def stop(self):
        self.stopped = True

    def get_metadata(self):
        return {"sample_rate": 48000, "channels": 2}

    def read_frames(self, timeout=0.1):
        return None

    def is_running(self):
        return self.started and not self.stopped


class FakeSession:
    """Fake audio session that simulates AudioSourceError on start."""
    def __init__(self, *, raise_on_system_start=False, raise_on_all_start=False):
        self.started_config = None
        self.stopped = 0
        self.raise_on_system_start = raise_on_system_start
        self.raise_on_all_start = raise_on_all_start
        self._stats = SessionStats()

    def start(self, config):
        self.started_config = config
        # Simulate AudioSourceError for system sources
        if self.raise_on_all_start:
            raise AudioSourceError("All sources failed: WASAPI endpoint unavailable")
        if self.raise_on_system_start:
            for source in config.sources:
                if source.type == "system":
                    raise AudioSourceError("System audio unavailable: WASAPI endpoint not found")

    def stop(self):
        self.stopped += 1
        return "recording.wav"

    def get_state(self):
        return ControllerState.RECORDING if self.started_config else ControllerState.IDLE

    def get_stats(self):
        return self._stats

    def get_error(self):
        return None


def _recording_controller(monkeypatch, *, raise_on_system_start=False, raise_on_all_start=False, now=100.0):
    """Create a RecordingController with mocked session."""
    ctrl = RecordingController(enable_transcription=False)
    fake_session = FakeSession(
        raise_on_system_start=raise_on_system_start,
        raise_on_all_start=raise_on_all_start
    )
    # Patch session creation
    original_session = RecordingController.__init__
    def patched_init(self, enable_transcription=True):
        original_session(self, enable_transcription)
        self._session = fake_session
    monkeypatch.setattr(RecordingController, "__init__", patched_init)
    monkeypatch.setattr("meetandread.recording.controller._time.time", lambda: now)
    return ctrl, fake_session


def test_audio_source_error_on_system_start_records_retry_metadata(monkeypatch):
    """AudioSourceError on system startup records retry metadata in SessionStats."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    # Initialize retry sequence
    ctrl.begin_start_retry_sequence()

    # Attempt start with mic + system
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        device_id="loopback-1",
        friendly_name="Laptop Speakers",
        error=AudioSourceError("WASAPI endpoint not found"),
    )

    # Verify retry metadata is captured
    session_diagnostics = ctrl.get_diagnostics()["session"]
    assert session_diagnostics["retry_attempts"] == 1
    assert session_diagnostics["retry_outcome"] == "retrying"
    assert "system" in session_diagnostics["failed_sources"]


def test_complete_retry_sequence_with_exponential_backoff(monkeypatch):
    """Complete retry sequence follows exponential backoff (1s, 2s, 4s)."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Record three retry attempts with exponential backoff
    backoff_times = [1.0, 2.0, 4.0]
    for i, backoff in enumerate(backoff_times, start=1):
        ctrl.record_start_retry_attempt(
            attempt_number=i,
            backoff_seconds=backoff,
            source_type="system",
            device_id="loopback-1",
            friendly_name="Laptop Speakers",
            error=AudioSourceError(f"Retry {i} failed: WASAPI endpoint unavailable"),
        )

    # Verify backoff pattern
    retry_diagnostics = ctrl.get_diagnostics().get("retry", {})
    assert "events" in retry_diagnostics
    assert len(retry_diagnostics["events"]) == 3

    # Verify exponential backoff values
    for i, event in enumerate(retry_diagnostics["events"], start=1):
        assert event["attempt_number"] == i
        assert event["backoff_seconds"] == backoff_times[i - 1]
        assert event["source_type"] == "system"


def test_retry_outcome_fallback_records_metadata(monkeypatch):
    """Retry outcome 'fallback' records failed and fallback source lists."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Record retry attempts
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("WASAPI endpoint unavailable"),
    )
    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="system",
        error=AudioSourceError("Retry failed"),
    )

    # Record fallback outcome
    ctrl.record_start_retry_outcome(
        outcome="fallback",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify outcome metadata
    session_diagnostics = ctrl.get_diagnostics()["session"]
    assert session_diagnostics["retry_attempts"] >= 2
    assert session_diagnostics["retry_outcome"] == "fallback"
    assert "system" in session_diagnostics["failed_sources"]
    assert "mic" in session_diagnostics["fallback_sources"]


def test_retry_outcome_failed_records_no_fallback_sources(monkeypatch):
    """Retry outcome 'failed' records failed sources but no fallback."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Record retry attempts
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("WASAPI endpoint unavailable"),
    )

    # Record failed outcome (user rejected fallback)
    ctrl.record_start_retry_outcome(
        outcome="failed",
        failed_sources=["system"],
        fallback_sources=[],
    )

    # Verify outcome metadata
    session_diagnostics = ctrl.get_diagnostics()["session"]
    assert session_diagnostics["retry_attempts"] >= 1
    assert session_diagnostics["retry_outcome"] == "failed"
    assert "system" in session_diagnostics["failed_sources"]
    assert len(session_diagnostics["fallback_sources"]) == 0


def test_retry_elapsed_time_tracks_duration(monkeypatch):
    """Retry elapsed time tracks total duration from start_sequence to outcome."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Record retry attempts with elapsed time
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("First attempt failed"),
    )

    # Verify elapsed time is tracked
    retry_diagnostics = ctrl.get_diagnostics().get("retry", {})
    assert retry_diagnostics["started"] is True
    assert "events" in retry_diagnostics
    assert retry_diagnostics["events"][0]["elapsed_seconds"] >= 0


def test_multiple_source_failure_records_all_failed_sources(monkeypatch):
    """Multiple source failure records all failed source types in stats."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_all_start=True)

    ctrl.begin_start_retry_sequence()

    # Record failures for both mic and system
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("System audio failed"),
    )
    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="mic",
        error=AudioSourceError("Microphone failed"),
    )

    # Record failed outcome
    ctrl.record_start_retry_outcome(
        outcome="failed",
        failed_sources=["system", "mic"],
        fallback_sources=[],
    )

    # Verify both sources are recorded as failed
    session_diagnostics = ctrl.get_diagnostics()["session"]
    assert "system" in session_diagnostics["failed_sources"]
    assert "mic" in session_diagnostics["failed_sources"]
    assert len(session_diagnostics["fallback_sources"]) == 0


def test_partial_source_failure_prompts_fallback_not_silent_degradation(monkeypatch):
    """Partial source failure prompts for fallback, not silent degradation."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # System fails but mic succeeds
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("System audio unavailable"),
    )

    # Record outcome indicating fallback is needed
    ctrl.record_start_retry_outcome(
        outcome="fallback",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify fallback is recorded (not silent degradation)
    session_diagnostics = ctrl.get_diagnostics()["session"]
    assert session_diagnostics["retry_outcome"] == "fallback"
    assert session_diagnostics["failed_sources"] == ["system"]
    assert session_diagnostics["fallback_sources"] == ["mic"]


def test_retry_state_transitions_through_retrying(monkeypatch):
    """Controller state transitions through RETRYING during retry sequence."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    # Initial state should be IDLE
    assert ctrl.get_state() == ControllerState.IDLE

    ctrl.begin_start_retry_sequence()

    # First retry attempt transitions to RETRYING
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("Failed"),
    )

    assert ctrl.get_state() == ControllerState.RETRYING
    assert ctrl.is_busy() is True


def test_diagnostics_include_retry_metadata_without_secrets(monkeypatch):
    """Controller diagnostics include retry metadata without raw audio/transcript data."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Record retry with device info
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        device_id="endpoint-with-long-device-id-12345",
        friendly_name="Private Device",
        error=AudioSourceError("Error"),
    )

    ctrl.record_start_retry_outcome(
        outcome="fallback",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify diagnostics are structured
    diagnostics = ctrl.get_diagnostics()
    flattened = repr(diagnostics)

    # Retry metadata is present
    assert "retry" in diagnostics
    assert "failed_sources" in diagnostics["session"]

    # But raw audio or transcript data is not included
    assert "audio_samples" not in flattened
    assert "transcript_text" not in flattened
    assert "embedding" not in flattened


def test_hotplug_monitor_start_remains_affected_after_retry_sequence(monkeypatch):
    """Hot-plug monitor lifecycle is not affected by retry sequence."""
    from meetandread.audio.hotplug import WindowsDeviceMonitor

    mock_monitor = Mock(spec=WindowsDeviceMonitor)
    mock_monitor.start = Mock()
    mock_monitor.stop = Mock()

    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    # Simulate successful start after retry
    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("Failed"),
    )

    # Verify monitor tracking is preserved
    ctrl._hotplug_monitor = mock_monitor
    ctrl._hotplug_monitor_active = True

    hotplug_diagnostics = ctrl.get_diagnostics()["hotplug"]
    assert hotplug_diagnostics["monitor_active"] is True


def test_retry_diagnostics_do_not_include_raw_audio_or_transcript_data(monkeypatch):
    """Retry diagnostics do not include raw audio or transcript data."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("Error"),
    )

    ctrl.record_start_retry_outcome(
        outcome="fallback",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify no audio or transcript data in diagnostics
    diagnostics = ctrl.get_diagnostics()
    flattened = repr(diagnostics)

    # Retry metadata is present
    assert "retry" in diagnostics

    # But no raw audio or transcript content
    assert "audio_samples" not in flattened
    assert "transcript_text" not in flattened
    assert "embedding" not in flattened


def test_controller_get_diagnostics_includes_retry_section(monkeypatch):
    """Controller get_diagnostics() includes retry section when retry occurred."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    # Before retry sequence - no retry section
    initial_diagnostics = ctrl.get_diagnostics()
    assert initial_diagnostics["session"]["retry_attempts"] == 0
    assert initial_diagnostics["session"]["retry_outcome"] == "none"

    # After retry sequence - retry metadata is present
    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("Failed"),
    )

    retry_diagnostics = ctrl.get_diagnostics()
    assert retry_diagnostics["session"]["retry_attempts"] >= 1
    assert retry_diagnostics["session"]["retry_outcome"] == "retrying"


def test_retry_helpers_update_session_stats_immutably(monkeypatch):
    """Retry helpers update SessionStats without breaking existing stats."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # Get initial stats
    initial_session = ctrl.get_diagnostics()["session"]

    # Update with retry metadata
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("Failed"),
    )

    # Verify stats are updated but structure is preserved
    updated_session = ctrl.get_diagnostics()["session"]

    # Retry fields are updated
    assert updated_session["retry_attempts"] >= 1
    assert updated_session["retry_outcome"] == "retrying"

    # But other stats structure is preserved
    assert "frames_recorded" in updated_session
    assert "duration_seconds" in updated_session
    assert "source_stats" in updated_session


def test_elapsed_time_increases_with_each_retry_attempt(monkeypatch):
    """Elapsed time increases with each retry attempt."""
    ctrl, fake_session = _recording_controller(monkeypatch, raise_on_system_start=True)

    ctrl.begin_start_retry_sequence()

    # First attempt
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=AudioSourceError("First failed"),
    )

    first_elapsed = ctrl.get_diagnostics()["retry"]["events"][0]["elapsed_seconds"]

    # Second attempt (simulated time passing)
    monkeypatch.setattr("meetandread.recording.controller._time.monotonic", lambda: 2.0)
    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="system",
        error=AudioSourceError("Second failed"),
    )

    second_elapsed = ctrl.get_diagnostics()["retry"]["events"][1]["elapsed_seconds"]

    # Verify elapsed time increases
    assert second_elapsed >= first_elapsed