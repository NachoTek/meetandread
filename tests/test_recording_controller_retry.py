import time

from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
)


class FakeSession:
    """Fake AudioSession that provides mutable SessionStats with retry fields."""

    def __init__(self):
        self._state_val = ControllerState.IDLE
        self._started_config = None
        # Eager init SessionStats to ensure same instance across all calls
        from meetandread.audio.session import SessionStats
        self._stats = SessionStats()
        self._stop_return = None

    def start(self, config):
        self._started_config = config
        self._state_val = ControllerState.RECORDING

    def stop(self):
        self._state_val = ControllerState.FINALIZED
        return self._stop_return or "recording.wav"

    def get_state(self):
        return self._state_val

    def get_stats(self):
        """Return the same mutable SessionStats instance across all calls."""
        return self._stats

    def get_error(self):
        return None


def _recording_controller(monkeypatch, *, now=100.0, enable_transcription=False):
    """Build a RecordingController with faked AudioSession."""
    ctrl = RecordingController(enable_transcription=enable_transcription)
    fake_session = FakeSession()
    monkeypatch.setattr(
        "meetandread.recording.controller.AudioSession",
        lambda: fake_session,
    )
    monkeypatch.setattr("meetandread.recording.controller._time.time", lambda: now)
    return ctrl, fake_session


def test_default_session_stats_retry_fields_are_backward_compatible(monkeypatch):
    """T01: Prove new retry stats default to safe backward-compatible values."""
    ctrl, fake_session = _recording_controller(monkeypatch)

    stats = fake_session.get_stats()
    assert stats.retry_attempts == 0
    assert stats.retry_outcome == "none"
    assert stats.failed_sources == []
    assert stats.fallback_sources == []

    # Verify diagnostics expose defaults
    diag = ctrl.get_diagnostics().get("session", {})
    assert diag.get("retry_attempts") == 0
    assert diag.get("retry_outcome") == "none"
    assert diag.get("failed_sources") == []
    assert diag.get("fallback_sources") == []


def test_begin_start_retry_sequence_resets_diagnostics_and_sets_stats_to_pending(monkeypatch):
    """T01: begin_start_retry_sequence clears retry events and sets retry_outcome to pending."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    # Record some prior retry state (must start with begin_start_retry_sequence)
    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        device_id="sys-1",
        friendly_name="Speakers (Realtek)",
        error=RuntimeError("endpoint unavailable"),
    )
    ctrl.record_start_retry_outcome(
        outcome="failed",
        failed_sources=["system"],
    )

    # Verify state was recorded
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 1
    assert stats.retry_outcome == "failed"
    assert stats.failed_sources == ["system"]

    # Reset via begin_start_retry_sequence
    ctrl.begin_start_retry_sequence()

    # Verify diagnostics cleared (events list)
    diag = ctrl.get_diagnostics()
    assert diag["retry"]["started"] is True
    assert diag["retry"]["events"] == []

    # Verify stats reset to pending/zero
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 0
    assert stats.retry_outcome == "pending"
    assert stats.failed_sources == []
    assert stats.fallback_sources == []


def test_record_start_retry_attempt_emits_retrying_state(monkeypatch):
    """T01: record_start_retry_attempt emits RETRYING state through callback."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    state_changes = []
    ctrl.on_state_change = lambda state: state_changes.append(state)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=RuntimeError("endpoint unavailable"),
    )

    # RETRYING state should be emitted
    assert ControllerState.RETRYING in state_changes
    assert ctrl.get_state() is ControllerState.RETRYING


def test_record_start_retry_attempt_increments_attempts_and_tracks_failed_sources(monkeypatch):
    """T01: Retry attempt metadata is captured in session stats."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        device_id="sys-1",
        error=RuntimeError("endpoint unavailable"),
    )

    # Verify stats persisted (FakeSession.get_stats() returns same instance)
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 1
    assert stats.retry_outcome == "retrying"
    assert "system" in stats.failed_sources

    # Second attempt with different source
    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="mic",
        device_id="mic-1",
        error=RuntimeError("device busy"),
    )

    # Stats should accumulate attempts and failed sources (same instance)
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 2
    assert set(stats.failed_sources) == {"system", "mic"}


def test_record_start_retry_attempt_sanitizes_inputs(monkeypatch):
    """T01: Retry metadata sanitizes truncates long/error values."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="  SYSTEM  ",  # Should be lowercased and stripped
        device_id=None,  # Should remain None
        friendly_name="Very Long Device Name That Exceeds Normal Length And Should Be Truncated",
        error=RuntimeError("A" * 300),  # Long error message
    )

    diag = ctrl.get_diagnostics()
    events = diag["retry"]["events"]
    assert len(events) == 1
    assert events[0]["source_type"] == "system"
    assert events[0]["device_id"] is None
    assert len(events[0]["friendly_name"]) <= 120
    assert len(events[0]["error_message"]) <= 120


def test_record_start_retry_outcome_sets_final_state_and_stats(monkeypatch):
    """T01: Final retry outcome is captured in stats and sanitized."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=3,
        backoff_seconds=4.0,
        source_type="system",
    )
    ctrl.record_start_retry_outcome(
        outcome="fallback to mic only",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify stats persisted (same SessionStats instance)
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 3
    assert stats.retry_outcome == "fallback_to_mic_only"
    assert stats.failed_sources == ["system"]
    assert stats.fallback_sources == ["mic"]

    # Verify diagnostics expose outcome (session stats are fetched fresh)
    diag = ctrl.get_diagnostics()
    assert diag["session"]["retry_outcome"] == "fallback_to_mic_only"
    assert diag["session"]["failed_sources"] == ["system"]
    assert diag["session"]["fallback_sources"] == ["mic"]


def test_is_busy_includes_retrying_state(monkeypatch):
    """T01: RETRYING state is considered busy (like STARTING/STOPPING)."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    assert ctrl.is_busy() is False

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
    )

    # RETRYING should report as busy
    assert ctrl.is_busy() is True


def test_retry_diagnostics_includes_elapsed_time_and_event_sequence(monkeypatch):
    """T01: Retry diagnostics expose elapsed time and ordered event list."""
    # Mock monotonic for elapsed time tracking
    monotonic_time = [100.0]
    def advance_time(delta):
        monotonic_time[0] += delta
        return monotonic_time[0]

    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)
    monkeypatch.setattr("meetandread.recording.controller._time.monotonic", lambda: monotonic_time[0])

    ctrl.begin_start_retry_sequence()
    advance_time(0.5)  # First attempt happens at 100.5
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
    )

    # Advance time by 2.5 seconds
    advance_time(2.5)

    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="system",
    )

    diag = ctrl.get_diagnostics()
    events = diag["retry"]["events"]

    assert len(events) == 2
    assert events[0]["attempt_number"] == 1
    assert events[0]["backoff_seconds"] == 1.0
    assert events[0]["elapsed_seconds"] >= 0.0
    assert events[1]["attempt_number"] == 2
    assert events[1]["elapsed_seconds"] >= 2.5  # ~2.5s elapsed since start


def test_retry_metadata_does_not_expose_raw_audio_or_transcript_data(monkeypatch):
    """T01: Retry metadata is sanitized (no audio/transcript/secrets)."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
        error=RuntimeError("AudioEndpointOpen failed"),
    )

    diag = ctrl.get_diagnostics()
    retry_events = diag["retry"]["events"]

    # Event fields are sanitized text only
    for event in retry_events:
        assert "raw" not in str(event).lower()
        assert "pcm" not in str(event).lower()
        assert "transcript" not in str(event).lower()
        assert "embedding" not in str(event).lower()
        assert isinstance(event.get("attempt_number"), int)
        assert isinstance(event.get("backoff_seconds"), (int, float))
        assert isinstance(event.get("elapsed_seconds"), (int, float))


def test_duplicate_failed_sources_are_deduplicated_in_stats(monkeypatch):
    """T01: Recording same failed source twice does not duplicate in stats."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()

    # Record same failed source twice
    ctrl.record_start_retry_attempt(
        attempt_number=1,
        backoff_seconds=1.0,
        source_type="system",
    )
    ctrl.record_start_retry_attempt(
        attempt_number=2,
        backoff_seconds=2.0,
        source_type="system",
    )

    # Stats should deduplicate failed sources (same SessionStats instance)
    stats = fake_session.get_stats()
    assert stats.failed_sources == ["system"]  # Deduplicated


def test_retry_outcome_sanitizes_to_safe_identifier(monkeypatch):
    """T01: Outcome strings are sanitized to lowercase underscore-safe IDs."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_outcome(
        outcome="  Fallback to MIC Only  ",  # Spaces and mixed case
        failed_sources=["system"],
    )

    # Stats should be sanitized (same SessionStats instance)
    stats = fake_session.get_stats()
    assert stats.retry_outcome == "fallback_to_mic_only"


def test_session_stats_retry_fields_are_mutable_and_resettable(monkeypatch):
    """T01: SessionStats retry fields are mutable and reset on new session."""
    ctrl, fake_session = _recording_controller(monkeypatch, now=100.0)

    # Record retry state
    ctrl.begin_start_retry_sequence()
    ctrl.record_start_retry_attempt(
        attempt_number=3,
        backoff_seconds=4.0,
        source_type="system",
    )
    ctrl.record_start_retry_outcome(
        outcome="fallback",
        failed_sources=["system"],
        fallback_sources=["mic"],
    )

    # Verify stats persisted (same SessionStats instance)
    stats = fake_session.get_stats()
    assert stats.retry_attempts == 3
    assert stats.failed_sources == ["system"]

    # Simulate session reset (new SessionStats with defaults)
    from meetandread.audio.session import SessionStats
    fake_session._stats = SessionStats()

    new_stats = fake_session.get_stats()
    assert new_stats.retry_attempts == 0
    assert new_stats.retry_outcome == "none"
    assert new_stats.failed_sources == []
    assert new_stats.fallback_sources == []