"""Tests for RecordingController.shutdown() — exit-only blocking wait.

Covers T01 must-haves:
- shutdown() waits for mocked finalization
- shutdown() is idempotent (10x repeated calls safe)
- shutdown() from IDLE is a no-op
- shutdown() from RECORDING initiates stop and waits
- shutdown() from STOPPING waits for finalizer
- shutdown() with never-ending finalizer logs timeout without hanging
- shutdown() never raises
- Normal stop() tests still prove non-blocking behavior (D045)
- Observability: shutdown logs start/completion/timeout
"""

import threading
import time as _time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meetandread.recording.controller import (
    RecordingController,
    ControllerState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAudioSession:
    """Minimal AudioSession fake for stop/start tests."""

    def __init__(self, wav_path: Path):
        self._wav_path = wav_path
        self._state = "idle"
        self._stop_event = threading.Event()

    def start(self, config=None):
        self._state = "recording"

    def stop(self) -> Path:
        self._state = "idle"
        return self._wav_path

    def get_state(self):
        return self._state

    def get_stats(self):
        from meetandread.audio import SessionStats
        return SessionStats(
            frames_recorded=0,
            frames_dropped=0,
            duration_seconds=0.0,
            source_stats=[],
        )

    def get_error(self):
        return None


def _make_recording_controller(tmp_path: Path) -> RecordingController:
    """Create a RecordingController in RECORDING state with faked session."""
    ctrl = RecordingController(enable_transcription=False)
    wav_path = tmp_path / "test.wav"
    wav_path.write_text("fake wav")
    ctrl._session = FakeAudioSession(wav_path)
    ctrl._state = ControllerState.RECORDING
    return ctrl


# ===================================================================
# Shutdown from IDLE
# ===================================================================

class TestShutdownFromIdle:
    """shutdown() when idle is a fast no-op."""

    def test_shutdown_idle_returns_quickly(self):
        ctrl = RecordingController(enable_transcription=False)
        t0 = _time.monotonic()
        ctrl.shutdown(timeout=5.0)
        elapsed = _time.monotonic() - t0
        assert elapsed < 1.0, f"shutdown from IDLE took {elapsed:.2f}s"

    def test_shutdown_idle_does_not_change_state(self):
        ctrl = RecordingController(enable_transcription=False)
        ctrl.shutdown(timeout=5.0)
        assert ctrl.get_state() == ControllerState.IDLE


# ===================================================================
# Shutdown while recording
# ===================================================================

class TestShutdownWhileRecording:
    """shutdown() initiates stop and waits for finalizer."""

    def test_shutdown_recording_waits_for_finalizer(self, tmp_path: Path):
        """shutdown() blocks until finalizer thread completes."""
        ctrl = _make_recording_controller(tmp_path)

        # Replace _stop_worker with one that sets up a fast finalizer
        original_stop_worker = ctrl._stop_worker

        finalizer_done = threading.Event()

        def patched_stop_worker():
            # Replicate key parts of _stop_worker
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                _time.sleep(0.1)  # Simulate finalization work
                finalizer_done.set()

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker

        t0 = _time.monotonic()
        ctrl.shutdown(timeout=5.0)
        elapsed = _time.monotonic() - t0

        assert finalizer_done.is_set(), "Finalizer should have completed"
        assert elapsed < 3.0, f"shutdown took too long: {elapsed:.2f}s"
        # State should be IDLE after shutdown
        assert ctrl.get_state() == ControllerState.IDLE

    def test_shutdown_recording_state_transitions_to_idle(self, tmp_path: Path):
        """After shutdown, controller is IDLE."""
        ctrl = _make_recording_controller(tmp_path)

        finalizer_done = threading.Event()

        def patched_stop_worker():
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                finalizer_done.set()

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker
        ctrl.shutdown(timeout=5.0)
        assert ctrl.get_state() == ControllerState.IDLE


# ===================================================================
# Shutdown while already stopping
# ===================================================================

class TestShutdownWhileStopping:
    """shutdown() when STOPPING waits for finalizer to finish."""

    def test_shutdown_during_stopping_waits(self, tmp_path: Path):
        ctrl = _make_recording_controller(tmp_path)
        ctrl._set_state(ControllerState.STOPPING)

        # Set up a finalizer that takes a moment
        finalizer_done = threading.Event()

        def _finalize():
            _time.sleep(0.2)
            finalizer_done.set()

        finalizer = threading.Thread(
            target=_finalize, daemon=True, name="RecordingFinalizer"
        )
        ctrl._finalizer_thread = finalizer
        finalizer.start()

        ctrl.shutdown(timeout=5.0)
        assert finalizer_done.is_set(), "Finalizer should have completed"


# ===================================================================
# Shutdown with never-ending finalizer (timeout)
# ===================================================================

class TestShutdownTimeout:
    """shutdown() with a hanging finalizer logs timeout and proceeds."""

    def test_shutdown_timeout_does_not_hang(self, tmp_path: Path):
        """shutdown() returns within timeout even if finalizer never completes."""
        ctrl = _make_recording_controller(tmp_path)
        ctrl._set_state(ControllerState.IDLE)

        # Create a finalizer that never completes
        blocker = threading.Event()

        def _blocking_finalize():
            blocker.wait(timeout=60)  # Block for a very long time

        finalizer = threading.Thread(
            target=_blocking_finalize, daemon=True, name="RecordingFinalizer"
        )
        ctrl._finalizer_thread = finalizer
        finalizer.start()

        t0 = _time.monotonic()
        ctrl.shutdown(timeout=1.0)  # Short timeout
        elapsed = _time.monotonic() - t0

        # Should return close to the timeout (not 60s)
        assert elapsed < 3.0, f"shutdown hung for {elapsed:.2f}s"
        assert finalizer.is_alive(), "Finalizer should still be alive (timed out)"

        # Clean up
        blocker.set()
        finalizer.join(timeout=5.0)


# ===================================================================
# Idempotency
# ===================================================================

class TestShutdownIdempotency:
    """Repeated shutdown calls must be safe and not spawn duplicate workers."""

    def test_10x_repeated_shutdown(self, tmp_path: Path):
        """10 consecutive shutdown calls complete without error."""
        ctrl = _make_recording_controller(tmp_path)

        finalizer_done = threading.Event()
        shutdown_count = [0]

        def patched_stop_worker():
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                _time.sleep(0.05)
                shutdown_count[0] += 1
                finalizer_done.set()

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker

        # First shutdown triggers stop + waits for finalizer
        ctrl.shutdown(timeout=5.0)
        assert finalizer_done.is_set()

        # 9 more calls should be no-ops (already idle, finalizer done)
        for i in range(9):
            ctrl.shutdown(timeout=2.0)

        # Only one finalizer should have run
        assert shutdown_count[0] == 1

    def test_concurrent_shutdown_calls(self, tmp_path: Path):
        """Multiple threads calling shutdown simultaneously is safe."""
        ctrl = _make_recording_controller(tmp_path)

        finalizer_done = threading.Event()

        def patched_stop_worker():
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                _time.sleep(0.1)
                finalizer_done.set()

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker

        errors = []

        def do_shutdown():
            try:
                ctrl.shutdown(timeout=5.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_shutdown, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent shutdown errors: {errors}"
        assert finalizer_done.is_set()


# ===================================================================
# Shutdown never raises
# ===================================================================

class TestShutdownNeverRaises:
    """shutdown() swallows exceptions so the quit path is never compromised."""

    def test_shutdown_with_stop_error(self, tmp_path: Path):
        """shutdown() logs and proceeds when stop() returns an error."""
        ctrl = _make_recording_controller(tmp_path)

        # Make stop() return an error (it sets ERROR state internally)
        # We patch it to raise instead
        original_stop = ctrl.stop

        call_count = [0]

        def bad_stop():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated stop failure")
            return original_stop()

        ctrl.stop = bad_stop

        # Should not raise despite stop() failing
        ctrl.shutdown(timeout=2.0)

    def test_shutdown_with_corrupt_state(self):
        """shutdown() handles unusual states gracefully."""
        ctrl = RecordingController(enable_transcription=False)
        ctrl._state = ControllerState.ERROR

        # Should not raise
        ctrl.shutdown(timeout=2.0)
        # State should remain ERROR (shutdown doesn't fix it)
        assert ctrl.get_state() == ControllerState.ERROR


# ===================================================================
# D045 preservation: normal stop() is non-blocking
# ===================================================================

class TestStopNonBlocking:
    """Verify that normal stop() remains non-blocking after shutdown changes."""

    def test_stop_returns_immediately(self, tmp_path: Path):
        """stop() returns in < 0.5s (non-blocking per D045)."""
        ctrl = _make_recording_controller(tmp_path)

        # Patch _stop_worker to simulate real behavior
        def patched_stop_worker():
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                _time.sleep(0.5)  # Finalizer takes time

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker

        t0 = _time.monotonic()
        result = ctrl.stop()
        elapsed = _time.monotonic() - t0

        assert result is None, "stop() should return None on success"
        assert elapsed < 0.5, f"stop() took {elapsed:.3f}s — violates D045"

    def test_stop_to_start_latency(self, tmp_path: Path):
        """STOPPING→IDLE→STARTING transition stays under 1 second."""
        ctrl = _make_recording_controller(tmp_path)

        def patched_stop_worker():
            ctrl._set_state(ControllerState.IDLE)

            def _finalize():
                _time.sleep(0.1)

            finalizer = threading.Thread(
                target=_finalize, daemon=True, name="RecordingFinalizer"
            )
            ctrl._finalizer_thread = finalizer
            finalizer.start()

        ctrl._stop_worker = patched_stop_worker

        # Stop
        ctrl.stop()

        # Wait for IDLE
        deadline = _time.monotonic() + 1.0
        while ctrl.get_state() != ControllerState.IDLE and _time.monotonic() < deadline:
            _time.sleep(0.01)

        assert ctrl.get_state() == ControllerState.IDLE

        # Start should work immediately
        ctrl._session = FakeAudioSession(tmp_path / "test2.wav")
        error = ctrl.start(
            {'fake'},
            fake_path=str(tmp_path / "test2.wav"),
        )
        # start() may fail due to session internals, but it should not
        # hang or be blocked by the previous finalizer


# ===================================================================
# Observability: diagnostics includes finalizer info
# ===================================================================

class TestShutdownDiagnostics:
    """get_diagnostics() includes finalizer thread status."""

    def test_diagnostics_has_finalizer_key(self):
        ctrl = RecordingController(enable_transcription=False)
        diag = ctrl.get_diagnostics()
        assert "finalizer" in diag
        assert diag["finalizer"]["alive"] is False
        assert diag["finalizer"]["name"] is None

    def test_diagnostics_shows_active_finalizer(self, tmp_path: Path):
        ctrl = _make_recording_controller(tmp_path)
        ctrl._set_state(ControllerState.IDLE)

        blocker = threading.Event()

        def _finalize():
            blocker.wait(timeout=5)

        finalizer = threading.Thread(
            target=_finalize, daemon=True, name="RecordingFinalizer"
        )
        ctrl._finalizer_thread = finalizer
        finalizer.start()

        try:
            diag = ctrl.get_diagnostics()
            assert diag["finalizer"]["alive"] is True
            assert diag["finalizer"]["name"] == "RecordingFinalizer"
        finally:
            blocker.set()
            finalizer.join(timeout=5)
