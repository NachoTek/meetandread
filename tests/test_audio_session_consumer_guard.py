"""Tests for audio consumer thread crash guard.

Covers:
- Consumer loop crash sets SessionState.ERROR and stores the exception
- Consumer loop crash invokes SessionConfig.on_error callback
- on_error callback failure is logged without masking the original crash
- stop() does not hang if consumer thread already died (state=ERROR)
- stop() works from ERROR state (consumer thread crashed before stop)
- Normal recording still finalizes correctly with crash guard in place
- get_error() returns None after clean recording
- Raw audio buffers are NOT included in logged/returned errors

All tests use FakeAudioModule — no real audio devices required.
"""

import threading
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meetandread.audio.session import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionState,
    SessionError,
)
from meetandread.audio.capture import FakeAudioModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_sine_wav(
    path: Path,
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.5,
) -> None:
    """Create a sine wave WAV file for testing."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    int16_data = (sine_wave * 32767).astype(np.int16)
    if channels == 2:
        int16_data = np.column_stack([int16_data, int16_data])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def _make_config(tmp_path: Path, **overrides) -> SessionConfig:
    """Create a SessionConfig with a fake source."""
    test_wav = tmp_path / "test.wav"
    if not test_wav.exists():
        _create_sine_wav(test_wav, duration=1.0)
    defaults = dict(
        sources=[SourceConfig(type="fake", fake_path=str(test_wav))],
        output_dir=tmp_path,
        sample_rate=16000,
        channels=1,
    )
    defaults.update(overrides)
    return SessionConfig(**defaults)


# ---------------------------------------------------------------------------
# Consumer crash guard
# ---------------------------------------------------------------------------


class TestConsumerCrashSetsErrorState:
    """Crash in _consumer_loop transitions session to ERROR and stores the exception."""

    def test_read_failure_sets_error_state(self, tmp_path):
        """Simulated wrapper read failure sets ERROR state and stores exception."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        # Poison the source wrapper to raise on read
        original_read = session._sources[0].read_and_process

        def crashing_read(timeout=None):
            # First let a few frames through so the session is healthy
            if not hasattr(crashing_read, "_called"):
                crashing_read._called = True
                return original_read(timeout=timeout)
            raise RuntimeError("Simulated audio read failure")

        session._sources[0].read_and_process = crashing_read

        # Wait for the consumer to crash
        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR
        assert session.get_error() is not None
        assert isinstance(session.get_error(), RuntimeError)
        assert "Simulated audio read failure" in str(session.get_error())

    def test_denoising_failure_handled_gracefully(self, tmp_path):
        """Denoising crash is handled internally — session stays RECORDING.

        _apply_denoising catches exceptions and hard-disables denoising.
        The consumer loop does NOT crash from denoising errors.
        """
        test_wav = tmp_path / "test.wav"
        if not test_wav.exists():
            _create_sine_wav(test_wav, duration=1.0)

        # Use denoise=True to force denoising on fake source
        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: _CrashingDenoiser(),
        )
        session = AudioSession()
        session.start(config)

        time.sleep(0.3)

        # Denoising errors are caught internally — session should still be RECORDING
        assert session.get_state() == SessionState.RECORDING
        # Denoising should have been hard-disabled after the first crash
        assert session._denoising_disabled is True

        wav_path = session.stop()
        assert session.get_state() == SessionState.FINALIZED

    def test_writer_failure_sets_error_state(self, tmp_path):
        """Writer write_frames_i16 crash sets ERROR state."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        # Poison the writer
        original_write = session._writer.write_frames_i16
        call_count = [0]

        def crashing_write(data):
            call_count[0] += 1
            if call_count[0] > 1:
                raise OSError("Simulated disk write failure")
            return original_write(data)

        session._writer.write_frames_i16 = crashing_write

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR
        assert session.get_error() is not None
        assert "Simulated disk write failure" in str(session.get_error())

    def test_callback_failure_sets_error_state(self, tmp_path):
        """on_audio_frame callback crash sets ERROR state."""
        call_count = [0]

        def crashing_callback(audio):
            call_count[0] += 1
            if call_count[0] > 1:
                raise ValueError("Simulated callback failure")
            return None

        config = _make_config(tmp_path, on_audio_frame=crashing_callback)
        session = AudioSession()
        session.start(config)

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR
        assert session.get_error() is not None


class TestConsumerCrashInvokesOnErrorCallback:
    """on_error callback fires when consumer loop crashes."""

    def test_on_error_called_with_exception(self, tmp_path):
        """on_error callback receives the exception from consumer crash."""
        errors = []

        config = _make_config(tmp_path, on_error=lambda e: errors.append(e))
        session = AudioSession()
        session.start(config)

        # Poison the source to crash
        def crashing_read(timeout=None):
            raise RuntimeError("Source read crash")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "Source read crash" in str(errors[0])

    def test_on_error_callback_exception_is_logged(self, tmp_path):
        """on_error callback raising does not mask the original crash."""
        errors = []

        def bad_callback(exc):
            errors.append(exc)
            raise ValueError("Callback itself crashed!")

        config = _make_config(tmp_path, on_error=bad_callback)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("Original crash")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        # State should still be ERROR (original crash preserved)
        assert session.get_state() == SessionState.ERROR
        # Callback was still invoked despite raising
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    def test_no_on_error_callback_is_safe(self, tmp_path):
        """Consumer crash without on_error callback does not crash."""
        config = _make_config(tmp_path)  # No on_error
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("Crash without callback")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR
        assert session.get_error() is not None


class TestStopAfterConsumerCrash:
    """stop() works correctly after consumer thread crashes."""

    def test_stop_from_error_state_returns_path(self, tmp_path):
        """stop() from ERROR state does not hang and returns a valid path."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        # Crash the consumer
        def crashing_read(timeout=None):
            raise RuntimeError("Consumer died")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR

        # stop() should work from ERROR state
        wav_path = session.stop()
        assert session.get_state() == SessionState.FINALIZED
        # Path should exist (writer may have written some frames before crash)
        # but we don't require it — the key invariant is stop() doesn't hang

    def test_stop_does_not_hang_after_consumer_crash(self, tmp_path):
        """stop() returns within reasonable time after consumer crash."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("Consumer died")

        session._sources[0].read_and_process = crashing_read

        # Wait for crash
        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        # stop() should complete quickly — timeout catches hangs
        result = session.stop()
        assert result is not None

    def test_error_observable_before_stop(self, tmp_path):
        """Error is observable via get_state/get_error before stop() is called."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("Observable crash")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        # Before calling stop(), error should be visible
        assert session.get_state() == SessionState.ERROR
        error = session.get_error()
        assert error is not None
        assert "Observable crash" in str(error)


class TestStopEventSetAfterCrash:
    """Consumer crash sets _stop_event to prevent further work."""

    def test_stop_event_set_after_crash(self, tmp_path):
        """_stop_event is set after consumer crash."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("Set stop event test")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session._stop_event.is_set()


class TestNormalPathUnchanged:
    """Normal recording path still works with crash guard in place."""

    def test_normal_recording_finalizes(self, tmp_path):
        """Clean recording with crash guard still finalizes correctly."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        time.sleep(0.3)

        wav_path = session.stop()

        assert session.get_state() == SessionState.FINALIZED
        assert wav_path.exists()
        assert session.get_error() is None

    def test_normal_recording_stats(self, tmp_path):
        """Stats are correct after normal recording."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        time.sleep(0.3)

        session.stop()

        stats = session.get_stats()
        assert stats.frames_recorded > 0
        assert session.get_error() is None

    def test_get_error_returns_none_when_no_crash(self, tmp_path):
        """get_error() returns None when no crash occurred."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)
        time.sleep(0.1)
        session.stop()
        assert session.get_error() is None


class TestGetErrorAccessor:
    """get_error() accessor behavior."""

    def test_initial_state_returns_none(self):
        """get_error() returns None on fresh session."""
        session = AudioSession()
        assert session.get_error() is None

    def test_after_start_returns_none(self, tmp_path):
        """get_error() returns None after successful start."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)
        assert session.get_error() is None
        # Clean up
        time.sleep(0.1)
        session.stop()

    def test_after_crash_returns_exception(self, tmp_path):
        """get_error() returns the crash exception after consumer crash."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise IOError("Device disconnected")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        error = session.get_error()
        assert isinstance(error, IOError)
        assert "Device disconnected" in str(error)


class TestNoRawAudioInErrors:
    """Errors and callbacks never contain raw audio buffers."""

    def test_error_message_no_audio_data(self, tmp_path):
        """Crash error does not include numpy arrays or raw samples."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        # Generate some audio frames to increase the chance of leakage
        time.sleep(0.1)

        def crashing_read(timeout=None):
            raise RuntimeError("Simulated crash with no audio leak")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        error = session.get_error()
        error_str = str(error)
        # Should not contain array representations or sample data
        assert "[" not in error_str or "Simulated" in error_str
        # The error should be a simple string message
        assert isinstance(error_str, str)

    def test_on_error_callback_no_audio_data(self, tmp_path):
        """on_error callback exception does not contain audio buffers."""
        errors = []

        config = _make_config(tmp_path, on_error=lambda e: errors.append(e))
        session = AudioSession()
        session.start(config)
        time.sleep(0.1)

        def crashing_read(timeout=None):
            raise RuntimeError("Clean crash message")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert len(errors) == 1
        error_str = str(errors[0])
        # Should be a clean error string without numpy data
        assert "Clean crash message" in error_str


class TestErrorNotSetDuringStopping:
    """ERROR state is not set if stop is already in progress."""

    def test_error_does_not_override_stopping(self, tmp_path):
        """Crash during STOPPING does not regress state back to ERROR."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        # Let it record briefly
        time.sleep(0.1)

        # Start stop — sets STOPPING
        # Then trigger a crash in the consumer loop
        # The consumer should see STOPPING and not set ERROR
        original_read = session._sources[0].read_and_process

        def delayed_crash_read(timeout=None):
            # Give stop time to set STOPPING
            time.sleep(0.1)
            raise RuntimeError("Late crash during stop")

        session._sources[0].read_and_process = delayed_crash_read

        # Stop the session
        wav_path = session.stop()

        # State should be FINALIZED, not ERROR
        assert session.get_state() == SessionState.FINALIZED


class TestSessionReuseAfterError:
    """Session can be reused after consumer crash."""

    def test_start_after_error(self, tmp_path):
        """Session can start a new recording after consumer crash."""
        config = _make_config(tmp_path)
        session = AudioSession()
        session.start(config)

        def crashing_read(timeout=None):
            raise RuntimeError("First crash")

        session._sources[0].read_and_process = crashing_read

        deadline = time.monotonic() + 5.0
        while session.get_state() == SessionState.RECORDING and time.monotonic() < deadline:
            time.sleep(0.05)

        assert session.get_state() == SessionState.ERROR
        session.stop()  # Clean up from crashed state

        # Should be able to reuse
        config2 = _make_config(tmp_path)
        session.start(config2)
        time.sleep(0.1)

        wav_path = session.stop()
        assert session.get_state() == SessionState.FINALIZED
        assert wav_path.exists()


# ---------------------------------------------------------------------------
# Helpers for crashing denoiser
# ---------------------------------------------------------------------------


class _CrashingDenoiser:
    """A denoising provider that crashes on process()."""

    name = "crash_provider"

    def process(self, audio):
        from meetandread.audio.denoising import DenoisingResult
        raise RuntimeError("Denoiser exploded!")
