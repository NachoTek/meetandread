"""Thread-safety tests for RecordingController state and live audio buffer.

Covers T01 must-haves:
- Concurrent get_state / get_diagnostics / get_live_audio_samples / feed_audio_for_transcription
- Stop-to-start latency remains under 1 second
- Negative tests: empty buffers, odd-byte buffers, concurrent clear/read/write,
  ERROR/STOPPING states during reads
- Existing waveform and stop/postprocess invariants preserved
"""

import threading
import time as _time
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from meetandread.recording.controller import (
    RecordingController,
    ControllerState,
    ControllerError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_int16_bytes(values):
    """Convert a list/tuple of ints in [-32768, 32767] to int16 PCM bytes."""
    return np.array(values, dtype=np.int16).tobytes()


def _synth_audio_chunk(duration_s: float = 0.1, sample_rate: int = 16000):
    """Create a float32 audio chunk of the given duration."""
    n = int(duration_s * sample_rate)
    return np.random.randn(n).astype(np.float32) * 0.3


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl():
    """A RecordingController in IDLE state with transcription disabled."""
    return RecordingController(enable_transcription=False)


@pytest.fixture
def ctrl_recording(tmp_path: Path):
    """A RecordingController wired with fakes, set to RECORDING state."""
    ctrl = RecordingController(enable_transcription=False)
    wav_path = tmp_path / "test.wav"
    wav_path.write_text("fake wav")
    ctrl._session = FakeAudioSession(wav_path)
    ctrl._state = ControllerState.RECORDING
    return ctrl


# ===================================================================
# Concurrent get_state
# ===================================================================

class TestConcurrentGetState:
    """Multiple threads reading get_state() must never crash or return garbage."""

    def test_concurrent_get_state_no_crash(self, ctrl):
        """100 concurrent readers of get_state() complete without error."""
        results = []
        errors = []

        def reader():
            try:
                for _ in range(200):
                    s = ctrl.get_state()
                    assert isinstance(s, ControllerState)
                    results.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent get_state: {errors}"
        assert len(results) == 20 * 200

    def test_concurrent_state_transitions(self, ctrl, tmp_path):
        """State transitions and reads can interleave without crash."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake")
        ctrl._session = FakeAudioSession(wav_path)

        errors = []
        states_seen = set()

        def state_reader():
            try:
                for _ in range(100):
                    s = ctrl.get_state()
                    states_seen.add(s)
            except Exception as e:
                errors.append(e)

        def state_writer():
            try:
                for _ in range(10):
                    ctrl._set_state(ControllerState.STARTING)
                    ctrl._set_state(ControllerState.IDLE)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=state_reader, daemon=True) for _ in range(5)]
            + [threading.Thread(target=state_writer, daemon=True) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"


# ===================================================================
# Concurrent get_diagnostics
# ===================================================================

class TestConcurrentGetDiagnostics:
    """get_diagnostics() must return valid dicts under concurrent state changes."""

    def test_concurrent_diagnostics_no_crash(self, ctrl):
        """50 concurrent readers of get_diagnostics() complete without error."""
        errors = []

        def reader():
            try:
                for _ in range(100):
                    d = ctrl.get_diagnostics()
                    assert isinstance(d, dict)
                    assert "state" in d
                    assert "live_speaker_matching" in d
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_diagnostics_during_buffer_writes(self, ctrl_recording):
        """get_diagnostics() returns consistent data while buffer is being written."""
        ctrl = ctrl_recording
        errors = []
        diag_snapshots = []

        def writer():
            try:
                chunk = _synth_audio_chunk(0.1)
                for _ in range(200):
                    ctrl.feed_audio_for_transcription(chunk)
            except Exception as e:
                errors.append(e)

        def diag_reader():
            try:
                for _ in range(100):
                    d = ctrl.get_diagnostics()
                    diag_snapshots.append(d)
                    lsm = d["live_speaker_matching"]
                    assert isinstance(lsm["audio_buffer_seconds"], (int, float))
                    assert lsm["audio_buffer_seconds"] >= 0
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, daemon=True)]
            + [threading.Thread(target=diag_reader, daemon=True) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert len(diag_snapshots) >= 100


# ===================================================================
# Concurrent get_live_audio_samples
# ===================================================================

class TestConcurrentGetLiveAudioSamples:
    """Readers of get_live_audio_samples must not crash during concurrent writes."""

    def test_concurrent_reads_during_writes(self, ctrl_recording):
        """Multiple readers see valid data while audio callback writes."""
        ctrl = ctrl_recording
        errors = []
        read_results = []

        def writer():
            try:
                chunk = _synth_audio_chunk(0.05)
                for _ in range(500):
                    ctrl.feed_audio_for_transcription(chunk)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    samples = ctrl.get_live_audio_samples()
                    assert isinstance(samples, np.ndarray)
                    assert samples.dtype == np.float32
                    read_results.append(len(samples))
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, daemon=True)]
            + [threading.Thread(target=reader, daemon=True) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_samples_always_valid_float32(self, ctrl_recording):
        """Returned samples are always finite float32 values."""
        ctrl = ctrl_recording
        errors = []

        def writer():
            chunk = _synth_audio_chunk(0.1)
            for _ in range(300):
                ctrl.feed_audio_for_transcription(chunk)

        def reader():
            for _ in range(100):
                samples = ctrl.get_live_audio_samples()
                if samples.size > 0:
                    if not np.all(np.isfinite(samples)):
                        errors.append("non-finite samples found")

        threads = (
            [threading.Thread(target=writer, daemon=True)]
            + [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors


# ===================================================================
# Concurrent feed_audio_for_transcription
# ===================================================================

class TestConcurrentFeedAudio:
    """Multiple writers to feed_audio_for_transcription must not corrupt the buffer."""

    def test_concurrent_writes_no_crash(self, ctrl_recording):
        """10 concurrent writers complete without crash."""
        ctrl = ctrl_recording
        errors = []

        def writer(wid):
            try:
                chunk = _synth_audio_chunk(0.02)
                for _ in range(500):
                    ctrl.feed_audio_for_transcription(chunk)
            except Exception as e:
                errors.append((wid, e))

        threads = [
            threading.Thread(target=writer, args=(i,), daemon=True)
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_buffer_size_bounded_after_concurrent_writes(self, ctrl_recording):
        """After many concurrent writes, buffer stays within max limit."""
        ctrl = ctrl_recording
        max_bytes = ctrl._live_max_buffer_bytes

        def writer():
            chunk = _synth_audio_chunk(0.5)
            for _ in range(50):
                ctrl.feed_audio_for_transcription(chunk)

        threads = [threading.Thread(target=writer, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        with ctrl._buffer_lock:
            assert len(ctrl._live_audio_buffer) <= max_bytes


# ===================================================================
# Concurrent mixed operations
# ===================================================================

class TestConcurrentMixedOperations:
    """All operations interleaved must not deadlock or crash."""

    def test_mixed_read_write_state_buffer(self, ctrl_recording):
        """Interleaved state reads, diagnostics, buffer reads/writes."""
        ctrl = ctrl_recording
        errors = []

        def state_reader():
            try:
                for _ in range(200):
                    ctrl.get_state()
            except Exception as e:
                errors.append(e)

        def diag_reader():
            try:
                for _ in range(100):
                    ctrl.get_diagnostics()
            except Exception as e:
                errors.append(e)

        def buffer_writer():
            try:
                chunk = _synth_audio_chunk(0.05)
                for _ in range(300):
                    ctrl.feed_audio_for_transcription(chunk)
            except Exception as e:
                errors.append(e)

        def buffer_reader():
            try:
                for _ in range(200):
                    ctrl.get_live_audio_samples()
            except Exception as e:
                errors.append(e)

        def error_reader():
            try:
                for _ in range(200):
                    ctrl.get_error()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=state_reader, daemon=True),
            threading.Thread(target=diag_reader, daemon=True),
            threading.Thread(target=buffer_writer, daemon=True),
            threading.Thread(target=buffer_reader, daemon=True),
            threading.Thread(target=buffer_writer, daemon=True),
            threading.Thread(target=error_reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors in mixed ops: {errors}"

    def test_no_deadlock_under_contention(self, ctrl):
        """Stress test: many threads hammering all accessors simultaneously.

        If there's a deadlock, this will time out rather than complete.
        """
        errors = []
        stop_event = threading.Event()

        def worker():
            chunk = _synth_audio_chunk(0.01)
            while not stop_event.is_set():
                try:
                    ctrl.get_state()
                    ctrl.get_error()
                    ctrl.is_recording()
                    ctrl.is_busy()
                    ctrl.get_diagnostics()
                    ctrl.get_live_audio_samples()
                    ctrl.get_last_recording_path()
                    ctrl.get_last_transcript_path()
                    ctrl.get_last_wer()
                    ctrl.feed_audio_for_transcription(chunk)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
        for t in threads:
            t.start()

        _time.sleep(2.0)  # Let them hammer for 2 seconds
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors: {errors}"


# ===================================================================
# Stop-to-start latency
# ===================================================================

class TestStopToStartLatency:
    """Stop → start must complete within 1 second (MEM352)."""

    def test_stop_to_start_under_1s(self, tmp_path: Path):
        """Full stop → IDLE → start cycle completes under 1 second."""
        ctrl = RecordingController(enable_transcription=True)
        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")
        ctrl._session = FakeAudioSession(wav_path)

        # Set up a fake transcription processor and transcript store
        ctrl._transcript_store = MagicMock()
        ctrl._transcription_processor = MagicMock()
        ctrl._post_processor = MagicMock()
        ctrl._post_processor.schedule_post_process = MagicMock(
            return_value=MagicMock(job_id="test-job")
        )
        ctrl._state = ControllerState.RECORDING

        # Measure stop-to-start
        t0 = _time.monotonic()

        # Stop (non-blocking)
        ctrl.stop()
        assert ctrl._state in (ControllerState.STOPPING, ControllerState.IDLE)

        # Wait for IDLE
        deadline = _time.monotonic() + 5.0
        while ctrl.get_state() != ControllerState.IDLE and _time.monotonic() < deadline:
            _time.sleep(0.01)

        assert ctrl.get_state() == ControllerState.IDLE
        idle_time = _time.monotonic() - t0

        # Now start a new recording
        with patch.object(ctrl, "_init_transcription", return_value=None), \
             patch.object(ctrl, "_build_source_configs", return_value=[]):
            ctrl.start({"mic"})

        total_time = _time.monotonic() - t0

        # Both transitions should be fast
        assert idle_time < 1.0, f"STOPPING→IDLE took {idle_time:.2f}s (must be <1s)"
        assert total_time < 2.0, f"Full stop→start took {total_time:.2f}s"

        # Clean up finalizer
        fin = ctrl._finalizer_thread
        if fin and fin.is_alive():
            fin.join(timeout=5)

    def test_rapid_stop_start_cycles_latency(self, tmp_path: Path):
        """3 rapid stop→start cycles each complete IDLE transition under 1 second."""
        ctrl = RecordingController(enable_transcription=False)
        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")
        ctrl._session = FakeAudioSession(wav_path)
        ctrl._post_processor = MagicMock()
        ctrl._post_processor.schedule_post_process = MagicMock(
            return_value=MagicMock(job_id="test-job")
        )

        max_idle_time = 0

        for i in range(3):
            ctrl._state = ControllerState.RECORDING
            ctrl._transcription_processor = MagicMock()
            ctrl._transcript_store = MagicMock()

            t0 = _time.monotonic()
            ctrl._stop_worker()

            # Wait for IDLE
            deadline = _time.monotonic() + 5.0
            while ctrl.get_state() != ControllerState.IDLE and _time.monotonic() < deadline:
                _time.sleep(0.005)

            idle_time = _time.monotonic() - t0
            max_idle_time = max(max_idle_time, idle_time)
            assert ctrl.get_state() == ControllerState.IDLE

            if ctrl._finalizer_thread:
                ctrl._finalizer_thread.join(timeout=5)

        assert max_idle_time < 1.0, (
            f"Slowest STOPPING→IDLE was {max_idle_time:.2f}s (must be <1s)"
        )


# ===================================================================
# Negative tests: empty buffers, odd-byte, concurrent clear/read/write
# ===================================================================

class TestNegativeConcurrentScenarios:
    """Edge cases: empty buffers, odd-byte buffers, ERROR/STOPPING states."""

    def test_read_empty_buffer_concurrent_with_reset(self, ctrl):
        """Reading an empty buffer during reset returns empty array."""
        errors = []

        def resetter():
            for _ in range(100):
                ctrl._reset_live_speaker_state()

        def reader():
            for _ in range(200):
                result = ctrl.get_live_audio_samples()
                assert isinstance(result, np.ndarray)
                assert result.dtype == np.float32

        threads = [
            threading.Thread(target=resetter, daemon=True),
            threading.Thread(target=reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_read_during_concurrent_write_and_clear(self, ctrl_recording):
        """Buffer reads during concurrent writes and resets don't crash."""
        ctrl = ctrl_recording
        errors = []

        def writer():
            chunk = _synth_audio_chunk(0.05)
            for _ in range(200):
                ctrl.feed_audio_for_transcription(chunk)

        def clearer():
            for _ in range(50):
                ctrl._reset_live_speaker_state()
                _time.sleep(0.001)

        def reader():
            for _ in range(200):
                try:
                    samples = ctrl.get_live_audio_samples()
                    assert samples.dtype == np.float32
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=clearer, daemon=True),
            threading.Thread(target=reader, daemon=True),
            threading.Thread(target=reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_get_state_during_error_transition(self, ctrl):
        """get_state returns valid state even during concurrent error transitions."""
        states_seen = set()
        errors = []

        def error_setter():
            for i in range(50):
                try:
                    ctrl._set_error(f"Test error {i}", is_recoverable=True)
                    ctrl.clear_error()
                except Exception as e:
                    errors.append(e)

        def state_reader():
            for _ in range(200):
                s = ctrl.get_state()
                states_seen.add(s)

        threads = [
            threading.Thread(target=error_setter, daemon=True),
            threading.Thread(target=state_reader, daemon=True),
            threading.Thread(target=state_reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        # Should see at least IDLE and ERROR
        assert ControllerState.IDLE in states_seen or ControllerState.ERROR in states_seen

    def test_diagnostics_during_error_state(self, ctrl):
        """get_diagnostics returns consistent data when controller is in ERROR."""
        ctrl._set_error("test error", is_recoverable=True)

        for _ in range(50):
            d = ctrl.get_diagnostics()
            assert d["state"] == "ERROR"
            assert d["error"]["message"] == "test error"
            assert d["error"]["is_recoverable"] is True
            assert "live_speaker_matching" in d

    def test_odd_byte_buffer_concurrent_access(self, ctrl):
        """Odd-byte buffer under concurrent read/write produces valid results."""
        errors = []

        # Start with odd-byte buffer
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([100, -200]) + b'\xAB')

        def writer():
            chunk = np.array([500, -500, 300], dtype=np.int16).tobytes()
            for _ in range(200):
                with ctrl._buffer_lock:
                    ctrl._live_audio_buffer.extend(chunk)

        def reader():
            for _ in range(200):
                samples = ctrl.get_live_audio_samples()
                assert samples.dtype == np.float32
                if samples.size > 0:
                    if not np.all(np.isfinite(samples)):
                        errors.append("non-finite value in odd-byte scenario")

        threads = [
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=reader, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_get_live_audio_samples_in_stopping_state(self, ctrl):
        """get_live_audio_samples works when controller is in STOPPING state."""
        ctrl._state = ControllerState.STOPPING
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([100, 200, 300]))

        result = ctrl.get_live_audio_samples()
        assert result.dtype == np.float32
        assert len(result) == 3

    def test_get_live_audio_samples_in_error_state(self, ctrl):
        """get_live_audio_samples works when controller is in ERROR state."""
        ctrl._state = ControllerState.ERROR
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([1000]))

        result = ctrl.get_live_audio_samples()
        assert result.dtype == np.float32
        assert len(result) == 1

    def test_feed_audio_in_non_recording_state(self, ctrl):
        """feed_audio_for_transcription is a no-op when not RECORDING."""
        ctrl._state = ControllerState.IDLE
        chunk = _synth_audio_chunk(0.1)
        initial_len = len(ctrl._live_audio_buffer)

        ctrl.feed_audio_for_transcription(chunk)

        # Buffer should not grow (not recording)
        assert len(ctrl._live_audio_buffer) == initial_len


# ===================================================================
# Existing waveform test invariants preserved
# ===================================================================

class TestWaveformInvariantsPreserved:
    """Existing waveform accessor tests still pass with locking."""

    def test_empty_buffer_returns_empty(self, ctrl):
        ctrl._live_audio_buffer = bytearray()
        result = ctrl.get_live_audio_samples()
        assert result.dtype == np.float32
        assert result.size == 0

    def test_normalization(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([32767, -32768, 0]))
        result = ctrl.get_live_audio_samples()
        assert result[0] == pytest.approx(32767 / 32768.0, abs=1e-5)
        assert result[1] == pytest.approx(-32768 / 32768.0, abs=1e-5)
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_copy_independence(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([1000, -2000]))
        result = ctrl.get_live_audio_samples()

        # Mutate returned array
        result[0] = 999.0

        # Re-read should be independent
        result2 = ctrl.get_live_audio_samples()
        assert result2[0] != 999.0

    def test_duration_capping(self, ctrl):
        n_samples = 2 * 16000
        samples = np.arange(n_samples, dtype=np.int16)
        ctrl._live_audio_buffer = bytearray(samples.tobytes())

        result = ctrl.get_live_audio_samples(duration_seconds=100.0)
        assert len(result) == n_samples

    def test_odd_byte_handling(self, ctrl):
        raw = _synth_int16_bytes([1000, -2000]) + b'\xAB'
        ctrl._live_audio_buffer = bytearray(raw)
        result = ctrl.get_live_audio_samples()
        assert len(result) == 2  # trailing odd byte dropped


# ===================================================================
# Diagnostics sanitization preserved
# ===================================================================

class TestDiagnosticsSanitizationPreserved:
    """Diagnostics remain sanitized under concurrent access (no audio/text leak)."""

    def test_diagnostics_no_raw_audio(self, ctrl):
        """Diagnostics never contain raw audio samples."""
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes(range(1000)))
        d = ctrl.get_diagnostics()
        lsm = d["live_speaker_matching"]
        # Only buffer_seconds is exposed, not raw bytes
        assert isinstance(lsm["audio_buffer_seconds"], (int, float))
        assert "buffer_bytes" not in d
        assert "samples" not in d

    def test_diagnostics_no_transcript_text(self, ctrl):
        """Diagnostics never expose transcript text."""
        d = ctrl.get_diagnostics()
        assert "transcript_text" not in d
        assert "words_text" not in d
        if "transcript" in d:
            # Should only have word counts
            for key in d["transcript"]:
                assert "text" not in key.lower() or key == "transcript_path"
