"""Issue #22 — mic capture diagnostics round.

Covers the three verified gaps from the issue:
- SoundDeviceSource.start() logs a sanitized stream-open INFO line (mirroring
  the loopback source), with a safe fallback when query_devices fails.
- AudioSourceWrapper logs resample-active / passthrough state at INFO.
- read_and_process never returns a 0-sample array (returns None instead), so
  _consumer_loop_inner never feeds empty chunks to on_audio_frame.

All tests are hardware-free: the stream layer is mocked/patched and sources
are built via __new__ construction (see test_audio_frame_drop_mitigation.py).
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.sounddevice_source import SoundDeviceSource
from meetandread.audio.session import (
    DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
)


def _make_sounddevice_source(queue_size: int = 10) -> SoundDeviceSource:
    src = SoundDeviceSource.__new__(SoundDeviceSource)
    src.device_id = 1
    src.channels = 1
    src.samplerate = 48000
    src.blocksize = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._stream = None
    src._running = False
    src._lock = __import__("threading").Lock()
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = "mic"
    return src


class _FakeSource:
    """Minimal source duck-type for AudioSourceWrapper (hardware-free)."""

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._queue: queue.Queue = queue.Queue(maxsize=10)

    def get_metadata(self):
        return {"sample_rate": self._sample_rate, "channels": self._channels}

    def start(self):
        pass

    def stop(self):
        pass

    def is_running(self):
        return False

    def read_frames(self, timeout=None):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_drop_telemetry(self):
        return {
            "block_size": DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
            "frames_dropped": 0,
            "frames_enqueued": 0,
            "total_callbacks": 0,
            "drop_rate": 0.0,
            "consecutive_frames_dropped": 0,
            "max_consecutive_frames_dropped": 0,
        }


# ---------------------------------------------------------------------------
# 1. Mic stream-open log
# ---------------------------------------------------------------------------


class TestMicStreamOpenLog:
    def test_start_logs_stream_open_line(self, caplog):
        src = _make_sounddevice_source()
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            return_value={"name": "Test Mic (WASAPI)"},
        ):
            with caplog.at_level("INFO", logger="meetandread.audio.capture.sounddevice_source"):
                src.start()

        assert src.is_running()
        open_logs = [r.getMessage() for r in caplog.records if "SoundDevice stream opened" in r.message]
        assert open_logs, f"no stream-open log in records: {[r.getMessage() for r in caplog.records]}"
        msg = open_logs[0]
        assert "Test Mic (WASAPI)" in msg
        assert "48000" in msg
        assert str(DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE) in msg

    def test_query_devices_failure_still_starts_and_logs(self, caplog):
        src = _make_sounddevice_source()
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            side_effect=RuntimeError("no devices"),
        ):
            with caplog.at_level("INFO", logger="meetandread.audio.capture.sounddevice_source"):
                src.start()

        assert src.is_running()
        assert any("SoundDevice stream opened" in r.getMessage() for r in caplog.records)

    def test_default_device_id_logs_default_name(self, caplog):
        src = _make_sounddevice_source()
        src.device_id = None
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            return_value={"name": "Whatever"},
        ):
            with caplog.at_level("INFO", logger="meetandread.audio.capture.sounddevice_source"):
                src.start()

        assert src.is_running()
        msg = next(r.getMessage() for r in caplog.records if "SoundDevice stream opened" in r.message)
        assert "default" in msg


# ---------------------------------------------------------------------------
# 2. Resample-state log
# ---------------------------------------------------------------------------


class TestResampleStateLog:
    def test_resampling_active_logged(self, caplog):
        source = _FakeSource(sample_rate=48000, channels=1)
        with caplog.at_level("INFO", logger="meetandread.audio.session"):
            AudioSourceWrapper(
                source,
                SourceConfig(type="mic"),
                target_rate=16000,
                target_channels=1,
            )
        assert any(
            "resampling active" in r.message
            and "48000" in r.message
            and "16000" in r.message
            for r in caplog.records
        )

    def test_passthrough_logged(self, caplog):
        source = _FakeSource(sample_rate=16000, channels=1)
        with caplog.at_level("INFO", logger="meetandread.audio.session"):
            AudioSourceWrapper(
                source,
                SourceConfig(type="mic"),
                target_rate=16000,
                target_channels=1,
            )
        assert any(
            "passthrough" in r.message and "16000" in r.getMessage() for r in caplog.records
        )


# ---------------------------------------------------------------------------
# 3. Zero-sample guard in read_and_process
# ---------------------------------------------------------------------------


class TestZeroSampleGuard:
    def test_empty_chunk_no_resampler_returns_none(self, caplog):
        source = _FakeSource(sample_rate=16000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="mic"), target_rate=16000, target_channels=1
        )
        source._queue.put(np.zeros((0, 1), dtype=np.float32))
        with caplog.at_level("DEBUG", logger="meetandread.audio.session"):
            result = wrapper.read_and_process(timeout=0.1)
        assert result is None
        assert any("zero-sample chunk after processing" in r.getMessage() for r in caplog.records)

    def test_empty_chunk_with_resampler_returns_none(self):
        source = _FakeSource(sample_rate=48000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="mic"), target_rate=16000, target_channels=1
        )
        source._queue.put(np.zeros((0, 1), dtype=np.float32))
        assert wrapper.read_and_process(timeout=0.1) is None

    def test_resample_chunk_zero_output_returns_none(self):
        source = _FakeSource(sample_rate=48000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="mic"), target_rate=16000, target_channels=1
        )
        wrapper._resampler = MagicMock()
        wrapper._resampler.resample_chunk.return_value = np.zeros((0, 1), dtype=np.float32)
        source._queue.put(np.zeros((8, 1), dtype=np.float32))
        assert wrapper.read_and_process(timeout=0.1) is None

    def test_nonempty_chunk_still_passes(self):
        source = _FakeSource(sample_rate=16000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="mic"), target_rate=16000, target_channels=1
        )
        source._queue.put(np.ones((16, 1), dtype=np.float32))
        result = wrapper.read_and_process(timeout=0.1)
        assert result is not None
        assert result.shape[0] > 0


# ---------------------------------------------------------------------------
# 4. Session-level: consumer never feeds 0-sample chunks to on_audio_frame
# ---------------------------------------------------------------------------


class TestConsumerNeverFeedsEmptyChunks:
    def test_empty_chunk_dropped_before_callback(self):
        session = AudioSession()
        source = _FakeSource(sample_rate=16000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="fake"), target_rate=16000, target_channels=1
        )
        session._sources = [wrapper]

        collected = []
        config = SessionConfig(
            sources=[SourceConfig(type="fake")],
            sample_rate=16000,
            channels=1,
            on_audio_frame=collected.append,
        )
        session._config = config
        session._writer = MagicMock()
        source._queue.put(np.zeros((0, 1), dtype=np.float32))
        source._queue.put(np.ones((32, 1), dtype=np.float32))

        # Drive the loop manually: replicate the main loop until the
        # source queue is drained (one pass per chunk).
        while not source._queue.empty():
            frames_list = []
            for w in session._sources:
                frames = w.read_and_process(timeout=0.05)
                if frames is not None:
                    frames = session._apply_denoising(frames, w)
                    frames_list.append(frames)

            if frames_list:
                mixed = session._mix_frames(frames_list)
                if session._config and session._config.on_audio_frame:
                    audio_for_transcription = (
                        mixed.flatten() if mixed.ndim > 1 else mixed
                    )
                    session._config.on_audio_frame(audio_for_transcription)

        assert collected, "expected at least one non-empty chunk to reach on_audio_frame"
        assert all(arr.shape[0] > 0 for arr in collected)

    def test_consumer_loop_inner_drops_empty_chunks(self):
        session = AudioSession()
        source = _FakeSource(sample_rate=16000, channels=1)
        wrapper = AudioSourceWrapper(
            source, SourceConfig(type="fake"), target_rate=16000, target_channels=1
        )
        session._sources = [wrapper]

        collected = []
        session._config = SessionConfig(
            sources=[SourceConfig(type="fake")],
            sample_rate=16000,
            channels=1,
            on_audio_frame=collected.append,
        )
        session._writer = MagicMock()

        # Queue: one empty chunk, one non-empty chunk, then nothing (loop sleeps).
        source._queue.put(np.zeros((0, 1), dtype=np.float32))
        source._queue.put(np.ones((32, 1), dtype=np.float32))

        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        # Wait until the non-empty chunk has been consumed (or timeout).
        deadline = time.monotonic() + 5.0
        while not collected and time.monotonic() < deadline:
            time.sleep(0.02)
        session._stop_event.set()
        t.join(timeout=5.0)
        assert not t.is_alive()

        assert collected, "expected at least one chunk to reach on_audio_frame"
        assert all(arr.shape[0] > 0 for arr in collected)
