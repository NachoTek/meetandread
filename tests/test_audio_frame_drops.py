"""Tests for audio frame-drop counting and exposure.

Covers source-level drop counters, sanitized logging, callback safety,
AudioSession aggregate stats, and thread-safety — all without real audio
hardware or files from .gitignore'd paths.
"""

import logging
import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.sounddevice_source import (
    MicSource,
    SoundDeviceSource,
    SystemSource,
)
from meetandread.audio.capture.pyaudiowpatch_source import PyAudioWPatchSource
from meetandread.audio.session import (
    AudioSession,
    SessionConfig,
    SessionStats,
    SourceConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indata(channels: int = 2, frames: int = 1024) -> np.ndarray:
    """Create a synthetic float32 audio buffer matching sounddevice's indata."""
    return np.zeros((frames, channels), dtype="float32")


def _make_paw_buffer(channels: int = 2, frames: int = 1024) -> bytes:
    """Create a synthetic float32 bytes buffer matching pyaudiowpatch's in_data."""
    return np.zeros((frames, channels), dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# SoundDeviceSource drop counting
# ---------------------------------------------------------------------------


class TestSoundDeviceSourceDropCounting:
    """Source-local counter and accessor on SoundDeviceSource."""

    def test_initial_count_is_zero(self):
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._frames_dropped = 0
        assert src.get_frames_dropped() == 0

    def test_increment_on_queue_full(self):
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = None
        src._source_label = "test"
        src._lock = threading.Lock()

        # Fill the queue
        src._queue.put(_make_indata())

        # Simulate callback — should drop and increment
        indata = _make_indata()
        src._callback(indata, 1024, {}, 0)
        assert src.get_frames_dropped() == 1

    def test_repeated_drops_accumulate(self):
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = None
        src._source_label = "test"
        src._lock = threading.Lock()

        # Fill the queue
        src._queue.put(_make_indata())

        for expected in range(1, 6):
            src._callback(_make_indata(), 1024, {}, 0)
            assert src.get_frames_dropped() == expected

    def test_no_increment_when_queue_has_room(self):
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=10)
        src._frames_dropped = 0
        src._on_frame_dropped = None
        src._source_label = "test"
        src._lock = threading.Lock()

        src._callback(_make_indata(), 1024, {}, 0)
        assert src.get_frames_dropped() == 0


class TestSoundDeviceSourceDropCallback:
    """on_frame_dropped callback behavior on SoundDeviceSource."""

    def test_callback_receives_label_and_count(self):
        drops = []
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = lambda label, count: drops.append((label, count))
        src._source_label = "mic"
        src._lock = threading.Lock()

        src._queue.put(_make_indata())
        src._callback(_make_indata(), 1024, {}, 0)
        assert drops == [("mic", 1)]

    def test_callback_failure_does_not_prevent_counter_increment(self):
        """A broken callback must not prevent the local counter from advancing."""
        def bad_callback(label, count):
            raise RuntimeError("boom")

        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = bad_callback
        src._source_label = "test"
        src._lock = threading.Lock()

        src._queue.put(_make_indata())
        # Should not raise despite bad callback
        src._callback(_make_indata(), 1024, {}, 0)
        assert src.get_frames_dropped() == 1

    def test_no_callback_when_none(self):
        """When on_frame_dropped is None, drops are still counted."""
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = None
        src._source_label = "test"
        src._lock = threading.Lock()

        src._queue.put(_make_indata())
        src._callback(_make_indata(), 1024, {}, 0)
        assert src.get_frames_dropped() == 1


class TestSoundDeviceSourceDropLogging:
    """INFO log emitted per drop."""

    def test_log_on_drop(self, caplog):
        src = SoundDeviceSource.__new__(SoundDeviceSource)
        src._queue = queue.Queue(maxsize=1)
        src._frames_dropped = 0
        src._on_frame_dropped = None
        src._source_label = "mic"
        src._lock = threading.Lock()

        src._queue.put(_make_indata())
        with caplog.at_level(logging.INFO, logger="meetandread.audio.capture.sounddevice_source"):
            src._callback(_make_indata(), 1024, {}, 0)

        assert any(
            "frame dropped" in rec.message.lower() and "mic" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# PyAudioWPatchSource drop counting
# ---------------------------------------------------------------------------


class TestPyAudioWPatchSourceDropCounting:
    """Source-local counter on PyAudioWPatchSource (no real hardware)."""

    @pytest.fixture
    def src(self):
        """Create a PyAudioWPatchSource without opening a stream.

        We patch the PyAudio constructor to avoid needing real hardware.
        """
        with patch("meetandread.audio.capture.pyaudiowpatch_source.pyaudiowpatch") as mock_paw:
            mock_paw.PyAudio.return_value = MagicMock()
            s = PyAudioWPatchSource.__new__(PyAudioWPatchSource)
            s._queue = queue.Queue(maxsize=1)
            s._running = True
            s._frames_dropped = 0
            s._on_frame_dropped = None
            s._source_label = "system"
            s.channels = 2
            s.dtype = "float32"
            s._lock = threading.Lock()
            return s

    def test_initial_count_is_zero(self, src):
        assert src.get_frames_dropped() == 0

    def test_increment_on_queue_full(self, src):
        # Fill the queue
        valid_buf = _make_paw_buffer()
        src._queue.put(np.zeros((1024, 2), dtype=np.float32))

        src._callback(valid_buf, 1024, None, 0)
        assert src.get_frames_dropped() == 1

    def test_no_increment_on_normal_enqueue(self, src):
        valid_buf = _make_paw_buffer()
        src._callback(valid_buf, 1024, None, 0)
        assert src.get_frames_dropped() == 0

    def test_callback_failure_does_not_prevent_increment(self, src):
        def bad_cb(label, count):
            raise RuntimeError("boom")

        src._on_frame_dropped = bad_cb
        src._queue.put(np.zeros((1024, 2), dtype=np.float32))
        valid_buf = _make_paw_buffer()

        with patch(
            "meetandread.audio.capture.pyaudiowpatch_source.pyaudiowpatch",
            paContinue=0,
        ):
            src._callback(valid_buf, 1024, None, 0)
        assert src.get_frames_dropped() == 1

    def test_invalid_buffer_not_counted_as_drop(self, src):
        """Malformed buffers (wrong size) should not be counted as queue drops."""
        # Queue is full, but buffer is wrong size — callback returns early
        src._queue.put(np.zeros((1024, 2), dtype=np.float32))
        bad_buf = b"\x00" * 10  # Too small
        src._callback(bad_buf, 1024, None, 0)
        assert src.get_frames_dropped() == 0


class TestPyAudioWPatchSourceDropLogging:
    """INFO log on drop from PyAudioWPatchSource."""

    def test_log_on_drop(self, caplog):
        with patch("meetandread.audio.capture.pyaudiowpatch_source.pyaudiowpatch") as mock_paw:
            mock_paw.paContinue = 0
            s = PyAudioWPatchSource.__new__(PyAudioWPatchSource)
            s._queue = queue.Queue(maxsize=1)
            s._running = True
            s._frames_dropped = 0
            s._on_frame_dropped = None
            s._source_label = "system"
            s.channels = 2
            s.dtype = "float32"
            s._lock = threading.Lock()

            s._queue.put(np.zeros((1024, 2), dtype=np.float32))
            valid_buf = _make_paw_buffer()
            with caplog.at_level(logging.INFO, logger="meetandread.audio.capture.pyaudiowpatch_source"):
                s._callback(valid_buf, 1024, None, 0)

            assert any(
                "frame dropped" in rec.message.lower() and "system" in rec.message
                for rec in caplog.records
            )


# ---------------------------------------------------------------------------
# SystemSource forwarding
# ---------------------------------------------------------------------------


class TestSystemSourceForwarding:
    """SystemSource delegates get_frames_dropped and on_frame_dropped to backend."""

    def test_get_frames_dropped_delegates_to_backend(self):
        """When backend is available, get_frames_dropped returns backend count."""
        src = SystemSource.__new__(SystemSource)
        mock_backend = MagicMock()
        mock_backend.get_frames_dropped.return_value = 7
        src._backend = mock_backend
        src.available = True

        assert src.get_frames_dropped() == 7

    def test_get_frames_dropped_zero_when_no_backend(self):
        src = SystemSource.__new__(SystemSource)
        src._backend = None
        src.available = False

        assert src.get_frames_dropped() == 0


# ---------------------------------------------------------------------------
# AudioSession aggregate stats
# ---------------------------------------------------------------------------


class TestAudioSessionAggregateDrops:
    """AudioSession._on_source_frame_dropped updates SessionStats."""

    def test_single_drop_increments_stats(self):
        session = AudioSession()
        assert session.get_stats().frames_dropped == 0

        session._on_source_frame_dropped("mic", 1)
        assert session.get_stats().frames_dropped == 1

    def test_multiple_drops_accumulate(self):
        session = AudioSession()
        for _ in range(5):
            session._on_source_frame_dropped("mic", 1)
        assert session.get_stats().frames_dropped == 5

    def test_fires_session_callback_with_aggregate(self):
        counts_seen = []
        config = SessionConfig(
            sources=[],
            on_frames_dropped=lambda c: counts_seen.append(c),
        )
        session = AudioSession()
        session._config = config

        session._on_source_frame_dropped("mic", 1)
        session._on_source_frame_dropped("mic", 2)
        assert counts_seen == [1, 2]

    def test_broken_session_callback_does_not_prevent_increment(self):
        def bad_cb(c):
            raise RuntimeError("boom")

        config = SessionConfig(
            sources=[],
            on_frames_dropped=bad_cb,
        )
        session = AudioSession()
        session._config = config

        # Must not raise
        session._on_source_frame_dropped("mic", 1)
        assert session.get_stats().frames_dropped == 1

    def test_thread_safety_under_concurrent_drops(self):
        """Multiple threads calling _on_source_frame_dropped concurrently
        must not lose increments."""
        session = AudioSession()
        n_threads = 4
        n_increments = 100

        barrier = threading.Barrier(n_threads)

        def increment_many():
            barrier.wait()
            for _ in range(n_increments):
                session._on_source_frame_dropped("mic", 1)

        threads = [threading.Thread(target=increment_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert session.get_stats().frames_dropped == n_threads * n_increments


class TestAudioSessionDropLogging:
    """Session-level INFO log on frame drops."""

    def test_log_on_source_drop(self, caplog):
        session = AudioSession()
        with caplog.at_level(logging.INFO, logger="meetandread.audio.session"):
            session._on_source_frame_dropped("mic", 1)

        assert any(
            "session frame drop" in rec.message.lower()
            for rec in caplog.records
        )
