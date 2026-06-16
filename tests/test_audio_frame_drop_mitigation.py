"""Focused tests for frame-drop mitigation telemetry.

Covers the larger capture block default, monotonic source counters, burst reset
behavior, and sanitized aggregate session stats without real audio hardware or
raw-audio diagnostics.
"""

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.pyaudiowpatch_source import PyAudioWPatchSource
from meetandread.audio.capture.sounddevice_source import SoundDeviceSource, SystemSource
from meetandread.audio.session import (
    DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
)


def _sounddevice_buffer(frames: int = 1024, channels: int = 2) -> np.ndarray:
    return np.zeros((frames, channels), dtype=np.float32)


def _pyaudio_buffer(frames: int = 1024, channels: int = 2) -> bytes:
    return np.zeros((frames, channels), dtype=np.float32).tobytes()


def _make_sounddevice_source(queue_size: int = 1) -> SoundDeviceSource:
    src = SoundDeviceSource.__new__(SoundDeviceSource)
    src.device_id = None
    src.channels = 2
    src.samplerate = 48000
    src.blocksize = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = "mic"
    return src


def _make_pyaudio_source(queue_size: int = 1) -> PyAudioWPatchSource:
    src = PyAudioWPatchSource.__new__(PyAudioWPatchSource)
    src.device_index = None
    src.channels = 2
    src.samplerate = 48000
    src.blocksize = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._running = True
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = "system"
    return src


def test_default_capture_block_size_is_named_and_larger_than_1024():
    assert DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE > 1024


def test_session_uses_default_capture_block_size_for_real_sources():
    session = AudioSession()
    created = []

    class FakeMic:
        def __init__(self, **kwargs):
            created.append(("mic", kwargs))
            self.blocksize = kwargs["blocksize"]

        def get_metadata(self):
            return {"sample_rate": 48000, "channels": 2, "block_size": self.blocksize}

        def get_drop_telemetry(self):
            return {
                "block_size": self.blocksize,
                "frames_dropped": 0,
                "frames_enqueued": 0,
                "total_callbacks": 0,
                "drop_rate": 0.0,
                "consecutive_frames_dropped": 0,
                "max_consecutive_frames_dropped": 0,
            }

        def start(self):
            pass

        def stop(self):
            pass

        def read_frames(self, timeout=None):
            return None

        def is_running(self):
            return False

    class FakeSystem(FakeMic):
        available = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created[-1] = ("system", kwargs)

    with patch("meetandread.audio.session.MicSource", FakeMic), patch(
        "meetandread.audio.session.SystemSource", FakeSystem
    ):
        wrappers = session._create_sources(
            SessionConfig(sources=[SourceConfig(type="mic"), SourceConfig(type="system")])
        )

    assert len(wrappers) == 2
    assert created == [
        (
            "mic",
            {
                "device_id": None,
                "blocksize": DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
                "queue_size": 10,
                "on_frame_dropped": session._on_source_frame_dropped,
            },
        ),
        (
            "system",
            {
                "device_id": None,
                "blocksize": DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
                "queue_size": 10,
                "on_frame_dropped": session._on_source_frame_dropped,
            },
        ),
    ]


def test_sounddevice_drop_bursts_reset_after_successful_enqueue_and_read():
    src = _make_sounddevice_source(queue_size=1)
    src._queue.put(_sounddevice_buffer())

    src._callback(_sounddevice_buffer(), 1024, {}, 0)
    src._callback(_sounddevice_buffer(), 1024, {}, 0)
    assert src.get_frames_dropped() == 2
    assert src.get_drop_telemetry()["consecutive_frames_dropped"] == 2
    assert src.get_drop_telemetry()["max_consecutive_frames_dropped"] == 2

    assert src.read_frames(timeout=0.01) is not None
    assert src.get_frames_dropped() == 2
    assert src.get_drop_telemetry()["consecutive_frames_dropped"] == 0

    src._callback(_sounddevice_buffer(), 1024, {}, 0)
    telemetry = src.get_drop_telemetry()
    assert telemetry["frames_enqueued"] == 1
    assert telemetry["total_callbacks"] == 3
    assert telemetry["drop_rate"] == pytest.approx(2 / 3)
    assert telemetry["max_consecutive_frames_dropped"] == 2


def test_pyaudio_drop_bursts_reset_after_successful_enqueue_and_read():
    src = _make_pyaudio_source(queue_size=1)
    src._queue.put(np.zeros((1024, 2), dtype=np.float32))

    with patch("meetandread.audio.capture.pyaudiowpatch_source.pyaudiowpatch") as paw:
        paw.paContinue = 0
        src._callback(_pyaudio_buffer(), 1024, None, 0)
        src._callback(_pyaudio_buffer(), 1024, None, 0)

    assert src.get_frames_dropped() == 2
    assert src.get_drop_telemetry()["consecutive_frames_dropped"] == 2
    assert src.get_drop_telemetry()["max_consecutive_frames_dropped"] == 2

    assert src.read_frames(timeout=0.01) is not None
    assert src.get_frames_dropped() == 2
    assert src.get_drop_telemetry()["consecutive_frames_dropped"] == 0

    with patch("meetandread.audio.capture.pyaudiowpatch_source.pyaudiowpatch") as paw:
        paw.paContinue = 0
        src._callback(_pyaudio_buffer(), 1024, None, 0)

    telemetry = src.get_drop_telemetry()
    assert telemetry["frames_enqueued"] == 1
    assert telemetry["total_callbacks"] == 3
    assert telemetry["drop_rate"] == pytest.approx(2 / 3)
    assert telemetry["max_consecutive_frames_dropped"] == 2


def test_system_source_forwards_drop_telemetry_to_backend():
    src = SystemSource.__new__(SystemSource)
    backend = MagicMock()
    backend.get_drop_telemetry.return_value = {
        "block_size": DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
        "frames_dropped": 3,
        "frames_enqueued": 7,
        "total_callbacks": 10,
        "drop_rate": 0.3,
        "consecutive_frames_dropped": 1,
        "max_consecutive_frames_dropped": 3,
    }
    src._backend = backend

    assert src.get_drop_telemetry()["drop_rate"] == pytest.approx(0.3)
    assert src.get_drop_telemetry()["max_consecutive_frames_dropped"] == 3


def test_audio_session_aggregates_sanitized_drop_stats():
    session = AudioSession()
    mic = _make_sounddevice_source(queue_size=1)
    system = _make_sounddevice_source(queue_size=1)
    system._source_label = "system"

    mic._queue.put(_sounddevice_buffer())
    mic._callback(_sounddevice_buffer(), 1024, {}, 0)
    mic._callback(_sounddevice_buffer(), 1024, {}, 0)

    system._callback(_sounddevice_buffer(), 1024, {}, 0)

    session._sources = [
        AudioSourceWrapper(mic, SourceConfig(type="mic")),
        AudioSourceWrapper(system, SourceConfig(type="system")),
    ]

    stats = session.get_stats()

    assert stats.frames_dropped == 2
    assert stats.drop_rate == pytest.approx(2 / 3)
    assert stats.consecutive_frames_dropped == 2
    assert stats.max_consecutive_frames_dropped == 2
    assert stats.capture_block_size == DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    assert stats.source_stats["mic"]["block_size"] == DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    assert stats.source_stats["mic"]["drop_rate"] == pytest.approx(1.0)
    assert stats.source_stats["system"]["drop_rate"] == pytest.approx(0.0)
    assert "raw_audio" not in stats.source_stats["mic"]
    assert "transcript" not in stats.source_stats["mic"]
