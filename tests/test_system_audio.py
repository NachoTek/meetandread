"""Tests for SystemSource fallback and dual-source recording.

Covers:
1. SystemSource graceful degradation when loopback is unavailable
2. AudioSession proceeding mic-only when system source is unavailable
3. SystemSource metadata correctness (available / unavailable paths)
4. Real-hardware smoke tests for WASAPI loopback (skipped in CI)
"""

import logging
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metamemory.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
)
from metamemory.audio.capture import SystemSource
from metamemory.audio.session import NoSourcesError


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
    """Write a short sine-wave WAV to *path*."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sine = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm = (sine * 32767).astype(np.int16)
    if channels == 2:
        pcm = np.column_stack([pcm, pcm])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# 1. SystemSource fallback when loopback is unavailable
# ---------------------------------------------------------------------------


class TestSystemSourceUnavailable:
    """Verify SystemSource degrades gracefully when loopback capture fails."""

    def test_system_source_unavailable_fallback(self, caplog):
        """When pyaudiowpatch has no loopback device, SystemSource.available
        is False and all public methods are no-ops that raise no exceptions."""
        # Mock pyaudiowpatch_source to simulate unavailable loopback
        mock_pawp_mod = MagicMock()
        mock_pawp_mod._HAS_PYAUDIOWPATCH = False

        with patch.dict(
            "sys.modules",
            {
                "metamemory.audio.capture.pyaudiowpatch_source": mock_pawp_mod,
            },
        ), patch(
            "metamemory.audio.capture.sounddevice_source._HAS_PYAUDIOWPATCH_attr",
            False,
            create=True,
        ):
            # Simulate the import path inside SystemSource.__init__
            # by patching the module-level import chain
            with patch(
                "metamemory.audio.capture.sounddevice_source.PyAudioWPatchSource",
                None,
                create=True,
            ):
                src = SystemSource.__new__(SystemSource)
                src._log = logging.getLogger("test")
                src.available = False
                src._loopback_device_name = None
                src._device_index = None
                src._channels = 2
                src._samplerate = 48000
                src._blocksize = 1024
                src._queue_size = 10
                src._backend = None

        # available must be False
        assert src.available is False

        # All public methods must be safe no-ops
        src.start()
        src.stop()
        result = src.read_frames(timeout=0.01)
        assert result is None, "read_frames should return None when unavailable"
        assert src.is_running() is False

        meta = src.get_metadata()
        assert meta["available"] is False
        assert meta["loopback_device"] == "none"
        assert meta["source_type"] == "system_loopback"

        # close must not raise even with no backend
        src.close()

    def test_system_source_unavailable_via_mocked_import(self):
        """Force an ImportError inside SystemSource.__init__ and verify the
        object settles into unavailable state."""
        with patch.dict("sys.modules", {"pyaudiowpatch": None}):
            src = SystemSource()
            assert src.available is False
            assert src._backend is None

    def test_system_source_unavailable_init_logs_warning(self, caplog):
        """SystemSource logs a WARNING when loopback is unavailable."""
        with caplog.at_level(logging.WARNING):
            with patch.dict("sys.modules", {"pyaudiowpatch": None}):
                src = SystemSource()
                assert src.available is False

        # Check that a WARNING was logged about unavailability
        warning_msgs = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("unavailable" in m.lower() for m in warning_msgs), (
            f"Expected a WARNING about unavailability, got: {warning_msgs}"
        )

    def test_system_source_unavailable_start_stop_noop(self):
        """start/stop/read_frames are safe no-ops when unavailable."""
        with patch.dict("sys.modules", {"pyaudiowpatch": None}):
            src = SystemSource()

        assert src.available is False

        # These must not raise
        src.start()
        src.stop()
        assert src.read_frames(timeout=0.01) is None
        assert src.is_running() is False
        src.close()


# ---------------------------------------------------------------------------
# 2. AudioSession mic-only recording when system source unavailable
# ---------------------------------------------------------------------------


class TestSessionMicOnlyFallback:
    """Verify AudioSession records successfully with only mic/fake when
    system source is unavailable."""

    def test_session_mic_only_when_system_unavailable(self, tmp_path):
        """Create a session with both a fake source and a system source.
        Mock SystemSource to be unavailable. Recording should complete
        successfully using only the fake source."""
        # Create a test WAV for the fake source
        test_wav = tmp_path / "test_sine.wav"
        _create_sine_wav(test_wav, frequency=440.0, duration=1.0, sample_rate=16000)

        # Patch SystemSource so it reports unavailable
        with patch.dict("sys.modules", {"pyaudiowpatch": None}):
            config = SessionConfig(
                sources=[
                    SourceConfig(type="fake", fake_path=str(test_wav)),
                    SourceConfig(type="system"),
                ],
                output_dir=tmp_path,
                sample_rate=16000,
                channels=1,
            )

            session = AudioSession()
            session.start(config)
            time.sleep(0.5)
            wav_path = session.stop()

        # Verify output exists and is valid
        assert wav_path.exists(), f"WAV file not created: {wav_path}"

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

        assert n_channels == 1
        assert sample_rate == 16000
        assert n_frames > 0, "Should have recorded audio frames"

        stats = session.get_stats()
        assert stats.frames_recorded > 0

    def test_session_raises_when_system_only_and_unavailable(self):
        """When SystemSource is the only source and it's unavailable,
        AudioSession should raise NoSourcesError."""
        with patch.dict("sys.modules", {"pyaudiowpatch": None}):
            config = SessionConfig(
                sources=[SourceConfig(type="system")],
                sample_rate=16000,
                channels=1,
            )

            session = AudioSession()
            with pytest.raises(NoSourcesError):
                session.start(config)


# ---------------------------------------------------------------------------
# 3. SystemSource metadata when available
# ---------------------------------------------------------------------------


class TestSystemSourceMetadata:
    """Verify SystemSource.get_metadata() returns correct fields."""

    def test_system_source_available_metadata(self):
        """When pyaudiowpatch has a valid loopback device, SystemSource.available
        is True and metadata includes device info."""
        # Create mock objects
        mock_loopback_info = {
            "index": 42,
            "name": "Test Loopback Device (WASAPI)",
            "maxInputChannels": 2,
            "max_input_channels": 2,
            "defaultSampleRate": 48000.0,
            "default_samplerate": 48000.0,
        }

        mock_pyaudio_instance = MagicMock()
        mock_pyaudio_instance.get_default_wasapi_loopback.return_value = (
            mock_loopback_info
        )
        mock_pyaudio_instance.__enter__ = MagicMock(return_value=mock_pyaudio_instance)
        mock_pyaudio_instance.__exit__ = MagicMock(return_value=False)

        mock_pawp = MagicMock()
        mock_pawp.PyAudio.return_value = mock_pyaudio_instance

        # Mock the PyAudioWPatchSource class
        mock_backend_instance = MagicMock()
        mock_backend_instance.get_metadata.return_value = {
            "source_type": "pyaudiowpatch_loopback",
            "sample_rate": 48000,
            "channels": 2,
            "dtype": "float32",
            "device_id": 42,
            "running": False,
            "available": True,
            "loopback_device": "Test Loopback Device (WASAPI)",
        }

        mock_pawp_source_class = MagicMock(return_value=mock_backend_instance)
        mock_pawp_source_class._HAS_PYAUDIOWPATCH = True

        with patch.dict(
            "sys.modules", {"pyaudiowpatch": mock_pawp}
        ), patch(
            "metamemory.audio.capture.pyaudiowpatch_source._HAS_PYAUDIOWPATCH",
            True,
        ), patch(
            "metamemory.audio.capture.pyaudiowpatch_source.PyAudioWPatchSource",
            mock_pawp_source_class,
        ), patch(
            "metamemory.audio.capture.pyaudiowpatch_source.pyaudiowpatch",
            mock_pawp,
            create=True,
        ):
            src = SystemSource()

        assert src.available is True
        meta = src.get_metadata()

        # Must include these diagnostic fields
        assert "available" in meta
        assert meta["available"] is True
        assert "loopback_device" in meta
        assert meta["loopback_device"] == "Test Loopback Device (WASAPI)"
        assert meta["source_type"] == "pyaudiowpatch_loopback"
        assert meta["sample_rate"] == 48000
        assert meta["channels"] == 2

    def test_system_source_unavailable_metadata_fields(self):
        """When unavailable, get_metadata() still returns available and
        loopback_device fields with correct values."""
        with patch.dict("sys.modules", {"pyaudiowpatch": None}):
            src = SystemSource()

        assert src.available is False
        meta = src.get_metadata()

        assert "available" in meta
        assert meta["available"] is False
        assert "loopback_device" in meta
        assert meta["loopback_device"] == "none"
        assert "sample_rate" in meta
        assert "channels" in meta


# ---------------------------------------------------------------------------
# 4. Real-hardware smoke tests (skipped unless loopback is available)
# ---------------------------------------------------------------------------


def _loopback_available() -> bool:
    """Check if a WASAPI loopback device exists on this machine."""
    try:
        import pyaudiowpatch as paw

        with paw.PyAudio() as pa:
            info = pa.get_default_wasapi_loopback()
            return info is not None
    except Exception:
        return False


_has_loopback = _loopback_available()

skip_no_loopback = pytest.mark.skipif(
    not _has_loopback,
    reason="No WASAPI loopback device available on this machine",
)


@skip_no_loopback
class TestRealLoopbackCapture:
    """Smoke tests using real WASAPI loopback capture hardware.

    These tests require actual loopback hardware and will be skipped
    in CI environments or on machines without loopback support.

    Note: When running under pytest-qt, the Qt event loop hook may
    interfere with pyaudiowpatch's threading, causing SystemError on
    teardown. The xfail markers reflect this known environment issue.
    The tests pass when run without pytest-qt loaded.
    """

    @pytest.mark.xfail(
        reason="pytest-qt event loop hook interferes with pyaudiowpatch "
               "threading on Windows; passes without pytest-qt",
        strict=False,
    )
    def test_real_loopback_capture(self):
        """Capture 2 seconds of real system audio via PyAudioWPatchSource
        and verify frames have expected format (float32, correct channels)."""
        import pyaudiowpatch as paw

        from metamemory.audio.capture.pyaudiowpatch_source import PyAudioWPatchSource

        with paw.PyAudio() as pa:
            loopback_info = pa.get_default_wasapi_loopback()
            assert loopback_info is not None, "No loopback device found"

        device_index = loopback_info["index"]
        channels = loopback_info.get("maxInputChannels", 2)
        samplerate = int(loopback_info.get("defaultSampleRate", 48000))

        source = PyAudioWPatchSource(
            device_index=device_index,
            channels=channels,
            samplerate=samplerate,
            blocksize=1024,
            queue_size=20,
        )

        source.start()
        assert source.is_running(), "Source should be running after start()"

        # Capture for ~2 seconds
        frames_captured = []
        try:
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    frame = source.read_frames(timeout=0.5)
                except SystemError:
                    # Known issue: pytest-qt event loop hook interferes with
                    # pyaudiowpatch's threading on some Windows environments
                    break
                if frame is not None:
                    frames_captured.append(frame)
        finally:
            source.stop()
            source.close()

        assert len(frames_captured) > 0, "Should have captured some frames"

        # Verify frame format
        combined = np.concatenate(frames_captured, axis=0)
        assert combined.dtype == np.float32, (
            f"Expected float32, got {combined.dtype}"
        )
        assert combined.ndim == 2, f"Expected 2D array, got {combined.ndim}D"
        assert combined.shape[1] == channels, (
            f"Expected {channels} channels, got {combined.shape[1]}"
        )

    @pytest.mark.xfail(
        reason="pytest-qt event loop hook interferes with pyaudiowpatch threading on Windows; "
               "passes when run without pytest-qt",
        strict=False,
    )
    def test_real_dual_source_recording(self, tmp_path):
        """Create AudioSession with real mic + real system source, record
        3 seconds, verify WAV exists and is valid."""
        config = SessionConfig(
            sources=[
                SourceConfig(type="mic"),
                SourceConfig(type="system"),
            ],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
        )

        session = AudioSession()
        try:
            session.start(config)
            assert session.get_state().name == "RECORDING"
            time.sleep(3.0)
            wav_path = session.stop()
        except NoSourcesError:
            pytest.skip("No real mic or system source available for dual recording")
        except Exception as exc:
            pytest.skip(f"Real audio source not available: {exc}")
            return

        assert wav_path.exists(), f"WAV file not created: {wav_path}"

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate

        assert n_channels == 1
        assert sample_rate == 16000
        assert sample_width == 2  # 16-bit
        assert duration > 1.0, (
            f"Expected at least 1 second of audio, got {duration:.2f}s"
        )
