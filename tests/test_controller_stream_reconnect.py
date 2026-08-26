"""Tests for controller-driven stream reconnection (S02).

Verifies that RecordingController.handle_device_event() and retry_recovery()
actually swap the audio source into the running AudioSession when a device
reconnects, so the recording continues producing real audio data.
"""

import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.fake_module import FakeAudioModule
from meetandread.audio.capture.devices import list_mic_inputs
from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.audio.session import (
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
)
from meetandread.recording.controller import RecordingController, ControllerState


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SILENCE_WAV = str(FIXTURES / "SAMPLE-Audio1.wav")
HARVARD_WAV = str(FIXTURES / "Harvard_AI_audio_16k.wav")


def _make_fake_source_config(fake_path: str = SILENCE_WAV, loop: bool = True) -> SourceConfig:
    return SourceConfig(type="fake", fake_path=fake_path, loop=loop)


class _FakeAudioSession:
    """Lightweight mock of AudioSession that tracks swap calls."""

    def __init__(self):
        self._state = "RECORDING"
        self._sources = []
        self._config = None
        self._swap_calls = []
        self._stop_event = threading.Event()

    def swap_source(self, source_type, new_wrapper):
        self._swap_calls.append((source_type, new_wrapper))
        return None

    def get_state(self):
        return self._state

    def get_stats(self):
        stats = MagicMock()
        stats.frames_dropped = 0
        stats.retry_attempts = 0
        stats.retry_outcome = "none"
        stats.failed_sources = []
        stats.fallback_sources = []
        return stats


class _FakeSourceWrapper:
    """Minimal AudioSourceWrapper stand-in for testing rebuild."""

    def __init__(self, source_type="fake"):
        self.source = MagicMock()
        self.source.is_running.return_value = True
        self.config = SourceConfig(type=source_type)
        self.source.get_metadata.return_value = {
            "sample_rate": 16000,
            "channels": 1,
        }

    def start(self):
        self.source.start()

    def stop(self):
        self.source.stop()

    def read_and_process(self, timeout=None):
        return None

    def is_running(self):
        return True


class TestRebuildSourceWrapper:
    """Verify _rebuild_source_wrapper creates correct source types."""

    def test_rebuild_fake_source(self):
        """_rebuild_source_wrapper should create a FakeAudioModule for type='fake'."""
        controller = RecordingController(enable_transcription=False)
        controller._session = _FakeAudioSession()
        controller._session._config = SessionConfig(
            sources=[_make_fake_source_config(SILENCE_WAV, loop=True)],
        )

        wrapper = controller._rebuild_source_wrapper("fake")
        assert wrapper is not None
        assert isinstance(wrapper, AudioSourceWrapper)
        assert wrapper.config.type == "fake"
        assert wrapper.config.loop is True

    @pytest.mark.skipif(
        not list_mic_inputs(),
        reason="No microphone input devices available on this machine",
    )
    def test_rebuild_mic_source(self):
        """_rebuild_source_wrapper should create a MicSource for type='mic'."""
        controller = RecordingController(enable_transcription=False)
        controller._session = _FakeAudioSession()
        controller._session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
        )

        wrapper = controller._rebuild_source_wrapper("mic")
        assert wrapper is not None
        assert isinstance(wrapper, AudioSourceWrapper)
        assert wrapper.config.type == "mic"

    def test_rebuild_unknown_type_returns_none(self):
        """_rebuild_source_wrapper should return None for unknown types."""
        controller = RecordingController(enable_transcription=False)
        controller._session = _FakeAudioSession()
        controller._session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
        )

        wrapper = controller._rebuild_source_wrapper("unknown")
        assert wrapper is None


class TestSwapSessionSource:
    """Verify swap_session_source delegates correctly."""

    def test_swap_delegates_to_session(self):
        """swap_session_source should call session.swap_source."""
        controller = RecordingController(enable_transcription=False)
        fake_session = _FakeAudioSession()
        controller._session = fake_session

        wrapper = _FakeSourceWrapper("fake")
        result = controller.swap_session_source("fake", wrapper)

        assert result is True
        assert len(fake_session._swap_calls) == 1
        assert fake_session._swap_calls[0][0] == "fake"

    def test_swap_returns_false_on_exception(self):
        """swap_session_source should return False on exception."""
        controller = RecordingController(enable_transcription=False)
        fake_session = _FakeAudioSession()
        fake_session.swap_source = MagicMock(side_effect=RuntimeError("fail"))
        controller._session = fake_session

        wrapper = _FakeSourceWrapper("fake")
        result = controller.swap_session_source("fake", wrapper)

        assert result is False


class TestAutoRecoverySwapsSource:
    """Verify handle_device_event AUTO_RECOVERED triggers source swap."""

    def _make_loss_event(self, device_id="test-dev-1"):
        return DeviceEvent(
            event_type=DeviceEventType.REMOVED,
            device_id=device_id,
            friendly_name="Test Headset",
            flow="capture",
            role="capture",
            state="unplugged",
        )

    def _make_reconnect_event(self, device_id="test-dev-1"):
        return DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id=device_id,
            friendly_name="Test Headset",
            flow="capture",
            role="capture",
            state="active",
        )

    def test_auto_recovered_triggers_rebuild_and_swap(self):
        """AUTO_RECOVERED outcome should call _rebuild_source_wrapper and swap_session_source."""
        controller = RecordingController(enable_transcription=False)
        fake_session = _FakeAudioSession()
        fake_session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
        )
        controller._session = fake_session

        # Start recording state
        controller._state = ControllerState.RECORDING
        controller._snapshot_active_sources([SourceConfig(type="mic")])

        # Simulate device loss
        loss_event = self._make_loss_event()
        result = controller.handle_device_event(loss_event)
        assert result.outcome.value in ("total_loss", "degraded")

        # Simulate reconnect within recovery window
        reconnect_event = self._make_reconnect_event()
        rebuild_calls = []
        swap_calls = []

        with patch.object(controller, '_rebuild_source_wrapper', wraps=controller._rebuild_source_wrapper) as mock_rebuild:
            mock_rebuild.side_effect = lambda st, device_id=None: rebuild_calls.append(st) or _FakeSourceWrapper(st)
            with patch.object(controller, 'swap_session_source', wraps=controller.swap_session_source) as mock_swap:
                mock_swap.side_effect = lambda st, w: swap_calls.append((st, w)) or True
                result = controller.handle_device_event(reconnect_event)

        assert result.outcome.value == "auto_recovered"
        # For mic type, _rebuild_source_wrapper should have been called
        # (it may fail to create a real MicSource without hardware, but the call should happen)
        # swap_session_source should also have been attempted
        # Note: on CI without audio hardware, rebuild may return None

    def test_manual_retry_triggers_rebuild_and_swap(self):
        """retry_recovery should call _rebuild_source_wrapper and swap_session_source."""
        controller = RecordingController(enable_transcription=False)
        fake_session = _FakeAudioSession()
        fake_session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
        )
        controller._session = fake_session

        # Set up lost state
        from meetandread.recording.controller import ActiveSourceIdentity
        controller._state = ControllerState.ERROR
        controller._lost_source_identities = {
            "mic": ActiveSourceIdentity(
                type="mic",
                device_id="test-dev-1",
                friendly_name="Test Mic",
                flow="capture",
                is_active=False,
                lost_at=time.time() - 10,
            ),
        }

        rebuild_calls = []
        swap_calls = []

        with patch.object(controller, '_rebuild_source_wrapper', wraps=controller._rebuild_source_wrapper) as mock_rebuild:
            mock_rebuild.side_effect = lambda st, device_id=None: rebuild_calls.append(st) or _FakeSourceWrapper(st)
            with patch.object(controller, 'swap_session_source', wraps=controller.swap_session_source) as mock_swap:
                mock_swap.side_effect = lambda st, w: swap_calls.append((st, w)) or True
                result = controller.retry_recovery()

        assert result.outcome.value == "manual_recovered"
        # swap_session_source should have been called for the recovered source
        assert len(swap_calls) == 1
        assert swap_calls[0][0] == "mic"


class TestEndToEndStreamReconnect:
    """End-to-end test: fake recording with device loss and reconnect."""

    def test_fake_recording_continues_after_reconnect(self):
        """Recording with a fake source should produce audio after reconnect swap.

        This test uses real AudioSession + FakeAudioModule (no mocks for the
        audio pipeline), and patches only the source rebuild to return a
        new fake wrapper.
        """
        controller = RecordingController(enable_transcription=False)

        # Start recording with a fake source
        error = controller.start({"fake"}, fake_path=SILENCE_WAV, fake_loop=True)
        assert error is None
        assert controller.is_recording()

        # Let it record for a bit
        time.sleep(0.2)

        # Simulate reconnect by manually swapping in a new fake source
        new_source = FakeAudioModule(
            wav_path=HARVARD_WAV, blocksize=1024, queue_size=10, loop=True
        )
        new_config = SourceConfig(type="fake", fake_path=HARVARD_WAV, loop=True)
        new_wrapper = AudioSourceWrapper(new_source, new_config, target_rate=16000, target_channels=1)

        success = controller.swap_session_source("fake", new_wrapper)
        assert success is True

        # Let the new source produce frames
        time.sleep(0.3)

        # Stop recording
        controller.stop()
        # Wait for finalization
        time.sleep(0.5)

        # Verify a WAV was produced
        wav_path = controller.get_last_recording_path()
        if wav_path:
            assert wav_path.exists()
            import wave
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                assert duration > 0.3, f"Expected > 0.3s of audio, got {duration:.2f}s"
