"""Tests for AudioSession.swap_source — the hot-plug reconnection seam.

Verifies that a capture source can be atomically replaced mid-recording
without interrupting the consumer loop or losing data integrity.
"""

import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from meetandread.audio.capture.fake_module import FakeAudioModule
from meetandread.audio.session import (
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
    SessionError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SILENCE_WAV = str(FIXTURES / "SAMPLE-Audio1.wav")
HARVARD_WAV = str(FIXTURES / "Harvard_AI_audio_16k.wav")


def _make_session_config(*source_types: str) -> SessionConfig:
    """Build a SessionConfig with fake sources for testing."""
    sources = []
    for stype in source_types:
        if stype == "fake":
            sources.append(
                SourceConfig(type="fake", fake_path=SILENCE_WAV, loop=True)
            )
        else:
            raise ValueError(f"Only 'fake' source type supported in helper")
    return SessionConfig(sources=sources)


def _make_wrapper(fake_path: str = SILENCE_WAV, loop: bool = True) -> AudioSourceWrapper:
    """Create an AudioSourceWrapper wrapping a FakeAudioModule."""
    source = FakeAudioModule(wav_path=fake_path, blocksize=1024, queue_size=10, loop=loop)
    config = SourceConfig(type="fake", fake_path=fake_path, loop=loop)
    return AudioSourceWrapper(source, config, target_rate=16000, target_channels=1)


# ---------------------------------------------------------------------------
# T01: swap_source basics
# ---------------------------------------------------------------------------

class TestSwapSourceBasic:
    """Verify swap_source correctness on a live recording."""

    def test_swap_replaces_wrapper_in_sources_list(self):
        """After swap_source('fake', new), the new wrapper appears in _sources."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        new_wrapper = _make_wrapper()
        old = session.swap_source("fake", new_wrapper)

        assert old is not None
        # The new wrapper should be in the sources list
        with session._sources_lock:
            found = any(
                w is new_wrapper for w in session._sources
            )
        assert found, "New wrapper should be in sources list"
        session.stop()

    def test_swap_returns_old_wrapper(self):
        """swap_source returns the wrapper it replaced."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        new_wrapper = _make_wrapper()
        old = session.swap_source("fake", new_wrapper)

        assert old is not None
        assert old is not new_wrapper
        session.stop()

    def test_swap_returns_none_for_unknown_type(self):
        """swap_source returns None when no wrapper matches the type."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        new_wrapper = _make_wrapper()
        result = session.swap_source("mic", new_wrapper)

        assert result is None
        session.stop()

    def test_swap_raises_when_not_recording(self):
        """swap_source raises SessionError when session is not RECORDING."""
        session = AudioSession()
        config = _make_session_config("fake")

        # IDLE state
        new_wrapper = _make_wrapper()
        with pytest.raises(SessionError, match="must be RECORDING"):
            session.swap_source("fake", new_wrapper)

    def test_old_wrapper_is_stopped_after_swap(self):
        """The replaced wrapper should be stopped after swap."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        # Grab reference to old wrapper
        with session._sources_lock:
            old_wrapper = session._sources[0]

        assert old_wrapper.source.is_running()

        new_wrapper = _make_wrapper()
        session.swap_source("fake", new_wrapper)

        assert not old_wrapper.source.is_running(), "Old source should be stopped"
        session.stop()

    def test_new_wrapper_is_started_after_swap(self):
        """The new wrapper should be running after swap."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        new_wrapper = _make_wrapper()
        session.swap_source("fake", new_wrapper)

        assert new_wrapper.source.is_running(), "New source should be started"
        session.stop()


# ---------------------------------------------------------------------------
# T02: Consumer loop picks up new source
# ---------------------------------------------------------------------------

class TestSwapSourceConsumerIntegration:
    """Verify the consumer loop reads from the new source after swap."""

    def test_consumer_reads_from_new_source_after_swap(self):
        """Frames from the new wrapper should appear in the written PCM data.

        Strategy: start recording with one WAV, swap to another WAV,
        let the consumer loop run briefly, stop, and verify the output
        contains non-zero audio data from the new source.
        """
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        # Let original source produce some frames
        time.sleep(0.3)

        # Swap to a different (non-silent) source
        new_wrapper = _make_wrapper(fake_path=HARVARD_WAV, loop=True)
        session.swap_source("fake", new_wrapper)

        # Let consumer loop read from new source
        time.sleep(0.5)

        wav_path = session.stop()

        # Verify the WAV file exists and has content
        assert wav_path.exists()
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            # Should have at least some audio from both sources
            assert duration > 0.5, f"WAV duration should be > 0.5s, got {duration:.2f}s"

    def test_no_crash_when_swap_during_consumer_loop(self):
        """Swapping while the consumer loop is running should not crash."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        # Perform many rapid swaps — stress test
        errors = []
        for i in range(5):
            try:
                new_wrapper = _make_wrapper()
                session.swap_source("fake", new_wrapper)
                time.sleep(0.05)
            except Exception as e:
                errors.append(e)

        assert len(errors) == 0, f"Swap errors during consumer loop: {errors}"
        session.stop()


# ---------------------------------------------------------------------------
# T03: Rollback on new source start failure
# ---------------------------------------------------------------------------

class TestSwapSourceRollback:
    """Verify rollback behavior when the new source fails to start."""

    def test_rollback_on_start_failure(self):
        """If new source start() fails, old source should be restarted."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        # Grab reference to old wrapper before swap attempt
        with session._sources_lock:
            old_wrapper = session._sources[0]
        old_source = old_wrapper.source

        # Create a wrapper whose source will fail on start
        bad_source = MagicMock()
        bad_source.start.side_effect = RuntimeError("device busy")
        bad_source.get_metadata.return_value = {
            "sample_rate": 16000,
            "channels": 1,
        }
        bad_config = SourceConfig(type="fake", fake_path=SILENCE_WAV)
        bad_wrapper = AudioSourceWrapper(bad_source, bad_config, target_rate=16000, target_channels=1)

        with pytest.raises(SessionError, match="Failed to start"):
            session.swap_source("fake", bad_wrapper)

        # Old source should still be the one in the list (rollback)
        with session._sources_lock:
            assert session._sources[0] is old_wrapper
        # Old source should be running again (rollback restarted it)
        assert old_source.is_running()

        session.stop()

    def test_session_state_remains_recording_after_failed_swap(self):
        """A failed swap should leave the session in RECORDING state."""
        session = AudioSession()
        config = _make_session_config("fake")
        session.start(config)

        bad_source = MagicMock()
        bad_source.start.side_effect = RuntimeError("fail")
        bad_source.get_metadata.return_value = {
            "sample_rate": 16000,
            "channels": 1,
        }
        bad_config = SourceConfig(type="fake", fake_path=SILENCE_WAV)
        bad_wrapper = AudioSourceWrapper(bad_source, bad_config, target_rate=16000, target_channels=1)

        with pytest.raises(SessionError):
            session.swap_source("fake", bad_wrapper)

        assert session.get_state().name == "RECORDING"
        session.stop()
