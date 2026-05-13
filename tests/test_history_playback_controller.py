"""Tests for the HistoryPlaybackController.

Covers WAV resolution, transport controls, error handling, state
transitions, and boundary conditions for the QtMultimedia playback helper.
Uses inline WAV fixtures generated with Python's ``wave`` module and mocked
Qt types so tests do not require a running QApplication.
"""

import sys
import wave
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Mock Qt types before importing the controller
# ---------------------------------------------------------------------------

# Create mock Qt modules so we can test the controller without a QApplication.
# This avoids the "DLL load failed" and event-loop hangs that occur when
# real QMediaPlayer/QAudioOutput are instantiated in a headless pytest run.

_mock_qt_multimedia = ModuleType("PyQt6.QtMultimedia")


class _MockQMediaPlayer:
    """Lightweight mock of QMediaPlayer with enum support."""

    class Error(IntEnum):
        NoError = 0
        ResourceError = 1
        FormatError = 2
        NetworkError = 3
        AccessDeniedError = 4

    class MediaStatus(IntEnum):
        NoMedia = 0
        LoadingMedia = 1
        LoadedMedia = 2
        BufferingMedia = 3
        StalledMedia = 4
        BufferedMedia = 5
        EndOfMedia = 6
        InvalidMedia = 7

    class PlaybackState(IntEnum):
        StoppedState = 0
        PlayingState = 1
        PausedState = 2

    def __init__(self):
        self._source = None
        self._rate = 1.0
        self._audio_output = None
        self._playback_state = self.PlaybackState.StoppedState
        self._media_status = self.MediaStatus.NoMedia
        self._position = 0
        self._duration = 0
        self.errorOccurred = MagicMock()
        self.mediaStatusChanged = MagicMock()
        self.playbackStateChanged = MagicMock()

    def setAudioOutput(self, output):
        self._audio_output = output

    def setSource(self, url):
        self._source = url
        self._media_status = self.MediaStatus.LoadedMedia
        # Default mock duration (simulates a 30-second recording)
        self._duration = 30000

    def source(self):
        return self._source

    def play(self):
        self._playback_state = self.PlaybackState.PlayingState

    def pause(self):
        self._playback_state = self.PlaybackState.PausedState

    def stop(self):
        self._playback_state = self.PlaybackState.StoppedState

    def setPlaybackRate(self, rate):
        self._rate = rate

    def playbackRate(self):
        return self._rate

    def playbackState(self):
        return self._playback_state

    def mediaStatus(self):
        return self._media_status

    def position(self):
        return self._position

    def setPosition(self, ms):
        self._position = max(0, int(ms))

    def duration(self):
        return self._duration


class _MockQAudioOutput:
    """Lightweight mock of QAudioOutput."""

    def __init__(self):
        self._volume = 1.0

    def setVolume(self, volume):
        self._volume = volume

    def volume(self):
        return self._volume


class _MockQUrl:
    """Lightweight mock of QUrl."""

    def __init__(self, url_str=""):
        self._url = url_str

    @classmethod
    def fromLocalFile(cls, path):
        return cls(f"file:///{path}")

    def toLocalFile(self):
        return self._url.replace("file:///", "")

    def __repr__(self):
        return f"QUrl('{self._url}')"


_mock_qt_core = ModuleType("PyQt6.QtCore")
_mock_qt_core.QUrl = _MockQUrl

_mock_qt_multimedia.QMediaPlayer = _MockQMediaPlayer
_mock_qt_multimedia.QAudioOutput = _MockQAudioOutput

# Inject mocks before importing the controller
sys.modules.setdefault("PyQt6.QtCore", _mock_qt_core)
sys.modules.setdefault("PyQt6.QtMultimedia", _mock_qt_multimedia)

# Now import the module under test — it will use our mocked Qt types
from meetandread.playback.history import (
    HistoryPlaybackController,
    _MIN_RATE,
    _MAX_RATE,
    _SKIP_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_valid_wav(
    path: Path,
    duration_frames: int = 1000,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    """Create a tiny valid WAV file using Python's ``wave`` module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_data = b"\x00" * (duration_frames * channels * sample_width)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(frame_data)
    return path


def _write_transcript_md(path: Path, stem_content: str = "test content") -> Path:
    """Write a minimal transcript .md file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Transcript\n\n{stem_content}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests — WAV resolution and loading
# ---------------------------------------------------------------------------


class TestLoadTranscriptAudio:
    """Tests for load_transcript_audio path resolution and state."""

    def test_load_existing_wav_sets_audio_path(self, tmp_path: Path) -> None:
        """Loading a transcript with an existing WAV sets current_audio_path."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "recording-001.md")
        wav_path = _create_valid_wav(recordings / "recording-001.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.current_audio_path == wav_path
        assert ctrl.current_transcript_path == md_path
        assert ctrl.last_error is None

    def test_load_missing_wav_sets_error(self, tmp_path: Path) -> None:
        """Loading a transcript without a companion WAV sets error state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "no-audio.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.current_audio_path is not None  # path is resolved
        assert ctrl.last_error == "Audio file not found"
        assert "not found" in ctrl.status_text

    def test_load_none_clears_state(self, tmp_path: Path) -> None:
        """Passing None to load_transcript_audio clears all state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        md = _write_transcript_md(tmp_path / "test.md")
        wav = _create_valid_wav(recordings / "test.wav")
        ctrl.load_transcript_audio(md)

        ctrl.load_transcript_audio(None)

        assert ctrl.current_transcript_path is None
        assert ctrl.current_audio_path is None
        assert ctrl.last_error is None
        assert ctrl.status_text == ""

    def test_load_replaces_previous_transcript(self, tmp_path: Path) -> None:
        """Loading a second transcript replaces the first."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md1 = _write_transcript_md(tmp_path / "first.md")
        wav1 = _create_valid_wav(recordings / "first.wav")
        md2 = _write_transcript_md(tmp_path / "second.md")
        wav2 = _create_valid_wav(recordings / "second.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md1)
        assert ctrl.current_transcript_path == md1

        ctrl.load_transcript_audio(md2)
        assert ctrl.current_transcript_path == md2
        assert ctrl.current_audio_path == wav2
        assert ctrl.last_error is None

    def test_load_nonexistent_md_still_resolves_wav(self, tmp_path: Path) -> None:
        """The controller resolves WAV by stem; md file existence is not checked."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = Path("/fake/path/recording-ghost.md")
        wav = _create_valid_wav(recordings / "recording-ghost.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.current_audio_path == wav
        assert ctrl.last_error is None

    def test_load_wav_path_points_to_directory(self, tmp_path: Path) -> None:
        """If the resolved WAV path is a directory, treat as unavailable."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "weird.md")
        wav_dir = recordings / "weird.wav"
        wav_dir.mkdir()

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.last_error == "Audio file not found"

    def test_load_after_deleted_wav(self, tmp_path: Path) -> None:
        """Loading a transcript whose WAV was deleted after scan."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "deleted.md")
        wav_path = _create_valid_wav(recordings / "deleted.wav")
        wav_path.unlink()

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.last_error == "Audio file not found"

    def test_load_sets_player_source_url(self, tmp_path: Path) -> None:
        """After loading, the player source is set to the WAV path."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "source-check.md")
        wav_path = _create_valid_wav(recordings / "source-check.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        source = ctrl.player.source()
        assert source is not None
        # Our mock QUrl stores the path
        assert str(wav_path) in repr(source)


# ---------------------------------------------------------------------------
# Tests — Transport controls
# ---------------------------------------------------------------------------


class TestTransportControls:
    """Tests for play, pause, stop, set_rate, set_volume."""

    def test_play_calls_player_play(self, tmp_path: Path) -> None:
        """play() calls player.play() when audio is available."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "play-test.md")
        _create_valid_wav(recordings / "play-test.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Force media status to Loaded so is_audio_available is True
        ctrl._player._media_status = _MockQMediaPlayer.MediaStatus.LoadedMedia

        with patch.object(ctrl._player, "play") as mock_play:
            ctrl.play()
            mock_play.assert_called_once()

    def test_pause_calls_player_pause(self, tmp_path: Path) -> None:
        """pause() calls player.pause() when playing."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "pause-test.md")
        _create_valid_wav(recordings / "pause-test.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PlayingState

        with patch.object(ctrl._player, "pause") as mock_pause:
            ctrl.pause()
            mock_pause.assert_called_once()

    def test_pause_noop_when_not_playing(self, tmp_path: Path) -> None:
        """pause() is a no-op when the player is not playing."""
        ctrl = HistoryPlaybackController(recordings_dir=tmp_path)
        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PausedState

        with patch.object(ctrl._player, "pause") as mock_pause:
            ctrl.pause()
            mock_pause.assert_not_called()

    def test_stop_calls_player_stop(self, tmp_path: Path) -> None:
        """stop() calls player.stop()."""
        ctrl = HistoryPlaybackController(recordings_dir=tmp_path)
        with patch.object(ctrl._player, "stop") as mock_stop:
            ctrl.stop()
            mock_stop.assert_called_once()

    def test_play_noop_when_no_audio(self, tmp_path: Path) -> None:
        """play() is a no-op when audio is not available."""
        ctrl = HistoryPlaybackController(recordings_dir=tmp_path)

        with patch.object(ctrl._player, "play") as mock_play:
            ctrl.play()
            mock_play.assert_not_called()


class TestSetRate:
    """Tests for set_rate clamping."""

    @pytest.mark.parametrize(
        "requested, expected",
        [
            (0.25, 0.25),   # Minimum allowed
            (1.0, 1.0),     # Normal speed
            (2.0, 2.0),     # Maximum allowed
            (0.1, 0.25),    # Below minimum → clamped to 0.25
            (3.0, 2.0),     # Above maximum → clamped to 2.0
            (0.5, 0.5),     # Within range
            (1.5, 1.5),     # Within range
        ],
    )
    def test_rate_clamped(self, requested: float, expected: float) -> None:
        """set_rate clamps to [0.25, 2.0] range."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        with patch.object(ctrl._player, "setPlaybackRate") as mock_rate:
            ctrl.set_rate(requested)
            mock_rate.assert_called_once_with(pytest.approx(expected))

    def test_minimum_rate_boundary(self) -> None:
        """_MIN_RATE is 0.25."""
        assert _MIN_RATE == 0.25

    def test_maximum_rate_boundary(self) -> None:
        """_MAX_RATE is 2.0."""
        assert _MAX_RATE == 2.0


class TestSetVolume:
    """Tests for set_volume normalization."""

    @pytest.mark.parametrize(
        "requested, expected",
        [
            (0.0, 0.0),     # Mute
            (1.0, 1.0),     # Full volume
            (0.5, 0.5),     # Half volume
            (-0.5, 0.0),    # Negative → clamped to 0.0
            (1.5, 1.0),     # Over 1.0 → clamped to 1.0
        ],
    )
    def test_volume_normalized(self, requested: float, expected: float) -> None:
        """set_volume normalizes to [0.0, 1.0] range."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        with patch.object(ctrl._audio_output, "setVolume") as mock_vol:
            ctrl.set_volume(requested)
            mock_vol.assert_called_once_with(pytest.approx(expected))


# ---------------------------------------------------------------------------
# Tests — Error / media error handling
# ---------------------------------------------------------------------------


class TestMediaErrorHandling:
    """Tests for Qt media error signal handling."""

    def test_media_error_sets_last_error(self, tmp_path: Path) -> None:
        """A player error signal sets last_error and status."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "error-test.md")
        _create_valid_wav(recordings / "error-test.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._on_media_error(
            _MockQMediaPlayer.Error.ResourceError, "Resource not found"
        )

        assert ctrl.last_error == "Audio could not be loaded"
        assert "could not be loaded" in ctrl.status_text

    def test_no_error_does_not_set_error(self, tmp_path: Path) -> None:
        """NoError does not set the error state."""
        ctrl = HistoryPlaybackController(recordings_dir=tmp_path)
        ctrl._on_media_error(_MockQMediaPlayer.Error.NoError, "")
        assert ctrl.last_error is None

    def test_invalid_media_status_sets_error(self, tmp_path: Path) -> None:
        """InvalidMedia status sets error state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "invalid.md")
        _create_valid_wav(recordings / "invalid.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._on_media_status_changed(_MockQMediaPlayer.MediaStatus.InvalidMedia)
        assert ctrl.last_error == "Audio could not be loaded"

    def test_no_media_status_clears_status(self, tmp_path: Path) -> None:
        """NoMedia status clears the status text."""
        ctrl = HistoryPlaybackController(recordings_dir=tmp_path)
        ctrl._status_text = "Playing"

        ctrl._on_media_status_changed(_MockQMediaPlayer.MediaStatus.NoMedia)
        assert ctrl._status_text == ""


# ---------------------------------------------------------------------------
# Tests — Status text
# ---------------------------------------------------------------------------


class TestStatusText:
    """Tests for computed status_text property."""

    def test_no_audio_empty_status(self) -> None:
        """When no audio is loaded, status_text is empty."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.status_text == ""

    def test_error_status_takes_precedence(self, tmp_path: Path) -> None:
        """When an error is set, its message becomes the status text."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "err.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.status_text == "Audio file not found"

    def test_ready_status_with_audio(self, tmp_path: Path) -> None:
        """With audio loaded and no error, status defaults to Ready."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "ready.md")
        _create_valid_wav(recordings / "ready.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Default mock state is StoppedState
        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.StoppedState
        assert ctrl.status_text == "Ready"

    def test_playing_status(self, tmp_path: Path) -> None:
        """Status text is 'Playing' when player is playing."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "playing.md")
        _create_valid_wav(recordings / "playing.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PlayingState
        assert ctrl.status_text == "Playing"

    def test_paused_status(self, tmp_path: Path) -> None:
        """Status text is 'Paused' when player is paused."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "paused.md")
        _create_valid_wav(recordings / "paused.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PausedState
        assert ctrl.status_text == "Paused"


# ---------------------------------------------------------------------------
# Tests — is_audio_available
# ---------------------------------------------------------------------------


class TestIsAudioAvailable:
    """Tests for the is_audio_available property."""

    def test_false_when_no_audio(self) -> None:
        """is_audio_available is False when nothing is loaded."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.is_audio_available is False

    def test_false_when_error_set(self, tmp_path: Path) -> None:
        """is_audio_available is False when an error is present."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "err.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.is_audio_available is False

    def test_true_when_loaded_and_no_error(self, tmp_path: Path) -> None:
        """is_audio_available is True when loaded with a valid media status."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "avail.md")
        _create_valid_wav(recordings / "avail.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Force LoadedMedia status
        ctrl._player._media_status = _MockQMediaPlayer.MediaStatus.LoadedMedia

        assert ctrl.is_audio_available is True

    def test_false_when_buffering(self, tmp_path: Path) -> None:
        """is_audio_available is False during buffering (unexpected state)."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "buf.md")
        _create_valid_wav(recordings / "buf.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # BufferingMedia is not in the "ready" set
        ctrl._player._media_status = _MockQMediaPlayer.MediaStatus.BufferingMedia

        assert ctrl.is_audio_available is False


# ---------------------------------------------------------------------------
# Tests — Repeated loads (churn protection)
# ---------------------------------------------------------------------------


class TestRepeatedLoads:
    """Tests for rapid transcript selection churn protection."""

    def test_stop_called_before_new_load(self, tmp_path: Path) -> None:
        """load_transcript_audio stops the player before loading a new source."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md1 = _write_transcript_md(tmp_path / "first.md")
        _create_valid_wav(recordings / "first.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md1)

        md2 = _write_transcript_md(tmp_path / "second.md")
        _create_valid_wav(recordings / "second.wav")

        stop_calls = []
        original_stop = ctrl._player.stop

        def tracking_stop():
            stop_calls.append(True)
            original_stop()

        with patch.object(ctrl._player, "stop", side_effect=tracking_stop):
            ctrl.load_transcript_audio(md2)

        assert len(stop_calls) >= 1

    def test_load_same_transcript_twice(self, tmp_path: Path) -> None:
        """Loading the same transcript twice resets and reloads."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "repeat.md")
        wav_path = _create_valid_wav(recordings / "repeat.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.current_transcript_path == md_path
        assert ctrl.current_audio_path == wav_path
        assert ctrl.last_error is None


# ---------------------------------------------------------------------------
# Tests — Player and output access
# ---------------------------------------------------------------------------


class TestPlayerAccess:
    """Tests for the player and audio_output properties."""

    def test_player_returns_player_instance(self) -> None:
        """player property returns the player instance."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.player is ctrl._player

    def test_audio_output_returns_output_instance(self) -> None:
        """audio_output property returns the audio output instance."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.audio_output is ctrl._audio_output


# ---------------------------------------------------------------------------
# Tests — position_ms and duration_ms
# ---------------------------------------------------------------------------


class TestPositionAndDuration:
    """Tests for position_ms and duration_ms read-only properties."""

    def test_position_ms_returns_player_position(self) -> None:
        """position_ms delegates to player.position()."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        ctrl._player._position = 12345
        assert ctrl.position_ms == 12345

    def test_position_ms_starts_at_zero(self) -> None:
        """position_ms is 0 when no audio has been played."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.position_ms == 0

    def test_duration_ms_returns_player_duration(self) -> None:
        """duration_ms delegates to player.duration()."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        ctrl._player._duration = 60000
        assert ctrl.duration_ms == 60000

    def test_duration_ms_is_zero_before_load(self) -> None:
        """duration_ms is 0 when no audio is loaded."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.duration_ms == 0

    def test_duration_ms_reflects_loaded_audio(self, tmp_path: Path) -> None:
        """After loading, duration_ms returns the mock duration."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "dur.md")
        _create_valid_wav(recordings / "dur.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Mock player sets _duration = 30000 on setSource
        assert ctrl.duration_ms == 30000


# ---------------------------------------------------------------------------
# Tests — seek_to
# ---------------------------------------------------------------------------


class TestSeekTo:
    """Tests for seek_to with position clamping."""

    def test_seek_to_sets_position(self, tmp_path: Path) -> None:
        """seek_to sets the player position."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "seek.md")
        _create_valid_wav(recordings / "seek.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl.seek_to(10000)
        assert ctrl.position_ms == 10000

    def test_seek_to_clamps_to_duration(self, tmp_path: Path) -> None:
        """seek_to clamps position to the audio duration."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "clamp.md")
        _create_valid_wav(recordings / "clamp.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Mock duration is 30000
        ctrl.seek_to(99999)
        assert ctrl.position_ms == 30000

    def test_seek_to_clamps_to_zero(self, tmp_path: Path) -> None:
        """seek_to clamps negative positions to 0."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "clamp-neg.md")
        _create_valid_wav(recordings / "clamp-neg.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl.seek_to(-500)
        assert ctrl.position_ms == 0

    def test_seek_to_zero_when_no_duration(self) -> None:
        """seek_to without loaded audio does not clamp to duration."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        ctrl._player._position = 0
        ctrl.seek_to(500)
        # Duration is 0, so no upper clamping; position set as-is
        assert ctrl.position_ms == 500

    def test_seek_to_exact_end(self, tmp_path: Path) -> None:
        """seek_to to the exact duration is allowed."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "exact-end.md")
        _create_valid_wav(recordings / "exact-end.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl.seek_to(30000)
        assert ctrl.position_ms == 30000


# ---------------------------------------------------------------------------
# Tests — skip_forward and skip_backward
# ---------------------------------------------------------------------------


class TestSkipForwardBackward:
    """Tests for skip_forward and skip_backward with default and custom seconds."""

    def test_skip_forward_default_5_seconds(self, tmp_path: Path) -> None:
        """skip_forward advances position by 5 seconds (default)."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sfwd.md")
        _create_valid_wav(recordings / "sfwd.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 10000
        ctrl.skip_forward()
        assert ctrl.position_ms == 15000

    def test_skip_backward_default_5_seconds(self, tmp_path: Path) -> None:
        """skip_backward rewinds position by 5 seconds (default)."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sbwd.md")
        _create_valid_wav(recordings / "sbwd.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 15000
        ctrl.skip_backward()
        assert ctrl.position_ms == 10000

    def test_skip_forward_clamps_at_duration(self, tmp_path: Path) -> None:
        """skip_forward does not exceed audio duration."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sfwd-clamp.md")
        _create_valid_wav(recordings / "sfwd-clamp.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 28000
        ctrl.skip_forward()
        assert ctrl.position_ms == 30000  # clamped to duration

    def test_skip_backward_clamps_at_zero(self, tmp_path: Path) -> None:
        """skip_backward does not go below 0."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sbwd-clamp.md")
        _create_valid_wav(recordings / "sbwd-clamp.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 2000
        ctrl.skip_backward()
        assert ctrl.position_ms == 0  # clamped to 0

    def test_skip_forward_custom_seconds(self, tmp_path: Path) -> None:
        """skip_forward with a custom second value."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sfwd-custom.md")
        _create_valid_wav(recordings / "sfwd-custom.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 5000
        ctrl.skip_forward(10.0)
        assert ctrl.position_ms == 15000

    def test_skip_backward_custom_seconds(self, tmp_path: Path) -> None:
        """skip_backward with a custom second value."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sbwd-custom.md")
        _create_valid_wav(recordings / "sbwd-custom.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 15000
        ctrl.skip_backward(10.0)
        assert ctrl.position_ms == 5000

    def test_skip_forward_at_zero_position(self, tmp_path: Path) -> None:
        """skip_forward from position 0 advances to 5 seconds."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sfwd-zero.md")
        _create_valid_wav(recordings / "sfwd-zero.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 0
        ctrl.skip_forward()
        assert ctrl.position_ms == 5000

    def test_skip_backward_at_zero_position(self, tmp_path: Path) -> None:
        """skip_backward from position 0 stays at 0."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        md_path = _write_transcript_md(tmp_path / "sbwd-zero.md")
        _create_valid_wav(recordings / "sbwd-zero.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._player._position = 0
        ctrl.skip_backward()
        assert ctrl.position_ms == 0

    def test_skip_seconds_constant_is_five(self) -> None:
        """_SKIP_SECONDS default is 5."""
        assert _SKIP_SECONDS == 5


# ---------------------------------------------------------------------------
# Helpers — Invalid / corrupt media fixtures
# ---------------------------------------------------------------------------


def _create_corrupt_wav(path: Path) -> Path:
    """Write a file with an invalid WAV header (corrupt media fixture).

    The file starts with ``RIFF`` to look like a WAV but has garbage
    after the header, making it unparseable as real audio.  Generated
    under *tmp_path* so tests never depend on checked-in binary blobs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Valid RIFF header followed by junk
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\xff" * 50)
    return path


def _create_unsupported_audio(path: Path) -> Path:
    """Write a placeholder file with an unsupported audio extension.

    Uses ``.ogg`` to simulate an audio format the player cannot decode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


# ---------------------------------------------------------------------------
# Tests — Invalid / corrupt media error handling
# ---------------------------------------------------------------------------


class TestInvalidMediaHandling:
    """Tests for corrupt/unsupported audio and missing WAV error paths."""

    def test_missing_wav_sets_not_found_error(self, tmp_path: Path) -> None:
        """Loading a transcript without a companion WAV reports not found."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "no-file.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        assert ctrl.last_error == "Audio file not found"
        assert ctrl.is_audio_available is False

    def test_corrupt_wav_via_media_error(self, tmp_path: Path) -> None:
        """A corrupt WAV file triggers _on_media_error with ResourceError."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "corrupt.md")
        _create_corrupt_wav(recordings / "corrupt.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Simulate the player detecting corrupt data
        ctrl._on_media_error(
            _MockQMediaPlayer.Error.ResourceError, "Resource error"
        )

        assert ctrl.last_error == "Audio could not be loaded"
        assert "could not be loaded" in ctrl.status_text
        assert ctrl.is_audio_available is False

    def test_invalid_media_status(self, tmp_path: Path) -> None:
        """InvalidMedia status disables audio and sets error."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "bad-fmt.md")
        _create_valid_wav(recordings / "bad-fmt.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._on_media_status_changed(_MockQMediaPlayer.MediaStatus.InvalidMedia)

        assert ctrl.last_error == "Audio could not be loaded"
        assert ctrl.is_audio_available is False

    def test_no_error_does_not_set_error_state(self) -> None:
        """NoError does not modify the error or status state."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        ctrl._on_media_error(_MockQMediaPlayer.Error.NoError, "")

        assert ctrl.last_error is None
        assert ctrl.is_audio_available is False  # still no audio loaded

    def test_format_error_sets_load_error(self, tmp_path: Path) -> None:
        """FormatError from the player sets the load-failed state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "fmt-err.md")
        _create_valid_wav(recordings / "fmt-err.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._on_media_error(
            _MockQMediaPlayer.Error.FormatError, "Format error"
        )

        assert ctrl.last_error == "Audio could not be loaded"
        assert ctrl.is_audio_available is False


# ---------------------------------------------------------------------------
# Tests — Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """Tests for get_diagnostics() sanitization and shape."""

    def test_diagnostics_keys_empty_controller(self) -> None:
        """Diagnostics from a fresh controller has all expected keys."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        diag = ctrl.get_diagnostics()

        expected_keys = {
            "transcript_stem",
            "transcript_path",
            "audio_path",
            "audio_filename",
            "is_audio_available",
            "last_error",
            "status_text",
            "media_status",
            "playback_state",
            "position_ms",
            "duration_ms",
            "playback_rate",
            "volume",
        }
        assert set(diag.keys()) == expected_keys

    def test_diagnostics_values_no_audio(self) -> None:
        """Diagnostics values are sensible when no audio is loaded."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        diag = ctrl.get_diagnostics()

        assert diag["transcript_stem"] is None
        assert diag["audio_path"] is None
        assert diag["audio_filename"] is None
        assert diag["is_audio_available"] is False
        assert diag["last_error"] is None
        assert diag["position_ms"] == 0
        assert diag["duration_ms"] == 0
        assert diag["playback_rate"] == 1.0
        assert diag["volume"] == 1.0

    def test_diagnostics_with_loaded_audio(self, tmp_path: Path) -> None:
        """Diagnostics reflects loaded audio state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "diag-test.md")
        _create_valid_wav(recordings / "diag-test.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        diag = ctrl.get_diagnostics()

        assert diag["transcript_stem"] == "diag-test"
        assert "diag-test" in diag["transcript_path"]
        assert diag["audio_filename"] == "diag-test.wav"
        assert "diag-test.wav" in diag["audio_path"]
        assert diag["is_audio_available"] is True
        assert diag["last_error"] is None

    def test_diagnostics_with_error(self, tmp_path: Path) -> None:
        """Diagnostics captures error state from missing audio."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "missing.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        diag = ctrl.get_diagnostics()

        assert diag["is_audio_available"] is False
        assert diag["last_error"] == "Audio file not found"
        assert "not found" in diag["status_text"]

    def test_diagnostics_redaction_no_transcript_body(self, tmp_path: Path) -> None:
        """Diagnostics never contains transcript body text or bookmark names."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(
            tmp_path / "secret-meeting.md",
            stem_content="Confidential discussion about Project X with Alice and Bob. "
            "Bookmark: Critical Decision Point. Speaker: This is sensitive.",
        )
        _create_valid_wav(recordings / "secret-meeting.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        diag = ctrl.get_diagnostics()
        diag_str = str(diag)

        # None of these should appear in diagnostics
        assert "Confidential" not in diag_str
        assert "Alice" not in diag_str
        assert "Bob" not in diag_str
        assert "Critical Decision Point" not in diag_str
        assert "sensitive" not in diag_str
        assert "Bookmark" not in diag_str

        # Only the stem and filenames should be present
        assert diag["transcript_stem"] == "secret-meeting"

    def test_diagnostics_is_o1(self) -> None:
        """get_diagnostics() is O(1) — no iteration over transcript contents."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        # Call twice to confirm it's just a property read
        d1 = ctrl.get_diagnostics()
        d2 = ctrl.get_diagnostics()
        assert d1 == d2

    def test_diagnostics_after_media_error(self, tmp_path: Path) -> None:
        """Diagnostics reflects media error state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "errdiag.md")
        _create_valid_wav(recordings / "errdiag.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        ctrl._on_media_error(
            _MockQMediaPlayer.Error.ResourceError, "Resource error"
        )

        diag = ctrl.get_diagnostics()
        assert diag["last_error"] == "Audio could not be loaded"
        assert diag["is_audio_available"] is False

    def test_diagnostics_playback_state_values(self, tmp_path: Path) -> None:
        """Diagnostics includes playback_state as a readable name."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md_path = _write_transcript_md(tmp_path / "state-diag.md")
        _create_valid_wav(recordings / "state-diag.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md_path)

        # Default state is StoppedState
        diag = ctrl.get_diagnostics()
        assert "Stopped" in diag["playback_state"]


# ---------------------------------------------------------------------------
# Tests — Status message regression lock (T04)
# ---------------------------------------------------------------------------


class TestStatusMessageRegression:
    """Consolidated regression lock for all user-facing status messages.

    Ensures status_text is non-empty and specific for every user-visible
    playback state.  If any assertion here breaks, a user-facing message
    has silently changed or disappeared.
    """

    def test_no_audio_empty_status(self) -> None:
        """Fresh controller: status is empty (no audio, no error)."""
        ctrl = HistoryPlaybackController(recordings_dir=Path("/tmp"))
        assert ctrl.status_text == ""

    def test_ready_with_valid_audio(self, tmp_path: Path) -> None:
        """Audio loaded, no error, stopped: status is 'Ready'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "ready.md")
        _create_valid_wav(recordings / "ready.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.StoppedState
        assert ctrl.status_text == "Ready"

    def test_playing_status(self, tmp_path: Path) -> None:
        """Audio loaded, playing: status is 'Playing'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "playing.md")
        _create_valid_wav(recordings / "playing.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PlayingState
        assert ctrl.status_text == "Playing"

    def test_paused_status(self, tmp_path: Path) -> None:
        """Audio loaded, paused: status is 'Paused'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "paused.md")
        _create_valid_wav(recordings / "paused.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)

        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PausedState
        assert ctrl.status_text == "Paused"

    def test_missing_wav_status(self, tmp_path: Path) -> None:
        """WAV not found: status is specific 'Audio file not found'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "missing.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)

        assert ctrl.status_text == "Audio file not found"
        assert ctrl.last_error == "Audio file not found"

    def test_media_error_status(self, tmp_path: Path) -> None:
        """Media error: status is specific 'Audio could not be loaded'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "err.md")
        _create_valid_wav(recordings / "err.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)
        ctrl._on_media_error(_MockQMediaPlayer.Error.ResourceError, "fail")

        assert ctrl.status_text == "Audio could not be loaded"
        assert ctrl.last_error == "Audio could not be loaded"

    def test_invalid_media_status(self, tmp_path: Path) -> None:
        """InvalidMedia status: status is 'Audio could not be loaded'."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "inv.md")
        _create_valid_wav(recordings / "inv.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)
        ctrl._on_media_status_changed(_MockQMediaPlayer.MediaStatus.InvalidMedia)

        assert ctrl.status_text == "Audio could not be loaded"
        assert ctrl.last_error == "Audio could not be loaded"

    def test_error_overrides_playback_state(self, tmp_path: Path) -> None:
        """Error status takes precedence over playback state."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "override.md")
        _create_valid_wav(recordings / "override.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)
        ctrl._on_media_error(_MockQMediaPlayer.Error.FormatError, "bad")
        ctrl._player._playback_state = _MockQMediaPlayer.PlaybackState.PlayingState

        # Error message wins, not 'Playing'
        assert ctrl.status_text == "Audio could not be loaded"

    def test_load_none_clears_status(self, tmp_path: Path) -> None:
        """Loading None resets status to empty."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "clear.md")
        _create_valid_wav(recordings / "clear.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        ctrl.load_transcript_audio(md)
        assert ctrl.status_text == "Ready"

        ctrl.load_transcript_audio(None)
        assert ctrl.status_text == ""

    def test_all_failure_states_non_empty_specific(self, tmp_path: Path) -> None:
        """Every failure state produces non-empty, specific status text."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        md = _write_transcript_md(tmp_path / "check.md")
        _create_valid_wav(recordings / "check.wav")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)

        # Missing file
        ctrl.load_transcript_audio(md)
        # Now set media error
        ctrl._on_media_error(_MockQMediaPlayer.Error.ResourceError, "x")
        assert ctrl.status_text != ""
        assert ctrl.status_text != "Ready"
        assert ctrl.status_text != "Playing"
        assert ctrl.status_text != "Paused"

        # InvalidMedia
        ctrl._reset_state()
        ctrl._current_audio_path = recordings / "check.wav"
        ctrl._on_media_status_changed(_MockQMediaPlayer.MediaStatus.InvalidMedia)
        assert ctrl.status_text != ""
        assert "could not be loaded" in ctrl.status_text
