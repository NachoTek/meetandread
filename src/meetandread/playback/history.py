"""History audio playback controller.

Wraps PyQt6 QtMultimedia's ``QMediaPlayer`` and ``QAudioOutput`` to provide
a clean boundary for the History view's audio playback. Resolves companion
WAV files from the recordings directory and exposes transport controls with
structured logging and observable state.

Design contract:
    - Transcript .md files live in ``transcripts/``; companion WAVs live in
      ``recordings/``, sharing the same filename stem.
    - One ``QMediaPlayer`` / ``QAudioOutput`` pair per controller instance
      (one per Settings History panel).
    - Rapid transcript selection churn: ``load_transcript_audio`` stops the
      previous source before loading the next one.
    - All state transitions are logged (stem only, no content).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

from meetandread.audio.storage.paths import get_recordings_dir

logger = logging.getLogger(__name__)

# Allowed playback rate range
_MIN_RATE = 0.25
_MAX_RATE = 2.0

# Default skip interval in seconds
_SKIP_SECONDS = 5


class HistoryPlaybackController:
    """Manages audio playback for a single History panel.

    Resolves a transcript .md path to its companion WAV in the recordings
    directory, loads it into a ``QMediaPlayer``, and exposes play/pause/stop,
    speed, and volume controls with error state.

    Attributes:
        current_transcript_path: The loaded transcript .md path, or None.
        current_audio_path: The resolved WAV path, or None.
        is_audio_available: True when a valid audio source is loaded and the
            player is in a playable state.
        last_error: Human-readable error string, or None.
        status_text: Concise status for the UI header (e.g. ``"Ready"``,
            ``"Audio file not found"``, ``"Audio could not be loaded"``).
    """

    def __init__(
        self,
        recordings_dir: Optional[Path] = None,
    ) -> None:
        self._recordings_dir = recordings_dir
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)

        self._current_transcript_path: Optional[Path] = None
        self._current_audio_path: Optional[Path] = None
        self._last_error: Optional[str] = None
        self._status_text: str = ""

        # Connect media error signal for structured failure visibility
        self._player.errorOccurred.connect(self._on_media_error)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

    # ------------------------------------------------------------------
    # Public read-only state
    # ------------------------------------------------------------------

    @property
    def current_transcript_path(self) -> Optional[Path]:
        """The currently loaded transcript .md path, or None."""
        return self._current_transcript_path

    @property
    def current_audio_path(self) -> Optional[Path]:
        """The resolved WAV path for the current transcript, or None."""
        return self._current_audio_path

    @property
    def is_audio_available(self) -> bool:
        """True when audio is loaded and the player can play."""
        if self._current_audio_path is None:
            return False
        if self._last_error is not None:
            return False
        status = self._player.mediaStatus()
        return status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        )

    @property
    def last_error(self) -> Optional[str]:
        """Human-readable error string, or None if no error."""
        return self._last_error

    @property
    def status_text(self) -> str:
        """Concise status for UI header display."""
        if self._status_text:
            return self._status_text
        if self._current_audio_path is None:
            return ""
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            return "Playing"
        if state == QMediaPlayer.PlaybackState.PausedState:
            return "Paused"
        return "Ready"

    @property
    def player(self) -> QMediaPlayer:
        """The underlying QMediaPlayer (for testing / signal wiring)."""
        return self._player

    @property
    def audio_output(self) -> QAudioOutput:
        """The underlying QAudioOutput (for testing / signal wiring)."""
        return self._audio_output

    @property
    def position_ms(self) -> int:
        """Current playback position in milliseconds."""
        return self._player.position()

    @property
    def duration_ms(self) -> int:
        """Total duration of the loaded audio in milliseconds.

        Returns 0 when no audio is loaded or duration is unknown.
        """
        return self._player.duration()

    # ------------------------------------------------------------------
    # Core: load transcript audio
    # ------------------------------------------------------------------

    def load_transcript_audio(self, md_path: Optional[Path]) -> None:
        """Resolve and load the companion WAV for a transcript .md file.

        Stops any currently playing media, resolves the WAV path in the
        recordings directory, and assigns it to the player. When the WAV
        is missing or the path is invalid, sets a disabled state with an
        appropriate error/status message.

        Args:
            md_path: Path to a transcript .md file, or None to unload.
        """
        # Stop and clear previous source (handles rapid selection churn)
        self._player.stop()
        self._reset_state()

        if md_path is None:
            logger.info("load_requested: md_path=None, unloading audio")
            self._status_text = ""
            return

        logger.info("load_requested: stem=%s", md_path.stem)

        self._current_transcript_path = md_path

        # Validate the path is a real file (not a dir, not metadata-derived)
        if not isinstance(md_path, Path):
            self._set_error("Invalid transcript path")
            logger.warning("load_error: path is not a Path object: %r", md_path)
            return

        # Resolve companion WAV in the recordings directory
        recordings = self._resolve_recordings_dir()
        wav_path = recordings / f"{md_path.stem}.wav"

        self._current_audio_path = wav_path

        if not wav_path.exists() or not wav_path.is_file():
            self._set_error("Audio file not found")
            logger.warning(
                "load_missing: stem=%s expected=%s", md_path.stem, wav_path
            )
            return

        # Assign source to player
        source_url = QUrl.fromLocalFile(str(wav_path))
        self._player.setSource(source_url)
        logger.info("load_ready: stem=%s source=%s", md_path.stem, wav_path.name)

    # ------------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback. No-op if audio is unavailable."""
        if not self.is_audio_available:
            logger.info("play: skipped, audio not available")
            return
        logger.info("play: stem=%s", self._stem_or_none())
        self._player.play()

    def pause(self) -> None:
        """Pause playback. No-op if not playing."""
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        logger.info("pause: stem=%s", self._stem_or_none())
        self._player.pause()

    def stop(self) -> None:
        """Stop playback and reset position to the start."""
        logger.info("stop: stem=%s", self._stem_or_none())
        self._player.stop()

    def set_rate(self, rate: float) -> None:
        """Set the playback rate, clamped to [0.25, 2.0].

        Args:
            rate: Desired playback speed multiplier (1.0 = normal).
        """
        clamped = max(_MIN_RATE, min(_MAX_RATE, float(rate)))
        logger.info("rate_changed: rate=%.2f (requested=%.2f)", clamped, rate)
        self._player.setPlaybackRate(clamped)

    def set_volume(self, volume: float) -> None:
        """Set the playback volume, normalized to [0.0, 1.0].

        Args:
            volume: Volume level from 0.0 (mute) to 1.0 (max).
        """
        normalized = max(0.0, min(1.0, float(volume)))
        logger.info("volume_changed: volume=%.2f (requested=%.2f)", normalized, volume)
        self._audio_output.setVolume(normalized)

    def seek_to(self, position_ms: int) -> None:
        """Seek to an absolute position, clamped to [0, duration].

        Args:
            position_ms: Target position in milliseconds.
        """
        position_ms = max(0, int(position_ms))
        duration = self._player.duration()
        if duration > 0:
            position_ms = min(position_ms, duration)
        logger.info(
            "seek_to: position_ms=%d duration=%d stem=%s",
            position_ms,
            duration,
            self._stem_or_none(),
        )
        self._player.setPosition(position_ms)

    def skip_forward(self, seconds: float = _SKIP_SECONDS) -> None:
        """Skip forward by the given number of seconds.

        Position is clamped to the end of the audio.

        Args:
            seconds: Number of seconds to skip (default: 5).
        """
        offset_ms = int(seconds * 1000)
        new_pos = self._player.position() + offset_ms
        duration = self._player.duration()
        if duration > 0:
            new_pos = min(new_pos, duration)
        logger.info(
            "skip_forward: seconds=%.1f position_ms=%d (from=%d) stem=%s",
            seconds,
            new_pos,
            self._player.position(),
            self._stem_or_none(),
        )
        self._player.setPosition(new_pos)

    def skip_backward(self, seconds: float = _SKIP_SECONDS) -> None:
        """Skip backward by the given number of seconds.

        Position is clamped to 0.

        Args:
            seconds: Number of seconds to skip (default: 5).
        """
        offset_ms = int(seconds * 1000)
        new_pos = max(0, self._player.position() - offset_ms)
        logger.info(
            "skip_backward: seconds=%.1f position_ms=%d (from=%d) stem=%s",
            seconds,
            new_pos,
            self._player.position(),
            self._stem_or_none(),
        )
        self._player.setPosition(new_pos)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset controller state for a new load."""
        self._current_transcript_path = None
        self._current_audio_path = None
        self._last_error = None
        self._status_text = ""

    def _set_error(self, message: str) -> None:
        """Set an error state with a status message."""
        self._last_error = message
        self._status_text = message

    def _resolve_recordings_dir(self) -> Path:
        """Return the recordings directory path."""
        if self._recordings_dir is not None:
            return self._recordings_dir
        return get_recordings_dir()

    def _stem_or_none(self) -> str:
        """Return the current transcript stem or '<none>'."""
        if self._current_transcript_path is not None:
            return self._current_transcript_path.stem
        return "<none>"

    # ------------------------------------------------------------------
    # Qt signal handlers
    # ------------------------------------------------------------------

    def _on_media_error(
        self,
        error: QMediaPlayer.Error,
        error_string: str,
    ) -> None:
        """Handle QMediaPlayer error signals."""
        if error == QMediaPlayer.Error.NoError:
            return
        self._set_error("Audio could not be loaded")
        logger.warning(
            "load_error: stem=%s error=%s message=%s",
            self._stem_or_none(),
            error,
            error_string,
        )

    def _on_media_status_changed(
        self,
        status: QMediaPlayer.MediaStatus,
    ) -> None:
        """Handle media status changes for state consistency."""
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._set_error("Audio could not be loaded")
            logger.warning(
                "load_error: stem=%s status=InvalidMedia",
                self._stem_or_none(),
            )
        elif status == QMediaPlayer.MediaStatus.NoMedia:
            self._status_text = ""

    def _on_playback_state_changed(
        self,
        state: QMediaPlayer.PlaybackState,
    ) -> None:
        """Handle playback state changes for status text updates."""
        # Status text is computed on-demand in the property; nothing to
        # update here, but the signal connection keeps the controller
        # eligible for future observability extensions.
        pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return sanitized playback diagnostics for testing/inspection.

        Exposes playback state, source names (stems/paths only), media
        status, error state, position, duration, rate, and volume.
        Does NOT expose transcript body text, bookmark names, speaker
        text, or audio content.

        Returns:
            Dict of sanitized diagnostic key/value pairs.
        """
        media_status = self._player.mediaStatus()
        playback_state = self._player.playbackState()

        # Resolve enum names safely (works with both real Qt and mocks)
        def _enum_name(enum_val: object, fallback: str) -> str:
            name = getattr(enum_val, "name", None)
            if name is not None:
                return name
            try:
                return str(enum_val)
            except Exception:
                return fallback

        return {
            "transcript_stem": (
                self._current_transcript_path.stem
                if self._current_transcript_path
                else None
            ),
            "transcript_path": (
                str(self._current_transcript_path)
                if self._current_transcript_path
                else None
            ),
            "audio_path": (
                str(self._current_audio_path)
                if self._current_audio_path
                else None
            ),
            "audio_filename": (
                self._current_audio_path.name
                if self._current_audio_path
                else None
            ),
            "is_audio_available": self.is_audio_available,
            "last_error": self._last_error,
            "status_text": self.status_text,
            "media_status": _enum_name(media_status, "unknown"),
            "playback_state": _enum_name(playback_state, "unknown"),
            "position_ms": self._player.position(),
            "duration_ms": self._player.duration(),
            "playback_rate": self._player.playbackRate(),
            "volume": self._audio_output.volume(),
        }
