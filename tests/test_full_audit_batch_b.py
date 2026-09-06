"""Full-audit batch B logging tests (issue #100).

Red-first suite for the storage/denoising/playback logging audit.
Per module: (a) DEBUG-trail tests for named internal events, (b) INFO
quietness (per-step events must not appear at INFO), (c) operational
facts asserted at INFO. Mirrors the caplog prior art in
tests/test_audio_session_denoising.py (caplog.at_level + r.getMessage()).

Fix round (PR #117 review): the NEW events carry pure numeric/count
outcomes (never stems, filenames, custom-root basenames, or hashes
derived from them), the denoise accepted event fires only after
validation succeeds, bookmark named= reflects caller-supplied names,
and release/provider lifecycle events exist. Privacy canaries drive a
distinctive stem and a distinctive custom-root basename through the
lane and assert they never appear in the trail at any log level.
"""

import json
import logging
import sys
import wave
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from meetandread.audio.storage import paths as storage_paths
from meetandread.audio.storage.pcm_part import PcmPartWriter, PcmMetadata
from meetandread.audio.storage.recovery import recover_part_files
from meetandread.audio.storage.wav_finalize import (
    finalize_part_to_wav,
    finalize_stem,
)
from meetandread.audio.denoising import SpectralGateProvider, create_provider
from meetandread.playback import bookmark as bookmark_mod
from meetandread.playback.bookmark import BookmarkError, BookmarkManager
from meetandread.transcription import transcript_footer


# ---------------------------------------------------------------------------
# Mock Qt types (mirrors tests/test_history_playback_controller.py)
# ---------------------------------------------------------------------------

_mock_qt_multimedia = ModuleType("PyQt6.QtMultimedia")


class _MockQMediaPlayer:
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
        self._playback_state = self.PlaybackState.StoppedState
        self._media_status = self.MediaStatus.NoMedia
        self._position = 0
        self._duration = 0
        self.errorOccurred = MagicMock()
        self.mediaStatusChanged = MagicMock()
        self.playbackStateChanged = MagicMock()
        self.positionChanged = MagicMock()
        self.durationChanged = MagicMock()

    def source(self):
        return self._source

    def setAudioOutput(self, output):
        pass

    def setSource(self, url):
        self._source = url
        self._media_status = self.MediaStatus.LoadedMedia
        self._duration = 30000

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
    def __init__(self):
        self._volume = 1.0

    def setVolume(self, volume):
        self._volume = volume

    def volume(self):
        return self._volume


_mock_qt_multimedia.QMediaPlayer = _MockQMediaPlayer
_mock_qt_multimedia.QAudioOutput = _MockQAudioOutput
sys.modules.setdefault("PyQt6.QtMultimedia", _mock_qt_multimedia)

try:
    from PyQt6.QtCore import QUrl as _RealQUrl  # noqa: F401
except ImportError:
    _mock_qt_core = ModuleType("PyQt6.QtCore")

    class _MockQUrl:
        def __init__(self, url_str=""):
            self._url = url_str

        @classmethod
        def fromLocalFile(cls, path):
            return cls(f"file:///{path}")

    _mock_qt_core.QUrl = _MockQUrl
    sys.modules.setdefault("PyQt6.QtCore", _mock_qt_core)

from meetandread.playback.history import HistoryPlaybackController  # noqa: E402


# ---------------------------------------------------------------------------
# caplog helpers
# ---------------------------------------------------------------------------


def _messages(caplog, logger_name: str) -> list:
    """Formatted messages captured from a specific logger."""
    return [r.getMessage() for r in caplog.records if r.name == logger_name]


def _debug_messages(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.DEBUG
    ]


def _starts_with(messages: list, prefix: str) -> list:
    return [m for m in messages if m.startswith(prefix)]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_part_files(
    recordings_dir: Path,
    stem: str,
    n_frames: int = 100,
    sample_rate: int = 16000,
) -> Path:
    """Write a valid .pcm.part + sidecar pair; returns the part path."""
    pcm = b"\x00\x01" * n_frames
    part = recordings_dir / f"{stem}.pcm.part"
    part.write_bytes(pcm)
    meta = recordings_dir / f"{stem}.pcm.part.json"
    meta.write_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "channels": 1,
                "sample_width_bytes": 2,
            }
        ),
        encoding="utf-8",
    )
    return part


def _write_bookmark_transcript(path: Path) -> Path:
    content = transcript_footer.join(
        "# Transcript\n\nbody\n",
        {"words": [], "bookmarks": []},
    )
    path.write_text(content, encoding="utf-8")
    return path


def _create_valid_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)
    return path


# ---------------------------------------------------------------------------
# Canary helpers (fix round: privacy canaries)
# ---------------------------------------------------------------------------

CANARY_STEM = "QuarterlyReviewSecretTitle"
CANARY_ROOT_NAME = "dks-private-username"

# Lane loggers whose new-event trail is audited by the canaries.
LANE_LOGGERS = [
    "meetandread.audio.storage.paths",
    "meetandread.audio.storage.pcm_part",
    "meetandread.audio.storage.recovery",
    "meetandread.audio.storage.wav_finalize",
    "meetandread.audio.denoising",
    "meetandread.playback.history",
    "meetandread.playback.bookmark",
]

# Pinned pre-existing events (on main before this lane) that still carry
# stems; they are an already-reviewed surface. The canaries inspect every
# record at every level and exempt only this vocabulary.
PINNED_STEM_EVENT_PREFIXES = (
    "load_requested:",
    "load_ready:",
    "load_missing:",
    "play:",
    "pause:",
    "stop:",
    "release_source:",
    "seek_to:",
    "skip_forward:",
    "skip_backward:",
    "bookmark_added:",
    "bookmark_deleted:",
    "Failed to recover",
)


def _drive_full_stem_lifecycle(recordings: Path, stem: str) -> None:
    """Drive a stem through part write, wav finalize, and recovery."""
    writer = PcmPartWriter.create(
        stem=stem,
        recordings_dir=recordings,
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
    )
    writer.write_frames_i16(b"\x00\x01" * 100)
    writer.flush()
    writer.close()
    finalize_stem(stem, recordings, delete_part=True)
    recovery_dir = recordings / "recovery"
    recovery_dir.mkdir()
    _write_part_files(recovery_dir, stem)
    recover_part_files(recovery_dir)


# ===========================================================================
# paths.py
# ===========================================================================

PATHS_LOGGER = "meetandread.audio.storage.paths"


class TestPathsLogging:
    def test_custom_path_debug_event(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        custom = tmp_path / "customrec"
        custom.mkdir()
        monkeypatch.setattr(
            storage_paths,
            "_resolve_custom_path",
            lambda field: custom if field == "recordings_path" else None,
        )
        with caplog.at_level(logging.DEBUG, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir()
        debugs = _debug_messages(caplog, PATHS_LOGGER)
        custom_lines = _starts_with(debugs, "storage_path_custom:")
        assert custom_lines
        assert "kind=recordings" in custom_lines[0]
        assert "source=custom" in custom_lines[0]
        assert "root=" not in custom_lines[0]

    def test_custom_path_info_root_line(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        custom = tmp_path / "customrec"
        custom.mkdir()
        monkeypatch.setattr(
            storage_paths,
            "_resolve_custom_path",
            lambda field: custom if field == "recordings_path" else None,
        )
        with caplog.at_level(logging.INFO, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir()
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == PATHS_LOGGER and r.levelno == logging.INFO
        ]
        resolved = _starts_with(infos, "storage_root_resolved: kind=recordings")
        assert resolved
        assert "source=custom" in resolved[0]
        assert "root=" not in resolved[0]

    def test_default_path_info_root_line_no_root(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(
            storage_paths, "_resolve_custom_path", lambda field: None
        )
        with caplog.at_level(logging.INFO, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir(base_dir=tmp_path)
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == PATHS_LOGGER and r.levelno == logging.INFO
        ]
        resolved = _starts_with(infos, "storage_root_resolved: kind=recordings")
        assert resolved
        assert "source=default" in resolved[0]
        assert "root=" not in resolved[0]

    def test_custom_root_basename_never_logged(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        custom = tmp_path / CANARY_ROOT_NAME
        custom.mkdir()
        monkeypatch.setattr(
            storage_paths,
            "_resolve_custom_path",
            lambda field: custom if field == "recordings_path" else None,
        )
        with caplog.at_level(logging.DEBUG, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir()
        for rec in caplog.records:
            if rec.name != PATHS_LOGGER:
                continue
            assert CANARY_ROOT_NAME not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_ROOT_NAME not in str(arg)

    def test_fallback_default_debug_event(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(
            storage_paths, "_resolve_custom_path", lambda field: None
        )
        with caplog.at_level(logging.DEBUG, logger=PATHS_LOGGER):
            storage_paths.get_transcripts_dir()
        debugs = _debug_messages(caplog, PATHS_LOGGER)
        assert _starts_with(debugs, "storage_path_fallback_default: kind=transcripts")

    def test_explicit_base_dir_debug_event(
        self, tmp_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=PATHS_LOGGER):
            storage_paths.get_logs_dir(base_dir=tmp_path)
        debugs = _debug_messages(caplog, PATHS_LOGGER)
        assert _starts_with(debugs, "storage_path_explicit_base: kind=logs")

    def test_info_quietness_no_step_events(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(
            storage_paths, "_resolve_custom_path", lambda field: None
        )
        with caplog.at_level(logging.INFO, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir()
        msgs = _messages(caplog, PATHS_LOGGER)
        assert not _starts_with(msgs, "storage_path_custom:")
        assert not _starts_with(msgs, "storage_path_fallback_default:")
        assert not _starts_with(msgs, "storage_path_explicit_base:")

    def test_privacy_no_absolute_paths(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        custom = tmp_path / "customrec"
        custom.mkdir()
        monkeypatch.setattr(
            storage_paths,
            "_resolve_custom_path",
            lambda field: custom if field == "recordings_path" else None,
        )
        with caplog.at_level(logging.DEBUG, logger=PATHS_LOGGER):
            storage_paths.get_recordings_dir()
            storage_paths.get_transcripts_dir()
        for msg in _messages(caplog, PATHS_LOGGER):
            assert str(tmp_path) not in msg
            assert "Users" not in msg
            assert "/" not in msg and "\\" not in msg


# ===========================================================================
# pcm_part.py
# ===========================================================================

PCM_LOGGER = "meetandread.audio.storage.pcm_part"


class TestPcmPartLogging:
    def _drive(self, tmp_path: Path, frames: bytes = b"\x00\x01" * 100) -> None:
        writer = PcmPartWriter.create(
            stem="test-stem",
            recordings_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
        )
        writer.write_frames_i16(frames)
        writer.flush()
        writer.close()

    def test_lifecycle_debug_events(self, tmp_path: Path, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=PCM_LOGGER):
            self._drive(tmp_path)
        debugs = _debug_messages(caplog, PCM_LOGGER)
        assert _starts_with(debugs, "part_created:")
        assert _starts_with(debugs, "part_appended:")
        assert _starts_with(debugs, "part_flushed:")
        assert _starts_with(debugs, "part_closed:")
        assert not any("stem=" in m or "id=" in m for m in debugs)

    def test_size_threshold_debug_event(self, tmp_path: Path, caplog) -> None:
        big = b"\x00\x01" * 300000  # 600000 bytes x2 crosses 1 MiB once
        with caplog.at_level(logging.DEBUG, logger=PCM_LOGGER):
            self._drive(tmp_path, frames=big * 2)
        debugs = _debug_messages(caplog, PCM_LOGGER)
        assert _starts_with(debugs, "part_size_threshold:")
        assert any("threshold_bytes=" in m for m in debugs)

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        with caplog.at_level(logging.INFO, logger=PCM_LOGGER):
            self._drive(tmp_path)
        msgs = _messages(caplog, PCM_LOGGER)
        assert not _starts_with(msgs, "part_created:")
        assert not _starts_with(msgs, "part_appended:")
        assert not _starts_with(msgs, "part_closed:")

    def test_privacy_no_absolute_paths(self, tmp_path: Path, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=PCM_LOGGER):
            self._drive(tmp_path)
        for msg in _messages(caplog, PCM_LOGGER):
            assert str(tmp_path) not in msg

    def test_canary_stem_never_logged(self, tmp_path: Path, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=PCM_LOGGER):
            _drive_full_stem_lifecycle(tmp_path, CANARY_STEM)
        for rec in caplog.records:
            if rec.name != PCM_LOGGER:
                continue
            assert CANARY_STEM not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg)


# ===========================================================================
# recovery.py
# ===========================================================================

RECOVERY_LOGGER = "meetandread.audio.storage.recovery"


class TestRecoveryLogging:
    def _make_mixed_dir(self, tmp_path: Path) -> Path:
        _write_part_files(tmp_path, "good-1")
        _write_part_files(tmp_path, "good-2")
        corrupt = tmp_path / "corrupt.pcm.part"
        corrupt.write_bytes(b"\x00\x01" * 50)
        (tmp_path / "corrupt.pcm.part.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        return tmp_path

    def test_per_file_debug_and_outcome_info(
        self, tmp_path: Path, caplog
    ) -> None:
        recordings = self._make_mixed_dir(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=RECOVERY_LOGGER):
            recovered = recover_part_files(recordings)
        assert len(recovered) == 2
        debugs = _debug_messages(caplog, RECOVERY_LOGGER)
        starts = _starts_with(debugs, "part_recovery_start:")
        assert len(starts) == 3
        assert "index=1" in starts[0] and "total=3" in starts[0]
        assert len(_starts_with(debugs, "part_recovered:")) == 2
        assert _starts_with(debugs, "part_recovery_backup:")
        # New events carry index/total/outcome counts, never stems or
        # stem-derived ids.
        for m in starts + _starts_with(debugs, "part_recovered:"):
            assert "id=" not in m
            assert "good-1" not in m and "good-2" not in m

    def test_outcome_info_line(self, tmp_path: Path, caplog) -> None:
        recordings = self._make_mixed_dir(tmp_path)
        with caplog.at_level(logging.INFO, logger=RECOVERY_LOGGER):
            recover_part_files(recordings)
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == RECOVERY_LOGGER and r.levelno == logging.INFO
        ]
        outcome = _starts_with(infos, "part_recovery:")
        assert len(outcome) == 1
        assert "recovered=2" in outcome[0]
        assert "skipped=0" in outcome[0]
        assert "failed=1" in outcome[0]

    def test_failure_keeps_existing_error(self, tmp_path: Path, caplog) -> None:
        recordings = self._make_mixed_dir(tmp_path)
        with caplog.at_level(logging.INFO, logger=RECOVERY_LOGGER):
            recover_part_files(recordings)
        errors = [
            r
            for r in caplog.records
            if r.name == RECOVERY_LOGGER and r.levelno == logging.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].getMessage().startswith("Failed to recover ")

    def test_delete_original_part_removed_debug(
        self, tmp_path: Path, caplog
    ) -> None:
        _write_part_files(tmp_path, "del-1")
        with caplog.at_level(logging.DEBUG, logger=RECOVERY_LOGGER):
            recover_part_files(tmp_path, delete_original=True)
        debugs = _debug_messages(caplog, RECOVERY_LOGGER)
        removed = _starts_with(debugs, "part_removed:")
        assert removed
        assert "id=" not in removed[0]
        assert "del-1" not in removed[0]
        assert not (tmp_path / "del-1.pcm.part").exists()

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        recordings = self._make_mixed_dir(tmp_path)
        with caplog.at_level(logging.INFO, logger=RECOVERY_LOGGER):
            recover_part_files(recordings)
        msgs = _messages(caplog, RECOVERY_LOGGER)
        assert not _starts_with(msgs, "part_recovery_start:")
        assert not _starts_with(msgs, "part_recovered:")
        assert not _starts_with(msgs, "part_recovery_backup:")

    def test_privacy_new_events_no_absolute_paths(
        self, tmp_path: Path, caplog
    ) -> None:
        recordings = self._make_mixed_dir(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=RECOVERY_LOGGER):
            recover_part_files(recordings)
        new_events = [
            m
            for m in _debug_messages(caplog, RECOVERY_LOGGER)
            + [
                r.getMessage()
                for r in caplog.records
                if r.name == RECOVERY_LOGGER and r.levelno == logging.INFO
            ]
            if not m.startswith("Failed to recover")
        ]
        assert new_events
        for msg in new_events:
            assert str(tmp_path) not in msg

    def test_canary_stem_never_logged(self, tmp_path: Path, caplog) -> None:
        _write_part_files(tmp_path, CANARY_STEM)
        with caplog.at_level(logging.DEBUG, logger=RECOVERY_LOGGER):
            recovered = recover_part_files(tmp_path)
        assert len(recovered) == 1
        for rec in caplog.records:
            if rec.name != RECOVERY_LOGGER:
                continue
            if rec.getMessage().startswith("Failed to recover"):
                continue  # pinned pre-existing ERROR may carry the path
            assert CANARY_STEM not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg)


# ===========================================================================
# wav_finalize.py
# ===========================================================================

WAV_LOGGER = "meetandread.audio.storage.wav_finalize"


class TestWavFinalizeLogging:
    def _part_pair(self, tmp_path: Path, n_frames: int = 16000):
        part = tmp_path / "fin.pcm.part"
        part.write_bytes(b"\x00\x01" * n_frames)
        meta = PcmMetadata(
            sample_rate=16000, channels=1, sample_width_bytes=2
        )
        return part, meta

    def test_finalize_debug_steps(self, tmp_path: Path, caplog) -> None:
        part, meta = self._part_pair(tmp_path)
        wav = tmp_path / "fin.wav"
        with caplog.at_level(logging.DEBUG, logger=WAV_LOGGER):
            finalize_part_to_wav(part, wav, meta)
        debugs = _debug_messages(caplog, WAV_LOGGER)
        assert _starts_with(debugs, "wav_pcm_read:")
        assert _starts_with(debugs, "wav_header_params:")
        assert any(
            m.startswith("wav_header_params:") and "frames=16000" in m
            for m in debugs
        )
        assert _starts_with(debugs, "wav_duration:")
        # New events carry byte/param facts only, never part/wav names
        # or ids derived from them.
        for m in _starts_with(debugs, "wav_pcm_read:"):
            assert "id=" not in m
            assert "fin.pcm.part" not in m

    def test_finalize_info_completion(self, tmp_path: Path, caplog) -> None:
        part, meta = self._part_pair(tmp_path)
        wav = tmp_path / "fin.wav"
        with caplog.at_level(logging.INFO, logger=WAV_LOGGER):
            finalize_part_to_wav(part, wav, meta)
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == WAV_LOGGER and r.levelno == logging.INFO
        ]
        done = _starts_with(infos, "wav_finalized:")
        assert len(done) == 1
        assert "duration_s=1.00" in done[0]
        assert "id=" not in done[0]
        assert "stem=" not in done[0] and "wav=" not in done[0]

    def test_finalize_stem_part_removed_debug(
        self, tmp_path: Path, caplog
    ) -> None:
        _write_part_files(tmp_path, "stemdel")
        with caplog.at_level(logging.DEBUG, logger=WAV_LOGGER):
            finalize_stem("stemdel", tmp_path, delete_part=True)
        debugs = _debug_messages(caplog, WAV_LOGGER)
        assert _starts_with(debugs, "part_removed:")
        assert not (tmp_path / "stemdel.pcm.part").exists()
        assert not (tmp_path / "stemdel.pcm.part.json").exists()

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        part, meta = self._part_pair(tmp_path)
        with caplog.at_level(logging.INFO, logger=WAV_LOGGER):
            finalize_part_to_wav(part, tmp_path / "fin.wav", meta)
        msgs = _messages(caplog, WAV_LOGGER)
        assert not _starts_with(msgs, "wav_pcm_read:")
        assert not _starts_with(msgs, "wav_header_params:")
        assert not _starts_with(msgs, "wav_duration:")

    def test_canary_stem_never_logged(self, tmp_path: Path, caplog) -> None:
        _write_part_files(tmp_path, CANARY_STEM)
        with caplog.at_level(logging.DEBUG, logger=WAV_LOGGER):
            finalize_stem(CANARY_STEM, tmp_path, delete_part=True)
        assert (tmp_path / f"{CANARY_STEM}.wav").exists()
        for rec in caplog.records:
            if rec.name != WAV_LOGGER:
                continue
            assert CANARY_STEM not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg)


# ===========================================================================
# denoising.py
# ===========================================================================

DENOISE_LOGGER = "meetandread.audio.denoising"


class TestDenoisingLogging:
    def test_accepted_frame_debug_event(self, caplog) -> None:
        provider = SpectralGateProvider()
        frame = np.zeros(512, dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            provider.process(frame)
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        assert _starts_with(debugs, "denoise_frame_accepted:")
        assert any("samples=512" in m for m in debugs)

    def test_overlap_buffered_debug_event(self, caplog) -> None:
        provider = SpectralGateProvider()
        frame = np.zeros(512, dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            provider.process(frame)
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        assert _starts_with(debugs, "denoise_overlap_buffered:")

    def test_fallback_frame_debug_event(self, caplog) -> None:
        provider = SpectralGateProvider()
        two_d = np.zeros((4, 8), dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            result = provider.process(two_d)
        assert result.fallback
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        assert _starts_with(debugs, "denoise_frame_fallback:")

    def test_bad_dtype_frame_fallback_not_accepted(self, caplog) -> None:
        provider = SpectralGateProvider()
        bad_dtype = np.full(512, "not-a-number", dtype=object)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            result = provider.process(bad_dtype)
        assert result.fallback
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        assert _starts_with(debugs, "denoise_frame_fallback:")
        assert not _starts_with(debugs, "denoise_frame_accepted:")

    def test_two_d_frame_fallback_not_accepted(self, caplog) -> None:
        provider = SpectralGateProvider()
        two_d = np.zeros((4, 8), dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            result = provider.process(two_d)
        assert result.fallback
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        assert _starts_with(debugs, "denoise_frame_fallback:")
        assert not _starts_with(debugs, "denoise_frame_accepted:")

    def test_processing_error_keeps_existing_warning(self, caplog) -> None:
        provider = SpectralGateProvider()

        def boom(frame):
            raise RuntimeError("kaboom")

        provider._spectral_gate = boom
        frame = np.zeros(8, dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            result = provider.process(frame)
        assert result.fallback
        warnings = [
            r
            for r in caplog.records
            if r.name == DENOISE_LOGGER and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].getMessage().startswith(
            "Denoising processing error, falling back to sanitized input"
        )

    def test_provider_created_debug_event(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger=DENOISE_LOGGER):
            create_provider("spectral_gate")
        debugs = _debug_messages(caplog, DENOISE_LOGGER)
        created = _starts_with(debugs, "provider_created:")
        assert created
        assert "provider=spectral_gate" in created[0]

    def test_info_quietness(self, caplog) -> None:
        provider = SpectralGateProvider()
        frame = np.zeros(512, dtype=np.float32)
        with caplog.at_level(logging.INFO, logger=DENOISE_LOGGER):
            provider.process(frame)
            provider.process(np.zeros((2, 4), dtype=np.float32))
            create_provider()
        msgs = _messages(caplog, DENOISE_LOGGER)
        assert not _starts_with(msgs, "denoise_frame_accepted:")
        assert not _starts_with(msgs, "denoise_overlap_buffered:")
        assert not _starts_with(msgs, "denoise_frame_fallback:")
        assert not _starts_with(msgs, "provider_created:")


# ===========================================================================
# history.py (extension events)
# ===========================================================================

HISTORY_LOGGER = "meetandread.playback.history"


class TestHistoryExtendedLogging:
    def _controller(self, tmp_path: Path):
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        return HistoryPlaybackController(recordings_dir=recordings), recordings

    def test_load_debug_internals(self, tmp_path: Path, caplog) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / "session-001.md"
        md.write_text("# T\n\nbody\n", encoding="utf-8")
        _create_valid_wav(recordings / "session-001.wav")
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.load_transcript_audio(md)
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        assert _starts_with(debugs, "load_resolve:")
        assert _starts_with(debugs, "load_source_set:")
        # New events never carry the stem, wav filename, or an id
        # derived from them.
        for m in _starts_with(debugs, "load_resolve:") + _starts_with(
            debugs, "load_source_set:"
        ):
            assert "session-001" not in m
            assert "wav=" not in m
            assert "id=" not in m
        # Existing INFO event vocabulary stays intact.
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == HISTORY_LOGGER and r.levelno == logging.INFO
        ]
        assert _starts_with(infos, "load_requested: stem=session-001")
        assert _starts_with(infos, "load_ready: stem=session-001")

    def test_load_missing_wav_debug_internal(self, tmp_path: Path, caplog) -> None:
        ctrl, _ = self._controller(tmp_path)
        md = tmp_path / "gone-001.md"
        md.write_text("# T\n", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.load_transcript_audio(md)
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        missing = _starts_with(debugs, "load_resolve_missing:")
        assert missing
        assert "id=" not in missing[0]
        assert "gone-001" not in missing[0]

    def test_seek_clamped_debug_event(self, tmp_path: Path, caplog) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / "session-001.md"
        md.write_text("# T\n", encoding="utf-8")
        _create_valid_wav(recordings / "session-001.wav")
        ctrl.load_transcript_audio(md)
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.seek_to(999999)
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        clamped = _starts_with(debugs, "seek_clamped:")
        assert clamped
        assert "clamped_ms=30000" in clamped[0]
        assert "session-001" not in clamped[0]

    def test_end_of_audio_debug_event(self, tmp_path: Path, caplog) -> None:
        ctrl, _ = self._controller(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl._on_media_status_changed(
                _MockQMediaPlayer.MediaStatus.EndOfMedia
            )
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        assert _starts_with(debugs, "media_end_of_audio:")

    def test_playback_state_changed_debug_event(
        self, tmp_path: Path, caplog
    ) -> None:
        ctrl, _ = self._controller(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl._on_playback_state_changed(
                _MockQMediaPlayer.PlaybackState.PlayingState
            )
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        assert _starts_with(debugs, "playback_state_changed:")

    def test_release_source_debug_event(self, tmp_path: Path, caplog) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / "session-001.md"
        md.write_text("# T\n", encoding="utf-8")
        _create_valid_wav(recordings / "session-001.wav")
        ctrl.load_transcript_audio(md)
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.release_source()
        debugs = _debug_messages(caplog, HISTORY_LOGGER)
        unset = _starts_with(debugs, "source_unset:")
        assert unset
        assert "session-001" not in unset[0]
        assert "id=" not in unset[0]

    def test_info_quietness_for_new_debug_events(
        self, tmp_path: Path, caplog
    ) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / "session-001.md"
        md.write_text("# T\n", encoding="utf-8")
        _create_valid_wav(recordings / "session-001.wav")
        with caplog.at_level(logging.INFO, logger=HISTORY_LOGGER):
            ctrl.load_transcript_audio(md)
            ctrl.seek_to(999999)
            ctrl._on_media_status_changed(
                _MockQMediaPlayer.MediaStatus.EndOfMedia
            )
            ctrl._on_playback_state_changed(
                _MockQMediaPlayer.PlaybackState.PlayingState
            )
            ctrl.release_source()
        msgs = _messages(caplog, HISTORY_LOGGER)
        assert not _starts_with(msgs, "load_resolve:")
        assert not _starts_with(msgs, "load_resolve_missing:")
        assert not _starts_with(msgs, "load_source_set:")
        assert not _starts_with(msgs, "seek_clamped:")
        assert not _starts_with(msgs, "media_end_of_audio:")
        assert not _starts_with(msgs, "playback_state_changed:")
        assert not _starts_with(msgs, "source_unset:")

    def test_new_events_privacy_no_absolute_paths(
        self, tmp_path: Path, caplog
    ) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / "session-001.md"
        md.write_text("# T\n", encoding="utf-8")
        _create_valid_wav(recordings / "session-001.wav")
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.load_transcript_audio(md)
            ctrl.seek_to(999999)
        new_msgs = [
            m
            for m in _debug_messages(caplog, HISTORY_LOGGER)
            if m.startswith(
                (
                    "load_resolve:",
                    "load_resolve_missing:",
                    "load_source_set:",
                    "seek_clamped:",
                    "media_end_of_audio:",
                    "playback_state_changed:",
                    "source_unset:",
                )
            )
        ]
        assert new_msgs
        for msg in new_msgs:
            assert str(tmp_path) not in msg
            assert "session-001" not in msg

    def test_canary_stem_never_logged(self, tmp_path: Path, caplog) -> None:
        ctrl, recordings = self._controller(tmp_path)
        md = tmp_path / f"{CANARY_STEM}.md"
        md.write_text("# T\n", encoding="utf-8")
        _create_valid_wav(recordings / f"{CANARY_STEM}.wav")
        with caplog.at_level(logging.DEBUG, logger=HISTORY_LOGGER):
            ctrl.load_transcript_audio(md)
            ctrl.seek_to(999999)
            ctrl._on_media_status_changed(
                _MockQMediaPlayer.MediaStatus.EndOfMedia
            )
            ctrl._on_playback_state_changed(
                _MockQMediaPlayer.PlaybackState.PlayingState
            )
            ctrl.release_source()
        checked = 0
        for rec in caplog.records:
            if rec.name != HISTORY_LOGGER:
                continue
            if rec.getMessage().startswith(PINNED_STEM_EVENT_PREFIXES):
                # Pinned INFO events (load_requested etc.) are a
                # separate, already-reviewed surface.
                continue
            checked += 1
            assert CANARY_STEM not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg)
        assert checked > 0


# ===========================================================================
# bookmark.py
# ===========================================================================

BOOKMARK_LOGGER = "meetandread.playback.bookmark"


class TestBookmarkLogging:
    def _manager(self, tmp_path: Path):
        path = _write_bookmark_transcript(tmp_path / "meet-001.md")
        return BookmarkManager(path)

    def test_add_debug_and_existing_info(self, tmp_path: Path, caplog) -> None:
        mgr = self._manager(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=BOOKMARK_LOGGER):
            mgr.add(position_ms=65000)
        debugs = _debug_messages(caplog, BOOKMARK_LOGGER)
        detail = _starts_with(debugs, "bookmark_add_detail:")
        assert detail
        assert "position_ms=65000" in detail[0]
        assert "id=" not in detail[0]
        assert "meet-001" not in detail[0]
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == BOOKMARK_LOGGER and r.levelno == logging.INFO
        ]
        added = _starts_with(infos, "bookmark_added:")
        assert added
        assert "stem=meet-001" in added[0]

    def test_add_detail_named_reflects_caller_supplied_name(
        self, tmp_path: Path, caplog
    ) -> None:
        mgr = self._manager(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=BOOKMARK_LOGGER):
            mgr.add(position_ms=1000, name="My Bookmark")
            mgr.add(position_ms=2000)
        details = _starts_with(
            _debug_messages(caplog, BOOKMARK_LOGGER), "bookmark_add_detail:"
        )
        assert len(details) == 2
        assert "named=True" in details[0]
        assert "named=False" in details[1]

    def test_delete_debug_and_existing_info(
        self, tmp_path: Path, caplog
    ) -> None:
        mgr = self._manager(tmp_path)
        bm = mgr.add(position_ms=1000)
        with caplog.at_level(logging.DEBUG, logger=BOOKMARK_LOGGER):
            mgr.delete(bm.created_at)
        debugs = _debug_messages(caplog, BOOKMARK_LOGGER)
        deleted = _starts_with(debugs, "bookmark_delete_detail:")
        assert deleted
        assert "id=" not in deleted[0]
        assert "meet-001" not in deleted[0]
        infos = [
            r.getMessage()
            for r in caplog.records
            if r.name == BOOKMARK_LOGGER and r.levelno == logging.INFO
        ]
        assert _starts_with(infos, "bookmark_deleted:")

    def test_write_failure_logs_warning(self, tmp_path: Path, caplog) -> None:
        mgr = self._manager(tmp_path)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        caplog.clear()
        from unittest.mock import patch as _patch

        with _patch.object(
            bookmark_mod, "_write_transcript", side_effect=boom
        ):
            with caplog.at_level(logging.WARNING, logger=BOOKMARK_LOGGER):
                with pytest.raises(OSError):
                    mgr.add(position_ms=1000)
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == BOOKMARK_LOGGER and r.levelno == logging.WARNING
        ]
        assert _starts_with(warnings, "bookmark_write_failed:")
        assert "stem=" not in warnings[0]
        assert "meet-001" not in warnings[0]

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        mgr = self._manager(tmp_path)
        bm = mgr.add(position_ms=1000)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=BOOKMARK_LOGGER):
            mgr.add(position_ms=2000)
            mgr.delete(bm.created_at)
        msgs = _messages(caplog, BOOKMARK_LOGGER)
        assert not _starts_with(msgs, "bookmark_add_detail:")
        assert not _starts_with(msgs, "bookmark_delete_detail:")

    def test_privacy_no_absolute_paths_or_names(
        self, tmp_path: Path, caplog
    ) -> None:
        mgr = self._manager(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=BOOKMARK_LOGGER):
            bm = mgr.add(position_ms=1000, name="SecretName")
            mgr.delete(bm.created_at)
        for msg in _messages(caplog, BOOKMARK_LOGGER):
            assert str(tmp_path) not in msg
            assert "SecretName" not in msg

    def test_canary_stem_never_logged(self, tmp_path: Path, caplog) -> None:
        path = _write_bookmark_transcript(tmp_path / f"{CANARY_STEM}.md")
        mgr = BookmarkManager(path)
        with caplog.at_level(logging.DEBUG, logger=BOOKMARK_LOGGER):
            bm = mgr.add(position_ms=1000, name="NamedByCaller")
            mgr.delete(bm.created_at)
        for rec in caplog.records:
            if rec.name != BOOKMARK_LOGGER:
                continue
            if rec.getMessage().startswith(PINNED_STEM_EVENT_PREFIXES):
                continue
            assert CANARY_STEM not in rec.getMessage()
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg)


# ===========================================================================
# Cross-lane canaries (fix round)
# ===========================================================================


class TestPrivacyCanaries:
    """End-to-end privacy canaries for the fix round.

    A distinctive recording stem and a distinctive custom-root basename
    flow through the full lane; assert no record from the lane's loggers
    — at ANY level — carries them. Only the pinned pre-existing event
    vocabulary (INFO stem lines, the recovery ERROR path) is exempt;
    the reviewer finding covers everything else.
    """

    def test_canary_stem_absent_across_lane(
        self, tmp_path: Path, caplog
    ) -> None:
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        _create_valid_wav(recordings / f"{CANARY_STEM}.wav")
        md = _write_bookmark_transcript(tmp_path / f"{CANARY_STEM}.md")

        ctrl = HistoryPlaybackController(recordings_dir=recordings)
        bookmark_mgr = BookmarkManager(md)
        storage = tmp_path / "storage"
        storage.mkdir()

        with caplog.at_level(logging.DEBUG, logger="meetandread"):
            ctrl.load_transcript_audio(md)
            ctrl.seek_to(999999)
            ctrl.release_source()
            bm = bookmark_mgr.add(position_ms=1000)
            bookmark_mgr.delete(bm.created_at)
            _drive_full_stem_lifecycle(storage, CANARY_STEM)

        checked = 0
        for rec in caplog.records:
            if rec.name not in LANE_LOGGERS:
                continue
            if rec.getMessage().startswith(PINNED_STEM_EVENT_PREFIXES):
                continue
            checked += 1
            msg = rec.getMessage()
            assert CANARY_STEM not in msg, msg
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg), (msg, arg)
        assert checked > 0

    def test_canary_custom_root_basename_absent(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        custom = tmp_path / CANARY_ROOT_NAME
        custom.mkdir()
        monkeypatch.setattr(
            storage_paths,
            "_resolve_custom_path",
            lambda field: custom if field == "recordings_path" else None,
        )
        with caplog.at_level(logging.DEBUG, logger="meetandread"):
            storage_paths.get_recordings_dir()
        checked = 0
        for rec in caplog.records:
            if rec.name not in LANE_LOGGERS:
                continue
            checked += 1
            msg = rec.getMessage()
            assert CANARY_ROOT_NAME not in msg, msg
            for arg in rec.args or ():
                assert CANARY_ROOT_NAME not in str(arg), (msg, arg)
        assert checked > 0

    def test_canary_stem_absent_on_write_failure(
        self, tmp_path: Path, caplog
    ) -> None:
        """Failure path: the write-failure WARNING carries no stem either."""
        from unittest.mock import patch as _patch

        path = _write_bookmark_transcript(tmp_path / f"{CANARY_STEM}.md")
        mgr = BookmarkManager(path)

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        with _patch.object(
            bookmark_mod, "_write_transcript", side_effect=_boom
        ):
            with caplog.at_level(logging.DEBUG, logger="meetandread"):
                with pytest.raises(OSError):
                    mgr.add(position_ms=1000)

        warnings = [
            r
            for r in caplog.records
            if r.name == BOOKMARK_LOGGER
            and r.levelno == logging.WARNING
        ]
        assert _starts_with(
            [r.getMessage() for r in warnings], "bookmark_write_failed:"
        )
        for rec in caplog.records:
            if rec.name not in LANE_LOGGERS:
                continue
            if rec.getMessage().startswith(PINNED_STEM_EVENT_PREFIXES):
                continue
            msg = rec.getMessage()
            assert CANARY_STEM not in msg, msg
            for arg in rec.args or ():
                assert CANARY_STEM not in str(arg), (msg, arg)
