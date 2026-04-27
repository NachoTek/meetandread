"""Tests for the ScrubRunner background re-transcription module."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.config.models import AppSettings
from meetandread.transcription.scrub import ScrubRunner
from meetandread.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def transcript_path(tmp_path: Path) -> Path:
    """A fake canonical transcript file."""
    p = tmp_path / "recording-2026-04-26-120000.md"
    p.write_text("# Transcript\n\nHello world\n")
    return p


# ---------------------------------------------------------------------------
# Sidecar naming
# ---------------------------------------------------------------------------

class TestSidecarNaming:
    def test_basic_naming(self, transcript_path: Path):
        result = ScrubRunner.sidecar_path(transcript_path, "small")
        assert result == transcript_path.parent / "recording-2026-04-26-120000_scrub_small.md"

    def test_different_models(self, transcript_path: Path):
        s1 = ScrubRunner.sidecar_path(transcript_path, "tiny")
        s2 = ScrubRunner.sidecar_path(transcript_path, "small")
        assert s1 != s2
        assert "tiny" in s1.name
        assert "small" in s2.name

    def test_stem_derived_from_transcript(self, tmp_path: Path):
        tp = tmp_path / "my-session.md"
        tp.write_text("")
        result = ScrubRunner.sidecar_path(tp, "base")
        assert result.name == "my-session_scrub_base.md"


# ---------------------------------------------------------------------------
# Scrub creates sidecar
# ---------------------------------------------------------------------------

class TestScrubCreatesSidecar:
    def test_scrub_creates_sidecar(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        # Create a minimal WAV file (16-bit mono, 16000 Hz, ~0.1 s silence)
        import struct, wave
        with wave.open(str(audio_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<160h", *([0] * 160)))

        # Mock the engine to avoid loading a real Whisper model
        fake_segment = MagicMock()
        fake_segment.text = "hello world"
        fake_segment.start = 0.0
        fake_segment.end = 1.0
        fake_segment.confidence = 90
        fake_segment.words = None

        mock_engine = MagicMock()
        mock_engine.transcribe_chunk.return_value = [fake_segment]

        progress_vals = []
        completed = threading.Event()
        result_holder = {}

        def on_progress(pct):
            progress_vals.append(pct)

        def on_complete(path, error):
            result_holder["path"] = path
            result_holder["error"] = error
            completed.set()

        runner = ScrubRunner(settings, on_progress=on_progress, on_complete=on_complete)

        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            sidecar_str = runner.scrub_recording(audio_path, transcript_path, "tiny")

        # Wait for background thread
        assert completed.wait(timeout=5), "Scrub did not complete in time"

        sidecar_path = Path(sidecar_str)
        assert sidecar_path.exists(), f"Sidecar not created at {sidecar_path}"
        assert "tiny" in sidecar_path.name
        assert result_holder.get("error") is None
        assert 100 in progress_vals

    def test_scrub_overwrites_existing_sidecar(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        import struct, wave
        with wave.open(str(audio_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<160h", *([0] * 160)))

        sidecar = ScrubRunner.sidecar_path(transcript_path, "base")
        sidecar.write_text("old content")

        fake_segment = MagicMock()
        fake_segment.text = "new"
        fake_segment.start = 0.0
        fake_segment.end = 1.0
        fake_segment.confidence = 95
        fake_segment.words = None

        mock_engine = MagicMock()
        mock_engine.transcribe_chunk.return_value = [fake_segment]

        completed = threading.Event()

        runner = ScrubRunner(settings, on_complete=lambda p, e: completed.set())
        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            runner.scrub_recording(audio_path, transcript_path, "base")

        assert completed.wait(timeout=5)
        content = sidecar.read_text()
        assert "new" in content
        assert "old content" not in content


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestScrubCancel:
    def test_cancel_stops_scrub(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        import struct, wave
        with wave.open(str(audio_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<160h", *([0] * 160)))

        completed = threading.Event()
        result_holder = {}

        def on_complete(path, error):
            result_holder["error"] = error
            completed.set()

        runner = ScrubRunner(settings, on_complete=on_complete)

        # Mock engine to be slow so cancel can fire
        def slow_transcribe(*args, **kwargs):
            import time
            time.sleep(10)
            return []

        mock_engine = MagicMock()
        mock_engine.transcribe_chunk.side_effect = slow_transcribe

        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            runner.scrub_recording(audio_path, transcript_path, "tiny")
            # Cancel before transcription finishes
            runner.cancel()

        # Wait for the thread to finish (should stop quickly, not 10s)
        if runner._thread:
            runner._thread.join(timeout=3)

        assert runner.is_cancelled


# ---------------------------------------------------------------------------
# Accept replaces transcript
# ---------------------------------------------------------------------------

class TestAcceptScrub:
    def test_accept_replaces_transcript(self, transcript_path: Path):
        sidecar = ScrubRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("# Scrubbed Transcript\n\nBetter text\n")

        result = ScrubRunner.accept_scrub(transcript_path, "small")

        assert result == transcript_path
        assert transcript_path.read_text() == "# Scrubbed Transcript\n\nBetter text\n"
        # Sidecar should be gone (moved, not copied)
        assert not sidecar.exists()

    def test_accept_missing_sidecar_raises(self, transcript_path: Path):
        with pytest.raises(FileNotFoundError, match="Sidecar not found"):
            ScrubRunner.accept_scrub(transcript_path, "small")


# ---------------------------------------------------------------------------
# Reject deletes sidecar
# ---------------------------------------------------------------------------

class TestRejectScrub:
    def test_reject_deletes_sidecar(self, transcript_path: Path):
        sidecar = ScrubRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("unwanted")

        ScrubRunner.reject_scrub(transcript_path, "small")

        assert not sidecar.exists()

    def test_reject_idempotent(self, transcript_path: Path):
        # Sidecar doesn't exist — should not raise
        ScrubRunner.reject_scrub(transcript_path, "small")
