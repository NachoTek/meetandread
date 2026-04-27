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


# ---------------------------------------------------------------------------
# UI accept/reject workflow — history list refresh
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    """Provide a QApplication for QWidget tests (one per session)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    """Create a FloatingTranscriptPanel for testing."""
    from meetandread.widgets.floating_panels import FloatingTranscriptPanel

    p = FloatingTranscriptPanel()
    yield p
    p.close()


class TestAcceptRejectUI:
    """Test Accept/Reject workflow refreshes the history list and viewer."""

    @staticmethod
    def _fake_meta(path: Path, word_count: int, recording_time: str = "2026-04-26T12:00:00"):
        """Build a minimal RecordingMeta-like object for _populate_history_list."""
        from dataclasses import dataclass

        @dataclass
        class FakeMeta:
            path: Path
            recording_time: str
            word_count: int
            speaker_count: int
            speakers: list
            duration_seconds: float
            wav_exists: bool

        return FakeMeta(
            path=path,
            recording_time=recording_time,
            word_count=word_count,
            speaker_count=1,
            speakers=["SPK_0"],
            duration_seconds=30.0,
            wav_exists=True,
        )

    def test_accept_updates_history_list_word_count(
        self, panel, tmp_path: Path
    ) -> None:
        """After accept, history list should reflect the new word count.

        Simulates: populate list → select item → accept scrub → list refreshes
        with updated word count from the new canonical transcript.
        """
        from meetandread.transcription.scrub import ScrubRunner

        # Create canonical transcript (original, 2 words)
        md_path = tmp_path / "recording-2026-04-26-120000.md"
        md_path.write_text("# Transcript\n\nHello world\n")
        md_str = str(md_path)

        # Populate list with original word count (2)
        panel._populate_history_list([self._fake_meta(md_path, word_count=2)])
        assert panel._history_list.count() == 1

        # Select the item
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._current_history_md_path = md_path

        # Create sidecar with more words (5)
        sidecar = ScrubRunner.sidecar_path(md_path, "small")
        sidecar.write_text("# Scrubbed\n\nOne two three four five\n")

        # Set scrub state
        panel._scrub_model_size = "small"
        panel._is_comparison_mode = True

        # Perform accept
        ScrubRunner.accept_scrub(md_path, "small")

        # Simulate the updated scan returning new word count
        with patch.object(panel, "_refresh_history") as mock_refresh:
            # Make _refresh_history actually repopulate with new word count
            def do_refresh():
                panel._populate_history_list(
                    [self._fake_meta(md_path, word_count=5)]
                )
            mock_refresh.side_effect = do_refresh

            panel._refresh_after_scrub()

        # List should have been refreshed with new word count
        assert panel._history_list.count() == 1
        updated_item = panel._history_list.item(0)
        assert "5 words" in updated_item.text()
        # Item should be re-selected
        assert panel._history_list.currentItem() is updated_item

    def test_reject_restores_original_view(
        self, panel, tmp_path: Path
    ) -> None:
        """After reject, the viewer shows the original transcript and
        the history list is still populated."""
        from meetandread.transcription.scrub import ScrubRunner

        # Create canonical transcript
        md_path = tmp_path / "recording-2026-04-26-120000.md"
        md_path.write_text("# Original\n\nOriginal content here\n")

        # Create sidecar
        sidecar = ScrubRunner.sidecar_path(md_path, "base")
        sidecar.write_text("# Scrubbed\n\nScrubbed content\n")

        # Populate and select
        panel._populate_history_list([self._fake_meta(md_path, word_count=3)])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._current_history_md_path = md_path
        panel._scrub_model_size = "base"
        panel._is_comparison_mode = True

        # Reject
        ScrubRunner.reject_scrub(md_path, "base")

        # Refresh — list should still be present
        with patch.object(panel, "_refresh_history") as mock_refresh:
            def do_refresh():
                panel._populate_history_list(
                    [self._fake_meta(md_path, word_count=3)]
                )
            mock_refresh.side_effect = do_refresh

            panel._refresh_after_scrub()

        # Original transcript unchanged
        assert "Original content" in md_path.read_text()
        # Sidecar gone
        assert not sidecar.exists()
        # List still populated and item selected
        assert panel._history_list.count() == 1
        assert panel._history_list.currentItem() is not None

    def test_reselect_history_item_finds_matching(
        self, panel, tmp_path: Path
    ) -> None:
        """_reselect_history_item selects the item matching the given path."""
        from PyQt6.QtCore import Qt

        md_path = tmp_path / "rec-abc.md"
        md_path.write_text("test")

        panel._populate_history_list([self._fake_meta(md_path, word_count=10)])
        # Initially no selection
        assert panel._history_list.currentItem() is None

        panel._reselect_history_item(md_path)

        selected = panel._history_list.currentItem()
        assert selected is not None
        assert selected.data(Qt.ItemDataRole.UserRole) == str(md_path)

    def test_reselect_history_item_no_match(
        self, panel, tmp_path: Path
    ) -> None:
        """_reselect_history_item is a no-op when path doesn't match any item."""
        md_path = tmp_path / "nonexistent.md"

        panel._populate_history_list([self._fake_meta(tmp_path / "other.md", word_count=5)])
        panel._reselect_history_item(md_path)

        # No crash, no selection
        assert panel._history_list.currentItem() is None
