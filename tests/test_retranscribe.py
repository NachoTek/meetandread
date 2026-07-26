"""Tests for the RetranscribeRunner background re-transcription module.

Covers sidecar naming, the re-transcribe lifecycle (start, progress, accept,
reject), cancellation, and speaker identification during re-transcription.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.config.models import AppSettings
from meetandread.transcription.engine import TranscriptionSuccess
from meetandread.transcription.retranscribe import RetranscribeRunner
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


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------

class TestModuleShape:
    def test_retranscribe_runner_exists(self):
        from meetandread.transcription import retranscribe
        assert hasattr(retranscribe, "RetranscribeRunner")

    def test_public_api_uses_retranscribe_names(self):
        """The runner exposes the canonical retranscribe_* public methods."""
        for name in ("retranscribe_recording", "accept_retranscribe",
                     "reject_retranscribe"):
            assert callable(getattr(RetranscribeRunner, name)), name

    def test_public_api_has_no_scrub_names(self):
        """No scrub-named public methods remain on the runner."""
        for name in ("scrub_recording", "accept_scrub", "reject_scrub"):
            assert not hasattr(RetranscribeRunner, name), name


# ---------------------------------------------------------------------------
# Sidecar naming — uses _retranscribe_ tag
# ---------------------------------------------------------------------------

class TestSidecarNaming:
    def test_basic_naming_uses_retranscribe_tag(self, transcript_path: Path):
        result = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert result == transcript_path.parent / (
            "recording-2026-04-26-120000_retranscribe_small.md"
        )

    def test_different_models_produce_different_paths(self, transcript_path: Path):
        s1 = RetranscribeRunner.sidecar_path(transcript_path, "tiny")
        s2 = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert s1 != s2
        assert "tiny" in s1.name
        assert "small" in s2.name

    def test_stem_derived_from_transcript(self, tmp_path: Path):
        tp = tmp_path / "my-session.md"
        tp.write_text("")
        result = RetranscribeRunner.sidecar_path(tp, "base")
        assert result.name == "my-session_retranscribe_base.md"

    def test_no_scrub_tag_in_retranscribe_path(self, transcript_path: Path):
        """The new naming must not accidentally use the legacy _scrub_ tag."""
        result = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert "_scrub_" not in result.name
        assert "_retranscribe_" in result.name


# ---------------------------------------------------------------------------
# Re-transcribe creates sidecar
# ---------------------------------------------------------------------------

class TestRetranscribeCreatesSidecar:
    def test_retranscribe_creates_sidecar(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        # Create a minimal WAV file (16-bit mono, 16000 Hz, ~0.1 s silence)
        import struct
        import wave
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
        mock_engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=[fake_segment])

        progress_vals = []
        completed = threading.Event()
        result_holder = {}

        def on_progress(pct):
            progress_vals.append(pct)

        def on_complete(path, error):
            result_holder["path"] = path
            result_holder["error"] = error
            completed.set()

        runner = RetranscribeRunner(settings, on_progress=on_progress, on_complete=on_complete)

        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            sidecar_str = runner.retranscribe_recording(audio_path, transcript_path, "tiny")

        # Wait for background thread
        assert completed.wait(timeout=5), "Re-transcribe did not complete in time"

        sidecar_path = Path(sidecar_str)
        assert sidecar_path.exists(), f"Sidecar not created at {sidecar_path}"
        assert "tiny" in sidecar_path.name
        assert "_retranscribe_" in sidecar_path.name
        assert result_holder.get("error") is None
        assert 100 in progress_vals

    def test_retranscribe_overwrites_existing_sidecar(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        import struct
        import wave
        with wave.open(str(audio_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<160h", *([0] * 160)))

        sidecar = RetranscribeRunner.sidecar_path(transcript_path, "base")
        sidecar.write_text("old content")

        fake_segment = MagicMock()
        fake_segment.text = "new"
        fake_segment.start = 0.0
        fake_segment.end = 1.0
        fake_segment.confidence = 95
        fake_segment.words = None

        mock_engine = MagicMock()
        mock_engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=[fake_segment])

        completed = threading.Event()

        runner = RetranscribeRunner(settings, on_complete=lambda p, e: completed.set())
        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            runner.retranscribe_recording(audio_path, transcript_path, "base")

        assert completed.wait(timeout=5)
        content = sidecar.read_text()
        assert "new" in content
        assert "old content" not in content


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestRetranscribeCancel:
    def test_cancel_stops_retranscribe(self, settings: AppSettings, tmp_path: Path):
        audio_path = tmp_path / "recording.wav"
        transcript_path = tmp_path / "recording.md"
        transcript_path.write_text("# Transcript\n")

        import struct
        import wave
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

        runner = RetranscribeRunner(settings, on_complete=on_complete)

        # Mock engine to be slow so cancel can fire
        def slow_transcribe(*args, **kwargs):
            import time
            time.sleep(10)
            return TranscriptionSuccess(segments=[])

        mock_engine = MagicMock()
        mock_engine.transcribe_chunk.side_effect = slow_transcribe

        with patch.object(runner, "_get_or_create_engine", return_value=mock_engine):
            runner.retranscribe_recording(audio_path, transcript_path, "tiny")
            # Cancel before transcription finishes
            runner.cancel()

        # Wait for the thread to finish (should stop quickly, not 10s)
        if runner._thread:
            runner._thread.join(timeout=3)

        assert runner.is_cancelled


# ---------------------------------------------------------------------------
# Accept / reject — sidecar path consistency
# ---------------------------------------------------------------------------

class TestAcceptRetranscribe:
    def test_accept_replaces_transcript(self, transcript_path: Path):
        sidecar = RetranscribeRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("# Re-transcribed\n\nBetter text\n")

        result = RetranscribeRunner.accept_retranscribe(transcript_path, "small")

        assert result == transcript_path
        assert transcript_path.read_text() == "# Re-transcribed\n\nBetter text\n"
        # Sidecar should be gone (moved, not copied)
        assert not sidecar.exists()

    def test_accept_missing_sidecar_raises(self, transcript_path: Path):
        with pytest.raises(FileNotFoundError, match="Sidecar not found"):
            RetranscribeRunner.accept_retranscribe(transcript_path, "small")


class TestRejectRetranscribe:
    def test_reject_deletes_sidecar(self, transcript_path: Path):
        sidecar = RetranscribeRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("unwanted")

        RetranscribeRunner.reject_retranscribe(transcript_path, "small")

        assert not sidecar.exists()

    def test_reject_idempotent(self, transcript_path: Path):
        # Sidecar doesn't exist — should not raise
        RetranscribeRunner.reject_retranscribe(transcript_path, "small")


# ---------------------------------------------------------------------------
# UI accept/reject workflow — history list refresh
# ---------------------------------------------------------------------------

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

        Simulates: populate list -> select item -> accept retranscribe ->
        list refreshes with updated word count from the new canonical transcript.
        """
        # Create canonical transcript (original, 2 words)
        md_path = tmp_path / "recording-2026-04-26-120000.md"
        md_path.write_text("# Transcript\n\nHello world\n")

        # Populate list with original word count (2)
        panel._populate_history_list([self._fake_meta(md_path, word_count=2)])
        assert panel._history_list.count() == 1

        # Select the item
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._current_history_md_path = md_path

        # Create sidecar with more words (5)
        sidecar = RetranscribeRunner.sidecar_path(md_path, "small")
        sidecar.write_text("# Re-transcribed\n\nOne two three four five\n")

        # Set retranscribe state
        panel._retranscribe.model_size = "small"
        panel._retranscribe.is_comparison_mode = True

        # Perform accept
        RetranscribeRunner.accept_retranscribe(md_path, "small")

        # Simulate the updated scan returning new word count
        with patch.object(panel, "_refresh_history") as mock_refresh:
            # Make _refresh_history actually repopulate with new word count
            def do_refresh():
                panel._populate_history_list(
                    [self._fake_meta(md_path, word_count=5)]
                )
            mock_refresh.side_effect = do_refresh

            panel._retranscribe._refresh_after_decision()

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
        # Create canonical transcript
        md_path = tmp_path / "recording-2026-04-26-120000.md"
        md_path.write_text("# Original\n\nOriginal content here\n")

        # Create sidecar
        sidecar = RetranscribeRunner.sidecar_path(md_path, "base")
        sidecar.write_text("# Re-transcribed\n\nRe-transcribed content\n")

        # Populate and select
        panel._populate_history_list([self._fake_meta(md_path, word_count=3)])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._current_history_md_path = md_path
        panel._retranscribe.model_size = "base"
        panel._retranscribe.is_comparison_mode = True

        # Reject
        RetranscribeRunner.reject_retranscribe(md_path, "base")

        # Refresh — list should still be present
        with patch.object(panel, "_refresh_history") as mock_refresh:
            def do_refresh():
                panel._populate_history_list(
                    [self._fake_meta(md_path, word_count=3)]
                )
            mock_refresh.side_effect = do_refresh

            panel._retranscribe._refresh_after_decision()

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


# ---------------------------------------------------------------------------
# Qt-safe retranscribe signal tests
# ---------------------------------------------------------------------------

class TestRetranscribeQtSafeSignals:
    """Verify retranscribe callbacks use PyQt signals instead of QTimer.singleShot.

    The flow now lives on RetranscribeController (shared with the settings
    panel, issue #33). Tests drive the controller through the panel:
    - _on_progress (background thread callback) emits _progress_sig
    - _progress_sig delivers to _on_progress_gui -> adapter.on_progress (button text)
    - _on_complete (background thread callback) emits _complete_sig
    - _complete_sig delivers to _on_complete_gui -> handle_complete
    """

    def test_progress_signal_updates_button_text(self, panel, qapp):
        """_on_retranscribe_progress emits signal that updates button text on GUI thread."""
        panel._retranscribe_btn.setEnabled(False)
        panel._retranscribe_btn.setText("Re-transcribing... 0%")

        # Simulate background thread calling the callback
        panel._retranscribe._on_progress(42)
        qapp.processEvents()

        assert panel._retranscribe_btn.text() == "Re-transcribing... 42%"

    def test_progress_signal_reaches_100(self, panel, qapp):
        """Progress signal delivers 100% correctly."""
        panel._retranscribe_btn.setEnabled(False)

        panel._retranscribe._on_progress(100)
        qapp.processEvents()

        assert panel._retranscribe_btn.text() == "Re-transcribing... 100%"

    def test_complete_signal_success_shows_comparison(self, panel, qapp, tmp_path):
        """_on_retranscribe_complete emits signal that triggers comparison view."""
        panel._retranscribe.is_retranscribing = True
        panel._retranscribe_btn.setEnabled(False)
        panel._retranscribe.model_size = "small"

        sidecar = tmp_path / "test_retranscribe_small.md"
        sidecar.write_text("**SPK_0**\nNew text.\n", encoding="utf-8")

        # Simulate background thread calling the completion callback
        panel._retranscribe._on_complete(str(sidecar), None)
        qapp.processEvents()

        assert panel._retranscribe.is_retranscribing is False
        assert panel._retranscribe_btn.isEnabled()
        assert panel._retranscribe.is_comparison_mode is True

    def test_complete_signal_error_reenables_button(self, panel, qapp):
        """_on_retranscribe_complete with error re-enables controls."""
        panel._retranscribe.is_retranscribing = True
        panel._retranscribe_btn.setEnabled(False)

        with patch("meetandread.widgets.floating_panels.QMessageBox.warning"):
            panel._retranscribe._on_complete("/fake/path.md", "Model load failed")
            qapp.processEvents()

        assert panel._retranscribe.is_retranscribing is False
        assert panel._retranscribe_btn.isEnabled()
        assert panel._retranscribe_btn.text() == "🔄 Re-transcribe"


class TestRetranscribeStartupFailure:
    """Verify RetranscribeRunner construction and startup failures are caught.

    Tests that exceptions during RetranscribeRunner() or retranscribe_recording()
    restore retranscribe state and show a warning dialog.
    """

    def test_runner_construction_failure_resets_state(self, panel, qapp, tmp_path):
        """If RetranscribeRunner() raises, retranscribe state is fully reset."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)
        md_path = tmp_path / "test.md"
        md_path.write_text("# Transcript\n")

        with patch(
            "meetandread.transcription.retranscribe.RetranscribeRunner",
            side_effect=RuntimeError("Whisper not available"),
        ), patch(
            "meetandread.widgets.floating_panels.QMessageBox.warning",
        ) as mock_warn:
            panel._retranscribe.start(wav_path, md_path, "tiny")
            qapp.processEvents()

        # State should be fully reset
        assert panel._retranscribe.is_retranscribing is False
        assert panel._retranscribe_btn.isEnabled()
        assert panel._retranscribe_btn.text() == "🔄 Re-transcribe"
        assert panel._retranscribe.is_comparison_mode is False
        assert panel._retranscribe.runner is None
        assert panel._retranscribe.sidecar_path is None
        # Warning shown
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args
        assert "Re-transcribe Failed" in call_args[0][1]

    def test_retranscribe_recording_startup_failure_resets_state(self, panel, qapp, tmp_path):
        """If retranscribe_recording() raises, retranscribe state is fully reset."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)
        md_path = tmp_path / "test.md"
        md_path.write_text("# Transcript\n")

        mock_runner = MagicMock()
        mock_runner.retranscribe_recording.side_effect = FileNotFoundError("Audio file vanished")

        with patch(
            "meetandread.transcription.retranscribe.RetranscribeRunner",
            return_value=mock_runner,
        ), patch(
            "meetandread.widgets.floating_panels.QMessageBox.warning",
        ) as mock_warn:
            panel._retranscribe.start(wav_path, md_path, "base")
            qapp.processEvents()

        assert panel._retranscribe.is_retranscribing is False
        assert panel._retranscribe_btn.isEnabled()
        assert panel._retranscribe_btn.text() == "🔄 Re-transcribe"
        mock_warn.assert_called_once()


# ---------------------------------------------------------------------------
# Speaker identification during re-transcribe (R025)
# ---------------------------------------------------------------------------

class TestRetranscribeSpeakerIdentification:
    """Tests for the _run_speaker_identification method in RetranscribeRunner."""

    def _make_store_with_words(self, num_words=10, duration=5.0) -> TranscriptStore:
        """Create a TranscriptStore with evenly-spaced words."""
        store = TranscriptStore()
        store.start_recording()
        words = []
        step = duration / num_words
        for i in range(num_words):
            words.append(Word(
                text=f"word{i}",
                start_time=i * step,
                end_time=(i + 1) * step,
                confidence=90,
                speaker_id=None,
            ))
        store.add_words(words)
        return store

    def test_speaker_id_graceful_when_diarizer_not_installed(self, settings):
        """If sherpa-onnx is not installed, the method returns store unchanged."""
        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words()

        with patch.dict("sys.modules", {"meetandread.speaker.diarizer": None}):
            result_store = runner._run_speaker_identification(
                Path("/fake/audio.wav"), store,
            )

        words = result_store.get_all_words()
        assert all(w.speaker_id is None for w in words)

    def test_speaker_id_skipped_when_disabled(self, settings):
        """When speaker diarization is disabled, return store unchanged."""
        settings.speaker.enabled = False
        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words()

        result_store = runner._run_speaker_identification(
            Path("/fake/audio.wav"), store,
        )
        words = result_store.get_all_words()
        assert all(w.speaker_id is None for w in words)

    def test_speaker_id_tags_words_from_diarization(self, settings):
        """Words within diarization segments get SPK_N speaker labels."""
        from meetandread.speaker.models import DiarizationResult, SpeakerSegment

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words(num_words=4, duration=4.0)

        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
                SpeakerSegment(start=2.0, end=4.0, speaker="spk1"),
            ],
            num_speakers=2,
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            with patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                with patch("meetandread.audio.storage.paths.get_recordings_dir", return_value=Path("/tmp")):
                    result_store = runner._run_speaker_identification(
                        Path("/fake/audio.wav"), store,
                    )

        words = result_store.get_all_words()
        assert words[0].speaker_id == "SPK_0"
        assert words[1].speaker_id == "SPK_0"
        assert words[2].speaker_id == "SPK_1"
        assert words[3].speaker_id == "SPK_1"

    def test_speaker_id_uses_matched_names(self, settings):
        """Known speakers from VoiceSignatureStore replace raw labels."""
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, SpeakerMatch, VoiceSignature,
        )

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words(num_words=2, duration=2.0)

        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(
                    embedding=np.random.rand(256).astype(np.float32),
                    speaker_label="spk0",
                    num_segments=1,
                ),
            },
            num_speakers=1,
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        mock_sig_store = MagicMock()
        mock_sig_store.__enter__ = MagicMock(return_value=mock_sig_store)
        mock_sig_store.__exit__ = MagicMock(return_value=False)
        mock_sig_store.find_match.return_value = SpeakerMatch(name="Alice", score=0.9, confidence="high")

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            with patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_sig_store):
                with patch("meetandread.audio.storage.paths.get_recordings_dir", return_value=Path("/tmp")):
                    result_store = runner._run_speaker_identification(
                        Path("/fake/audio.wav"), store,
                    )

        words = result_store.get_all_words()
        assert words[0].speaker_id == "Alice"
        assert words[1].speaker_id == "Alice"
        # Verify speaker_matches metadata was stored
        assert hasattr(result_store, "_speaker_matches")
        assert result_store._speaker_matches["spk0"]["identity_name"] == "Alice"

    def test_speaker_id_diarization_failure_returns_unchanged(self, settings):
        """If diarization fails, return store with no speaker labels."""
        from meetandread.speaker.models import DiarizationResult

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words()

        result = DiarizationResult(error="diarization failed")

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            result_store = runner._run_speaker_identification(
                Path("/fake/audio.wav"), store,
            )

        words = result_store.get_all_words()
        assert all(w.speaker_id is None for w in words)

    def test_speaker_id_no_match_keeps_spk_label(self, settings):
        """Without a known signature, words keep the SPK_N default label."""
        from meetandread.speaker.models import DiarizationResult, SpeakerSegment, VoiceSignature

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words(num_words=2, duration=2.0)

        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(
                    embedding=np.random.rand(256).astype(np.float32),
                    speaker_label="spk0",
                    num_segments=1,
                ),
            },
            num_speakers=1,
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        mock_sig_store = MagicMock()
        mock_sig_store.__enter__ = MagicMock(return_value=mock_sig_store)
        mock_sig_store.__exit__ = MagicMock(return_value=False)
        # No match found
        mock_sig_store.find_match.return_value = None

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            with patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_sig_store):
                with patch("meetandread.audio.storage.paths.get_recordings_dir", return_value=Path("/tmp")):
                    result_store = runner._run_speaker_identification(
                        Path("/fake/audio.wav"), store,
                    )

        words = result_store.get_all_words()
        assert words[0].speaker_id == "SPK_0"
        assert words[1].speaker_id == "SPK_0"
        # Verify raw profile was saved to signature store (same as controller)
        mock_sig_store.save_signature.assert_called_once()
        call_args = mock_sig_store.save_signature.call_args
        assert call_args[0][0] == "SPK_0"  # display label

    def test_speaker_id_carries_forward_original_identity(self, settings, tmp_path):
        """Re-transcribe carries forward speaker identity from original transcript."""
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words(num_words=2, duration=2.0)

        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(
                    embedding=np.random.rand(256).astype(np.float32),
                    speaker_label="spk0",
                    num_segments=1,
                ),
            },
            num_speakers=1,
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        mock_sig_store = MagicMock()
        mock_sig_store.__enter__ = MagicMock(return_value=mock_sig_store)
        mock_sig_store.__exit__ = MagicMock(return_value=False)
        # No voice match — but identity should carry from original
        mock_sig_store.find_match.return_value = None

        # Create a fake original transcript with speaker_matches
        original_md = tmp_path / "original.md"
        original_md.write_text(
            "# Transcript\n\n**SPK_0**\n\nHello world.\n"
            "\n---\n\n<!-- METADATA: "
            '{"speaker_matches": {"SPK_0": {"identity_name": "Commercial Guy", '
            '"score": 1.0, "confidence": "manual"}}} -->',
            encoding="utf-8",
        )

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            with patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_sig_store):
                with patch("meetandread.audio.storage.paths.get_recordings_dir", return_value=Path("/tmp")):
                    result_store = runner._run_speaker_identification(
                        Path("/fake/audio.wav"), store,
                        original_transcript_path=original_md,
                    )

        words = result_store.get_all_words()
        assert words[0].speaker_id == "Commercial Guy"
        assert words[1].speaker_id == "Commercial Guy"
        assert result_store._speaker_matches["spk0"]["identity_name"] == "Commercial Guy"
        assert result_store._speaker_matches["spk0"]["confidence"] == "carried"

    def test_speaker_id_carries_forward_with_integer_labels(self, settings, tmp_path):
        """Carry-forward works when diarizer returns integer speaker labels."""
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )

        runner = RetranscribeRunner(settings)
        store = self._make_store_with_words(num_words=2, duration=2.0)

        # sherpa-onnx returns integer speaker labels
        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker=0),
            ],
            signatures={
                0: VoiceSignature(
                    embedding=np.random.rand(256).astype(np.float32),
                    speaker_label=0,
                    num_segments=1,
                ),
            },
            num_speakers=1,
        )

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = result

        mock_sig_store = MagicMock()
        mock_sig_store.__enter__ = MagicMock(return_value=mock_sig_store)
        mock_sig_store.__exit__ = MagicMock(return_value=False)
        mock_sig_store.find_match.return_value = None

        # Original transcript has SPK_0 mapped to identity
        original_md = tmp_path / "original.md"
        original_md.write_text(
            "# Transcript\n\n**SPK_0**\n\nHello world.\n"
            "\n---\n\n<!-- METADATA: "
            '{"speaker_matches": {"SPK_0": {"identity_name": "Commercial Guy", '
            '"score": 1.0, "confidence": "manual"}}} -->',
            encoding="utf-8",
        )

        with patch("meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer):
            with patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_sig_store):
                with patch("meetandread.audio.storage.paths.get_recordings_dir", return_value=Path("/tmp")):
                    result_store = runner._run_speaker_identification(
                        Path("/fake/audio.wav"), store,
                        original_transcript_path=original_md,
                    )

        words = result_store.get_all_words()
        assert words[0].speaker_id == "Commercial Guy"
        assert words[1].speaker_id == "Commercial Guy"
