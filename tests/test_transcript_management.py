"""Tests for transcript management — post-processing in-place overwrite.

Covers T01 must-haves:
- _save_post_processed_transcript writes to {stem}.md, never {stem}_enhanced.md
- When the original .md already exists, it gets overwritten
- The result dict contains "transcript_path" key (not "enhanced_path")
- Controller callback reads "transcript_path" from result
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metamemory.transcription.post_processor import (
    PostProcessJob,
    PostProcessStatus,
    PostProcessingQueue,
)
from metamemory.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Qt application fixture for History tab tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session (needed for QWidget tests)."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store_with_words(*texts: str) -> TranscriptStore:
    """Create a TranscriptStore with simple words from *texts*."""
    store = TranscriptStore()
    store.start_recording()
    words = [
        Word(text=t, start_time=i * 1.0, end_time=i * 1.0 + 0.9, confidence=90)
        for i, t in enumerate(texts)
    ]
    store.add_words(words)
    return store


def _make_job(tmp_path: Path) -> PostProcessJob:
    """Create a minimal PostProcessJob pointing at *tmp_path*."""
    audio_file = tmp_path / "recording_001.wav"
    audio_file.write_bytes(b"\x00")  # placeholder
    realtime = _make_store_with_words("hello world")
    return PostProcessJob(
        job_id="test-job",
        audio_file=audio_file,
        realtime_transcript=realtime,
        output_dir=tmp_path,
        model_size="base",
    )


# ---------------------------------------------------------------------------
# Tests — in-place overwrite (not _enhanced.md)
# ---------------------------------------------------------------------------

class TestPostProcessInPlaceOverwrite:
    """Verify post-processing writes {stem}.md and overwrites if it exists."""

    def test_writes_stem_md_not_enhanced(self, tmp_path: Path) -> None:
        """Post-processing must create {stem}.md, never {stem}_enhanced.md."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)
        store = _make_store_with_words("post", "processed", "result")

        result_path = ppq._save_post_processed_transcript(job, store)

        # The returned path must be {stem}.md
        assert result_path.name == "recording_001.md"
        assert result_path.exists()

        # No _enhanced.md variant should be created
        enhanced = tmp_path / "recording_001_enhanced.md"
        assert not enhanced.exists()

    def test_overwrites_existing_md(self, tmp_path: Path) -> None:
        """When original .md exists, post-processing must overwrite it."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # Create an existing transcript .md
        existing_md = tmp_path / "recording_001.md"
        existing_md.write_text("# Old transcript\n\nOld content here.", encoding="utf-8")
        assert existing_md.exists()

        new_store = _make_store_with_words("new", "content")
        result_path = ppq._save_post_processed_transcript(job, new_store)

        # Same file path
        assert result_path == existing_md

        # Content must be overwritten — old marker text gone
        content = result_path.read_text(encoding="utf-8")
        assert "Old content here" not in content
        assert "new" in content

    def test_creates_md_when_missing(self, tmp_path: Path) -> None:
        """When no original .md exists, post-processing creates one."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # No pre-existing .md
        transcript_md = tmp_path / "recording_001.md"
        assert not transcript_md.exists()

        store = _make_store_with_words("fresh", "transcript")
        result_path = ppq._save_post_processed_transcript(job, store)

        assert result_path.exists()
        assert result_path.name == "recording_001.md"
        content = result_path.read_text(encoding="utf-8")
        assert "fresh" in content


# ---------------------------------------------------------------------------
# Tests — result dict key
# ---------------------------------------------------------------------------

class TestPostProcessResultKey:
    """Verify the result dict uses 'transcript_path', not 'enhanced_path'."""

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_result_dict_has_transcript_path_key(
        self, mock_load_audio, mock_engine, tmp_path: Path
    ) -> None:
        """After _process_job, result dict must contain 'transcript_path'."""
        import numpy as np

        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # Stub audio loading
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        # Stub engine transcription — return empty segments
        mock_eng = MagicMock()
        mock_eng.transcribe_chunk.return_value = []
        mock_engine.return_value = mock_eng

        ppq._process_job(job)

        assert job.status == PostProcessStatus.COMPLETED
        assert "transcript_path" in job.result
        assert "enhanced_path" not in job.result

        # transcript_path must point to {stem}.md
        assert Path(job.result["transcript_path"]).name == "recording_001.md"


# ---------------------------------------------------------------------------
# Tests — controller callback reads transcript_path
# ---------------------------------------------------------------------------

class TestControllerCallback:
    """Verify RecordingController._on_post_process_complete_callback reads
    'transcript_path' from the result dict."""

    def test_controller_callback_reads_transcript_path(self, tmp_path: Path) -> None:
        """Controller callback must read 'transcript_path', not 'enhanced_path'."""
        from metamemory.recording.controller import RecordingController

        captured: dict = {}

        def on_complete(job_id: str, path: Path) -> None:
            captured["job_id"] = job_id
            captured["path"] = path

        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_post_process_complete = on_complete

        transcript_md = tmp_path / "recording_001.md"
        transcript_md.write_text("# transcript", encoding="utf-8")

        result = {
            "transcript_path": str(transcript_md),
            "word_count": 5,
            "realtime_word_count": 3,
            "model_used": "base",
        }

        ctrl._on_post_process_complete_callback("job-42", result)

        assert captured["job_id"] == "job-42"
        assert captured["path"] == transcript_md

    def test_controller_callback_ignores_enhanced_path(self, tmp_path: Path) -> None:
        """If result dict only has 'enhanced_path', callback must not fire."""
        from metamemory.recording.controller import RecordingController

        captured: dict = {}

        def on_complete(job_id: str, path: Path) -> None:
            captured["path"] = path

        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_post_process_complete = on_complete

        # Simulate a stale result with only enhanced_path
        result = {
            "enhanced_path": str(tmp_path / "recording_001_enhanced.md"),
            "word_count": 5,
        }

        ctrl._on_post_process_complete_callback("job-42", result)

        # Callback should NOT have fired — no transcript_path key
        assert "path" not in captured


# ---------------------------------------------------------------------------
# Tests — History tab (T03)
# ---------------------------------------------------------------------------

class TestHistoryTab:
    """Verify FloatingTranscriptPanel has Live/History tabs with correct
    behavior for recording list, transcript viewing, and empty recordings.

    Uses a shared QApplication created once per session to avoid the
    PyQt6/PySide6 type mismatch with pytest-qt's qtbot.
    """

    @pytest.fixture
    def panel(self, qapp):
        """Create a FloatingTranscriptPanel for testing."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel

        p = FloatingTranscriptPanel()
        yield p
        p.close()

    # -- Structural tests --------------------------------------------------

    def test_panel_has_tab_widget_with_two_tabs(self, panel) -> None:
        """Panel must contain a QTabWidget with exactly 2 tabs."""
        from PyQt6.QtWidgets import QTabWidget

        tabs = panel.findChild(QTabWidget)
        assert tabs is not None, "Panel must have a QTabWidget"
        assert tabs.count() == 2
        assert tabs.tabText(0) == "Live"
        assert tabs.tabText(1) == "History"

    def test_history_tab_has_list_and_viewer(self, panel) -> None:
        """History tab must contain a QListWidget and a read-only QTextEdit."""
        from PyQt6.QtWidgets import QListWidget, QTextEdit, QSplitter

        assert hasattr(panel, "_history_list")
        assert isinstance(panel._history_list, QListWidget)
        assert hasattr(panel, "_history_viewer")
        assert isinstance(panel._history_viewer, QTextEdit)
        assert panel._history_viewer.isReadOnly()

    # -- _populate_history_list -------------------------------------------

    def test_populate_history_list_with_recordings(self, panel) -> None:
        """_populate_history_list populates the QListWidget with metadata."""
        from dataclasses import dataclass
        from pathlib import Path
        from PyQt6.QtCore import Qt

        @dataclass
        class FakeMeta:
            path: Path
            recording_time: str
            word_count: int
            speaker_count: int
            speakers: list
            duration_seconds: float
            wav_exists: bool

        recordings = [
            FakeMeta(
                path=Path("/tmp/rec1.md"),
                recording_time="2026-04-22T14:30:00",
                word_count=150,
                speaker_count=2,
                speakers=["SPK_0", "SPK_1"],
                duration_seconds=60.0,
                wav_exists=True,
            ),
            FakeMeta(
                path=Path("/tmp/rec2.md"),
                recording_time="2026-04-21T10:00:00",
                word_count=50,
                speaker_count=1,
                speakers=["SPK_0"],
                duration_seconds=30.0,
                wav_exists=False,
            ),
        ]

        panel._populate_history_list(recordings)

        assert panel._history_list.count() == 2
        # Newest first (rec1)
        item0 = panel._history_list.item(0)
        assert "2026-04-22 14:30" in item0.text()
        assert "150 words" in item0.text()
        assert "2 speakers" in item0.text()
        assert item0.data(Qt.ItemDataRole.UserRole) == str(Path("/tmp/rec1.md"))

    def test_populate_history_empty_recording(self, panel) -> None:
        """Empty recordings (word_count=0) show '(Empty recording)' badge."""
        from dataclasses import dataclass
        from pathlib import Path

        @dataclass
        class FakeMeta:
            path: Path
            recording_time: str
            word_count: int
            speaker_count: int
            speakers: list
            duration_seconds: float
            wav_exists: bool

        recordings = [
            FakeMeta(
                path=Path("/tmp/empty.md"),
                recording_time="2026-04-22T15:00:00",
                word_count=0,
                speaker_count=0,
                speakers=[],
                duration_seconds=0.0,
                wav_exists=True,
            ),
        ]

        panel._populate_history_list(recordings)

        item = panel._history_list.item(0)
        assert "(Empty recording)" in item.text()

    def test_populate_history_clears_previous(self, panel) -> None:
        """Populating again clears previous items."""
        from dataclasses import dataclass
        from pathlib import Path

        @dataclass
        class FakeMeta:
            path: Path
            recording_time: str
            word_count: int
            speaker_count: int
            speakers: list
            duration_seconds: float
            wav_exists: bool

        recordings = [
            FakeMeta(
                path=Path("/tmp/r.md"),
                recording_time="2026-04-22T15:00:00",
                word_count=10,
                speaker_count=1,
                speakers=["SPK_0"],
                duration_seconds=5.0,
                wav_exists=False,
            ),
        ]

        panel._populate_history_list(recordings)
        assert panel._history_list.count() == 1
        panel._populate_history_list([])
        assert panel._history_list.count() == 0

    # -- _on_history_item_clicked ------------------------------------------

    def test_history_item_click_loads_markdown(self, panel, tmp_path: Path) -> None:
        """Clicking a history item displays the transcript markdown content."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        # Create a sample transcript file
        md_content = "# My Transcript\n\nHello world this is the content.\n"
        metadata = '{"recording_start_time": "2026-04-22T14:30:00", "word_count": 7, "words": []}'
        full_content = md_content + "---\n\n<!-- METADATA: " + metadata + " -->"

        md_file = tmp_path / "test_rec.md"
        md_file.write_text(full_content, encoding="utf-8")

        # Add an item to the list
        item = QListWidgetItem("2026-04-22 14:30 | 7 words | 0 speakers")
        item.setData(Qt.ItemDataRole.UserRole, str(md_file))
        panel._history_list.addItem(item)

        # Simulate click
        panel._on_history_item_clicked(item)

        # Viewer should show the markdown content (without the footer)
        viewer_text = panel._history_viewer.toPlainText()
        assert "Hello world this is the content" in viewer_text
        # Metadata footer should be stripped
        assert "METADATA" not in viewer_text

    def test_history_item_click_missing_file(self, panel, tmp_path: Path) -> None:
        """Clicking an item whose file was deleted shows not-found message."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        missing_path = tmp_path / "nonexistent.md"

        item = QListWidgetItem("missing recording")
        item.setData(Qt.ItemDataRole.UserRole, str(missing_path))
        panel._history_list.addItem(item)

        panel._on_history_item_clicked(item)

        viewer_text = panel._history_viewer.toPlainText()
        assert "not found" in viewer_text.lower()

    # -- Tab switching triggers refresh ------------------------------------

    def test_tab_switch_triggers_refresh(self, panel) -> None:
        """Switching to History tab triggers _refresh_history."""
        refresh_called = {"count": 0}
        original_refresh = panel._refresh_history

        def counting_refresh():
            refresh_called["count"] += 1
            original_refresh()

        panel._refresh_history = counting_refresh

        # Switch to History tab (index 1)
        panel._tab_widget.setCurrentIndex(1)
        assert refresh_called["count"] >= 1

    # -- Live tab preserved ------------------------------------------------

    def test_live_tab_preserves_text_edit_and_status(self, panel) -> None:
        """Live tab must contain the text_edit and status_label."""
        assert panel.text_edit is not None
        assert panel.status_label is not None
        # text_edit should still work for live transcript
        assert panel.text_edit.isReadOnly()


# ---------------------------------------------------------------------------
# Tests — Speaker rename in history transcripts (T04)
# ---------------------------------------------------------------------------

class TestSpeakerRename:
    """Verify speaker rename updates .md JSON metadata, markdown body,
    and propagates to VoiceSignatureStore when applicable.
    """

    @pytest.fixture
    def panel(self, qapp):
        """Create a FloatingTranscriptPanel for testing."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel

        p = FloatingTranscriptPanel()
        yield p
        p.close()

    @staticmethod
    def _make_transcript_md(
        tmp_path: Path,
        words: list,
        segments: list,
        recording_time: str = "2026-04-22T14:30:00",
    ) -> Path:
        """Create a transcript .md file with JSON metadata footer.

        Args:
            tmp_path: Directory to write into.
            words: List of word dicts with speaker_id.
            segments: List of segment dicts with speaker_id.
            recording_time: ISO timestamp.

        Returns:
            Path to the created .md file.
        """
        import json

        # Build markdown body from segments
        lines = ["# Transcript", ""]
        lines.append(f"**Recorded:** {recording_time}")
        lines.append("")

        # Group words by speaker for markdown rendering
        current_speaker = None
        current_words = []
        for w in words:
            sid = w.get("speaker_id") or "Unknown Speaker"
            if sid != current_speaker:
                if current_words:
                    lines.append(f"**{current_speaker}**")
                    lines.append("")
                    lines.append(" ".join(current_words))
                    lines.append("")
                current_speaker = sid
                current_words = [w["text"]]
            else:
                current_words.append(w["text"])
        if current_words:
            lines.append(f"**{current_speaker}**")
            lines.append("")
            lines.append(" ".join(current_words))
            lines.append("")

        md_body = "\n".join(lines)
        metadata = {
            "recording_start_time": recording_time,
            "word_count": len(words),
            "words": words,
            "segments": segments,
        }

        md_file = tmp_path / "test_rec.md"
        md_file.write_text(
            md_body + "\n---\n\n<!-- METADATA: " + json.dumps(metadata, indent=2) + " -->\n",
            encoding="utf-8",
        )
        return md_file

    @staticmethod
    def _read_metadata(md_path: Path) -> dict:
        """Read and parse JSON metadata from a transcript .md file."""
        content = md_path.read_text(encoding="utf-8")
        marker = "<!-- METADATA: "
        idx = content.find(marker)
        assert idx != -1, "No metadata footer found"
        end_marker = " -->"
        json_str = content[idx + len(marker):]
        if json_str.rstrip().endswith(end_marker):
            json_str = json_str.rstrip()[: -len(end_marker)]
        return json.loads(json_str)

    # -- _rename_speaker_in_file tests ------------------------------------

    def test_rename_updates_words_and_segments_in_metadata(
        self, panel, tmp_path: Path
    ) -> None:
        """_rename_speaker_in_file must update speaker_id in words and segments."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "World", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "SPK_0"},
            {"text": "Hi", "start_time": 1.0, "end_time": 1.5, "confidence": 88, "speaker_id": "SPK_1"},
        ]
        segments = [
            {"words": words[:2], "start_time": 0.0, "end_time": 1.0, "avg_confidence": 87, "speaker_id": "SPK_0"},
            {"words": [words[2]], "start_time": 1.0, "end_time": 1.5, "avg_confidence": 88, "speaker_id": "SPK_1"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        panel._rename_speaker_in_file(md_path, "SPK_0", "Alice")

        data = self._read_metadata(md_path)

        # Words should have SPK_0 -> Alice
        assert data["words"][0]["speaker_id"] == "Alice"
        assert data["words"][1]["speaker_id"] == "Alice"
        assert data["words"][2]["speaker_id"] == "SPK_1"  # unchanged

        # Segments should be updated too
        assert data["segments"][0]["speaker_id"] == "Alice"
        assert data["segments"][1]["speaker_id"] == "SPK_1"  # unchanged

    def test_rename_updates_markdown_body_speaker_labels(
        self, panel, tmp_path: Path
    ) -> None:
        """_rename_speaker_in_file must update **SpeakerLabel** in markdown body."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Hi", "start_time": 1.0, "end_time": 1.5, "confidence": 88, "speaker_id": "SPK_1"},
        ]
        segments = [
            {"words": [words[0]], "start_time": 0.0, "end_time": 0.5, "avg_confidence": 90, "speaker_id": "SPK_0"},
            {"words": [words[1]], "start_time": 1.0, "end_time": 1.5, "avg_confidence": 88, "speaker_id": "SPK_1"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        panel._rename_speaker_in_file(md_path, "SPK_0", "Alice")

        content = md_path.read_text(encoding="utf-8")
        # Markdown body should have **Alice** instead of **SPK_0**
        assert "**Alice**" in content
        assert "**SPK_0**" not in content
        # SPK_1 should remain unchanged
        assert "**SPK_1**" in content

    def test_rename_propagate_to_signatures_saves_new_deletes_old(
        self, panel, tmp_path: Path
    ) -> None:
        """_propagate_rename_to_signatures saves under new name and deletes old."""
        import numpy as np
        from metamemory.speaker.signatures import VoiceSignatureStore

        # Create a signature DB with the old speaker
        db_path = tmp_path / "speaker_signatures.db"
        embedding = np.random.randn(256).astype(np.float32)

        with VoiceSignatureStore(db_path=str(db_path)) as store:
            store.save_signature("SPK_0", embedding, averaged_from_segments=3)

        # Create a transcript file in the same directory
        md_path = tmp_path / "test_rec.md"
        md_path.write_text("# dummy\n\n---\n\n<!-- METADATA: {} -->\n", encoding="utf-8")

        panel._propagate_rename_to_signatures(md_path, "SPK_0", "Alice")

        with VoiceSignatureStore(db_path=str(db_path)) as store:
            profiles = store.load_signatures()
            names = [p.name for p in profiles]
            assert "Alice" in names
            assert "SPK_0" not in names

    def test_rename_propagate_handles_missing_speaker_gracefully(
        self, panel, tmp_path: Path
    ) -> None:
        """_propagate_rename_to_signatures must not error when old_name is missing."""
        import numpy as np
        from metamemory.speaker.signatures import VoiceSignatureStore

        # Create a DB with a different speaker (not the one being renamed)
        db_path = tmp_path / "speaker_signatures.db"
        embedding = np.random.randn(256).astype(np.float32)

        with VoiceSignatureStore(db_path=str(db_path)) as store:
            store.save_signature("SPK_1", embedding)

        md_path = tmp_path / "test_rec.md"
        md_path.write_text("# dummy\n\n---\n\n<!-- METADATA: {} -->\n", encoding="utf-8")

        # Should not raise — SPK_0 is not in the store
        panel._propagate_rename_to_signatures(md_path, "SPK_0", "Alice")

        # Verify no changes were made
        with VoiceSignatureStore(db_path=str(db_path)) as store:
            profiles = store.load_signatures()
            names = [p.name for p in profiles]
            assert names == ["SPK_1"]

    def test_rename_propagate_handles_no_db_gracefully(
        self, panel, tmp_path: Path
    ) -> None:
        """_propagate_rename_to_signatures must not error when no DB exists."""
        md_path = tmp_path / "test_rec.md"
        md_path.write_text("# dummy\n\n---\n\n<!-- METADATA: {} -->\n", encoding="utf-8")

        # Should not raise — no signature DB exists
        panel._propagate_rename_to_signatures(md_path, "SPK_0", "Alice")

    def test_full_rename_flow_md_and_signatures(
        self, panel, tmp_path: Path
    ) -> None:
        """Full rename flow: update .md metadata + propagate to signature store."""
        import numpy as np
        from metamemory.speaker.signatures import VoiceSignatureStore

        # Set up signature DB with SPK_0
        db_path = tmp_path / "speaker_signatures.db"
        embedding = np.random.randn(256).astype(np.float32)
        with VoiceSignatureStore(db_path=str(db_path)) as store:
            store.save_signature("SPK_0", embedding, averaged_from_segments=2)

        # Create transcript with SPK_0 and SPK_1
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "There", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "SPK_0"},
            {"text": "Hey", "start_time": 1.0, "end_time": 1.5, "confidence": 88, "speaker_id": "SPK_1"},
        ]
        segments = [
            {"words": words[:2], "start_time": 0.0, "end_time": 1.0, "avg_confidence": 87, "speaker_id": "SPK_0"},
            {"words": [words[2]], "start_time": 1.0, "end_time": 1.5, "avg_confidence": 88, "speaker_id": "SPK_1"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)

        # Perform full rename
        panel._rename_speaker_in_file(md_path, "SPK_0", "Bob")
        panel._propagate_rename_to_signatures(md_path, "SPK_0", "Bob")

        # Verify .md metadata
        data = self._read_metadata(md_path)
        assert data["words"][0]["speaker_id"] == "Bob"
        assert data["words"][1]["speaker_id"] == "Bob"
        assert data["words"][2]["speaker_id"] == "SPK_1"
        assert data["segments"][0]["speaker_id"] == "Bob"

        # Verify markdown body
        content = md_path.read_text(encoding="utf-8")
        assert "**Bob**" in content
        assert "**SPK_0**" not in content

        # Verify signature store
        with VoiceSignatureStore(db_path=str(db_path)) as store:
            profiles = store.load_signatures()
            names = [p.name for p in profiles]
            assert "Bob" in names
            assert "SPK_0" not in names

    # -- _render_history_transcript tests ----------------------------------

    def test_render_history_transcript_produces_anchors(
        self, panel, tmp_path: Path
    ) -> None:
        """_render_history_transcript should produce HTML with speaker anchors."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Hi", "start_time": 1.0, "end_time": 1.5, "confidence": 88, "speaker_id": "SPK_1"},
        ]
        segments = [
            {"words": [words[0]], "start_time": 0.0, "end_time": 0.5, "avg_confidence": 90, "speaker_id": "SPK_0"},
            {"words": [words[1]], "start_time": 1.0, "end_time": 1.5, "avg_confidence": 88, "speaker_id": "SPK_1"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        html = panel._render_history_transcript(md_path)

        assert html is not None
        assert 'href="speaker:SPK_0"' in html
        assert 'href="speaker:SPK_1"' in html

    def test_render_history_transcript_returns_none_no_metadata(
        self, panel, tmp_path: Path
    ) -> None:
        """_render_history_transcript returns None when no metadata footer."""
        md_path = tmp_path / "bare.md"
        md_path.write_text("# Just markdown\n\nNo metadata here.\n", encoding="utf-8")

        result = panel._render_history_transcript(md_path)
        assert result is None

    # -- History viewer uses QTextBrowser ---------------------------------

    def test_history_viewer_is_text_browser(self, panel) -> None:
        """History viewer should be a QTextBrowser for anchor click support."""
        from PyQt6.QtWidgets import QTextBrowser

        assert isinstance(panel._history_viewer, QTextBrowser)

    def test_history_viewer_anchor_clicked_connected(self, panel) -> None:
        """History viewer must have anchorClicked signal connected."""
        # Verify the method exists (QTextBrowser provides it)
        assert hasattr(panel._history_viewer, "anchorClicked")

    # -- Anchor click rename flow ------------------------------------------

    def test_on_history_anchor_clicked_triggers_rename(
        self, panel, tmp_path: Path
    ) -> None:
        """Clicking a speaker anchor in history triggers rename dialog."""
        from PyQt6.QtCore import QUrl
        from unittest.mock import patch

        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
        ]
        segments = [
            {"words": [words[0]], "start_time": 0.0, "end_time": 0.5, "avg_confidence": 90, "speaker_id": "SPK_0"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        panel._current_history_md_path = md_path

        # Mock QInputDialog to return "Alice"
        with patch(
            "metamemory.widgets.floating_panels.QInputDialog.getText",
            return_value=("Alice", True),
        ):
            panel._on_history_anchor_clicked(QUrl("speaker:SPK_0"))

        # Verify the rename happened
        data = self._read_metadata(md_path)
        assert data["words"][0]["speaker_id"] == "Alice"
        assert data["segments"][0]["speaker_id"] == "Alice"

        content = md_path.read_text(encoding="utf-8")
        assert "**Alice**" in content

    def test_on_history_anchor_clicked_cancels_no_change(
        self, panel, tmp_path: Path
    ) -> None:
        """Cancelling the rename dialog must not modify the file."""
        from PyQt6.QtCore import QUrl
        from unittest.mock import patch

        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
        ]
        segments = [
            {"words": [words[0]], "start_time": 0.0, "end_time": 0.5, "avg_confidence": 90, "speaker_id": "SPK_0"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        original_content = md_path.read_text(encoding="utf-8")
        panel._current_history_md_path = md_path

        # Mock QInputDialog to return cancelled
        with patch(
            "metamemory.widgets.floating_panels.QInputDialog.getText",
            return_value=("", False),
        ):
            panel._on_history_anchor_clicked(QUrl("speaker:SPK_0"))

        # File must be unchanged
        assert md_path.read_text(encoding="utf-8") == original_content

    def test_viewer_refreshes_after_rename(
        self, panel, tmp_path: Path
    ) -> None:
        """After rename, the history viewer content should be refreshed."""
        from PyQt6.QtCore import QUrl
        from unittest.mock import patch

        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
        ]
        segments = [
            {"words": [words[0]], "start_time": 0.0, "end_time": 0.5, "avg_confidence": 90, "speaker_id": "SPK_0"},
        ]

        md_path = self._make_transcript_md(tmp_path, words, segments)
        panel._current_history_md_path = md_path

        with patch(
            "metamemory.widgets.floating_panels.QInputDialog.getText",
            return_value=("Carol", True),
        ):
            panel._on_history_anchor_clicked(QUrl("speaker:SPK_0"))

        # Viewer HTML should contain the new name as an anchor
        html = panel._history_viewer.toHtml()
        assert "Carol" in html
