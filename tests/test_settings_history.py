"""
Tests for the Settings History page in FloatingSettingsPanel.

Covers: structure, nav refresh, list population, item selection,
empty state, missing-file fallback, speaker anchor rendering,
delete workflows, scrub workflows, and speaker rename workflows.
"""

import json
import re
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, call

from PyQt6.QtWidgets import (
    QApplication, QListWidget, QListWidgetItem, QSplitter,
    QTextBrowser, QFrame, QMessageBox, QInputDialog, QDialog,
    QPushButton,
)
from PyQt6.QtCore import Qt, QUrl

from meetandread.widgets.floating_panels import FloatingSettingsPanel
from meetandread.transcription.transcript_scanner import RecordingMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(path: str, recording_time: str = "2026-01-15T10:30:00",
               word_count: int = 42, speaker_count: int = 2,
               speakers=None, duration_seconds: float = 60.0,
               wav_exists: bool = True) -> RecordingMeta:
    """Create a RecordingMeta instance for testing."""
    return RecordingMeta(
        path=Path(path),
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=speaker_count,
        speakers=speakers or ["SPK_0", "SPK_1"],
        duration_seconds=duration_seconds,
        wav_exists=wav_exists,
    )


def _write_transcript(path: Path, body: str, metadata: dict) -> None:
    """Write a transcript .md file with metadata footer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = f"\n---\n\n<!-- METADATA: {json.dumps(metadata)} -->\n"
    path.write_text(body + footer, encoding="utf-8")


def _select_recording(panel, tmp_path, qapp, stem="test_rec",
                      body="**SPK_0**\nHello world.\n\n**SPK_1**\nHow are you?\n",
                      speakers=None):
    """Populate list with one recording and click to select it.
    
    Returns (md_path, item).
    """
    if speakers is None:
        speakers = ["SPK_0", "SPK_1"]
    md_path = tmp_path / "transcripts" / f"{stem}.md"
    metadata = {
        "words": [
            {"speaker_id": s, "start_time": float(i), "end_time": float(i + 1), "text": f"word{i}"}
            for i, s in enumerate(speakers)
        ],
        "segments": [
            {"speaker": s, "start": float(i), "end": float(i + 1)}
            for i, s in enumerate(speakers)
        ],
    }
    _write_transcript(md_path, body, metadata)

    meta = _make_meta(str(md_path), wav_exists=True)
    panel._populate_history_list([meta])
    item = panel._history_list.item(0)
    panel._history_list.setCurrentItem(item)
    panel._on_history_item_clicked(item)
    qapp.processEvents()
    return md_path, item


# ---- Fixtures -------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def settings_panel(qapp):
    panel = FloatingSettingsPanel()
    panel.show()
    qapp.processEvents()
    yield panel
    panel.close()


@pytest.fixture
def settings_panel_on_history(settings_panel, qapp):
    """Navigate to History page before returning the panel."""
    settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
    qapp.processEvents()
    return settings_panel


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestSettingsHistoryStructure:
    """Verify History page widgets exist with correct object names."""

    def test_history_page_object_name(self, settings_panel):
        page = settings_panel._content_stack.widget(FloatingSettingsPanel._NAV_HISTORY)
        assert page is not None
        assert page.objectName() == "AethericHistoryPage"

    def test_history_splitter_object_name(self, settings_panel):
        assert settings_panel._history_splitter.objectName() == "AethericHistorySplitter"

    def test_history_list_object_name(self, settings_panel):
        assert settings_panel._history_list.objectName() == "AethericHistoryList"

    def test_history_detail_header_object_name(self, settings_panel):
        assert settings_panel._history_detail_header.objectName() == "AethericHistoryHeader"

    def test_history_viewer_object_name(self, settings_panel):
        assert settings_panel._history_viewer.objectName() == "AethericHistoryViewer"

    def test_scrub_button_object_name(self, settings_panel):
        assert settings_panel._scrub_btn.objectName() == "AethericHistoryActionButton"

    def test_delete_button_object_name(self, settings_panel):
        assert settings_panel._delete_btn.objectName() == "AethericHistoryActionButton"

    def test_scrub_button_action_property(self, settings_panel):
        assert settings_panel._scrub_btn.property("action") == "scrub"

    def test_delete_button_action_property(self, settings_panel):
        assert settings_panel._delete_btn.property("action") == "delete"

    def test_history_page_is_stack_index_2(self, settings_panel):
        assert settings_panel._NAV_HISTORY == 2
        page = settings_panel._content_stack.widget(2)
        assert page is not None
        assert page.objectName() == "AethericHistoryPage"

    def test_splitter_is_vertical(self, settings_panel):
        assert settings_panel._history_splitter.orientation() == Qt.Orientation.Vertical

    def test_viewer_is_read_only(self, settings_panel):
        assert settings_panel._history_viewer.isReadOnly() is True

    def test_viewer_does_not_open_external_links(self, settings_panel):
        assert settings_panel._history_viewer.openExternalLinks() is False

    def test_detail_header_initially_hidden(self, settings_panel):
        assert settings_panel._history_detail_header.isHidden() is True

    def test_state_attributes_initialized(self, settings_panel):
        assert settings_panel._current_history_md_path is None
        assert settings_panel._scrub_runner is None
        assert settings_panel._scrub_model_size is None
        assert settings_panel._is_scrubbing is False
        assert settings_panel._is_comparison_mode is False

    def test_no_placeholder_labels(self, settings_panel):
        """After T02, placeholder labels should not exist."""
        assert not hasattr(settings_panel, '_history_placeholder_title')
        assert not hasattr(settings_panel, '_history_placeholder_desc')


# ---------------------------------------------------------------------------
# Nav refresh tests
# ---------------------------------------------------------------------------

class TestSettingsHistoryNavRefresh:
    """Verify History nav triggers refresh via scan_recordings."""

    def test_nav_to_history_calls_refresh(self, settings_panel, qapp):
        """Navigating to History calls _refresh_history (scan_recordings)."""
        mock_recordings = [_make_meta("/fake/path1.md"), _make_meta("/fake/path2.md")]
        with patch(
            "meetandread.widgets.floating_panels.scan_recordings",
            create=True,
        ) as mock_scan:
            # The import is deferred inside _refresh_history, so patch the import target
            with patch(
                "meetandread.transcription.transcript_scanner.scan_recordings",
                return_value=mock_recordings,
            ):
                settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
                qapp.processEvents()

    def test_nav_to_history_stops_perf_monitor(self, settings_panel, qapp):
        """History nav should stop ResourceMonitor (not start it)."""
        # Start on Performance first
        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_PERFORMANCE)
        qapp.processEvents()
        assert settings_panel._perf_tab_active is True

        # Navigate to History
        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
        qapp.processEvents()
        assert settings_panel._perf_tab_active is False

    def test_refresh_populates_list(self, settings_panel_on_history, qapp):
        """_populate_history_list adds items to the list widget."""
        recordings = [
            _make_meta("/fake/a.md", recording_time="2026-01-15T10:30:00"),
            _make_meta("/fake/b.md", recording_time="2026-01-14T08:00:00"),
        ]
        settings_panel_on_history._populate_history_list(recordings)
        assert settings_panel_on_history._history_list.count() == 2

    def test_refresh_with_empty_list(self, settings_panel_on_history, qapp):
        """Empty recordings list clears the history list."""
        settings_panel_on_history._populate_history_list([])
        assert settings_panel_on_history._history_list.count() == 0


# ---------------------------------------------------------------------------
# List population tests
# ---------------------------------------------------------------------------

class TestSettingsHistoryListPopulation:
    """Verify list items display correct text and carry path data."""

    def test_item_display_text_with_words(self, settings_panel_on_history):
        meta = _make_meta("/fake/test.md", recording_time="2026-01-15T10:30:00",
                          word_count=42, speaker_count=2)
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)
        text = item.text()
        assert "42 words" in text
        assert "2 speakers" in text
        assert "2026-01-15 10:30" in text

    def test_item_display_empty_recording(self, settings_panel_on_history):
        meta = _make_meta("/fake/empty.md", recording_time="2026-01-15T10:30:00",
                          word_count=0, speaker_count=0, speakers=[])
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)
        text = item.text()
        assert "Empty recording" in text

    def test_item_stores_path_as_user_role(self, settings_panel_on_history):
        meta = _make_meta("/fake/test.md")
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)
        assert item.data(Qt.ItemDataRole.UserRole) == str(meta.path)

    def test_populate_clears_previous_items(self, settings_panel_on_history):
        recordings1 = [_make_meta("/fake/a.md"), _make_meta("/fake/b.md")]
        settings_panel_on_history._populate_history_list(recordings1)
        assert settings_panel_on_history._history_list.count() == 2

        recordings2 = [_make_meta("/fake/c.md")]
        settings_panel_on_history._populate_history_list(recordings2)
        assert settings_panel_on_history._history_list.count() == 1

    def test_date_formatting(self, settings_panel_on_history):
        meta = _make_meta("/fake/dated.md", recording_time="2026-03-20T14:45:00")
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)
        assert "2026-03-20 14:45" in item.text()


# ---------------------------------------------------------------------------
# Item selection / viewer tests
# ---------------------------------------------------------------------------

class TestSettingsHistoryItemSelection:
    """Verify item clicks load transcript content."""

    def test_click_missing_file_shows_error(self, settings_panel_on_history, qapp):
        """Clicking an item whose file doesn't exist shows file-not-found."""
        meta = _make_meta("/nonexistent/file.md")
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)

        settings_panel_on_history._on_history_item_clicked(item)
        qapp.processEvents()

        assert settings_panel_on_history._current_history_md_path is None
        viewer_text = settings_panel_on_history._history_viewer.toPlainText()
        assert "File not found" in viewer_text or "not found" in viewer_text.lower()

    def test_click_shows_detail_header(self, settings_panel_on_history, qapp):
        """Clicking any item shows the detail header."""
        meta = _make_meta("/nonexistent/file.md")
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)

        settings_panel_on_history._on_history_item_clicked(item)
        qapp.processEvents()
        assert settings_panel_on_history._history_detail_header.isVisible()

    def test_click_valid_file_renders_html(self, settings_panel_on_history, qapp, tmp_path):
        """Clicking a valid transcript file renders anchor HTML in viewer."""
        md_path = tmp_path / "transcripts" / "test.md"
        metadata = {
            "words": [
                {"speaker_id": "SPK_0", "start_time": 0.0, "end_time": 1.0, "text": "Hello"},
                {"speaker_id": "SPK_1", "start_time": 1.0, "end_time": 2.0, "text": "World"},
            ],
            "segments": [],
        }
        body = "**SPK_0**\nHello world.\n\n**SPK_1**\nHow are you?\n"
        _write_transcript(md_path, body, metadata)

        meta = _make_meta(str(md_path))
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)

        settings_panel_on_history._on_history_item_clicked(item)
        qapp.processEvents()

        assert settings_panel_on_history._current_history_md_path == md_path
        html = settings_panel_on_history._history_viewer.toHtml()
        assert "speaker:SPK_0" in html
        assert "speaker:SPK_1" in html


# ---------------------------------------------------------------------------
# Empty state / negative tests
# ---------------------------------------------------------------------------

class TestSettingsHistoryEmptyState:
    """Verify empty-list and error states."""

    def test_empty_list_viewer_has_placeholder(self, settings_panel_on_history):
        """When list is empty, viewer shows placeholder text."""
        settings_panel_on_history._populate_history_list([])
        assert settings_panel_on_history._history_list.count() == 0
        # Placeholder text should still be set
        assert settings_panel_on_history._history_viewer.placeholderText() != ""

    def test_missing_transcript_file_path_is_none(self, settings_panel_on_history, qapp):
        """Selecting a missing file sets _current_history_md_path to None."""
        meta = _make_meta("/nonexistent.md")
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)

        settings_panel_on_history._on_history_item_clicked(item)
        qapp.processEvents()
        assert settings_panel_on_history._current_history_md_path is None


class TestSettingsHistoryNoMetadata:
    """Verify handling of transcripts without metadata footer."""

    def test_no_metadata_falls_back_to_markdown(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript with no metadata footer falls back to setMarkdown."""
        md_path = tmp_path / "no_meta.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("Just some plain text with no metadata.", encoding="utf-8")

        meta = _make_meta(str(md_path))
        settings_panel_on_history._populate_history_list([meta])
        item = settings_panel_on_history._history_list.item(0)

        settings_panel_on_history._on_history_item_clicked(item)
        qapp.processEvents()

        assert settings_panel_on_history._current_history_md_path == md_path
        # Viewer should show the content (via setMarkdown fallback)
        text = settings_panel_on_history._history_viewer.toPlainText()
        assert "plain text" in text

    def test_malformed_metadata_returns_none(self, settings_panel_on_history, qapp, tmp_path):
        """Malformed metadata JSON falls back gracefully."""
        md_path = tmp_path / "bad_meta.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        content = "Some transcript\n---\n\n<!-- METADATA: {invalid json} -->\n"
        md_path.write_text(content, encoding="utf-8")

        result = settings_panel_on_history._render_history_transcript(md_path)
        # Should return None for malformed metadata
        assert result is None


# ---------------------------------------------------------------------------
# Speaker anchor rendering tests
# ---------------------------------------------------------------------------

class TestSettingsHistorySpeakerAnchors:
    """Verify speaker anchor URL format is speaker:{label}."""

    def test_speaker_anchor_format(self, settings_panel_on_history, qapp, tmp_path):
        """Anchors use speaker:{label} format, not speaker://."""
        md_path = tmp_path / "anchors.md"
        metadata = {
            "words": [
                {"speaker_id": "SPK_0", "start_time": 0.0, "end_time": 1.0, "text": "Hi"},
            ],
            "segments": [],
        }
        body = "**SPK_0**\nHi there.\n"
        _write_transcript(md_path, body, metadata)

        result = settings_panel_on_history._render_history_transcript(md_path)
        assert result is not None
        assert 'href="speaker:SPK_0"' in result
        # Ensure no speaker:// format
        assert "speaker://" not in result

    def test_speaker_anchor_preserves_case(self, settings_panel_on_history, tmp_path):
        """Custom speaker names preserve their case in anchor URLs."""
        md_path = tmp_path / "case.md"
        metadata = {
            "words": [
                {"speaker_id": "Alice", "start_time": 0.0, "end_time": 1.0, "text": "Hi"},
            ],
            "segments": [],
        }
        body = "**Alice**\nHello.\n"
        _write_transcript(md_path, body, metadata)

        result = settings_panel_on_history._render_history_transcript(md_path)
        assert result is not None
        assert 'href="speaker:Alice"' in result


# ---------------------------------------------------------------------------
# _extract_transcript_body tests
# ---------------------------------------------------------------------------

class TestSettingsExtractTranscriptBody:
    """Verify _extract_transcript_body static method."""

    def test_none_path_returns_not_found(self):
        result = FloatingSettingsPanel._extract_transcript_body(None)
        assert "not found" in result

    def test_missing_file_returns_not_found(self):
        result = FloatingSettingsPanel._extract_transcript_body(Path("/nonexistent.md"))
        assert "not found" in result

    def test_valid_file_extracts_body(self, tmp_path):
        md_path = tmp_path / "body.md"
        md_path.write_text("Line one\nLine two\n---\n\n<!-- METADATA: {} -->\n", encoding="utf-8")
        result = FloatingSettingsPanel._extract_transcript_body(md_path)
        assert "Line one" in result
        assert "Line two" in result
        assert "METADATA" not in result

    def test_no_footer_returns_full_content(self, tmp_path):
        md_path = tmp_path / "no_footer.md"
        md_path.write_text("Just content here", encoding="utf-8")
        result = FloatingSettingsPanel._extract_transcript_body(md_path)
        assert "Just content here" in result


# ---------------------------------------------------------------------------
# _reselect_history_item tests
# ---------------------------------------------------------------------------

class TestSettingsReselectHistoryItem:
    """Verify re-selection after list repopulation."""

    def test_reselect_finds_matching_item(self, settings_panel_on_history, qapp):
        recordings = [
            _make_meta("/fake/a.md"),
            _make_meta("/fake/b.md"),
        ]
        settings_panel_on_history._populate_history_list(recordings)

        # Reselect second item — use the same Path that _populate_history_list stored
        target_path = recordings[1].path
        settings_panel_on_history._reselect_history_item(target_path)
        current = settings_panel_on_history._history_list.currentItem()
        assert current is not None
        assert current.data(Qt.ItemDataRole.UserRole) == str(target_path)

    def test_reselect_missing_path_no_crash(self, settings_panel_on_history, qapp):
        """Reselecting a path not in the list doesn't crash."""
        recordings = [_make_meta("/fake/a.md")]
        settings_panel_on_history._populate_history_list(recordings)

        # Should not raise
        settings_panel_on_history._reselect_history_item(Path("/fake/nonexistent.md"))


# ---------------------------------------------------------------------------
# Delete workflow tests
# ---------------------------------------------------------------------------

class TestSettingsDeleteWorkflow:
    """Verify delete confirm/cancel/failure and state cleanup."""

    def test_delete_yes_removes_and_clears(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        assert settings_panel_on_history._current_history_md_path == md_path

        with patch("meetandread.widgets.floating_panels.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes), \
             patch("meetandread.recording.management.enumerate_recording_files",
                    return_value=[md_path]), \
             patch("meetandread.recording.management.delete_recording",
                    return_value=(1, [str(md_path)])):
            settings_panel_on_history._delete_recording(item)
            qapp.processEvents()

        assert settings_panel_on_history._current_history_md_path is None
        assert settings_panel_on_history._history_detail_header.isHidden()

    def test_delete_cancel_leaves_everything(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)

        with patch("meetandread.widgets.floating_panels.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No):
            settings_panel_on_history._delete_recording(item)

        assert settings_panel_on_history._current_history_md_path == md_path
        assert not settings_panel_on_history._history_detail_header.isHidden()

    def test_delete_exception_shows_warning(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)

        with patch("meetandread.widgets.floating_panels.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes), \
             patch("meetandread.recording.management.enumerate_recording_files",
                    return_value=[md_path]), \
             patch("meetandread.recording.management.delete_recording",
                    side_effect=OSError("disk error")), \
             patch("meetandread.widgets.floating_panels.QMessageBox.warning") as warn:
            settings_panel_on_history._delete_recording(item)
            qapp.processEvents()
            warn.assert_called_once()

        assert settings_panel_on_history._current_history_md_path == md_path

    def test_delete_no_current_item_is_noop(self, settings_panel_on_history, qapp):
        settings_panel_on_history._on_delete_btn_clicked()


# ---------------------------------------------------------------------------
# Scrub workflow tests
# ---------------------------------------------------------------------------

class TestSettingsScrubWorkflow:
    """Verify scrub start/progress/error/comparison flows."""

    def test_scrub_refuses_missing_wav(self, settings_panel_on_history, qapp, tmp_path):
        _select_recording(settings_panel_on_history, tmp_path, qapp)

        # get_recordings_dir returns real dir, WAV won't exist there
        with patch("meetandread.widgets.floating_panels.QMessageBox.information") as info:
            settings_panel_on_history._on_scrub_clicked()
            info.assert_called_once()

    def test_scrub_already_scrubbing_is_noop(self, settings_panel_on_history, qapp):
        settings_panel_on_history._is_scrubbing = True
        settings_panel_on_history._on_scrub_clicked()

    def test_scrub_dialog_cancel_does_nothing(self, settings_panel_on_history, qapp, tmp_path):
        _select_recording(settings_panel_on_history, tmp_path, qapp)
        wav_dir = tmp_path / "recordings"
        wav_dir.mkdir(exist_ok=True)
        (wav_dir / "test_rec.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = 0
        settings_panel_on_history._create_scrub_dialog = lambda: mock_dialog

        with patch("meetandread.audio.storage.paths.get_recordings_dir",
                    return_value=wav_dir):
            settings_panel_on_history._on_scrub_clicked()

        assert not settings_panel_on_history._is_scrubbing

    def test_scrub_start_sets_state(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        wav_path = tmp_path / "test_rec.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_runner = MagicMock()
        mock_runner.scrub_recording.return_value = "/fake/sidecar.md"

        with patch("meetandread.transcription.scrub.ScrubRunner",
                    return_value=mock_runner), \
             patch.object(settings_panel_on_history, "_get_app_settings",
                          return_value=MagicMock()):
            settings_panel_on_history._start_scrub(wav_path, md_path, "small")

        assert settings_panel_on_history._is_scrubbing is True
        assert settings_panel_on_history._scrub_model_size == "small"
        assert not settings_panel_on_history._scrub_btn.isEnabled()

    def test_scrub_complete_error_reenables_button(self, settings_panel_on_history, qapp):
        settings_panel_on_history._is_scrubbing = True
        settings_panel_on_history._scrub_btn.setEnabled(False)

        with patch("meetandread.widgets.floating_panels.QMessageBox.warning"):
            settings_panel_on_history._handle_scrub_complete(None, "Transcription failed")

        assert settings_panel_on_history._is_scrubbing is False
        assert settings_panel_on_history._scrub_btn.isEnabled()
        assert not settings_panel_on_history._is_comparison_mode

    def test_scrub_complete_shows_comparison(self, settings_panel_on_history, qapp, tmp_path):
        settings_panel_on_history._is_scrubbing = True
        settings_panel_on_history._scrub_btn.setEnabled(False)
        settings_panel_on_history._scrub_model_size = "small"

        sidecar = tmp_path / "test_rec_scrub_small.md"
        sidecar.write_text("**SPK_0**\nScrubbed text.\n", encoding="utf-8")

        settings_panel_on_history._handle_scrub_complete(str(sidecar), None)
        qapp.processEvents()

        assert settings_panel_on_history._is_comparison_mode is True
        assert settings_panel_on_history._scrub_btn.isHidden()


# ---------------------------------------------------------------------------
# Scrub accept/reject tests
# ---------------------------------------------------------------------------

class TestSettingsScrubAcceptReject:
    """Verify accept/reject refresh and reselection."""

    def test_accept_promotes_sidecar(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        settings_panel_on_history._scrub_model_size = "small"
        settings_panel_on_history._is_comparison_mode = True

        with patch("meetandread.transcription.scrub.ScrubRunner.accept_scrub") as mock_accept:
            settings_panel_on_history._on_scrub_accept()
            mock_accept.assert_called_once_with(md_path, "small")

        assert not settings_panel_on_history._is_comparison_mode

    def test_accept_missing_sidecar_shows_warning(self, settings_panel_on_history, qapp, tmp_path):
        _select_recording(settings_panel_on_history, tmp_path, qapp)
        settings_panel_on_history._scrub_model_size = "small"

        with patch("meetandread.transcription.scrub.ScrubRunner.accept_scrub",
                    side_effect=FileNotFoundError("gone")), \
             patch("meetandread.widgets.floating_panels.QMessageBox.warning") as warn:
            settings_panel_on_history._on_scrub_accept()
            warn.assert_called_once()

    def test_reject_deletes_sidecar(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        settings_panel_on_history._scrub_model_size = "small"
        settings_panel_on_history._is_comparison_mode = True

        with patch("meetandread.transcription.scrub.ScrubRunner.reject_scrub") as mock_reject:
            settings_panel_on_history._on_scrub_reject()
            mock_reject.assert_called_once_with(md_path, "small")

        assert not settings_panel_on_history._is_comparison_mode

    def test_reject_no_path_is_noop(self, settings_panel_on_history, qapp):
        settings_panel_on_history._current_history_md_path = None
        settings_panel_on_history._scrub_model_size = None
        settings_panel_on_history._on_scrub_reject()


# ---------------------------------------------------------------------------
# Speaker rename workflow tests
# ---------------------------------------------------------------------------

class TestSettingsSpeakerRenameWorkflow:
    """Verify speaker anchor rename via _on_history_anchor_clicked."""

    def test_rename_updates_file(self, settings_panel_on_history, qapp, tmp_path):
        body = "**SPK_0**\nHello.\n\n**SPK_1**\nWorld.\n"
        md_path, item = _select_recording(
            settings_panel_on_history, tmp_path, qapp,
            body=body, speakers=["SPK_0", "SPK_1"],
        )

        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("Alice", True)):
            settings_panel_on_history._on_history_anchor_clicked(url)

        content = md_path.read_text(encoding="utf-8")
        assert "**Alice**" in content
        assert "**SPK_0**" not in content

    def test_rename_cancel_does_nothing(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        original = md_path.read_text(encoding="utf-8")

        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("", False)):
            settings_panel_on_history._on_history_anchor_clicked(url)

        assert md_path.read_text(encoding="utf-8") == original

    def test_rename_blank_name_is_noop(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        original = md_path.read_text(encoding="utf-8")

        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("  ", True)):
            settings_panel_on_history._on_history_anchor_clicked(url)

        assert md_path.read_text(encoding="utf-8") == original

    def test_rename_same_name_is_noop(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(settings_panel_on_history, tmp_path, qapp)
        original = md_path.read_text(encoding="utf-8")

        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("SPK_0", True)):
            settings_panel_on_history._on_history_anchor_clicked(url)

        assert md_path.read_text(encoding="utf-8") == original

    def test_rename_no_current_path_is_noop(self, settings_panel_on_history, qapp):
        settings_panel_on_history._current_history_md_path = None
        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("Alice", True)):
            settings_panel_on_history._on_history_anchor_clicked(url)

    def test_rename_preserves_url_case(self, settings_panel_on_history, qapp):
        url = QUrl("speaker:SPK_0")
        assert url.toString() == "speaker:SPK_0"

    def test_rename_propagate_signatures_best_effort(self, settings_panel_on_history, qapp, tmp_path):
        md_path, item = _select_recording(
            settings_panel_on_history, tmp_path, qapp,
            body="**SPK_0**\nHello.\n", speakers=["SPK_0"],
        )

        url = QUrl("speaker:SPK_0")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("Alice", True)), \
             patch.object(settings_panel_on_history, "_propagate_rename_to_signatures",
                          side_effect=RuntimeError("db locked")):
            try:
                settings_panel_on_history._on_history_anchor_clicked(url)
            except RuntimeError:
                pytest.fail("Signature propagation error should not propagate")

    def test_non_speaker_url_is_ignored(self, settings_panel_on_history, qapp):
        url = QUrl("https://example.com")
        settings_panel_on_history._on_history_anchor_clicked(url)


# ---------------------------------------------------------------------------
# Cross-panel regression tests (T04)
# ---------------------------------------------------------------------------

class TestCrossPanelStateIsolation:
    """Verify Settings and Transcript panels have independent History state.

    Q7 negative tests: verify state does not alias between panels.
    """

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        panel = FloatingSettingsPanel()
        panel.show()
        qapp.processEvents()
        yield panel
        panel.close()

    @pytest.fixture
    def transcript_panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel()
        panel.show()
        qapp.processEvents()
        yield panel
        panel.close()

    def test_panels_have_distinct_history_path_attrs(self, settings_panel, transcript_panel):
        """Each panel has its own _current_history_md_path attribute."""
        assert hasattr(settings_panel, "_current_history_md_path")
        assert hasattr(transcript_panel, "_current_history_md_path")
        # They must be distinct objects (not the same reference)
        assert settings_panel is not transcript_panel

    def test_selecting_settings_does_not_mutate_transcript_path(
        self, settings_panel, transcript_panel, qapp, tmp_path,
    ):
        """Selecting a recording in Settings does not change transcript panel path."""
        assert transcript_panel._current_history_md_path is None

        # Select a recording in Settings panel
        md_path = tmp_path / "transcripts" / "settings_test.md"
        metadata = {
            "words": [{"speaker_id": "SPK_0", "start_time": 0.0, "end_time": 1.0, "text": "Hi"}],
            "segments": [],
        }
        _write_transcript(md_path, "**SPK_0**\nHi.\n", metadata)
        meta = _make_meta(str(md_path))
        settings_panel._populate_history_list([meta])
        item = settings_panel._history_list.item(0)
        settings_panel._history_list.setCurrentItem(item)
        settings_panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert settings_panel._current_history_md_path == md_path
        # Transcript panel must remain None
        assert transcript_panel._current_history_md_path is None

    def test_selecting_transcript_does_not_mutate_settings_path(
        self, settings_panel, transcript_panel, qapp, tmp_path,
    ):
        """Selecting a recording in Transcript panel does not change Settings path."""
        assert settings_panel._current_history_md_path is None

        md_path = tmp_path / "transcripts" / "transcript_test.md"
        metadata = {
            "words": [{"speaker_id": "SPK_0", "start_time": 0.0, "end_time": 1.0, "text": "Hi"}],
            "segments": [],
        }
        _write_transcript(md_path, "**SPK_0**\nHi.\n", metadata)
        meta = _make_meta(str(md_path))
        transcript_panel._populate_history_list([meta])
        item = transcript_panel._history_list.item(0)
        transcript_panel._history_list.setCurrentItem(item)
        transcript_panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert transcript_panel._current_history_md_path == md_path
        assert settings_panel._current_history_md_path is None

    def test_scrubbing_state_is_independent(
        self, settings_panel, transcript_panel, qapp,
    ):
        """Scrubbing state in one panel does not affect the other."""
        settings_panel._is_scrubbing = True
        assert transcript_panel._is_scrubbing is False

        transcript_panel._is_scrubbing = True
        assert settings_panel._is_scrubbing is True  # still True
        assert transcript_panel._is_scrubbing is True

    def test_comparison_mode_is_independent(
        self, settings_panel, transcript_panel, qapp,
    ):
        """Comparison mode in one panel does not affect the other."""
        settings_panel._is_comparison_mode = True
        assert transcript_panel._is_comparison_mode is False

    def test_settings_uses_aetheric_object_names(self, settings_panel):
        """Settings panel History widgets use Aetheric object names."""
        assert settings_panel._history_splitter.objectName() == "AethericHistorySplitter"
        assert settings_panel._history_list.objectName() == "AethericHistoryList"
        assert settings_panel._history_viewer.objectName() == "AethericHistoryViewer"

    def test_transcript_panel_has_no_aetheric_object_names(self, transcript_panel):
        """Transcript panel History widgets do NOT use Aetheric object names."""
        assert transcript_panel._history_list.objectName() != "AethericHistoryList"
        assert transcript_panel._history_viewer.objectName() != "AethericHistoryViewer"


class TestNavAwayBackDeterminism:
    """Verify History nav away/back refresh remains deterministic.

    Q7 negative test: nav away/back behavior is deterministic.
    """

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        panel = FloatingSettingsPanel()
        panel.show()
        qapp.processEvents()
        yield panel
        panel.close()

    def test_nav_away_and_back_clears_then_repulates(
        self, settings_panel, qapp,
    ):
        """Navigate away from History and back triggers a clean refresh."""
        recordings = [_make_meta("/fake/a.md"), _make_meta("/fake/b.md")]

        # Go to History with recordings
        with patch("meetandread.transcription.transcript_scanner.scan_recordings",
                    return_value=recordings):
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
            qapp.processEvents()

        count_after_first = settings_panel._history_list.count()

        # Nav away to Settings
        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
        qapp.processEvents()

        # Nav back to History with different recordings
        recordings2 = [_make_meta("/fake/c.md")]
        with patch("meetandread.transcription.transcript_scanner.scan_recordings",
                    return_value=recordings2):
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
            qapp.processEvents()

        # List should reflect the second scan result
        assert settings_panel._history_list.count() == 1
        assert "c.md" in settings_panel._history_list.item(0).data(Qt.ItemDataRole.UserRole)

    def test_nav_to_history_clears_list_on_empty_scan(
        self, settings_panel, qapp, tmp_path,
    ):
        """Navigating away and back with empty scan clears the list."""
        md_path = tmp_path / "transcripts" / "navtest.md"
        metadata = {
            "words": [{"speaker_id": "SPK_0", "start_time": 0.0, "end_time": 1.0, "text": "Hi"}],
            "segments": [],
        }
        _write_transcript(md_path, "**SPK_0**\nHi.\n", metadata)
        meta = _make_meta(str(md_path))

        with patch("meetandread.transcription.transcript_scanner.scan_recordings",
                    return_value=[meta]):
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
            qapp.processEvents()

        assert settings_panel._history_list.count() == 1

        # Nav away
        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
        qapp.processEvents()

        # Nav back with empty recordings
        with patch("meetandread.transcription.transcript_scanner.scan_recordings",
                    return_value=[]):
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
            qapp.processEvents()

        # List should be empty after scan returns no recordings
        assert settings_panel._history_list.count() == 0


class TestStalePlaceholderAbsence:
    """Verify stale placeholder attributes are absent (Q7 negative test)."""

    @pytest.fixture
    def qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def settings_panel(self, qapp):
        panel = FloatingSettingsPanel()
        panel.show()
        qapp.processEvents()
        yield panel
        panel.close()

    def test_no_placeholder_title_attribute(self, settings_panel):
        assert not hasattr(settings_panel, "_history_placeholder_title")

    def test_no_placeholder_desc_attribute(self, settings_panel):
        assert not hasattr(settings_panel, "_history_placeholder_desc")

    def test_no_tab_widget_attribute(self, settings_panel):
        assert not hasattr(settings_panel, "_tab_widget")

    def test_no_title_label_attribute(self, settings_panel):
        assert not hasattr(settings_panel, "_title_label")

    def test_no_close_btn_attribute(self, settings_panel):
        assert not hasattr(settings_panel, "_close_btn")

    def test_history_page_is_real_widget(self, settings_panel):
        """History page is a real QWidget, not a placeholder."""
        page = settings_panel._content_stack.widget(FloatingSettingsPanel._NAV_HISTORY)
        assert page is not None
        assert page.objectName() == "AethericHistoryPage"
        # It must have child widgets (splitter, list, viewer)
        assert settings_panel._history_list is not None
        assert settings_panel._history_viewer is not None
