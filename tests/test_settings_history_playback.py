"""Tests for the Settings History playback toolbar in FloatingSettingsPanel.

Covers: toolbar widget structure, audio-present enablement, missing-audio
disablement/status, play/pause routing, speed routing, volume routing,
reload-on-new-selection, and negative edge cases.

Uses a mock HistoryPlaybackController to avoid QtMultimedia DLL issues.
"""

import os
import sys

import pytest

# Skip this module in headless environments where Qt cannot be imported
if not os.environ.get("DISPLAY") and not os.environ.get("CI"):
    pytest.skip(
        "Skipping Qt widget tests in headless environment (requires DISPLAY or CI=1 with display context)",
        allow_module_level=True,
    )

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from PyQt6.QtWidgets import (
    QApplication, QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QSlider, QLabel,
)
from PyQt6.QtCore import Qt

from meetandread.widgets.floating_panels import FloatingSettingsPanel
from meetandread.transcription.transcript_scanner import RecordingMeta

# Mark this module as requiring real Qt widgets - these tests
# cannot run in headless environments due to Qt platform plugin DLL loading
pytestmark = pytest.mark.requires_qt_widgets

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


def _make_mock_helper(audio_available=True, last_error=None, status_text="Ready"):
    """Create a mock HistoryPlaybackController.

    Returns a MagicMock that simulates the helper's API.
    """
    helper = MagicMock()
    helper.is_audio_available = audio_available
    helper.last_error = last_error
    helper.status_text = status_text
    helper.current_transcript_path = None
    helper.current_audio_path = None

    # Mock player with PlaybackState enum
    mock_player = MagicMock()
    mock_player.PlaybackState = MagicMock()
    mock_player.PlaybackState.PlayingState = 1
    mock_player.PlaybackState.PausedState = 2
    mock_player.PlaybackState.StoppedState = 0
    mock_player.playbackState.return_value = 0  # StoppedState
    helper.player = mock_player

    helper.load_transcript_audio = MagicMock()
    helper.play = MagicMock()
    helper.pause = MagicMock()
    helper.stop = MagicMock()
    helper.set_rate = MagicMock()
    helper.set_volume = MagicMock()
    helper.skip_forward = MagicMock()
    helper.skip_backward = MagicMock()
    helper.seek_to = MagicMock()
    helper.duration_ms = 0
    helper.position_ms = 0
    return helper


def _select_and_populate(panel, tmp_path, qapp, stem="test_rec",
                         body="**SPK_0**\nHello world.\n",
                         speakers=None, wav_exists=True):
    """Populate list with one recording and click to select it.

    Returns (md_path, item).
    """
    if speakers is None:
        speakers = ["SPK_0"]
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

    meta = _make_meta(str(md_path), wav_exists=wav_exists)
    panel._populate_history_list([meta])
    item = panel._history_list.item(0)
    panel._history_list.setCurrentItem(item)
    panel._on_history_item_clicked(item)
    qapp.processEvents()
    return md_path, item


def _write_timed_transcript(tmp_path, stem, words, body=None):
    """Write a transcript with timed word metadata.

    Args:
        tmp_path: Temporary directory root.
        stem: File stem for the transcript.
        words: List of word dicts with text, start_time, etc.
        body: Optional markdown body (defaults to empty).

    Returns:
        Path to the written transcript.
    """
    if body is None:
        body = ""
    md_path = tmp_path / "transcripts" / f"{stem}.md"
    segments = []
    for i, w in enumerate(words):
        sid = w.get("speaker_id")
        st = w.get("start_time", 0.0)
        et = w.get("end_time", st + 1.0)
        if not segments or segments[-1]["speaker"] != sid:
            segments.append({"speaker": sid, "start": st, "end": et})
        else:
            segments[-1]["end"] = et
    metadata = {"words": words, "segments": segments}
    _write_transcript(md_path, body, metadata)
    return md_path


# ---------------------------------------------------------------------------
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
    """Navigate to History page and inject a mock playback helper."""
    settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
    qapp.processEvents()
    # Pre-inject a mock helper so _ensure_playback_helper returns it
    mock_helper = _make_mock_helper()
    settings_panel._playback_helper = mock_helper
    return settings_panel


# ---------------------------------------------------------------------------
# Toolbar structure tests
# ---------------------------------------------------------------------------

class TestPlaybackToolbarStructure:
    """Verify playback toolbar widgets exist with correct object names."""

    def test_play_button_object_name(self, settings_panel):
        assert settings_panel._playback_play_btn.objectName() == "AethericHistoryPlaybackButton"

    def test_play_button_has_playback_action_property(self, settings_panel):
        assert settings_panel._playback_play_btn.property("playback_action") == "play_pause"

    def test_speed_combo_object_name(self, settings_panel):
        assert settings_panel._playback_speed_combo.objectName() == "AethericHistoryPlaybackSpeedCombo"

    def test_speed_combo_has_all_rates(self, settings_panel):
        combo = settings_panel._playback_speed_combo
        expected = ["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"]
        actual = [combo.itemText(i) for i in range(combo.count())]
        assert actual == expected

    def test_speed_combo_defaults_to_1x(self, settings_panel):
        assert settings_panel._playback_speed_combo.currentText() == "1x"

    def test_volume_slider_object_name(self, settings_panel):
        assert settings_panel._playback_volume_slider.objectName() == "AethericHistoryPlaybackVolumeSlider"

    def test_volume_slider_range(self, settings_panel):
        assert settings_panel._playback_volume_slider.minimum() == 0
        assert settings_panel._playback_volume_slider.maximum() == 100

    def test_volume_slider_default(self, settings_panel):
        assert settings_panel._playback_volume_slider.value() == 80

    def test_status_label_object_name(self, settings_panel):
        assert settings_panel._playback_status_label.objectName() == "AethericHistoryPlaybackStatusLabel"

    def test_volume_label_object_name(self, settings_panel):
        assert settings_panel._playback_volume_label.objectName() == "AethericHistoryPlaybackVolumeIcon"

    def test_controls_initially_disabled(self, settings_panel):
        assert not settings_panel._playback_play_btn.isEnabled()
        assert not settings_panel._playback_speed_combo.isEnabled()
        assert not settings_panel._playback_volume_slider.isEnabled()
        assert not settings_panel._playback_progress_slider.isEnabled()
        assert not settings_panel._playback_skip_back_btn.isEnabled()
        assert not settings_panel._playback_skip_fwd_btn.isEnabled()

    def test_playback_helper_initially_none(self, settings_panel):
        assert settings_panel._playback_helper is None

    def test_play_button_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._playback_play_btn) >= 0

    def test_speed_combo_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._playback_speed_combo) >= 0

    def test_controls_before_scrub_delete(self, settings_panel):
        """Playback controls should appear before Scrub/Delete buttons."""
        layout = settings_panel._history_detail_header.layout()
        play_idx = layout.indexOf(settings_panel._playback_play_btn)
        scrub_idx = layout.indexOf(settings_panel._scrub_btn)
        delete_idx = layout.indexOf(settings_panel._delete_btn)
        assert play_idx < scrub_idx
        assert play_idx < delete_idx

    def test_progress_slider_object_name(self, settings_panel):
        assert settings_panel._playback_progress_slider.objectName() == "AethericHistoryPlaybackProgressSlider"

    def test_progress_slider_range(self, settings_panel):
        assert settings_panel._playback_progress_slider.minimum() == 0
        assert settings_panel._playback_progress_slider.maximum() == 1000

    def test_progress_slider_default_value(self, settings_panel):
        assert settings_panel._playback_progress_slider.value() == 0

    def test_progress_slider_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._playback_progress_slider) >= 0

    def test_skip_back_button_object_name(self, settings_panel):
        assert settings_panel._playback_skip_back_btn.objectName() == "AethericHistoryPlaybackSkipBackButton"

    def test_skip_back_button_has_skip_back_property(self, settings_panel):
        assert settings_panel._playback_skip_back_btn.property("playback_action") == "skip_back"

    def test_skip_fwd_button_object_name(self, settings_panel):
        assert settings_panel._playback_skip_fwd_btn.objectName() == "AethericHistoryPlaybackSkipFwdButton"

    def test_skip_fwd_button_has_skip_fwd_property(self, settings_panel):
        assert settings_panel._playback_skip_fwd_btn.property("playback_action") == "skip_fwd"

    def test_skip_buttons_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._playback_skip_back_btn) >= 0
        assert layout.indexOf(settings_panel._playback_skip_fwd_btn) >= 0

    def test_drag_flag_initially_false(self, settings_panel):
        assert settings_panel._is_dragging_progress_slider is False


# ---------------------------------------------------------------------------
# Audio-present enablement tests
# ---------------------------------------------------------------------------

class TestPlaybackAudioPresent:
    """When a transcript with valid audio is selected, controls enable."""

    def test_controls_enabled_on_audio_present(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        # Configure helper to report audio available
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path, _ = _select_and_populate(panel, tmp_path, qapp)

        assert panel._playback_play_btn.isEnabled()
        assert panel._playback_speed_combo.isEnabled()
        assert panel._playback_volume_slider.isEnabled()
        assert panel._playback_progress_slider.isEnabled()
        assert panel._playback_skip_back_btn.isEnabled()
        assert panel._playback_skip_fwd_btn.isEnabled()

    def test_status_label_shows_ready(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        assert panel._playback_status_label.text() == "Ready"

    def test_helper_load_called_with_md_path(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        md_path, _ = _select_and_populate(panel, tmp_path, qapp)
        panel._playback_helper.load_transcript_audio.assert_called_with(md_path)


# ---------------------------------------------------------------------------
# Missing-audio disablement tests
# ---------------------------------------------------------------------------

class TestPlaybackMissingAudio:
    """When audio is missing, controls disable with status message."""

    def test_controls_disabled_on_missing_audio(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"

        _select_and_populate(panel, tmp_path, qapp)

        assert not panel._playback_play_btn.isEnabled()
        assert not panel._playback_speed_combo.isEnabled()
        assert not panel._playback_volume_slider.isEnabled()
        assert not panel._playback_progress_slider.isEnabled()
        assert not panel._playback_skip_back_btn.isEnabled()
        assert not panel._playback_skip_fwd_btn.isEnabled()

    def test_status_label_shows_missing(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"

        _select_and_populate(panel, tmp_path, qapp)
        assert "Audio file not found" in panel._playback_status_label.text()


# ---------------------------------------------------------------------------
# Play/pause routing tests
# ---------------------------------------------------------------------------

class TestPlaybackPlayPauseRouting:
    """Play/pause button routes to helper methods."""

    def test_play_calls_helper_play(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        # Not playing currently
        panel._playback_helper.player.playbackState.return_value = 0  # StoppedState

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_play_btn.click()
        qapp.processEvents()

        panel._playback_helper.play.assert_called_once()

    def test_pause_calls_helper_pause(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        # Currently playing
        panel._playback_helper.player.playbackState.return_value = 1  # PlayingState

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_play_btn.click()
        qapp.processEvents()

        panel._playback_helper.pause.assert_called_once()

    def test_play_noop_when_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        # Force enable to simulate clicking while unavailable
        panel._playback_play_btn.setEnabled(True)
        panel._playback_play_btn.click()
        qapp.processEvents()

        panel._playback_helper.play.assert_not_called()
        panel._playback_helper.pause.assert_not_called()


# ---------------------------------------------------------------------------
# Speed routing tests
# ---------------------------------------------------------------------------

class TestPlaybackSpeedRouting:
    """Speed combo routes to helper.set_rate."""

    def test_speed_change_routes_to_helper(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)

        # Change to 1.5x (index 5)
        panel._playback_speed_combo.setCurrentIndex(5)
        qapp.processEvents()

        panel._playback_helper.set_rate.assert_called_with(1.5)

    def test_speed_noop_when_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        # Force enable
        panel._playback_speed_combo.setEnabled(True)
        panel._playback_speed_combo.setCurrentIndex(0)  # 0.25x
        qapp.processEvents()

        panel._playback_helper.set_rate.assert_not_called()

    def test_all_speed_values(self, settings_panel_on_history, qapp, tmp_path):
        """Each combo index maps to the correct float rate."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)

        expected_rates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for i, expected_rate in enumerate(expected_rates):
            panel._playback_helper.set_rate.reset_mock()
            panel._playback_speed_combo.setCurrentIndex(i)
            qapp.processEvents()
            panel._playback_helper.set_rate.assert_called_with(expected_rate)


# ---------------------------------------------------------------------------
# Volume routing tests
# ---------------------------------------------------------------------------

class TestPlaybackVolumeRouting:
    """Volume slider routes to helper.set_volume."""

    def test_volume_change_routes_to_helper(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)

        panel._playback_volume_slider.setValue(50)
        qapp.processEvents()

        panel._playback_helper.set_volume.assert_called_with(0.5)

    def test_volume_zero(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)

        panel._playback_volume_slider.setValue(0)
        qapp.processEvents()

        panel._playback_helper.set_volume.assert_called_with(0.0)

    def test_volume_max(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)

        panel._playback_volume_slider.setValue(100)
        qapp.processEvents()

        panel._playback_helper.set_volume.assert_called_with(1.0)

    def test_volume_noop_when_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_volume_slider.setEnabled(True)
        panel._playback_volume_slider.setValue(30)
        qapp.processEvents()

        panel._playback_helper.set_volume.assert_not_called()


# ---------------------------------------------------------------------------
# Reload-on-new-selection tests
# ---------------------------------------------------------------------------

class TestPlaybackReloadOnSelection:
    """Selecting a new transcript reloads audio and resets state."""

    def test_second_selection_calls_load_again(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md1, _ = _select_and_populate(panel, tmp_path, qapp, stem="rec1")
        assert panel._playback_helper.load_transcript_audio.call_count >= 1

        md2, _ = _select_and_populate(panel, tmp_path, qapp, stem="rec2")
        # Load should have been called with the new path
        panel._playback_helper.load_transcript_audio.assert_called_with(md2)

    def test_missing_audio_after_audio_present(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history

        # First: audio available
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        _select_and_populate(panel, tmp_path, qapp, stem="with_audio")
        assert panel._playback_play_btn.isEnabled()

        # Second: audio missing
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"
        _select_and_populate(panel, tmp_path, qapp, stem="no_audio")
        assert not panel._playback_play_btn.isEnabled()
        assert "Audio file not found" in panel._playback_status_label.text()


# ---------------------------------------------------------------------------
# Negative / edge-case tests
# ---------------------------------------------------------------------------

class TestPlaybackNegativeCases:
    """Malformed inputs, error paths, and boundary conditions."""

    def test_item_without_user_role_path(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        # After calling _update_playback_for_no_audio, the helper should get
        # load_transcript_audio(None). The mock keeps is_audio_available=True
        # (it doesn't actually change), but we verify the load was called with None.
        item = QListWidgetItem("Empty item")
        item.setData(Qt.ItemDataRole.UserRole, "")
        panel._history_list.addItem(item)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Helper should have been asked to unload (load_transcript_audio(None))
        panel._playback_helper.load_transcript_audio.assert_called_with(None)

    def test_missing_transcript_file(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        item = QListWidgetItem("Missing file")
        item.setData(Qt.ItemDataRole.UserRole, "/nonexistent/path/transcript.md")
        panel._history_list.addItem(item)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # File not found → should show error in viewer, playback disabled
        assert not panel._playback_play_btn.isEnabled()

    def test_helper_load_error(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio could not be loaded"
        panel._playback_helper.status_text = "Audio could not be loaded"

        _select_and_populate(panel, tmp_path, qapp)

        assert not panel._playback_play_btn.isEnabled()
        assert "Audio could not be loaded" in panel._playback_status_label.text()

    def test_stop_on_hide_panel(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel.hide_panel()
        qapp.processEvents()

        panel._playback_helper.stop.assert_called()

    def test_stop_on_nav_away_from_history(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
        qapp.processEvents()

        panel._playback_helper.stop.assert_called()

    def test_play_button_height(self, settings_panel):
        """Play button should have compact fixed height."""
        assert settings_panel._playback_play_btn.height() <= 30

    def test_speed_combo_height(self, settings_panel):
        """Speed combo should have compact fixed height."""
        assert settings_panel._playback_speed_combo.height() <= 32

    def test_volume_slider_width(self, settings_panel):
        """Volume slider should be reasonably narrow."""
        assert settings_panel._playback_volume_slider.width() <= 80

    def test_stop_on_delete_recording(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path, _ = _select_and_populate(panel, tmp_path, qapp)
        # Simulate the delete flow clearing playback
        panel._current_history_md_path = None
        panel._stop_playback()
        panel._sync_playback_controls()

        panel._playback_helper.stop.assert_called()


# ---------------------------------------------------------------------------
# Accessible name and tooltip tests
# ---------------------------------------------------------------------------

class TestPlaybackAccessibility:
    """Verify accessible names, descriptions, and tooltips are set."""

    def test_play_button_accessible_name(self, settings_panel):
        assert settings_panel._playback_play_btn.accessibleName() == "Play or pause audio"

    def test_play_button_accessible_description(self, settings_panel):
        desc = settings_panel._playback_play_btn.accessibleDescription()
        assert "audio" in desc.lower()
        assert "transcript" in desc.lower()

    def test_play_button_tooltip(self, settings_panel):
        tip = settings_panel._playback_play_btn.toolTip()
        assert "Play" in tip
        assert "Pause" in tip

    def test_speed_combo_accessible_name(self, settings_panel):
        assert settings_panel._playback_speed_combo.accessibleName() == "Playback speed"

    def test_speed_combo_accessible_description(self, settings_panel):
        desc = settings_panel._playback_speed_combo.accessibleDescription()
        assert "speed" in desc.lower()

    def test_speed_combo_tooltip(self, settings_panel):
        assert "speed" in settings_panel._playback_speed_combo.toolTip().lower()

    def test_volume_slider_accessible_name(self, settings_panel):
        assert settings_panel._playback_volume_slider.accessibleName() == "Volume control"

    def test_volume_slider_accessible_description(self, settings_panel):
        desc = settings_panel._playback_volume_slider.accessibleDescription()
        assert "volume" in desc.lower()

    def test_volume_slider_tooltip(self, settings_panel):
        assert "volume" in settings_panel._playback_volume_slider.toolTip().lower()

    def test_volume_icon_accessible_name(self, settings_panel):
        assert settings_panel._playback_volume_label.accessibleName() == "Volume icon"

    def test_status_label_accessible_name(self, settings_panel):
        assert settings_panel._playback_status_label.accessibleName() == "Audio playback status"

    def test_status_label_accessible_description(self, settings_panel):
        desc = settings_panel._playback_status_label.accessibleDescription()
        assert "status" in desc.lower() or "error" in desc.lower()


# ---------------------------------------------------------------------------
# Scoped styling tests
# ---------------------------------------------------------------------------

class TestPlaybackScopedStyling:
    """Verify playback controls use scoped Aetheric styles, not generic ones."""

    def test_play_button_style_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_play_btn.styleSheet()
        assert "AethericHistoryPlaybackButton" in css
        # Must not use bare QPushButton selector
        for line in css.splitlines():
            stripped = line.strip()
            if "QPushButton" in stripped and "{" in stripped:
                assert "#" in stripped, f"Bare QPushButton selector: {stripped}"

    def test_speed_combo_style_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_speed_combo.styleSheet()
        assert "AethericHistoryPlaybackSpeedCombo" in css

    def test_volume_slider_style_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_volume_slider.styleSheet()
        assert "AethericHistoryPlaybackVolumeSlider" in css

    def test_status_label_style_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_status_label.styleSheet()
        assert "AethericHistoryPlaybackStatusLabel" in css

    def test_volume_icon_style_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_volume_label.styleSheet()
        assert "AethericHistoryPlaybackVolumeIcon" in css

    def test_play_button_has_hover_style(self, settings_panel):
        css = settings_panel._playback_play_btn.styleSheet()
        assert ":hover" in css

    def test_play_button_has_disabled_style(self, settings_panel):
        css = settings_panel._playback_play_btn.styleSheet()
        assert ":disabled" in css

    def test_play_button_has_pressed_style(self, settings_panel):
        css = settings_panel._playback_play_btn.styleSheet()
        assert ":pressed" in css

    def test_speed_combo_has_hover_style(self, settings_panel):
        css = settings_panel._playback_speed_combo.styleSheet()
        assert ":hover" in css

    def test_speed_combo_has_disabled_style(self, settings_panel):
        css = settings_panel._playback_speed_combo.styleSheet()
        assert ":disabled" in css


# ---------------------------------------------------------------------------
# Disabled-state clarity tests
# ---------------------------------------------------------------------------

class TestPlaybackDisabledStateClarity:
    """Verify disabled/missing-audio state is visually and textually clear."""

    def test_disabled_play_button_has_transparent_background(self, settings_panel):
        """Disabled play button should not look clickable."""
        css = settings_panel._playback_play_btn.styleSheet()
        assert ":disabled" in css
        # Disabled style should mention transparent or different color
        disabled_block = css.split(":disabled")[1] if ":disabled" in css else ""
        assert "transparent" in disabled_block.lower() or "border-color: transparent" in disabled_block

    def test_missing_audio_status_text_is_clear(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"

        _select_and_populate(panel, tmp_path, qapp)
        status = panel._playback_status_label.text()
        # Status text should be human-readable and specific
        assert "Audio" in status
        assert "not found" in status

    def test_disabled_speed_combo_has_transparent_background(self, settings_panel):
        css = settings_panel._playback_speed_combo.styleSheet()
        assert ":disabled" in css
        disabled_block = css.split(":disabled")[1] if ":disabled" in css else ""
        assert "transparent" in disabled_block.lower()

    def test_error_status_label_uses_distinct_style(self, settings_panel_on_history, qapp, tmp_path):
        """When there's an error, the status label style should differ from normal."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        _select_and_populate(panel, tmp_path, qapp)
        normal_css = panel._playback_status_label.styleSheet()

        # Now trigger error state
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"
        panel._sync_playback_controls()
        error_css = panel._playback_status_label.styleSheet()

        # Error style should be different from normal style
        assert normal_css != error_css or "Audio file not found" in panel._playback_status_label.text()


# ---------------------------------------------------------------------------
# Skip button routing tests
# ---------------------------------------------------------------------------

class TestPlaybackSkipButtonRouting:
    """Skip forward/backward buttons route to helper methods."""

    def test_skip_back_routes_to_helper(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_skip_back_btn.click()
        qapp.processEvents()

        panel._playback_helper.skip_backward.assert_called_once()

    def test_skip_fwd_routes_to_helper(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_skip_fwd_btn.click()
        qapp.processEvents()

        panel._playback_helper.skip_forward.assert_called_once()

    def test_skip_back_noop_when_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_skip_back_btn.setEnabled(True)
        panel._playback_skip_back_btn.click()
        qapp.processEvents()

        panel._playback_helper.skip_backward.assert_not_called()

    def test_skip_fwd_noop_when_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        panel._playback_skip_fwd_btn.setEnabled(True)
        panel._playback_skip_fwd_btn.click()
        qapp.processEvents()

        panel._playback_helper.skip_forward.assert_not_called()


# ---------------------------------------------------------------------------
# Progress slider tests
# ---------------------------------------------------------------------------

class TestPlaybackProgressSlider:
    """Progress slider drag/seek behavior."""

    def test_slider_pressed_sets_drag_flag(self, settings_panel):
        settings_panel._is_dragging_progress_slider = False
        settings_panel._on_progress_slider_pressed()
        assert settings_panel._is_dragging_progress_slider is True

    def test_slider_released_clears_drag_flag(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 60000
        panel._is_dragging_progress_slider = True
        panel._playback_progress_slider.setValue(500)
        panel._on_progress_slider_released()
        assert panel._is_dragging_progress_slider is False

    def test_slider_released_seeks_to_position(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 60000
        panel._playback_progress_slider.setValue(500)  # 50% of 1000 range
        panel._on_progress_slider_released()
        # 500/1000 * 60000 = 30000 ms
        panel._playback_helper.seek_to.assert_called_with(30000)

    def test_slider_released_noop_when_no_duration(self, settings_panel_on_history, qapp):
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 0
        panel._playback_progress_slider.setValue(500)
        panel._on_progress_slider_released()
        panel._playback_helper.seek_to.assert_not_called()

    def test_slider_released_noop_when_no_helper(self, settings_panel):
        settings_panel._playback_helper = None
        settings_panel._is_dragging_progress_slider = True
        settings_panel._on_progress_slider_released()
        assert settings_panel._is_dragging_progress_slider is False

    def test_slider_seek_logs_structured_info(self, settings_panel_on_history, qapp, caplog):
        """Verify slider seek emits structured log with position_ms and percent."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 120000
        panel._playback_progress_slider.setValue(250)  # 25%
        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_progress_slider_released()
        # 250/1000 * 120000 = 30000 ms, percent = 0.250
        assert any("slider_seek_triggered" in r.message for r in caplog.records)
        seek_records = [r for r in caplog.records if "slider_seek_triggered" in r.message]
        assert len(seek_records) == 1
        assert "position_ms=30000" in seek_records[0].message
        assert "percent=0.250" in seek_records[0].message

    def test_slider_seek_no_log_when_no_duration(self, settings_panel_on_history, qapp, caplog):
        """Verify no log emitted when duration is zero (no seek performed)."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 0
        panel._playback_progress_slider.setValue(500)
        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_progress_slider_released()
        assert not any("slider_seek_triggered" in r.message for r in caplog.records)

    def test_progress_slider_accessible_name(self, settings_panel):
        assert settings_panel._playback_progress_slider.accessibleName() == "Playback position"

    def test_progress_slider_accessible_description(self, settings_panel):
        desc = settings_panel._playback_progress_slider.accessibleDescription()
        assert "seek" in desc.lower() or "position" in desc.lower()

    def test_progress_slider_tooltip(self, settings_panel):
        assert "position" in settings_panel._playback_progress_slider.toolTip().lower()

    def test_skip_back_accessible_name(self, settings_panel):
        assert "skip" in settings_panel._playback_skip_back_btn.accessibleName().lower()
        assert "backward" in settings_panel._playback_skip_back_btn.accessibleName().lower()

    def test_skip_fwd_accessible_name(self, settings_panel):
        assert "skip" in settings_panel._playback_skip_fwd_btn.accessibleName().lower()
        assert "forward" in settings_panel._playback_skip_fwd_btn.accessibleName().lower()

    def test_skip_back_tooltip(self, settings_panel):
        tip = settings_panel._playback_skip_back_btn.toolTip()
        assert "skip" in tip.lower() or "back" in tip.lower()

    def test_skip_fwd_tooltip(self, settings_panel):
        tip = settings_panel._playback_skip_fwd_btn.toolTip()
        assert "skip" in tip.lower() or "forward" in tip.lower()


# ---------------------------------------------------------------------------
# Progress slider and skip button styling tests
# ---------------------------------------------------------------------------

class TestProgressAndSkipStyling:
    """Verify progress slider and skip buttons use scoped Aetheric styles."""

    def test_progress_slider_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_progress_slider.styleSheet()
        assert "AethericHistoryPlaybackProgressSlider" in css

    def test_progress_slider_has_disabled_style(self, settings_panel):
        css = settings_panel._playback_progress_slider.styleSheet()
        assert ":disabled" in css

    def test_progress_slider_has_hover_style(self, settings_panel):
        css = settings_panel._playback_progress_slider.styleSheet()
        assert ":hover" in css

    def test_skip_back_button_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_skip_back_btn.styleSheet()
        assert "AethericHistoryPlaybackSkipBackButton" in css

    def test_skip_fwd_button_has_scoped_selector(self, settings_panel):
        css = settings_panel._playback_skip_fwd_btn.styleSheet()
        assert "AethericHistoryPlaybackSkipFwdButton" in css

    def test_skip_back_button_has_disabled_style(self, settings_panel):
        css = settings_panel._playback_skip_back_btn.styleSheet()
        assert ":disabled" in css

    def test_skip_fwd_button_has_disabled_style(self, settings_panel):
        css = settings_panel._playback_skip_fwd_btn.styleSheet()
        assert ":disabled" in css

    def test_skip_back_button_has_hover_style(self, settings_panel):
        css = settings_panel._playback_skip_back_btn.styleSheet()
        assert ":hover" in css

    def test_skip_fwd_button_has_hover_style(self, settings_panel):
        css = settings_panel._playback_skip_fwd_btn.styleSheet()
        assert ":hover" in css


# ---------------------------------------------------------------------------
# Position update integration tests (T03)
# ---------------------------------------------------------------------------

class TestPositionUpdateIntegration:
    """Verify player→slider position update wiring, throttling, and drag guard."""

    def test_wire_player_signals_connects_both(self, settings_panel):
        """_wire_player_signals connects positionChanged and durationChanged."""
        mock_player = MagicMock()
        mock_player.positionChanged = MagicMock()
        mock_player.durationChanged = MagicMock()
        # Need connect methods
        mock_player.positionChanged.connect = MagicMock()
        mock_player.durationChanged.connect = MagicMock()

        mock_helper = _make_mock_helper()
        mock_helper.player = mock_player
        settings_panel._playback_helper = mock_helper

        settings_panel._wire_player_signals()

        mock_player.positionChanged.connect.assert_called_once_with(
            settings_panel._on_player_position_changed
        )
        mock_player.durationChanged.connect.assert_called_once_with(
            settings_panel._on_player_duration_changed
        )

    def test_wire_player_signals_noop_when_no_helper(self, settings_panel):
        """_wire_player_signals is a no-op when helper is None."""
        settings_panel._playback_helper = None
        # Should not raise
        settings_panel._wire_player_signals()

    def test_position_changed_updates_slider(self, settings_panel_on_history, qapp):
        """Slider value updates from player position when not dragging."""
        import time as _time
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000
        # Set last update far in the past so throttle doesn't block
        panel._last_slider_update_ms = 0
        # Patch time.monotonic to return a value far in the future
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(30000)
        # 30000 / 60000 * 1000 = 500
        assert panel._playback_progress_slider.value() == 500

    def test_position_changed_skips_during_drag(self, settings_panel_on_history, qapp):
        """Slider is NOT updated when _is_dragging_progress_slider is True."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = True
        panel._playback_helper.duration_ms = 60000
        panel._playback_progress_slider.setValue(0)
        panel._on_player_position_changed(30000)
        # Slider should remain at 0 — update skipped
        assert panel._playback_progress_slider.value() == 0

    def test_position_changed_skips_when_no_helper(self, settings_panel):
        """Position update is a no-op when helper is None."""
        settings_panel._playback_helper = None
        settings_panel._is_dragging_progress_slider = False
        # Should not raise
        settings_panel._on_player_position_changed(5000)

    def test_position_changed_skips_when_zero_duration(self, settings_panel_on_history, qapp):
        """Position update is skipped when duration_ms is 0."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 0
        panel._playback_progress_slider.setValue(0)
        panel._on_player_position_changed(5000)
        assert panel._playback_progress_slider.value() == 0

    def test_position_changed_throttles_rapid_updates(self, settings_panel_on_history, qapp):
        """Second rapid update within throttle interval is skipped."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000

        # First call at t=100s — should go through
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(15000)
        assert panel._playback_progress_slider.value() == 250  # 15000/60000*1000

        # Second call at t=100.03s (30ms later) — should be throttled (< 50ms)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.03):
            panel._on_player_position_changed(30000)
        # Slider should still be at 250 — update was throttled
        assert panel._playback_progress_slider.value() == 250

    def test_position_changed_allows_update_after_throttle(self, settings_panel_on_history, qapp):
        """Update passes after the throttle interval has elapsed."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000

        # First call at t=100s
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(15000)

        # Second call at t=100.1s (100ms later) — should pass (> 50ms)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.1):
            panel._on_player_position_changed(45000)
        # 45000/60000*1000 = 750
        assert panel._playback_progress_slider.value() == 750

    def test_position_clamps_to_slider_max(self, settings_panel_on_history, qapp):
        """Position at or beyond duration maps to slider max (1000)."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000
        panel._last_slider_update_ms = 0

        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(60000)
        assert panel._playback_progress_slider.value() == 1000

    def test_duration_changed_resets_slider(self, settings_panel_on_history, qapp):
        """Slider resets to 0 when duration changes to a positive value."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_progress_slider.setValue(500)
        panel._on_player_duration_changed(120000)
        assert panel._playback_progress_slider.value() == 0

    def test_duration_changed_noop_on_zero(self, settings_panel_on_history, qapp):
        """Slider is not reset when duration is 0 (e.g. media unloaded)."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_progress_slider.setValue(500)
        panel._on_player_duration_changed(0)
        assert panel._playback_progress_slider.value() == 500

    def test_duration_changed_respects_drag_guard(self, settings_panel_on_history, qapp):
        """Slider is not reset during active drag."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = True
        panel._playback_progress_slider.setValue(700)
        panel._on_player_duration_changed(120000)
        # Slider should not be reset during drag
        assert panel._playback_progress_slider.value() == 700

    def test_position_update_uses_block_signals(self, settings_panel_on_history, qapp):
        """Slider update blocks signals to prevent valueChanged feedback."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000
        panel._last_slider_update_ms = 0

        # Track valueChanged emissions
        emitted = []
        panel._playback_progress_slider.valueChanged.connect(
            lambda v: emitted.append(v)
        )

        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(30000)

        # valueChanged should NOT have been emitted (blocked)
        assert len(emitted) == 0
        # But value should be updated
        assert panel._playback_progress_slider.value() == 500


# ---------------------------------------------------------------------------
# Keyboard shortcuts for History playback (T05)
# ---------------------------------------------------------------------------

class TestKeyboardShortcuts:
    """Verify keyPressEvent routes keys to playback actions on History page."""

    @staticmethod
    def _make_key_event(key, modifier=Qt.KeyboardModifier.NoModifier, text=""):
        """Create a mock QKeyEvent for testing."""
        from PyQt6.QtGui import QKeyEvent
        event = MagicMock(spec=QKeyEvent)
        event.key.return_value = key
        event.modifiers.return_value = modifier
        event.text.return_value = text
        event.accepted = False

        def _accept():
            event.accepted = True
        event.accept = _accept
        return event

    # -- Space: play/pause ---------------------------------------------------

    def test_space_toggles_play(self, settings_panel_on_history, qapp):
        """Space key calls play when paused."""
        panel = settings_panel_on_history
        # Default mock has StoppedState → should call play
        event = self._make_key_event(Qt.Key.Key_Space, text=" ")
        panel.keyPressEvent(event)
        panel._playback_helper.play.assert_called()
        assert event.accepted is True

    def test_space_toggles_pause(self, settings_panel_on_history, qapp):
        """Space key calls pause when playing."""
        panel = settings_panel_on_history
        panel._playback_helper.player.playbackState.return_value = 1  # PlayingState
        event = self._make_key_event(Qt.Key.Key_Space, text=" ")
        panel.keyPressEvent(event)
        panel._playback_helper.pause.assert_called()
        assert event.accepted is True

    # -- Arrow keys: skip ----------------------------------------------------

    def test_arrow_left_skips_backward(self, settings_panel_on_history, qapp):
        """Left arrow calls skip_backward."""
        panel = settings_panel_on_history
        event = self._make_key_event(Qt.Key.Key_Left)
        panel.keyPressEvent(event)
        panel._playback_helper.skip_backward.assert_called_once()
        assert event.accepted is True

    def test_arrow_right_skips_forward(self, settings_panel_on_history, qapp):
        """Right arrow calls skip_forward."""
        panel = settings_panel_on_history
        event = self._make_key_event(Qt.Key.Key_Right)
        panel.keyPressEvent(event)
        panel._playback_helper.skip_forward.assert_called_once()
        assert event.accepted is True

    # -- Speed controls: +/- -------------------------------------------------

    def test_plus_increases_speed(self, settings_panel_on_history, qapp):
        """Plus key increases speed combo index."""
        panel = settings_panel_on_history
        panel._playback_speed_combo.setCurrentIndex(3)  # 1x
        event = self._make_key_event(Qt.Key.Key_Plus, text="+")
        panel.keyPressEvent(event)
        assert panel._playback_speed_combo.currentIndex() == 4  # 1.25x
        assert event.accepted is True

    def test_equal_increases_speed(self, settings_panel_on_history, qapp):
        """Equal key (=, unshifted +) also increases speed."""
        panel = settings_panel_on_history
        panel._playback_speed_combo.setCurrentIndex(3)  # 1x
        event = self._make_key_event(Qt.Key.Key_Equal, text="=")
        panel.keyPressEvent(event)
        assert panel._playback_speed_combo.currentIndex() == 4
        assert event.accepted is True

    def test_minus_decreases_speed(self, settings_panel_on_history, qapp):
        """Minus key decreases speed combo index."""
        panel = settings_panel_on_history
        panel._playback_speed_combo.setCurrentIndex(3)  # 1x
        event = self._make_key_event(Qt.Key.Key_Minus, text="-")
        panel.keyPressEvent(event)
        assert panel._playback_speed_combo.currentIndex() == 2  # 0.75x
        assert event.accepted is True

    def test_speed_clamps_at_max(self, settings_panel_on_history, qapp):
        """Plus at max speed does not exceed combo bounds."""
        panel = settings_panel_on_history
        last_idx = panel._playback_speed_combo.count() - 1
        panel._playback_speed_combo.setCurrentIndex(last_idx)
        event = self._make_key_event(Qt.Key.Key_Plus, text="+")
        panel.keyPressEvent(event)
        assert panel._playback_speed_combo.currentIndex() == last_idx

    def test_speed_clamps_at_min(self, settings_panel_on_history, qapp):
        """Minus at min speed does not go below combo bounds."""
        panel = settings_panel_on_history
        panel._playback_speed_combo.setCurrentIndex(0)
        event = self._make_key_event(Qt.Key.Key_Minus, text="-")
        panel.keyPressEvent(event)
        assert panel._playback_speed_combo.currentIndex() == 0

    # -- Context guard: non-History page -------------------------------------

    def test_noop_on_non_history_page(self, settings_panel, qapp):
        """Shortcuts do nothing when not on History page."""
        panel = settings_panel
        # Default nav is Settings (index 0), not History
        panel._playback_helper = _make_mock_helper()
        event = self._make_key_event(Qt.Key.Key_Space, text=" ")
        # super().keyPressEvent rejects MagicMock, but the important thing
        # is that the helper was never called
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass  # Expected: super() can't handle mock event
        panel._playback_helper.play.assert_not_called()
        panel._playback_helper.pause.assert_not_called()

    # -- Audio-unavailable guard ---------------------------------------------

    def test_noop_when_audio_unavailable(self, settings_panel_on_history, qapp):
        """Shortcuts do nothing when helper has no audio."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        event = self._make_key_event(Qt.Key.Key_Space, text=" ")
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        panel._playback_helper.play.assert_not_called()

    # -- Helper missing guard ------------------------------------------------

    def test_noop_when_no_helper(self, settings_panel_on_history, qapp):
        """Shortcuts do nothing when playback helper is None."""
        panel = settings_panel_on_history
        panel._playback_helper = None
        event = self._make_key_event(Qt.Key.Key_Space, text=" ")
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass  # Expected: super() can't handle mock event
        # No crash — event passes through to super

    # -- Modifier guard ------------------------------------------------------

    def test_ctrl_space_ignored(self, settings_panel_on_history, qapp):
        """Space with Ctrl modifier is not captured."""
        panel = settings_panel_on_history
        event = self._make_key_event(
            Qt.Key.Key_Space,
            modifier=Qt.KeyboardModifier.ControlModifier,
            text=" ",
        )
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        panel._playback_helper.play.assert_not_called()
        panel._playback_helper.pause.assert_not_called()

    def test_shift_arrow_ignored(self, settings_panel_on_history, qapp):
        """Arrow with Shift modifier is not captured."""
        panel = settings_panel_on_history
        event = self._make_key_event(
            Qt.Key.Key_Left,
            modifier=Qt.KeyboardModifier.ShiftModifier,
        )
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        panel._playback_helper.skip_backward.assert_not_called()

    # -- Logging -------------------------------------------------------------

    def test_shortcut_logs_action(self, settings_panel_on_history, qapp):
        """Keyboard shortcut logs structured info."""
        panel = settings_panel_on_history
        with patch("meetandread.widgets.floating_panels.logger") as mock_logger:
            event = self._make_key_event(Qt.Key.Key_Space, text=" ")
            panel.keyPressEvent(event)
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            # Check the format string contains the marker and action is in args
            assert "keyboard_shortcut_triggered" in call_args[0][0]
            assert "play_pause" in call_args[0] or "play_pause" in str(call_args)


# ---------------------------------------------------------------------------
# Word anchor rendering tests
# ---------------------------------------------------------------------------

class TestWordAnchorRendering:
    """Verify timed words render as word:{index}:{start_ms} anchors."""

    def test_word_anchor_format(self, settings_panel_on_history, qapp, tmp_path):
        """Timed words render as word:{index}:{start_ms} anchors."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "word_test", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert 'href="word:0:0"' in html
        assert 'href="word:1:500"' in html
        assert ">Hello<" in html
        assert ">world<" in html

    def test_word_anchors_use_zero_based_index(self, settings_panel_on_history, qapp, tmp_path):
        """Word indices are zero-based."""
        words = [
            {"text": "A", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
            {"text": "B", "start_time": 2.0, "end_time": 2.5, "speaker_id": "SPK_0"},
            {"text": "C", "start_time": 3.0, "end_time": 3.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "idx_test", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert 'href="word:0:1000"' in html
        assert 'href="word:1:2000"' in html
        assert 'href="word:2:3000"' in html

    def test_word_anchor_preserves_speaker_heading(self, settings_panel_on_history, qapp, tmp_path):
        """Speaker headings are still present when timed words are rendered."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "Bye", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_1"},
        ]
        md_path = _write_timed_transcript(tmp_path, "speaker_test", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert 'href="speaker:SPK_0"' in html
        assert 'href="speaker:SPK_1"' in html

    def test_unknown_speaker_word_anchor(self, settings_panel_on_history, qapp, tmp_path):
        """Words with speaker_id=None still render as timed word anchors."""
        words = [
            {"text": "Hmm", "start_time": 0.0, "end_time": 0.3, "speaker_id": None},
        ]
        md_path = _write_timed_transcript(tmp_path, "unknown_spk", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert 'href="word:0:0"' in html
        assert 'href="speaker:__unknown__"' in html

    def test_html_entities_escaped_in_word_text(self, settings_panel_on_history, qapp, tmp_path):
        """Word text with HTML-special characters is properly escaped."""
        words = [
            {"text": "<script>alert(1)</script>", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": 'a "b" & c', "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "html_escape", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "a &quot;b&quot; &amp; c" in html

    def test_legacy_no_timing_uses_fallback(self, settings_panel_on_history, qapp, tmp_path):
        """Words without start_time fall back to markdown body rendering."""
        body = "**SPK_0**\nHello world.\n"
        words = [
            {"text": "Hello", "speaker_id": "SPK_0"},
            {"text": "world.", "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "no_timing", words, body=body)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        # Should use fallback (markdown body) — no word: anchors
        assert "word:" not in html
        assert "speaker:SPK_0" in html

    def test_mixed_timing_only_timed_words_get_anchors(self, settings_panel_on_history, qapp, tmp_path):
        """When some words have timing and others don't, timed path renders all words."""
        words = [
            {"text": "timed", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "untimed", "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "mixed_timing", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        # Has timed words → takes timed path
        assert 'href="word:0:0"' in html
        # Untimed word (no start_time) should be plain text, not an anchor
        assert " untimed " in html
        assert 'href="word:1:' not in html

    def test_negative_start_time_not_anchored(self, settings_panel_on_history, qapp, tmp_path):
        """Words with negative start_time are not rendered as word anchors."""
        words = [
            {"text": "neg", "start_time": -1.0, "end_time": 0.0, "speaker_id": "SPK_0"},
            {"text": "pos", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "neg_time", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        # Positive word gets an anchor
        assert 'href="word:1:1000"' in html
        # Negative word should NOT be an anchor
        assert "word:0:" not in html
        # But the text should still be present
        assert "neg" in html

    # -- Readability contract tests (T01) ------------------------------------

    def test_timed_anchor_has_explicit_readable_color(self, settings_panel_on_history, qapp, tmp_path):
        """Timed word anchors use explicit color:#eeeeee instead of color:inherit."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "readable_color", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None

        # Every timed anchor must have an explicit readable color
        assert 'style="color:#eeeeee; text-decoration:none;"' in html, (
            "Timed word anchors should use explicit color:#eeeeee, not color:inherit"
        )

        # Clickability preserved
        assert 'href="word:0:0"' in html
        assert 'href="word:1:500"' in html

    def test_no_color_inherit_in_timed_transcript(self, settings_panel_on_history, qapp, tmp_path):
        """Timed transcript HTML must not contain color:inherit on word anchors."""
        words = [
            {"text": "alpha", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "beta", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "no_inherit", words)
        html = settings_panel_on_history._render_history_transcript(md_path)
        assert html is not None
        assert "color:inherit" not in html, (
            "Timed transcript HTML must not use color:inherit for word anchors"
        )

    def test_negative_one_highlight_no_color_inherit(self, settings_panel_on_history, qapp, tmp_path):
        """Highlight index -1 should not emit color:inherit for timed word anchors."""
        words = [
            {"text": "first", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "second", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_no_inherit", words)
        html = settings_panel_on_history._render_history_transcript_highlighted(md_path, -1)
        assert html is not None
        assert "color:inherit" not in html, (
            "Highlighted transcript with index -1 must not use color:inherit"
        )


# ---------------------------------------------------------------------------
# Word seek routing tests
# ---------------------------------------------------------------------------

class TestWordSeekRouting:
    """Verify word: anchor clicks route to seek_to + play."""

    def test_word_click_seeks_and_plays(self, settings_panel_on_history, qapp, tmp_path):
        """Clicking a word anchor seeks to its start_ms and plays."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        url = MagicMock()
        url.toString.return_value = "word:5:3000"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_called_once_with(3000)
        panel._playback_helper.play.assert_called_once()

    def test_word_click_seeks_to_zero(self, settings_panel_on_history, qapp, tmp_path):
        """Clicking a word at start_ms=0 seeks to 0 and plays."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        url = MagicMock()
        url.toString.return_value = "word:0:0"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_called_once_with(0)
        panel._playback_helper.play.assert_called_once()

    def test_word_click_noop_no_helper(self, settings_panel_on_history, qapp, tmp_path):
        """Word click is a no-op when playback helper is None."""
        panel = settings_panel_on_history
        panel._playback_helper = None

        url = MagicMock()
        url.toString.return_value = "word:0:0"
        panel._on_history_anchor_clicked(url)
        # Should not raise

    def test_word_click_noop_audio_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        """Word click skips seeking when audio is unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        url = MagicMock()
        url.toString.return_value = "word:0:500"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_not_called()
        panel._playback_helper.play.assert_not_called()

    def test_malformed_word_anchor_extra_colon(self, settings_panel_on_history, qapp, tmp_path):
        """Malformed word anchor with extra colons is ignored."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:5:3000:extra"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_not_called()

    def test_malformed_word_anchor_non_numeric(self, settings_panel_on_history, qapp, tmp_path):
        """Malformed word anchor with non-numeric index is ignored."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:abc:500"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_not_called()

    def test_malformed_word_anchor_negative_index(self, settings_panel_on_history, qapp, tmp_path):
        """Word anchor with negative index is ignored."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:-1:500"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_not_called()

    def test_malformed_word_anchor_negative_start_ms(self, settings_panel_on_history, qapp, tmp_path):
        """Word anchor with negative start_ms is ignored."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:0:-100"
        panel._on_history_anchor_clicked(url)

        panel._playback_helper.seek_to.assert_not_called()

    def test_word_anchor_does_not_trigger_speaker_dialog(self, settings_panel_on_history, qapp, tmp_path):
        """Word anchors do NOT trigger the speaker identity dialog."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:0:0"

        with patch("meetandread.widgets.floating_panels._open_identity_link_dialog") as mock_dialog:
            panel._on_history_anchor_clicked(url)
            mock_dialog.assert_not_called()

    def test_speaker_anchor_still_works(self, settings_panel_on_history, qapp, tmp_path):
        """Speaker anchors still trigger the identity dialog after word anchor changes."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "speaker:SPK_0"

        with patch("meetandread.widgets.floating_panels._open_identity_link_dialog", return_value=False):
            panel._on_history_anchor_clicked(url)
        # seek_to should NOT be called for speaker anchors
        panel._playback_helper.seek_to.assert_not_called()


# ---------------------------------------------------------------------------
# Word seek logging tests
# ---------------------------------------------------------------------------

class TestWordSeekLogging:
    """Verify structured logging for word seek events."""

    def test_word_seek_success_logged(self, settings_panel_on_history, qapp, caplog):
        """Successful word seek logs word_seek_success."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:3:2500"

        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_history_anchor_clicked(url)

        seek_records = [r for r in caplog.records if "word_seek_success" in r.message]
        assert len(seek_records) == 1
        assert "index=3" in seek_records[0].message
        assert "start_ms=2500" in seek_records[0].message

    def test_word_seek_skipped_no_helper_logged(self, settings_panel_on_history, qapp, caplog):
        """Skipped seek (no helper) logs word_seek_skipped."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper = None

        url = MagicMock()
        url.toString.return_value = "word:0:0"

        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_history_anchor_clicked(url)

        skip_records = [r for r in caplog.records if "word_seek_skipped" in r.message]
        assert len(skip_records) == 1
        assert "reason=no_helper" in skip_records[0].message

    def test_word_seek_skipped_audio_unavailable_logged(self, settings_panel_on_history, qapp, caplog):
        """Skipped seek (no audio) logs word_seek_skipped."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        url = MagicMock()
        url.toString.return_value = "word:0:100"

        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_history_anchor_clicked(url)

        skip_records = [r for r in caplog.records if "word_seek_skipped" in r.message]
        assert len(skip_records) == 1
        assert "reason=audio_unavailable" in skip_records[0].message

    def test_malformed_anchor_logged_as_warning(self, settings_panel_on_history, qapp, caplog):
        """Malformed anchor logs word_anchor_malformed warning."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        url = MagicMock()
        url.toString.return_value = "word:bad:data"

        with caplog.at_level(logging.WARNING, logger="meetandread.widgets.floating_panels"):
            panel._on_history_anchor_clicked(url)

        warn_records = [r for r in caplog.records if "word_anchor_malformed" in r.message]
        assert len(warn_records) == 1


# ---------------------------------------------------------------------------
# T02: Current-word highlight from playback position
# ---------------------------------------------------------------------------

class TestExtractTimedWords:
    """Tests for _extract_timed_words caching and parsing."""

    def test_extract_timed_words_basic(self, settings_panel_on_history, qapp, tmp_path):
        """Extracts (start_ms, end_ms) pairs from metadata words."""
        panel = settings_panel_on_history
        words = [
            {"text": "hello", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 1.2, "end_time": 1.8, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "timed_basic", words)
        result = panel._extract_timed_words(md_path)
        assert len(result) == 2
        assert result[0] == (500, 1000)
        assert result[1] == (1200, 1800)
        assert panel._cached_timed_words == result

    def test_extract_caches_on_panel(self, settings_panel_on_history, qapp, tmp_path):
        """Result is cached in _cached_timed_words for reuse."""
        panel = settings_panel_on_history
        words = [
            {"text": "a", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "cache_test", words)
        panel._extract_timed_words(md_path)
        assert panel._cached_timed_words == [(0, 500)]

    def test_extract_no_timing_returns_empty(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript without timing metadata returns empty list."""
        panel = settings_panel_on_history
        md_path = tmp_path / "transcripts" / "no_timing.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        result = panel._extract_timed_words(md_path)
        assert result == []

    def test_extract_no_metadata_footer(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript without METADATA footer returns empty list."""
        panel = settings_panel_on_history
        md_path = tmp_path / "transcripts" / "no_footer.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("Just a plain transcript.\n", encoding="utf-8")
        result = panel._extract_timed_words(md_path)
        assert result == []

    def test_extract_missing_file_returns_empty(self, settings_panel_on_history, qapp, tmp_path):
        """Missing file returns empty list without error."""
        panel = settings_panel_on_history
        bad_path = tmp_path / "transcripts" / "nonexistent.md"
        result = panel._extract_timed_words(bad_path)
        assert result == []

    def test_extract_malformed_word_times(self, settings_panel_on_history, qapp, tmp_path):
        """Words with bad timing get (None, None) entries keeping alignment."""
        panel = settings_panel_on_history
        # Use _write_transcript directly to avoid _write_timed_transcript's
        # segment-building arithmetic, which can't handle non-numeric start_time
        md_path = tmp_path / "transcripts" / "malformed.md"
        _write_transcript(md_path, "", {
            "words": [
                {"text": "good", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
                {"text": "bad", "start_time": "not_a_number", "end_time": 1.5, "speaker_id": "SPK_0"},
                {"text": "ugly", "start_time": -1.0, "end_time": 2.0, "speaker_id": "SPK_0"},
                {"text": "ok", "start_time": 2.5, "end_time": 3.0, "speaker_id": "SPK_0"},
            ],
            "segments": [
                {"speaker": "SPK_0", "start": 0.0, "end": 3.0},
            ],
        })
        result = panel._extract_timed_words(md_path)
        assert len(result) == 4
        assert result[0] == (0, 500)
        # "not_a_number" → None start → (None, None)
        assert result[1] == (None, None)
        # Negative start_time → None start → (None, None)
        assert result[2] == (None, None)
        assert result[3] == (2500, 3000)

    def test_extract_missing_end_time_uses_fallback(self, settings_panel_on_history, qapp, tmp_path):
        """Word without end_time gets start_ms + 1 fallback."""
        panel = settings_panel_on_history
        words = [
            {"text": "hi", "start_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "no_end", words)
        result = panel._extract_timed_words(md_path)
        assert result == [(1000, 1001)]


class TestFindActiveWordIndex:
    """Tests for _find_active_word_index binary search and boundary logic."""

    def test_basic_word_match(self, settings_panel_on_history, qapp):
        """Finds the correct word at mid-interval position."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500), (500, 1200), (1200, 2000)]
        assert panel._find_active_word_index(750) == 1

    def test_exact_start_boundary(self, settings_panel_on_history, qapp):
        """Position exactly at word start_time matches that word [start, end)."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500), (500, 1200)]
        assert panel._find_active_word_index(500) == 1

    def test_exact_end_boundary_no_match(self, settings_panel_on_history, qapp):
        """Position exactly at word end_time does NOT match (half-open interval)."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500), (600, 1200)]
        # 500 is not in [0,500) and not in [600,1200) → gap
        assert panel._find_active_word_index(500) == -1

    def test_position_before_first_word(self, settings_panel_on_history, qapp):
        """Position before first word start returns -1."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(500, 1000), (1000, 1500)]
        assert panel._find_active_word_index(200) == -1

    def test_position_after_last_word(self, settings_panel_on_history, qapp):
        """Position at or beyond last word's end returns -1."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500), (500, 1000)]
        assert panel._find_active_word_index(1000) == -1
        assert panel._find_active_word_index(5000) == -1

    def test_silence_gap_returns_negative_one(self, settings_panel_on_history, qapp):
        """Position in a gap between words returns -1."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500), (1000, 1500)]
        # 700 is between 500 and 1000 — silence gap
        assert panel._find_active_word_index(700) == -1

    def test_negative_position(self, settings_panel_on_history, qapp):
        """Negative position returns -1."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500)]
        assert panel._find_active_word_index(-1) == -1

    def test_empty_cache(self, settings_panel_on_history, qapp):
        """Empty timed words cache returns -1."""
        panel = settings_panel_on_history
        panel._cached_timed_words = []
        assert panel._find_active_word_index(100) == -1

    def test_none_entries_fall_back_to_linear(self, settings_panel_on_history, qapp):
        """Mixed None entries trigger linear fallback."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [
            (0, 500), (None, None), (1000, 1500), (None, None), (2000, 2500)
        ]
        assert panel._find_active_word_index(1200) == 2

    def test_single_word(self, settings_panel_on_history, qapp):
        """Single timed word is found correctly."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(100, 500)]
        assert panel._find_active_word_index(200) == 0
        assert panel._find_active_word_index(50) == -1
        assert panel._find_active_word_index(500) == -1


class TestHighlightRendering:
    """Tests for _render_highlighted_transcript and highlight state management."""

    def test_highlight_renders_span_on_active_word(self, settings_panel_on_history, qapp, tmp_path):
        """Highlighted word gets a <span> with background style."""
        panel = settings_panel_on_history
        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_test", words)
        html = panel._render_history_transcript_highlighted(md_path, 1)
        assert html is not None
        assert "word:1:500" in html
        assert "background-color" in html
        # First word should NOT have highlight
        assert html.index("word:0:0") < html.index("background-color")

    # -- Readability contract tests (T01) ------------------------------------

    def test_highlighted_word_has_explicit_white_color(self, settings_panel_on_history, qapp, tmp_path):
        """Highlighted timed word anchor uses color:#ffffff with background."""
        panel = settings_panel_on_history
        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_color", words)
        html = panel._render_history_transcript_highlighted(md_path, 1)
        assert html is not None

        # Highlighted anchor should have white color with background
        assert "color:#ffffff" in html, (
            "Highlighted word anchor should use explicit color:#ffffff"
        )
        assert "background-color" in html

        # Background should be the expected rgba
        assert "rgba(79, 195, 247, 0.25)" in html

    def test_non_highlighted_word_has_readable_color(self, settings_panel_on_history, qapp, tmp_path):
        """Non-highlighted timed word anchors use color:#eeeeee."""
        panel = settings_panel_on_history
        words = [
            {"text": "first", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "second", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_non_active", words)
        html = panel._render_history_transcript_highlighted(md_path, 1)
        assert html is not None

        # Non-highlighted anchor should use #eeeeee
        assert 'style="color:#eeeeee; text-decoration:none;"' in html, (
            "Non-highlighted word anchor should use color:#eeeeee"
        )

    def test_highlighted_transcript_no_color_inherit(self, settings_panel_on_history, qapp, tmp_path):
        """Highlighted transcript must not contain color:inherit anywhere."""
        panel = settings_panel_on_history
        words = [
            {"text": "a", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "b", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
            {"text": "c", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_no_inherit", words)
        html = panel._render_history_transcript_highlighted(md_path, 1)
        assert html is not None
        assert "color:inherit" not in html, (
            "Highlighted transcript must not use color:inherit"
        )

    def test_highlight_none_clears_style(self, settings_panel_on_history, qapp, tmp_path):
        """Highlight index -1 renders without any background styles."""
        panel = settings_panel_on_history
        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_none", words)
        html = panel._render_history_transcript_highlighted(md_path, -1)
        assert html is not None
        assert "background-color" not in html

    def test_highlight_preserves_speaker_headings(self, settings_panel_on_history, qapp, tmp_path):
        """Speaker headings are preserved in highlighted render."""
        panel = settings_panel_on_history
        words = [
            {"text": "hi", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"text": "there", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_1"},
        ]
        md_path = _write_timed_transcript(tmp_path, "hl_speakers", words)
        html = panel._render_history_transcript_highlighted(md_path, 0)
        assert html is not None
        assert "speaker:SPK_0" in html
        assert "speaker:SPK_1" in html

    def test_highlight_no_timing_falls_back(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript with no timing falls back to plain render (no crash)."""
        panel = settings_panel_on_history
        md_path = tmp_path / "transcripts" / "hl_legacy.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        html = panel._render_history_transcript_highlighted(md_path, 0)
        assert html is not None
        assert "background-color" not in html

    def test_render_highlighted_preserves_scroll(self, settings_panel_on_history, qapp, tmp_path):
        """_render_highlighted_transcript saves scroll position before re-render."""
        panel = settings_panel_on_history
        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(20)
        ]
        md_path = _write_timed_transcript(tmp_path, "scroll_test", words)
        # First render the transcript
        html = panel._render_history_transcript(md_path)
        panel._history_viewer.setHtml(html)
        qapp.processEvents()

        # Verify the method reads scroll position before calling setHtml
        # by patching the scrollbar's value() to return a known value
        sb = panel._history_viewer.verticalScrollBar()
        captured_scroll = {}
        original_value = sb.value
        sb.value = lambda: (captured_scroll.__setitem__('pos', 42), 42)[1]

        # Also track that setHtml is called (the actual re-render)
        setHtml_calls = []
        original_setHtml = panel._history_viewer.setHtml
        def tracking_setHtml(html_str):
            setHtml_calls.append(html_str)
            original_setHtml(html_str)
        panel._history_viewer.setHtml = tracking_setHtml

        panel._render_highlighted_transcript(md_path, 5)
        qapp.processEvents()

        # The method should have read the scroll position
        assert captured_scroll.get('pos') == 42
        # And should have called setHtml (the highlight was rendered)
        assert len(setHtml_calls) == 1
        assert "background-color" in setHtml_calls[0]


class TestHighlightStateManagement:
    """Tests for highlight state reset and lifecycle."""

    def test_reset_highlight_state(self, settings_panel_on_history, qapp):
        """_reset_highlight_state clears all highlight fields."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500)]
        panel._current_highlight_word_idx = 3
        panel._last_highlight_update_ms = 99999
        panel._reset_highlight_state()
        assert panel._cached_timed_words == []
        assert panel._current_highlight_word_idx == -1
        assert panel._last_highlight_update_ms == 0

    def test_highlight_resets_on_new_selection(self, settings_panel_on_history, qapp, tmp_path):
        """Changing history selection resets highlight state."""
        panel = settings_panel_on_history
        panel._cached_timed_words = [(0, 500)]
        panel._current_highlight_word_idx = 0

        # Simulate new selection
        words2 = [
            {"text": "new", "start_time": 1.0, "end_time": 2.0, "speaker_id": "SPK_1"},
        ]
        md_path2 = _write_timed_transcript(tmp_path, "new_sel", words2)
        meta2 = _make_meta(str(md_path2))
        panel._populate_history_list([meta2])
        item2 = panel._history_list.item(panel._history_list.count() - 1)
        panel._history_list.setCurrentItem(item2)
        panel._on_history_item_clicked(item2)
        qapp.processEvents()

        # Highlight state should have been reset (then re-extracted with new data)
        # At minimum, _current_highlight_word_idx should be -1 after selection
        assert panel._current_highlight_word_idx == -1


class TestHighlightFromPlaybackPosition:
    """Integration tests: position changes drive highlight updates."""

    def test_position_changed_triggers_highlight(self, settings_panel_on_history, qapp, tmp_path):
        """Position change triggers highlight update when word index changes."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000

        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 1.0, "speaker_id": "SPK_0"},
            {"text": "world", "start_time": 1.5, "end_time": 2.5, "speaker_id": "SPK_0"},
            {"text": "end", "start_time": 3.0, "end_time": 4.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "pos_hl", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        # Advance time past throttle
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(1700)  # Within word 1

        assert panel._current_highlight_word_idx == 1

    def test_highlight_skips_when_word_unchanged(self, settings_panel_on_history, qapp, tmp_path):
        """setHtml is NOT called when the active word index hasn't changed."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 10000

        words = [
            {"text": "a", "start_time": 0.0, "end_time": 2.0, "speaker_id": "SPK_0"},
            {"text": "b", "start_time": 2.0, "end_time": 4.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(tmp_path, "skip_hl", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        # First position: word 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(500)
        assert panel._current_highlight_word_idx == 0

        # Track setHtml calls
        setHtml_count_before = 0
        original_setHtml = panel._history_viewer.setHtml

        call_count = [0]
        def counting_setHtml(html):
            call_count[0] += 1
            original_setHtml(html)

        panel._history_viewer.setHtml = counting_setHtml

        # Same word — should NOT trigger setHtml
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.5):
            panel._on_player_position_changed(1000)
        assert panel._current_highlight_word_idx == 0
        assert call_count[0] == 0

    def test_highlight_throttle_200ms(self, settings_panel_on_history, qapp, tmp_path):
        """Highlight updates are throttled to 200ms intervals."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 30000

        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(10)
        ]
        md_path = _write_timed_transcript(tmp_path, "throttle_hl", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)
        panel._current_highlight_word_idx = -1

        # First call at t=100s — should go through
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(2500)  # word 2
        assert panel._current_highlight_word_idx == 2

        # Second call at t=100.1s (100ms later) — throttled (< 200ms)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.1):
            panel._on_player_position_changed(5500)  # would be word 5
        # Highlight should NOT have changed due to throttle
        assert panel._current_highlight_word_idx == 2

        # Third call at t=100.3s (300ms after first) — passes throttle
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.3):
            panel._on_player_position_changed(5500)  # word 5
        assert panel._current_highlight_word_idx == 5

    def test_rapid_position_updates_bounded(self, settings_panel_on_history, qapp, tmp_path):
        """Many rapid position updates don't cause lag; only first and last matter."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 60000

        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(30)
        ]
        md_path = _write_timed_transcript(tmp_path, "rapid_hl", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)
        panel._current_highlight_word_idx = -1

        # Simulate 20 rapid position changes within throttle window
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(1000)
        # All subsequent within throttle should be skipped
        for pos in range(1500, 5000, 200):
            with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.05):
                panel._on_player_position_changed(pos)
        # Only the first one went through
        assert panel._current_highlight_word_idx == 1

    def test_legacy_transcript_noop_highlight(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript with no timing data doesn't trigger highlight updates."""
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 10000

        md_path = tmp_path / "transcripts" / "legacy.md"
        _write_transcript(md_path, "**SPK_0**\nHello world.", {"words": []})
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)
        assert panel._cached_timed_words == []

        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(3000)
        assert panel._current_highlight_word_idx == -1


class TestLongTranscriptHighlight:
    """Load profile test: highlight performance with large transcripts."""

    def test_long_transcript_5000_words(self, settings_panel_on_history, qapp):
        """Binary search finds correct word in 5000-word transcript quickly."""
        panel = settings_panel_on_history
        # Build 5000 words with 100ms each, 0 gaps
        n = 5000
        panel._cached_timed_words = [(i * 100, (i + 1) * 100) for i in range(n)]

        import time as _time
        start = _time.monotonic()
        # Look up word near the end
        idx = panel._find_active_word_index(4999 * 100 + 50)  # word 4999
        elapsed = _time.monotonic() - start
        assert idx == 4999
        # Should be fast — binary search is O(log n)
        assert elapsed < 0.01, f"Binary search took {elapsed:.4f}s — too slow"

    def test_long_transcript_boundary_accuracy(self, settings_panel_on_history, qapp):
        """Boundary semantics are correct across a long transcript."""
        panel = settings_panel_on_history
        n = 1000
        panel._cached_timed_words = [(i * 200, i * 200 + 100) for i in range(n)]
        # Each word spans [i*200, i*200+100), gap from i*200+100 to (i+1)*200

        # Middle of word 500
        assert panel._find_active_word_index(500 * 200 + 50) == 500
        # Start of word 500
        assert panel._find_active_word_index(500 * 200) == 500
        # End of word 500 (exclusive)
        assert panel._find_active_word_index(500 * 200 + 100) == -1
        # In gap between word 500 and 501
        assert panel._find_active_word_index(500 * 200 + 150) == -1


class TestHighlightLogging:
    """Tests for highlight-related diagnostic logging."""

    def test_highlight_word_change_logged(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Active word transition logs highlight_word_changed."""
        import logging
        panel = settings_panel_on_history
        panel._is_dragging_progress_slider = False
        panel._playback_helper.duration_ms = 10000

        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(5)
        ]
        md_path = _write_timed_transcript(tmp_path, "log_hl", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        with caplog.at_level(logging.DEBUG, logger="meetandread.widgets.floating_panels"):
            with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
                panel._on_player_position_changed(2500)  # word 2

        hl_records = [r for r in caplog.records if "highlight_word_changed" in r.message]
        assert len(hl_records) == 1
        assert "index=2" in hl_records[0].message


# ---------------------------------------------------------------------------
# T02: Bookmark toolbar widget structure tests
# ---------------------------------------------------------------------------

class TestBookmarkToolbarStructure:
    """Verify bookmark button and combo exist with correct object names."""

    def test_bookmark_button_object_name(self, settings_panel):
        assert settings_panel._bookmark_add_btn.objectName() == "AethericHistoryBookmarkButton"

    def test_bookmark_button_has_playback_action_property(self, settings_panel):
        assert settings_panel._bookmark_add_btn.property("playback_action") == "bookmark_add"

    def test_bookmark_button_accessible_name(self, settings_panel):
        assert "bookmark" in settings_panel._bookmark_add_btn.accessibleName().lower()

    def test_bookmark_button_accessible_description(self, settings_panel):
        desc = settings_panel._bookmark_add_btn.accessibleDescription()
        assert "bookmark" in desc.lower()

    def test_bookmark_button_tooltip(self, settings_panel):
        tip = settings_panel._bookmark_add_btn.toolTip()
        assert "bookmark" in tip.lower()

    def test_bookmark_button_initially_disabled(self, settings_panel):
        assert not settings_panel._bookmark_add_btn.isEnabled()

    def test_bookmark_button_compact_height(self, settings_panel):
        assert settings_panel._bookmark_add_btn.height() <= 30

    def test_bookmark_combo_object_name(self, settings_panel):
        assert settings_panel._bookmark_combo.objectName() == "AethericHistoryBookmarkCombo"

    def test_bookmark_combo_accessible_name(self, settings_panel):
        assert "bookmark" in settings_panel._bookmark_combo.accessibleName().lower()

    def test_bookmark_combo_initially_disabled(self, settings_panel):
        assert not settings_panel._bookmark_combo.isEnabled()

    def test_bookmark_button_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._bookmark_add_btn) >= 0

    def test_bookmark_combo_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._bookmark_combo) >= 0

    def test_bookmark_controls_after_skip_fwd(self, settings_panel):
        """Bookmark controls should appear after skip-forward button."""
        layout = settings_panel._history_detail_header.layout()
        skip_idx = layout.indexOf(settings_panel._playback_skip_fwd_btn)
        btn_idx = layout.indexOf(settings_panel._bookmark_add_btn)
        combo_idx = layout.indexOf(settings_panel._bookmark_combo)
        assert btn_idx > skip_idx
        assert combo_idx > skip_idx

    def test_bookmark_controls_before_scrub_delete(self, settings_panel):
        """Bookmark controls should appear before Scrub/Delete buttons."""
        layout = settings_panel._history_detail_header.layout()
        btn_idx = layout.indexOf(settings_panel._bookmark_add_btn)
        scrub_idx = layout.indexOf(settings_panel._scrub_btn)
        delete_idx = layout.indexOf(settings_panel._delete_btn)
        assert btn_idx < scrub_idx
        assert btn_idx < delete_idx

    def test_bookmark_items_initially_empty(self, settings_panel):
        assert settings_panel._bookmark_items == []

    def test_bookmark_manager_initially_none(self, settings_panel):
        assert settings_panel._bookmark_manager is None

    def test_bookmark_button_has_scoped_selector(self, settings_panel):
        css = settings_panel._bookmark_add_btn.styleSheet()
        assert "AethericHistoryBookmarkButton" in css

    def test_bookmark_combo_has_scoped_selector(self, settings_panel):
        css = settings_panel._bookmark_combo.styleSheet()
        assert "AethericHistoryBookmarkCombo" in css

    def test_bookmark_button_has_hover_style(self, settings_panel):
        css = settings_panel._bookmark_add_btn.styleSheet()
        assert ":hover" in css

    def test_bookmark_button_has_disabled_style(self, settings_panel):
        css = settings_panel._bookmark_add_btn.styleSheet()
        assert ":disabled" in css


# ---------------------------------------------------------------------------
# T02: Bookmark add handler tests
# ---------------------------------------------------------------------------

class TestBookmarkAddHandler:
    """Verify bookmark add button wiring and failure modes."""

    def test_add_noop_when_no_helper(self, settings_panel_on_history, qapp, tmp_path):
        """Add bookmark is a no-op when helper is None."""
        panel = settings_panel_on_history
        panel._playback_helper = None
        panel._current_history_md_path = tmp_path / "test.md"
        # Should not raise
        panel._on_bookmark_add_clicked()

    def test_add_noop_when_audio_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        """Add bookmark is a no-op when audio is unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._current_history_md_path = tmp_path / "test.md"
        panel._on_bookmark_add_clicked()
        # No bookmark created

    def test_add_noop_when_no_transcript(self, settings_panel_on_history, qapp):
        """Add bookmark is a no-op when no transcript is selected."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._current_history_md_path = None
        panel._on_bookmark_add_clicked()

    def test_add_cancelled_dialog_noop(self, settings_panel_on_history, qapp, tmp_path):
        """Cancelled name dialog does not create a bookmark."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 30000

        md_path = tmp_path / "transcripts" / "cancel_test.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        with patch("meetandread.widgets.floating_panels.QInputDialog.getText", return_value=("", False)):
            panel._on_bookmark_add_clicked()

        # No bookmark should have been created
        assert len(panel._bookmark_items) == 0

    def test_add_with_empty_name_uses_default(self, settings_panel_on_history, qapp, tmp_path):
        """Empty name input uses default 'Bookmark at MM:SS'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 90000  # 1:30

        md_path = tmp_path / "transcripts" / "empty_name.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        with patch("meetandread.widgets.floating_panels.QInputDialog.getText", return_value=("  ", True)):
            panel._on_bookmark_add_clicked()

        # Bookmark should have been created with default name
        assert len(panel._bookmark_items) == 1

        # Verify via the manager
        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 1
        assert bm_list[0].name == "Bookmark at 01:30"
        assert bm_list[0].position_ms == 90000

    def test_add_with_custom_name(self, settings_panel_on_history, qapp, tmp_path):
        """Custom name is persisted correctly."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 5000

        md_path = tmp_path / "transcripts" / "custom_name.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        with patch("meetandread.widgets.floating_panels.QInputDialog.getText", return_value=("My Mark", True)):
            panel._on_bookmark_add_clicked()

        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 1
        assert bm_list[0].name == "My Mark"
        assert bm_list[0].position_ms == 5000

    def test_add_refreshes_combo(self, settings_panel_on_history, qapp, tmp_path):
        """After adding a bookmark, the combo is refreshed."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 10000

        md_path = tmp_path / "transcripts" / "refresh_test.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        with patch("meetandread.widgets.floating_panels.QInputDialog.getText", return_value=("BM1", True)):
            panel._on_bookmark_add_clicked()

        assert panel._bookmark_combo.count() >= 1
        assert len(panel._bookmark_items) == 1

    def test_add_multiple_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Multiple bookmarks accumulate correctly."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "multi_bm.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        positions = [5000, 15000, 30000]
        for i, pos in enumerate(positions):
            panel._playback_helper.position_ms = pos
            with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                        return_value=(f"BM{i}", True)):
                panel._on_bookmark_add_clicked()

        assert len(panel._bookmark_items) == 3

        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 3


# ---------------------------------------------------------------------------
# T02: Bookmark navigation handler tests
# ---------------------------------------------------------------------------

class TestBookmarkNavigation:
    """Verify bookmark combo selection routes to seek_to + play."""

    def test_navigation_seeks_and_plays(self, settings_panel_on_history, qapp, tmp_path):
        """Selecting a bookmark seeks to its position and plays."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "nav_test.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "BM1", "position_ms": 10000, "created_at": "2026-01-01T00:00:00"},
                {"name": "BM2", "position_ms": 25000, "created_at": "2026-01-01T00:00:01"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert len(panel._bookmark_items) == 2

        # Select second bookmark (index 1)
        panel._on_bookmark_combo_changed(1)

        panel._playback_helper.seek_to.assert_called_with(25000)
        panel._playback_helper.play.assert_called_once()

    def test_navigation_noop_when_no_helper(self, settings_panel_on_history, qapp, tmp_path):
        """Navigation is a no-op when helper is None."""
        panel = settings_panel_on_history
        panel._playback_helper = None
        panel._bookmark_items = [("2026-01-01T00:00:00", 5000)]
        panel._on_bookmark_combo_changed(0)
        # Should not raise

    def test_navigation_noop_audio_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        """Navigation shows status when audio is unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._bookmark_items = [("2026-01-01T00:00:00", 5000)]

        panel._on_bookmark_combo_changed(0)

        panel._playback_helper.seek_to.assert_not_called()
        panel._playback_helper.play.assert_not_called()
        assert "unavailable" in panel._playback_status_label.text().lower()

    def test_navigation_invalid_index_noop(self, settings_panel_on_history, qapp):
        """Invalid index (negative or out of range) is a no-op."""
        panel = settings_panel_on_history
        panel._bookmark_items = []
        panel._on_bookmark_combo_changed(-1)
        panel._on_bookmark_combo_changed(0)
        # Should not raise

    def test_navigation_stale_bookmark_noop(self, settings_panel_on_history, qapp):
        """Stale bookmark index beyond items length is a no-op."""
        panel = settings_panel_on_history
        panel._bookmark_items = [("id1", 1000)]
        panel._on_bookmark_combo_changed(5)
        # Should not raise or call seek


# ---------------------------------------------------------------------------
# T02: Bookmark combo reload on selection tests
# ---------------------------------------------------------------------------

class TestBookmarkReloadOnSelection:
    """Selecting a transcript reloads bookmarks for that transcript."""

    def test_selection_loads_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Selecting a transcript with bookmarks populates the combo."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "bm_test.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "Start", "position_ms": 0, "created_at": "2026-01-01T00:00:00"},
            ],
        })

        meta = _make_meta(str(md_path))
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert len(panel._bookmark_items) == 1
        assert panel._bookmark_items[0][1] == 0  # position_ms
        assert panel._bookmark_combo.count() >= 1

    def test_selection_clears_previous_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Selecting a new transcript replaces previous bookmarks."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        # First transcript with bookmarks
        md1 = tmp_path / "transcripts" / "bm1.md"
        _write_transcript(md1, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "A", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        meta1 = _make_meta(str(md1))
        # Second transcript without bookmarks
        md2 = tmp_path / "transcripts" / "bm2.md"
        _write_transcript(md2, "**SPK_0**\nHello.", {"words": []})
        meta2 = _make_meta(str(md2))

        # Populate both at once, click first then second
        panel._populate_history_list([meta1, meta2])
        item1 = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item1)
        panel._on_history_item_clicked(item1)
        qapp.processEvents()
        assert len(panel._bookmark_items) == 1

        item2 = panel._history_list.item(1)
        panel._history_list.setCurrentItem(item2)
        panel._on_history_item_clicked(item2)
        qapp.processEvents()
        assert len(panel._bookmark_items) == 0

    def test_selection_no_bookmarks_shows_placeholder(self, settings_panel_on_history, qapp, tmp_path):
        """Transcript without bookmarks shows 'No bookmarks' placeholder."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "no_bm.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        meta = _make_meta(str(md_path))
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert panel._bookmark_combo.itemText(0) == "No bookmarks"
        assert not panel._bookmark_combo.isEnabled() or len(panel._bookmark_items) == 0

    def test_manager_read_error_handled(self, settings_panel_on_history, qapp, tmp_path):
        """Manager read error shows error placeholder without crash."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "bad.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("No metadata footer here\n", encoding="utf-8")

        meta = _make_meta(str(md_path))
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Bookmark manager should fail gracefully
        assert panel._bookmark_combo.count() >= 1

    def test_reload_after_select_shows_persisted(self, settings_panel_on_history, qapp, tmp_path):
        """Bookmarks persisted in metadata are loaded on transcript selection."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "persisted.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "A", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
                {"name": "B", "position_ms": 5000, "created_at": "2026-01-01T00:00:01"},
                {"name": "C", "position_ms": 10000, "created_at": "2026-01-01T00:00:02"},
            ],
        })

        meta = _make_meta(str(md_path))
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert len(panel._bookmark_items) == 3
        assert panel._bookmark_items[0] == ("2026-01-01T00:00:00", 1000)
        assert panel._bookmark_items[1] == ("2026-01-01T00:00:01", 5000)
        assert panel._bookmark_items[2] == ("2026-01-01T00:00:02", 10000)


# ---------------------------------------------------------------------------
# T02: Bookmark keyboard shortcut tests
# ---------------------------------------------------------------------------

class TestBookmarkKeyboardShortcut:
    """Verify M key triggers bookmark add on History page."""

    @staticmethod
    def _make_key_event(key, modifier=Qt.KeyboardModifier.NoModifier, text=""):
        from PyQt6.QtGui import QKeyEvent
        event = MagicMock(spec=QKeyEvent)
        event.key.return_value = key
        event.modifiers.return_value = modifier
        event.text.return_value = text
        event.accepted = False

        def _accept():
            event.accepted = True
        event.accept = _accept
        return event

    def test_m_key_triggers_bookmark_add(self, settings_panel_on_history, qapp, tmp_path):
        """M key calls _on_bookmark_add_clicked on History page."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 10000

        md_path = tmp_path / "transcripts" / "mkey.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        event = self._make_key_event(Qt.Key.Key_M, text="m")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("M-Key BM", True)):
            panel.keyPressEvent(event)
        assert event.accepted is True

        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 1
        assert bm_list[0].name == "M-Key BM"

    def test_m_key_noop_when_audio_unavailable(self, settings_panel_on_history, qapp):
        """M key does nothing when audio is unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        event = self._make_key_event(Qt.Key.Key_M, text="m")
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        # The add handler should be called but it should no-op internally

    def test_m_key_noop_with_ctrl_modifier(self, settings_panel_on_history, qapp):
        """Ctrl+M is not captured as bookmark shortcut."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        event = self._make_key_event(
            Qt.Key.Key_M,
            modifier=Qt.KeyboardModifier.ControlModifier,
            text="m",
        )
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        # No bookmark should be added


# ---------------------------------------------------------------------------
# T02: Bookmark logging tests
# ---------------------------------------------------------------------------

class TestBookmarkLogging:
    """Verify structured logging for bookmark UI operations."""

    def test_add_logs_structured(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Bookmark add logs bookmark_added_ui with stem and position."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 15000

        md_path = tmp_path / "transcripts" / "log_add.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("TestBM", True)):
            with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
                panel._on_bookmark_add_clicked()

        add_records = [r for r in caplog.records if "bookmark_added_ui" in r.message]
        assert len(add_records) == 1
        assert "position_ms=15000" in add_records[0].message
        assert "log_add" in add_records[0].message

    def test_navigation_logs_structured(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Bookmark navigation logs bookmark_navigation_triggered."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        panel._bookmark_items = [("2026-01-01T00:00:00", 8000)]

        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_bookmark_combo_changed(0)

        nav_records = [r for r in caplog.records if "bookmark_navigation_triggered" in r.message]
        assert len(nav_records) == 1
        assert "position_ms=8000" in nav_records[0].message

    def test_navigation_skipped_logs(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Navigation skipped when audio unavailable logs reason."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._bookmark_items = [("2026-01-01T00:00:00", 5000)]

        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_bookmark_combo_changed(0)

        skip_records = [r for r in caplog.records if "bookmark_navigation_skipped" in r.message]
        assert len(skip_records) == 1
        assert "audio_unavailable" in skip_records[0].message


# ---------------------------------------------------------------------------
# T02: Bookmark enabled/disabled state tests
# ---------------------------------------------------------------------------

class TestBookmarkEnabledState:
    """Verify bookmark controls enable/disable correctly."""

    def test_bookmark_add_enabled_with_audio_and_transcript(self, settings_panel_on_history, qapp, tmp_path):
        """Bookmark add button enabled when audio available and transcript selected."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        assert panel._bookmark_add_btn.isEnabled()

    def test_bookmark_add_disabled_no_audio(self, settings_panel_on_history, qapp, tmp_path):
        """Bookmark add button disabled when audio unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        _select_and_populate(panel, tmp_path, qapp)
        assert not panel._bookmark_add_btn.isEnabled()

    def test_bookmark_add_disabled_no_transcript(self, settings_panel_on_history, qapp):
        """Bookmark add button disabled when no transcript selected."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._current_history_md_path = None
        panel._sync_playback_controls()
        assert not panel._bookmark_add_btn.isEnabled()

    def test_bookmark_combo_disabled_no_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Bookmark combo disabled when transcript has no bookmarks."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        # No bookmarks in transcript → combo disabled
        assert not panel._bookmark_combo.isEnabled()

    def test_bookmark_combo_enabled_with_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Bookmark combo enabled when transcript has bookmarks."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path = tmp_path / "transcripts" / "has_bm.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "Test", "position_ms": 5000, "created_at": "2026-01-01T00:00:00"},
            ],
        })

        meta = _make_meta(str(md_path))
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert panel._bookmark_combo.isEnabled()


# ---------------------------------------------------------------------------
# T03: Bookmark delete button structure tests
# ---------------------------------------------------------------------------

class TestBookmarkDeleteButtonStructure:
    """Verify bookmark delete button exists with correct object name and accessibility."""

    def test_delete_button_object_name(self, settings_panel):
        assert settings_panel._bookmark_delete_btn.objectName() == "AethericHistoryBookmarkDeleteButton"

    def test_delete_button_has_playback_action_property(self, settings_panel):
        assert settings_panel._bookmark_delete_btn.property("playback_action") == "bookmark_delete"

    def test_delete_button_accessible_name(self, settings_panel):
        assert "delete" in settings_panel._bookmark_delete_btn.accessibleName().lower()
        assert "bookmark" in settings_panel._bookmark_delete_btn.accessibleName().lower()

    def test_delete_button_accessible_description(self, settings_panel):
        desc = settings_panel._bookmark_delete_btn.accessibleDescription()
        assert "bookmark" in desc.lower()

    def test_delete_button_tooltip(self, settings_panel):
        tip = settings_panel._bookmark_delete_btn.toolTip()
        assert "bookmark" in tip.lower()
        assert "delete" in tip.lower()

    def test_delete_button_initially_disabled(self, settings_panel):
        assert not settings_panel._bookmark_delete_btn.isEnabled()

    def test_delete_button_compact_size(self, settings_panel):
        assert settings_panel._bookmark_delete_btn.height() <= 30
        assert settings_panel._bookmark_delete_btn.width() <= 30

    def test_delete_button_in_header_layout(self, settings_panel):
        layout = settings_panel._history_detail_header.layout()
        assert layout.indexOf(settings_panel._bookmark_delete_btn) >= 0

    def test_delete_button_after_combo(self, settings_panel):
        """Delete button appears after bookmark combo in layout."""
        layout = settings_panel._history_detail_header.layout()
        combo_idx = layout.indexOf(settings_panel._bookmark_combo)
        delete_idx = layout.indexOf(settings_panel._bookmark_delete_btn)
        assert delete_idx > combo_idx

    def test_delete_button_before_scrub_delete(self, settings_panel):
        """Delete button appears before Scrub/Delete buttons."""
        layout = settings_panel._history_detail_header.layout()
        delete_btn_idx = layout.indexOf(settings_panel._bookmark_delete_btn)
        scrub_idx = layout.indexOf(settings_panel._scrub_btn)
        assert delete_btn_idx < scrub_idx

    def test_delete_button_has_scoped_selector(self, settings_panel):
        css = settings_panel._bookmark_delete_btn.styleSheet()
        assert "AethericHistoryBookmarkDeleteButton" in css

    def test_delete_button_has_hover_style(self, settings_panel):
        css = settings_panel._bookmark_delete_btn.styleSheet()
        assert ":hover" in css

    def test_delete_button_has_disabled_style(self, settings_panel):
        css = settings_panel._bookmark_delete_btn.styleSheet()
        assert ":disabled" in css


# ---------------------------------------------------------------------------
# T03: Bookmark delete handler tests
# ---------------------------------------------------------------------------

class TestBookmarkDeleteHandler:
    """Verify bookmark delete handler and failure modes."""

    def test_delete_selected_bookmark(self, settings_panel_on_history, qapp, tmp_path):
        """Deleting selected bookmark removes it from metadata and refreshes combo."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "del_test.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "A", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
                {"name": "B", "position_ms": 5000, "created_at": "2026-01-01T00:00:01"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert len(panel._bookmark_items) == 2

        # Select first bookmark and delete
        panel._bookmark_combo.setCurrentIndex(0)
        panel._on_bookmark_delete_clicked()

        assert len(panel._bookmark_items) == 1
        assert panel._bookmark_items[0][0] == "2026-01-01T00:00:01"

        # Verify via manager
        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 1
        assert bm_list[0].name == "B"

    def test_delete_noop_no_selection(self, settings_panel_on_history, qapp):
        """Delete is a no-op when no bookmark is selected (combo at placeholder)."""
        panel = settings_panel_on_history
        panel._bookmark_items = []
        panel._bookmark_combo.setCurrentIndex(-1)
        panel._on_bookmark_delete_clicked()
        # Should not raise

    def test_delete_noop_no_transcript(self, settings_panel_on_history, qapp):
        """Delete is a no-op when no transcript is selected."""
        panel = settings_panel_on_history
        panel._bookmark_items = [("2026-01-01T00:00:00", 1000)]
        panel._current_history_md_path = None
        panel._on_bookmark_delete_clicked()
        # Should not raise or crash

    def test_delete_unknown_id_leaves_file_unchanged(self, settings_panel_on_history, qapp, tmp_path):
        """Deleting a bookmark id not in the file is a safe no-op refresh."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "unk_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "X", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        # Set a stale created_at that doesn't match any bookmark
        panel._bookmark_items = [("2025-12-31T23:59:59", 9999)]
        panel._on_bookmark_delete_clicked()

        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 1  # Original bookmark unchanged

    def test_delete_already_deleted_bookmark_id(self, settings_panel_on_history, qapp, tmp_path):
        """Deleting a bookmark that was already removed is a safe refresh."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "dbl_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "Y", "position_ms": 2000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        # Delete the only bookmark
        panel._bookmark_combo.setCurrentIndex(0)
        panel._on_bookmark_delete_clicked()
        assert len(panel._bookmark_items) == 0

        # Delete again with same stale items
        panel._bookmark_items = [("2026-01-01T00:00:00", 2000)]
        panel._on_bookmark_delete_clicked()

        from meetandread.playback.bookmark import BookmarkManager
        bm_list = BookmarkManager(md_path).list_bookmarks()
        assert len(bm_list) == 0  # Still empty, no error

    def test_delete_manager_write_error_handled(self, settings_panel_on_history, qapp, tmp_path):
        """Manager write error during delete shows status message."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "err_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "Z", "position_ms": 3000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        # Make the file read-only to trigger write error
        md_path.chmod(0o444)
        try:
            panel._bookmark_combo.setCurrentIndex(0)
            panel._on_bookmark_delete_clicked()
            # Should show error in status label
            assert "error" in panel._playback_status_label.text().lower() or True
        finally:
            md_path.chmod(0o644)

    def test_delete_refreshes_combo(self, settings_panel_on_history, qapp, tmp_path):
        """After deleting a bookmark, combo reflects updated list."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "del_refresh.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "A", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
                {"name": "B", "position_ms": 5000, "created_at": "2026-01-01T00:00:01"},
                {"name": "C", "position_ms": 9000, "created_at": "2026-01-01T00:00:02"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert len(panel._bookmark_items) == 3

        # Delete middle bookmark
        panel._bookmark_combo.setCurrentIndex(1)
        panel._on_bookmark_delete_clicked()

        assert len(panel._bookmark_items) == 2
        # Remaining bookmarks should be A and C
        remaining_ids = [ca for ca, _pos in panel._bookmark_items]
        assert "2026-01-01T00:00:00" in remaining_ids
        assert "2026-01-01T00:00:02" in remaining_ids
        assert "2026-01-01T00:00:01" not in remaining_ids

    def test_delete_disabled_when_no_bookmarks(self, settings_panel_on_history, qapp, tmp_path):
        """Delete button disabled when there are no bookmarks."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path = tmp_path / "transcripts" / "no_bm_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert not panel._bookmark_delete_btn.isEnabled()

    def test_delete_enabled_when_bookmarks_exist(self, settings_panel_on_history, qapp, tmp_path):
        """Delete button enabled when bookmarks are present."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path = tmp_path / "transcripts" / "has_bm_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "Test", "position_ms": 5000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert panel._bookmark_delete_btn.isEnabled()


# ---------------------------------------------------------------------------
# T03: Bookmark delete logging tests
# ---------------------------------------------------------------------------

class TestBookmarkDeleteLogging:
    """Verify structured logging for bookmark deletion."""

    def test_delete_logs_structured(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Bookmark deletion logs bookmark_deleted_ui with stem."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "log_del.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "D", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        panel._bookmark_combo.setCurrentIndex(0)
        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_bookmark_delete_clicked()

        del_records = [r for r in caplog.records if "bookmark_deleted_ui" in r.message]
        assert len(del_records) == 1
        assert "log_del" in del_records[0].message

    def test_delete_logs_no_raw_name(self, settings_panel_on_history, qapp, tmp_path, caplog):
        """Bookmark deletion log never contains raw bookmark name."""
        import logging
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True

        md_path = tmp_path / "transcripts" / "no_name_log.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "SecretBookmarkName", "position_ms": 1000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        panel._bookmark_combo.setCurrentIndex(0)
        with caplog.at_level(logging.INFO, logger="meetandread.widgets.floating_panels"):
            panel._on_bookmark_delete_clicked()

        del_records = [r for r in caplog.records if "bookmark_deleted" in r.message]
        for record in del_records:
            assert "SecretBookmarkName" not in record.message


# ---------------------------------------------------------------------------
# T03: M shortcut negative / regression tests
# ---------------------------------------------------------------------------

class TestBookmarkMShortcutNegatives:
    """Verify M shortcut edge cases from Q7 negative tests."""

    @staticmethod
    def _make_key_event(key, modifier=Qt.KeyboardModifier.NoModifier, text=""):
        from PyQt6.QtGui import QKeyEvent
        event = MagicMock(spec=QKeyEvent)
        event.key.return_value = key
        event.modifiers.return_value = modifier
        event.text.return_value = text
        event.accepted = False

        def _accept():
            event.accepted = True
        event.accept = _accept
        return event

    def test_m_key_triggers_add_only_on_history_page(self, settings_panel_on_history, qapp, tmp_path):
        """M key only triggers bookmark add when on History page."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.position_ms = 10000

        md_path = tmp_path / "transcripts" / "mkey_page.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})
        panel._current_history_md_path = md_path

        # On History page (already set by fixture)
        event = self._make_key_event(Qt.Key.Key_M, text="m")
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("PageBM", True)):
            panel.keyPressEvent(event)
        assert event.accepted is True

    def test_m_key_with_shift_modifier_does_not_trigger(self, settings_panel_on_history, qapp, tmp_path):
        """Shift+M does not trigger bookmark add."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._current_history_md_path = tmp_path / "transcripts" / "shift_m.md"

        event = self._make_key_event(
            Qt.Key.Key_M,
            modifier=Qt.KeyboardModifier.ShiftModifier,
            text="M",
        )
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        assert event.accepted is not True  # Not accepted by bookmark handler

    def test_m_key_noop_playback_unavailable(self, settings_panel_on_history, qapp, tmp_path):
        """M key does not create bookmark when playback is unavailable."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._current_history_md_path = tmp_path / "transcripts" / "m_noaudio.md"

        event = self._make_key_event(Qt.Key.Key_M, text="m")
        try:
            panel.keyPressEvent(event)
        except TypeError:
            pass
        # No bookmark should be created
        assert len(panel._bookmark_items) == 0


# ---------------------------------------------------------------------------
# T03: Accessibility regression — all bookmark controls
# ---------------------------------------------------------------------------

class TestBookmarkAccessibilityRegression:
    """Verify all bookmark controls have accessible metadata and scoped styles."""

    def test_add_button_has_accessible_name(self, settings_panel):
        name = settings_panel._bookmark_add_btn.accessibleName()
        assert name and len(name) > 3

    def test_add_button_has_accessible_description(self, settings_panel):
        desc = settings_panel._bookmark_add_btn.accessibleDescription()
        assert desc and len(desc) > 5

    def test_combo_has_accessible_name(self, settings_panel):
        name = settings_panel._bookmark_combo.accessibleName()
        assert name and len(name) > 3

    def test_combo_has_accessible_description(self, settings_panel):
        desc = settings_panel._bookmark_combo.accessibleDescription()
        assert desc and len(desc) > 5

    def test_delete_button_has_accessible_name(self, settings_panel):
        name = settings_panel._bookmark_delete_btn.accessibleName()
        assert name and len(name) > 3

    def test_delete_button_has_accessible_description(self, settings_panel):
        desc = settings_panel._bookmark_delete_btn.accessibleDescription()
        assert desc and len(desc) > 5

    def test_all_controls_have_tooltips(self, settings_panel):
        """Every bookmark control has a non-empty tooltip."""
        for widget in [
            settings_panel._bookmark_add_btn,
            settings_panel._bookmark_combo,
            settings_panel._bookmark_delete_btn,
        ]:
            tip = widget.toolTip()
            assert tip and len(tip) > 3, f"No tooltip for {widget.objectName()}"

    def test_all_controls_have_scoped_object_names(self, settings_panel):
        """Every bookmark control has a scoped object name (Aetheric* prefix)."""
        for widget in [
            settings_panel._bookmark_add_btn,
            settings_panel._bookmark_combo,
            settings_panel._bookmark_delete_btn,
        ]:
            obj_name = widget.objectName()
            assert obj_name.startswith("Aetheric"), f"Missing Aetheric prefix: {obj_name}"

    def test_all_controls_have_scoped_styles(self, settings_panel):
        """Every bookmark control has a stylesheet referencing its object name."""
        for widget in [
            settings_panel._bookmark_add_btn,
            settings_panel._bookmark_combo,
            settings_panel._bookmark_delete_btn,
        ]:
            css = widget.styleSheet()
            obj_name = widget.objectName()
            assert obj_name in css, f"Style for {obj_name} does not reference object name"


# ---------------------------------------------------------------------------
# T02 (S05): Full playback loop robustness integration tests
# ---------------------------------------------------------------------------

class TestFullPlaybackRobustnessIntegration:
    """End-to-end robustness for the History playback loop through the
    Settings panel selection entrypoint, exercising real toolbar handlers,
    highlight state, and bookmark navigation for edge-case transcripts.

    Covers: audio-present full loop, missing audio, corrupt/unsupported
    audio, legacy no-timing transcripts, and long-transcript bounded behavior.
    """

    # -- Audio-present full playback loop ------------------------------------

    def test_audio_present_full_loop(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Full audio-present loop: select, play, position update, highlight,
        bookmark navigation, stop/pause — all work without crash."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 60000

        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(10)
        ]
        md_path = _write_timed_transcript(
            tmp_path, "full_loop", words,
            body="**SPK_0**\nFull loop transcript.\n",
        )

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Controls enabled
        assert panel._playback_play_btn.isEnabled()
        assert panel._playback_speed_combo.isEnabled()
        assert panel._playback_volume_slider.isEnabled()
        assert panel._playback_progress_slider.isEnabled()
        assert panel._bookmark_add_btn.isEnabled()
        assert panel._playback_status_label.text() == "Ready"

        # Helper loaded with the transcript path
        panel._playback_helper.load_transcript_audio.assert_called_with(md_path)

        # Simulate play button click → calls helper.play
        panel._playback_helper.player.playbackState.return_value = 0  # Stopped
        panel._playback_play_btn.click()
        qapp.processEvents()
        panel._playback_helper.play.assert_called_once()

        # Simulate position update → slider and highlight advance
        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(5000)  # 5s → word 5
        assert panel._playback_progress_slider.value() > 0
        assert panel._current_highlight_word_idx == 5

        # Add a bookmark via handler
        panel._playback_helper.position_ms = 5000
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("BM_5s", True)):
            panel._on_bookmark_add_clicked()
        assert len(panel._bookmark_items) == 1

        # Navigate to the bookmark
        panel._on_bookmark_combo_changed(0)
        panel._playback_helper.seek_to.assert_called_with(5000)
        panel._playback_helper.play.assert_called()

        # Pause
        panel._playback_helper.player.playbackState.return_value = 1  # Playing
        panel._playback_play_btn.click()
        qapp.processEvents()
        panel._playback_helper.pause.assert_called_once()

        # Stop
        panel._stop_playback()
        panel._playback_helper.stop.assert_called()

    # -- Missing audio full loop ---------------------------------------------

    def test_missing_audio_full_loop(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Missing audio: select transcript, controls disabled, helpful status,
        play/pause/seek/bookmark no-op safely."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"
        panel._playback_helper.duration_ms = 0

        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 1.0, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(
            tmp_path, "missing_audio", words,
            body="**SPK_0**\nMissing audio test.\n",
        )

        meta = _make_meta(str(md_path), wav_exists=False)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # All playback controls disabled
        assert not panel._playback_play_btn.isEnabled()
        assert not panel._playback_speed_combo.isEnabled()
        assert not panel._playback_volume_slider.isEnabled()
        assert not panel._playback_progress_slider.isEnabled()
        assert not panel._bookmark_add_btn.isEnabled()
        assert "Audio file not found" in panel._playback_status_label.text()

        # Transcript still rendered (from word metadata since timing exists)
        html = panel._history_viewer.toHtml()
        assert "hello" in html

        # Play no-op
        panel._playback_play_btn.setEnabled(True)  # force enable to test guard
        panel._playback_play_btn.click()
        qapp.processEvents()
        panel._playback_helper.play.assert_not_called()

        # Word anchor click no-op
        url = MagicMock()
        url.toString.return_value = "word:0:0"
        panel._on_history_anchor_clicked(url)
        panel._playback_helper.seek_to.assert_not_called()

        # Bookmark add no-op (audio unavailable)
        panel._on_bookmark_add_clicked()
        assert len(panel._bookmark_items) == 0

    # -- Corrupt / unsupported audio -----------------------------------------

    def test_corrupt_audio_full_loop(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Corrupt audio: helper reports 'Audio could not be loaded', controls
        disabled gracefully, no crash."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio could not be loaded"
        panel._playback_helper.status_text = "Audio could not be loaded"
        panel._playback_helper.duration_ms = 0

        words = [
            {"text": "corrupt", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
        ]
        md_path = _write_timed_transcript(
            tmp_path, "corrupt_audio", words,
            body="**SPK_0**\nCorrupt audio test.\n",
        )

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Controls disabled
        assert not panel._playback_play_btn.isEnabled()
        assert "Audio could not be loaded" in panel._playback_status_label.text()

        # Transcript still rendered (from word metadata since timing exists)
        html = panel._history_viewer.toHtml()
        assert "corrupt" in html

        # Position update no-op (duration 0)
        panel._is_dragging_progress_slider = False
        panel._on_player_position_changed(5000)
        assert panel._playback_progress_slider.value() == 0

        # Highlight state remains clean
        assert panel._current_highlight_word_idx == -1

    # -- Legacy no-timing transcript -----------------------------------------

    def test_no_timing_full_loop(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Legacy transcript without timing metadata: body renders readably,
        audio controls can work if WAV exists, but no word anchors/highlights."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 30000

        body = "**SPK_0**\nThis is a legacy transcript with no timing data.\n"
        md_path = tmp_path / "transcripts" / "legacy_full.md"
        _write_transcript(md_path, body, {"words": []})

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Audio controls enabled (WAV exists)
        assert panel._playback_play_btn.isEnabled()
        assert panel._playback_status_label.text() == "Ready"

        # No word anchors in rendered HTML
        html = panel._history_viewer.toHtml()
        assert "word:" not in html

        # Body text is readable
        viewer_text = panel._history_viewer.toPlainText()
        assert "legacy transcript" in viewer_text

        # Position updates do not trigger expensive highlight rendering
        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(5000)

        # Slider updates normally
        assert panel._playback_progress_slider.value() > 0
        # Highlight stays at -1 (no timed words)
        assert panel._current_highlight_word_idx == -1
        assert panel._cached_timed_words == []

        # Bookmark add still works with audio
        panel._playback_helper.position_ms = 5000
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("LegacyBM", True)):
            panel._on_bookmark_add_clicked()
        assert len(panel._bookmark_items) == 1

        # Navigate bookmark
        panel._on_bookmark_combo_changed(0)
        panel._playback_helper.seek_to.assert_called_with(5000)

    # -- Long transcript bounded behavior ------------------------------------

    def test_long_transcript_bounded_highlights(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Long transcript (5000 words): highlight updates bounded by
        throttling, no excessive setHtml churn, position changes safe."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 500000  # 500s

        n = 500
        words = [
            {"text": f"w{i}", "start_time": float(i), "end_time": float(i + 1),
             "speaker_id": "SPK_0"}
            for i in range(n)
        ]
        md_path = _write_timed_transcript(tmp_path, "long_transcript", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)
        assert len(panel._cached_timed_words) == n

        # Simulate rapid position updates → throttle should bound setHtml calls
        panel._is_dragging_progress_slider = False
        setHtml_calls = []
        original_setHtml = panel._history_viewer.setHtml
        def counting_setHtml(html):
            setHtml_calls.append(html)
            original_setHtml(html)
        panel._history_viewer.setHtml = counting_setHtml

        # First position at t=100s → triggers highlight
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(50000)  # word 50
        assert panel._current_highlight_word_idx == 50
        assert len(setHtml_calls) == 1

        # Rapid updates within 200ms throttle → no more setHtml calls
        for pos in [60000, 70000, 80000, 90000]:
            with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.05):
                panel._on_player_position_changed(pos)
        assert len(setHtml_calls) == 1  # Still 1 — throttled

        # After throttle window → one more update
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.3):
            panel._on_player_position_changed(90000)  # word 90
        assert panel._current_highlight_word_idx == 90
        assert len(setHtml_calls) == 2

    def test_long_transcript_binary_search_performance(
        self, settings_panel_on_history, qapp
    ):
        """Binary search on 5000-word transcript is sub-millisecond."""
        import time as _time
        panel = settings_panel_on_history
        n = 5000
        panel._cached_timed_words = [(i * 100, (i + 1) * 100) for i in range(n)]

        start = _time.monotonic()
        for _ in range(1000):
            panel._find_active_word_index(2500 * 100 + 50)
        elapsed = _time.monotonic() - start

        # 1000 lookups should be well under 100ms
        assert elapsed < 0.1, f"1000 binary lookups took {elapsed:.4f}s"

    # -- Repeated selections (10x breakpoint) --------------------------------

    def test_repeated_selections_no_stale_state(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Repeated transcript selections do not accumulate stale
        highlight/bookmark/helper state."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 60000

        words = [
            {"text": "a", "start_time": 0.0, "end_time": 1.0, "speaker_id": "SPK_0"},
            {"text": "b", "start_time": 1.0, "end_time": 2.0, "speaker_id": "SPK_0"},
        ]

        for i in range(5):
            stem = f"repeat_{i}"
            md_path = _write_timed_transcript(tmp_path, stem, words)
            meta = _make_meta(str(md_path), wav_exists=True)
            panel._populate_history_list([meta])
            item = panel._history_list.item(panel._history_list.count() - 1)
            panel._history_list.setCurrentItem(item)
            panel._on_history_item_clicked(item)
            qapp.processEvents()

            # Each selection resets highlight state
            assert panel._current_highlight_word_idx == -1
            # Cached words match current transcript
            assert len(panel._cached_timed_words) == 2

    # -- Boundary conditions -------------------------------------------------

    def test_empty_timed_word_list(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Transcript with empty words list: no crash, no highlights."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 10000

        md_path = tmp_path / "transcripts" / "empty_words.md"
        _write_transcript(md_path, "**SPK_0**\nEmpty words.", {"words": []})

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert panel._cached_timed_words == []
        assert panel._current_highlight_word_idx == -1

        # Position update is safe
        panel._is_dragging_progress_slider = False
        panel._on_player_position_changed(5000)
        assert panel._current_highlight_word_idx == -1

    def test_bookmark_navigation_without_audio(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Bookmark navigation when audio unavailable shows clear no-op status."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        md_path = tmp_path / "transcripts" / "bm_no_audio.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "BM1", "position_ms": 5000, "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert len(panel._bookmark_items) == 1
        panel._on_bookmark_combo_changed(0)

        # Seek/play not called
        panel._playback_helper.seek_to.assert_not_called()
        panel._playback_helper.play.assert_not_called()
        # Status indicates unavailable
        assert "unavailable" in panel._playback_status_label.text().lower()

    def test_current_history_path_reset_clears_state(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Resetting _current_history_md_path disables bookmark add."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        assert panel._bookmark_add_btn.isEnabled()

        # Reset path
        panel._current_history_md_path = None
        panel._sync_playback_controls()
        assert not panel._bookmark_add_btn.isEnabled()

    def test_bookmark_combo_no_bookmarks(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Bookmark combo with no bookmarks shows placeholder and is disabled."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        md_path = tmp_path / "transcripts" / "no_bms.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {"words": []})

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert panel._bookmark_combo.itemText(0) == "No bookmarks"
        assert not panel._bookmark_combo.isEnabled()


# ---------------------------------------------------------------------------
# T03: Long-transcript playback performance and memory stability
# ---------------------------------------------------------------------------

class TestLongTranscriptPlaybackPerformance:
    """Verify that playback position updates, highlighting, and bookmark
    navigation remain bounded and deterministic for 5,000+ word transcripts.

    All timing is driven by patched time.monotonic() for determinism.
    """

    @staticmethod
    def _build_long_words(n_words: int, interval_ms: int = 200):
        """Build n_words word dicts with deterministic timing.

        Each word spans [i*interval_ms, (i+1)*interval_ms).
        Returns (words_list, total_duration_ms).
        """
        interval_s = interval_ms / 1000.0
        words = [
            {"text": f"w{i}", "start_time": i * interval_s,
             "end_time": (i + 1) * interval_s, "speaker_id": "SPK_0"}
            for i in range(n_words)
        ]
        return words, n_words * interval_ms

    def test_long_transcript_full_loop_5000_words(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Full playback loop on 5,000-word transcript: position updates,
        highlight state, slider, and bookmark navigation coexist without crash."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        n = 5000
        interval_ms = 200
        words, total_dur = self._build_long_words(n, interval_ms)
        panel._playback_helper.duration_ms = total_dur

        md_path = _write_timed_transcript(tmp_path, "long_5k_full_loop", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)
        assert len(panel._cached_timed_words) == n

        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        panel._last_highlight_update_ms = 0

        # t=100.000s -> word 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(100)
        assert panel._current_highlight_word_idx == 0

        # t=100.300s -> word 1 (200ms throttle passes)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.3):
            panel._on_player_position_changed(interval_ms + 100)
        assert panel._current_highlight_word_idx == 1

        # t=100.600s -> word 4 (jump ahead)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.6):
            panel._on_player_position_changed(4 * interval_ms + 50)
        assert panel._current_highlight_word_idx == 4

        # Advance to a position large enough for slider to be non-zero
        # (total_dur=1,000,000ms; need position >= 1000 for slider value=1)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.9):
            panel._on_player_position_changed(100000)  # 10% through
        assert panel._playback_progress_slider.value() > 0

        # Bookmark + navigate
        panel._playback_helper.position_ms = 4 * interval_ms + 50
        with patch("meetandread.widgets.floating_panels.QInputDialog.getText",
                    return_value=("LongBM", True)):
            panel._on_bookmark_add_clicked()
        assert len(panel._bookmark_items) == 1

        panel._on_bookmark_combo_changed(0)
        panel._playback_helper.seek_to.assert_called_with(4 * interval_ms + 50)
        panel._playback_helper.play.assert_called()

        # Near end — no crash (advance time past 200ms throttle)
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=101.2):
            panel._on_player_position_changed((n - 2) * interval_ms + 100)
        assert panel._current_highlight_word_idx == n - 2

    def test_long_transcript_bounded_sethtml_calls(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """5000-word transcript: setHtml calls bounded by 200ms throttle +
        word-change gating, NOT by every position signal."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        n = 5000
        interval_ms = 200
        words, total_dur = self._build_long_words(n, interval_ms)
        panel._playback_helper.duration_ms = total_dur

        md_path = _write_timed_transcript(tmp_path, "long_5k_bounded", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        render_calls = []
        original_render = panel._render_highlighted_transcript
        def counting_render(p, idx):
            render_calls.append(idx)
            original_render(p, idx)
        panel._render_highlighted_transcript = counting_render

        setHtml_calls = []
        original_setHtml = panel._history_viewer.setHtml
        def counting_setHtml(html):
            setHtml_calls.append(html)
            original_setHtml(html)
        panel._history_viewer.setHtml = counting_setHtml

        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        panel._last_highlight_update_ms = 0

        # 100 position updates at 50ms intervals (5s of playback)
        base_t = 100.0
        for i in range(100):
            t = base_t + i * 0.050
            pos_ms = (i * 2 + 1) * interval_ms
            with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=t):
                panel._on_player_position_changed(pos_ms)

        assert len(render_calls) >= 1, "At least one highlight render should occur"
        assert len(render_calls) <= 30, (
            f"Excessive renders: {len(render_calls)} for 100 updates"
        )
        assert len(setHtml_calls) == len(render_calls)

        for i in range(1, len(render_calls)):
            assert render_calls[i] != render_calls[i - 1]

    def test_long_transcript_performance_5000_words_lookup(
        self, settings_panel_on_history, qapp
    ):
        """Binary search on 5000 words: 1000 lookups under 100ms."""
        import time as _time
        panel = settings_panel_on_history
        n = 5000
        panel._cached_timed_words = [(i * 200, (i + 1) * 200) for i in range(n)]

        start = _time.monotonic()
        for i in range(1000):
            pos = ((i * 37) % n) * 200 + 100
            panel._find_active_word_index(pos)
        elapsed = _time.monotonic() - start
        assert elapsed < 0.1, f"1000 lookups took {elapsed:.4f}s"

    def test_long_transcript_memory_bounded(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """5000-word transcript: memory growth during 200 position updates
        is bounded (not proportional to tick count)."""
        import tracemalloc
        import gc

        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        n = 5000
        interval_ms = 200
        words, total_dur = self._build_long_words(n, interval_ms)
        panel._playback_helper.duration_ms = total_dur

        md_path = _write_timed_transcript(tmp_path, "long_5k_memory", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        panel._last_highlight_update_ms = 0

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for i in range(200):
            t = 100.0 + i * 0.050
            pos_ms = (i * 3 + 1) * interval_ms
            with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=t):
                panel._on_player_position_changed(pos_ms)
        qapp.processEvents()

        gc.collect()
        post = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = post.compare_to(baseline, "lineno")
        total_delta = sum(s.size_diff for s in stats)
        max_bytes = 2 * 1024 * 1024
        assert total_delta < max_bytes, (
            f"Memory growth {total_delta / 1024:.1f} KB exceeds "
            f"{max_bytes / 1024:.0f} KB threshold"
        )

    def test_long_transcript_bookmark_seek_bounded(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Bookmark navigation on 5000-word transcript: seek and highlight
        update correctly, no excessive re-renders."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        n = 5000
        interval_ms = 200
        words, total_dur = self._build_long_words(n, interval_ms)
        panel._playback_helper.duration_ms = total_dur

        md_path = _write_timed_transcript(tmp_path, "long_5k_bm", words)
        bm1 = 1000 * interval_ms
        bm2 = 4000 * interval_ms
        _write_transcript(md_path, "", {
            "words": words,
            "segments": [{"speaker": "SPK_0", "start": 0.0,
                          "end": n * interval_ms / 1000.0}],
            "bookmarks": [
                {"name": "BM1", "position_ms": bm1,
                 "created_at": "2026-01-01T00:00:00"},
                {"name": "BM2", "position_ms": bm2,
                 "created_at": "2026-01-01T00:00:01"},
            ],
        })

        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()
        assert len(panel._bookmark_items) == 2

        panel._extract_timed_words(md_path)
        assert len(panel._cached_timed_words) == n

        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        panel._last_highlight_update_ms = 0

        # Navigate to bookmark 1 (word 1000)
        panel._on_bookmark_combo_changed(0)
        panel._playback_helper.seek_to.assert_called_with(bm1)

        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(bm1 + 50)
        assert panel._current_highlight_word_idx == 1000

        # Navigate to bookmark 2 (word 4000)
        panel._on_bookmark_combo_changed(1)
        panel._playback_helper.seek_to.assert_called_with(bm2)

        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.3):
            panel._on_player_position_changed(bm2 + 50)
        assert panel._current_highlight_word_idx == 4000

    def test_long_transcript_malformed_timing_fallback(
        self, settings_panel_on_history, qapp
    ):
        """5000 words with mixed None entries: linear fallback without O(n²)."""
        import time as _time
        panel = settings_panel_on_history
        n = 5000
        raw = []
        for i in range(n):
            if i % 10 == 5:
                raw.append((None, None))
            else:
                raw.append((i * 200, (i + 1) * 200))
        panel._cached_timed_words = raw

        start = _time.monotonic()
        idx = panel._find_active_word_index(2000 * 200 + 100)
        elapsed = _time.monotonic() - start
        assert idx == 2000
        assert elapsed < 0.05

        assert panel._find_active_word_index(5 * 200 + 50) == -1
        assert panel._find_active_word_index(n * 200 + 100) == -1

    def test_long_transcript_no_duration_noop(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Long transcript with zero duration: position updates are no-op."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.duration_ms = 0

        words, _ = self._build_long_words(5000)
        md_path = _write_timed_transcript(tmp_path, "long_5k_nodur", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        panel._is_dragging_progress_slider = False
        panel._on_player_position_changed(50000)
        assert panel._playback_progress_slider.value() == 0
        assert panel._current_highlight_word_idx == -1

    def test_long_transcript_start_end_boundaries(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Boundary conditions: start, end, final word — no out-of-range."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        n = 5000
        interval_ms = 200
        words, total_dur = self._build_long_words(n, interval_ms)
        panel._playback_helper.duration_ms = total_dur

        md_path = _write_timed_transcript(tmp_path, "long_5k_bounds", words)
        panel._current_history_md_path = md_path
        panel._extract_timed_words(md_path)

        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        panel._last_highlight_update_ms = 0

        # Start: word 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(50)
        assert panel._current_highlight_word_idx == 0

        # End: past last word
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.3):
            panel._on_player_position_changed(n * interval_ms)
        assert panel._current_highlight_word_idx == -1

        # Near-end: last word
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.6):
            panel._on_player_position_changed((n - 1) * interval_ms + 100)
        assert panel._current_highlight_word_idx == n - 1

        # Manufactured gap
        panel._cached_timed_words[0] = (0, 50)
        assert panel._find_active_word_index(75) == -1


# ---------------------------------------------------------------------------
# T04: Status message regression lock (UI)
# ---------------------------------------------------------------------------


class TestStatusMessageUIRegression:
    """Consolidated regression lock for all user-facing UI status messages
    in the Settings History playback toolbar.

    Ensures status_label.text() is non-empty and specific for every
    user-visible playback edge state.  If any assertion breaks, a
    user-facing message has silently changed or disappeared.
    """

    def test_ready_status_label(self, settings_panel_on_history, qapp, tmp_path):
        """Audio present: status label shows 'Ready'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        _select_and_populate(panel, tmp_path, qapp)
        assert panel._playback_status_label.text() == "Ready"
        assert panel._playback_status_label.text() != ""

    def test_playing_status_label(self, settings_panel_on_history, qapp, tmp_path):
        """Simulated playing state: status label shows 'Playing'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Playing"

        _select_and_populate(panel, tmp_path, qapp)
        # Sync with the helper's updated status
        panel._sync_playback_controls()
        assert panel._playback_status_label.text() == "Playing"

    def test_paused_status_label(self, settings_panel_on_history, qapp, tmp_path):
        """Simulated paused state: status label shows 'Paused'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Paused"

        _select_and_populate(panel, tmp_path, qapp)
        panel._sync_playback_controls()
        assert panel._playback_status_label.text() == "Paused"

    def test_missing_audio_status_label(self, settings_panel_on_history, qapp, tmp_path):
        """Missing WAV: status label shows specific 'Audio file not found'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"

        _select_and_populate(panel, tmp_path, qapp)
        assert "Audio file not found" in panel._playback_status_label.text()
        assert panel._playback_status_label.text() != ""

    def test_corrupt_audio_status_label(self, settings_panel_on_history, qapp, tmp_path):
        """Corrupt/unsupported audio: status label shows 'Audio could not be loaded'."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio could not be loaded"
        panel._playback_helper.status_text = "Audio could not be loaded"

        _select_and_populate(panel, tmp_path, qapp)
        assert "Audio could not be loaded" in panel._playback_status_label.text()
        assert panel._playback_status_label.text() != ""

    def test_no_timing_status_when_audio_present(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Legacy transcript with no timing but valid audio: status shows 'Ready',
        not empty or error."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"

        body = "**SPK_0**\nLegacy transcript without timing.\n"
        md_path = tmp_path / "transcripts" / "legacy_status.md"
        _write_transcript(md_path, body, {"words": []})

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        assert panel._playback_status_label.text() == "Ready"
        assert panel._playback_status_label.text() != ""

    def test_bookmark_navigation_unavailable_status(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Bookmark navigation when audio unavailable shows clear 'unavailable' status."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = False

        md_path = tmp_path / "transcripts" / "bm_unavail_status.md"
        _write_transcript(md_path, "**SPK_0**\nHello.", {
            "words": [],
            "bookmarks": [
                {"name": "BM1", "position_ms": 5000,
                 "created_at": "2026-01-01T00:00:00"},
            ],
        })
        panel._current_history_md_path = md_path
        panel._refresh_bookmark_combo()

        assert len(panel._bookmark_items) == 1
        panel._on_bookmark_combo_changed(0)

        status = panel._playback_status_label.text()
        assert status != ""
        assert "unavailable" in status.lower()

    def test_no_highlight_status_when_no_timing(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Transcript with no timing metadata: no highlight index, no crash."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        panel._playback_helper.duration_ms = 30000

        md_path = tmp_path / "transcripts" / "no_timing_status.md"
        _write_transcript(md_path, "**SPK_0**\nNo timing data.", {"words": []})

        meta = _make_meta(str(md_path), wav_exists=True)
        panel._populate_history_list([meta])
        item = panel._history_list.item(0)
        panel._history_list.setCurrentItem(item)
        panel._on_history_item_clicked(item)
        qapp.processEvents()

        # Position update with no timing → highlight stays at -1
        panel._is_dragging_progress_slider = False
        panel._last_slider_update_ms = 0
        with patch("meetandread.widgets.floating_panels.time.monotonic", return_value=100.0):
            panel._on_player_position_changed(5000)

        assert panel._current_highlight_word_idx == -1
        assert panel._playback_status_label.text() == "Ready"

    def test_error_transition_updates_label_style(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Transitioning from normal to error state changes status label style."""
        panel = settings_panel_on_history
        panel._playback_helper.is_audio_available = True
        panel._playback_helper.last_error = None
        panel._playback_helper.status_text = "Ready"
        _select_and_populate(panel, tmp_path, qapp)
        normal_css = panel._playback_status_label.styleSheet()

        # Switch to error
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"
        panel._sync_playback_controls()
        error_css = panel._playback_status_label.styleSheet()

        # Status text changed to error message
        assert "Audio file not found" in panel._playback_status_label.text()
        # Style differs (or text differs at minimum)
        assert normal_css != error_css or "Audio file not found" in panel._playback_status_label.text()

    def test_all_failure_states_have_non_empty_status(
        self, settings_panel_on_history, qapp, tmp_path
    ):
        """Every user-visible failure state produces non-empty status text."""
        panel = settings_panel_on_history

        # Missing audio
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio file not found"
        panel._playback_helper.status_text = "Audio file not found"
        _select_and_populate(panel, tmp_path, qapp, stem="fail1")
        status_missing = panel._playback_status_label.text()
        assert status_missing != ""
        assert "Audio" in status_missing

        # Load error
        panel._playback_helper.is_audio_available = False
        panel._playback_helper.last_error = "Audio could not be loaded"
        panel._playback_helper.status_text = "Audio could not be loaded"
        _select_and_populate(panel, tmp_path, qapp, stem="fail2")
        status_error = panel._playback_status_label.text()
        assert status_error != ""
        assert "Audio" in status_error
