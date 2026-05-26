"""Tests for the post-processing pending indicator in History list items.

R028: History recording list items should show a visual indicator when
post-processing (diarization, speaker identification) is still pending.

The indicator is shown when RecordingMeta.speaker_count == 0, which means
no speakers have been identified yet — either post-processing hasn't run
or it's still in progress.

Tests target the _build_history_display_text helper and the italic styling
applied by _populate_history_list, without requiring full widget construction.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel

from meetandread.transcription.transcript_scanner import RecordingMeta


# Ensure QApplication exists for QFont/QLabel tests
_app = QApplication.instance() or QApplication(sys.argv)


def _make_meta(
    path: str = "recording-2026-01-01-120000.md",
    word_count: int = 100,
    speaker_count: int = 2,
    speakers: list = None,
    recording_time: str = "2026-01-01T12:00:00",
    duration_seconds: float = 60.0,
    wav_exists: bool = True,
) -> RecordingMeta:
    """Create a RecordingMeta with sensible defaults."""
    return RecordingMeta(
        path=Path(path),
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=speaker_count,
        speakers=speakers or [f"SPK_{i}" for i in range(speaker_count)],
        duration_seconds=duration_seconds,
        wav_exists=wav_exists,
    )


# ---------------------------------------------------------------------------
# Test: Display text logic
# ---------------------------------------------------------------------------

class TestHistoryDisplayText:
    """Verify display text generation for history items."""

    def test_no_speakers_includes_indicator(self):
        """Items with speaker_count==0 and word_count>0 should show indicator."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "processing speakers" in text.lower(), (
            f"Expected 'processing speakers' in text, got: {text}"
        )

    def test_has_speakers_no_indicator(self):
        """Items with speaker_count>0 should NOT show the indicator."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "processing speakers" not in text.lower(), (
            f"Unexpected indicator in text: {text}"
        )

    def test_has_speakers_shows_count(self):
        """Items with speakers should show the speaker count."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "3 speakers" in text

    def test_empty_recording_no_speaker_indicator(self):
        """Empty recordings (word_count==0) show 'Empty recording', not speaker indicator."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(word_count=0, speaker_count=0, speakers=[])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "empty recording" in text.lower(), (
            f"Expected 'Empty recording' in text, got: {text}"
        )
        assert "processing speakers" not in text.lower()

    def test_one_speaker_shows_singular(self):
        """Items with 1 speaker should show '1 speaker' (singular)."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=1, speakers=["SPK_0"])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "1 speaker" in text

    def test_no_speakers_italic_flag(self):
        """Items with speaker_count==0 should be flagged for italic styling."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        _, italic = FloatingSettingsPanel._build_history_display_text(meta, return_italic=True)
        assert italic is True

    def test_has_speakers_not_italic(self):
        """Items with speakers should NOT be flagged for italic."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=2)
        _, italic = FloatingSettingsPanel._build_history_display_text(meta, return_italic=True)
        assert italic is False

    def test_empty_recording_not_italic(self):
        """Empty recordings should not get italic styling."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(word_count=0, speaker_count=0, speakers=[])
        _, italic = FloatingSettingsPanel._build_history_display_text(meta, return_italic=True)
        assert italic is False

    def test_renamed_recording_keeps_indicator(self):
        """Custom-named recording with no speakers still shows indicator."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(
            path="My Meeting.md",
            speaker_count=0,
            speakers=[],
        )
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "processing speakers" in text.lower()
