"""Tests for the post-processing pending indicator in History list items.

R028: History recording list items should show a visual indicator of their
Post-processing state. Since issue #62 this is the per-row status pill
(color-coded, with tooltip); the conflated 'Manual Action Required' label
is gone. A Recording with no speakers and no Outcome is Stalled — surfaced
via the pill, not via the row text.

Tests target the _build_history_display_text / _build_history_status_pill
helpers without requiring full widget construction.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_footer import PostProcessOutcome
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
    outcome: PostProcessOutcome = None,
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
        post_process_outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Test: Display text logic
# ---------------------------------------------------------------------------

class TestHistoryDisplayText:
    """Verify display text generation for history items."""

    def test_no_speakers_shows_word_count_only(self):
        """No-speaker recordings show word count; the pill carries status."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "100 words" in text
        assert "manual action required" not in text.lower(), (
            f"'Manual Action Required' must not appear, got: {text}"
        )

    def test_has_speakers_no_manual_action(self):
        """Items with speaker_count>0 never show the manual-action indicator."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "manual action required" not in text.lower(), (
            f"Unexpected indicator in text: {text}"
        )

    def test_has_speakers_shows_count(self):
        """Items with speakers should show the speaker count."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "3 speakers" in text

    def test_empty_recording_no_speaker_indicator(self):
        """Empty recordings (word_count==0) show 'Empty recording'."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(word_count=0, speaker_count=0, speakers=[])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "empty recording" in text.lower(), (
            f"Expected 'Empty recording' in text, got: {text}"
        )
        assert "manual action required" not in text.lower()

    def test_one_speaker_shows_singular(self):
        """Items with 1 speaker should show '1 speaker' (singular)."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=1, speakers=["SPK_0"])
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "1 speaker" in text

    def test_zero_speakers_italic_only_while_live(self):
        """Stalled rows are not italic; live jobs are."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        from meetandread.transcription.post_processor import PostProcessStatus

        meta = _make_meta(speaker_count=0, speakers=[])
        _, stalled_italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
        )
        _, live_italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.RUNNING,
        )
        assert stalled_italic is False
        assert live_italic is True

    def test_renamed_recording_keeps_word_count(self):
        """Custom-named recording with no speakers shows words + pill status."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(
            path="My Meeting.md",
            speaker_count=0,
            speakers=[],
        )
        text = FloatingSettingsPanel._build_history_display_text(meta)
        assert "100 words" in text


# ---------------------------------------------------------------------------
# Test: Stalled zero-speaker recordings surface via the pill
# ---------------------------------------------------------------------------

class TestStalledPillIndicator:
    """Zero-speaker Stalled recordings are surfaced by the pill, not text."""

    def test_stalled_no_speakers_pill_is_not_post_processed_when_disabled(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, tooltip = FloatingSettingsPanel._build_history_status_pill(
            meta, post_processing_enabled=False,
        )
        assert text == "Completed"
        assert kind == "not-post-processed"
        assert "manual action" not in tooltip.lower()

    def test_completed_zero_speakers_pill_is_yellow(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        outcome = PostProcessOutcome(
            status=transcript_footer.STATUS_COMPLETED,
            attempted_at="2026-08-14T10:00:00",
        )
        meta = _make_meta(speaker_count=0, speakers=[], outcome=outcome)
        _, kind, tooltip = FloatingSettingsPanel._build_history_status_pill(meta)
        assert kind == "completed-warning"
        assert "Speakers not identified" in tooltip
