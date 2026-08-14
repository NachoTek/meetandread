"""The Library scanner surfaces the Post-processing Outcome (issue #62).

``parse_metadata`` is the Library's read path for saved Recordings.  It now
decodes the durable Post-processing Outcome from the Transcript Footer so
the Library UI can render per-row status pills.  A Recording with no Outcome
is Stalled — represented here simply as ``post_process_outcome is None``.
"""

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_footer import PostProcessOutcome
from meetandread.transcription.transcript_scanner import parse_metadata
from tests.footer_test_helpers import write_transcript


def _failed_outcome() -> PostProcessOutcome:
    return PostProcessOutcome(
        status=transcript_footer.STATUS_FAILED,
        stage=transcript_footer.STAGE_ENGINE_LOAD,
        error="Could not load model",
        attempted_at="2026-08-14T10:00:00",
    )


class TestScannerSurfacesOutcome:
    """RecordingMeta carries the Outcome decoded from the Transcript Footer."""

    def test_failed_outcome_surfaced(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello.",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "post_process": _failed_outcome().to_block(),
            },
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.post_process_outcome == _failed_outcome()

    def test_completed_outcome_surfaced(self, tmp_path):
        outcome = PostProcessOutcome(
            status=transcript_footer.STATUS_COMPLETED,
            attempted_at="2026-08-14T10:00:00",
        )
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello.",
            {"post_process": outcome.to_block()},
        )

        meta = parse_metadata(md)

        assert meta.post_process_outcome == outcome

    def test_no_outcome_is_stalled_none(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello.",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.post_process_outcome is None

    def test_unrecognised_outcome_block_is_stalled_none(self, tmp_path):
        """A footer written by another version must read as Stalled, not crash."""
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello.",
            {"post_process": {"status": "mystery"}},
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.post_process_outcome is None
