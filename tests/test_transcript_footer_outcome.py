"""The Transcript Footer carries the durable Post-processing Outcome (issue #62).

The Outcome is the terminal result of Post-processing for a Recording —
Completed (including zero-speaker results) or Failed (with the failing stage
and reason).  It lives and dies with the Transcript: this module owns the
canonical ``post_process`` block format inside the Transcript Footer.

These tests pin the Outcome block codec and the file-level read/write
operations through the public ``transcript_footer`` interface only.
"""

from pathlib import Path

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_footer import PostProcessOutcome
from tests.footer_test_helpers import write_transcript


def _completed(**overrides) -> PostProcessOutcome:
    base = dict(
        status=transcript_footer.STATUS_COMPLETED,
        attempted_at="2026-08-14T10:00:00",
    )
    base.update(overrides)
    return PostProcessOutcome(**base)


def _failed(**overrides) -> PostProcessOutcome:
    base = dict(
        status=transcript_footer.STATUS_FAILED,
        stage=transcript_footer.STAGE_TRANSCRIBE,
        error="Post-processing transcription failed: runtime: boom",
        attempted_at="2026-08-14T10:00:00",
    )
    base.update(overrides)
    return PostProcessOutcome(**base)


# ---------------------------------------------------------------------------
# Block codec: to_block / outcome_from_block round-trip
# ---------------------------------------------------------------------------


class TestOutcomeBlockRoundTrip:
    """Outcome blocks encode and decode through the canonical footer shape."""

    def test_completed_round_trip(self):
        outcome = _completed()

        decoded = transcript_footer.outcome_from_block(outcome.to_block())

        assert decoded == outcome

    def test_failed_round_trip_keeps_stage_and_error(self):
        outcome = _failed()

        decoded = transcript_footer.outcome_from_block(outcome.to_block())

        assert decoded == outcome
        assert decoded.stage == transcript_footer.STAGE_TRANSCRIBE
        assert "boom" in decoded.error

    def test_completed_block_has_no_error(self):
        block = _completed().to_block()

        assert "error" not in block
        assert "stage" not in block

    def test_audio_missing_failed_round_trip(self):
        outcome = _failed(stage=transcript_footer.STAGE_AUDIO_MISSING,
                          error="Audio file missing")

        decoded = transcript_footer.outcome_from_block(outcome.to_block())

        assert decoded == outcome


class TestOutcomeFromBlockTolerance:
    """Decoding tolerates unusable blocks by returning None."""

    def test_none_block(self):
        assert transcript_footer.outcome_from_block(None) is None

    def test_non_dict_block(self):
        assert transcript_footer.outcome_from_block("completed") is None

    def test_unknown_status(self):
        block = {"status": "cancelled", "attempted_at": "2026-08-14T10:00:00"}

        assert transcript_footer.outcome_from_block(block) is None

    def test_missing_status(self):
        assert transcript_footer.outcome_from_block({"attempted_at": "x"}) is None

    def test_unknown_stage(self):
        block = {"status": "failed", "stage": "mystery", "attempted_at": "x"}

        assert transcript_footer.outcome_from_block(block) is None


# ---------------------------------------------------------------------------
# read_post_process_outcome: content-level read
# ---------------------------------------------------------------------------


class TestReadOutcomeFromContent:
    """``read_post_process_outcome`` decodes the Outcome from a full Transcript."""

    def test_reads_outcome_from_footer(self):
        content = transcript_footer.join(
            "# Transcript\n\nHello.",
            {"post_process": _failed().to_block(), "word_count": 1},
        )

        assert transcript_footer.read_post_process_outcome(content) == _failed()

    def test_returns_none_without_block(self):
        content = transcript_footer.join("# Transcript", {"word_count": 0})

        assert transcript_footer.read_post_process_outcome(content) is None

    def test_returns_none_without_footer(self):
        assert transcript_footer.read_post_process_outcome("# No footer") is None

    def test_reads_outcome_from_crlf_footer(self):
        """A transcript saved with Windows CRLF line endings still yields its
        Outcome — the durable Outcome must survive Windows editors."""
        content = transcript_footer.join(
            "# Transcript\n\nHello.",
            {"post_process": _failed().to_block(), "word_count": 1},
        ).replace("\n", "\r\n")

        assert transcript_footer.read_post_process_outcome(content) == _failed()


# ---------------------------------------------------------------------------
# write_post_process_outcome: file-level write preserving body + metadata
# ---------------------------------------------------------------------------


class TestWriteOutcomeToFile:
    """``write_post_process_outcome`` edits only the post_process block."""

    def test_write_then_read_round_trip(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello world.",
            {"recording_start_time": "2026-08-14T09:00:00", "word_count": 2},
        )

        ok = transcript_footer.write_post_process_outcome(md, _completed())

        assert ok is True
        assert transcript_footer.read_post_process_outcome(
            md.read_text(encoding="utf-8")
        ) == _completed()

    def test_write_preserves_body_and_other_metadata(self, tmp_path):
        body = "# Transcript\n\nHello world."
        metadata = {
            "recording_start_time": "2026-08-14T09:00:00",
            "word_count": 2,
            "speaker_matches": {"SPK_0": {"identity_name": "Alice"}},
        }
        md = write_transcript(tmp_path / "recording.md", body, metadata)

        transcript_footer.write_post_process_outcome(md, _completed())

        split = transcript_footer.split(md.read_text(encoding="utf-8"))
        assert split is not None
        new_body, new_metadata = split
        assert new_body == body
        assert new_metadata["recording_start_time"] == "2026-08-14T09:00:00"
        assert new_metadata["speaker_matches"] == metadata["speaker_matches"]

    def test_write_replaces_existing_outcome(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md", "# Transcript", {"post_process": _failed().to_block()},
        )

        transcript_footer.write_post_process_outcome(md, _completed())

        assert transcript_footer.read_post_process_outcome(
            md.read_text(encoding="utf-8")
        ) == _completed()

    def test_write_returns_false_when_file_missing(self, tmp_path):
        missing = tmp_path / "nope.md"

        ok = transcript_footer.write_post_process_outcome(missing, _completed())

        assert ok is False

    def test_write_returns_false_without_footer(self, tmp_path):
        md = tmp_path / "bare.md"
        md.write_text("# Just markdown\n", encoding="utf-8")

        ok = transcript_footer.write_post_process_outcome(md, _completed())

        assert ok is False


# ---------------------------------------------------------------------------
# clear_post_process_outcome: file-level removal (issue #63 — Retry)
# ---------------------------------------------------------------------------


class TestClearOutcomeFromFile:
    """``clear_post_process_outcome`` removes only the post_process block."""

    def test_clear_then_read_returns_none(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md",
            "# Transcript\n\nHello world.",
            {"post_process": _failed().to_block(), "word_count": 2},
        )

        ok = transcript_footer.clear_post_process_outcome(md)

        assert ok is True
        assert transcript_footer.read_post_process_outcome(
            md.read_text(encoding="utf-8")
        ) is None

    def test_clear_preserves_body_and_other_metadata(self, tmp_path):
        body = "# Transcript\n\nHello world."
        metadata = {
            "recording_start_time": "2026-08-14T09:00:00",
            "word_count": 2,
            "speaker_matches": {"SPK_0": {"identity_name": "Alice"}},
            "post_process": _failed().to_block(),
        }
        md = write_transcript(tmp_path / "recording.md", body, metadata)

        transcript_footer.clear_post_process_outcome(md)

        split = transcript_footer.split(md.read_text(encoding="utf-8"))
        assert split is not None
        new_body, new_metadata = split
        assert new_body == body
        assert new_metadata["recording_start_time"] == "2026-08-14T09:00:00"
        assert new_metadata["speaker_matches"] == metadata["speaker_matches"]
        assert "post_process" not in new_metadata

    def test_clear_without_existing_outcome_is_false(self, tmp_path):
        md = write_transcript(
            tmp_path / "recording.md", "# Transcript", {"word_count": 0},
        )

        ok = transcript_footer.clear_post_process_outcome(md)

        assert ok is False

    def test_clear_returns_false_when_file_missing(self, tmp_path):
        missing = tmp_path / "nope.md"

        ok = transcript_footer.clear_post_process_outcome(missing)

        assert ok is False

    def test_clear_returns_false_without_footer(self, tmp_path):
        md = tmp_path / "bare.md"
        md.write_text("# Just markdown\n", encoding="utf-8")

        ok = transcript_footer.clear_post_process_outcome(md)

        assert ok is False
