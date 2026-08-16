"""Accepting a re-transcribe must preserve the Post-processing Outcome.

Bug report (QA on #61/#62): after accepting a re-transcribed version of a
Completed recording, its pill flipped to Queued.  Cause: the sidecar is
promoted over the canonical transcript with a fresh Transcript Footer that
carries no ``post_process`` Outcome — the recording became Stalled and the
requeue scan re-queued it (which would later overwrite the user's accepted
version with the post-processing output).

Accept semantics (fix): the canonical Outcome carries into the promoted
transcript; an Outcome-less canonical gains a Completed Outcome — an
explicit user accept is a terminal Post-processing state, so the requeue
scan can never clobber it.
"""

from pathlib import Path

from meetandread.transcription import transcript_footer
from meetandread.transcription.retranscribe import RetranscribeRunner
from meetandread.transcription.transcript_footer import PostProcessOutcome

from tests.footer_test_helpers import write_transcript


def _outcome_at(path: Path):
    return transcript_footer.read_post_process_outcome(
        path.read_text(encoding="utf-8")
    )


def _make_sidecar(md_path: Path, model: str = "small") -> Path:
    sidecar = RetranscribeRunner.sidecar_path(md_path, model)
    write_transcript(
        sidecar,
        "# Re-transcribed\n\nnew words",
        {"recording_start_time": "2026-08-14T09:00:00", "word_count": 2},
    )
    return sidecar


class TestAcceptPreservesOutcome:
    def test_completed_outcome_carries_into_promoted_transcript(self, tmp_path):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path,
            "# Original\n\nold words",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "post_process": PostProcessOutcome(
                    status=transcript_footer.STATUS_COMPLETED,
                    attempted_at="2026-08-14T10:00:00",
                ).to_block(),
            },
        )
        _make_sidecar(md_path)

        RetranscribeRunner.accept_retranscribe(md_path, "small")

        # The accepted body survived the promotion...
        assert "Re-transcribed" in md_path.read_text(encoding="utf-8")
        # ...and so did the Outcome — the recording stays Completed,
        # never Stalled/Queued.
        outcome = _outcome_at(md_path)
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_COMPLETED
        assert outcome.attempted_at == "2026-08-14T10:00:00"

    def test_failed_outcome_carries_too(self, tmp_path):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path,
            "# Original",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "post_process": PostProcessOutcome(
                    status=transcript_footer.STATUS_FAILED,
                    attempted_at="2026-08-14T10:00:00",
                    stage=transcript_footer.STAGE_DEPENDENCY,
                    error="sherpa-onnx is required for Speaker identification.",
                    dependency="sherpa-onnx",
                ).to_block(),
            },
        )
        _make_sidecar(md_path)

        RetranscribeRunner.accept_retranscribe(md_path, "small")

        outcome = _outcome_at(md_path)
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_FAILED
        assert outcome.stage == transcript_footer.STAGE_DEPENDENCY
        assert outcome.dependency == "sherpa-onnx"


class TestAcceptCompletesOutcomelessRecording:
    def test_stalled_recording_gains_completed_outcome(self, tmp_path):
        """A canonical with no Outcome (Stalled) must not stay Stalled
        after an accept — the requeue scan would later overwrite the
        user's accepted transcript with the post-processing output."""
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path,
            "# Original\n\nold words",
            {"recording_start_time": "2026-08-14T09:00:00", "word_count": 2},
        )
        assert _outcome_at(md_path) is None
        _make_sidecar(md_path)

        RetranscribeRunner.accept_retranscribe(md_path, "small")

        outcome = _outcome_at(md_path)
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_COMPLETED
        assert outcome.attempted_at  # stamped with the accept time

    def test_other_metadata_is_preserved(self, tmp_path):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path,
            "# Original",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "custom_field": "keep me",
            },
        )
        _make_sidecar(md_path)

        RetranscribeRunner.accept_retranscribe(md_path, "small")

        content = md_path.read_text(encoding="utf-8")
        # The promoted file keeps its own footer fields plus the Outcome;
        # the canonical's unrelated fields are not smuggled in beyond
        # what the sidecar already preserved (recording time).
        assert transcript_footer.parse(content) is not None
        assert _outcome_at(md_path) is not None


class TestAcceptEdgeCases:
    def test_missing_sidecar_still_raises(self, tmp_path):
        import pytest

        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path, "# Original", {"recording_start_time": "2026-08-14T09:00:00"}
        )

        with pytest.raises(FileNotFoundError):
            RetranscribeRunner.accept_retranscribe(md_path, "small")

    def test_outcomeless_canonical_without_usable_footer_promotes_anyway(
        self, tmp_path
    ):
        """A canonical without a parseable footer still gets promoted;
        the Outcome write is best-effort and must not block the accept."""
        md_path = tmp_path / "recording_a.md"
        md_path.write_text("# no footer here\n\njust text", encoding="utf-8")
        _make_sidecar(md_path)

        result = RetranscribeRunner.accept_retranscribe(md_path, "small")

        assert result == md_path
        assert "Re-transcribed" in md_path.read_text(encoding="utf-8")
