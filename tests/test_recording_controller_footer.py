"""Recording metadata handling rewrites the Transcript Footer through the canonical interface.

``RecordingController._compute_and_store_wer`` is the Recording metadata
read-and-rewrite path: it reads the post-processed Transcript Footer,
appends a ``wer`` field, and writes the file back.  These tests pin its
migration onto the canonical ``transcript_footer`` interface: the rewrite
goes through ``split`` and ``join``, the body and unrelated footer data are
preserved, and an earlier marker-like footer quoted in the Transcript body
can no longer shadow the real (final) footer.

Only the four public Transcript Footer operations and the public
``RecordingController`` API are exercised here.  No private marker literals
or framing constants are referenced.
"""

from meetandread.recording.controller import RecordingController
from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_store import TranscriptStore, Word
from tests.footer_test_helpers import patch_split_and_join, write_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _controller_with_realtime_words(words: list[str]) -> RecordingController:
    ctrl = RecordingController(enable_transcription=False)
    store = TranscriptStore()
    store.start_recording()
    t = 0.0
    ws = []
    for text in words:
        ws.append(Word(text=text, start_time=t, end_time=t + 0.5, confidence=90))
        t += 0.5
    store.add_words(ws)
    ctrl._transcript_store = store
    return ctrl


# ---------------------------------------------------------------------------
# Delegation: the rewrite goes through canonical split and join
# ---------------------------------------------------------------------------


class TestWerRewriteDelegatesToCanonicalInterface:
    """The WER rewrite reads through ``split`` and writes through ``join``."""

    def test_calls_canonical_split_and_join(self, tmp_path):
        ctrl = _controller_with_realtime_words(["hello", "world"])
        md = tmp_path / "recording.md"
        write_transcript(
            md,
            "# Transcript\n\nHello world.",
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 2,
             "words": [{"text": "hello"}, {"text": "world"}]},
        )

        with patch_split_and_join() as (mock_split, mock_join):
            ctrl._compute_and_store_wer({"transcript_path": str(md)})
            mock_split.assert_called_once()
            mock_join.assert_called_once()

    def test_wer_value_stored_on_controller(self, tmp_path):
        ctrl = _controller_with_realtime_words(["hello", "world"])
        md = tmp_path / "recording.md"
        write_transcript(
            md,
            "# Transcript\n\nHello world.",
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 2,
             "words": [{"text": "hello"}, {"text": "world"}]},
        )

        ctrl._compute_and_store_wer({"transcript_path": str(md)})

        # realtime == post-processed -> WER 0.0
        assert ctrl.get_last_wer() == 0.0


# ---------------------------------------------------------------------------
# Read-and-rewrite: body and unrelated footer data preserved, wer appended
# ---------------------------------------------------------------------------


class TestWerRewritePreservesTranscript:
    """The rewrite keeps the Markdown body and the rest of the footer intact."""

    def test_preserves_body_and_appends_wer(self, tmp_path):
        ctrl = _controller_with_realtime_words(["hello", "world"])
        md = tmp_path / "recording.md"
        body = "# Transcript\n\nHello world."
        write_transcript(
            md,
            body,
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 2,
             "words": [{"text": "hello"}, {"text": "world"}],
             "speaker_matches": {"SPK_0": {"identity_name": "Alice"}}},
        )

        ctrl._compute_and_store_wer({"transcript_path": str(md)})

        result = transcript_footer.split(md.read_text(encoding="utf-8"))
        assert result is not None
        recovered_body, metadata = result
        assert recovered_body == body
        assert metadata["recording_start_time"] == "2026-04-22T10:00:00"
        assert metadata["speaker_matches"]["SPK_0"]["identity_name"] == "Alice"
        assert metadata["wer"] == 0.0


# ---------------------------------------------------------------------------
# Hardening: a marker-like footer in the body cannot shadow the real footer
# ---------------------------------------------------------------------------


class TestWerRewriteBodyMarkerCannotShadow:
    """WER is computed from the LAST footer, so body text quoting the format is harmless."""

    def test_earlier_complete_footer_in_body_is_ignored(self, tmp_path):
        ctrl = _controller_with_realtime_words(["hello", "world"])
        body = (
            "# Transcript\n\nHello world.\n\n"
            "The footer format looks like:\n\n"
            + transcript_footer.join(
                "discarded",
                {"recording_start_time": "1999-01-01T00:00:00", "word_count": 1,
                 "words": [{"text": "goodbye"}]},
            )
            + "But that was only a discussion of the format.\n"
        )
        md = tmp_path / "recording.md"
        write_transcript(
            md,
            body,
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 2,
             "words": [{"text": "hello"}, {"text": "world"}]},
        )

        ctrl._compute_and_store_wer({"transcript_path": str(md)})

        # The real (last) footer's words match realtime -> WER 0.0.
        # The earlier body footer's "goodbye" must not be used.
        assert ctrl.get_last_wer() == 0.0
