"""Re-transcribe reads Recording time and Speaker matches through the canonical Transcript Footer.

Re-transcription carries the original Recording's ``recording_start_time``
and source Speaker identity assignments forward onto the sidecar store.
These tests pin that migration onto the canonical ``transcript_footer``
interface: both reads delegate to ``transcript_footer.parse``, and an
earlier marker-like footer quoted in the Transcript body can no longer
shadow the real (final) footer.

Only the four public Transcript Footer operations and the public
``RetranscribeRunner`` read helpers are exercised here.  No private marker
literals or framing constants are referenced.
"""

from datetime import datetime
from pathlib import Path

import pytest

from meetandread.transcription import transcript_footer
from meetandread.transcription.retranscribe import RetranscribeRunner
from tests.footer_test_helpers import fresh_store, patch_parse, write_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _preserve_recording_time: reads recording_start_time via canonical parse
# ---------------------------------------------------------------------------


class TestPreserveRecordingTimeDelegatesToCanonicalParse:
    """Re-transcribe preserves the original Recording time through ``transcript_footer.parse``."""

    def test_calls_canonical_parse(self, tmp_path):
        md = write_transcript(
            tmp_path / "original.md",
            "# Transcript",
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 0, "words": []},
        )
        store = fresh_store()

        with patch_parse() as mock_parse:
            RetranscribeRunner._preserve_recording_time(md, store)
            mock_parse.assert_called_once()

    def test_preserves_real_recording_time(self, tmp_path):
        md = write_transcript(
            tmp_path / "original.md",
            "# Transcript",
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 0, "words": []},
        )
        store = fresh_store()

        RetranscribeRunner._preserve_recording_time(md, store)

        assert store._recording_start_time == datetime.fromisoformat("2026-04-22T10:00:00")

    def test_missing_path_is_a_noop(self, tmp_path):
        store = fresh_store()
        before = store._recording_start_time

        RetranscribeRunner._preserve_recording_time(tmp_path / "ghost.md", store)

        assert store._recording_start_time == before


class TestPreserveRecordingTimeBodyMarkerCannotShadow:
    """An earlier footer quoted in the body must not be read as the Recording time."""

    def test_earlier_complete_footer_in_body_is_ignored(self, tmp_path):
        body = (
            "# Transcript\n\n"
            "The footer format looks like:\n\n"
            + transcript_footer.join(
                "discarded",
                {"recording_start_time": "1999-01-01T00:00:00", "word_count": 0, "words": []},
            )
            + "But that was only a discussion of the format.\n"
        )
        md = write_transcript(
            tmp_path / "original.md",
            body,
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 0, "words": []},
        )
        store = fresh_store()

        RetranscribeRunner._preserve_recording_time(md, store)

        assert store._recording_start_time == datetime.fromisoformat("2026-04-22T10:00:00")


# ---------------------------------------------------------------------------
# _read_speaker_identities: recovers source Speaker matches via canonical parse
# ---------------------------------------------------------------------------


class TestReadSpeakerIdentitiesDelegatesToCanonicalParse:
    """Re-transcribe recovers source Speaker matches through ``transcript_footer.parse``."""

    def test_calls_canonical_parse(self, tmp_path):
        md = write_transcript(
            tmp_path / "original.md",
            "# Transcript",
            {"speaker_matches": {"SPK_0": {"identity_name": "Alice"}}},
        )

        with patch_parse() as mock_parse:
            RetranscribeRunner._read_speaker_identities(md)
            mock_parse.assert_called_once()

    def test_recovers_named_identities(self, tmp_path):
        md = write_transcript(
            tmp_path / "original.md",
            "# Transcript",
            {
                "speaker_matches": {
                    "SPK_0": {"identity_name": "Alice"},
                    "SPK_1": {"identity_name": None},
                }
            },
        )

        result = RetranscribeRunner._read_speaker_identities(md)

        assert result == {"SPK_0": "Alice"}


class TestReadSpeakerIdentitiesBodyMarkerCannotShadow:
    """An earlier footer quoted in the body must not shadow the real Speaker matches."""

    def test_earlier_complete_footer_in_body_is_ignored(self, tmp_path):
        body = (
            "# Transcript\n\n"
            "Earlier footer:\n\n"
            + transcript_footer.join(
                "discarded",
                {"speaker_matches": {"FAKE": {"identity_name": "Imposter"}}},
            )
            + "More body text.\n"
        )
        md = write_transcript(
            tmp_path / "original.md",
            body,
            {"speaker_matches": {"SPK_0": {"identity_name": "Commercial Guy"}}},
        )

        result = RetranscribeRunner._read_speaker_identities(md)

        assert result == {"SPK_0": "Commercial Guy"}


# ---------------------------------------------------------------------------
# Behavior preserved: unusable footers yield empty results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "# Transcript\n\nNo footer here.\n",
        "# Transcript\n\n---\n\n<!-- METADATA: {not valid json} -->\n",
    ],
    ids=["no_footer", "malformed_json"],
)
class TestReadSpeakerIdentitiesStillTolerant:
    def test_returns_empty(self, tmp_path, content):
        md = tmp_path / "bad.md"
        md.write_text(content, encoding="utf-8")

        assert RetranscribeRunner._read_speaker_identities(md) == {}
