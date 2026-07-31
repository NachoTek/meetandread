"""Library scanning reads Recording metadata through the canonical Transcript Footer.

``parse_metadata`` is the Library's read path for saved Recordings.  These
tests pin its migration onto the canonical ``transcript_footer`` interface:
metadata reads delegate to ``transcript_footer.parse``, and an earlier
marker-like footer quoted in the Transcript body can no longer shadow the
real (final) footer.

Only the four public Transcript Footer operations and the public
``parse_metadata`` / ``scan_recordings`` API are exercised here.  No private
marker literals or framing constants are referenced.
"""

import pytest

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_scanner import parse_metadata
from tests.footer_test_helpers import patch_parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transcript_with_footer(body: str, metadata: dict) -> str:
    """Build a Transcript whose final footer is canonical."""
    return transcript_footer.join(body, metadata)


# ---------------------------------------------------------------------------
# Delegation: parse_metadata reads through the canonical parse operation
# ---------------------------------------------------------------------------


class TestScannerDelegatesToCanonicalParse:
    """Library scanning obtains Recording metadata through ``transcript_footer.parse``."""

    def test_parse_metadata_calls_canonical_parse(self, tmp_path):
        md = tmp_path / "recording.md"
        md.write_text(
            _transcript_with_footer(
                "# Transcript\n\nHello.",
                {"recording_start_time": "2026-04-22T10:00:00", "word_count": 1, "words": []},
            ),
            encoding="utf-8",
        )

        with patch_parse() as mock_parse:
            parse_metadata(md)
            mock_parse.assert_called_once()

    def test_parse_metadata_recovers_fields_via_canonical_parse(self, tmp_path):
        md = tmp_path / "recording.md"
        md.write_text(
            _transcript_with_footer(
                "# Transcript\n\nHello world.",
                {
                    "recording_start_time": "2026-04-22T10:00:00",
                    "word_count": 2,
                    "words": [
                        {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "S1"},
                        {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "S1"},
                    ],
                },
            ),
            encoding="utf-8",
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.recording_time == "2026-04-22T10:00:00"
        assert meta.word_count == 2
        assert meta.speakers == ["S1"]


# ---------------------------------------------------------------------------
# Hardening: a marker-like footer in the body cannot shadow the real footer
# ---------------------------------------------------------------------------


class TestBodyMarkerCannotShadowRealFooter:
    """The Library reads the LAST footer, so body text quoting the format is harmless."""

    def test_earlier_complete_footer_in_body_is_ignored(self, tmp_path):
        """A complete earlier footer embedded in the body must not be read as the metadata."""
        body = (
            "# Transcript\n\n"
            "The transcript format appends a footer that looks like:\n\n"
            + transcript_footer.join(
                "discarded body",
                {"recording_start_time": "1999-01-01T00:00:00", "word_count": 999, "words": []},
            )
            + "But that was just a discussion of the format.\n"
        )
        md = tmp_path / "recording.md"
        md.write_text(
            _transcript_with_footer(
                body,
                {"recording_start_time": "2026-04-22T10:00:00", "word_count": 0, "words": []},
            ),
            encoding="utf-8",
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.recording_time == "2026-04-22T10:00:00"
        assert meta.word_count == 0


# ---------------------------------------------------------------------------
# Behavior preserved: malformed / missing footers still skipped
# ---------------------------------------------------------------------------


class TestScannerStillSkipsBadInputs:
    """Migration preserves the Library's tolerance of unusable footers."""

    @pytest.mark.parametrize(
        "content",
        [
            "# Just a transcript\n\nNo footer here.\n",
            "# Transcript\n\n---\n\n<!-- METADATA: {not valid json} -->\n",
        ],
        ids=["no_footer", "malformed_json"],
    )
    def test_returns_none(self, tmp_path, content):
        md = tmp_path / "bad.md"
        md.write_text(content, encoding="utf-8")

        assert parse_metadata(md) is None
