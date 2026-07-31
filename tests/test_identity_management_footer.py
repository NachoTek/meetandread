"""Speaker identity management reads/rewrites through the canonical Transcript Footer.

Library-wide identity operations scan transcript footers and rewrite them:
``scan_identity_usage`` and ``_find_transcripts_with_label`` read metadata, and
``replace_speaker_label_in_file`` is a read-and-rewrite path.  These tests pin
their migration onto the canonical ``transcript_footer`` interface: metadata
reads delegate to ``parse`` and the read-and-rewrite path delegates to ``split``
and ``join``.

Only the four public Transcript Footer operations and the public
``identity_management`` API are exercised here.  No private marker literals,
framing constants, or local footer helpers are referenced.
"""

from meetandread.speaker.identity_management import (
    _find_transcripts_with_label,
    replace_speaker_label_in_file,
    scan_identity_usage,
)
from meetandread.transcription import transcript_footer
from tests.footer_test_helpers import (
    patch_join,
    patch_parse,
    patch_split,
    write_transcript,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _speaker_metadata(name: str = "SPK_0") -> dict:
    return {
        "recording_start_time": "2026-04-22T10:00:00",
        "word_count": 1,
        "words": [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "speaker_id": name}],
        "segments": [{"start_time": 0.0, "end_time": 0.5, "speaker_id": name, "speaker": name}],
        "speaker_matches": {name: {"identity_name": name}},
    }


_BODY = "# Transcript\n\n**SPK_0**\n\nHi"


# ---------------------------------------------------------------------------
# Delegation: metadata-only reads use canonical parse
# ---------------------------------------------------------------------------


class TestIdentityReadsDelegateToCanonicalParse:
    """scan_identity_usage and _find_transcripts_with_label read through parse."""

    def test_scan_calls_canonical_parse(self, tmp_path):
        md = tmp_path / "transcripts" / "rec.md"
        md.parent.mkdir()
        write_transcript(md, _BODY, _speaker_metadata("Alice"))

        with patch_parse() as mock_parse:
            scan_identity_usage(md.parent, ["Alice"])
            mock_parse.assert_called()

    def test_find_transcripts_with_label_calls_canonical_parse(self, tmp_path):
        md = tmp_path / "transcripts" / "rec.md"
        md.parent.mkdir()
        write_transcript(md, _BODY, _speaker_metadata("SPK_0"))

        with patch_parse() as mock_parse:
            _find_transcripts_with_label(md.parent, "SPK_0")
            mock_parse.assert_called()


# ---------------------------------------------------------------------------
# Delegation: read-and-rewrite uses canonical split and join
# ---------------------------------------------------------------------------


class TestReplaceLabelDelegatesToCanonicalInterface:
    """replace_speaker_label_in_file reads through split and writes through join."""

    def test_replace_calls_canonical_split_and_join(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY, _speaker_metadata("SPK_0"))

        with patch_split() as mock_split, patch_join() as mock_join:
            replace_speaker_label_in_file(md, "SPK_0", "Alice")
            mock_split.assert_called_once()
            mock_join.assert_called_once()

    def test_replace_round_trips_through_canonical_interface(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY, _speaker_metadata("SPK_0"))

        replace_speaker_label_in_file(md, "SPK_0", "Alice")

        body, metadata = transcript_footer.split(md.read_text(encoding="utf-8"))  # type: ignore[misc]
        assert body == _BODY.replace("**SPK_0**", "**Alice**")
        assert metadata["recording_start_time"] == "2026-04-22T10:00:00"
        assert metadata["words"][0]["speaker_id"] == "Alice"
        assert metadata["segments"][0]["speaker_id"] == "Alice"
        assert metadata["segments"][0]["speaker"] == "Alice"
        assert metadata["speaker_matches"]["SPK_0"]["identity_name"] == "Alice"


# ---------------------------------------------------------------------------
# Guard: a marker-like footer in the body cannot shadow the real footer
# ---------------------------------------------------------------------------


class TestIdentityManagementBodyMarkerCannotShadow:
    """Metadata is read from the LAST footer, so body text quoting the format is harmless."""

    def test_scan_ignores_earlier_footer_in_body(self, tmp_path):
        body = (
            "# Transcript\n\n"
            "Earlier footer:\n\n"
            + transcript_footer.join(
                "discarded",
                {"word_count": 1, "words": [{"text": "x", "speaker_id": "FAKE"}]},
            )
            + "More body text.\n"
        )
        md = tmp_path / "transcripts" / "rec.md"
        md.parent.mkdir()
        write_transcript(md, body, _speaker_metadata("Alice"))

        usage = scan_identity_usage(md.parent, ["Alice", "FAKE"])

        assert usage["Alice"].total_mentions == 1
        assert usage["FAKE"].total_mentions == 0
