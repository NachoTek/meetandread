"""Speaker identity linking reads and rewrites through the canonical Transcript Footer.

``link_identity`` and ``rename_identity`` rewrite a Transcript's footer: they
read the Markdown body plus metadata, update words, segments, speaker matches,
and body headings, then write the file back.  These tests pin their migration
onto the canonical ``transcript_footer`` interface: the read-and-rewrite path
delegates to ``split`` and ``join``, and the metadata-only helper delegates to
``parse``.

Only the four public Transcript Footer operations and the public
``identity_linking`` API are exercised here.  No private marker literals or
framing constants are referenced.
"""

from meetandread.speaker.identity_linking import link_identity, rename_identity
from meetandread.transcription import transcript_footer
from tests.footer_test_helpers import (
    patch_join,
    patch_split,
    write_transcript,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _words_speakers() -> list[dict]:
    return [
        {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
        {"text": "Hi", "start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_1"},
    ]


def _segments_speakers() -> list[dict]:
    return [
        {"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0", "speaker": "SPK_0"},
        {"start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_1", "speaker": "SPK_1"},
    ]


def _speaker_metadata() -> dict:
    return {
        "recording_start_time": "2026-04-22T10:00:00",
        "word_count": 2,
        "words": _words_speakers(),
        "segments": _segments_speakers(),
        "speaker_matches": {"SPK_0": {"identity_name": "SPK_0"}},
    }


_BODY_WITH_HEADINGS = "# Transcript\n\n**SPK_0**\n\nHello\n\n**SPK_1**\n\nHi"


# ---------------------------------------------------------------------------
# Delegation: read-and-rewrite uses canonical split and join
# ---------------------------------------------------------------------------


class TestLinkIdentityDelegatesToCanonicalInterface:
    """link_identity reads through split and writes through join."""

    def test_link_calls_canonical_split_and_join(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY_WITH_HEADINGS, _speaker_metadata())

        with patch_split() as mock_split, patch_join() as mock_join:
            link_identity(md, "SPK_0", "Alice")
            mock_split.assert_called_once()
            mock_join.assert_called_once()

    def test_rename_calls_canonical_split_and_join(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY_WITH_HEADINGS, _speaker_metadata())

        with patch_split() as mock_split, patch_join() as mock_join:
            rename_identity(md, "SPK_0", "Alice")
            mock_split.assert_called_once()
            mock_join.assert_called_once()


# ---------------------------------------------------------------------------
# Round-trip: words, segments, speaker matches, and headings are preserved
# ---------------------------------------------------------------------------


class TestLinkIdentityPreservesThroughCanonicalInterface:
    """Rewrites keep words, segments, speaker matches, and Markdown headings."""

    def test_link_preserves_words_segments_matches_and_headings(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY_WITH_HEADINGS, _speaker_metadata())

        link_identity(md, "SPK_0", "Alice")

        body, metadata = transcript_footer.split(md.read_text(encoding="utf-8"))  # type: ignore[misc]
        assert "**Alice**" in body
        assert "**SPK_0**" not in body
        assert "**SPK_1**" in body
        assert metadata["recording_start_time"] == "2026-04-22T10:00:00"
        assert [w["speaker_id"] for w in metadata["words"]] == ["Alice", "SPK_1"]
        assert [s["speaker_id"] for s in metadata["segments"]] == ["Alice", "SPK_1"]
        assert [s["speaker"] for s in metadata["segments"]] == ["Alice", "SPK_1"]
        assert metadata["speaker_matches"]["SPK_0"]["identity_name"] == "Alice"

    def test_rename_preserves_words_segments_and_headings(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, _BODY_WITH_HEADINGS, _speaker_metadata())

        rename_identity(md, "SPK_0", "Bob")

        body, metadata = transcript_footer.split(md.read_text(encoding="utf-8"))  # type: ignore[misc]
        assert "**Bob**" in body
        assert "**SPK_0**" not in body
        assert "**SPK_1**" in body
        assert [w["speaker_id"] for w in metadata["words"]] == ["Bob", "SPK_1"]


# ---------------------------------------------------------------------------
# Guard: a marker-like footer in the body cannot shadow the real footer
# ---------------------------------------------------------------------------


class TestLinkIdentityBodyMarkerCannotShadow:
    """Identities are read from the LAST footer, so body text quoting the format is harmless."""

    def test_link_ignores_earlier_footer_in_body(self, tmp_path):
        body = (
            "# Transcript\n\n"
            "Earlier footer:\n\n"
            + transcript_footer.join(
                "discarded",
                {"word_count": 0, "words": [{"text": "x", "speaker_id": "FAKE"}]},
            )
            + "More body text.\n"
        )
        md = tmp_path / "rec.md"
        write_transcript(md, body, _speaker_metadata())

        link_identity(md, "SPK_0", "Alice")

        _, metadata = transcript_footer.split(md.read_text(encoding="utf-8"))  # type: ignore[misc]
        # The real footer's words were rewritten; the body's fake footer is untouched.
        assert [w["speaker_id"] for w in metadata["words"]] == ["Alice", "SPK_1"]
