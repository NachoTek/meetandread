"""Bookmark read-and-rewrite goes through the canonical Transcript Footer.

``BookmarkManager`` edits a Transcript's footer to add, delete, and list
playback bookmarks.  These tests pin its migration onto the canonical
``transcript_footer`` interface: the read-and-rewrite path delegates to
``split`` and ``join`` and the read-only path delegates to ``parse``.

Only the four public Transcript Footer operations and the public
``BookmarkManager`` API are exercised here.  No private marker literals,
framing constants, or identity_management helpers are referenced.
"""

from pathlib import Path

from meetandread.playback.bookmark import BookmarkManager
from meetandread.transcription import transcript_footer
from tests.footer_test_helpers import patch_join, patch_parse, patch_split, write_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Delegation: read-and-rewrite uses canonical split and join
# ---------------------------------------------------------------------------


class TestBookmarkRewriteDelegatesToCanonicalInterface:
    """Bookmark add/delete read through ``split`` and write through ``join``."""

    def test_add_calls_canonical_split_and_join(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(md, "# Transcript", {"word_count": 0, "words": [], "bookmarks": []})
        mgr = BookmarkManager(md)

        with patch_split() as mock_split, patch_join() as mock_join:
            mgr.add(position_ms=1000, name="Intro")
            mock_split.assert_called_once()
            mock_join.assert_called_once()

    def test_delete_calls_canonical_split_and_join(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(
            md,
            "# Transcript",
            {"word_count": 0, "words": [],
             "bookmarks": [{"name": "A", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"}]},
        )
        mgr = BookmarkManager(md)

        with patch_split() as mock_split, patch_join() as mock_join:
            assert mgr.delete(created_at="2026-05-01T10:00:00+00:00") is True
            mock_split.assert_called_once()
            mock_join.assert_called_once()

    def test_add_round_trips_through_canonical_interface(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(
            md,
            "# Transcript\n\nBody.",
            {"recording_start_time": "2026-04-22T10:00:00", "word_count": 0, "words": [],
             "speaker_matches": {"SPK_0": {"identity_name": "Alice"}}, "bookmarks": []},
        )
        mgr = BookmarkManager(md)

        mgr.add(position_ms=5000, name="Intro")

        body, metadata = transcript_footer.split(md.read_text(encoding="utf-8"))  # type: ignore[misc]
        assert body == "# Transcript\n\nBody."
        assert metadata["recording_start_time"] == "2026-04-22T10:00:00"
        assert metadata["speaker_matches"]["SPK_0"]["identity_name"] == "Alice"
        assert len(metadata["bookmarks"]) == 1
        assert metadata["bookmarks"][0]["name"] == "Intro"


class TestBookmarkReadDelegatesToCanonicalParse:
    """Bookmark listing reads through ``transcript_footer.parse``."""

    def test_list_calls_canonical_parse(self, tmp_path):
        md = tmp_path / "rec.md"
        write_transcript(
            md,
            "# Transcript",
            {"word_count": 0, "words": [],
             "bookmarks": [{"name": "A", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"}]},
        )
        mgr = BookmarkManager(md)

        with patch_parse() as mock_parse:
            mgr.list_bookmarks()
            mock_parse.assert_called_once()


# ---------------------------------------------------------------------------
# Guard: a marker-like footer in the body cannot shadow the real footer
# ---------------------------------------------------------------------------


class TestBookmarkBodyMarkerCannotShadow:
    """Bookmarks are read from the LAST footer, so body text quoting the format is harmless."""

    def test_list_ignores_earlier_footer_in_body(self, tmp_path):
        body = (
            "# Transcript\n\n"
            "Earlier footer:\n\n"
            + transcript_footer.join(
                "discarded",
                {"word_count": 0, "words": [],
                 "bookmarks": [{"name": "FAKE", "position_ms": 1, "created_at": "1999-01-01T00:00:00+00:00"}]},
            )
            + "More body text.\n"
        )
        md = tmp_path / "rec.md"
        write_transcript(
            md,
            body,
            {"word_count": 0, "words": [],
             "bookmarks": [{"name": "REAL", "position_ms": 2000, "created_at": "2026-05-01T10:00:00+00:00"}]},
        )
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()

        assert len(result) == 1
        assert result[0].name == "REAL"
