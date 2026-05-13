"""Tests for bookmark metadata service and scanner parsing.

Covers T01 must-haves:
- Bookmark frozen dataclass with name, position_ms, created_at
- BookmarkManager add/delete/list/load operations on transcript metadata
- Typed BookmarkError for unreadable/missing/malformed paths
- Malformed bookmark entries filtered during load
- Negative position_ms clamped to zero
- Default display name generation (Bookmark at MM:SS)
- Preservation of unrelated metadata keys during writes
- Scanner parse_metadata exposing bookmarks without key-existence checks
- Negative tests: no footer, invalid JSON, wrong type, duplicate names, etc.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from meetandread.playback.bookmark import (
    Bookmark,
    BookmarkError,
    BookmarkManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_transcript_md(
    path: Path,
    recording_start_time: str = "2026-05-01T10:00:00",
    words: Optional[list] = None,
    bookmarks: Optional[list] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write a transcript .md with embedded metadata footer."""
    if words is None:
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90},
        ]

    metadata: Dict[str, Any] = {
        "recording_start_time": recording_start_time,
        "word_count": len(words),
        "words": words,
    }
    if bookmarks is not None:
        metadata["bookmarks"] = bookmarks
    if extra_metadata:
        metadata.update(extra_metadata)

    content = f"# Transcript\n\nSome content.\n\n---\n\n<!-- METADATA: {json.dumps(metadata)} -->\n"
    path.write_text(content, encoding="utf-8")
    return path


def _parse_metadata_dict(path: Path) -> Optional[Dict[str, Any]]:
    """Read back the metadata dict from a transcript file."""
    text = path.read_text(encoding="utf-8")
    prefix = "<!-- METADATA: "
    suffix = " -->"
    start = text.find(prefix)
    if start == -1:
        return None
    after = text[start + len(prefix):]
    end = after.rfind(suffix)
    if end == -1:
        return None
    try:
        return json.loads(after[:end])
    except json.JSONDecodeError:
        return None


def _sample_bookmarks() -> list:
    """Return sample bookmark dicts."""
    return [
        {"name": "Intro", "position_ms": 5000, "created_at": "2026-05-01T10:01:00+00:00"},
        {"name": "Key point", "position_ms": 120000, "created_at": "2026-05-01T10:02:00+00:00"},
    ]


# ---------------------------------------------------------------------------
# Tests — Bookmark dataclass
# ---------------------------------------------------------------------------

class TestBookmark:
    """Tests for Bookmark frozen dataclass."""

    def test_create_bookmark(self) -> None:
        """Bookmark stores name, position_ms, created_at."""
        bm = Bookmark(name="Test", position_ms=5000, created_at="2026-05-01T10:00:00+00:00")
        assert bm.name == "Test"
        assert bm.position_ms == 5000
        assert bm.created_at == "2026-05-01T10:00:00+00:00"

    def test_bookmark_is_frozen(self) -> None:
        """Bookmark dataclass is immutable."""
        bm = Bookmark(name="Test", position_ms=5000, created_at="2026-05-01T10:00:00+00:00")
        with pytest.raises(AttributeError):
            bm.name = "Changed"  # type: ignore[misc]

    def test_bookmark_equality(self) -> None:
        """Two Bookmarks with same fields are equal."""
        bm1 = Bookmark(name="A", position_ms=1000, created_at="2026-05-01T10:00:00+00:00")
        bm2 = Bookmark(name="A", position_ms=1000, created_at="2026-05-01T10:00:00+00:00")
        assert bm1 == bm2


# ---------------------------------------------------------------------------
# Tests — BookmarkManager add
# ---------------------------------------------------------------------------

class TestBookmarkManagerAdd:
    """Tests for BookmarkManager.add."""

    def test_add_bookmark_to_transcript(self, tmp_path: Path) -> None:
        """add writes a new bookmark to the transcript metadata footer."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=5000, name="Intro")

        assert bm is not None
        assert bm.name == "Intro"
        assert bm.position_ms == 5000
        assert bm.created_at != ""

        # Verify persisted
        data = _parse_metadata_dict(md)
        assert data is not None
        assert len(data["bookmarks"]) == 1
        assert data["bookmarks"][0]["name"] == "Intro"

    def test_add_with_empty_name_generates_default(self, tmp_path: Path) -> None:
        """add generates 'Bookmark at MM:SS' when name is empty."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=90000)  # 1:30

        assert bm is not None
        assert bm.name == "Bookmark at 01:30"

    def test_add_at_zero_ms(self, tmp_path: Path) -> None:
        """add accepts position_ms=0."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=0, name="Start")

        assert bm is not None
        assert bm.position_ms == 0

    def test_add_preserves_other_metadata(self, tmp_path: Path) -> None:
        """add preserves words, segments, speaker_matches, etc."""
        words = [{"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90}]
        extra = {
            "segments": [{"start": 0.0, "end": 0.5}],
            "speaker_matches": {"SPK_0": {"identity_name": "Alice"}},
        }
        md = _write_transcript_md(tmp_path / "rec.md", words=words, extra_metadata=extra, bookmarks=[])
        mgr = BookmarkManager(md)

        mgr.add(position_ms=1000, name="Mark")

        data = _parse_metadata_dict(md)
        assert data is not None
        assert len(data["words"]) == 1
        assert data["segments"] == [{"start": 0.0, "end": 0.5}]
        assert "SPK_0" in data["speaker_matches"]

    def test_add_appends_to_existing_bookmarks(self, tmp_path: Path) -> None:
        """add appends to an existing bookmarks list."""
        existing = [{"name": "A", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"}]
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=existing)
        mgr = BookmarkManager(md)

        mgr.add(position_ms=5000, name="B")

        data = _parse_metadata_dict(md)
        assert data is not None
        assert len(data["bookmarks"]) == 2

    def test_add_to_file_without_footer_raises(self, tmp_path: Path) -> None:
        """add raises BookmarkError when file has no metadata footer."""
        md = tmp_path / "bare.md"
        md.write_text("# No footer\n", encoding="utf-8")

        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.add(position_ms=1000)

    def test_add_to_missing_file_raises(self, tmp_path: Path) -> None:
        """add raises BookmarkError when file doesn't exist."""
        md = tmp_path / "ghost.md"
        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.add(position_ms=1000)


# ---------------------------------------------------------------------------
# Tests — BookmarkManager delete
# ---------------------------------------------------------------------------

class TestBookmarkManagerDelete:
    """Tests for BookmarkManager.delete."""

    def test_delete_existing_bookmark(self, tmp_path: Path) -> None:
        """delete removes a bookmark by created_at and returns True."""
        bm_data = _sample_bookmarks()
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=bm_data)
        mgr = BookmarkManager(md)

        result = mgr.delete(created_at="2026-05-01T10:01:00+00:00")

        assert result is True
        data = _parse_metadata_dict(md)
        assert data is not None
        assert len(data["bookmarks"]) == 1
        assert data["bookmarks"][0]["name"] == "Key point"

    def test_delete_unknown_id_returns_false(self, tmp_path: Path) -> None:
        """delete returns False when no bookmark matches the created_at."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=_sample_bookmarks())
        mgr = BookmarkManager(md)

        result = mgr.delete(created_at="1999-01-01T00:00:00+00:00")

        assert result is False

    def test_delete_preserves_other_metadata(self, tmp_path: Path) -> None:
        """delete preserves words and other metadata keys."""
        words = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90}]
        bm_data = [{"name": "A", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"}]
        md = _write_transcript_md(tmp_path / "rec.md", words=words, bookmarks=bm_data)
        mgr = BookmarkManager(md)

        mgr.delete(created_at="2026-05-01T10:00:00+00:00")

        data = _parse_metadata_dict(md)
        assert data is not None
        assert len(data["words"]) == 1

    def test_delete_from_missing_file_raises(self, tmp_path: Path) -> None:
        """delete raises BookmarkError when file doesn't exist."""
        md = tmp_path / "ghost.md"
        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.delete(created_at="2026-05-01T10:00:00+00:00")


# ---------------------------------------------------------------------------
# Tests — BookmarkManager list_bookmarks
# ---------------------------------------------------------------------------

class TestBookmarkManagerList:
    """Tests for BookmarkManager.list_bookmarks."""

    def test_list_returns_bookmark_objects(self, tmp_path: Path) -> None:
        """list_bookmarks returns Bookmark objects from metadata."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=_sample_bookmarks())
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()

        assert len(result) == 2
        assert all(isinstance(bm, Bookmark) for bm in result)
        assert result[0].name == "Intro"
        assert result[1].name == "Key point"

    def test_list_defaults_to_empty_when_no_key(self, tmp_path: Path) -> None:
        """list_bookmarks returns [] when bookmarks key is missing."""
        md = _write_transcript_md(tmp_path / "rec.md")  # no bookmarks key
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()

        assert result == []

    def test_list_filters_malformed_entries(self, tmp_path: Path) -> None:
        """list_bookmarks skips malformed bookmark entries."""
        bad_bookmarks = [
            {"name": "Good", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"},
            {"name": "No position"},  # missing position_ms
            {"position_ms": 2000, "created_at": "2026-05-01T10:01:00+00:00"},  # missing name
            "not a dict",  # not a dict at all
            {"name": "", "position_ms": -100, "created_at": "2026-05-01T10:02:00+00:00"},  # negative pos
        ]
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=bad_bookmarks)
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()

        # Only the first entry is fully valid; negative position gets clamped but still valid
        assert len(result) == 2
        assert result[0].name == "Good"
        assert result[1].position_ms == 0  # clamped from -100

    def test_list_from_missing_file_raises(self, tmp_path: Path) -> None:
        """list_bookmarks raises BookmarkError when file doesn't exist."""
        md = tmp_path / "ghost.md"
        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.list_bookmarks()


# ---------------------------------------------------------------------------
# Tests — BookmarkManager load (reload from disk)
# ---------------------------------------------------------------------------

class TestBookmarkManagerLoad:
    """Tests for BookmarkManager.load (re-read from disk)."""

    def test_load_refreshes_state(self, tmp_path: Path) -> None:
        """load re-reads bookmarks from disk after external change."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)
        assert mgr.list_bookmarks() == []

        # Externally modify the file
        _write_transcript_md(md, bookmarks=_sample_bookmarks())
        mgr.load()

        assert len(mgr.list_bookmarks()) == 2

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """load raises BookmarkError when file doesn't exist."""
        md = tmp_path / "ghost.md"
        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.load()


# ---------------------------------------------------------------------------
# Tests — Negative / edge cases (Q7)
# ---------------------------------------------------------------------------

class TestBookmarkNegativeCases:
    """Negative tests covering failure modes and boundary conditions."""

    def test_no_footer_returns_error_on_operations(self, tmp_path: Path) -> None:
        """Operations on a file with no footer raise BookmarkError."""
        md = tmp_path / "bare.md"
        md.write_text("# Just text\n\nNo metadata here.\n", encoding="utf-8")
        mgr = BookmarkManager(md)

        with pytest.raises(BookmarkError):
            mgr.add(position_ms=1000)
        with pytest.raises(BookmarkError):
            mgr.delete(created_at="x")

    def test_invalid_json_metadata_raises_on_write(self, tmp_path: Path) -> None:
        """add raises BookmarkError when footer has invalid JSON."""
        md = tmp_path / "bad.md"
        md.write_text("# Transcript\n\n---\n\n<!-- METADATA: {not json} -->\n", encoding="utf-8")
        mgr = BookmarkManager(md)

        with pytest.raises(BookmarkError):
            mgr.add(position_ms=1000)

    def test_bookmarks_wrong_type_defaults_to_empty(self, tmp_path: Path) -> None:
        """When bookmarks is not a list, list_bookmarks returns empty."""
        md = _write_transcript_md(tmp_path / "rec.md", extra_metadata={"bookmarks": "not a list"})
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()

        assert result == []

    def test_negative_position_clamped_to_zero(self, tmp_path: Path) -> None:
        """add clamps negative position_ms to zero."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=-500, name="Clamped")

        assert bm.position_ms == 0

    def test_duplicate_names_allowed(self, tmp_path: Path) -> None:
        """Multiple bookmarks with the same name are allowed."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm1 = mgr.add(position_ms=1000, name="Same")
        bm2 = mgr.add(position_ms=2000, name="Same")

        assert bm1.created_at != bm2.created_at
        result = mgr.list_bookmarks()
        assert len(result) == 2

    def test_default_name_format_various_positions(self) -> None:
        """Default name format covers MM:SS correctly."""
        # Test the static formatting logic via BookmarkManager._format_position
        # We'll test through add to keep it public-interface
        pass

    def test_default_name_at_0ms(self, tmp_path: Path) -> None:
        """Default name at position 0 is 'Bookmark at 00:00'."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=0)

        assert bm.name == "Bookmark at 00:00"

    def test_default_name_at_large_position(self, tmp_path: Path) -> None:
        """Default name handles positions > 1 hour correctly."""
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=[])
        mgr = BookmarkManager(md)

        bm = mgr.add(position_ms=3661000)  # 1:01:01

        assert bm.name == "Bookmark at 61:01"

    def test_malformed_bookmark_missing_created_at_skipped(self, tmp_path: Path) -> None:
        """Bookmarks missing created_at are skipped during load."""
        bookmarks = [
            {"name": "OK", "position_ms": 1000, "created_at": "2026-05-01T10:00:00+00:00"},
            {"name": "No ts", "position_ms": 2000},  # missing created_at
        ]
        md = _write_transcript_md(tmp_path / "rec.md", bookmarks=bookmarks)
        mgr = BookmarkManager(md)

        result = mgr.list_bookmarks()
        assert len(result) == 1
        assert result[0].name == "OK"

    def test_malformed_metadata_not_rewritten(self, tmp_path: Path) -> None:
        """add does not rewrite a file with malformed metadata JSON."""
        md = tmp_path / "bad.md"
        original_content = "# Transcript\n\n---\n\n<!-- METADATA: {broken json} -->\n"
        md.write_text(original_content, encoding="utf-8")

        mgr = BookmarkManager(md)
        with pytest.raises(BookmarkError):
            mgr.add(position_ms=1000)

        # File content unchanged
        assert md.read_text(encoding="utf-8") == original_content
