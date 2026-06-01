"""Bookmark metadata service for History playback.

Manages bookmark reads/writes on transcript metadata footers using the
same footer marker/rebuild pattern as identity_management.  Bookmarks
are stored as JSON objects with ``name``, ``position_ms``, and
``created_at``; ``created_at`` is the stable id for deletion.

PII constraint: bookmark names are user content.  This module's logger
messages use transcript stem, bookmark count, and position only — never
raw bookmark names or transcript text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from meetandread.utils.file_utils import atomic_write
from typing import Any, Dict, List, Optional

from meetandread.speaker.identity_management import (
    parse_metadata_footer as _parse_metadata_footer,
    split_metadata_footer as _split_metadata_footer,
    _rebuild_transcript,
)

logger = logging.getLogger(__name__)

# Metadata footer parsing delegated to identity_management


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BookmarkError(Exception):
    """Base exception for bookmark metadata operations."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bookmark:
    """A single playback bookmark.

    Attributes:
        name: Display name for the bookmark.
        position_ms: Playback position in milliseconds (>= 0).
        created_at: ISO-8601 timestamp string — the stable identifier
            for deletion.
    """

    name: str
    position_ms: int
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_position(position_ms: int) -> str:
    """Format milliseconds as MM:SS for default bookmark names."""
    total_seconds = max(position_ms, 0) // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _parse_bookmark_entry(entry: Any) -> Optional[Bookmark]:
    """Parse a single bookmark dict, returning None for malformed entries.

    Clamps negative ``position_ms`` to zero.
    """
    if not isinstance(entry, dict):
        return None

    name = entry.get("name")
    position_ms = entry.get("position_ms")
    created_at = entry.get("created_at")

    # All three fields required
    if not isinstance(name, str) or not isinstance(created_at, str):
        return None
    if not isinstance(position_ms, (int, float)):
        return None

    # Clamp negative positions to zero
    position_ms = max(int(position_ms), 0)

    return Bookmark(
        name=name,
        position_ms=position_ms,
        created_at=created_at,
    )


def _read_transcript(path: Path) -> tuple[str, Dict[str, Any]]:
    """Read and parse a transcript file, returning (md_body, metadata).

    Raises BookmarkError if the file is unreadable or has no valid footer.
    """
    if not path.exists():
        raise BookmarkError(f"Transcript file not found: {path.name}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BookmarkError(
            f"Cannot read transcript {path.name}: {exc}"
        ) from exc

    result = _split_metadata_footer(content)
    if result is None:
        raise BookmarkError(f"No metadata footer in {path.name}")

    return result


def _write_transcript(path: Path, md_body: str, metadata: Dict[str, Any]) -> None:
    """Rebuild and write a transcript file atomically."""
    new_content = _rebuild_transcript(md_body, metadata)
    atomic_write(path, new_content)


# ---------------------------------------------------------------------------
# BookmarkManager
# ---------------------------------------------------------------------------


class BookmarkManager:
    """Manages bookmark metadata for a single transcript file.

    Each operation reads/writes the transcript file's metadata footer.
    Callers should construct a new manager or call ``load()`` to refresh
    after external changes.
    """

    def __init__(self, transcript_path: Path) -> None:
        self._path = transcript_path

    def add(
        self,
        position_ms: int,
        name: str = "",
    ) -> Bookmark:
        """Add a bookmark at the given playback position.

        Args:
            position_ms: Playback position in milliseconds.
                Negative values are clamped to zero.
            name: Display name. When empty, generates
                ``"Bookmark at MM:SS"`` from the position.

        Returns:
            The newly created Bookmark.

        Raises:
            BookmarkError: If the transcript is unreadable or malformed.
        """
        position_ms = max(int(position_ms), 0)

        if not name:
            name = f"Bookmark at {_format_position(position_ms)}"

        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc).isoformat()

        bm = Bookmark(name=name, position_ms=position_ms, created_at=created_at)

        md_body, metadata = _read_transcript(self._path)
        bookmarks = metadata.get("bookmarks", [])
        if not isinstance(bookmarks, list):
            bookmarks = []
        bookmarks.append({"name": bm.name, "position_ms": bm.position_ms, "created_at": bm.created_at})
        metadata["bookmarks"] = bookmarks

        _write_transcript(self._path, md_body, metadata)

        logger.info(
            "bookmark_added: stem=%s count=%d position_ms=%d",
            self._path.stem,
            len(bookmarks),
            position_ms,
        )

        return bm

    def delete(self, created_at: str) -> bool:
        """Delete a bookmark by its ``created_at`` identifier.

        Args:
            created_at: The stable timestamp id of the bookmark to remove.

        Returns:
            True if a bookmark was removed, False if not found.

        Raises:
            BookmarkError: If the transcript is unreadable or malformed.
        """
        md_body, metadata = _read_transcript(self._path)
        bookmarks = metadata.get("bookmarks", [])
        if not isinstance(bookmarks, list):
            bookmarks = []

        original_len = len(bookmarks)
        filtered = [bm for bm in bookmarks if bm.get("created_at") != created_at]

        if len(filtered) == original_len:
            return False

        metadata["bookmarks"] = filtered
        _write_transcript(self._path, md_body, metadata)

        logger.info(
            "bookmark_deleted: stem=%s count=%d",
            self._path.stem,
            len(filtered),
        )

        return True

    def list_bookmarks(self) -> List[Bookmark]:
        """Return all bookmarks parsed from the transcript metadata.

        Malformed entries are silently skipped.  Missing ``bookmarks`` key
        defaults to an empty list.

        Raises:
            BookmarkError: If the transcript is unreadable.
        """
        try:
            content = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BookmarkError(
                f"Cannot read transcript {self._path.name}: {exc}"
            ) from exc

        data = _parse_metadata_footer(content)
        if data is None:
            raise BookmarkError(f"No metadata footer in {self._path.name}")

        raw_bookmarks = data.get("bookmarks", [])
        if not isinstance(raw_bookmarks, list):
            logger.warning(
                "bookmarks key is not a list in %s, defaulting to empty",
                self._path.name,
            )
            return []

        result: List[Bookmark] = []
        for entry in raw_bookmarks:
            bm = _parse_bookmark_entry(entry)
            if bm is not None:
                result.append(bm)

        return result

    def load(self) -> None:
        """Force a re-read of the transcript from disk.

        This is a no-op in terms of return value; callers should call
        ``list_bookmarks()`` after ``load()`` to get fresh data.

        Raises:
            BookmarkError: If the transcript is unreadable or has no footer.
        """
        # Validate that the file can be read and parsed
        _read_transcript(self._path)
