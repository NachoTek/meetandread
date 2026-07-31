"""Shared helpers for Transcript Footer migration tests.

The non-identity Transcript Footer consumers were migrated onto the
canonical ``transcript_footer`` interface (issue #49).  Their focused tests
share the same patch-and-fixture shape, so the duplication lives here once.

Only the four public Transcript Footer operations are exercised by these
helpers — no private marker literals or framing constants are referenced.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_store import TranscriptStore


@contextmanager
def patch_parse():
    """Patch the canonical ``parse`` as seen by a migrated module."""
    with patch.object(
        transcript_footer, "parse", wraps=transcript_footer.parse
    ) as mock_parse:
        yield mock_parse


@contextmanager
def patch_split():
    """Patch the canonical ``split`` as seen by a migrated module."""
    with patch.object(
        transcript_footer, "split", wraps=transcript_footer.split
    ) as mock_split:
        yield mock_split


@contextmanager
def patch_join():
    """Patch the canonical ``join`` as seen by a migrated module."""
    with patch.object(
        transcript_footer, "join", wraps=transcript_footer.join
    ) as mock_join:
        yield mock_join


@contextmanager
def patch_split_and_join():
    """Patch the canonical ``split`` and ``join`` together."""
    with patch_split() as mock_split, patch_join() as mock_join:
        yield mock_split, mock_join


def write_transcript(path: Path, body: str, metadata: dict) -> Path:
    """Write a Transcript whose final footer is canonical."""
    path.write_text(transcript_footer.join(body, metadata), encoding="utf-8")
    return path


def fresh_store() -> TranscriptStore:
    """A TranscriptStore with recording started and no words."""
    store = TranscriptStore()
    store.start_recording()
    return store
