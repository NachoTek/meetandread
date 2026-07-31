"""Transcript persistence uses the canonical Transcript Footer write operation.

``TranscriptStore.save_to_file`` is the write-side source of truth for saved
Transcripts.  These tests verify that saving delegates Transcript Footer
construction to the canonical ``transcript_footer.join`` operation and that the resulting file
round-trips through the public ``parse`` / ``split`` interface, regardless of
the Markdown body's trailing-newline state.

Only the four public Transcript Footer operations and the public
``TranscriptStore.save_to_file`` API are exercised here.  No private marker
literals, framing constants, or parser helpers are referenced.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def patch_join():
    """Patch the canonical ``join`` as seen by ``transcript_store``.

    ``save_to_file`` resolves ``join`` off the ``transcript_footer`` module at
    call time, so patching that attribute records the delegation while still
    producing a real file via the wrapped operation.
    """
    with patch.object(
        transcript_footer,
        "join",
        wraps=transcript_footer.join,
    ) as mock_join:
        yield mock_join


def _store_with_hello(speaker_id: str = "SPK_0") -> TranscriptStore:
    """Build a minimal TranscriptStore holding a single word."""
    store = TranscriptStore()
    store.start_recording()
    store.add_words([Word("Hello", 0.0, 0.5, 90, speaker_id=speaker_id)])
    return store


# ---------------------------------------------------------------------------
# Delegation: save_to_file constructs the Transcript Footer through the canonical join
# ---------------------------------------------------------------------------


class TestSaveDelegatesToCanonicalJoin:
    """Saving a Transcript delegates Transcript Footer construction to ``transcript_footer.join``."""

    def test_save_calls_canonical_join(self, tmp_path):
        store = _store_with_hello()
        dest = tmp_path / "transcript.md"

        with patch_join() as mock_join:
            store.save_to_file(dest)
            mock_join.assert_called_once()

    def test_join_receives_markdown_body_and_metadata(self, tmp_path):
        store = _store_with_hello(speaker_id="SPK_1")
        dest = tmp_path / "transcript.md"

        with patch_join() as mock_join:
            store.save_to_file(dest, speaker_matches={"SPK_1": {"identity_name": "Alice"}})
            body, metadata = mock_join.call_args.args

        assert "Hello" in body
        assert metadata["word_count"] == 1
        assert metadata["recording_start_time"] is not None
        assert metadata["speaker_matches"]["SPK_1"]["identity_name"] == "Alice"


# ---------------------------------------------------------------------------
# Round-trip through the canonical public interface
# ---------------------------------------------------------------------------


class TestSavedFileRoundTripsThroughCanonicalInterface:
    """Newly saved files parse and split successfully through the public interface."""

    def test_parse_recovers_full_metadata(self, tmp_path):
        store = _store_with_hello(speaker_id="SPK_0")
        dest = tmp_path / "transcript.md"
        store.save_to_file(dest, speaker_matches={"SPK_0": {"identity_name": "David"}})

        content = dest.read_text(encoding="utf-8")
        metadata = transcript_footer.parse(content)

        assert metadata is not None
        assert metadata["recording_start_time"] is not None
        assert metadata["word_count"] == 1
        assert metadata["words"][0]["text"] == "Hello"
        assert len(metadata["segments"]) == 1
        assert metadata["segments"][0]["speaker_id"] == "SPK_0"
        assert metadata["speaker_matches"]["SPK_0"]["identity_name"] == "David"

    def test_split_recovers_body_and_metadata(self, tmp_path):
        store = _store_with_hello()
        dest = tmp_path / "transcript.md"
        store.save_to_file(dest)

        content = dest.read_text(encoding="utf-8")
        result = transcript_footer.split(content)

        assert result is not None
        body, metadata = result
        assert "# Transcript" in body
        assert "Hello" in body
        assert metadata["word_count"] == 1

    def test_saved_markdown_retains_horizontal_rule_separation(self, tmp_path):
        """The human-readable body keeps a horizontal rule separating it from the Transcript Footer."""
        store = _store_with_hello()
        dest = tmp_path / "transcript.md"
        store.save_to_file(dest)

        content = dest.read_text(encoding="utf-8")
        # A blank line precedes the thematic break so it stays a horizontal
        # rule rather than a setext heading underline.
        assert "\n\n---\n" in content
        body, _ = transcript_footer.split(content)
        assert body.startswith("# Transcript")


# ---------------------------------------------------------------------------
# Trailing-newline states: the body round-trips byte-for-byte through save
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trailing",
    ["", "\n", "\n\n", "\n\n\n"],
    ids=["no_newline", "one_newline", "two_newlines", "many_newlines"],
)
class TestSavedBodyRoundTripsAcrossTrailingNewlines:
    """Persistence preserves the Markdown body for any trailing-newline state."""

    def test_split_recovers_body_exactly(self, trailing, tmp_path, monkeypatch):
        store = _store_with_hello()
        body = "# Transcript\n\nHello world." + trailing

        monkeypatch.setattr(TranscriptStore, "to_markdown", lambda self, **kw: body)
        dest = tmp_path / "transcript.md"
        store.save_to_file(dest)

        content = dest.read_text(encoding="utf-8")
        result = transcript_footer.split(content)
        assert result is not None
        recovered_body, metadata = result
        assert recovered_body == body
        assert metadata["word_count"] == 1
