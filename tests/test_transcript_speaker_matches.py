"""Tests for speaker match metadata serialization in TranscriptStore.

Covers T01 must-haves:
- save_to_file(path) remains backward compatible (empty speaker_matches).
- save_to_file(path, speaker_matches={...}) writes valid JSON metadata.
- Matched labels serialize as {identity_name, score, confidence} dicts.
- Unmatched labels serialize as JSON null.
- Default / no-argument saves include an empty speaker_matches map.
- Empty transcript and empty speaker map serialize to valid metadata.
- speaker_matches preserved through to_dict() round-trip.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from meetandread.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_metadata_footer(md_path: Path) -> Dict[str, Any]:
    """Read a transcript .md file and return the parsed JSON metadata dict."""
    text = md_path.read_text(encoding="utf-8")
    prefix = "<!-- METADATA: "
    suffix = " -->"
    start = text.index(prefix) + len(prefix)
    end = text.rindex(suffix)
    return json.loads(text[start:end])


def _store_with_words(*words_args) -> TranscriptStore:
    """Create a TranscriptStore pre-loaded with the given Word tuples.

    Each positional arg is ``(text, start, end, confidence[, speaker_id])``.
    """
    store = TranscriptStore()
    store.start_recording()
    for args in words_args:
        store.add_words([Word(*args)])
    return store


# Sample speaker_matches fixtures
MATCHED_LABELS: Dict[str, Optional[Dict[str, Any]]] = {
    "spk_0": {
        "identity_name": "Alice",
        "score": 0.92,
        "confidence": "high",
    },
    "spk_1": {
        "identity_name": "Bob",
        "score": 0.78,
        "confidence": "medium",
    },
}

MIXED_LABELS: Dict[str, Optional[Dict[str, Any]]] = {
    "spk_0": {
        "identity_name": "Alice",
        "score": 0.92,
        "confidence": "high",
    },
    "spk_1": None,  # unmatched
}


# ---------------------------------------------------------------------------
# Tests — to_dict
# ---------------------------------------------------------------------------

class TestToDictSpeakerMatches:
    """Tests for TranscriptStore.to_dict() speaker_matches parameter."""

    def test_default_empty_speaker_matches(self) -> None:
        """to_dict() with no args returns an empty speaker_matches map."""
        store = TranscriptStore()
        data = store.to_dict()

        assert "speaker_matches" in data
        assert data["speaker_matches"] == {}

    def test_explicit_none_produces_empty_map(self) -> None:
        """to_dict(speaker_matches=None) produces an empty speaker_matches map."""
        store = TranscriptStore()
        data = store.to_dict(speaker_matches=None)

        assert data["speaker_matches"] == {}

    def test_passed_matches_preserved(self) -> None:
        """to_dict(speaker_matches={...}) preserves the mapping exactly."""
        store = TranscriptStore()
        data = store.to_dict(speaker_matches=MATCHED_LABELS)

        assert data["speaker_matches"] == MATCHED_LABELS
        assert data["speaker_matches"]["spk_0"]["identity_name"] == "Alice"

    def test_none_values_in_map(self) -> None:
        """to_dict preserves None (unmatched) entries in speaker_matches."""
        store = TranscriptStore()
        data = store.to_dict(speaker_matches=MIXED_LABELS)

        assert data["speaker_matches"]["spk_1"] is None
        assert data["speaker_matches"]["spk_0"]["identity_name"] == "Alice"

    def test_existing_keys_unchanged(self) -> None:
        """Adding speaker_matches does not alter existing metadata keys."""
        store = _store_with_words(("hello", 0.0, 0.5, 90, "spk_0"))
        data = store.to_dict(speaker_matches=MATCHED_LABELS)

        assert data["word_count"] == 1
        assert len(data["words"]) == 1
        assert data["words"][0]["text"] == "hello"
        assert len(data["segments"]) == 1
        assert data["recording_start_time"] is not None


# ---------------------------------------------------------------------------
# Tests — save_to_file
# ---------------------------------------------------------------------------

class TestSaveToFileSpeakerMatches:
    """Tests for TranscriptStore.save_to_file() with speaker_matches."""

    def test_backward_compatible_no_matches(self, tmp_path: Path) -> None:
        """save_to_file(path) without speaker_matches includes empty map."""
        store = _store_with_words(("hello", 0.0, 0.5, 90, "spk_0"))
        md = tmp_path / "recording.md"
        store.save_to_file(md)

        data = _parse_metadata_footer(md)
        assert data["speaker_matches"] == {}

    def test_matched_labels_serialized(self, tmp_path: Path) -> None:
        """Matched speaker labels serialize as {identity_name, score, confidence}."""
        store = _store_with_words(("hello", 0.0, 0.5, 90, "spk_0"))
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches=MATCHED_LABELS)

        data = _parse_metadata_footer(md)
        matches = data["speaker_matches"]
        assert matches["spk_0"] == {
            "identity_name": "Alice",
            "score": 0.92,
            "confidence": "high",
        }

    def test_unmatched_labels_null(self, tmp_path: Path) -> None:
        """Unmatched speaker labels serialize as JSON null."""
        store = _store_with_words(("hello", 0.0, 0.5, 90, "spk_0"))
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches=MIXED_LABELS)

        data = _parse_metadata_footer(md)
        assert data["speaker_matches"]["spk_1"] is None

    def test_raw_labels_unchanged(self, tmp_path: Path) -> None:
        """Raw diarization labels are preserved as-is (e.g. spk_0, not SPK_0)."""
        labels = {"spk_2": {"identity_name": "Carol", "score": 0.85, "confidence": "high"}}
        store = _store_with_words(("word", 0.0, 0.5, 90, "spk_2"))
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches=labels)

        data = _parse_metadata_footer(md)
        assert "spk_2" in data["speaker_matches"]
        # Verify the file content has the raw label literally
        text = md.read_text(encoding="utf-8")
        assert '"spk_2"' in text

    def test_empty_transcript_empty_matches(self, tmp_path: Path) -> None:
        """Empty transcript with empty speaker_matches serializes validly."""
        store = TranscriptStore()
        store.start_recording()
        md = tmp_path / "empty.md"
        store.save_to_file(md, speaker_matches={})

        data = _parse_metadata_footer(md)
        assert data["speaker_matches"] == {}
        assert data["word_count"] == 0

    def test_empty_matches_with_words(self, tmp_path: Path) -> None:
        """Transcript with words but empty speaker_matches is valid."""
        store = _store_with_words(
            ("alpha", 0.0, 0.5, 90, "spk_0"),
            ("beta", 0.5, 1.0, 85, "spk_1"),
        )
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches={})

        data = _parse_metadata_footer(md)
        assert data["speaker_matches"] == {}
        assert data["word_count"] == 2

    def test_json_serializable_with_matches(self, tmp_path: Path) -> None:
        """The full metadata footer is valid JSON when speaker_matches included."""
        store = _store_with_words(("hi", 0.0, 0.3, 95, "spk_0"))
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches=MIXED_LABELS)

        # If _parse_metadata_footer succeeds, JSON was valid
        data = _parse_metadata_footer(md)
        assert isinstance(data["speaker_matches"], dict)

    def test_none_match_entry_serializes_to_null(self, tmp_path: Path) -> None:
        """Explicit None entry becomes JSON null in the metadata file."""
        store = _store_with_words(("hey", 0.0, 0.3, 95, "spk_99"))
        md = tmp_path / "recording.md"
        store.save_to_file(md, speaker_matches={"spk_99": None})

        data = _parse_metadata_footer(md)
        assert data["speaker_matches"]["spk_99"] is None

    def test_save_without_matches_preserves_word_data(self, tmp_path: Path) -> None:
        """Calling save_to_file without speaker_matches doesn't lose word data."""
        store = _store_with_words(
            ("one", 0.0, 0.5, 90, "spk_0"),
            ("two", 0.5, 1.0, 85, "spk_1"),
        )
        md = tmp_path / "recording.md"
        store.save_to_file(md)

        data = _parse_metadata_footer(md)
        assert len(data["words"]) == 2
        assert data["words"][0]["text"] == "one"
        assert data["words"][1]["text"] == "two"
