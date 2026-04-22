"""Tests for transcript_scanner module.

Covers T02 must-haves:
- RecordingMeta dataclass with all specified fields
- parse_metadata correctly extracts JSON footer and builds RecordingMeta
- scan_recordings returns sorted list, skips _enhanced.md
- Graceful handling of malformed/missing metadata
"""

import json
from pathlib import Path
from typing import List, Optional

import pytest

from metamemory.transcription.transcript_scanner import (
    RecordingMeta,
    parse_metadata,
    scan_recordings,
)
from metamemory.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_transcript_md(
    path: Path,
    recording_start_time: Optional[str] = "2026-04-22T10:00:00",
    words: Optional[list] = None,
    word_count: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write a minimal transcript .md with embedded metadata footer.

    Args:
        path: Target file path.
        recording_start_time: ISO timestamp for the recording.
        words: List of word dicts (text, start_time, end_time, confidence, speaker_id).
        word_count: Override word_count in metadata. Auto-computed from words if None.
        extra_metadata: Additional keys merged into the metadata dict.

    Returns:
        The written file path.
    """
    if words is None:
        words = []

    metadata = {
        "recording_start_time": recording_start_time,
        "word_count": word_count if word_count is not None else len(words),
        "words": words,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    content = f"# Transcript\n\nSome content.\n\n---\n\n<!-- METADATA: {json.dumps(metadata)} -->\n"
    path.write_text(content, encoding="utf-8")
    return path


def _sample_words() -> list:
    """Return a sample words list with two speakers."""
    return [
        {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Speaker_1"},
        {"text": "world", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "Speaker_1"},
        {"text": "Hi", "start_time": 1.2, "end_time": 1.5, "confidence": 88, "speaker_id": "Speaker_2"},
        {"text": "there", "start_time": 1.5, "end_time": 2.0, "confidence": 92, "speaker_id": "Speaker_2"},
    ]


# ---------------------------------------------------------------------------
# Tests — parse_metadata
# ---------------------------------------------------------------------------

class TestParseMetadata:
    """Tests for parse_metadata function."""

    def test_parses_valid_metadata(self, tmp_path: Path) -> None:
        """parse_metadata extracts all fields from a well-formed .md file."""
        words = _sample_words()
        md = _write_transcript_md(tmp_path / "recording-001.md", words=words)

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.path == md
        assert meta.recording_time == "2026-04-22T10:00:00"
        assert meta.word_count == 4
        assert meta.speaker_count == 2
        assert meta.speakers == ["Speaker_1", "Speaker_2"]
        assert meta.duration_seconds == 2.0  # max end_time
        assert meta.wav_exists is False

    def test_wav_exists_true(self, tmp_path: Path) -> None:
        """wav_exists is True when a companion .wav file exists."""
        md = _write_transcript_md(tmp_path / "recording-001.md", words=_sample_words())
        wav = tmp_path / "recording-001.wav"
        wav.write_bytes(b"\x00")

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.wav_exists is True

    def test_wav_exists_false(self, tmp_path: Path) -> None:
        """wav_exists is False when no companion .wav file."""
        md = _write_transcript_md(tmp_path / "recording-001.md", words=_sample_words())

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.wav_exists is False

    def test_returns_none_for_missing_metadata_footer(self, tmp_path: Path) -> None:
        """parse_metadata returns None when no METADATA footer is present."""
        md = tmp_path / "bare.md"
        md.write_text("# Just a transcript\n\nNo footer here.\n", encoding="utf-8")

        assert parse_metadata(md) is None

    def test_returns_none_for_malformed_json(self, tmp_path: Path) -> None:
        """parse_metadata returns None when the JSON inside the footer is invalid."""
        md = tmp_path / "bad.md"
        md.write_text(
            "# Transcript\n\n---\n\n<!-- METADATA: {not valid json} -->\n",
            encoding="utf-8",
        )

        assert parse_metadata(md) is None

    def test_returns_none_for_unreadable_file(self, tmp_path: Path) -> None:
        """parse_metadata returns None when the file cannot be read."""
        nonexistent = tmp_path / "ghost.md"
        assert parse_metadata(nonexistent) is None

    def test_empty_recording_zero_words(self, tmp_path: Path) -> None:
        """RecordingMeta for a recording with no words."""
        md = _write_transcript_md(tmp_path / "empty.md", words=[])

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.word_count == 0
        assert meta.speaker_count == 0
        assert meta.speakers == []
        assert meta.duration_seconds == 0.0

    def test_duration_from_word_end_times(self, tmp_path: Path) -> None:
        """duration_seconds is the max end_time across all words."""
        words = [
            {"text": "a", "start_time": 0.0, "end_time": 1.0, "confidence": 90, "speaker_id": None},
            {"text": "b", "start_time": 1.0, "end_time": 5.5, "confidence": 90, "speaker_id": None},
            {"text": "c", "start_time": 5.5, "end_time": 3.0, "confidence": 90, "speaker_id": None},
        ]
        md = _write_transcript_md(tmp_path / "dur.md", words=words)

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.duration_seconds == 5.5

    def test_speakers_skip_none(self, tmp_path: Path) -> None:
        """Words with speaker_id=None are not counted as speakers."""
        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "confidence": 90, "speaker_id": None},
        ]
        md = _write_transcript_md(tmp_path / "nospeaker.md", words=words)

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.speaker_count == 0
        assert meta.speakers == []

    def test_unique_speakers_deduped(self, tmp_path: Path) -> None:
        """Same speaker_id appearing in multiple words is listed once."""
        words = [
            {"text": "a", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "A"},
            {"text": "b", "start_time": 0.5, "end_time": 1.0, "confidence": 90, "speaker_id": "A"},
            {"text": "c", "start_time": 1.0, "end_time": 1.5, "confidence": 90, "speaker_id": "B"},
        ]
        md = _write_transcript_md(tmp_path / "dedup.md", words=words)

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.speakers == ["A", "B"]
        assert meta.speaker_count == 2

    def test_no_recording_start_time_defaults_empty(self, tmp_path: Path) -> None:
        """When recording_start_time is missing, recording_time is empty string."""
        md = _write_transcript_md(
            tmp_path / "notime.md",
            recording_start_time=None,
            words=[{"text": "x", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )

        meta = parse_metadata(md)

        assert meta is not None
        assert meta.recording_time == ""

    def test_companion_file_from_real_transcript_store(self, tmp_path: Path) -> None:
        """Verify parse_metadata works on output from TranscriptStore.save_to_file."""
        store = TranscriptStore()
        store.start_recording()
        store.add_words([
            Word(text="real", start_time=0.0, end_time=0.5, confidence=90, speaker_id="S1"),
            Word(text="deal", start_time=0.5, end_time=1.0, confidence=85, speaker_id="S1"),
        ])

        md_path = tmp_path / "live-recording.md"
        store.save_to_file(md_path)

        meta = parse_metadata(md_path)

        assert meta is not None
        assert meta.word_count == 2
        assert meta.speaker_count == 1
        assert meta.speakers == ["S1"]
        assert meta.duration_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests — scan_recordings
# ---------------------------------------------------------------------------

class TestScanRecordings:
    """Tests for scan_recordings function."""

    def test_scans_directory_returns_list(self, tmp_path: Path) -> None:
        """scan_recordings returns parsed recordings from the given directory."""
        _write_transcript_md(
            tmp_path / "recording-a.md",
            recording_start_time="2026-04-20T08:00:00",
            words=[{"text": "hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        _write_transcript_md(
            tmp_path / "recording-b.md",
            recording_start_time="2026-04-21T12:00:00",
            words=[{"text": "world", "start_time": 0.0, "end_time": 1.0, "confidence": 85, "speaker_id": None}],
        )

        results = scan_recordings(tmp_path)

        assert len(results) == 2

    def test_sorts_newest_first(self, tmp_path: Path) -> None:
        """scan_recordings returns results sorted by recording_time descending."""
        _write_transcript_md(
            tmp_path / "recording-old.md",
            recording_start_time="2026-01-01T10:00:00",
            words=[{"text": "old", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        _write_transcript_md(
            tmp_path / "recording-new.md",
            recording_start_time="2026-04-22T10:00:00",
            words=[{"text": "new", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        _write_transcript_md(
            tmp_path / "recording-mid.md",
            recording_start_time="2026-03-15T14:00:00",
            words=[{"text": "mid", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )

        results = scan_recordings(tmp_path)

        assert len(results) == 3
        assert results[0].recording_time == "2026-04-22T10:00:00"
        assert results[1].recording_time == "2026-03-15T14:00:00"
        assert results[2].recording_time == "2026-01-01T10:00:00"

    def test_skips_enhanced_md_files(self, tmp_path: Path) -> None:
        """scan_recordings skips *_enhanced.md files (backwards compat)."""
        _write_transcript_md(
            tmp_path / "recording-good.md",
            recording_start_time="2026-04-22T10:00:00",
            words=[{"text": "good", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        # Write an _enhanced.md file (legacy format)
        _write_transcript_md(
            tmp_path / "recording-old_enhanced.md",
            recording_start_time="2026-04-21T10:00:00",
            words=[{"text": "enhanced", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )

        results = scan_recordings(tmp_path)

        assert len(results) == 1
        assert results[0].path.name == "recording-good.md"

    def test_filters_out_none_results(self, tmp_path: Path) -> None:
        """scan_recordings silently skips files with no valid metadata."""
        _write_transcript_md(
            tmp_path / "valid.md",
            recording_start_time="2026-04-22T10:00:00",
            words=[{"text": "ok", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        # Write a file with no metadata
        (tmp_path / "invalid.md").write_text("# No metadata\n", encoding="utf-8")

        results = scan_recordings(tmp_path)

        assert len(results) == 1
        assert results[0].path.name == "valid.md"

    def test_empty_directory(self, tmp_path: Path) -> None:
        """scan_recordings returns empty list for a directory with no .md files."""
        results = scan_recordings(tmp_path)
        assert results == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """scan_recordings returns empty list for a non-existent directory."""
        ghost = tmp_path / "does_not_exist"
        results = scan_recordings(ghost)
        assert results == []

    def test_wav_exists_in_scan_results(self, tmp_path: Path) -> None:
        """scan_recordings correctly reports wav_exists for each recording."""
        md = _write_transcript_md(
            tmp_path / "recording-with-wav.md",
            recording_start_time="2026-04-22T10:00:00",
            words=[{"text": "audio", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )
        (tmp_path / "recording-with-wav.wav").write_bytes(b"\x00")

        _write_transcript_md(
            tmp_path / "recording-no-wav.md",
            recording_start_time="2026-04-21T10:00:00",
            words=[{"text": "silent", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None}],
        )

        results = scan_recordings(tmp_path)

        assert len(results) == 2
        # Sorted newest first
        assert results[0].wav_exists is True
        assert results[1].wav_exists is False
