"""Tests for speaker match metadata wiring in RecordingController.

Covers T02 must-haves:
- _speaker_matches_metadata() returns {} when no diarization result.
- _speaker_matches_metadata() returns {} when diarization failed.
- _speaker_matches_metadata() returns {} when segments are empty.
- _speaker_matches_metadata() serialises matched labels with
  identity_name/score/confidence and unmatched labels as None.
- _save_transcript() writes speaker_matches into the saved .md file.
- get_diagnostics() exposes sanitised match counts/labels without names.
- Multiple segments for the same label produce a single metadata entry.
- Saving still returns None on filesystem exceptions through existing path.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

import pytest

from meetandread.recording.controller import RecordingController
from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.speaker.models import (
    DiarizationResult,
    SpeakerSegment,
    SpeakerMatch,
    VoiceSignature,
)


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


def _make_controller_with_store(tmp_path: Path) -> RecordingController:
    """Create a controller with a transcript store and last_wav_path set."""
    ctrl = RecordingController(enable_transcription=False)
    ctrl._transcript_store = TranscriptStore()
    ctrl._transcript_store.start_recording()
    ctrl._transcript_store.add_words([
        Word("hello", 0.0, 0.5, 90),
        Word("world", 0.5, 1.0, 85),
    ])
    ctrl._last_wav_path = tmp_path / "recording.wav"
    ctrl._last_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ctrl._last_wav_path.write_text("fake wav")
    return ctrl


def _make_diarization_result(
    segments: list | None = None,
    matches: dict | None = None,
    error: str | None = None,
) -> DiarizationResult:
    """Build a DiarizationResult with sensible defaults."""
    return DiarizationResult(
        segments=segments or [],
        matches=matches or {},
        error=error,
        num_speakers=len({s.speaker for s in (segments or [])}),
    )


# ---------------------------------------------------------------------------
# _speaker_matches_metadata tests
# ---------------------------------------------------------------------------

class TestSpeakerMatchesMetadata:
    """Tests for RecordingController._speaker_matches_metadata()."""

    def test_returns_empty_when_no_result(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        assert ctrl._speaker_matches_metadata() == {}

    def test_returns_empty_when_failed_result(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        ctrl._last_diarization_result = _make_diarization_result(error="boom")
        assert ctrl._speaker_matches_metadata() == {}

    def test_returns_empty_when_empty_segments(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        ctrl._last_diarization_result = _make_diarization_result(segments=[])
        assert ctrl._speaker_matches_metadata() == {}

    def test_matched_label_serialised(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
        ]
        matches = {
            "spk0": SpeakerMatch(name="Alice", score=0.92, confidence="high"),
        }
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches=matches,
        )
        result = ctrl._speaker_matches_metadata()
        assert result == {
            "spk0": {
                "identity_name": "Alice",
                "score": 0.92,
                "confidence": "high",
            },
        }

    def test_unmatched_label_is_none(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.0, end=2.0, speaker="spk1"),
        ]
        matches = {
            "spk0": SpeakerMatch(name="Alice", score=0.92, confidence="high"),
        }
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches=matches,
        )
        result = ctrl._speaker_matches_metadata()
        assert result["spk1"] is None
        assert result["spk0"]["identity_name"] == "Alice"

    def test_multiple_segments_same_label_single_entry(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.5, end=2.5, speaker="spk0"),
            SpeakerSegment(start=3.0, end=4.0, speaker="spk0"),
        ]
        matches = {
            "spk0": SpeakerMatch(name="Bob", score=0.85, confidence="medium"),
        }
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches=matches,
        )
        result = ctrl._speaker_matches_metadata()
        # Only one entry for spk0, not three
        assert len(result) == 1
        assert "spk0" in result
        assert result["spk0"]["identity_name"] == "Bob"

    def test_all_unmatched_labels(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.0, end=2.0, speaker="spk1"),
        ]
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches={},
        )
        result = ctrl._speaker_matches_metadata()
        assert result == {"spk0": None, "spk1": None}

    def test_result_without_succeeded_attribute_returns_empty(self, tmp_path: Path):
        """Defensive: object with missing 'succeeded' attribute returns {}."""
        ctrl = _make_controller_with_store(tmp_path)
        ctrl._last_diarization_result = object()
        assert ctrl._speaker_matches_metadata() == {}


# ---------------------------------------------------------------------------
# _save_transcript integration tests
# ---------------------------------------------------------------------------

class TestSaveTranscriptWithSpeakerMatches:
    """Tests that _save_transcript() persists speaker_matches into the .md file."""

    @patch("meetandread.recording.controller.RecordingController._speaker_matches_metadata")
    def test_metadata_in_saved_file(self, mock_meta, tmp_path: Path):
        """Transcript .md contains speaker_matches in metadata footer."""
        ctrl = _make_controller_with_store(tmp_path)
        mock_meta.return_value = {
            "spk0": {
                "identity_name": "Alice",
                "score": 0.92,
                "confidence": "high",
            },
            "spk1": None,
        }

        # Patch get_transcripts_dir to return tmp_path
        with patch(
            "meetandread.recording.controller.get_transcripts_dir",
            return_value=tmp_path,
            create=True,
        ):
            transcript_path = ctrl._save_transcript()

        assert transcript_path is not None
        assert transcript_path.exists()
        metadata = _parse_metadata_footer(transcript_path)
        assert "speaker_matches" in metadata
        assert metadata["speaker_matches"]["spk0"]["identity_name"] == "Alice"
        assert metadata["speaker_matches"]["spk1"] is None

    def test_no_diarization_saves_empty_matches(self, tmp_path: Path):
        """When no diarization result, saved metadata has empty speaker_matches."""
        ctrl = _make_controller_with_store(tmp_path)

        with patch(
            "meetandread.recording.controller.get_transcripts_dir",
            return_value=tmp_path,
            create=True,
        ):
            transcript_path = ctrl._save_transcript()

        assert transcript_path is not None
        metadata = _parse_metadata_footer(transcript_path)
        assert metadata["speaker_matches"] == {}

    def test_save_returns_none_on_filesystem_error(self, tmp_path: Path):
        """_save_transcript returns None when the save itself fails."""
        ctrl = _make_controller_with_store(tmp_path)

        with patch(
            "meetandread.recording.controller.get_transcripts_dir",
            return_value=tmp_path,
            create=True,
        ):
            # Force save_to_file to raise
            with patch.object(
                TranscriptStore, "save_to_file", side_effect=OSError("disk full")
            ):
                result = ctrl._save_transcript()

        assert result is None


# ---------------------------------------------------------------------------
# get_diagnostics sanitisation tests
# ---------------------------------------------------------------------------

class TestDiagnosticsSanitised:
    """Tests that get_diagnostics() includes sanitised speaker-match info."""

    def test_diagnostics_with_diarization(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.0, end=2.0, speaker="spk1"),
        ]
        matches = {
            "spk0": SpeakerMatch(name="Alice", score=0.92, confidence="high"),
        }
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches=matches,
        )

        diag = ctrl.get_diagnostics()
        dia = diag["diarization"]
        assert dia["succeeded"] is True
        assert dia["segment_count"] == 2
        assert dia["match_count"] == 1
        assert dia["matched_label_count"] == 1
        assert sorted(dia["labels"]) == ["spk0", "spk1"]

    def test_diagnostics_no_speaker_names_leaked(self, tmp_path: Path):
        """Diagnostics dict must not contain identity names."""
        ctrl = _make_controller_with_store(tmp_path)
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
        ]
        matches = {
            "spk0": SpeakerMatch(name="SecretName", score=0.9, confidence="high"),
        }
        ctrl._last_diarization_result = _make_diarization_result(
            segments=segments, matches=matches,
        )

        diag = ctrl.get_diagnostics()
        diag_str = json.dumps(diag)
        assert "SecretName" not in diag_str
        # Ensure the key-level dict doesn't leak either
        assert "Alice" not in diag_str
        assert "identity_name" not in diag_str

    def test_diagnostics_no_diarization(self, tmp_path: Path):
        ctrl = _make_controller_with_store(tmp_path)
        diag = ctrl.get_diagnostics()
        assert "diarization" not in diag
