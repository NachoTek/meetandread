"""Tests for transcript management — post-processing in-place overwrite.

Covers T01 must-haves:
- _save_post_processed_transcript writes to {stem}.md, never {stem}_enhanced.md
- When the original .md already exists, it gets overwritten
- The result dict contains "transcript_path" key (not "enhanced_path")
- Controller callback reads "transcript_path" from result
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metamemory.transcription.post_processor import (
    PostProcessJob,
    PostProcessStatus,
    PostProcessingQueue,
)
from metamemory.transcription.transcript_store import TranscriptStore, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store_with_words(*texts: str) -> TranscriptStore:
    """Create a TranscriptStore with simple words from *texts*."""
    store = TranscriptStore()
    store.start_recording()
    words = [
        Word(text=t, start_time=i * 1.0, end_time=i * 1.0 + 0.9, confidence=90)
        for i, t in enumerate(texts)
    ]
    store.add_words(words)
    return store


def _make_job(tmp_path: Path) -> PostProcessJob:
    """Create a minimal PostProcessJob pointing at *tmp_path*."""
    audio_file = tmp_path / "recording_001.wav"
    audio_file.write_bytes(b"\x00")  # placeholder
    realtime = _make_store_with_words("hello world")
    return PostProcessJob(
        job_id="test-job",
        audio_file=audio_file,
        realtime_transcript=realtime,
        output_dir=tmp_path,
        model_size="base",
    )


# ---------------------------------------------------------------------------
# Tests — in-place overwrite (not _enhanced.md)
# ---------------------------------------------------------------------------

class TestPostProcessInPlaceOverwrite:
    """Verify post-processing writes {stem}.md and overwrites if it exists."""

    def test_writes_stem_md_not_enhanced(self, tmp_path: Path) -> None:
        """Post-processing must create {stem}.md, never {stem}_enhanced.md."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)
        store = _make_store_with_words("post", "processed", "result")

        result_path = ppq._save_post_processed_transcript(job, store)

        # The returned path must be {stem}.md
        assert result_path.name == "recording_001.md"
        assert result_path.exists()

        # No _enhanced.md variant should be created
        enhanced = tmp_path / "recording_001_enhanced.md"
        assert not enhanced.exists()

    def test_overwrites_existing_md(self, tmp_path: Path) -> None:
        """When original .md exists, post-processing must overwrite it."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # Create an existing transcript .md
        existing_md = tmp_path / "recording_001.md"
        existing_md.write_text("# Old transcript\n\nOld content here.", encoding="utf-8")
        assert existing_md.exists()

        new_store = _make_store_with_words("new", "content")
        result_path = ppq._save_post_processed_transcript(job, new_store)

        # Same file path
        assert result_path == existing_md

        # Content must be overwritten — old marker text gone
        content = result_path.read_text(encoding="utf-8")
        assert "Old content here" not in content
        assert "new" in content

    def test_creates_md_when_missing(self, tmp_path: Path) -> None:
        """When no original .md exists, post-processing creates one."""
        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # No pre-existing .md
        transcript_md = tmp_path / "recording_001.md"
        assert not transcript_md.exists()

        store = _make_store_with_words("fresh", "transcript")
        result_path = ppq._save_post_processed_transcript(job, store)

        assert result_path.exists()
        assert result_path.name == "recording_001.md"
        content = result_path.read_text(encoding="utf-8")
        assert "fresh" in content


# ---------------------------------------------------------------------------
# Tests — result dict key
# ---------------------------------------------------------------------------

class TestPostProcessResultKey:
    """Verify the result dict uses 'transcript_path', not 'enhanced_path'."""

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_result_dict_has_transcript_path_key(
        self, mock_load_audio, mock_engine, tmp_path: Path
    ) -> None:
        """After _process_job, result dict must contain 'transcript_path'."""
        import numpy as np

        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ppq = PostProcessingQueue(settings=settings)
        job = _make_job(tmp_path)

        # Stub audio loading
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        # Stub engine transcription — return empty segments
        mock_eng = MagicMock()
        mock_eng.transcribe_chunk.return_value = []
        mock_engine.return_value = mock_eng

        ppq._process_job(job)

        assert job.status == PostProcessStatus.COMPLETED
        assert "transcript_path" in job.result
        assert "enhanced_path" not in job.result

        # transcript_path must point to {stem}.md
        assert Path(job.result["transcript_path"]).name == "recording_001.md"


# ---------------------------------------------------------------------------
# Tests — controller callback reads transcript_path
# ---------------------------------------------------------------------------

class TestControllerCallback:
    """Verify RecordingController._on_post_process_complete_callback reads
    'transcript_path' from the result dict."""

    def test_controller_callback_reads_transcript_path(self, tmp_path: Path) -> None:
        """Controller callback must read 'transcript_path', not 'enhanced_path'."""
        from metamemory.recording.controller import RecordingController

        captured: dict = {}

        def on_complete(job_id: str, path: Path) -> None:
            captured["job_id"] = job_id
            captured["path"] = path

        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_post_process_complete = on_complete

        transcript_md = tmp_path / "recording_001.md"
        transcript_md.write_text("# transcript", encoding="utf-8")

        result = {
            "transcript_path": str(transcript_md),
            "word_count": 5,
            "realtime_word_count": 3,
            "model_used": "base",
        }

        ctrl._on_post_process_complete_callback("job-42", result)

        assert captured["job_id"] == "job-42"
        assert captured["path"] == transcript_md

    def test_controller_callback_ignores_enhanced_path(self, tmp_path: Path) -> None:
        """If result dict only has 'enhanced_path', callback must not fire."""
        from metamemory.recording.controller import RecordingController

        captured: dict = {}

        def on_complete(job_id: str, path: Path) -> None:
            captured["path"] = path

        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_post_process_complete = on_complete

        # Simulate a stale result with only enhanced_path
        result = {
            "enhanced_path": str(tmp_path / "recording_001_enhanced.md"),
            "word_count": 5,
        }

        ctrl._on_post_process_complete_callback("job-42", result)

        # Callback should NOT have fired — no transcript_path key
        assert "path" not in captured
