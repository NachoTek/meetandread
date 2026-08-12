"""Tests for per-Recording Post-processing lifecycle status (issue #19).

The History list shows each Recording's Post-processing lifecycle state
(Queued / Processing NN% / Failed / Manual Action Required / complete) as a
per-row label, replacing the misleading static '(processing speakers)' text.
The lower transcript viewer keeps showing the Live Transcript so a recent
Recording is readable even when the Post-processor is backed up.

Three seams are covered here:

1. ``PostProcessingQueue.get_status_for_audio`` / ``get_progress_for_audio`` —
   the queue reports a Recording's job status and progress.
2. ``RecordingController.get_post_processing_state`` /
   ``get_post_processing_progress`` — controller façades over the queue,
   tolerant of a disabled (None) queue.
3. ``FloatingSettingsPanel._build_history_display_text`` — pure helper mapping
   (meta, status, progress) to the per-row label, mirroring the existing
   static-helper test pattern (runs without a display context).
"""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

from meetandread.config.models import AppSettings
from meetandread.transcription.post_processor import (
    PostProcessingQueue,
    PostProcessJob,
    PostProcessStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    audio_file: Path,
    job_id: str = "job-1",
    status: PostProcessStatus = PostProcessStatus.PENDING,
    progress: int = 0,
) -> PostProcessJob:
    """Create a minimal PostProcessJob for testing."""
    return PostProcessJob(
        job_id=job_id,
        audio_file=audio_file,
        realtime_transcript=None,
        output_dir=audio_file.parent,
        model_size="base",
        status=status,
        progress=progress,
    )


def _new_queue() -> PostProcessingQueue:
    """Construct a PostProcessingQueue without starting the worker."""
    return PostProcessingQueue(AppSettings())


def _make_controller_with_queue(queue: Optional[object]):
    """Construct a RecordingController and inject *queue* as its post-processor."""
    from meetandread.recording.controller import RecordingController

    ctrl = RecordingController(enable_transcription=True)
    ctrl._post_processor = queue
    return ctrl


# ---------------------------------------------------------------------------
# Slice 1: PostProcessingQueue status + progress queries
# ---------------------------------------------------------------------------


class TestPostProcessingQueueStatusQuery:
    """Verify the queue's status + progress queries used by the History rows."""

    def test_running_job_reports_running_status(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.RUNNING, progress=42)

        assert queue.get_status_for_audio(audio) == PostProcessStatus.RUNNING
        assert queue.get_progress_for_audio(audio) == 42

    def test_pending_job_reports_pending_status(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.PENDING)

        assert queue.get_status_for_audio(audio) == PostProcessStatus.PENDING

    def test_failed_job_reports_failed_status(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.FAILED)

        assert queue.get_status_for_audio(audio) == PostProcessStatus.FAILED

    def test_completed_job_reports_completed_status(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.COMPLETED)

        assert queue.get_status_for_audio(audio) == PostProcessStatus.COMPLETED

    def test_no_job_reports_none(self, tmp_path):
        queue = _new_queue()
        assert queue.get_status_for_audio(tmp_path / "any.wav") is None
        assert queue.get_progress_for_audio(tmp_path / "any.wav") is None

    def test_completed_job_progress_is_none(self, tmp_path):
        """Progress only applies to in-flight jobs."""
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.COMPLETED, progress=100)

        assert queue.get_progress_for_audio(audio) is None

    def test_match_by_stem(self, tmp_path):
        """WAV lives in recordings/; probe via the transcript companion path."""
        stored = tmp_path / "recordings" / "recording-2026-08-12-090000.wav"
        probe = tmp_path / "transcripts" / "recording-2026-08-12-090000.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(stored, status=PostProcessStatus.RUNNING, progress=45)

        assert queue.get_status_for_audio(probe) == PostProcessStatus.RUNNING
        assert queue.get_progress_for_audio(probe) == 45

    def test_pending_prefers_over_terminal_for_same_stem(self, tmp_path):
        """If both a terminal and a non-terminal job exist, the live one wins."""
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["old"] = _make_job(audio, job_id="old", status=PostProcessStatus.FAILED)
        queue._jobs["new"] = _make_job(audio, job_id="new", status=PostProcessStatus.RUNNING, progress=30)

        assert queue.get_status_for_audio(audio) == PostProcessStatus.RUNNING


# ---------------------------------------------------------------------------
# Slice 2: RecordingController state + progress façades
# ---------------------------------------------------------------------------


class TestRecordingControllerStateCheck:
    """Verify the controller façades over the Post-processing queue."""

    def test_state_returns_queue_status(self, tmp_path):
        queue = MagicMock()
        queue.get_status_for_audio.return_value = PostProcessStatus.RUNNING
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_state(md_path) == PostProcessStatus.RUNNING

    def test_state_returns_none_when_queue_reports_none(self, tmp_path):
        queue = MagicMock()
        queue.get_status_for_audio.return_value = None
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_state(md_path) is None

    def test_state_returns_none_when_post_processing_disabled(self, tmp_path):
        """A None queue (post-processing disabled) reports no state."""
        ctrl = _make_controller_with_queue(None)
        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_state(md_path) is None

    def test_state_returns_none_when_queue_raises(self, tmp_path):
        queue = MagicMock()
        queue.get_status_for_audio.side_effect = RuntimeError("boom")
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_state(md_path) is None

    def test_progress_returns_queue_progress(self, tmp_path):
        queue = MagicMock()
        queue.get_progress_for_audio.return_value = 55
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) == 55

    def test_progress_returns_none_when_disabled(self, tmp_path):
        ctrl = _make_controller_with_queue(None)
        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) is None

    def test_state_delegates_transcript_path_to_queue(self, tmp_path):
        queue = MagicMock()
        queue.get_status_for_audio.return_value = None
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-2026-08-12-090000.md"
        ctrl.get_post_processing_state(md_path)
        queue.get_status_for_audio.assert_called_once_with(md_path)


# ---------------------------------------------------------------------------
# Slice 3: FloatingSettingsPanel._build_history_display_text lifecycle labels
# ---------------------------------------------------------------------------


def _make_meta(
    path: str = "recording-2026-01-01-120000.md",
    word_count: int = 100,
    speaker_count: int = 2,
    speakers: Optional[list] = None,
    recording_time: str = "2026-01-01T12:00:00",
    duration_seconds: float = 60.0,
    wav_exists: bool = True,
):
    from meetandread.transcription.transcript_scanner import RecordingMeta

    return RecordingMeta(
        path=Path(path),
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=speaker_count,
        speakers=speakers or [f"SPK_{i}" for i in range(speaker_count)],
        duration_seconds=duration_seconds,
        wav_exists=wav_exists,
    )


class TestHistoryLifecycleDisplay:
    """Verify the per-row lifecycle labels from issue #19."""

    def test_running_shows_processing_percent(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.RUNNING, post_process_progress=45,
        )
        assert "Processing 45%" in text
        assert italic is True

    def test_pending_shows_queued(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.PENDING,
        )
        assert "Queued" in text
        assert italic is True

    def test_failed_shows_failed(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.FAILED,
        )
        assert "Failed" in text
        assert italic is True

    def test_cancelled_shows_manual_action_required(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.CANCELLED,
        )
        assert "Manual Action Required" in text
        assert italic is True

    def test_no_job_no_speakers_shows_manual_action_required(self):
        """Replaces the old '(processing speakers)' for stalled recordings."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "Manual Action Required" in text
        assert "processing speakers" not in text.lower()
        assert italic is True

    def test_completed_via_speakers_shows_count(self):
        """A Recording with speakers is complete — shows speaker count."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "3 speakers" in text
        assert italic is False

    def test_completed_job_falls_through_to_speakers(self):
        """An explicit COMPLETED job also shows the speaker count."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=2)
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.COMPLETED,
        )
        assert "2 speakers" in text
        assert italic is False

    def test_empty_recording_shows_empty_label(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(word_count=0, speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "empty recording" in text.lower()
        assert italic is False

    def test_running_progress_defaults_to_zero_when_unknown(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, _ = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True,
            post_process_status=PostProcessStatus.RUNNING, post_process_progress=None,
        )
        assert "Processing 0%" in text
