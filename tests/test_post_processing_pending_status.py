"""Tests for the 'Post Processing pending' History detail status (issue #12).

The History detail view replaces the fragile live-transcription preview
with a simple status message while Post-processing is still in flight for
a Recording. Once Post-processing completes, the normal speaker-labeled
transcript is shown.

This file covers three seams that together decide the status:

1. ``PostProcessingQueue.has_pending_job_for_audio`` — does the queue have
   a PENDING or RUNNING job for a given Recording's audio?
2. ``RecordingController.is_post_processing_pending`` — controller-level
   façade over the queue, tolerant of a disabled (None) queue.
3. ``FloatingSettingsPanel._history_detail_status`` — pure helper returning
   the status string (or None to render normally), mirroring the existing
   ``_build_history_display_text`` static-helper pattern.

Tests target the pure-logic seams so they run without a display context.
"""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from meetandread.config.models import AppSettings
from meetandread.transcription.post_processor import (
    PostProcessingQueue,
    PostProcessJob,
    PostProcessStatus,
)


# ---------------------------------------------------------------------------
# Slice 1: PostProcessingQueue.has_pending_job_for_audio
# ---------------------------------------------------------------------------


def _make_job(
    audio_file: Path,
    job_id: str = "job-1",
    status: PostProcessStatus = PostProcessStatus.PENDING,
) -> PostProcessJob:
    """Create a minimal PostProcessJob for testing."""
    return PostProcessJob(
        job_id=job_id,
        audio_file=audio_file,
        realtime_transcript=None,
        output_dir=audio_file.parent,
        model_size="base",
        status=status,
    )


def _new_queue() -> PostProcessingQueue:
    """Construct a PostProcessingQueue without starting the worker."""
    return PostProcessingQueue(AppSettings())


class TestPostProcessingQueuePendingQuery:
    """Verify the queue's pending-job query used by the History view."""

    def test_pending_job_matches_returns_true(self, tmp_path):
        audio = tmp_path / "recording-2026-01-01-120000.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.PENDING)

        assert queue.has_pending_job_for_audio(audio) is True

    def test_running_job_matches_returns_true(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.RUNNING)

        assert queue.has_pending_job_for_audio(audio) is True

    def test_completed_job_returns_false(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.COMPLETED)

        assert queue.has_pending_job_for_audio(audio) is False

    def test_failed_job_returns_false(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.FAILED)

        assert queue.has_pending_job_for_audio(audio) is False

    def test_cancelled_job_returns_false(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.CANCELLED)

        assert queue.has_pending_job_for_audio(audio) is False

    def test_no_jobs_returns_false(self, tmp_path):
        queue = _new_queue()
        assert queue.has_pending_job_for_audio(tmp_path / "any.wav") is False

    def test_different_recording_returns_false(self, tmp_path):
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(tmp_path / "recording-a.wav")
        assert queue.has_pending_job_for_audio(tmp_path / "recording-b.wav") is False

    def test_match_by_stem_not_whole_path(self, tmp_path):
        """A recording's WAV lives in recordings/ but may be probed by stem.

        The queue stores absolute audio paths; callers may pass an
        equivalently-stemmed path. Matching by stem keeps the check robust.
        """
        stored = tmp_path / "recordings" / "recording-2026-01-01-120000.wav"
        probe = tmp_path / "transcripts" / "recording-2026-01-01-120000.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(stored, status=PostProcessStatus.RUNNING)

        assert queue.has_pending_job_for_audio(probe) is True


# ---------------------------------------------------------------------------
# Slice 2: RecordingController.is_post_processing_pending
# ---------------------------------------------------------------------------


def _make_controller_with_queue(queue: Optional[object]):
    """Construct a RecordingController and inject *queue* as its post-processor."""
    from meetandread.recording.controller import RecordingController

    ctrl = RecordingController(enable_transcription=True)
    ctrl._post_processor = queue
    return ctrl


class TestRecordingControllerPendingCheck:
    """Verify the controller façade used by the History detail view."""

    def test_returns_true_when_queue_reports_pending(self, tmp_path):
        queue = MagicMock()
        queue.has_pending_job_for_audio.return_value = True
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-2026-01-01-120000.md"
        assert ctrl.is_post_processing_pending(md_path) is True

    def test_returns_false_when_queue_reports_not_pending(self, tmp_path):
        queue = MagicMock()
        queue.has_pending_job_for_audio.return_value = False
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.is_post_processing_pending(md_path) is False

    def test_returns_false_when_post_processing_disabled(self, tmp_path):
        """A None queue (post-processing disabled) is never 'pending'."""
        ctrl = _make_controller_with_queue(None)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.is_post_processing_pending(md_path) is False

    def test_returns_false_when_queue_raises(self, tmp_path):
        """A failing query must not crash the History view."""
        queue = MagicMock()
        queue.has_pending_job_for_audio.side_effect = RuntimeError("boom")
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.is_post_processing_pending(md_path) is False

    def test_delegates_transcript_path_to_queue(self, tmp_path):
        """The transcript .md path is forwarded to the queue (stem-matched)."""
        queue = MagicMock()
        queue.has_pending_job_for_audio.return_value = False
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-2026-08-12-090000.md"
        ctrl.is_post_processing_pending(md_path)

        queue.has_pending_job_for_audio.assert_called_once_with(md_path)

    def test_end_to_end_with_real_queue_pending(self, tmp_path):
        """Controller + real queue: a RUNNING job is recognised as pending."""
        audio = tmp_path / "recordings" / "recording-2026-08-12-090000.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.RUNNING)
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-2026-08-12-090000.md"
        assert ctrl.is_post_processing_pending(md_path) is True


# ---------------------------------------------------------------------------
# Slice 3: FloatingSettingsPanel._history_detail_status
# ---------------------------------------------------------------------------


class TestHistoryDetailStatusHelper:
    """Verify the pure helper that decides the History detail status text.

    Mirrors the static-helper test pattern in test_post_processing_indicator
    so it runs without constructing the full widget (no display required).
    """

    def test_pending_returns_status_message(self, tmp_path):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        controller = MagicMock()
        controller.is_post_processing_pending.return_value = True
        md_path = tmp_path / "transcripts" / "recording-x.md"

        status = FloatingSettingsPanel._history_detail_status(md_path, controller)
        assert status is not None
        assert "post processing pending" in status.lower()

    def test_not_pending_returns_none(self, tmp_path):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        controller = MagicMock()
        controller.is_post_processing_pending.return_value = False
        md_path = tmp_path / "transcripts" / "recording-x.md"

        assert FloatingSettingsPanel._history_detail_status(md_path, controller) is None

    def test_no_controller_returns_none(self, tmp_path):
        """Panel constructed without a controller renders normally."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert FloatingSettingsPanel._history_detail_status(md_path, None) is None

    def test_controller_without_method_returns_none(self, tmp_path):
        """A controller lacking the façade degrades to normal rendering."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        md_path = tmp_path / "transcripts" / "recording-x.md"
        # Plain object has no is_post_processing_pending attribute
        assert FloatingSettingsPanel._history_detail_status(md_path, object()) is None

    def test_message_text_matches_spec(self, tmp_path):
        """Issue #12 wording: 'Post Processing pending...'."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        controller = MagicMock()
        controller.is_post_processing_pending.return_value = True
        md_path = tmp_path / "transcripts" / "recording-x.md"

        status = FloatingSettingsPanel._history_detail_status(md_path, controller)
        assert status == "Post Processing pending..."

    def test_controller_raises_returns_none(self, tmp_path):
        """A failing controller check must not crash the History view."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        controller = MagicMock()
        controller.is_post_processing_pending.side_effect = RuntimeError("boom")
        md_path = tmp_path / "transcripts" / "recording-x.md"

        assert FloatingSettingsPanel._history_detail_status(md_path, controller) is None


# ---------------------------------------------------------------------------
# Slice 4: PostProcessingQueue.get_progress_for_audio
# ---------------------------------------------------------------------------


class TestPostProcessingQueueProgressQuery:
    """Verify the queue's progress query used to render the percentage."""

    def test_pending_job_returns_its_progress(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        job = _make_job(audio, status=PostProcessStatus.PENDING)
        job.progress = 25
        queue._jobs["job-1"] = job

        assert queue.get_progress_for_audio(audio) == 25

    def test_running_job_returns_its_progress(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        job = _make_job(audio, status=PostProcessStatus.RUNNING)
        job.progress = 80
        queue._jobs["job-1"] = job

        assert queue.get_progress_for_audio(audio) == 80

    def test_zero_progress_returned_for_fresh_pending_job(self, tmp_path):
        """A just-queued job reports 0%, which is distinct from 'no job'."""
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.PENDING)

        assert queue.get_progress_for_audio(audio) == 0

    def test_completed_job_returns_none(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.COMPLETED)

        assert queue.get_progress_for_audio(audio) is None

    def test_failed_job_returns_none(self, tmp_path):
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        queue._jobs["job-1"] = _make_job(audio, status=PostProcessStatus.FAILED)

        assert queue.get_progress_for_audio(audio) is None

    def test_no_jobs_returns_none(self, tmp_path):
        queue = _new_queue()
        assert queue.get_progress_for_audio(tmp_path / "any.wav") is None

    def test_match_by_stem(self, tmp_path):
        stored = tmp_path / "recordings" / "recording-2026-08-12-090000.wav"
        probe = tmp_path / "transcripts" / "recording-2026-08-12-090000.wav"
        queue = _new_queue()
        job = _make_job(stored, status=PostProcessStatus.RUNNING)
        job.progress = 45
        queue._jobs["job-1"] = job

        assert queue.get_progress_for_audio(probe) == 45

    def test_has_pending_delegates_to_progress_query(self, tmp_path):
        """has_pending_job_for_audio stays consistent with get_progress_for_audio."""
        audio = tmp_path / "recording-x.wav"
        queue = _new_queue()
        job = _make_job(audio, status=PostProcessStatus.RUNNING)
        job.progress = 60
        queue._jobs["job-1"] = job

        pending = queue.has_pending_job_for_audio(audio)
        progress = queue.get_progress_for_audio(audio)
        assert pending is True
        assert progress == 60
        # consistency: progress is not None iff pending is True
        assert (progress is not None) == pending


# ---------------------------------------------------------------------------
# Slice 5: RecordingController.get_post_processing_progress
# ---------------------------------------------------------------------------


class TestRecordingControllerProgressCheck:
    """Verify the controller façade for progress percentage."""

    def test_returns_progress_when_queue_reports_pending(self, tmp_path):
        queue = MagicMock()
        queue.get_progress_for_audio.return_value = 42
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) == 42

    def test_returns_none_when_queue_reports_none(self, tmp_path):
        queue = MagicMock()
        queue.get_progress_for_audio.return_value = None
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) is None

    def test_returns_none_when_post_processing_disabled(self, tmp_path):
        """A None queue (post-processing disabled) reports no progress."""
        ctrl = _make_controller_with_queue(None)
        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) is None

    def test_returns_none_when_queue_raises(self, tmp_path):
        queue = MagicMock()
        queue.get_progress_for_audio.side_effect = RuntimeError("boom")
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-x.md"
        assert ctrl.get_post_processing_progress(md_path) is None

    def test_delegates_transcript_path_to_queue(self, tmp_path):
        queue = MagicMock()
        queue.get_progress_for_audio.return_value = None
        ctrl = _make_controller_with_queue(queue)

        md_path = tmp_path / "transcripts" / "recording-2026-08-12-090000.md"
        ctrl.get_post_processing_progress(md_path)
        queue.get_progress_for_audio.assert_called_once_with(md_path)


# ---------------------------------------------------------------------------
# Slice 6: FloatingSettingsPanel._format_post_processing_status
# ---------------------------------------------------------------------------


class TestFormatPostProcessingStatus:
    """Verify the pure helper that formats the pending status text."""

    def test_known_progress_appends_percentage(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        assert (
            FloatingSettingsPanel._format_post_processing_status(45)
            == "Post Processing pending... 45%"
        )

    def test_zero_progress_appends_zero(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        assert (
            FloatingSettingsPanel._format_post_processing_status(0)
            == "Post Processing pending... 0%"
        )

    def test_none_progress_omits_percentage(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        assert (
            FloatingSettingsPanel._format_post_processing_status(None)
            == "Post Processing pending..."
        )

    def test_full_progress_still_pending(self):
        """100% may be reported before the completion refresh re-renders."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        assert (
            FloatingSettingsPanel._format_post_processing_status(100)
            == "Post Processing pending... 100%"
        )
