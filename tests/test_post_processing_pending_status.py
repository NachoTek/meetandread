"""Tests for per-Recording Post-processing lifecycle status (issues #19, #62).

The Library's History list shows each Recording's Post-processing state as
a per-row color-coded status pill (Queued / Processing NN% / Completed /
Failed / Not post-processed) with a tooltip.  The conflated 'Manual Action
Required' label is gone: a Recording with no Outcome is Stalled and is
re-queued automatically when its Audio exists and Post-processing is
enabled.

Four seams are covered here:

1. ``PostProcessingQueue.get_status_for_audio`` / ``get_progress_for_audio`` —
   the queue reports a Recording's job status and progress.
2. ``RecordingController.get_post_processing_state`` /
   ``get_post_processing_progress`` / ``requeue_stalled_recordings`` —
   controller façades over the queue, tolerant of a disabled (None) queue.
3. ``FloatingSettingsPanel._build_history_display_text`` — pure helper mapping
   (meta, status) to the row's text (label + words + speaker count).
4. ``FloatingSettingsPanel._build_history_status_pill`` — pure helper mapping
   (meta, status, outcome, post-processing-enabled) to the row's pill.
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


class TestRecordingControllerRequeueFaçade:
    """Verify the controller façade over the Stalled requeue scan (issue #62)."""

    def test_requeue_delegates_to_queue(self):
        queue = MagicMock()
        queue.requeue_stalled_recordings.return_value = 3
        ctrl = _make_controller_with_queue(queue)

        assert ctrl.requeue_stalled_recordings() == 3
        queue.requeue_stalled_recordings.assert_called_once_with()

    def test_requeue_returns_zero_when_post_processing_disabled(self):
        ctrl = _make_controller_with_queue(None)

        assert ctrl.requeue_stalled_recordings() == 0

    def test_requeue_returns_zero_when_queue_raises(self):
        queue = MagicMock()
        queue.requeue_stalled_recordings.side_effect = RuntimeError("boom")
        ctrl = _make_controller_with_queue(queue)

        assert ctrl.requeue_stalled_recordings() == 0


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
    outcome=None,
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
        post_process_outcome=outcome,
    )


def _completed_outcome():
    from meetandread.transcription import transcript_footer
    from meetandread.transcription.transcript_footer import PostProcessOutcome

    return PostProcessOutcome(
        status=transcript_footer.STATUS_COMPLETED,
        attempted_at="2026-08-14T10:00:00",
    )


def _failed_outcome(stage=None, error="boom"):
    from meetandread.transcription import transcript_footer
    from meetandread.transcription.transcript_footer import PostProcessOutcome

    return PostProcessOutcome(
        status=transcript_footer.STATUS_FAILED,
        stage=stage or transcript_footer.STAGE_TRANSCRIBE,
        error=error,
        attempted_at="2026-08-14T10:00:00",
    )


class TestHistoryLifecycleDisplay:
    """Verify the per-row display text (label + words + speaker count)."""

    def test_has_speakers_shows_count(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=3)
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "3 speakers" in text
        assert italic is False

    def test_one_speaker_singular(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=1, speakers=["SPK_0"])
        text, _ = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "1 speaker" in text

    def test_zero_speakers_shows_words_only(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        text, _ = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "words" in text
        assert "speaker" not in text

    def test_empty_recording_shows_empty_label(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(word_count=0, speaker_count=0, speakers=[])
        text, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=None,
        )
        assert "empty recording" in text.lower()
        assert italic is False

    def test_live_job_keeps_row_italic(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        meta = _make_meta(speaker_count=0, speakers=[])
        _, italic = FloatingSettingsPanel._build_history_display_text(
            meta, return_italic=True, post_process_status=PostProcessStatus.RUNNING,
        )
        assert italic is True

    def test_no_manual_action_required_anywhere(self):
        """'Manual Action Required' is removed from the UI entirely (issue #62)."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        for status in (None, PostProcessStatus.PENDING, PostProcessStatus.RUNNING,
                       PostProcessStatus.FAILED, PostProcessStatus.CANCELLED):
            meta = _make_meta(speaker_count=0, speakers=[])
            text, _ = FloatingSettingsPanel._build_history_display_text(
                meta, return_italic=True, post_process_status=status,
            )
            assert "manual action" not in text.lower(), text
            pill_text, _, _ = FloatingSettingsPanel._build_history_status_pill(
                meta, post_process_status=status,
            )
            assert "manual action" not in pill_text.lower(), pill_text


# ---------------------------------------------------------------------------
# Slice 4: FloatingSettingsPanel._build_history_status_pill (issue #62)
# ---------------------------------------------------------------------------


class TestHistoryStatusPill:
    """Verify the per-row status pill mapping (text, kind, tooltip)."""

    def _pill(self, meta, **kwargs):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        return FloatingSettingsPanel._build_history_status_pill(meta, **kwargs)

    def test_running_shows_processing_percent(self):
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, tooltip = self._pill(
            meta, post_process_status=PostProcessStatus.RUNNING,
            post_process_progress=45,
        )
        assert "Processing 45%" == text
        assert kind == "processing"
        assert tooltip

    def test_pending_shows_queued(self):
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, _ = self._pill(meta, post_process_status=PostProcessStatus.PENDING)
        assert text == "Queued"
        assert kind == "queued"

    def test_completed_with_speakers_is_green(self):
        meta = _make_meta(speaker_count=2, outcome=_completed_outcome())
        text, kind, tooltip = self._pill(meta)
        assert text == "Completed"
        assert kind == "completed"
        assert "successful" in tooltip.lower()

    def test_completed_zero_speakers_is_yellow(self):
        meta = _make_meta(speaker_count=0, speakers=[], outcome=_completed_outcome())
        text, kind, tooltip = self._pill(meta)
        assert text == "Completed"
        assert kind == "completed-warning"
        assert "speakers not identified" in tooltip.lower()

    def test_failed_outcome_is_red_with_stage_tooltip(self):
        meta = _make_meta(
            speaker_count=0, speakers=[],
            outcome=_failed_outcome(
                stage="engine-load", error="torch not available",
            ),
        )
        text, kind, tooltip = self._pill(meta)
        assert text == "Failed"
        assert kind == "failed"
        assert "model loading" in tooltip.lower()
        assert "torch not available" in tooltip

    def test_failed_outcome_audio_missing(self):
        meta = _make_meta(
            speaker_count=0, speakers=[],
            outcome=_failed_outcome(stage="audio-missing", error="Audio file missing"),
        )
        text, kind, tooltip = self._pill(meta)
        assert text == "Failed"
        assert kind == "failed"
        assert "audio file missing" in tooltip.lower()

    def test_stalled_with_post_processing_disabled_is_red_not_post_processed(self):
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, tooltip = self._pill(
            meta, post_process_status=None, post_processing_enabled=False,
        )
        assert text == "Completed"
        assert kind == "not-post-processed"
        assert "not post-processed" in tooltip.lower()

    def test_stalled_with_post_processing_enabled_is_queued(self):
        """Stalled recordings are re-queued automatically when enabled."""
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, _ = self._pill(
            meta, post_process_status=None, post_processing_enabled=True,
        )
        assert text == "Queued"
        assert kind == "queued"

    def test_stalled_unknown_enabled_is_not_post_processed(self):
        """Without a controller, assume not post-processed (conservative)."""
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, _ = self._pill(
            meta, post_process_status=None, post_processing_enabled=None,
        )
        assert kind == "not-post-processed"

    def test_live_failed_job_without_outcome_is_failed(self):
        """In-session failure shows Failed even before the footer re-scan."""
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, tooltip = self._pill(
            meta, post_process_status=PostProcessStatus.FAILED,
            post_processing_enabled=True,
        )
        assert text == "Failed"
        assert kind == "failed"

    def test_cancelled_job_shows_as_stalled_flow_not_cancelled(self):
        """CANCELLED no longer appears in the UI; the Recording is re-queued."""
        meta = _make_meta(speaker_count=0, speakers=[])
        text, kind, _ = self._pill(
            meta, post_process_status=PostProcessStatus.CANCELLED,
            post_processing_enabled=True,
        )
        assert "cancel" not in text.lower()
        assert kind == "queued"

    def test_live_job_wins_over_outcome(self):
        """An in-flight re-attempt takes precedence over a stored Outcome."""
        meta = _make_meta(speaker_count=2, outcome=_completed_outcome())
        text, kind, _ = self._pill(
            meta, post_process_status=PostProcessStatus.RUNNING,
            post_process_progress=30,
        )
        assert text == "Processing 30%"
        assert kind == "processing"
