"""Controller wiring for preemptible post-processing and Retry (issue #63).

RecordingController façades over the Post-processing queue:

* ``preempt_post_processing`` / record-start — starting a live Recording
  preempts a running Post-processing job immediately, with no dialog.  The
  live Recording always outranks background work (ADR-0002 supersedes the
  old defer-and-let-it-finish behavior).  The idle-wait gate for queued
  jobs remains.
* ``retry_post_processing`` — the user-initiated Retry for a Failed
  recording: clears the Failed Outcome from the Transcript Footer,
  preempts a running job, and schedules the Retry at the FRONT of the
  queue with default Post-processing settings (no model picker, no
  sidecar; the transcript is overwritten in place).
* ``is_post_processing_running`` — queue-busy probe for the Retry confirm
  dialog.
* ``get_post_process_failure`` — the failure payload (stage, error,
  user_initiated) surfacing actively in the UI.
"""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
)
from meetandread.transcription import transcript_footer
from meetandread.transcription.post_processor import (
    PostProcessJob,
    PostProcessStatus,
)
from meetandread.transcription.transcript_footer import PostProcessOutcome

from tests.footer_test_helpers import write_transcript


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePreemptibleQueue:
    """Fake PostProcessingQueue recording preempt/retry interactions."""

    def __init__(self, running: bool = False):
        self.preempt_calls: list = []
        self.scheduled: list = []
        self.running = running
        self.jobs: dict = {}

    def preempt_current_job(self, reason: str = "") -> bool:
        self.preempt_calls.append(reason)
        return True

    def is_job_running(self) -> bool:
        return self.running

    def schedule_post_process(
        self,
        audio_file: Path,
        realtime_transcript,
        output_dir: Path,
        model_size: Optional[str] = None,
        *,
        front: bool = False,
        user_initiated: bool = False,
    ) -> PostProcessJob:
        job = PostProcessJob(
            job_id=f"job-{len(self.scheduled) + 1}",
            audio_file=audio_file,
            realtime_transcript=realtime_transcript,
            output_dir=output_dir,
            model_size=model_size or "base",
            user_initiated=user_initiated,
        )
        self.scheduled.append(
            {
                "job": job,
                "audio_file": audio_file,
                "realtime_transcript": realtime_transcript,
                "output_dir": output_dir,
                "model_size": model_size,
                "front": front,
                "user_initiated": user_initiated,
            }
        )
        self.jobs[job.job_id] = job
        return job

    def cancel_job(self, job_id: str, reason: str = "") -> bool:
        raise AssertionError("preemption must not use terminal cancellation")

    def cancel_current_job(self, reason: str = "") -> bool:
        raise AssertionError("preemption must not use terminal cancellation")

    def get_job_status(self, job_id: str) -> Optional[PostProcessJob]:
        return self.jobs.get(job_id)


def _controller_with_queue(queue) -> RecordingController:
    ctrl = RecordingController(enable_transcription=True)
    ctrl._post_processor = queue
    return ctrl


def _failed_transcript(tmp_path: Path) -> Path:
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir(exist_ok=True)
    return write_transcript(
        transcripts / "recording-x.md",
        "# Transcript\n\nhello",
        {
            "word_count": 1,
            "post_process": PostProcessOutcome(
                status=transcript_footer.STATUS_FAILED,
                stage=transcript_footer.STAGE_TRANSCRIBE,
                error="boom",
                attempted_at="2026-08-14T10:00:00",
            ).to_block(),
        },
    )


def _recordings_dir(tmp_path: Path, monkeypatch, wav_name: str = "recording-x.wav") -> Path:
    from meetandread.audio.storage import paths as paths_mod

    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    (recordings / wav_name).write_bytes(b"RIFF")
    monkeypatch.setattr(paths_mod, "get_recordings_dir", lambda: recordings)
    return recordings


# ---------------------------------------------------------------------------
# Record-start preemption (ADR-0002)
# ---------------------------------------------------------------------------


class TestPreemptPostProcessingFaçade:
    """preempt_post_processing delegates to the queue, safely."""

    def test_delegates_with_reason(self):
        queue = FakePreemptibleQueue()
        ctrl = _controller_with_queue(queue)

        ctrl.preempt_post_processing(reason="new recording starting")

        assert queue.preempt_calls == ["new recording starting"]

    def test_no_queue_is_noop(self):
        ctrl = _controller_with_queue(None)
        ctrl.preempt_post_processing()  # must not raise

    def test_queue_exception_swallowed(self):
        queue = MagicMock()
        queue.preempt_current_job.side_effect = RuntimeError("boom")
        ctrl = _controller_with_queue(queue)

        ctrl.preempt_post_processing()  # must not raise

    def test_queue_without_preempt_api_swallowed(self):
        """Older/fake queues without the preempt API are tolerated."""
        ctrl = _controller_with_queue(MagicMock(spec=[]))
        ctrl.preempt_post_processing()  # must not raise


class TestStartPreemptsRunningJob:
    """start() preempts (not cancels) a running Post-processing job."""

    def _start_with_valid_sources(self, ctrl):
        from meetandread.audio import SourceConfig

        with patch.object(ctrl, "_init_transcription", return_value=None), \
             patch.object(ctrl, "_build_source_configs") as mock_sources, \
             patch("meetandread.recording.controller.AudioSession") as MockSession:
            mock_sources.return_value = [SourceConfig(type="mic")]
            MockSession.return_value = MagicMock()
            return ctrl.start({"mic"})

    def test_start_preempts_with_no_dialog(self, tmp_path):
        queue = FakePreemptibleQueue()
        ctrl = _controller_with_queue(queue)
        ctrl._state = ControllerState.IDLE

        self._start_with_valid_sources(ctrl)

        assert queue.preempt_calls, "start() must preempt in-flight post-processing"
        assert all("recording" in r for r in queue.preempt_calls)

    def test_aborted_start_does_not_preempt(self, tmp_path):
        """A start that fails early validation leaves the running job alone.

        (A start that fails later — e.g. device open failure — still
        preempts: the recording genuinely attempted to begin.)
        """
        queue = FakePreemptibleQueue()
        ctrl = _controller_with_queue(queue)
        ctrl._state = ControllerState.IDLE

        ctrl.start(set())  # no sources → fails validation

        assert queue.preempt_calls == []

    def test_start_while_recording_does_not_preempt(self, tmp_path):
        queue = FakePreemptibleQueue()
        ctrl = _controller_with_queue(queue)
        ctrl._state = ControllerState.RECORDING

        ctrl.start({"mic"})

        assert queue.preempt_calls == []


# ---------------------------------------------------------------------------
# Retry façade
# ---------------------------------------------------------------------------


class TestRetryPostProcessing:
    """retry_post_processing re-runs a Failed recording's post-processing."""

    def test_retry_clears_outcome_preempts_and_schedules_front(
        self, tmp_path, monkeypatch
    ):
        md_path = _failed_transcript(tmp_path)
        _recordings_dir(tmp_path, monkeypatch)
        queue = FakePreemptibleQueue(running=True)
        ctrl = _controller_with_queue(queue)

        job_id = ctrl.retry_post_processing(md_path)

        assert job_id == "job-1"
        # The Failed Outcome is cleared from the Transcript Footer.
        assert transcript_footer.read_post_process_outcome(
            md_path.read_text(encoding="utf-8")
        ) is None
        # A running job was preempted so the Retry runs first.
        assert len(queue.preempt_calls) == 1
        # The Retry is scheduled at the front, user-initiated, with default
        # settings — no model picker, no sidecar.
        (scheduled,) = queue.scheduled
        assert scheduled["front"] is True
        assert scheduled["user_initiated"] is True
        assert scheduled["model_size"] is None
        assert scheduled["realtime_transcript"] is None
        assert scheduled["audio_file"].name == "recording-x.wav"
        assert scheduled["output_dir"] == md_path.parent

    def test_retry_when_idle_still_schedules_front(self, tmp_path, monkeypatch):
        """Queue idle → no dialog, Retry runs immediately (front)."""
        md_path = _failed_transcript(tmp_path)
        _recordings_dir(tmp_path, monkeypatch)
        queue = FakePreemptibleQueue(running=False)
        ctrl = _controller_with_queue(queue)

        assert ctrl.retry_post_processing(md_path) is not None
        # preempt_current_job is harmless when idle.
        assert len(queue.preempt_calls) == 1
        assert queue.scheduled[0]["front"] is True

    def test_retry_without_queue_returns_none(self, tmp_path, monkeypatch):
        md_path = _failed_transcript(tmp_path)
        _recordings_dir(tmp_path, monkeypatch)
        ctrl = _controller_with_queue(None)

        assert ctrl.retry_post_processing(md_path) is None

    def test_retry_without_audio_returns_none_and_keeps_outcome(
        self, tmp_path, monkeypatch
    ):
        md_path = _failed_transcript(tmp_path)
        from meetandread.audio.storage import paths as paths_mod

        empty = tmp_path / "recordings"
        empty.mkdir(exist_ok=True)
        monkeypatch.setattr(paths_mod, "get_recordings_dir", lambda: empty)

        queue = FakePreemptibleQueue()
        ctrl = _controller_with_queue(queue)

        assert ctrl.retry_post_processing(md_path) is None
        # The Failed Outcome survives: nothing was re-scheduled.
        assert queue.scheduled == []
        assert transcript_footer.read_post_process_outcome(
            md_path.read_text(encoding="utf-8")
        ) is not None

    def test_retry_swallows_schedule_errors(self, tmp_path, monkeypatch):
        md_path = _failed_transcript(tmp_path)
        _recordings_dir(tmp_path, monkeypatch)
        queue = FakePreemptibleQueue()
        queue.schedule_post_process = MagicMock(side_effect=RuntimeError("boom"))
        ctrl = _controller_with_queue(queue)

        assert ctrl.retry_post_processing(md_path) is None


# ---------------------------------------------------------------------------
# Queue-busy probe + failure payload
# ---------------------------------------------------------------------------


class TestIsPostProcessingRunning:
    """is_post_processing_running drives the Retry confirm dialog."""

    def test_true_when_queue_running(self):
        ctrl = _controller_with_queue(FakePreemptibleQueue(running=True))
        assert ctrl.is_post_processing_running() is True

    def test_false_when_queue_idle(self):
        ctrl = _controller_with_queue(FakePreemptibleQueue(running=False))
        assert ctrl.is_post_processing_running() is False

    def test_false_without_queue(self):
        ctrl = _controller_with_queue(None)
        assert ctrl.is_post_processing_running() is False

    def test_false_when_queue_raises(self):
        queue = MagicMock()
        queue.is_job_running.side_effect = RuntimeError("boom")
        ctrl = _controller_with_queue(queue)
        assert ctrl.is_post_processing_running() is False


class TestGetPostProcessFailure:
    """get_post_process_failure yields the active-dialog payload."""

    def _failed_job(self, tmp_path: Path, **kwargs) -> PostProcessJob:
        defaults = dict(
            job_id="job-9",
            audio_file=tmp_path / "recording-x.wav",
            realtime_transcript=None,
            output_dir=tmp_path,
            model_size="base",
            status=PostProcessStatus.FAILED,
            error="boom",
            error_stage=transcript_footer.STAGE_ENGINE_LOAD,
            user_initiated=True,
        )
        defaults.update(kwargs)
        return PostProcessJob(**defaults)

    def test_failed_user_initiated_job(self, tmp_path):
        queue = FakePreemptibleQueue()
        queue.jobs["job-9"] = self._failed_job(tmp_path)
        ctrl = _controller_with_queue(queue)

        failure = ctrl.get_post_process_failure("job-9")

        assert failure == {
            "stage": transcript_footer.STAGE_ENGINE_LOAD,
            "error": "boom",
            "user_initiated": True,
            "transcript_path": str(tmp_path / "recording-x.md"),
        }

    def test_background_failure_is_not_user_initiated(self, tmp_path):
        queue = FakePreemptibleQueue()
        queue.jobs["job-9"] = self._failed_job(
            tmp_path, user_initiated=False, error_stage=None
        )
        ctrl = _controller_with_queue(queue)

        failure = ctrl.get_post_process_failure("job-9")

        assert failure["user_initiated"] is False
        assert failure["stage"] == transcript_footer.STAGE_TRANSCRIBE

    def test_none_when_job_missing(self):
        ctrl = _controller_with_queue(FakePreemptibleQueue())
        assert ctrl.get_post_process_failure("nope") is None

    def test_none_when_job_completed(self, tmp_path):
        queue = FakePreemptibleQueue()
        queue.jobs["job-9"] = self._failed_job(
            tmp_path, status=PostProcessStatus.COMPLETED
        )
        ctrl = _controller_with_queue(queue)
        assert ctrl.get_post_process_failure("job-9") is None

    def test_none_without_queue(self):
        ctrl = _controller_with_queue(None)
        assert ctrl.get_post_process_failure("job-9") is None

    def test_none_when_queue_raises(self):
        queue = MagicMock()
        queue.get_job_status.side_effect = RuntimeError("boom")
        ctrl = _controller_with_queue(queue)
        assert ctrl.get_post_process_failure("job-9") is None
