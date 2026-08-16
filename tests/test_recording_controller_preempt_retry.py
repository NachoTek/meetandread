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

import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

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
# The idle gate must be closed BEFORE the preempt request (issue #63)
# ---------------------------------------------------------------------------


class TestIsPostProcessingGate:
    """is_post_processing_gated is the queue's idle-wait gate predicate.

    STARTING (and RETRYING) must count as busy: the preempt request fires
    after the state leaves IDLE, so the gate has to be closed by then or
    the just-preempted job is re-dequeued and — being shielded — runs
    un-preemptibly through the whole recording.
    """

    @pytest.mark.parametrize(
        "state,busy",
        [
            (ControllerState.STARTING, True),
            (ControllerState.RETRYING, True),
            (ControllerState.RECORDING, True),
            (ControllerState.IDLE, False),
            (ControllerState.STOPPING, False),
            (ControllerState.ERROR, False),
        ],
    )
    def test_gate_covers_starting_and_recording(self, state, busy):
        ctrl = RecordingController(enable_transcription=False)
        ctrl._state = state

        assert ctrl.is_post_processing_gated() is busy

    def test_gate_is_thread_safe_snapshot(self):
        ctrl = RecordingController(enable_transcription=False)
        ctrl._state = ControllerState.RECORDING
        ctrl.is_post_processing_gated()  # must not raise under lock churn


class TestPostProcessorQueuePersistsAcrossRecordings:
    """_ensure_post_processor reuses one queue for the controller's life.

    Replacing the queue per recording would (a) discard the loaded engine
    cache, and (b) start() a second worker whose pending-job recovery
    re-enqueues the previous queue's still-persisted (preempted) job as a
    fresh unshielded copy — defeating record-start preemption (issue #63).
    """

    def _settings(self, enabled=True):
        settings = MagicMock()
        settings.transcription.enable_postprocessing = enabled
        settings.transcription.postprocess_model_size = "base"
        return settings

    def test_created_once_and_reused(self, tmp_path, monkeypatch):
        from meetandread.audio.storage import paths as paths_mod

        data = tmp_path / "queue-data"
        data.mkdir()
        monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)

        ctrl = RecordingController(enable_transcription=False)
        ctrl._ensure_post_processor(self._settings())
        first = ctrl._post_processor
        try:
            assert first is not None

            ctrl._ensure_post_processor(self._settings())

            assert ctrl._post_processor is first
        finally:
            if ctrl._post_processor:
                ctrl._post_processor.stop()

    def test_not_created_when_post_processing_disabled(self):
        ctrl = RecordingController(enable_transcription=False)

        ctrl._ensure_post_processor(self._settings(enabled=False))

        assert ctrl._post_processor is None

    def test_existing_fake_queue_is_not_replaced(self):
        """A queue injected by callers/tests survives re-initialization."""
        ctrl = RecordingController(enable_transcription=False)
        fake = MagicMock()
        ctrl._post_processor = fake

        ctrl._ensure_post_processor(self._settings())

        assert ctrl._post_processor is fake
        fake.start.assert_not_called()


class TestRecordStartPreemptsEndToEnd:
    """Record-start preemption against a REAL queue + worker (issue #63).

    Reproduces the production wiring: the queue's idle gate is the
    controller's gate predicate, the job runs on the queue's worker, and
    the preempt request arrives from the controller.  The preempted job
    must park while the recording is live and complete afterwards without
    user intervention — no queue replacement, no unshielded recovery copy.
    """

    def test_preempted_job_parks_while_recording_and_completes_after(
        self, tmp_path, monkeypatch
    ):
        import numpy as np

        from meetandread.transcription.post_processor import (
            PostProcessingQueue,
            PostProcessStatus,
        )
        from tests.test_post_processing_preempt import (
            FakeEngine,
            _wait_for,
            _window_samples,
        )
        from meetandread.audio.storage import paths as paths_mod

        data = tmp_path / "queue-data"
        data.mkdir()
        monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)

        settings = MagicMock()
        settings.transcription.postprocess_model_size = "base"

        ctrl = RecordingController(enable_transcription=False)
        windows = 2
        first_window_started = threading.Event()
        release_first_window = threading.Event()

        def on_chunk(index):
            if index == 1:
                # Hold the first window open until the test has fired the
                # preempt request — otherwise the faked job finishes before
                # preemption can arrive.
                first_window_started.set()
                assert release_first_window.wait(timeout=10)

        queue = PostProcessingQueue(
            settings=settings,
            is_recording_callback=ctrl.is_post_processing_gated,
            auto_requeue_stalled=False,
        )
        engine = FakeEngine(on_chunk=on_chunk)
        queue._engines["base"] = engine
        queue._load_audio_file = lambda path: np.zeros(
            windows * _window_samples(queue), dtype=np.float32
        )
        ctrl._post_processor = queue

        audio = tmp_path / "recording-x.wav"
        audio.write_bytes(b"RIFF")
        job = queue.schedule_post_process(audio, None, tmp_path)

        queue.start()
        try:
            assert first_window_started.wait(timeout=10), "job never started"

            # Simulate the post-validation part of start(): gate closes
            # (STARTING), preempt fires, recording goes live.
            ctrl._state = ControllerState.STARTING
            ctrl.preempt_post_processing(reason="new recording starting")
            ctrl._state = ControllerState.RECORDING
            release_first_window.set()

            # The job steps aside at the next cooperative checkpoint.
            assert _wait_for(
                lambda: job.status == PostProcessStatus.PENDING
            ), f"job was not preempted: {job.status}"
            assert job.shielded is True
            partial_windows = len(engine.calls)
            assert partial_windows >= 1

            # While the recording is live the parked job must NOT re-run.
            time.sleep(0.6)
            assert job.status == PostProcessStatus.PENDING
            assert len(engine.calls) == partial_windows

            # Recording ends — the parked job finishes on its own.
            ctrl._state = ControllerState.IDLE
            assert _wait_for(
                lambda: job.status == PostProcessStatus.COMPLETED
            ), f"job did not complete after recording: {job.status}"
            # Partial progress was redone: full re-run after the partial.
            assert len(engine.calls) == partial_windows + windows
        finally:
            ctrl._state = ControllerState.IDLE
            queue.stop()


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
