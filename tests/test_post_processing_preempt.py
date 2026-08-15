"""Post-processing jobs are preemptible, not just cancellable (issue #63).

ADR-0002: a preempt request is honored cooperatively per-segment inside the
transcription loop (the engine wrapper checks between segments).  A preempted
job returns to the FRONT of the queue — it is not terminal — and is shielded
from re-preemption until it completes, so a fast-failing Retry cannot starve
a long job.  Partial transcription progress is lost and redone.

Seams covered here:

1. ``_PendingJobQueue`` — the FIFO pending lane with a priority front lane
   (preempted jobs and front-enqueued Retries) that always drains first.
2. ``PostProcessingQueue.preempt_job`` / ``preempt_current_job`` — the
   preempt request API and its refusal rules (shield, non-RUNNING, unknown).
3. ``PostProcessingQueue._transcribe_audio_segmented`` — the engine wrapper
   that checks the preempt/cancel flags between segments and offsets
   per-window timestamps back into whole-audio time.
4. Worker-loop end-to-end — a job preempted mid-transcription returns to the
   front, completes without user intervention, and a front-enqueued Retry
   runs before the preempted job.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from meetandread.transcription import transcript_footer
from meetandread.transcription.engine import (
    TranscriptionSegment,
    TranscriptionSuccess,
    WordInfo,
)
from meetandread.transcription.post_processor import (
    PostProcessingQueue,
    PostProcessJob,
    PostProcessStatus,
    _JobCancelled,
    _JobPreempted,
    _PendingJobQueue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(model: str = "base") -> MagicMock:
    settings = MagicMock()
    settings.transcription.postprocess_model_size = model
    return settings


def _make_queue(
    tmp_path: Path,
    monkeypatch,
    *,
    is_recording=None,
    start: bool = True,
) -> PostProcessingQueue:
    """Construct a PostProcessingQueue whose queue file lives in tmp_path.

    The worker is started against the empty tmp queue file (so pending-job
    recovery is a no-op) BEFORE any job is scheduled — mirroring the
    production order and keeping schedule()'s auto-start from running
    recovery over entries it just persisted.
    """
    from meetandread.audio.storage import paths as paths_mod

    data = tmp_path / "queue-data"
    data.mkdir(exist_ok=True)
    monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)
    kwargs = {"settings": _settings(), "auto_requeue_stalled": False}
    if is_recording is not None:
        kwargs["is_recording_callback"] = is_recording
    queue = PostProcessingQueue(**kwargs)
    if start:
        queue.start()
    return queue


class FakeEngine:
    """Engine fake that records per-window calls and runs an optional hook.

    ``on_chunk`` (if set) fires at the START of each ``transcribe_chunk``
    call, simulating a concurrent preempt/cancel request arriving from
    another thread while the window is being transcribed.
    """

    def __init__(self, on_chunk=None):
        self.calls: List[int] = []
        self.first_call_at: Optional[float] = None
        self.on_chunk = on_chunk

    def transcribe_chunk(self, audio, word_level=False):
        if self.first_call_at is None:
            self.first_call_at = time.monotonic()
        index = len(self.calls) + 1
        self.calls.append(len(audio))
        if self.on_chunk is not None:
            self.on_chunk(index)
        return TranscriptionSuccess(
            segments=[
                TranscriptionSegment(
                    text="hello",
                    confidence=90,
                    start=0.0,
                    end=1.0,
                    words=[WordInfo(text="hello", start=0.0, end=1.0, confidence=90)],
                )
            ]
        )


def _window_samples(queue: PostProcessingQueue) -> int:
    return int(queue.TRANSCRIBE_WINDOW_SECONDS * 16000)


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# Slice 1: _PendingJobQueue — front lane drains before the back lane
# ---------------------------------------------------------------------------


class TestPendingJobQueueOrdering:
    """Front-lane jobs (preempted / Retry) run ahead of normal queued jobs."""

    def test_front_lane_drains_before_back_lane(self):
        q = _PendingJobQueue()
        back_a, front_a, front_b, back_b = "back-a", "front-a", "front-b", "back-b"

        q.put(back_a)
        q.put_front(front_a)
        q.put_front(front_b)
        q.put(back_b)

        assert q.get(timeout=0.1) is front_a
        assert q.get(timeout=0.1) is front_b
        assert q.get(timeout=0.1) is back_a
        assert q.get(timeout=0.1) is back_b

    def test_get_returns_none_on_timeout(self):
        q = _PendingJobQueue()
        assert q.get(timeout=0.05) is None

    def test_front_lane_is_fifo(self):
        """Two Retries queued at the front keep their schedule order."""
        q = _PendingJobQueue()
        q.put_front("r1")
        q.put_front("r2")
        assert q.get(timeout=0.1) == "r1"
        assert q.get(timeout=0.1) == "r2"

    def test_retry_enqueued_during_preempt_runs_before_preempted_job(self):
        """Retry enqueued at the front while a job is being preempted lands
        AHEAD of the preempted job (which re-enters the front lane later)."""
        q = _PendingJobQueue()
        q.put("backlog")
        q.put_front("retry")  # Retry scheduled while A is still RUNNING
        q.put_front("preempted-a")  # A lands in the front lane afterwards

        assert q.get(timeout=0.1) == "retry"
        assert q.get(timeout=0.1) == "preempted-a"
        assert q.get(timeout=0.1) == "backlog"


# ---------------------------------------------------------------------------
# Slice 2: preempt_job / preempt_current_job request semantics
# ---------------------------------------------------------------------------


def _job(tmp_path: Path, status=PostProcessStatus.PENDING, **kwargs) -> PostProcessJob:
    defaults = dict(
        job_id="job-1",
        audio_file=tmp_path / "recording-x.wav",
        realtime_transcript=None,
        output_dir=tmp_path,
        model_size="base",
        status=status,
    )
    defaults.update(kwargs)
    return PostProcessJob(**defaults)


class TestPreemptJob:
    """preempt_job honors RUNNING unshielded jobs only."""

    def test_preempt_running_unshielded_job(self, tmp_path):
        queue = PostProcessingQueue(settings=_settings())
        job = _job(tmp_path, status=PostProcessStatus.RUNNING)
        queue._jobs[job.job_id] = job

        assert queue.preempt_job(job.job_id, reason="record-start") is True
        assert job.preempt_requested is True
        assert job.preempt_reason == "record-start"

    def test_preempt_shielded_job_refused(self, tmp_path):
        queue = PostProcessingQueue(settings=_settings())
        job = _job(tmp_path, status=PostProcessStatus.RUNNING, shielded=True)
        queue._jobs[job.job_id] = job

        assert queue.preempt_job(job.job_id) is False
        assert job.preempt_requested is False

    def test_preempt_pending_job_refused(self, tmp_path):
        """Preemption targets the running job; a queued job just waits."""
        queue = PostProcessingQueue(settings=_settings())
        job = _job(tmp_path, status=PostProcessStatus.PENDING)
        queue._jobs[job.job_id] = job

        assert queue.preempt_job(job.job_id) is False
        assert job.preempt_requested is False

    @pytest.mark.parametrize(
        "status",
        [
            PostProcessStatus.COMPLETED,
            PostProcessStatus.FAILED,
            PostProcessStatus.CANCELLED,
        ],
    )
    def test_preempt_terminal_job_refused(self, tmp_path, status):
        queue = PostProcessingQueue(settings=_settings())
        job = _job(tmp_path, status=status)
        queue._jobs[job.job_id] = job

        assert queue.preempt_job(job.job_id) is False

    def test_preempt_unknown_job_returns_false(self):
        queue = PostProcessingQueue(settings=_settings())
        assert queue.preempt_job("nope") is False

    def test_preempt_current_job_delegates(self, tmp_path):
        queue = PostProcessingQueue(settings=_settings())
        job = _job(tmp_path, status=PostProcessStatus.RUNNING)
        queue._jobs[job.job_id] = job
        queue._current_job = job

        assert queue.preempt_current_job(reason="user retry") is True
        assert job.preempt_requested is True
        assert job.preempt_reason == "user retry"

    def test_preempt_current_job_without_running_job(self):
        queue = PostProcessingQueue(settings=_settings())
        assert queue.preempt_current_job() is False


class TestIsJobRunning:
    """is_job_running drives the Retry confirm dialog (queue busy?)."""

    def test_true_while_job_running(self, tmp_path):
        queue = PostProcessingQueue(settings=_settings())
        queue._jobs["a"] = _job(tmp_path, status=PostProcessStatus.RUNNING)
        assert queue.is_job_running() is True

    def test_false_when_only_pending(self, tmp_path):
        queue = PostProcessingQueue(settings=_settings())
        queue._jobs["a"] = _job(tmp_path, status=PostProcessStatus.PENDING)
        assert queue.is_job_running() is False

    def test_false_when_empty(self):
        queue = PostProcessingQueue(settings=_settings())
        assert queue.is_job_running() is False


class TestScheduleFrontAndUserInitiated:
    """schedule_post_process gains front-of-queue + user_initiated flags."""

    def test_schedule_front_lands_ahead_of_back_jobs(self, tmp_path, monkeypatch):
        # No worker thread: pretend it is running so schedule()'s auto-start
        # is a no-op and the lanes can be inspected directly.
        queue = _make_queue(tmp_path, monkeypatch, start=False)
        queue._is_running = True
        try:
            audio = tmp_path / "recording-a.wav"
            audio.write_bytes(b"RIFF")

            first = queue.schedule_post_process(audio, None, tmp_path)
            retry = queue.schedule_post_process(
                audio, None, tmp_path, front=True, user_initiated=True
            )

            assert retry.user_initiated is True
            assert first.user_initiated is False
            # Front lane drains first: the retry job is dequeued before `first`.
            assert queue._job_queue.get(timeout=0.1) is retry
            assert queue._job_queue.get(timeout=0.1) is first
        finally:
            queue._is_running = False
            queue.stop()


# ---------------------------------------------------------------------------
# Slice 3: _transcribe_audio_segmented — the preemptible engine wrapper
# ---------------------------------------------------------------------------


class TestTranscribeSegmented:
    """The wrapper transcribes window-by-window, checking flags between."""

    def _audio(self, queue, windows: int) -> np.ndarray:
        return np.zeros(windows * _window_samples(queue), dtype=np.float32)

    def test_short_audio_single_window_no_offset(self):
        queue = PostProcessingQueue(settings=_settings())
        engine = FakeEngine()
        audio = np.zeros(16000, dtype=np.float32)  # 1s — well under one window
        job = _job(Path("."))

        segments = queue._transcribe_audio_segmented(engine, audio, job)

        assert len(engine.calls) == 1
        assert segments[0].start == 0.0

    def test_multi_window_offsets_timestamps(self):
        queue = PostProcessingQueue(settings=_settings())
        engine = FakeEngine()
        audio = self._audio(queue, windows=3)
        job = _job(Path("."))

        segments = queue._transcribe_audio_segmented(engine, audio, job)

        assert len(engine.calls) == 3
        # Window 2 starts at TRANSCRIBE_WINDOW_SECONDS; the engine's
        # window-relative 0.0..1.0 must be offset into whole-audio time.
        assert segments[0].start == pytest.approx(0.0)
        assert segments[1].start == pytest.approx(queue.TRANSCRIBE_WINDOW_SECONDS)
        assert segments[2].start == pytest.approx(2 * queue.TRANSCRIBE_WINDOW_SECONDS)
        assert segments[2].words[0].start == pytest.approx(
            2 * queue.TRANSCRIBE_WINDOW_SECONDS
        )

    def test_preempt_flag_stops_before_next_window(self):
        queue = PostProcessingQueue(settings=_settings())
        job_ref = {}

        def on_chunk(index):
            if index == 2:
                job_ref["job"].preempt_requested = True

        job_ref["job"] = job = _job(Path("."))
        engine = FakeEngine(on_chunk=on_chunk)
        audio = self._audio(queue, windows=4)

        with pytest.raises(_JobPreempted):
            queue._transcribe_audio_segmented(engine, audio, job)

        # Preempt honored after the in-flight window completed: within one
        # segment of the request, not after the whole audio.
        assert len(engine.calls) == 2

    def test_cancel_flag_stops_before_next_window(self):
        queue = PostProcessingQueue(settings=_settings())
        job_ref = {}

        def on_chunk(index):
            if index == 1:
                job_ref["job"].cancel_requested = True

        job_ref["job"] = job = _job(Path("."))
        engine = FakeEngine(on_chunk=on_chunk)
        audio = self._audio(queue, windows=3)

        with pytest.raises(_JobCancelled):
            queue._transcribe_audio_segmented(engine, audio, job)

        assert len(engine.calls) == 1

    def test_engine_error_raises_post_process_failure(self):
        from meetandread.transcription.engine import TranscriptionError

        queue = PostProcessingQueue(settings=_settings())

        class BrokenEngine:
            def transcribe_chunk(self, audio, word_level=False):
                return TranscriptionError(error_type="model_error", message="boom")

        with pytest.raises(Exception) as excinfo:
            queue._transcribe_audio_segmented(
                BrokenEngine(), np.zeros(16000, dtype=np.float32), _job(Path("."))
            )
        assert excinfo.value.stage == transcript_footer.STAGE_TRANSCRIBE

    def test_progress_advances_within_transcribe_band(self):
        progress: List[int] = []
        queue = PostProcessingQueue(
            settings=_settings(), on_progress=lambda jid, pct: progress.append(pct)
        )
        engine = FakeEngine()
        audio = self._audio(queue, windows=2)
        job = _job(Path("."))

        queue._transcribe_audio_segmented(engine, audio, job)

        assert progress  # progress reported between windows
        assert all(35 <= p <= 80 for p in progress)
        assert progress[-1] == 80


# ---------------------------------------------------------------------------
# Slice 4: worker loop — preemption returns the job to the front
# ---------------------------------------------------------------------------


class TestWorkerPreemptionEndToEnd:
    """A preempted job returns to the front, is shielded, and completes."""

    def test_preempted_job_returns_to_front_and_completes(
        self, tmp_path, monkeypatch
    ):
        gate = {"recording": True}
        queue = _make_queue(
            tmp_path, monkeypatch, is_recording=lambda: gate["recording"]
        )
        audio_a = tmp_path / "recording-a.wav"
        audio_a.write_bytes(b"RIFF")

        windows = 3
        preempt_results: List[bool] = []
        job_a_ref: dict = {}

        def on_chunk_a(index):
            if index == 2:
                # First pass: preempt A mid-transcription.
                preempt_results.append(
                    queue.preempt_job(job_a_ref["id"], reason="record-start")
                )

        engine_a = FakeEngine(on_chunk=on_chunk_a)
        engine_b = FakeEngine()
        queue._engines["base"] = engine_a
        queue._engines["small"] = engine_b
        queue._load_audio_file = lambda path: np.zeros(
            windows * _window_samples(queue), dtype=np.float32
        )

        job_a = queue.schedule_post_process(audio_a, None, tmp_path)
        job_a_ref["id"] = job_a.job_id
        # A user-initiated Retry for another recording, enqueued at the
        # front while A waits behind the recording gate.
        audio_b = tmp_path / "recording-b.wav"
        audio_b.write_bytes(b"RIFF")
        job_b = queue.schedule_post_process(
            audio_b, None, tmp_path, model_size="small",
            front=True, user_initiated=True,
        )

        completed: dict = {}
        done = threading.Event()

        def on_complete(job_id, result):
            completed[job_id] = dict(result)
            if job_a.job_id in completed and job_b.job_id in completed:
                done.set()

        queue._on_complete = on_complete

        try:
            gate["recording"] = False
            assert _wait_for(done.is_set), f"jobs did not complete: {completed}"

            # The Retry ran first (front lane) — before A ever started.
            assert engine_b.first_call_at is not None
            assert engine_a.first_call_at is not None
            assert engine_b.first_call_at < engine_a.first_call_at

            # A was preempted after its second window and re-ran from scratch.
            assert preempt_results == [True]
            assert len(engine_a.calls) == windows + 2

            # Both jobs completed; the preempted job needed no user action.
            assert job_a.status == PostProcessStatus.COMPLETED
            assert job_b.status == PostProcessStatus.COMPLETED
            assert job_a.shielded is True
            assert completed[job_a.job_id]["model_used"] == "base"
            assert completed[job_b.job_id]["model_used"] == "small"
        finally:
            queue.stop()

    def test_shielded_job_cannot_be_preempted_again(self, tmp_path, monkeypatch):
        """After preemption the re-run is shielded; a Retry waits its turn."""
        gate = {"recording": True}
        queue = _make_queue(
            tmp_path, monkeypatch, is_recording=lambda: gate["recording"]
        )
        audio_a = tmp_path / "recording-a.wav"
        audio_a.write_bytes(b"RIFF")
        windows = 2
        second_attempt_preempt: List[bool] = []
        job_a_ref: dict = {}
        first_pass = {"done": False}

        def on_chunk_a(index):
            if not first_pass["done"]:
                if index == 1:
                    first_pass["done"] = queue.preempt_job(job_a_ref["id"])
            else:
                # Second (shielded) attempt: preemption must be refused.
                second_attempt_preempt.append(queue.preempt_job(job_a_ref["id"]))

        engine_a = FakeEngine(on_chunk=on_chunk_a)
        queue._engines["base"] = engine_a
        queue._load_audio_file = lambda path: np.zeros(
            windows * _window_samples(queue), dtype=np.float32
        )

        job_a = queue.schedule_post_process(audio_a, None, tmp_path)
        job_a_ref["id"] = job_a.job_id

        try:
            gate["recording"] = False
            assert _wait_for(
                lambda: job_a.status == PostProcessStatus.COMPLETED
            ), f"job did not complete: {job_a.status}"

            assert first_pass["done"] is True
            # The shield held during the second attempt.
            assert second_attempt_preempt
            assert all(result is False for result in second_attempt_preempt)
            # Full re-run: partial (1 window) + full (2 windows).
            assert len(engine_a.calls) == windows + 1
        finally:
            queue.stop()

    def test_cancelled_mid_transcription_is_terminal_and_preserves_transcript(
        self, tmp_path, monkeypatch
    ):
        queue = _make_queue(tmp_path, monkeypatch)
        audio_a = tmp_path / "recording-a.wav"
        audio_a.write_bytes(b"RIFF")
        original_md = tmp_path / "recording-a.md"
        original_md.write_text("# Original\n\nkeep me", encoding="utf-8")

        def on_chunk(index):
            if index == 1:
                queue.cancel_current_job(reason="test cancel")

        engine_a = FakeEngine(on_chunk=on_chunk)
        queue._engines["base"] = engine_a
        queue._load_audio_file = lambda path: np.zeros(
            3 * _window_samples(queue), dtype=np.float32
        )

        job = queue.schedule_post_process(audio_a, None, tmp_path)

        try:
            assert _wait_for(
                lambda: job.status == PostProcessStatus.CANCELLED
            ), f"job not cancelled: {job.status}"
            time.sleep(0.2)  # let any stray write land

            assert original_md.read_text(encoding="utf-8") == "# Original\n\nkeep me"
            # Cancelled mid-transcription: the engine saw one window only.
            assert len(engine_a.calls) == 1
            assert queue.get_status_for_audio(audio_a) == PostProcessStatus.CANCELLED
        finally:
            queue.stop()

    def test_preempted_job_stays_persisted_until_terminal(self, tmp_path, monkeypatch):
        """A preempted job remains in the persisted queue file (still pending)."""
        gate = {"recording": True}
        queue = _make_queue(
            tmp_path, monkeypatch, is_recording=lambda: gate["recording"]
        )
        audio_a = tmp_path / "recording-a.wav"
        audio_a.write_bytes(b"RIFF")
        windows = 3
        job_a_ref: dict = {}
        second_run_paused = threading.Event()
        release_second_run = threading.Event()

        def on_chunk(index):
            if index == 2:
                queue.preempt_job(job_a_ref["id"])
            elif index == 4:
                # Second run is now RUNNING (and shielded): pause it here so
                # the test can inspect the persisted queue mid-flight.
                second_run_paused.set()
                release_second_run.wait(timeout=10)

        engine_a = FakeEngine(on_chunk=on_chunk)
        queue._engines["base"] = engine_a
        queue._load_audio_file = lambda path: np.zeros(
            windows * _window_samples(queue), dtype=np.float32
        )

        job_a = queue.schedule_post_process(audio_a, None, tmp_path)
        job_a_ref["id"] = job_a.job_id

        try:
            gate["recording"] = False
            assert second_run_paused.wait(timeout=10), "second run never started"

            # The preempted job is RUNNING its shielded re-attempt and is
            # still persisted — a crash here must not lose it.
            assert job_a.status == PostProcessStatus.RUNNING
            assert job_a.shielded is True
            assert any(
                e.get("job_id") == job_a.job_id for e in queue._read_queue_file()
            )

            release_second_run.set()
            assert _wait_for(
                lambda: job_a.status == PostProcessStatus.COMPLETED
            ), f"job did not complete: {job_a.status}"
            # Terminal: the persisted entry is removed again.
            assert all(
                e.get("job_id") != job_a.job_id
                for e in queue._read_queue_file()
            )
        finally:
            release_second_run.set()
            queue.stop()
