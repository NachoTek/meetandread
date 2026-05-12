"""Tests for RecordingController stop-worker immediate-IDLE semantics
and post-processing cancellation/diarization wiring.

Covers T02 must-haves:
- _stop_worker reaches IDLE without calling _run_diarization directly.
- _stop_worker schedules PostProcessingQueue and returns to IDLE immediately.
- cancel_post_processing() is idempotent and safe with no queue/job.
- start() cancels in-flight post-processing before beginning a new recording.
- _on_post_process_complete_callback stores diarization_result.
- Queue scheduling/cancellation exceptions do not leave STOPPING stuck.
- Negative tests: slow/failed diarization, start after stop, malformed results.
"""

import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock, PropertyMock, call

import pytest

from meetandread.recording.controller import (
    RecordingController,
    ControllerState,
    ControllerError,
)
from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.transcription.post_processor import (
    PostProcessingQueue,
    PostProcessJob,
    PostProcessStatus,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeAudioSession:
    """Minimal AudioSession fake that returns a WAV path on stop."""

    def __init__(self, wav_path: Path):
        self._wav_path = wav_path
        self._state = "idle"

    def start(self, config=None):
        self._state = "recording"

    def stop(self) -> Path:
        self._state = "idle"
        return self._wav_path

    def get_state(self):
        return self._state

    def get_stats(self):
        from meetandread.audio import SessionStats
        return SessionStats(
            frames_recorded=0,
            frames_dropped=0,
            duration_seconds=0.0,
            source_stats=[],
        )


class FakePostProcessingQueue:
    """Fake PostProcessingQueue that tracks schedule/cancel calls."""

    def __init__(self, slow_diarize: bool = False):
        self.scheduled_jobs: List[PostProcessJob] = []
        self.cancelled_job_ids: List[str] = []
        self.cancel_current_called: int = 0
        self._started = False
        self._slow_diarize = slow_diarize
        self._job_counter = 0

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def schedule_post_process(
        self,
        audio_file: Path,
        realtime_transcript: TranscriptStore,
        output_dir: Path,
        model_size: Optional[str] = None,
    ) -> PostProcessJob:
        self._job_counter += 1
        job = PostProcessJob(
            job_id=f"fake-job-{self._job_counter}",
            audio_file=audio_file,
            realtime_transcript=realtime_transcript,
            output_dir=output_dir,
            model_size=model_size or "base",
        )
        self.scheduled_jobs.append(job)
        return job

    def cancel_job(self, job_id: str, reason: str = "") -> bool:
        self.cancelled_job_ids.append(job_id)
        return True

    def cancel_current_job(self, reason: str = "") -> bool:
        self.cancel_current_called += 1
        return True

    def get_job_status(self, job_id: str) -> Optional[PostProcessJob]:
        for j in self.scheduled_jobs:
            if j.job_id == job_id:
                return j
        return None


class FakeAccumulatingProcessor:
    """Minimal fake for AccumulatingTranscriptionProcessor."""

    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def load_model(self, progress_callback=None):
        pass

    def feed_audio(self, chunk):
        pass

    def get_stats(self):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(tmp_path: Path, *, enable_postprocess: bool = True) -> RecordingController:
    """Create a RecordingController with fakes wired in."""
    ctrl = RecordingController(enable_transcription=True)
    # Replace session
    wav_path = tmp_path / "test.wav"
    wav_path.write_text("fake wav")
    ctrl._session = FakeAudioSession(wav_path)

    if enable_postprocess:
        ctrl._post_processor = FakePostProcessingQueue()

    return ctrl


def _setup_transcription(ctrl: RecordingController):
    """Wire in a fake transcript store and processor."""
    ctrl._transcript_store = TranscriptStore()
    ctrl._transcript_store.start_recording()
    ctrl._transcript_store.add_words([
        Word("hello", 0.0, 0.5, 90),
        Word("world", 0.5, 1.0, 85),
    ])
    ctrl._transcription_processor = FakeAccumulatingProcessor()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestStopWorkerImmediateIdle:
    """_stop_worker reaches IDLE immediately without calling _run_diarization."""

    def test_stop_worker_reaches_idle(self, tmp_path: Path):
        """After _stop_worker, controller state is IDLE."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        ctrl._stop_worker()

        assert ctrl._state == ControllerState.IDLE

    def test_stop_worker_does_not_call_run_diarization(self, tmp_path: Path):
        """_stop_worker should NOT call _run_diarization directly."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        with patch.object(ctrl, "_run_diarization") as mock_diary:
            ctrl._stop_worker()
            mock_diary.assert_not_called()

    def test_stop_worker_does_not_call_run_diarization_for_postprocess(self, tmp_path: Path):
        """_stop_worker should NOT call _run_diarization_for_postprocess either."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        with patch.object(ctrl, "_run_diarization_for_postprocess") as mock_diary:
            ctrl._stop_worker()
            mock_diary.assert_not_called()

    def test_stop_worker_schedules_postprocess(self, tmp_path: Path):
        """_stop_worker should schedule a post-processing job."""
        ctrl = _make_controller(tmp_path)
        fake_queue = ctrl._post_processor
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        ctrl._stop_worker()

        assert len(fake_queue.scheduled_jobs) == 1
        job = fake_queue.scheduled_jobs[0]
        assert ctrl._post_process_job_id == job.job_id

    def test_stop_worker_saves_transcript_before_idle(self, tmp_path: Path):
        """Transcript is saved before going to IDLE."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        with patch.object(ctrl, "_save_transcript", return_value=tmp_path / "test.md") as mock_save:
            ctrl._stop_worker()
            mock_save.assert_called_once()

        assert ctrl._last_transcript_path == tmp_path / "test.md"

    def test_stop_worker_notifies_recording_complete(self, tmp_path: Path):
        """on_recording_complete callback is called with paths."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        received = {}
        ctrl.on_recording_complete = lambda wav, transcript: received.update(
            wav=wav, transcript=transcript
        )

        ctrl._stop_worker()

        assert "wav" in received
        assert received["wav"] is not None

    def test_stop_worker_error_goes_to_error_state(self, tmp_path: Path):
        """If session.stop() raises, controller goes to ERROR state."""
        ctrl = _make_controller(tmp_path)
        ctrl._state = ControllerState.RECORDING

        # Make session.stop raise
        ctrl._session.stop = MagicMock(side_effect=RuntimeError("audio device error"))

        ctrl._stop_worker()

        assert ctrl._state == ControllerState.ERROR
        assert ctrl._error is not None
        assert "audio device error" in ctrl._error.message

    def test_stop_worker_no_transcript_still_idle(self, tmp_path: Path):
        """When no transcript store, worker still reaches IDLE."""
        ctrl = _make_controller(tmp_path)
        ctrl._transcript_store = None
        ctrl._transcription_processor = None
        ctrl._state = ControllerState.RECORDING

        ctrl._stop_worker()

        assert ctrl._state == ControllerState.IDLE

    def test_stop_worker_no_postprocess_queue_still_idle(self, tmp_path: Path):
        """When no post-processing queue, worker still reaches IDLE."""
        ctrl = _make_controller(tmp_path, enable_postprocess=False)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        ctrl._stop_worker()

        assert ctrl._state == ControllerState.IDLE


class TestCancelPostProcessing:
    """cancel_post_processing() is idempotent and safe."""

    def test_cancel_with_no_queue(self, tmp_path: Path):
        """No-op when _post_processor is None."""
        ctrl = _make_controller(tmp_path, enable_postprocess=False)
        ctrl._post_processor = None
        # Should not raise
        ctrl.cancel_post_processing()

    def test_cancel_with_no_job(self, tmp_path: Path):
        """No-op when _post_process_job_id is None."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = None
        ctrl.cancel_post_processing()
        # cancel_current_job should still be called for safety
        assert ctrl._post_processor.cancel_current_called == 1

    def test_cancel_cancels_known_job(self, tmp_path: Path):
        """Cancels the tracked job by ID."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "job-abc"
        ctrl.cancel_post_processing()
        assert "job-abc" in ctrl._post_processor.cancelled_job_ids
        assert ctrl._post_process_job_id is None

    def test_cancel_clears_job_id(self, tmp_path: Path):
        """After cancellation, _post_process_job_id is cleared."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "job-123"
        ctrl.cancel_post_processing()
        assert ctrl._post_process_job_id is None

    def test_cancel_idempotent(self, tmp_path: Path):
        """Calling twice doesn't raise."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "job-1"
        ctrl.cancel_post_processing()
        ctrl.cancel_post_processing()  # Second call with None job_id
        # Only one explicit cancel_job call (the second call has job_id=None)
        assert "job-1" in ctrl._post_processor.cancelled_job_ids

    def test_cancel_handles_exception(self, tmp_path: Path):
        """Exceptions from cancel_job are caught and logged."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "job-err"
        ctrl._post_processor.cancel_job = MagicMock(side_effect=RuntimeError("broken"))
        # Should not raise
        ctrl.cancel_post_processing()

    def test_cancel_cancel_current_handles_exception(self, tmp_path: Path):
        """Exceptions from cancel_current_job are caught."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = None
        ctrl._post_processor.cancel_current_job = MagicMock(side_effect=RuntimeError("broken"))
        ctrl.cancel_post_processing()


class TestStartCancelsPostProcessing:
    """start() cancels in-flight post-processing before beginning."""

    def test_start_cancels_postprocess(self, tmp_path: Path):
        """start() should call cancel_post_processing before recording."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "old-job"
        ctrl._state = ControllerState.IDLE

        with patch.object(ctrl, "cancel_post_processing") as mock_cancel:
            # Let start() proceed past state validation but fail at source
            try:
                ctrl.start({"mic"})
            except Exception:
                pass
            mock_cancel.assert_called_once()

    def test_start_after_stop_cancels_old_job(self, tmp_path: Path):
        """Recording again after stop cancels the previous post-process job."""
        ctrl = _make_controller(tmp_path)
        fake_queue = ctrl._post_processor
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        # Stop — schedules a job
        ctrl._stop_worker()
        assert ctrl._state == ControllerState.IDLE
        assert len(fake_queue.scheduled_jobs) == 1

        # Start a new recording — should cancel the old job
        with patch.object(ctrl, "_init_transcription", return_value=None):
            with patch.object(ctrl, "_build_source_configs", return_value=[]):
                ctrl.start({"mic"})

        # The old job should have been cancelled
        assert len(fake_queue.cancelled_job_ids) > 0


class TestPostProcessCompleteCallback:
    """_on_post_process_complete_callback stores diarization_result."""

    def test_stores_diarization_result(self, tmp_path: Path):
        """Callback stores diarization_result in _last_diarization_result."""
        ctrl = _make_controller(tmp_path)

        mock_result_obj = MagicMock()
        mock_result_obj.num_speakers = 2

        result_dict = {
            "transcript_path": str(tmp_path / "transcript.md"),
            "word_count": 10,
            "realtime_word_count": 8,
            "model_used": "base",
            "diarization_result": mock_result_obj,
        }

        # Create the transcript file for WER computation
        transcript_path = tmp_path / "transcript.md"
        transcript_path.write_text("# Transcript\n\nhello world\n\n---\n\n<!-- METADATA: {\"words\": []} -->\n")

        ctrl._on_post_process_complete_callback("job-1", result_dict)

        assert ctrl._last_diarization_result is mock_result_obj

    def test_no_diarization_result_does_not_overwrite(self, tmp_path: Path):
        """When result has no diarization_result, _last_diarization_result is not overwritten."""
        ctrl = _make_controller(tmp_path)

        # Pre-set a result
        prev = MagicMock()
        ctrl._last_diarization_result = prev

        result_dict = {
            "transcript_path": str(tmp_path / "transcript.md"),
            "word_count": 10,
            "realtime_word_count": 8,
            "model_used": "base",
        }

        ctrl._on_post_process_complete_callback("job-2", result_dict)

        # Should still be the previous value
        assert ctrl._last_diarization_result is prev

    def test_none_diarization_result_does_not_overwrite(self, tmp_path: Path):
        """When diarization_result is None, _last_diarization_result is not overwritten."""
        ctrl = _make_controller(tmp_path)

        prev = MagicMock()
        ctrl._last_diarization_result = prev

        result_dict = {
            "transcript_path": str(tmp_path / "transcript.md"),
            "word_count": 10,
            "realtime_word_count": 8,
            "model_used": "base",
            "diarization_result": None,
        }

        ctrl._on_post_process_complete_callback("job-3", result_dict)

        assert ctrl._last_diarization_result is prev

    def test_calls_on_post_process_complete(self, tmp_path: Path):
        """External callback is still called."""
        ctrl = _make_controller(tmp_path)

        transcript_path = tmp_path / "transcript.md"
        transcript_path.write_text("# Transcript\n\nhello world\n\n---\n\n<!-- METADATA: {\"words\": []} -->\n")

        received = {}
        ctrl.on_post_process_complete = lambda jid, path: received.update(jid=jid, path=path)

        result_dict = {
            "transcript_path": str(transcript_path),
            "word_count": 10,
            "realtime_word_count": 8,
            "model_used": "base",
        }

        ctrl._on_post_process_complete_callback("job-4", result_dict)

        assert received["jid"] == "job-4"
        assert received["path"] == transcript_path

    def test_malformed_result_no_transcript_path(self, tmp_path: Path):
        """Callback handles missing transcript_path gracefully."""
        ctrl = _make_controller(tmp_path)

        received = []
        ctrl.on_post_process_complete = lambda jid, path: received.append((jid, path))

        result_dict = {
            "word_count": 10,
            "realtime_word_count": 8,
        }

        # Should not raise
        ctrl._on_post_process_complete_callback("job-5", result_dict)
        # Callback should NOT have been called (no valid transcript_path)
        assert len(received) == 0


class TestStopWorkerNoStuckOnFailure:
    """Queue scheduling/cancellation exceptions cannot leave STOPPING stuck."""

    def test_schedule_exception_goes_to_error(self, tmp_path: Path):
        """If schedule_post_process raises, controller goes to ERROR not STOPPING."""
        ctrl = _make_controller(tmp_path)
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        # Make schedule_post_process raise
        ctrl._post_processor.schedule_post_process = MagicMock(
            side_effect=RuntimeError("queue broken")
        )

        ctrl._stop_worker()

        # Should have gone to ERROR, not stuck at STOPPING
        assert ctrl._state == ControllerState.ERROR
        assert "queue broken" in ctrl._error.message


class TestNegativeScenarios:
    """Negative tests for edge cases."""

    def test_slow_diarization_does_not_block_idle(self, tmp_path: Path):
        """Even if diarization is slow, controller is already IDLE."""
        ctrl = _make_controller(tmp_path)
        fake_queue = FakePostProcessingQueue(slow_diarize=True)
        ctrl._post_processor = fake_queue
        _setup_transcription(ctrl)
        ctrl._state = ControllerState.RECORDING

        # The _stop_worker should complete quickly (no inline diarization)
        start = _time.monotonic()
        ctrl._stop_worker()
        elapsed = _time.monotonic() - start

        assert ctrl._state == ControllerState.IDLE
        # Should be fast — under 1 second since diarization is not inline
        assert elapsed < 1.0

    def test_failed_diarization_in_callback_non_fatal(self, tmp_path: Path):
        """If diarization callback fails, post-processing still completes."""
        # This tests that the PostProcessingQueue catches diarization errors
        # (tested in test_transcript_management.py), but the controller
        # wiring should handle a None diarization_result gracefully.
        ctrl = _make_controller(tmp_path)

        result_dict = {
            "transcript_path": str(tmp_path / "transcript.md"),
            "word_count": 10,
            "realtime_word_count": 8,
            "model_used": "base",
            "diarization_result": None,  # Failed diarization returns None
        }

        transcript_path = tmp_path / "transcript.md"
        transcript_path.write_text("# Transcript\n\nhello\n\n---\n\n<!-- METADATA: {\"words\": []} -->\n")

        # Should not raise
        ctrl._on_post_process_complete_callback("job-fail", result_dict)

    def test_start_cancels_noop_when_no_queue(self, tmp_path: Path):
        """start() with no post-processor doesn't crash."""
        ctrl = _make_controller(tmp_path, enable_postprocess=False)
        ctrl._post_processor = None
        ctrl._state = ControllerState.IDLE

        # Should not raise during cancel_post_processing
        with patch.object(ctrl, "_init_transcription", return_value=None):
            with patch.object(ctrl, "_build_source_configs", return_value=[]):
                error = ctrl.start({"mic"})

        # "No valid audio sources" error from empty source configs
        assert error is not None
        assert "No valid audio sources" in error.message

    def test_malformed_completion_result_empty_dict(self, tmp_path: Path):
        """Empty result dict doesn't crash the callback."""
        ctrl = _make_controller(tmp_path)

        # Should not raise
        ctrl._on_post_process_complete_callback("job-empty", {})

    def test_cancel_post_processing_exception_safe(self, tmp_path: Path):
        """cancel_post_processing swallows all exceptions."""
        ctrl = _make_controller(tmp_path)
        ctrl._post_process_job_id = "bad-job"
        ctrl._post_processor.cancel_job = MagicMock(side_effect=Exception("fatal"))
        ctrl._post_processor.cancel_current_job = MagicMock(side_effect=Exception("also-fatal"))

        # Should not raise
        ctrl.cancel_post_processing()
