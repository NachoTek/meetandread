"""Post-processing writes durable Outcomes and re-queues Stalled recordings.

Issue #62: on a job's terminal transition the queue writes a Post-processing
Outcome into the Recording's Transcript Footer; a Stalled Recording (no
Outcome) is automatically re-queued when its Audio still exists and
Post-processing is enabled; a no-Outcome Recording whose Audio is gone gets
a Failed (audio-missing) Outcome.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetandread.config.models import AppSettings
from meetandread.transcription import transcript_footer
from meetandread.transcription.engine import (
    TranscriptionError,
    TranscriptionSegment,
    TranscriptionSuccess,
    WordInfo,
)
from meetandread.transcription.post_processor import (
    PostProcessJob,
    PostProcessStatus,
    PostProcessingQueue,
)
from meetandread.transcription.transcript_footer import PostProcessOutcome
from meetandread.transcription.transcript_store import TranscriptStore, Word
from tests.footer_test_helpers import write_transcript

import meetandread.audio.storage.paths as paths_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_with_words(*texts: str, speaker_id=None) -> TranscriptStore:
    store = TranscriptStore()
    store.start_recording()
    words = [
        Word(
            text=t,
            start_time=i * 1.0,
            end_time=i * 1.0 + 0.9,
            confidence=90,
            speaker_id=speaker_id,
        )
        for i, t in enumerate(texts)
    ]
    store.add_words(words)
    return store


def _make_queue(tmp_path: Path, monkeypatch, settings: AppSettings = None):
    """Construct a queue whose persistence file lands in *tmp_path*."""
    data = tmp_path / "queue-data"
    data.mkdir(exist_ok=True)
    monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)
    return PostProcessingQueue(settings or AppSettings())


def _make_job(tmp_path: Path, job_id: str = "job-1", with_transcript=True) -> PostProcessJob:
    audio_file = tmp_path / f"recording_{job_id}.wav"
    audio_file.write_bytes(b"\x00")
    if with_transcript:
        write_transcript(
            tmp_path / f"recording_{job_id}.md",
            "# Transcript\n\nlive words",
            {"recording_start_time": "2026-08-14T09:00:00", "word_count": 2},
        )
    return PostProcessJob(
        job_id=job_id,
        audio_file=audio_file,
        realtime_transcript=_store_with_words("hello world", speaker_id="SPK_0"),
        output_dir=tmp_path,
        model_size="base",
    )


def _read_outcome(md_path: Path):
    return transcript_footer.read_post_process_outcome(
        md_path.read_text(encoding="utf-8")
    )


def _engine_returning(segments):
    engine = MagicMock()
    engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=segments)
    return engine


def _single_word_segment(text: str = "hello"):
    return TranscriptionSegment(
        text=text,
        confidence=90,
        start=0.0,
        end=1.0,
        words=[WordInfo(text=text, start=0.0, end=1.0, confidence=90)],
    )


# ---------------------------------------------------------------------------
# Terminal transitions write durable Outcomes
# ---------------------------------------------------------------------------


class TestCompletedJobWritesOutcome:
    """A completed job writes a Completed Outcome into the Transcript Footer."""

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_completed_outcome_written(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        mock_engine.return_value = _engine_returning([_single_word_segment()])

        queue._process_job(job)

        assert job.status == PostProcessStatus.COMPLETED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_COMPLETED
        assert outcome.error is None
        assert outcome.attempted_at  # ISO timestamp recorded

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_zero_speaker_completion_is_completed(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        """No-speech is a legitimate result, not a failure."""
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = PostProcessJob(
            job_id="quiet",
            audio_file=tmp_path / "recording_quiet.wav",
            realtime_transcript=None,
            output_dir=tmp_path,
            model_size="base",
        )
        job.audio_file.write_bytes(b"\x00")
        write_transcript(
            tmp_path / "recording_quiet.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        mock_engine.return_value = _engine_returning([])  # no speech

        queue._process_job(job)

        assert job.status == PostProcessStatus.COMPLETED
        outcome = _read_outcome(tmp_path / "recording_quiet.md")
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_COMPLETED


class TestFailedJobWritesOutcome:
    """A failed job writes a Failed Outcome with stage + error + attempted_at."""

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    def test_engine_load_failure_stage(self, mock_engine, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_engine.side_effect = RuntimeError("torch not available")

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_FAILED
        assert outcome.stage == transcript_footer.STAGE_ENGINE_LOAD
        assert "torch not available" in outcome.error
        assert outcome.attempted_at

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_transcription_error_stage(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        engine = MagicMock()
        engine.transcribe_chunk.return_value = TranscriptionError(
            error_type="oom", message="out of memory"
        )
        mock_engine.return_value = engine

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.stage == transcript_footer.STAGE_TRANSCRIBE
        assert "out of memory" in outcome.error

    def test_missing_audio_stage(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        job.audio_file.unlink()  # audio gone before the job ran

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.stage == transcript_footer.STAGE_AUDIO_MISSING

    def test_failed_outcome_write_failure_does_not_crash(self, tmp_path, monkeypatch):
        """Outcome write failure must not crash or block the terminal state."""
        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        job.audio_file.unlink()
        # Transcript file exists but has no usable footer: replace it.
        (tmp_path / "recording_job-1.md").write_text("# Bare markdown\n", "utf-8")

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        # The stem is quarantined so the requeue scan cannot hot-loop it.
        assert job.audio_file.stem in queue._no_outcome_stems


class TestCancelledJobWritesNoOutcome:
    """Cancellation leaves the Recording Stalled (no Outcome)."""

    def test_cancelled_before_start_writes_no_outcome(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        job.cancel_requested = True

        queue._process_job(job)

        assert job.status == PostProcessStatus.CANCELLED
        assert _read_outcome(tmp_path / "recording_job-1.md") is None


# ---------------------------------------------------------------------------
# Requeue predicate
# ---------------------------------------------------------------------------


def _meta(tmp_path: Path, stem: str, *, outcome=None, wav_exists=True):
    from meetandread.transcription.transcript_scanner import RecordingMeta

    return RecordingMeta(
        path=tmp_path / "transcripts" / f"{stem}.md",
        recording_time="2026-08-14T09:00:00",
        word_count=10,
        speaker_count=1,
        speakers=["SPK_0"],
        duration_seconds=30.0,
        wav_exists=wav_exists,
        post_process_outcome=outcome,
    )


def _active_job(tmp_path: Path, stem: str, status=PostProcessStatus.PENDING):
    return PostProcessJob(
        job_id=f"live-{stem}",
        audio_file=tmp_path / "recordings" / f"{stem}.wav",
        realtime_transcript=None,
        output_dir=tmp_path / "transcripts",
        model_size="base",
        status=status,
    )


class TestRequeuePredicate:
    """Requeue requires: no Outcome ∧ audio ∧ enabled ∧ not already queued."""

    def test_stalled_with_audio_and_enabled_is_requeueable(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is True

    def test_recording_with_outcome_is_not_requeued(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        meta = _meta(
            tmp_path,
            "recording_a",
            outcome=PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            ),
        )

        assert queue._should_requeue_recording(meta) is False

    def test_missing_audio_is_not_requeued(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        meta = _meta(tmp_path, "recording_a", wav_exists=False)

        assert queue._should_requeue_recording(meta) is False

    def test_post_processing_disabled_is_not_requeued(self, tmp_path, monkeypatch):
        settings = AppSettings()
        settings.transcription.enable_postprocessing = False
        queue = _make_queue(tmp_path, monkeypatch, settings)
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False

    def test_speakers_disabled_is_not_requeued(self, tmp_path, monkeypatch):
        settings = AppSettings()
        settings.speaker.enabled = False
        queue = _make_queue(tmp_path, monkeypatch, settings)
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False

    def test_live_pending_job_is_not_requeued(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        job = _active_job(tmp_path, "recording_a", status=PostProcessStatus.PENDING)
        with queue._jobs_lock:
            queue._jobs[job.job_id] = job
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False

    def test_live_running_job_is_not_requeued(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        job = _active_job(tmp_path, "recording_a", status=PostProcessStatus.RUNNING)
        with queue._jobs_lock:
            queue._jobs[job.job_id] = job
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False

    def test_persisted_job_entry_is_not_requeued(self, tmp_path, monkeypatch):
        queue = _make_queue(tmp_path, monkeypatch)
        entries = [
            {
                "job_id": "persisted-1",
                "audio_file": str(tmp_path / "recordings" / "recording_a.wav"),
                "output_dir": str(tmp_path / "transcripts"),
                "model_size": "base",
            }
        ]
        queue._write_queue_file(entries)
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False

    def test_terminal_job_this_session_is_requeueable(self, tmp_path, monkeypatch):
        """A cancelled (outcome-less terminal) job leaves the Recording Stalled."""
        queue = _make_queue(tmp_path, monkeypatch)
        job = _active_job(tmp_path, "recording_a", status=PostProcessStatus.CANCELLED)
        with queue._jobs_lock:
            queue._jobs[job.job_id] = job
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is True

    def test_no_outcome_stem_is_not_requeued(self, tmp_path, monkeypatch):
        """A stem whose Outcome write failed is quarantined for this session."""
        queue = _make_queue(tmp_path, monkeypatch)
        queue._no_outcome_stems.add("recording_a")
        meta = _meta(tmp_path, "recording_a")

        assert queue._should_requeue_recording(meta) is False


# ---------------------------------------------------------------------------
# Requeue scan (integration through the transcript scanner)
# ---------------------------------------------------------------------------


class TestRequeueScan:
    """The scan re-queues Stalled recordings and marks missing-audio ones."""

    @pytest.fixture()
    def dirs(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "transcripts"
        recordings = tmp_path / "recordings"
        data = tmp_path / "data"
        transcripts.mkdir()
        recordings.mkdir()
        data.mkdir()
        monkeypatch.setattr(
            "meetandread.transcription.transcript_scanner.get_recordings_dir",
            lambda: recordings,
        )
        monkeypatch.setattr(paths_mod, "get_transcripts_dir", lambda: transcripts)
        monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)
        monkeypatch.setattr(paths_mod, "get_recordings_dir", lambda: recordings)
        return transcripts, recordings, data

    def _queue(self):
        """A queue that looks running without spawning a worker thread.

        Scheduling through the scan would otherwise start() the real worker
        and load a real Whisper engine; faking _is_running keeps the
        scheduled jobs PENDING for deterministic assertions.
        """
        queue = PostProcessingQueue(AppSettings())
        queue._is_running = True
        return queue

    def test_stalled_recording_is_requeued(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        queue = self._queue()

        count = queue.requeue_stalled_recordings()

        assert count == 1
        assert queue.get_status_for_audio(recordings / "recording_a.wav") == (
            PostProcessStatus.PENDING
        )

    def test_recording_with_outcome_is_left_alone(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "post_process": PostProcessOutcome(
                    status=transcript_footer.STATUS_COMPLETED,
                    attempted_at="2026-08-14T10:00:00",
                ).to_block(),
            },
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        queue = PostProcessingQueue(AppSettings())

        assert queue.requeue_stalled_recordings() == 0

    def test_missing_audio_gets_failed_outcome(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_gone.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        # No WAV for recording_gone.
        queue = PostProcessingQueue(AppSettings())

        count = queue.requeue_stalled_recordings()

        assert count == 0
        outcome = _read_outcome(transcripts / "recording_gone.md")
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_FAILED
        assert outcome.stage == transcript_footer.STAGE_AUDIO_MISSING
        # And it is not requeued on a second scan.
        assert queue.requeue_stalled_recordings() == 0

    def test_scan_skips_when_post_processing_disabled(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        settings = AppSettings()
        settings.transcription.enable_postprocessing = False
        queue = PostProcessingQueue(settings)

        assert queue.requeue_stalled_recordings() == 0
        assert queue.get_status_for_audio(recordings / "recording_a.wav") is None

    def test_enabling_post_processing_requeues_red_rows(self, dirs):
        """A 'not post-processed' row becomes requeued once enabled (live settings)."""
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        settings = AppSettings()
        settings.transcription.enable_postprocessing = False
        queue = PostProcessingQueue(settings)
        queue._is_running = True
        assert queue.requeue_stalled_recordings() == 0

        settings.transcription.enable_postprocessing = True  # same live object

        assert queue.requeue_stalled_recordings() == 1
        assert queue.get_status_for_audio(recordings / "recording_a.wav") == (
            PostProcessStatus.PENDING
        )

    def test_scan_does_not_double_enqueue_live_job(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        queue = self._queue()

        queue.requeue_stalled_recordings()
        count2 = queue.requeue_stalled_recordings()

        assert count2 == 0  # second scan sees the now-PENDING job

    def test_startup_scan_runs_after_recovery(self, dirs):
        """start(): recover → dependency-repair conversion → Stalled scan."""
        transcripts, recordings, _ = dirs
        queue = PostProcessingQueue(AppSettings())
        calls: list[str] = []

        with patch.object(
            queue, "_recover_pending_jobs", side_effect=lambda: calls.append("recover")
        ), patch.object(
            queue,
            "requeue_dependency_failed_recordings",
            side_effect=lambda: calls.append("convert"),
        ), patch.object(
            queue,
            "requeue_stalled_recordings",
            side_effect=lambda: calls.append("scan"),
        ):
            try:
                queue.start()
            finally:
                queue.stop()

        assert calls == ["recover", "convert", "scan"]

    def test_terminal_transition_triggers_scan(self, dirs):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_b.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:00:00"},
        )
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {"recording_start_time": "2026-08-14T09:05:00"},
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        queue = PostProcessingQueue(AppSettings())
        # Pretend the worker is live so the terminal hook scans (no real
        # worker thread: the scheduled job stays PENDING for assertions).
        queue._is_running = True

        job = PostProcessJob(
            job_id="terminal",
            audio_file=recordings / "recording_b.wav",
            realtime_transcript=None,
            output_dir=transcripts,
            model_size="base",
            status=PostProcessStatus.COMPLETED,
        )
        with patch.object(
            queue, "requeue_stalled_recordings", wraps=queue.requeue_stalled_recordings
        ) as mock_scan:
            queue._on_job_terminal(job)
            assert mock_scan.called

        # recording_a (Stalled, unrelated to the terminal job) got requeued.
        assert queue.get_status_for_audio(recordings / "recording_a.wav") == (
            PostProcessStatus.PENDING
        )
