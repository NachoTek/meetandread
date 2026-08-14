"""Post-processing queue for hybrid transcription system.

When recording stops, this system:
1. Queues the full audio file for re-transcription with a stronger model
2. Runs transcription in a background thread to avoid blocking UI
3. Overwrites the original transcript .md in-place with the stronger result
4. Allows easy model swapping for different use cases

HYBRID TRANSCRIPTION FLOW:
┌─────────────┐    Real-time    ┌──────────────┐
│  Audio      │ ──stream─────→ │  Tiny Model  │ ──UI────→ Display
│  Capture    │    (chunked)   │  (fast)      │
└─────────────┘                └──────────────┘
         │                           │
         │           Stop Recording  │
         ▼                           ▼
┌──────────────────────────────────────────┐
│  Full Audio File                       │
│  (original recording)                  │
└──────────────────────────────────────────┘
                    │
                    ▼ Post-processing queue
        ┌──────────────────────┐
        │  Stronger Model      │ (base/small)
        │  (better accuracy)     │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Enhanced Transcript │
        │  (saved alongside)   │
        └──────────────────────┘

Usage:
    # During recording
    post_processor = PostProcessingQueue(config)
    
    # When recording stops
    post_processor.schedule_post_process(
        audio_file=wav_path,
        realtime_transcript=transcript_store,
        output_dir=output_path.parent
    )
    
    # Check progress
    status = post_processor.get_status()
"""

import logging
import threading
import queue
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any, Set
import numpy as np

logger = logging.getLogger(__name__)

from meetandread.config.models import AppSettings  # noqa: E402
from meetandread.transcription.engine import WhisperTranscriptionEngine, TranscriptionSegment  # noqa: E402
from meetandread.transcription import transcript_footer  # noqa: E402
from meetandread.transcription.transcript_footer import PostProcessOutcome  # noqa: E402
from meetandread.transcription.transcript_store import TranscriptStore, Word  # noqa: E402
from meetandread.audio.utils import load_wav_as_float32_mono  # noqa: E402


class PostProcessStatus(Enum):
    """Status of a post-processing job."""
    PENDING = auto()      # Queued but not started
    RUNNING = auto()      # Currently processing
    COMPLETED = auto()    # Successfully completed
    FAILED = auto()       # Failed with error
    CANCELLED = auto()    # Cancelled by caller


class PostProcessFailure(Exception):
    """A post-processing failure tagged with its failing stage.

    The stage is one of the Transcript Footer Outcome stages (see
    ``transcript_footer``); it is written into the Recording's Failed
    Outcome so the Library can explain *where* Post-processing broke.
    """

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass
class PostProcessJob:
    """A single post-processing job.

    Attributes:
        job_id: Unique identifier for this job
        audio_file: Path to the audio file to transcribe
        realtime_transcript: The real-time transcript for comparison
        output_dir: Directory to save enhanced transcript
        model_size: Whisper model size for post-processing
        status: Current status of the job
        progress: Progress percentage (0-100)
        result: Result data after completion
        error: Error message if failed
        cancel_requested: True when cancellation has been requested
        cancel_reason: Optional reason string for the cancellation
        diarization_error: Warning/error from diarization step (non-fatal)
        attempted_at: ISO timestamp captured when the job started running;
            recorded in the durable Post-processing Outcome
    """
    job_id: str
    audio_file: Path
    realtime_transcript: Optional[TranscriptStore]
    output_dir: Path
    model_size: str
    status: PostProcessStatus = PostProcessStatus.PENDING
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    cancel_reason: Optional[str] = None
    diarization_error: Optional[str] = None
    attempted_at: Optional[str] = None


class PostProcessingQueue:
    """Manages post-processing transcription jobs.
    
    Runs transcription with a stronger model after recording stops,
    providing higher quality transcripts for archival while maintaining
    real-time performance during recording.
    
    The queue processes jobs in a background thread to avoid blocking
    the UI or recording operations.
    
    Example:
        queue = PostProcessingQueue(settings)
        
        # Schedule post-processing when recording stops
        job = queue.schedule_post_process(
            audio_file=wav_path,
            realtime_transcript=transcript_store,
            output_dir=output_dir
        )
        
        # Check status later
        status = queue.get_job_status(job.job_id)
        if status.status == PostProcessStatus.COMPLETED:
            print(f"Transcript: {status.result['transcript_path']}")
    """
    
    def __init__(
        self,
        settings: AppSettings,
        on_progress: Optional[Callable[[str, int], None]] = None,
        on_complete: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        is_recording_callback: Optional[Callable[[], bool]] = None,
        diarize_callback: Optional[Callable[[Path], Any]] = None,
        apply_speaker_labels_callback: Optional[
            Callable[[TranscriptStore, Any], None]
        ] = None,
        auto_requeue_stalled: bool = True,
    ):
        """Initialize the post-processing queue.

        Args:
            settings: Application settings containing model configuration
            on_progress: Callback(job_id, progress_pct) for progress updates
            on_complete: Callback(job_id, result) when job completes
            is_recording_callback: Callable returning True while recording is
                active.  When provided, the queue waits (without busy-spinning)
                for the callback to return False before starting each job.
            diarize_callback: Callable(wav_path) -> DiarizationResult.
                When provided, diarization runs before stronger-model
                transcription.  Errors are caught and recorded in
                ``job.diarization_error`` but do not block transcription.
            apply_speaker_labels_callback: Callable(transcript_store,
                diarization_result) that applies speaker labels to a
                transcript store. Called with the diarization result after
                stronger-model transcription completes.
            auto_requeue_stalled: When True (default), the queue scans the
                Library for Stalled recordings (no Post-processing Outcome)
                at startup — after pending-job recovery — and again each time
                a job reaches a terminal state, re-queuing those whose Audio
                still exists while Post-processing is enabled (issue #62).
        """
        self._settings = settings
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._is_recording_callback = is_recording_callback
        self._diarize_callback = diarize_callback
        self._apply_speaker_labels_callback = apply_speaker_labels_callback
        self._auto_requeue_stalled = auto_requeue_stalled

        # Job queue
        self._job_queue: queue.Queue[PostProcessJob] = queue.Queue()
        self._jobs: Dict[str, PostProcessJob] = {}
        self._jobs_lock = threading.Lock()

        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_event = threading.Event()

        # Engine cache - one engine per model size
        self._engines: Dict[str, WhisperTranscriptionEngine] = {}
        self._engines_lock = threading.Lock()

        # Currently running job (for cancel_current_job)
        self._current_job: Optional[PostProcessJob] = None
        self._current_job_lock = threading.Lock()

        # Stalled-requeue scans (startup + after each terminal job)
        self._requeue_lock = threading.Lock()
        # Recording stems whose Outcome could not be written (unusable
        # Transcript Footer). Quarantined for this session so the requeue
        # scan cannot hot-loop an unwritable transcript.
        self._no_outcome_stems: Set[str] = set()

        # Queue persistence
        from meetandread.audio.storage.paths import get_data_dir
        self._queue_file = get_data_dir() / "post_processing_queue.json"
        self._queue_file_lock = threading.Lock()  # serialises read-modify-write

    def start(self) -> None:
        """Start the background worker thread."""
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PostProcessingWorker"
        )
        self._worker_thread.start()
        logger.info("PostProcessingQueue worker started")

        # Recover any persisted pending jobs, then re-queue Stalled
        # recordings (no Outcome) — recovery first so the scan sees the
        # recovered jobs and does not double-enqueue them.
        self._recover_pending_jobs()
        if self._auto_requeue_stalled:
            self.requeue_stalled_recordings()
    
    def stop(self) -> None:
        """Stop the background worker thread."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        
        logger.info("PostProcessingQueue worker stopped")
    
    def schedule_post_process(
        self,
        audio_file: Path,
        realtime_transcript: Optional[TranscriptStore],
        output_dir: Path,
        model_size: Optional[str] = None
    ) -> PostProcessJob:
        """Schedule a post-processing job.
        
        Args:
            audio_file: Path to the recorded audio file
            realtime_transcript: The real-time transcript for comparison, or
                ``None`` when the Recording has none (a re-queued Stalled
                Recording is post-processed from its Audio alone)
            output_dir: Directory to save the enhanced transcript
            model_size: Model size for post-processing (default from settings)
        
        Returns:
            The scheduled job
        """
        import uuid
        
        # Use configured post-process model or default
        if model_size is None:
            model_size = self._settings.transcription.postprocess_model_size
            if not model_size or model_size == "auto":
                # Default to base for post-processing if not set
                model_size = "base"
        
        job = PostProcessJob(
            job_id=str(uuid.uuid4())[:8],
            audio_file=audio_file,
            realtime_transcript=realtime_transcript,
            output_dir=output_dir,
            model_size=model_size
        )
        
        with self._jobs_lock:
            self._jobs[job.job_id] = job
        
        self._job_queue.put(job)
        self._persist_job(job)
        logger.info(
            "Scheduled post-processing job %s with model %s", job.job_id, model_size
        )
        
        # Ensure worker is running
        if not self._is_running:
            self.start()
        
        return job
    
    def get_job_status(self, job_id: str) -> Optional[PostProcessJob]:
        """Get the current status of a job.
        
        Args:
            job_id: The job ID to check
        
        Returns:
            The job status or None if not found
        """
        with self._jobs_lock:
            return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[PostProcessJob]:
        """Get all jobs (pending, running, and completed).
        
        Returns:
            List of all jobs
        """
        with self._jobs_lock:
            return list(self._jobs.values())

    _PENDING_STATUSES = (
        PostProcessStatus.PENDING,
        PostProcessStatus.RUNNING,
    )

    def get_status_for_audio(self, audio_path: Path) -> Optional[PostProcessStatus]:
        """Return the Post-processing job status for a Recording, if any.

        Used by the Library rows to pick a per-row status pill
        (Queued / Processing NN% / Failed; terminal states fall back to
        the durable Outcome in the Transcript Footer).  Matching is by
        file stem so a Recording's WAV (in ``recordings/``) is recognised
        via its transcript companion path.

        When both a non-terminal (PENDING/RUNNING) and a terminal job exist
        for the same stem, the live one wins so an in-flight re-attempt is
        not masked by a stale FAILED entry.

        Returns ``None`` when no job targets this Recording.
        """
        target_stem = audio_path.stem
        with self._jobs_lock:
            terminal: Optional[PostProcessStatus] = None
            for job in self._jobs.values():
                if job.audio_file.stem != target_stem:
                    continue
                if job.status in self._PENDING_STATUSES:
                    return job.status
                terminal = job.status
            return terminal

    def get_progress_for_audio(self, audio_path: Path) -> Optional[int]:
        """Return the progress percent (0-100) of the in-flight job, or None.

        Returns the PENDING/RUNNING job's progress percent, or ``None`` when
        no in-flight job targets this Recording (complete, failed, cancelled,
        or never scheduled).
        """
        target_stem = audio_path.stem
        with self._jobs_lock:
            for job in self._jobs.values():
                if job.status in self._PENDING_STATUSES and job.audio_file.stem == target_stem:
                    return job.progress
            return None
    
    def cancel_job(self, job_id: str, reason: str = "") -> bool:
        """Request cancellation of a specific job.
        
        If the job is PENDING, it is immediately marked CANCELLED.
        If the job is RUNNING, the cancellation flag is set and the
        worker will abort at the next checkpoint.
        
        Args:
            job_id: The job ID to cancel
            reason: Optional reason for cancellation
        
        Returns:
            True if the job was found and cancellation was requested,
            False if the job was not found or already terminal.
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        
        if job is None:
            logger.warning("cancel_job: unknown job_id %s", job_id)
            return False
        
        if job.status in (PostProcessStatus.COMPLETED, PostProcessStatus.FAILED, PostProcessStatus.CANCELLED):
            logger.info("cancel_job: job %s already terminal (%s)", job_id, job.status.name)
            return False
        
        job.cancel_requested = True
        job.cancel_reason = reason or "cancelled by caller"
        
        if job.status == PostProcessStatus.PENDING:
            job.status = PostProcessStatus.CANCELLED
            logger.info(
                "Job %s cancelled while PENDING: %s", job_id, job.cancel_reason
            )
        else:
            logger.info(
                "Job %s cancellation requested while RUNNING: %s",
                job_id, job.cancel_reason,
            )
        
        return True
    
    def cancel_current_job(self, reason: str = "") -> bool:
        """Request cancellation of the currently running job.
        
        Args:
            reason: Optional reason for cancellation
        
        Returns:
            True if a running job was found and cancellation was requested.
        """
        with self._current_job_lock:
            job = self._current_job
        
        if job is None:
            return False
        
        return self.cancel_job(job.job_id, reason)
    
    def _worker_loop(self) -> None:
        """Background worker thread that processes jobs."""
        logger.info("Post-processing worker loop started")
        
        while self._is_running and not self._stop_event.is_set():
            try:
                # Get job with timeout to allow checking stop_event
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error in post-processing worker loop: %s", e)
                continue
            
            # Check if job was already cancelled while queued
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info(
                    "Job %s skipped (cancelled while queued)", job.job_id
                )
                self._finalize_terminal_job(job)
                continue
            
            # Idle-wait gate: if an is_recording_callback is provided and
            # recording is still active, wait for it to become idle before
            # processing.  This prevents post-processing from consuming CPU
            # while a new recording is starting.
            if self._is_recording_callback is not None:
                waited = False
                while (
                    self._is_running
                    and not self._stop_event.is_set()
                    and not job.cancel_requested
                    and self._is_recording_callback()
                ):
                    if not waited:
                        logger.info(
                            "Job %s idle-wait: recording active, deferring start",
                            job.job_id,
                        )
                        waited = True
                    self._stop_event.wait(timeout=0.5)
                
                if job.cancel_requested:
                    job.status = PostProcessStatus.CANCELLED
                    logger.info(
                        "Job %s cancelled during idle-wait", job.job_id
                    )
                    self._finalize_terminal_job(job)
                    continue
                
                if waited:
                    logger.info(
                        "Job %s idle-wait ended, proceeding", job.job_id
                    )
            
            # Track current job for cancel_current_job()
            with self._current_job_lock:
                self._current_job = job
            
            try:
                self._process_job(job)
            finally:
                with self._current_job_lock:
                    if self._current_job is job:
                        self._current_job = None
                self._finalize_terminal_job(job)

    def _finalize_terminal_job(self, job: PostProcessJob) -> None:
        """Handle a job that just reached a terminal state.

        Removes its persisted entry (so the stem no longer reads as
        queued to the Stalled requeue scan) and fires the drain-until-empty
        requeue hook.
        """
        if job.status in (
            PostProcessStatus.COMPLETED,
            PostProcessStatus.FAILED,
            PostProcessStatus.CANCELLED,
        ):
            self._unpersist_job(job.job_id)
            self._on_job_terminal(job)

    def _on_job_terminal(self, job: PostProcessJob) -> None:
        """Drain-until-empty hook fired after a job reaches a terminal state.

        Re-runs the Stalled requeue scan: any recording that became
        requeueable (e.g. a Cancelled job's Recording) is scheduled again,
        and each of those jobs will fire this hook on completion until a
        scan enqueues nothing.
        """
        if not self._auto_requeue_stalled:
            return
        if not self._is_running or self._stop_event.is_set():
            return
        try:
            self.requeue_stalled_recordings()
        except Exception as exc:
            logger.warning("Stalled-requeue scan failed (non-fatal): %s", exc)
    
    def _process_job(self, job: PostProcessJob) -> None:
        """Process a single post-processing job.

        On a terminal transition the job's durable Post-processing Outcome
        is written into the Recording's Transcript Footer: Completed (even
        for zero-speaker results) or Failed with the failing stage and
        reason.  A cancelled job writes no Outcome — the Recording stays
        Stalled and is picked up by the requeue scan.

        Args:
            job: The job to process
        """
        logger.info("Processing job %s with model %s", job.job_id, job.model_size)

        transcript_path: Optional[Path] = None
        try:
            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled before start", job.job_id)
                return

            # Update status; stamp the attempt time recorded in the Outcome.
            job.status = PostProcessStatus.RUNNING
            job.attempted_at = datetime.now().isoformat()
            self._update_progress(job, 10)

            # A Recording without its Audio can never be post-processed.
            if not job.audio_file.exists():
                raise PostProcessFailure(
                    transcript_footer.STAGE_AUDIO_MISSING,
                    f"Audio file missing: {job.audio_file.name}",
                )

            # Load or get engine
            logger.info("Job %s: loading engine for model %s", job.job_id, job.model_size)
            try:
                engine = self._get_or_create_engine(job.model_size)
            except PostProcessFailure:
                raise
            except Exception as exc:
                raise PostProcessFailure(
                    transcript_footer.STAGE_ENGINE_LOAD, str(exc)
                ) from exc
            logger.info("Job %s: engine loaded", job.job_id)
            self._update_progress(job, 15)

            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled after engine load", job.job_id)
                return

            # ---- Diarization step (optional, before transcription) ----
            diarization_result = None
            if self._diarize_callback is not None:
                try:
                    logger.info(
                        "Job %s: running diarization on %s",
                        job.job_id, job.audio_file.name,
                    )
                    diarization_result = self._diarize_callback(job.audio_file)
                    self._update_progress(job, 25)
                except ImportError:
                    job.diarization_error = "diarization dependencies not available"
                    logger.warning(
                        "Job %s: diarization skipped — %s",
                        job.job_id, job.diarization_error,
                    )
                    self._update_progress(job, 25)
                except Exception as exc:
                    job.diarization_error = str(exc)
                    logger.warning(
                        "Job %s: diarization failed (non-fatal): %s",
                        job.job_id, exc,
                    )
                    self._update_progress(job, 25)

            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled after diarization", job.job_id)
                return

            # Read audio file and transcribe with the stronger model.
            try:
                audio_data = self._load_audio_file(job.audio_file)
                self._update_progress(job, 35)

                logger.info(
                    "Transcribing %d samples with %s model for job %s...",
                    len(audio_data), job.model_size, job.job_id,
                )
                segments = engine.transcribe_chunk(audio_data, word_level=True)
            except PostProcessFailure:
                raise
            except Exception as exc:
                raise PostProcessFailure(
                    transcript_footer.STAGE_TRANSCRIBE, str(exc)
                ) from exc
            self._update_progress(job, 80)

            # Unwrap typed result (M019 changed transcribe_chunk to return
            # TranscriptionSuccess | TranscriptionError instead of raw segments)
            from meetandread.transcription.engine import TranscriptionError
            if isinstance(segments, TranscriptionError):
                raise PostProcessFailure(
                    transcript_footer.STAGE_TRANSCRIBE,
                    f"Post-processing transcription failed: "
                    f"{segments.error_type}: {segments.message}",
                )
            segments = segments.segments

            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled after transcription", job.job_id)
                return

            # Create post-processed transcript
            enhanced_store = self._create_post_processed_transcript(segments)
            self._update_progress(job, 85)

            # Transfer speaker labels from realtime transcript to post-processed words.
            # The realtime transcript has speaker labels from diarization; the
            # post-processed transcript has better word text/timing from the
            # stronger model but no speaker info.  Merge by time overlap.
            if job.realtime_transcript:
                self._transfer_speaker_labels(job.realtime_transcript, enhanced_store)

            # Apply diarization speaker labels if diarization ran successfully.
            if diarization_result is not None and self._apply_speaker_labels_callback is not None:
                try:
                    self._apply_speaker_labels_callback(enhanced_store, diarization_result)
                except Exception as exc:
                    logger.warning(
                        "Job %s: apply speaker labels failed (non-fatal): %s",
                        job.job_id, exc, exc_info=True,
                    )

            # ---- Checkpoint: not cancelled before save ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info(
                    "Job %s cancelled before save — transcript NOT overwritten",
                    job.job_id,
                )
                return

            self._update_progress(job, 90)

            # Save post-processed transcript (overwrites original .md).
            # Build speaker_matches from the post-processing diarization
            # result (which has identity matches from VoiceSignatureStore).
            # Falls back to carrying forward the realtime transcript's
            # matches when diarization didn't run.
            speaker_matches = None
            if diarization_result is not None and diarization_result.matches:
                speaker_matches = {}
                for label, match in diarization_result.matches.items():
                    speaker_matches[str(label)] = {
                        "identity_name": match.name,
                        "score": match.score,
                        "confidence": match.confidence,
                    }
            elif job.realtime_transcript:
                try:
                    base_name = job.audio_file.stem
                    original_path = job.output_dir / f"{base_name}.md"
                    speaker_matches = self._read_speaker_matches(original_path)
                except Exception:
                    logger.debug(
                        "Failed to read speaker_matches from %s "
                        "(transcript will save without them)",
                        original_path,
                    )
            transcript_path = self._save_post_processed_transcript(
                job, enhanced_store, speaker_matches=speaker_matches,
            )
            self._update_progress(job, 100)

            # Write the durable Completed Outcome.  A zero-speaker result is
            # a legitimate completion, not a failure.
            self._write_outcome(
                transcript_path,
                PostProcessOutcome(
                    status=transcript_footer.STATUS_COMPLETED,
                    attempted_at=job.attempted_at,
                ),
            )

            # Mark complete
            job.status = PostProcessStatus.COMPLETED
            job.result = {
                "transcript_path": str(transcript_path),
                "word_count": enhanced_store.get_word_count(),
                "realtime_word_count": (
                    job.realtime_transcript.get_word_count()
                    if job.realtime_transcript else 0
                ),
                "model_used": job.model_size,
                "diarization_result": diarization_result,
            }

            logger.info(
                "Job %s completed. Transcript: %s", job.job_id, transcript_path
            )

            # Notify completion
            if self._on_complete:
                self._on_complete(job.job_id, job.result)

        except Exception as e:
            job.status = PostProcessStatus.FAILED
            job.error = str(e)
            logger.error(
                "Job %s failed: %s", job.job_id, e, exc_info=True,
            )

            # Write the durable Failed Outcome with the failing stage.
            failed_outcome_path = (
                transcript_path
                if transcript_path is not None
                else job.output_dir / f"{job.audio_file.stem}.md"
            )
            stage = getattr(
                e, "stage", transcript_footer.STAGE_TRANSCRIBE,
            )
            self._write_outcome(
                failed_outcome_path,
                PostProcessOutcome(
                    status=transcript_footer.STATUS_FAILED,
                    attempted_at=job.attempted_at or datetime.now().isoformat(),
                    stage=stage,
                    error=str(e),
                ),
            )

            # Notify completion even on failure so UI can update
            if self._on_complete:
                try:
                    self._on_complete(job.job_id, {
                        "error": str(e),
                        "status": "failed",
                    })
                except Exception:
                    logger.debug(
                        "on_complete callback error (failure notification): "
                        "job_id=%s",
                        job.job_id,
                    )

    def _write_outcome(
        self, transcript_path: Optional[Path], outcome: PostProcessOutcome
    ) -> None:
        """Write a durable Post-processing Outcome into a Transcript Footer.

        A Recording whose Outcome cannot be written (missing file or no
        usable Transcript Footer) is quarantined for this session so the
        Stalled requeue scan cannot repeatedly re-run Post-processing that
        can never record its result.
        """
        if transcript_path is None:
            return
        written = False
        try:
            written = transcript_footer.write_post_process_outcome(
                transcript_path, outcome
            )
        except Exception as exc:
            logger.error(
                "Failed to write Post-processing Outcome to %s: %s",
                transcript_path, exc,
            )
        if written:
            return
        self._no_outcome_stems.add(transcript_path.stem)
        logger.error(
            "Cannot write Post-processing Outcome — no usable Transcript "
            "Footer in %s (recording quarantined from requeue this session)",
            transcript_path,
        )

    def _get_or_create_engine(self, model_size: str) -> WhisperTranscriptionEngine:
        """Get cached engine or create new one.
        
        Args:
            model_size: The model size to use
        
        Returns:
            WhisperTranscriptionEngine instance
        """
        with self._engines_lock:
            if model_size not in self._engines:
                logger.info("Creating new engine for model %s", model_size)
                engine = WhisperTranscriptionEngine(
                    model_size=model_size,
                    device="cpu",
                    compute_type="int8"
                )
                engine.load_model()
                self._engines[model_size] = engine
            
            return self._engines[model_size]
    
    def _load_audio_file(self, audio_file: Path) -> np.ndarray:
        """Load audio file into numpy array.

        Delegates to the shared :func:`load_wav_as_float32_mono` utility.

        Args:
            audio_file: Path to audio file

        Returns:
            Audio samples as float32 numpy array
        """
        return load_wav_as_float32_mono(audio_file)
    
    def _create_post_processed_transcript(
        self,
        segments: List[TranscriptionSegment],
    ) -> TranscriptStore:
        """Create TranscriptStore from transcription segments.

        Splits multi-word Whisper tokens into individual words, dividing each
        segment's real ``[start, end]`` evenly across its words.

        Timestamps are taken verbatim from Whisper. They are never stretched
        to fill the audio duration: previously the words were linearly rescaled
        when Whisper under-reported the duration, but that embedded silence
        into the word spans and made the playback highlight drift ahead of the
        audio during pauses (issue #21). Real timestamps preserve inter-word
        and trailing silence as gaps, which the highlighter holds on.

        Args:
            segments: Transcription segments from Whisper
        
        Returns:
            TranscriptStore with words
        """
        store = TranscriptStore()
        store.start_recording()
        
        words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                # Use word-level data if available, but split multi-word entries
                # (Whisper may return full sentences as single "word" tokens)
                for word_info in segment.words:
                    raw_text = word_info.text if hasattr(word_info, 'text') else str(word_info)
                    w_start = word_info.start if hasattr(word_info, 'start') else 0.0
                    w_end = word_info.end if hasattr(word_info, 'end') else 0.0
                    w_conf = word_info.confidence if hasattr(word_info, 'confidence') else 85

                    parts = raw_text.split()
                    if len(parts) <= 1:
                        # Single word — use as-is
                        words.append(Word(
                            text=raw_text.strip(),
                            start_time=w_start,
                            end_time=w_end,
                            confidence=w_conf,
                            speaker_id=None,
                        ))
                    else:
                        # Multi-word text — split and distribute timing evenly
                        duration = w_end - w_start
                        per_word = duration / len(parts)
                        for i, part in enumerate(parts):
                            words.append(Word(
                                text=part,
                                start_time=w_start + i * per_word,
                                end_time=w_start + (i + 1) * per_word,
                                confidence=w_conf,
                                speaker_id=None,
                            ))
            else:
                # Create words from segment text
                segment_words = segment.text.split()
                word_duration = (segment.end - segment.start) / max(1, len(segment_words))
                
                for i, word_text in enumerate(segment_words):
                    word = Word(
                        text=word_text,
                        start_time=segment.start + (i * word_duration),
                        end_time=segment.start + ((i + 1) * word_duration),
                        confidence=segment.confidence,
                        speaker_id=None
                    )
                    words.append(word)
        
        if words:
            store.add_words(words)
        
        return store

    def _transfer_speaker_labels(
        self,
        realtime_store: TranscriptStore,
        postproc_store: TranscriptStore,
    ) -> None:
        """Transfer speaker labels from realtime to post-processed words.

        Uses nearest-midpoint matching: for each post-processed word, find
        the realtime word whose midpoint is closest.  Then fills any still-
        untagged words with the dominant speaker from the realtime transcript.

        Args:
            realtime_store: The realtime transcript with speaker labels.
            postproc_store: The post-processed transcript (no speaker info).
        """
        rt_words = realtime_store.get_all_words()
        pp_words = postproc_store.get_all_words()
        if not rt_words or not pp_words:
            return

        # Only transfer if realtime words actually have labels
        labeled = [w for w in rt_words if w.speaker_id is not None]
        if not labeled:
            return

        # Pre-compute midpoints for labeled realtime words
        rt_mids = [(w, (w.start_time + w.end_time) / 2) for w in labeled]

        transferred = 0
        for pp_word in pp_words:
            pp_mid = (pp_word.start_time + pp_word.end_time) / 2
            # Find nearest realtime word by midpoint distance
            nearest = min(rt_mids, key=lambda x: abs(x[1] - pp_mid))
            pp_word.speaker_id = nearest[0].speaker_id
            transferred += 1

        logger.info(
            "Transferred speaker labels: %d/%d post-processed words",
            transferred, len(pp_words),
        )

    @staticmethod
    def _read_speaker_matches(transcript_path: Path) -> Optional[dict]:
        """Read speaker_matches from an existing transcript file.

        Args:
            transcript_path: Path to the .md transcript.

        Returns:
            The speaker_matches dict, or None if not found.
        """
        try:
            content = transcript_path.read_text(encoding="utf-8")
        except OSError:
            return None

        data = transcript_footer.parse(content)
        if data is None:
            return None
        return data.get("speaker_matches")

    def _save_post_processed_transcript(
        self, job: PostProcessJob, store: TranscriptStore,
        speaker_matches: Optional[dict] = None,
    ) -> Path:
        """Save post-processed transcript by overwriting the original .md in-place.

        Derives the original transcript path from the audio file stem:
        ``{audio_file.stem}.md`` in the same output directory.

        Preserves the original ``recording_start_time`` from the existing
        transcript so the history list shows the real recording date instead
        of the post-processing completion time.

        Args:
            job: The job being processed
            store: The transcript store to save
            speaker_matches: Optional speaker match metadata from the
                realtime transcript's diarization result.

        Returns:
            Path to the (over)written transcript file
        """
        base_name = job.audio_file.stem
        transcript_path = job.output_dir / f"{base_name}.md"

        # Preserve original recording_start_time from the existing transcript
        if transcript_path.exists():
            logger.debug(
                "Overwriting existing transcript in-place: %s", transcript_path
            )
            self._preserve_recording_time(transcript_path, store)
        else:
            logger.debug(
                "Creating new transcript (no prior .md found): %s", transcript_path
            )

        store.save_to_file(transcript_path, speaker_matches=speaker_matches)

        logger.info(
            "Saved post-processed transcript to %s", transcript_path
        )
        return transcript_path

    @staticmethod
    def _preserve_recording_time(
        original_path: Path, store: TranscriptStore
    ) -> None:
        """Read recording_start_time from an existing transcript and set it
        on the new store so the original recording date survives overwrites.
        """
        from datetime import datetime as dt

        try:
            content = original_path.read_text(encoding="utf-8")
            data = transcript_footer.parse(content)
            if data is None:
                return
            original_time = data.get("recording_start_time")
            if original_time:
                store.set_recording_start_time(
                    dt.fromisoformat(original_time)
                )
        except (OSError, ValueError):
            pass
    
    def _update_progress(self, job: PostProcessJob, progress: int) -> None:
        """Update job progress and notify.
        
        Args:
            job: The job to update
            progress: Progress percentage (0-100)
        """
        job.progress = progress
        if self._on_progress:
            self._on_progress(job.job_id, progress)
    
    def clear_completed_jobs(self) -> None:
        """Clear completed, failed, and cancelled jobs from memory."""
        with self._jobs_lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.status in (
                    PostProcessStatus.COMPLETED,
                    PostProcessStatus.FAILED,
                    PostProcessStatus.CANCELLED,
                )
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
    
    # ------------------------------------------------------------------
    # Queue persistence
    # ------------------------------------------------------------------
    
    def _persist_job(self, job: PostProcessJob) -> None:
        """Append a job to the persistent queue file.
        
        Serialised by _queue_file_lock so concurrent persist/unpersist
        cannot lose entries via read-modify-write races.
        """
        try:
            with self._queue_file_lock:
                entries = self._read_queue_file()
                
                # Append new entry (avoid duplicates)
                if not any(e.get("job_id") == job.job_id for e in entries):
                    entries.append({
                        "job_id": job.job_id,
                        "audio_file": str(job.audio_file),
                        "output_dir": str(job.output_dir),
                        "model_size": job.model_size,
                        "scheduled_at": time.time(),
                    })
                
                self._write_queue_file(entries)
        except Exception as exc:
            logger.warning("Failed to persist job %s: %s", job.job_id, exc)
    
    def _unpersist_job(self, job_id: str) -> None:
        """Remove a completed/failed/cancelled job from the queue file.
        
        Serialised by _queue_file_lock so concurrent persist/unpersist
        cannot lose entries via read-modify-write races.
        """
        try:
            with self._queue_file_lock:
                entries = self._read_queue_file()
                filtered = [e for e in entries if e.get("job_id") != job_id]
                self._write_queue_file(filtered)
        except Exception as exc:
            logger.warning("Failed to unpersist job %s: %s", job_id, exc)
    
    def _read_queue_file(self) -> List[dict]:
        """Read the queue file, returning a list of job entries."""
        if not self._queue_file.exists():
            return []
        try:
            data = json.loads(self._queue_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []
    
    def _write_queue_file(self, entries: List[dict]) -> None:
        """Write entries to the queue file atomically."""
        self._queue_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._queue_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        tmp.replace(self._queue_file)
    
    def _recover_pending_jobs(self) -> None:
        """Re-queue any jobs found in the persistent queue file.
        
        Called on startup. Jobs whose audio files no longer exist are
        silently dropped.  The realtime transcript is not recoverable
        (it was in-memory), so post-processing creates a fresh one from
        the stronger-model transcription alone.
        """
        entries = self._read_queue_file()
        if not entries:
            return
        
        recovered = 0
        dropped = 0
        for entry in entries:
            audio_path = Path(entry.get("audio_file", ""))
            if not audio_path.exists():
                logger.info(
                    "Recovery: dropping job %s (audio gone: %s)",
                    entry.get("job_id"), audio_path,
                )
                dropped += 1
                continue
            
            output_dir = Path(entry.get("output_dir", audio_path.parent))
            model_size = entry.get("model_size", "base")
            
            # Create a minimal job without realtime_transcript (will be None)
            job = PostProcessJob(
                job_id=entry.get("job_id", ""),
                audio_file=audio_path,
                realtime_transcript=None,
                output_dir=output_dir,
                model_size=model_size,
            )
            
            with self._jobs_lock:
                self._jobs[job.job_id] = job
            self._job_queue.put(job)
            recovered += 1
            logger.info(
                "Recovery: re-queued job %s for %s (model=%s)",
                job.job_id, audio_path.name, model_size,
            )
        
        if recovered or dropped:
            # Re-write the file with only the recovered entries so that
            # if the app crashes before they complete, they're still on disk.
            # _unpersist_job will remove each entry as it finishes.
            with self._queue_file_lock:
                persisted = [
                    e for e in entries
                    if Path(e.get("audio_file", "")).exists()
                ]
                self._write_queue_file(persisted)
            logger.info(
                "Recovery complete: %d re-queued, %d dropped", recovered, dropped,
            )

    # ------------------------------------------------------------------
    # Stalled requeue (issue #62)
    # ------------------------------------------------------------------

    def requeue_stalled_recordings(self) -> int:
        """Scan the Library and re-queue Stalled recordings.

        A Stalled recording has no Post-processing Outcome — Post-processing
        never ran, was lost, or was interrupted.  Per recording:

        * Outcome present            → nothing to do.
        * Audio gone                 → write a Failed (audio-missing) Outcome
          so the row surfaces as Failed instead of lingering as a zombie.
        * otherwise, when Post-processing and Speakers are enabled (current
          settings, read live) and no live or persisted job targets it →
          schedule a fresh post-processing job.

        Runs at startup (after pending-job recovery) and after each job
        reaches a terminal state; each re-queued job re-triggers the scan on
        its own completion, draining until a scan enqueues nothing.

        Returns:
            The number of recordings re-queued.
        """
        with self._requeue_lock:
            from meetandread.transcription.transcript_scanner import scan_recordings

            try:
                recordings = scan_recordings()
            except Exception as exc:
                logger.warning("Stalled scan could not read Library: %s", exc)
                return 0

            requeued = 0
            for meta in recordings:
                if meta.post_process_outcome is not None:
                    continue
                stem = meta.path.stem
                if not meta.wav_exists:
                    # No Outcome and the Audio is gone: Post-processing can
                    # never run. Record the Failed Outcome durably.
                    self._write_outcome(
                        meta.path,
                        PostProcessOutcome(
                            status=transcript_footer.STATUS_FAILED,
                            attempted_at=datetime.now().isoformat(),
                            stage=transcript_footer.STAGE_AUDIO_MISSING,
                            error="Audio file missing",
                        ),
                    )
                    continue
                if not self._should_requeue_recording(meta):
                    continue
                from meetandread.audio.storage.paths import get_recordings_dir

                wav_path = get_recordings_dir() / f"{stem}.wav"
                self.schedule_post_process(
                    audio_file=wav_path,
                    realtime_transcript=None,
                    output_dir=meta.path.parent,
                )
                requeued += 1
                logger.info(
                    "Stalled requeue: scheduled post-processing for %s", stem,
                )
            if requeued:
                logger.info("Stalled requeue scan: %d recording(s) re-queued", requeued)
            return requeued

    def _should_requeue_recording(self, meta) -> bool:
        """Return True when a Stalled RecordingMeta should be re-queued.

        Predicate (issue #62): no Outcome ∧ Audio still exists ∧
        Post-processing enabled ∧ Speakers enabled ∧ not already queued or
        in-flight (live in-memory job or persisted queue entry).
        """
        if meta.post_process_outcome is not None:
            return False
        if not meta.wav_exists:
            return False
        stem = meta.path.stem
        if stem in self._no_outcome_stems:
            return False
        settings = self._settings
        if not getattr(settings, "transcription", None):
            return False
        if not (
            settings.transcription.enable_postprocessing
            and settings.speaker.enabled
        ):
            return False
        return not self._has_queued_job_for_stem(stem)

    def _has_queued_job_for_stem(self, stem: str) -> bool:
        """True when a live (in-memory) or persisted job targets *stem*."""
        with self._jobs_lock:
            for job in self._jobs.values():
                if (
                    job.status in self._PENDING_STATUSES
                    and job.audio_file.stem == stem
                ):
                    return True
        for entry in self._read_queue_file():
            if Path(entry.get("audio_file", "")).stem == stem:
                return True
        return False
