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
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
import numpy as np
import wave

logger = logging.getLogger(__name__)

from meetandread.config.models import TranscriptionSettings, AppSettings
from meetandread.transcription.engine import WhisperTranscriptionEngine, TranscriptionSegment
from meetandread.transcription.transcript_store import TranscriptStore, Word


class PostProcessStatus(Enum):
    """Status of a post-processing job."""
    PENDING = auto()      # Queued but not started
    RUNNING = auto()      # Currently processing
    COMPLETED = auto()    # Successfully completed
    FAILED = auto()       # Failed with error
    CANCELLED = auto()    # Cancelled by caller


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
    """
    job_id: str
    audio_file: Path
    realtime_transcript: TranscriptStore
    output_dir: Path
    model_size: str
    status: PostProcessStatus = PostProcessStatus.PENDING
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    cancel_reason: Optional[str] = None
    diarization_error: Optional[str] = None


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
        """
        self._settings = settings
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._is_recording_callback = is_recording_callback
        self._diarize_callback = diarize_callback
        self._apply_speaker_labels_callback = apply_speaker_labels_callback
        
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
        realtime_transcript: TranscriptStore,
        output_dir: Path,
        model_size: Optional[str] = None
    ) -> PostProcessJob:
        """Schedule a post-processing job.
        
        Args:
            audio_file: Path to the recorded audio file
            realtime_transcript: The real-time transcript for comparison
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
    
    def _process_job(self, job: PostProcessJob) -> None:
        """Process a single post-processing job.
        
        Args:
            job: The job to process
        """
        logger.info("Processing job %s with model %s", job.job_id, job.model_size)
        
        try:
            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled before start", job.job_id)
                return
            
            # Update status
            job.status = PostProcessStatus.RUNNING
            self._update_progress(job, 10)
            
            # Load or get engine
            engine = self._get_or_create_engine(job.model_size)
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
            
            # Read audio file
            audio_data = self._load_audio_file(job.audio_file)
            self._update_progress(job, 35)
            
            # Transcribe with stronger model
            logger.info(
                "Transcribing %d samples with %s model for job %s...",
                len(audio_data), job.model_size, job.job_id,
            )
            segments = engine.transcribe_chunk(audio_data)
            self._update_progress(job, 80)
            
            # ---- Checkpoint: not cancelled ----
            if job.cancel_requested:
                job.status = PostProcessStatus.CANCELLED
                logger.info("Job %s cancelled after transcription", job.job_id)
                return
            
            # Create post-processed transcript
            audio_duration = len(audio_data) / 16000.0  # 16kHz sample rate
            enhanced_store = self._create_post_processed_transcript(segments, audio_duration)
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
                        job.job_id, exc,
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
            # Carry forward speaker_matches from the realtime transcript's
            # diarization result so identity metadata survives the overwrite.
            speaker_matches = None
            if job.realtime_transcript:
                try:
                    base_name = job.audio_file.stem
                    original_path = job.output_dir / f"{base_name}.md"
                    speaker_matches = self._read_speaker_matches(original_path)
                except Exception:
                    pass

            transcript_path = self._save_post_processed_transcript(
                job, enhanced_store, speaker_matches=speaker_matches,
            )
            self._update_progress(job, 100)
            
            # Mark complete
            job.status = PostProcessStatus.COMPLETED
            job.result = {
                "transcript_path": str(transcript_path),
                "word_count": enhanced_store.get_word_count(),
                "realtime_word_count": job.realtime_transcript.get_word_count(),
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
            logger.error("Job %s failed: %s", job.job_id, e)
    
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
        
        Args:
            audio_file: Path to audio file
        
        Returns:
            Audio samples as float32 numpy array
        """
        import wave
        import struct
        
        with wave.open(str(audio_file), 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all frames
            raw_data = wf.readframes(n_frames)
            
            # Convert to numpy
            if sample_width == 2:  # 16-bit
                fmt = f"{n_frames * n_channels}h"
                samples = struct.unpack(fmt, raw_data)
                audio = np.array(samples, dtype=np.float32) / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to mono if stereo
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling (for production, use proper resampling library)
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            
            return audio
    
    def _create_post_processed_transcript(
        self, 
        segments: List[TranscriptionSegment],
        audio_duration: float = 0.0,
    ) -> TranscriptStore:
        """Create TranscriptStore from transcription segments.
        
        Splits multi-word Whisper tokens into individual words and rescales
        timestamps when Whisper's reported duration is significantly shorter
        than the actual audio duration.

        Args:
            segments: Transcription segments from Whisper
            audio_duration: Actual audio duration in seconds (0 = unknown).
                Used to rescale Whisper timestamps when they under-report.
        
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
        
        # Rescale timestamps if Whisper's reported duration is significantly
        # shorter than the actual audio duration. This happens when Whisper's
        # timestamps don't cover the full audio (common with smaller models or
        # compressed speech). We proportionally stretch the timestamps to fill
        # the actual audio duration.
        if words and audio_duration > 0:
            whisper_end = max(w.end_time for w in words)
            # Only rescale if Whisper reports less than 80% of the actual duration
            if whisper_end > 0 and whisper_end < audio_duration * 0.8:
                scale = audio_duration / whisper_end
                for w in words:
                    w.start_time = w.start_time * scale
                    w.end_time = w.end_time * scale
        
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
        import json as _json

        try:
            content = transcript_path.read_text(encoding="utf-8")
            marker = "\n---\n\n<!-- METADATA: "
            idx = content.find(marker)
            if idx < 0:
                return None
            data = _json.loads(content[idx + len(marker) :].rstrip(" -->\n"))
            return data.get("speaker_matches")
        except Exception:
            return None
    
    def _save_post_processed_transcript(
        self, job: PostProcessJob, store: TranscriptStore,
        speaker_matches: Optional[dict] = None,
    ) -> Path:
        """Save post-processed transcript by overwriting the original .md in-place.

        Derives the original transcript path from the audio file stem:
        ``{audio_file.stem}.md`` in the same output directory.

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

        if transcript_path.exists():
            logger.debug(
                "Overwriting existing transcript in-place: %s", transcript_path
            )
        else:
            logger.debug(
                "Creating new transcript (no prior .md found): %s", transcript_path
            )

        store.save_to_file(transcript_path, speaker_matches=speaker_matches)

        logger.info(
            "Saved post-processed transcript to %s", transcript_path
        )
        return transcript_path
    
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
