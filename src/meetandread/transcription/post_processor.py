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
        on_complete: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """Initialize the post-processing queue.
        
        Args:
            settings: Application settings containing model configuration
            on_progress: Callback(job_id, progress_pct) for progress updates
            on_complete: Callback(job_id, result) when job completes
        """
        self._settings = settings
        self._on_progress = on_progress
        self._on_complete = on_complete
        
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
        print("DEBUG: PostProcessingQueue worker started")
    
    def stop(self) -> None:
        """Stop the background worker thread."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        
        print("DEBUG: PostProcessingQueue worker stopped")
    
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
        print(f"DEBUG: Scheduled post-processing job {job.job_id} with model {model_size}")
        
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
    
    def _worker_loop(self) -> None:
        """Background worker thread that processes jobs."""
        print("DEBUG: Post-processing worker loop started")
        
        while self._is_running and not self._stop_event.is_set():
            try:
                # Get job with timeout to allow checking stop_event
                job = self._job_queue.get(timeout=0.5)
                self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in post-processing worker: {e}")
    
    def _process_job(self, job: PostProcessJob) -> None:
        """Process a single post-processing job.
        
        Args:
            job: The job to process
        """
        print(f"DEBUG: Processing job {job.job_id} with model {job.model_size}")
        
        try:
            # Update status
            job.status = PostProcessStatus.RUNNING
            self._update_progress(job, 10)
            
            # Load or get engine
            engine = self._get_or_create_engine(job.model_size)
            self._update_progress(job, 20)
            
            # Read audio file
            audio_data = self._load_audio_file(job.audio_file)
            self._update_progress(job, 30)
            
            # Transcribe with stronger model
            print(f"DEBUG: Transcribing {len(audio_data)} samples with {job.model_size} model...")
            segments = engine.transcribe_chunk(audio_data)
            self._update_progress(job, 80)
            
            # Create post-processed transcript
            audio_duration = len(audio_data) / 16000.0  # 16kHz sample rate
            enhanced_store = self._create_post_processed_transcript(segments, audio_duration)
            self._update_progress(job, 90)
            
            # Save post-processed transcript (overwrites original .md)
            transcript_path = self._save_post_processed_transcript(job, enhanced_store)
            self._update_progress(job, 100)
            
            # Mark complete
            job.status = PostProcessStatus.COMPLETED
            job.result = {
                "transcript_path": str(transcript_path),
                "word_count": enhanced_store.get_word_count(),
                "realtime_word_count": job.realtime_transcript.get_word_count(),
                "model_used": job.model_size
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
            print(f"ERROR: Job {job.job_id} failed: {e}")
    
    def _get_or_create_engine(self, model_size: str) -> WhisperTranscriptionEngine:
        """Get cached engine or create new one.
        
        Args:
            model_size: The model size to use
        
        Returns:
            WhisperTranscriptionEngine instance
        """
        with self._engines_lock:
            if model_size not in self._engines:
                print(f"DEBUG: Creating new engine for model {model_size}")
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
    
    def _save_post_processed_transcript(self, job: PostProcessJob, store: TranscriptStore) -> Path:
        """Save post-processed transcript by overwriting the original .md in-place.

        Derives the original transcript path from the audio file stem:
        ``{audio_file.stem}.md`` in the same output directory.

        Args:
            job: The job being processed
            store: The transcript store to save

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

        store.save_to_file(transcript_path)

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
        """Clear completed and failed jobs from memory."""
        with self._jobs_lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.status in (PostProcessStatus.COMPLETED, PostProcessStatus.FAILED)
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
