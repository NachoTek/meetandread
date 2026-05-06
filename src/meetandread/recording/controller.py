"""Recording controller - UI-friendly wrapper around AudioSession.

Provides non-blocking recording control with proper error handling
and state management for UI integration. Includes hybrid transcription:
- Real-time: tiny model for immediate display using accumulating processor
- Post-process: stronger model after recording stops
"""

import logging
import threading
import time as _time


def _ts(): return _time.strftime("%H:%M:%S")
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Set, Callable, List

from meetandread.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionState,
    SessionError,
    NoSourcesError,
)
from meetandread.audio.capture import AudioSourceError
from meetandread.transcription.accumulating_processor import AccumulatingTranscriptionProcessor, SegmentResult
from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.transcription.post_processor import PostProcessingQueue, PostProcessStatus
from meetandread.config.manager import ConfigManager

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """Controller states for UI state management."""
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class ControllerError:
    """Error information for UI display."""
    message: str
    is_recoverable: bool = True


class RecordingController:
    """UI-friendly controller for recording operations.
    
    Wraps AudioSession with:
    - Non-blocking stop/finalize (runs on worker thread)
    - Clear error state for UI display
    - Simple source selection API
    - Callback support for state changes
    - Real-time transcription using accumulating processor
    
    HYBRID TRANSCRIPTION ARCHITECTURE:
    - Real-time: AccumulatingTranscriptionProcessor with tiny model
      * 60s window for context
      * Updates every 2 seconds
      * 3s silence detection for phrase breaks
    - Post-process: Stronger model (base/small) after recording stops
    
    Example:
        controller = RecordingController()
        controller.on_state_change = lambda state: print(f"State: {state}")
        controller.on_error = lambda err: print(f"Error: {err.message}")
        controller.on_phrase_result = lambda result: print(f"Phrase: {result.text}")
        
        # Start recording
        error = controller.start({'mic', 'system'})
        if error:
            print(f"Failed to start: {error.message}")
        
        # Stop recording (non-blocking)
        controller.stop()
    """
    
    def __init__(self, enable_transcription: bool = True):
        """Initialize the recording controller.
        
        Args:
            enable_transcription: Whether to enable real-time transcription
        """
        self._session = AudioSession()
        self._state = ControllerState.IDLE
        self._error: Optional[ControllerError] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._last_wav_path: Optional[Path] = None
        self._last_transcript_path: Optional[Path] = None
        
        # HYBRID TRANSCRIPTION
        self.enable_transcription = enable_transcription
        self._transcription_processor: Optional[AccumulatingTranscriptionProcessor] = None
        self._transcript_store: Optional[TranscriptStore] = None
        self._post_processor: Optional[PostProcessingQueue] = None
        self._post_process_job_id: Optional[str] = None
        self._config_manager = ConfigManager()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[ControllerState], None]] = None
        self.on_error: Optional[Callable[[ControllerError], None]] = None
        self.on_recording_complete: Optional[Callable[[Path, Optional[Path]], None]] = None
        self.on_phrase_result: Optional[Callable[[SegmentResult], None]] = None  # For accumulating processor results
        self.on_post_process_complete: Optional[Callable[[str, Path], None]] = None  # job_id, transcript_path
        
        # Audio feed tracking
        self._audio_chunks_fed = 0
        
        # Speaker diarization result (kept for pin-to-name UX)
        self._last_diarization_result: Optional[object] = None  # DiarizationResult
        
        # Auto-WER from last post-processing (None until computed)
        self._last_wer: Optional[float] = None

        # --- Live speaker matching state ---
        self._live_audio_buffer = bytearray()  # raw int16 PCM bytes
        self._live_max_buffer_bytes = 12 * 16000 * 2  # 12s at 16kHz int16 = 384000
        self._live_extractor = None  # lazily-created SpeakerEmbeddingExtractor
        self._live_extractor_available: Optional[bool] = None  # None=unchecked, True/False
        self._live_store_available: Optional[bool] = None
        self._live_match_attempts = 0
        self._live_match_hits = 0
        self._live_match_fallbacks = 0
        self._live_last_status: str = "disabled"  # sanitized status string
        self._live_last_error_class: Optional[str] = None
        self._live_last_error_message: Optional[str] = None
        self._live_last_attempt_ts: Optional[float] = None
        self._live_min_audio_bytes = 8 * 16000 * 2  # 8s at 16kHz int16 = 256000
        self._live_attempt_interval = 2.0  # seconds between match attempts
        self._live_extractor_lock = threading.Lock()  # serialize extractor use
        
    
    def _set_state(self, state: ControllerState) -> None:
        """Update state and notify listeners."""
        self._state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception as e:
                print(f"[{_ts()}] ERROR: State change callback failed: {e}")
    
    def _set_error(self, message: str, is_recoverable: bool = True) -> ControllerError:
        """Set error state and notify listeners."""
        self._error = ControllerError(message, is_recoverable)
        self._set_state(ControllerState.ERROR)
        if self.on_error:
            try:
                self.on_error(self._error)
            except Exception as e:
                print(f"[{_ts()}] ERROR: Error callback failed: {e}")
        return self._error
    
    def clear_error(self) -> None:
        """Clear error state and return to idle."""
        self._error = None
        if self._state == ControllerState.ERROR:
            self._set_state(ControllerState.IDLE)
    
    def get_state(self) -> ControllerState:
        """Get current controller state."""
        # Sync with underlying session state if needed
        session_state = self._session.get_state()
        if session_state == SessionState.RECORDING and self._state != ControllerState.RECORDING:
            self._set_state(ControllerState.RECORDING)
        return self._state
    
    def get_error(self) -> Optional[ControllerError]:
        """Get current error if any."""
        return self._error
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._state == ControllerState.RECORDING
    
    def is_busy(self) -> bool:
        """Check if controller is busy (starting, stopping, etc.)."""
        return self._state in (ControllerState.STARTING, ControllerState.STOPPING)
    
    def start(
        self,
        selected_sources: Set[str],
        *,
        fake_path: Optional[str] = None,
        fake_denoise: bool = False,
        fake_loop: bool = False,
    ) -> Optional[ControllerError]:
        """Start recording from selected sources.
        
        Args:
            selected_sources: Set of source types ('mic', 'system', 'fake')
            fake_path: Path to WAV file for fake source (required when
                'fake' is in selected_sources).
            fake_denoise: Whether to apply denoising to the fake source.
                Only meaningful when 'fake' is in selected_sources.
            fake_loop: Whether to loop the fake source WAV file.
        
        Returns:
            ControllerError if start failed, None on success
        """
        # Validate state
        if self._state in (ControllerState.RECORDING, ControllerState.STARTING):
            return self._set_error("Already recording", is_recoverable=True)
        
        if self._state == ControllerState.STOPPING:
            return self._set_error("Cannot start while stopping", is_recoverable=True)
        
        # Validate sources
        if not selected_sources:
            return self._set_error(
                "No audio source selected. Enable microphone or system audio.",
                is_recoverable=True
            )
        
        # Clear any previous error
        self.clear_error()
        self._set_state(ControllerState.STARTING)
        print("DEBUG: Starting recording...")
        
        try:
            # Initialize transcription if enabled
            if self.enable_transcription:
                print("DEBUG: Initializing transcription...")
                error = self._init_transcription()
                if error:
                    # Log warning but continue with recording
                    print(f"Warning: Transcription not available: {error.message}")
            
            # Build source configs
            source_configs = self._build_source_configs(
                selected_sources,
                fake_path=fake_path,
                fake_denoise=fake_denoise,
                fake_loop=fake_loop,
            )
            
            if not source_configs:
                return self._set_error(
                    "No valid audio sources configured",
                    is_recoverable=True
                )
            
            # Read denoising settings from persisted config with safe fallbacks
            denoise_enabled = False  # safe default (disabled due to spectral gate artifacts)
            denoise_provider = "spectral_gate"  # safe default
            denoise_budget_ms = 200.0  # safe default
            try:
                settings = self._config_manager.get_settings()
                ts = settings.transcription

                # Validate enabled — must be actual bool
                raw_enabled = ts.microphone_denoising_enabled
                denoise_enabled = raw_enabled if isinstance(raw_enabled, bool) else False

                # Validate provider — must be a non-empty string in the allowed set
                raw_provider = ts.microphone_denoising_provider
                from meetandread.audio.denoising import VALID_PROVIDER_NAMES
                if isinstance(raw_provider, str) and raw_provider in VALID_PROVIDER_NAMES:
                    denoise_provider = raw_provider
                else:
                    if raw_provider:
                        logger.warning(
                            "Invalid denoising provider '%s' in config, "
                            "falling back to '%s'",
                            raw_provider, denoise_provider,
                        )

                # Validate budget — must be a positive number
                raw_budget = ts.microphone_denoising_latency_budget_ms
                if isinstance(raw_budget, (int, float)) and raw_budget > 0:
                    denoise_budget_ms = float(raw_budget)
                else:
                    if raw_budget is not None and raw_budget != 200:
                        logger.warning(
                            "Invalid denoising latency budget %r in config, "
                            "falling back to %.0fms",
                            raw_budget, denoise_budget_ms,
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to read denoising config, using defaults: %s", exc
                )

            # Tag mic sources with denoise=True when denoising is enabled
            if denoise_enabled:
                for sc in source_configs:
                    if sc.type == "mic" and sc.denoise is None:
                        sc.denoise = True

            # If fake_denoise was requested, force denoising on at the session
            # level regardless of persisted config (test harness override)
            if fake_denoise:
                denoise_enabled = True

            # Create and start session
            config = SessionConfig(
                sources=source_configs,
                enable_microphone_denoising=denoise_enabled,
                denoising_provider_name=denoise_provider if denoise_enabled else None,
                denoising_latency_budget_ms=denoise_budget_ms,
            )
            logger.info(
                "Denoising config: enabled=%s provider=%s budget=%.0fms",
                denoise_enabled, denoise_provider, denoise_budget_ms,
            )
            # Wire audio callback to feed transcription processor
            if self.enable_transcription and self._transcription_processor:
                config.on_audio_frame = self.feed_audio_for_transcription
                print("DEBUG: Audio callback wired to transcription processor")
            
            self._session = AudioSession()
            self._session.start(config)
            print("DEBUG: Audio session started")
            
            # Start transcription if available
            if self._transcription_processor:
                print("DEBUG: Starting transcription processor...")
                print(f"[{_ts()}] DEBUG: Transcription processor exists: {self._transcription_processor is not None}")
                print(f"[{_ts()}] DEBUG: Processor on_result callback: {self._transcription_processor.on_result is not None}")
                self._transcription_processor.start()
                print("DEBUG: Transcription processor started")
            
            self._audio_chunks_fed = 0
            self._reset_live_speaker_state()
            self._set_state(ControllerState.RECORDING)
            print("DEBUG: Recording started successfully")
            return None
            
        except NoSourcesError as e:
            return self._set_error(f"No sources: {e}", is_recoverable=True)
        except AudioSourceError as e:
            return self._set_error(f"Audio device error: {e}", is_recoverable=True)
        except SessionError as e:
            return self._set_error(f"Session error: {e}", is_recoverable=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._set_error(f"Unexpected error: {e}", is_recoverable=False)
    
    def stop(self) -> Optional[ControllerError]:
        """Stop recording and finalize to WAV.
        
        This is non-blocking - finalization happens on a worker thread.
        
        Returns:
            ControllerError if stop cannot be initiated, None if stop started
        """
        if self._state != ControllerState.RECORDING:
            return self._set_error("Not currently recording", is_recoverable=True)
        
        print("DEBUG: Stopping recording...")
        self._set_state(ControllerState.STOPPING)
        
        # Run stop/finalize in worker thread to avoid blocking UI
        self._worker_thread = threading.Thread(
            target=self._stop_worker,
            daemon=True,
            name="RecordingStopWorker"
        )
        self._worker_thread.start()
        print("DEBUG: Stop worker thread started")
        
        return None
    
    def _stop_worker(self) -> None:
        """Worker thread that handles stop and finalization."""
        try:
            # Stop transcription first to flush results
            if self._transcription_processor:
                print("DEBUG: Stopping transcription processor...")
                self._transcription_processor.stop()
                print("DEBUG: Transcription processor stopped")
                self._transcription_processor = None
            
            # Stop audio session
            print("DEBUG: Stopping audio session...")
            wav_path = self._session.stop()
            self._last_wav_path = wav_path
            print(f"[{_ts()}] DEBUG: Audio saved to: {wav_path}")
            
            # --- Speaker diarization (post-processing step) ---
            if self._transcript_store and self._last_wav_path:
                self._run_diarization(self._last_wav_path)
            
            # Save transcript if available
            transcript_path = None
            if self._transcript_store and self._last_wav_path:
                print(f"[{_ts()}] DEBUG: Saving transcript ({self._transcript_store.get_word_count()} words)...")
                transcript_path = self._save_transcript()
                self._last_transcript_path = transcript_path
                print(f"[{_ts()}] DEBUG: Transcript saved to: {transcript_path}")
            
            # Schedule post-processing with stronger model
            if self._post_processor and self._last_wav_path and self._transcript_store:
                from meetandread.audio.storage.paths import get_transcripts_dir
                print("DEBUG: Scheduling post-processing job...")
                job = self._post_processor.schedule_post_process(
                    audio_file=self._last_wav_path,
                    realtime_transcript=self._transcript_store,
                    output_dir=get_transcripts_dir()
                )
                self._post_process_job_id = job.job_id
                print(f"[{_ts()}] DEBUG: Post-processing job scheduled: {job.job_id}")
            
            self._set_state(ControllerState.IDLE)
            print("DEBUG: Recording stopped, state set to IDLE")
            
            # Notify completion
            if self.on_recording_complete:
                try:
                    self.on_recording_complete(wav_path, transcript_path)
                except Exception as e:
                    print(f"[{_ts()}] ERROR: Recording complete callback failed: {e}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set_error(f"Failed to finalize recording: {e}", is_recoverable=False)
    
    def _init_transcription(self) -> Optional[ControllerError]:
        """Initialize transcription components.
        
        HYBRID TRANSCRIPTION:
        - Uses AccumulatingTranscriptionProcessor for real-time
          * 60s window for context
          * Updates every 2 seconds
          * 3s silence detection for phrase breaks
        - Post-processing uses stronger model (scheduled on stop)
        
        Returns:
            ControllerError if initialization failed, None on success
        """
        try:
            # Get transcription settings from config
            settings = self._config_manager.get_settings()
            
            # HYBRID: Always use tiny for real-time (fastest)
            # Post-processing will use stronger model
            realtime_model = settings.transcription.realtime_model_size
            print(f"[{_ts()}] DEBUG: Initializing accumulating transcription with {realtime_model} model")
            
            # Create transcript store
            self._transcript_store = TranscriptStore()
            self._transcript_store.start_recording()
            print("DEBUG: Transcript store initialized")
            
            # Create accumulating transcription processor
            # Configuration optimized for meetings:
            # - 60s window for good context
            # - 2s update frequency for responsiveness
            # - 3s silence timeout for natural turn-taking
            self._transcription_processor = AccumulatingTranscriptionProcessor(
                model_size=realtime_model,
                window_size=60.0,  # 60 seconds of context
                update_frequency=2.0,  # Update every 2 seconds
                silence_timeout=3.0  # 3 seconds of silence = phrase complete
            )
            
            # Load model (tiny takes 1-2 seconds)
            print(f"[{_ts()}] DEBUG: Loading {realtime_model} model for real-time transcription...")
            self._transcription_processor.load_model(
                progress_callback=lambda p: print(f"Loading {realtime_model} model: {p}%")
            )
            print(f"[{_ts()}] DEBUG: {realtime_model} model loaded successfully")
            
            # Wire up the phrase result callback
            self._transcription_processor.on_result = self._on_phrase_result
            print("DEBUG: Transcription result callback wired")
            
            # Initialize post-processing queue (for after recording stops)
            if settings.transcription.enable_postprocessing:
                print("DEBUG: Initializing post-processing queue")
                self._post_processor = PostProcessingQueue(
                    settings=settings,
                    on_progress=self._on_post_process_progress,
                    on_complete=self._on_post_process_complete_callback
                )
                self._post_processor.start()
            
            return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ControllerError(
                message=f"Failed to initialize transcription: {e}",
                is_recoverable=True
            )
    
    def _on_phrase_result(self, result: SegmentResult) -> None:
        """Handle segment result from accumulating transcription processor.
        
        Args:
            result: SegmentResult with text, confidence, and completion status
        """
        print(f"DEBUG Controller: Segment: '{result.text}' [conf: {result.confidence}%, final: {result.is_final}, idx: {result.segment_index}]")
        
        # Attempt live speaker matching (conservative; attaches name only
        # for high-confidence known-speaker matches)
        try:
            name = self._try_live_speaker_match()
            if name is not None:
                result.speaker_id = name
        except Exception:
            pass  # Never block phrase result delivery
        
        # Convert SegmentResult to Word objects for storage
        if self._transcript_store:
            words = self._segment_to_words(result)
            if words:
                self._transcript_store.add_words(words)
                print(f"DEBUG Controller: Added {len(words)} words to transcript store (total words: {self._transcript_store.get_word_count()})")
        
        # Notify UI callback
        if self.on_phrase_result:
            try:
                self.on_phrase_result(result)
            except Exception as e:
                print(f"[{_ts()}] ERROR: Segment result callback failed: {e}")
    
    def _segment_to_words(self, result: SegmentResult) -> List[Word]:
        """Convert a SegmentResult to Word objects.
        
        Args:
            result: SegmentResult from accumulating processor
        
        Returns:
            List of Word objects for storage
        """
        words = []
        text_parts = result.text.split()
        
        if not text_parts:
            return words
        
        # Distribute timing across words
        duration = result.end_time - result.start_time
        word_duration = duration / len(text_parts) if text_parts else 0
        
        for i, word_text in enumerate(text_parts):
            word_start = result.start_time + (i * word_duration)
            word_end = word_start + word_duration
            
            word = Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                confidence=result.confidence,
                speaker_id=None
            )
            words.append(word)
        
        return words
    
    def _on_post_process_progress(self, job_id: str, progress: int) -> None:
        """Handle post-processing progress updates.
        
        Args:
            job_id: The job identifier
            progress: Progress percentage (0-100)
        """
        print(f"[{_ts()}] DEBUG: Post-processing job {job_id}: {progress}%")
    
    def _on_post_process_complete_callback(self, job_id: str, result: dict) -> None:
        """Handle post-processing completion.
        
        Args:
            job_id: The job identifier
            result: Result dictionary with transcript_path, etc.
        """
        print(f"[{_ts()}] DEBUG: Post-processing job {job_id} completed!")
        print(f"[{_ts()}] DEBUG: Post-processed transcript: {result.get('transcript_path')}")
        print(f"[{_ts()}] DEBUG: Real-time words: {result.get('realtime_word_count')}")
        print(f"[{_ts()}] DEBUG: Post-processed words: {result.get('word_count')}")
        
        # --- Auto-WER calculation ---
        self._compute_and_store_wer(result)

        if self.on_post_process_complete:
            transcript_path_str = result.get('transcript_path')
            if transcript_path_str and isinstance(transcript_path_str, str):
                transcript_path = Path(transcript_path_str)
                try:
                    self.on_post_process_complete(job_id, transcript_path)
                except Exception as e:
                    print(f"[{_ts()}] ERROR: Post-process complete callback failed: {e}")

    def _compute_and_store_wer(self, result: dict) -> None:
        """Compute WER between realtime and post-processed transcripts and append to file.

        Extracts realtime words from the in-memory TranscriptStore, reads the
        post-processed words from the saved .md file's metadata JSON footer,
        calculates WER via calculate_wer(), and appends a 'wer' field to the
        file's metadata JSON footer.

        Args:
            result: Post-processing result dict with 'transcript_path' key.
        """
        try:
            from meetandread.performance.wer import calculate_wer

            # Gather realtime text from the in-memory store
            realtime_text = ""
            if self._transcript_store:
                words = self._transcript_store.get_all_words()
                realtime_text = " ".join(w.text for w in words)

            # Read post-processed text from the saved .md file
            transcript_path_str = result.get('transcript_path')
            if not transcript_path_str:
                return

            transcript_path = Path(transcript_path_str)
            if not transcript_path.exists():
                logger.warning("Cannot compute WER: transcript file not found: %s", transcript_path)
                return

            content = transcript_path.read_text(encoding="utf-8")
            footer_marker = "\n---\n\n<!-- METADATA: "
            marker_idx = content.find(footer_marker)
            if marker_idx == -1:
                logger.warning("Cannot compute WER: no metadata footer in %s", transcript_path)
                return

            # Parse metadata JSON
            import json
            metadata_text = content[marker_idx + len(footer_marker):]
            if metadata_text.strip().endswith(" -->"):
                metadata_text = metadata_text.strip()[:-len(" -->")]
            data = json.loads(metadata_text)

            # Extract post-processed words
            postproc_words = data.get("words", [])
            postproc_text = " ".join(w.get("text", "") for w in postproc_words)

            if not realtime_text.strip() and not postproc_text.strip():
                logger.info("Both transcripts empty — skipping WER calculation")
                return

            wer_value = calculate_wer(realtime_text, postproc_text)
            logger.info(
                "Auto-WER for %s: %.3f (realtime: %d words, postproc: %d words)",
                transcript_path.name, wer_value,
                len(realtime_text.split()) if realtime_text else 0,
                len(postproc_words),
            )

            # Append WER to the metadata and rewrite the file
            data["wer"] = wer_value

            # Rebuild the file: markdown body + updated metadata footer
            md_body = content[:marker_idx]
            updated_json = json.dumps(data, indent=2)
            new_content = md_body + footer_marker + updated_json + " -->\n"
            transcript_path.write_text(new_content, encoding="utf-8")

            # Store WER value for UI access
            self._last_wer = wer_value

        except Exception as exc:
            logger.error("Auto-WER computation failed: %s", exc, exc_info=True)
    
    def feed_audio_for_transcription(self, audio_chunk) -> None:
        """Feed audio chunk to transcription processor.
        
        This is called from the audio capture consumer thread
        to provide audio data for transcription.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array
        """
        if self._transcription_processor and self._state == ControllerState.RECORDING:
            self._transcription_processor.feed_audio(audio_chunk)
            
            # Debug logging
            self._audio_chunks_fed += 1
            if self._audio_chunks_fed % 100 == 0:
                stats = self._transcription_processor.get_stats()
                print(f"[{_ts()}] DEBUG: Fed {self._audio_chunks_fed} audio chunks, buffer: {stats.get('buffer_duration', 0):.1f}s")

        # Buffer raw PCM for live speaker matching (only while recording)
        if self._state == ControllerState.RECORDING and audio_chunk is not None:
            try:
                import numpy as np
                chunk = audio_chunk
                if isinstance(chunk, np.ndarray):
                    # Convert float32 to int16 PCM bytes for the live buffer
                    clamped = np.clip(chunk, -1.0, 1.0)
                    pcm_int16 = (clamped * 32767).astype(np.int16)
                    self._live_audio_buffer.extend(pcm_int16.tobytes())
                    # Trim to rolling window
                    if len(self._live_audio_buffer) > self._live_max_buffer_bytes:
                        excess = len(self._live_audio_buffer) - self._live_max_buffer_bytes
                        del self._live_audio_buffer[:excess]
            except Exception:
                pass  # Non-critical; matching simply won't have audio
    
    # ------------------------------------------------------------------
    # Live speaker matching (conservative, for CC overlay)
    # ------------------------------------------------------------------

    def _ensure_live_extractor(self) -> bool:
        """Lazily initialize the speaker embedding extractor for live matching.

        Returns True if the extractor is available (or was successfully
        created), False otherwise.  Failures are sticky for the session.
        """
        if self._live_extractor_available is True:
            return True
        if self._live_extractor_available is False:
            return False

        try:
            import sherpa_onnx
            from meetandread.speaker.model_downloader import ensure_embedding_model

            emb_path = ensure_embedding_model()
            if not emb_path or not emb_path.exists():
                self._live_extractor_available = False
                self._live_last_status = "model_unavailable"
                logger.info("Live speaker matching disabled: embedding model not found")
                return False

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_path),
            )
            if not config.validate():
                self._live_extractor_available = False
                self._live_last_status = "model_unavailable"
                logger.info("Live speaker matching disabled: extractor config invalid")
                return False

            self._live_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._live_extractor_available = True
            self._live_last_status = "enabled"
            logger.info("Live speaker matching enabled")
            return True

        except (ImportError, Exception) as exc:
            self._live_extractor_available = False
            self._live_last_error_class = type(exc).__name__
            self._live_last_error_message = str(exc)[:120]
            self._live_last_status = "extractor_error"
            logger.info(
                "Live speaker matching disabled: %s", type(exc).__name__
            )
            return False

    def _try_live_speaker_match(self) -> Optional[str]:
        """Attempt to match buffered live audio against known speakers.

        Returns the matched speaker name for high-confidence matches,
        or None.  Thread-safe via the extractor lock.  Never raises —
        all failures degrade to None.
        """
        now = _time.monotonic()

        # Rate-limit: don't attempt more than once per interval
        if (self._live_last_attempt_ts is not None
                and now - self._live_last_attempt_ts < self._live_attempt_interval):
            return None

        # Check audio buffer has enough data
        if len(self._live_audio_buffer) < self._live_min_audio_bytes:
            self._live_last_status = "insufficient_audio"
            return None

        # Check speaker settings
        try:
            settings = self._config_manager.get_settings()
            if not settings.speaker.enabled:
                self._live_last_status = "disabled"
                return None
        except Exception:
            pass  # Default to trying

        # Lazy-init extractor (non-blocking check)
        if not self._ensure_live_extractor():
            return None

        # Serialize extraction to avoid concurrent sherpa-onnx use
        if not self._live_extractor_lock.acquire(blocking=False):
            return None

        try:
            self._live_last_attempt_ts = now
            self._live_match_attempts += 1

            # Convert int16 PCM buffer to mono float32 numpy array
            import numpy as np
            raw_bytes = bytes(self._live_audio_buffer)
            pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = pcm_int16.astype(np.float32) / 32768.0

            # Create stream and compute embedding
            stream = self._live_extractor.create_stream()
            stream.accept_waveform(16000, audio_float32)
            stream.input_finished()

            if not self._live_extractor.is_ready(stream):
                self._live_last_status = "insufficient_audio"
                self._live_match_fallbacks += 1
                return None

            embedding = self._live_extractor.compute(stream)

            if embedding is None or len(embedding) == 0:
                self._live_last_status = "no_match"
                self._live_match_fallbacks += 1
                return None

            # Look up against known signatures
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir

            db_path = get_recordings_dir() / "speaker_signatures.db"
            try:
                with VoiceSignatureStore(db_path=db_path) as store:
                    match = store.find_match(
                        embedding,
                        threshold=0.75,  # Conservative: no match below this
                    )
            except Exception as store_exc:
                self._live_last_error_class = type(store_exc).__name__
                self._live_last_error_message = str(store_exc)[:120]
                self._live_last_status = "store_error"
                self._live_match_fallbacks += 1
                return None

            if match is None or match.confidence != "high":
                self._live_last_status = (
                    "high_confidence_match_without_name"
                    if match is not None and match.confidence != "high"
                    else "no_match"
                )
                self._live_match_fallbacks += 1
                return None

            # High-confidence match found
            self._live_match_hits += 1
            self._live_last_status = "matched"
            return match.name

        except Exception as exc:
            self._live_last_error_class = type(exc).__name__
            self._live_last_error_message = str(exc)[:120]
            self._live_last_status = "extractor_error"
            self._live_match_fallbacks += 1
            logger.debug(
                "Live speaker matching error: %s", type(exc).__name__
            )
            return None

        finally:
            self._live_extractor_lock.release()

    def _reset_live_speaker_state(self) -> None:
        """Reset live speaker matching state for a new recording."""
        self._live_audio_buffer = bytearray()
        self._live_match_attempts = 0
        self._live_match_hits = 0
        self._live_match_fallbacks = 0
        self._live_last_status = "disabled"
        self._live_last_error_class = None
        self._live_last_error_message = None
        self._live_last_attempt_ts = None
        # Keep extractor available across recordings for efficiency
        # (don't recreate it for each recording session)
        pass

    def _run_diarization(self, wav_path: Path) -> None:
        """Run speaker diarization on the saved WAV and tag transcript words.

        Post-processing step executed AFTER the WAV is saved and BEFORE the
        transcript is saved. Gracefully degrades if sherpa-onnx is not
        installed — logs a warning and returns without tagging.

        Args:
            wav_path: Path to the saved WAV file.
        """
        try:
            from meetandread.speaker.diarizer import Diarizer
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir
        except ImportError:
            logger.warning(
                "sherpa-onnx not installed — speaker diarization skipped. "
                "Install sherpa-onnx to enable speaker identification."
            )
            return

        try:
            settings = self._config_manager.get_settings()
            speaker_cfg = settings.speaker

            if not speaker_cfg.enabled:
                logger.info("Speaker diarization disabled in settings — skipped")
                return

            logger.info("Running speaker diarization on %s", wav_path.name)

            # (1) Run diarization
            diarizer = Diarizer(
                clustering_threshold=speaker_cfg.clustering_threshold,
                min_duration_on=speaker_cfg.min_duration_on,
                min_duration_off=speaker_cfg.min_duration_off,
            )
            logger.info(
                "Diarization config: clustering=%.2f min_duration_on=%.2f min_duration_off=%.2f",
                speaker_cfg.clustering_threshold,
                speaker_cfg.min_duration_on,
                speaker_cfg.min_duration_off,
            )
            result = diarizer.diarize(wav_path)

            if not result.succeeded:
                logger.error(
                    "Diarization failed for %s: %s", wav_path.name, result.error
                )
                return

            if not result.segments:
                logger.info("No speaker segments detected in %s", wav_path.name)
                return

            logger.info(
                "Diarized %s: %d segments, %d speakers",
                wav_path.name, len(result.segments), result.num_speakers,
            )

            # --- Clean up noisy over-segmentation ---
            from meetandread.speaker.diarizer import cleanup_diarization_segments
            pre_cleanup_count = len(result.segments)
            result.segments = cleanup_diarization_segments(result.segments)
            if len(result.segments) != pre_cleanup_count:
                logger.info(
                    "Diarization cleanup in controller: %d -> %d segments "
                    "(gap_threshold=%.2fs, short_threshold=%.2fs)",
                    pre_cleanup_count, len(result.segments),
                    0.2, 0.5,
                )

            # (2) Save diarization embeddings to signature store and match
            #     against known identities.
            #
            #     Every new speaker gets a raw-label profile (SPK_N) so the
            #     user can later link it to an identity via the history
            #     dialog.  If the embedding already matches a known
            #     identity, we skip the raw save and record the match
            #     instead.
            db_path = get_recordings_dir() / "speaker_signatures.db"
            with VoiceSignatureStore(db_path=db_path) as store:
                for label, sig in result.signatures.items():
                    match = store.find_match(
                        sig.embedding,
                        threshold=speaker_cfg.confidence_threshold,
                    )
                    if match:
                        result.matches[label] = match
                        logger.debug(
                            "Matched %s -> '%s' (score=%.4f, confidence=%s)",
                            label, match.name, match.score, match.confidence,
                        )
                    else:
                        # Save raw profile so it can be linked later.
                        # Use the same display label (SPK_N) that
                        # _apply_speaker_labels will assign to words.
                        display_label = result.speaker_label_for(label)
                        emb = np.asarray(sig.embedding, dtype=np.float32)
                        store.save_signature(
                            display_label,
                            emb,
                            averaged_from_segments=sig.num_segments,
                        )
                        logger.debug(
                            "Saved raw profile '%s' to signature store (no known match)",
                            display_label,
                        )

            # (3) Tag transcript words with speaker labels
            self._apply_speaker_labels(result)

            # Store result for pin-to-name UX
            self._last_diarization_result = result

        except Exception as exc:
            logger.error(
                "Speaker diarization error for %s: %s",
                wav_path.name, exc, exc_info=True,
            )

    def _apply_speaker_labels(self, result: "DiarizationResult") -> None:
        """Tag transcript store words with speaker IDs from diarization.

        Strategy:

        1. **Midpoint overlap**: For each word, check if its midpoint
           falls within a diarization segment. Assign the segment's
           speaker label.

        2. **Single-speaker fill**: When only 1 speaker is detected,
           the diarization segment often covers only part of the audio
           (pyannote is conservative on short recordings). In this
           case, assign ALL words to that single speaker regardless
           of segment boundaries.

        3. **Nearest-speaker fill**: For multi-speaker recordings,
           words that fall in gaps between segments are assigned to
           the temporally nearest segment's speaker.

        Args:
            result: A successful DiarizationResult with segments and matches.
        """
        assert self._transcript_store is not None
        from meetandread.speaker.models import DiarizationResult

        words = self._transcript_store.get_all_words()
        if not words:
            return

        # Build a mapping from raw label -> display label
        label_map: dict[str, str] = {}
        for seg in result.segments:
            raw = seg.speaker
            if raw not in label_map:
                label_map[raw] = result.speaker_label_for(raw)

        if not label_map:
            return

        # --- Pass 1: midpoint overlap ---
        tagged_count = 0
        for word in words:
            word_mid = (word.start_time + word.end_time) / 2
            for seg in result.segments:
                if seg.start <= word_mid <= seg.end:
                    word.speaker_id = label_map[seg.speaker]
                    tagged_count += 1
                    break

        # --- Pass 2: fill untagged words ---
        if tagged_count < len(words):
            if len(label_map) == 1:
                # Single speaker: assign all words to that speaker
                single_label = next(iter(label_map.values()))
                for word in words:
                    if word.speaker_id is None:
                        word.speaker_id = single_label
                        tagged_count += 1
            else:
                # Multi-speaker: assign to nearest segment's speaker
                sorted_segments = sorted(result.segments, key=lambda s: s.start)
                for word in words:
                    if word.speaker_id is not None:
                        continue
                    word_mid = (word.start_time + word.end_time) / 2
                    nearest_label = min(
                        sorted_segments,
                        key=lambda s: min(
                            abs(word_mid - s.start),
                            abs(word_mid - s.end),
                        ),
                    ).speaker
                    word.speaker_id = label_map[nearest_label]
                    tagged_count += 1

        logger.info(
            "Tagged %d/%d words with speaker labels (%d speakers)",
            tagged_count, len(words), len(label_map),
        )

    def _speaker_matches_metadata(self) -> dict:
        """Build a raw-label-keyed speaker matches map for transcript persistence.

        Returns an empty dict when there is no successful diarization result.
        Otherwise collects every unique raw label from detected segments and
        serialises matched ``SpeakerMatch`` objects as
        ``{"identity_name": ..., "score": ..., "confidence": ...}``; detected-
        but-unmatched labels are serialised as ``None``.

        Returns:
            Dict mapping raw labels (e.g. ``"spk0"``) to match dicts or None.
        """
        result = self._last_diarization_result
        if result is None or not getattr(result, "succeeded", False):
            return {}

        segments = getattr(result, "segments", [])
        if not segments:
            return {}

        matches_map: dict = {}
        # Collect all unique raw labels from segments
        raw_labels: set[str] = set()
        for seg in segments:
            raw_labels.add(seg.speaker)

        for label in sorted(raw_labels):
            match = getattr(result, "matches", {}).get(label)
            if match is not None:
                matches_map[label] = {
                    "identity_name": match.name,
                    "score": match.score,
                    "confidence": match.confidence,
                }
            else:
                matches_map[label] = None

        return matches_map

    def _save_transcript(self) -> Optional[Path]:
        """Save transcript to file.

        Persists speaker match metadata alongside transcript words so
        downstream consumers can resolve raw diarization labels without
        re-running diarization.

        Returns:
            Path to saved transcript file, or None if no transcript
        """
        if not self._transcript_store or not self._last_wav_path:
            return None
        
        try:
            from meetandread.audio.storage.paths import get_transcripts_dir
            
            # Create transcript filename based on WAV filename
            wav_stem = self._last_wav_path.stem
            transcripts_dir = get_transcripts_dir()
            transcript_path = transcripts_dir / f"{wav_stem}.md"

            # Build speaker matches metadata from last diarization result
            speaker_matches = self._speaker_matches_metadata()
            
            # Save as markdown with metadata
            self._transcript_store.save_to_file(
                transcript_path,
                speaker_matches=speaker_matches,
            )
            
            return transcript_path
            
        except Exception as e:
            print(f"Failed to save transcript: {e}")
            return None
    
    def _build_source_configs(
        self,
        selected_sources: Set[str],
        *,
        fake_path: Optional[str] = None,
        fake_denoise: bool = False,
        fake_loop: bool = False,
    ) -> List[SourceConfig]:
        """Build SourceConfig list from selected source types.
        
        Args:
            selected_sources: Set of source type strings.
            fake_path: Path to WAV file (required for 'fake' source).
            fake_denoise: Apply denoising to the fake source.
            fake_loop: Loop the fake source WAV file.
        
        Returns:
            List of SourceConfig objects for valid sources.
        """
        configs = []
        
        for source_type in selected_sources:
            source_type = source_type.lower().strip()
            
            if source_type == 'mic':
                configs.append(SourceConfig(type='mic', gain=1.0))
            elif source_type == 'system':
                configs.append(SourceConfig(type='system', gain=0.8))
            elif source_type == 'fake':
                if not fake_path:
                    logger.warning("Fake source requested without fake_path — skipping")
                    continue
                configs.append(SourceConfig(
                    type='fake',
                    fake_path=fake_path,
                    loop=fake_loop,
                    denoise=True if fake_denoise else None,
                ))
        
        return configs
    
    def get_last_recording_path(self) -> Optional[Path]:
        """Get path to the most recently completed recording."""
        return self._last_wav_path
    
    def get_last_transcript_path(self) -> Optional[Path]:
        """Get path to the most recently completed transcript."""
        return self._last_transcript_path
    
    def get_transcript_store(self) -> Optional[TranscriptStore]:
        """Get the current transcript store (for UI access during recording)."""
        return self._transcript_store
    
    def pin_speaker_name(self, raw_label: str, name: str) -> None:
        """Pin a user-chosen name to a speaker and save the voice signature.

        After pinning, re-checks all unmatched speakers against the updated
        signature store, then updates transcript word labels.

        Args:
            raw_label: Raw speaker label from diarization (e.g. "spk0").
            name: User-chosen display name for this speaker.
        """
        if not self._last_diarization_result or not self._last_transcript_path:
            logger.warning(
                "Cannot pin speaker '%s': no diarization result available",
                raw_label,
            )
            return

        result = self._last_diarization_result
        if not result.succeeded or raw_label not in result.signatures:
            logger.warning(
                "Cannot pin speaker '%s': no signature found in diarization result",
                raw_label,
            )
            return

        sig = result.signatures[raw_label]

        # Save or update the voice signature in the store
        from meetandread.audio.storage.paths import get_recordings_dir
        db_path = get_recordings_dir() / "speaker_signatures.db"
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
            with VoiceSignatureStore(db_path=db_path) as store:
                existing = store.find_match(sig.embedding, threshold=0.99)
                if existing and existing.name == name:
                    # Already saved — update the embedding average
                    store.update_signature(name, sig.embedding)
                else:
                    store.save_signature(name, sig.embedding, sig.num_segments)

                logger.info("Saved voice signature for '%s' (was %s)", name, raw_label)

                # Update the in-memory result mapping
                from meetandread.speaker.models import SpeakerMatch
                result.matches[raw_label] = SpeakerMatch(
                    name=name, score=1.0, confidence="high",
                )

                # Re-check all unmatched speakers against updated store
                for label, label_sig in result.signatures.items():
                    if label in result.matches:
                        continue  # Already matched (including the just-pinned one)
                    match = store.find_match(
                        label_sig.embedding,
                        threshold=self._config_manager.get_settings().speaker.confidence_threshold,
                    )
                    if match:
                        result.matches[label] = match
                        logger.info(
                            "Re-checked %s -> '%s' (score=%.4f)",
                            label, match.name, match.score,
                        )

            # Re-apply speaker labels to transcript words
            if self._transcript_store:
                self._apply_speaker_labels(result)

        except Exception as exc:
            logger.error("Failed to pin speaker '%s': %s", name, exc, exc_info=True)

    def get_speaker_names(self) -> dict:
        """Return current speaker label mapping from the last diarization.

        Returns:
            Dict mapping raw labels (e.g. "spk0") to display names
            (e.g. "Alice" or "SPK_0").
        """
        if not self._last_diarization_result:
            return {}
        result = self._last_diarization_result
        names = {}
        seen_labels = set()
        # From segments, collect all unique raw labels
        if hasattr(result, 'segments'):
            for seg in result.segments:
                if seg.speaker not in seen_labels:
                    seen_labels.add(seg.speaker)
                    names[seg.speaker] = result.speaker_label_for(seg.speaker)
        return names

    def get_last_wer(self) -> Optional[float]:
        """Return the WER value from the last auto-WER computation.

        Returns:
            WER as float (0.0–1.0+) or None if not yet computed.
        """
        return self._last_wer

    def get_diagnostics(self) -> dict:
        """Return sanitized controller diagnostics for testing/inspection.

        Exposes controller state, recording paths, session stats (including
        denoising stats), VAD/transcription stats when present, and
        diarization result metadata. Does NOT expose raw audio, transcript
        text, embeddings, or secrets.

        Returns:
            Dict of sanitized diagnostic key/value pairs.
        """
        diag: dict = {
            "state": self._state.name,
            "last_wav_path": str(self._last_wav_path) if self._last_wav_path else None,
            "last_transcript_path": str(self._last_transcript_path) if self._last_transcript_path else None,
        }

        # Session stats
        try:
            stats = self._session.get_stats()
            diag["session"] = {
                "frames_recorded": stats.frames_recorded,
                "frames_dropped": stats.frames_dropped,
                "duration_seconds": stats.duration_seconds,
                "source_stats": stats.source_stats,
                "denoising": {
                    "provider": stats.denoising.provider,
                    "enabled": stats.denoising.enabled,
                    "active": stats.denoising.active,
                    "processed_frame_count": stats.denoising.processed_frame_count,
                    "fallback_count": stats.denoising.fallback_count,
                    "avg_latency_ms": stats.denoising.avg_latency_ms,
                    "max_latency_ms": stats.denoising.max_latency_ms,
                    "budget_exceeded_count": stats.denoising.budget_exceeded_count,
                    "last_error_class": stats.denoising.last_error_class,
                },
            }
        except Exception:
            pass

        # Transcription / VAD stats
        if self._transcription_processor:
            try:
                ts = self._transcription_processor.get_stats()
                diag["transcription"] = {
                    "buffer_duration": ts.get("buffer_duration"),
                    "segments_processed": ts.get("segments_processed"),
                }
                # VAD stats if available
                vad = getattr(self._transcription_processor, "get_vad_stats", None)
                if callable(vad):
                    vs = vad()
                    diag["vad"] = vs
            except Exception:
                pass

        # Transcript store stats
        if self._transcript_store:
            try:
                words = self._transcript_store.get_all_words()
                diag["transcript"] = {
                    "word_count": len(words),
                    "words_with_speaker": sum(1 for w in words if w.speaker_id is not None),
                }
            except Exception:
                pass

        # Diarization result metadata
        if self._last_diarization_result:
            try:
                result = self._last_diarization_result
                raw_labels: set = set()
                if hasattr(result, "segments"):
                    for seg in result.segments:
                        raw_labels.add(seg.speaker)
                matches = getattr(result, "matches", {})
                matched_labels = {l for l in raw_labels if l in matches}
                diag["diarization"] = {
                    "succeeded": result.succeeded,
                    "num_speakers": result.num_speakers,
                    "segment_count": len(result.segments),
                    "error": result.error,
                    "match_count": len(matched_labels),
                    "matched_label_count": len(matched_labels),
                    "labels": sorted(raw_labels),
                }
            except Exception:
                pass

        # Live speaker matching diagnostics (sanitized — no names/embeddings)
        diag["live_speaker_matching"] = {
            "enabled": self._live_extractor_available is True,
            "extractor_available": self._live_extractor_available,
            "store_available": self._live_store_available,
            "audio_buffer_seconds": round(
                len(self._live_audio_buffer) / (16000 * 2), 1
            ),
            "attempts": self._live_match_attempts,
            "matches": self._live_match_hits,
            "fallbacks": self._live_match_fallbacks,
            "last_status": self._live_last_status,
            "last_error_class": self._live_last_error_class,
            "last_error_message": self._live_last_error_message,
            "last_attempt_ts": self._live_last_attempt_ts,
        }

        # Error info
        if self._error:
            diag["error"] = {
                "message": self._error.message,
                "is_recoverable": self._error.is_recoverable,
            }

        return diag
