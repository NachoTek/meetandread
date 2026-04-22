"""Recording controller - UI-friendly wrapper around AudioSession.

Provides non-blocking recording control with proper error handling
and state management for UI integration. Includes hybrid transcription:
- Real-time: tiny model for immediate display using accumulating processor
- Post-process: stronger model after recording stops
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Set, Callable, List

from metamemory.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionState,
    SessionError,
    NoSourcesError,
)
from metamemory.audio.capture import AudioSourceError
from metamemory.transcription.accumulating_processor import AccumulatingTranscriptionProcessor, SegmentResult
from metamemory.transcription.transcript_store import TranscriptStore, Word
from metamemory.transcription.post_processor import PostProcessingQueue, PostProcessStatus
from metamemory.config.manager import ConfigManager

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
        
    
    def _set_state(self, state: ControllerState) -> None:
        """Update state and notify listeners."""
        self._state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception as e:
                print(f"ERROR: State change callback failed: {e}")
    
    def _set_error(self, message: str, is_recoverable: bool = True) -> ControllerError:
        """Set error state and notify listeners."""
        self._error = ControllerError(message, is_recoverable)
        self._set_state(ControllerState.ERROR)
        if self.on_error:
            try:
                self.on_error(self._error)
            except Exception as e:
                print(f"ERROR: Error callback failed: {e}")
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
    
    def start(self, selected_sources: Set[str]) -> Optional[ControllerError]:
        """Start recording from selected sources.
        
        Args:
            selected_sources: Set of source types ('mic', 'system', 'fake')
        
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
            source_configs = self._build_source_configs(selected_sources)
            
            if not source_configs:
                return self._set_error(
                    "No valid audio sources configured",
                    is_recoverable=True
                )
            
            # Create and start session
            config = SessionConfig(sources=source_configs)
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
                print(f"DEBUG: Transcription processor exists: {self._transcription_processor is not None}")
                print(f"DEBUG: Processor on_result callback: {self._transcription_processor.on_result is not None}")
                self._transcription_processor.start()
                print("DEBUG: Transcription processor started")
            
            self._audio_chunks_fed = 0
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
            print(f"DEBUG: Audio saved to: {wav_path}")
            
            # --- Speaker diarization (post-processing step) ---
            if self._transcript_store and self._last_wav_path:
                self._run_diarization(self._last_wav_path)
            
            # Save transcript if available
            transcript_path = None
            if self._transcript_store and self._last_wav_path:
                print(f"DEBUG: Saving transcript ({self._transcript_store.get_word_count()} words)...")
                transcript_path = self._save_transcript()
                self._last_transcript_path = transcript_path
                print(f"DEBUG: Transcript saved to: {transcript_path}")
            
            # Schedule post-processing with stronger model
            if self._post_processor and self._last_wav_path and self._transcript_store:
                print("DEBUG: Scheduling post-processing job...")
                job = self._post_processor.schedule_post_process(
                    audio_file=self._last_wav_path,
                    realtime_transcript=self._transcript_store,
                    output_dir=self._last_wav_path.parent
                )
                self._post_process_job_id = job.job_id
                print(f"DEBUG: Post-processing job scheduled: {job.job_id}")
            
            self._set_state(ControllerState.IDLE)
            print("DEBUG: Recording stopped, state set to IDLE")
            
            # Notify completion
            if self.on_recording_complete:
                try:
                    self.on_recording_complete(wav_path, transcript_path)
                except Exception as e:
                    print(f"ERROR: Recording complete callback failed: {e}")
                
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
            print(f"DEBUG: Initializing accumulating transcription with {realtime_model} model")
            
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
            print(f"DEBUG: Loading {realtime_model} model for real-time transcription...")
            self._transcription_processor.load_model(
                progress_callback=lambda p: print(f"Loading {realtime_model} model: {p}%")
            )
            print(f"DEBUG: {realtime_model} model loaded successfully")
            
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
                print(f"ERROR: Segment result callback failed: {e}")
    
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
        print(f"DEBUG: Post-processing job {job_id}: {progress}%")
    
    def _on_post_process_complete_callback(self, job_id: str, result: dict) -> None:
        """Handle post-processing completion.
        
        Args:
            job_id: The job identifier
            result: Result dictionary with transcript_path, etc.
        """
        print(f"DEBUG: Post-processing job {job_id} completed!")
        print(f"DEBUG: Post-processed transcript: {result.get('transcript_path')}")
        print(f"DEBUG: Real-time words: {result.get('realtime_word_count')}")
        print(f"DEBUG: Post-processed words: {result.get('word_count')}")
        
        if self.on_post_process_complete:
            transcript_path_str = result.get('transcript_path')
            if transcript_path_str and isinstance(transcript_path_str, str):
                transcript_path = Path(transcript_path_str)
                try:
                    self.on_post_process_complete(job_id, transcript_path)
                except Exception as e:
                    print(f"ERROR: Post-process complete callback failed: {e}")
    
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
                print(f"DEBUG: Fed {self._audio_chunks_fed} audio chunks, buffer: {stats.get('buffer_duration', 0):.1f}s")
    
    def _run_diarization(self, wav_path: Path) -> None:
        """Run speaker diarization on the saved WAV and tag transcript words.

        Post-processing step executed AFTER the WAV is saved and BEFORE the
        transcript is saved. Gracefully degrades if sherpa-onnx is not
        installed — logs a warning and returns without tagging.

        Args:
            wav_path: Path to the saved WAV file.
        """
        try:
            from metamemory.speaker.diarizer import Diarizer
            from metamemory.speaker.signatures import VoiceSignatureStore
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
            diarizer = Diarizer(clustering_threshold=speaker_cfg.clustering_threshold)
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

            # (2) Match speaker embeddings against known signatures
            db_path = wav_path.parent / "speaker_signatures.db"
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

        For each diarized segment, finds words whose time range overlaps
        and assigns the resolved speaker label (known name or SPK_N).

        Args:
            result: A successful DiarizationResult with segments and matches.
        """
        assert self._transcript_store is not None
        from metamemory.speaker.models import DiarizationResult

        words = self._transcript_store.get_all_words()
        if not words:
            return

        # Build a mapping from raw label -> display label
        label_map: dict[str, str] = {}
        for seg in result.segments:
            raw = seg.speaker
            if raw not in label_map:
                label_map[raw] = result.speaker_label_for(raw)

        tagged_count = 0
        for word in words:
            word_mid = (word.start_time + word.end_time) / 2
            for seg in result.segments:
                if seg.start <= word_mid <= seg.end:
                    word.speaker_id = label_map[seg.speaker]
                    tagged_count += 1
                    break

        logger.info(
            "Tagged %d/%d words with speaker labels (%d speakers)",
            tagged_count, len(words), len(label_map),
        )

    def _save_transcript(self) -> Optional[Path]:
        """Save transcript to file.
        
        Returns:
            Path to saved transcript file, or None if no transcript
        """
        if not self._transcript_store or not self._last_wav_path:
            return None
        
        try:
            # Create transcript filename based on WAV filename
            wav_stem = self._last_wav_path.stem
            transcript_path = self._last_wav_path.parent / f"{wav_stem}.md"
            
            # Save as markdown with metadata
            self._transcript_store.save_to_file(transcript_path)
            
            return transcript_path
            
        except Exception as e:
            print(f"Failed to save transcript: {e}")
            return None
    
    def _build_source_configs(self, selected_sources: Set[str]) -> List[SourceConfig]:
        """Build SourceConfig list from selected source types."""
        configs = []
        
        for source_type in selected_sources:
            source_type = source_type.lower().strip()
            
            if source_type == 'mic':
                configs.append(SourceConfig(type='mic', gain=1.0))
            elif source_type == 'system':
                configs.append(SourceConfig(type='system', gain=0.8))
            elif source_type == 'fake':
                # For testing - handled separately in test scenarios
                pass
        
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
        db_path = self._last_transcript_path.parent / "speaker_signatures.db"
        try:
            from metamemory.speaker.signatures import VoiceSignatureStore
            with VoiceSignatureStore(db_path=db_path) as store:
                existing = store.find_match(sig.embedding, threshold=0.99)
                if existing and existing.name == name:
                    # Already saved — update the embedding average
                    store.update_signature(name, sig.embedding)
                else:
                    store.save_signature(name, sig.embedding, sig.num_segments)

                logger.info("Saved voice signature for '%s' (was %s)", name, raw_label)

                # Update the in-memory result mapping
                from metamemory.speaker.models import SpeakerMatch
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
