"""Recording controller - UI-friendly wrapper around AudioSession.

Provides non-blocking recording control with proper error handling
and state management for UI integration. Includes hybrid transcription:
- Real-time: tiny model for immediate display
- Post-process: stronger model after recording stops
"""

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
from metamemory.transcription.streaming_pipeline import RealTimeTranscriptionProcessor, PipelineResult
from metamemory.transcription.transcript_store import TranscriptStore, Word
from metamemory.transcription.engine import WordInfo
from metamemory.transcription.post_processor import PostProcessingQueue, PostProcessStatus
from metamemory.config.manager import ConfigManager


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
    - Real-time transcription integration
    
    Example:
        controller = RecordingController()
        controller.on_state_change = lambda state: print(f"State: {state}")
        controller.on_error = lambda err: print(f"Error: {err.message}")
        controller.on_word_received = lambda word: print(f"Word: {word.text}")
        
        # Start recording
        error = controller.start({'mic', 'system'})
        if error:
            print(f"Failed to start: {error.message}")
        
        # Stop recording (non-blocking)
        controller.stop(lambda path, transcript_path: 
            print(f"Saved to: {path}, transcript: {transcript_path}"))
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
        self._transcription_processor: Optional[RealTimeTranscriptionProcessor] = None
        self._transcript_store: Optional[TranscriptStore] = None
        self._post_processor: Optional[PostProcessingQueue] = None
        self._post_process_job_id: Optional[str] = None
        self._config_manager = ConfigManager()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[ControllerState], None]] = None
        self.on_error: Optional[Callable[[ControllerError], None]] = None
        self.on_recording_complete: Optional[Callable[[Path, Optional[Path]], None]] = None
        self.on_transcript_update: Optional[Callable[[List[Word]], None]] = None
        self.on_word_received: Optional[Callable[[Word], None]] = None
        self.on_post_process_complete: Optional[Callable[[str, Path], None]] = None  # job_id, enhanced_path
        

    
    def _set_state(self, state: ControllerState) -> None:
        """Update state and notify listeners."""
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)
    
    def _set_error(self, message: str, is_recoverable: bool = True) -> ControllerError:
        """Set error state and notify listeners."""
        self._error = ControllerError(message, is_recoverable)
        self._set_state(ControllerState.ERROR)
        if self.on_error:
            self.on_error(self._error)
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
        
        try:
            # Initialize transcription if enabled
            if self.enable_transcription:
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
            if self.enable_transcription:
                config.on_audio_frame = self.feed_audio_for_transcription
            self._session = AudioSession()
            self._session.start(config)
            
            # Start transcription if available
            if self._transcription_processor:
                self._transcription_processor.start()
                # Start polling for results
                self._start_transcription_polling()
            
            self._set_state(ControllerState.RECORDING)
            return None
            
        except NoSourcesError as e:
            return self._set_error(f"No sources: {e}", is_recoverable=True)
        except AudioSourceError as e:
            return self._set_error(f"Audio device error: {e}", is_recoverable=True)
        except SessionError as e:
            return self._set_error(f"Session error: {e}", is_recoverable=True)
        except Exception as e:
            return self._set_error(f"Unexpected error: {e}", is_recoverable=False)
    
    def stop(self, on_complete: Optional[Callable[[Path, Optional[Path]], None]] = None) -> Optional[ControllerError]:
        """Stop recording and finalize to WAV.
        
        This is non-blocking - finalization happens on a worker thread.
        
        Args:
            on_complete: Callback when finalization completes (receives wav path and transcript path)
        
        Returns:
            ControllerError if stop cannot be initiated, None if stop started
        """
        if self._state != ControllerState.RECORDING:
            return self._set_error("Not currently recording", is_recoverable=True)
        
        self._set_state(ControllerState.STOPPING)
        
        # Store callback
        if on_complete:
            self.on_recording_complete = on_complete
        
        # Stop transcription polling
        self._stop_transcription_polling()
        
        # Run stop/finalize in worker thread to avoid blocking UI
        self._worker_thread = threading.Thread(
            target=self._stop_worker,
            daemon=True,
        )
        self._worker_thread.start()
        
        return None
    
    def _stop_worker(self) -> None:
        """Worker thread that handles stop and finalization."""
        try:
            # Stop transcription first to flush results
            if self._transcription_processor:
                # Get any remaining results
                print("DEBUG: Flushing final transcription results...")
                self._poll_transcription_results()
                self._transcription_processor.stop()
                self._transcription_processor = None
            
            # Stop audio session
            print("DEBUG: Stopping audio session...")
            wav_path = self._session.stop()
            self._last_wav_path = wav_path
            print(f"DEBUG: Audio saved to: {wav_path}")
            
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
            
            # Notify completion
            if self.on_recording_complete:
                self.on_recording_complete(wav_path, transcript_path)
                
        except Exception as e:
            self._set_error(f"Failed to finalize recording: {e}", is_recoverable=False)
    
    def _init_transcription(self) -> Optional[ControllerError]:
        """Initialize transcription components.
        
        HYBRID TRANSCRIPTION:
        - Uses tiny model for real-time transcription (fast, responsive)
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
            print(f"DEBUG: Initializing real-time transcription with {realtime_model} model")
            
            # Create transcript store
            self._transcript_store = TranscriptStore()
            self._transcript_store.start_recording()
            
            # Create transcription processor with tiny model
            self._transcription_processor = RealTimeTranscriptionProcessor(
                config=settings.transcription,
                model_size=realtime_model
            )
            
            # Load model (tiny takes 1-2 seconds)
            if not self._transcription_processor.is_model_loaded():
                print(f"DEBUG: Loading {realtime_model} model for real-time transcription...")
                self._transcription_processor.load_model(
                    progress_callback=lambda p: print(f"Loading {realtime_model} model: {p}%")
                )
                print(f"DEBUG: {realtime_model} model loaded successfully")
            
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
            return ControllerError(
                message=f"Failed to initialize transcription: {e}",
                is_recoverable=True
            )
    
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
            result: Result dictionary with enhanced_path, etc.
        """
        print(f"DEBUG: Post-processing job {job_id} completed!")
        print(f"DEBUG: Enhanced transcript: {result.get('enhanced_path')}")
        print(f"DEBUG: Real-time words: {result.get('realtime_word_count')}")
        print(f"DEBUG: Enhanced words: {result.get('word_count')}")
        
        if self.on_post_process_complete:
            enhanced_path_str = result.get('enhanced_path')
            if enhanced_path_str and isinstance(enhanced_path_str, str):
                enhanced_path = Path(enhanced_path_str)
                self.on_post_process_complete(job_id, enhanced_path)
    
    def _start_transcription_polling(self) -> None:
        """Start polling for transcription results."""
        # Audio frames are fed via on_audio_frame callback in SessionConfig
        # Poll for transcription results in a separate thread
        self._transcription_poll_thread = threading.Thread(
            target=self._transcription_poll_loop,
            daemon=True,
            name="TranscriptionPoller"
        )
        self._transcription_poll_thread.start()
    
    def _stop_transcription_polling(self) -> None:
        """Stop polling for transcription results."""
        self._stop_polling = True
        if hasattr(self, '_transcription_poll_thread') and self._transcription_poll_thread:
            self._transcription_poll_thread.join(timeout=1.0)
    
    def _transcription_poll_loop(self) -> None:
        """Background thread that polls for transcription results."""
        self._stop_polling = False
        while not self._stop_polling and self._state == ControllerState.RECORDING:
            self._poll_transcription_results()
            # Poll every 100ms
            import time
            time.sleep(0.1)
    
    def _poll_transcription_results(self) -> None:
        """Poll for and process transcription results."""
        if not self._transcription_processor or not self._transcript_store:
            return
        
        # Get new results
        results = self._transcription_processor.get_results()
        
        # DEBUG: Track results
        if results:
            print(f"DEBUG: Got {len(results)} transcription results")
        
        if not results:
            return
        
        # Convert PipelineResults to Words
        new_words = []
        for result in results:
            print(f"DEBUG: Processing result: text='{result.text}', words={len(result.words)}, confidence={result.confidence}")
            # Create Word objects from result
            for word_info in result.words:
                word = Word(
                    text=getattr(word_info, 'word', str(word_info)),
                    start_time=getattr(word_info, 'start', result.start_time),
                    end_time=getattr(word_info, 'end', result.end_time),
                    confidence=result.confidence,
                    is_enhanced=False,
                    speaker_id=None
                )
                new_words.append(word)
                
                # Notify individual word callback
                if self.on_word_received:
                    self.on_word_received(word)
        
        # Add to store
        print(f"DEBUG: Adding {len(new_words)} words to transcript store")
        if new_words:
            self._transcript_store.add_words(new_words)
            print(f"DEBUG: Transcript store now has {len(self._transcript_store.get_all_words())} words")
            
            # Notify batch update callback
            if self.on_transcript_update:
                self.on_transcript_update(new_words)
    
    def feed_audio_for_transcription(self, audio_chunk) -> None:
        """Feed audio chunk to transcription processor.
        
        This should be called from the audio capture consumer thread
        to provide audio data for transcription.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array
        """
        if self._transcription_processor and self._state == ControllerState.RECORDING:
            self._transcription_processor.feed_audio(audio_chunk)
            # DEBUG: Print every 100th chunk to verify audio is flowing
            if not hasattr(self, '_feed_count'):
                self._feed_count = 0
            self._feed_count += 1
            if self._feed_count % 100 == 0:
                print(f"DEBUG: Fed {self._feed_count} audio chunks, last size: {len(audio_chunk)}")
    
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
                # For testing - would need fake_path in real usage
                # This is handled specially in the widget
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
