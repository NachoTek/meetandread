"""Streaming pipeline for real-time transcription.

Orchestrates all transcription components into a unified pipeline:
- AudioRingBuffer: Thread-safe audio buffering
- VADChunkingProcessor: Intelligent audio segmentation
- WhisperTranscriptionEngine: Whisper model inference (whisper.cpp backend)
- LocalAgreementBuffer: Prevents text flickering

This class runs transcription in a background thread to avoid blocking
the audio capture or UI threads.

Compatible with whisper.cpp via pywhispercpp (CPU-only, no PyTorch DLLs).
"""

import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, List, Callable, Any
from pathlib import Path
import numpy as np

from metamemory.config.models import TranscriptionSettings
from metamemory.transcription.audio_buffer import AudioRingBuffer
from metamemory.transcription.vad_processor import VADChunkingProcessor
from metamemory.transcription.engine import WhisperTranscriptionEngine, TranscriptionSegment
from metamemory.transcription.local_agreement import LocalAgreementBuffer


@dataclass
class PipelineResult:
    """Result from the transcription pipeline.
    
    Attributes:
        text: Transcribed text
        confidence: Confidence score (0-100)
        start_time: Start timestamp in seconds from recording start
        end_time: End timestamp
        words: List of word-level data with timestamps and confidence
    """
    text: str
    confidence: int
    start_time: float
    end_time: float
    words: List[Any]  # List of WordInfo or similar


class RealTimeTranscriptionProcessor:
    """Orchestrates real-time transcription from audio capture to text output.
    
    This class integrates all transcription components and runs inference
    in a background thread to maintain low latency and avoid blocking.
    
    Threading Model:
    - Audio capture thread: Calls feed_audio() from AudioSession consumer
    - Processing thread: Runs inference in background (_processing_loop)
    - UI thread: Calls get_results() to retrieve new segments
    
    All component access is thread-safe through internal locking.
    
    Example:
        config = TranscriptionSettings(min_chunk_size_sec=1.0, agreement_threshold=2)
        processor = RealTimeTranscriptionProcessor(config)
        
        # Load model (can be done before starting)
        processor.load_model(lambda progress: print(f"Loading: {progress}%"))
        
        # Start processing
        processor.start()
        
        # Feed audio from capture callback
        processor.feed_audio(audio_chunk)
        
        # Get results in UI update loop
        results = processor.get_results()
        
        # Stop when done
        processor.stop()
    """
    
    def __init__(self, config: TranscriptionSettings):
        """Initialize the transcription processor.
        
        Args:
            config: Transcription configuration including chunk sizes and thresholds
        """
        self._config = config
        
        # Core components
        self._audio_buffer = AudioRingBuffer(max_seconds=30, sample_rate=16000)
        self._vad_processor = VADChunkingProcessor(
            min_chunk_size_sec=config.min_chunk_size_sec,
            sample_rate=16000
        )
        # For streaming, use agreement_threshold=1 to commit immediately
        # (threshold=2 is for static audio where same text appears multiple times)
        streaming_threshold = max(1, config.agreement_threshold - 1)
        self._agreement_buffer = LocalAgreementBuffer(
            agreement_threshold=streaming_threshold
        )
        
        # Engine is created but model not loaded yet
        self._engine: Optional[WhisperTranscriptionEngine] = None
        self._model_size: str = "base"  # Default, can be changed before load
        self._model_device: str = "cpu"
        self._model_compute_type: str = "int8"
        
        # Threading
        self._processing_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_event = threading.Event()
        self._result_queue: queue.Queue[PipelineResult] = queue.Queue()
        self._buffer_lock = threading.Lock()
        
        # State tracking
        self._recording_start_time: Optional[float] = None
        self._total_samples_processed = 0
        self._last_vad_was_speech = False
        
        # Callback for model loading progress
        self._load_progress_callback: Optional[Callable[[int], None]] = None
    
    def set_model_config(self, model_size: str, device: str = "cpu", 
                         compute_type: str = "int8") -> None:
        """Configure model parameters before loading.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cpu", "cuda", etc.)
            compute_type: Compute type ("int8", "float16", "float32")
        """
        self._model_size = model_size
        self._model_device = device
        self._model_compute_type = compute_type
    
    def load_model(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Load the Whisper model.
        
        This can take 5-10 seconds depending on model size and hardware.
        Call before start() for immediate transcription availability.
        
        Args:
            progress_callback: Optional callback(int: 0-100) for loading progress
        """
        self._load_progress_callback = progress_callback
        
        if progress_callback:
            progress_callback(0)
        
        # Create and load the engine
        self._engine = WhisperTranscriptionEngine(
            model_size=self._model_size,
            device=self._model_device,
            compute_type=self._model_compute_type
        )
        
        if progress_callback:
            progress_callback(50)
        
        self._engine.load_model()
        
        if progress_callback:
            progress_callback(100)
    
    def is_model_loaded(self) -> bool:
        """Check if the Whisper model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        if self._engine is None:
            return False
        return self._engine.is_model_loaded()
    
    def start(self) -> None:
        """Start the transcription processing thread.
        
        Raises:
            RuntimeError: If called when already running
        """
        if self._is_running:
            raise RuntimeError("Processor is already running")
        
        # Reset state
        self._recording_start_time = time.time()
        self._total_samples_processed = 0
        self._last_vad_was_speech = False
        
        # Clear any previous results
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start processing thread
        self._is_running = True
        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="TranscriptionProcessor"
        )
        self._processing_thread.start()
    
    def stop(self) -> None:
        """Stop the transcription processing thread.
        
        Waits up to 5 seconds for the processing thread to finish.
        """
        if not self._is_running:
            return
        
        self._is_running = False
        self._stop_event.set()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
            self._processing_thread = None
    
    def feed_audio(self, chunk: np.ndarray) -> None:
        """Feed audio data from the capture thread.
        
        This method is thread-safe and can be called from any thread,
        typically from the AudioSession consumer thread.
        
        Args:
            chunk: Audio samples as float32 numpy array at 16kHz mono
        """
        with self._buffer_lock:
            self._audio_buffer.append(chunk)
            
            # Update VAD state (we assume speech if we got audio)
            # In a full implementation, this would come from a real VAD
            self._vad_processor.feed_audio(chunk, vad_is_speech=True)
    
    def get_results(self) -> List[PipelineResult]:
        """Get new transcription results (non-blocking).
        
        Call this from the UI thread periodically to retrieve
        new transcribed segments.
        
        Returns:
            List of PipelineResult objects since last call
        """
        results = []
        try:
            while True:
                result = self._result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def _processing_loop(self) -> None:
        """Background thread that processes audio chunks.
        
        This runs continuously while is_running is True, checking
        for audio in the buffer and running Whisper inference.
        """
        while self._is_running and not self._stop_event.is_set():
            # Check if we have enough audio to process
            should_process = False
            chunk_to_process = None
            
            with self._buffer_lock:
                # Check if we should extract a chunk
                if self._vad_processor.should_process():
                    chunk_to_process = self._vad_processor.get_chunk()
                    should_process = chunk_to_process is not None
            
            if should_process and chunk_to_process is not None and self._engine is not None:
                # Ensure model is loaded
                if not self._engine.is_model_loaded():
                    self._engine.load_model()
                
                # Run transcription (this is the slow part - 0.5-2s)
                try:
                    print(f"DEBUG: Transcribing chunk of {len(chunk_to_process)} samples...")
                    segments = self._engine.transcribe_chunk(chunk_to_process)
                    print(f"DEBUG: Transcription returned {len(segments)} segments")
                    
                    # Process segments through agreement buffer
                    self._process_segments(segments, chunk_to_process)
                    
                    # Update tracking
                    self._total_samples_processed += len(chunk_to_process)
                    
                except Exception as e:
                    # Log error but continue processing
                    print(f"Transcription error: {e}")
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
            else:
                # Wait a bit longer if no audio to process
                time.sleep(0.05)
    
    def _process_segments(self, segments: List[TranscriptionSegment], 
                         audio_chunk: np.ndarray) -> None:
        """Process transcription segments and update results.
        
        Args:
            segments: Transcription segments from Whisper
            audio_chunk: The audio chunk that was transcribed
        """
        if not segments:
            return
        
        # Get the text from all segments
        full_text = " ".join([seg.text for seg in segments]).strip()
        
        if not full_text:
            return
        
        # Pass through agreement buffer to prevent flickering
        committed_text = self._agreement_buffer.process_iteration(full_text)
        print(f"DEBUG: Agreement buffer committed: '{committed_text}'")
        
        if committed_text:
            # Calculate timing
            chunk_duration = len(audio_chunk) / 16000  # 16kHz sample rate
            end_time = self._total_samples_processed / 16000
            start_time = end_time - chunk_duration
            
            # Calculate average confidence
            avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
            
            # Collect all words
            all_words = []
            for seg in segments:
                if hasattr(seg, 'words') and seg.words:
                    all_words.extend(seg.words)
            
            # Create result
            result = PipelineResult(
                text=committed_text,
                confidence=int(avg_confidence),
                start_time=max(0, start_time),
                end_time=end_time,
                words=all_words
            )
            
            # Queue for UI
            self._result_queue.put(result)
    
    def get_stats(self) -> dict:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        with self._buffer_lock:
            buffer_duration = self._audio_buffer.get_total_duration()
        
        return {
            "is_running": self._is_running,
            "is_model_loaded": self.is_model_loaded(),
            "model_size": self._model_size,
            "buffer_duration_sec": buffer_duration,
            "total_samples_processed": self._total_samples_processed,
            "pending_results": self._result_queue.qsize(),
        }
    
    def reset(self) -> None:
        """Reset the processor for a new recording session.
        
        Clears all buffers and state, but keeps the loaded model.
        """
        with self._buffer_lock:
            # Clear audio buffer
            self._audio_buffer = AudioRingBuffer(max_seconds=30, sample_rate=16000)
            
            # Reset VAD processor
            self._vad_processor = VADChunkingProcessor(
                min_chunk_size_sec=self._config.min_chunk_size_sec,
                sample_rate=16000
            )
        
        # Reset agreement buffer
        self._agreement_buffer = LocalAgreementBuffer(
            agreement_threshold=self._config.agreement_threshold
        )
        
        # Clear results queue
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state
        self._recording_start_time = None
        self._total_samples_processed = 0
        self._last_vad_was_speech = False
