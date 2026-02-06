"""
Proper real-time transcription implementation matching whisper_real_time reference.

Key differences from previous approach:
1. Accumulate audio over time (not chunk-by-chunk isolation)
2. Re-transcribe accumulated buffer for context continuity
3. Detect phrase breaks via timeout (3s silence)
4. Update display in-place (edit current line, don't add new items)
5. Confidence calculated per phrase (not per chunk)
6. 60-second window for good meeting context

This implementation is optimized for meetings and calls:
- 60s window provides enough context for accurate transcription
- 2s update frequency is responsive without overwhelming the CPU
- 3s silence detection for natural turn-taking
"""

import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Queue, Empty


@dataclass
class SegmentResult:
    """Result of transcribing a single segment."""
    text: str
    confidence: int
    start_time: float
    end_time: float
    segment_index: int  # Position in the phrase for UI matching
    is_final: bool  # True if this segment is from a completed phrase
    phrase_start: bool = False  # True if this is the first segment of a new phrase


class AccumulatingTranscriptionProcessor:
    """
    Real-time transcription that accumulates audio and re-transcribes for context.
    
    Architecture (matching whisper_real_time reference):
    1. Audio captured continuously
    2. Accumulated in buffer (phrase_bytes)
    3. Every 2 seconds or on 3s silence, transcribe accumulated audio
    4. Display updates in-place (current phrase edited, not new items added)
    5. After 3s silence, start new phrase
    
    Configuration for meetings:
    - window_size: 60 seconds (configurable, default for good context)
    - update_frequency: 2 seconds (responsive updates)
    - silence_timeout: 3 seconds (natural turn-taking)
    
    This provides:
    - Better accuracy (context from accumulated audio)
    - Lower latency (only transcribe when needed, not every chunk)
    - Natural phrase breaks (based on silence detection)
    - Meets < 2s latency requirement for Phase 2
    """
    
    def __init__(
        self,
        model_size: str = "tiny",
        window_size: float = 60.0,  # 60s buffer for good meeting context
        update_frequency: float = 2.0,  # Update every 2 seconds
        silence_timeout: float = 3.0,  # 3s silence = phrase complete
    ):
        """
        Initialize the accumulating transcription processor.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small")
            window_size: Maximum audio buffer size in seconds (default 60s)
            update_frequency: How often to run transcription (seconds)
            silence_timeout: Silence duration before considering phrase complete (seconds)
        """
        self.model_size = model_size
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.silence_timeout = silence_timeout
        
        # Maximum bytes in buffer (16kHz, 16-bit = 2 bytes per sample)
        self._max_buffer_bytes = int(window_size * 16000 * 2)
        
        # Accumulated audio buffer (raw bytes)
        self._phrase_bytes = bytes()
        
        # Transcription engine
        self._engine = None
        
        # Threading
        self._is_running = False
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        
        # Result queue for UI
        self._result_queue: Queue[SegmentResult] = Queue()
        
        # Timing
        self._last_audio_time: Optional[datetime] = None
        self._recording_start_time: Optional[datetime] = None
        self._last_update_time: Optional[datetime] = None
        
        # Callbacks
        self.on_result: Optional[Callable[[SegmentResult], None]] = None
        
        # Thread safety for model access
        self._model_lock = threading.Lock()
        
        # Debug counters
        self._audio_chunks_fed = 0
        self._total_samples_processed = 0
        self._transcription_count = 0
        self._result_counter = 0  # For tracking unique result IDs

        # Deduplication tracking
        self._last_transcribed_text = ""  # For deduplication
        self._last_phrase_start_time: Optional[datetime] = None  # Track phrase timing
        self._min_phrase_duration = 0.3  # Minimum audio duration before transcription (seconds)
        self._last_emitted_text = ""  # Track last emitted text for deduplication
        
        # Phrase tracking
        self._new_phrase_started = False  # Flag to indicate start of new phrase
        
        # Segment index tracking to prevent duplicate emission
        # Tracks the last segment index that was emitted to the UI
        self._last_emitted_segment_index = -1  # -1 means nothing emitted yet
    
    def load_model(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Load the Whisper model."""
        from metamemory.transcription.engine import WhisperTranscriptionEngine
        
        print(f"DEBUG: Loading {self.model_size} model for accumulating transcription...")
        self._engine = WhisperTranscriptionEngine(
            model_size=self.model_size,
            device="cpu",
            compute_type="int8"
        )
        self._engine.load_model(progress_callback=progress_callback)
        print(f"DEBUG: Model loaded successfully")
    
    def start(self) -> None:
        """Start the transcription processing loop."""
        if self._is_running:
            print("DEBUG: Processor already running")
            return
        
        self._is_running = True
        self._stop_event.clear()
        self._recording_start_time = datetime.utcnow()
        self._last_audio_time = None
        self._last_update_time = None
        self._phrase_bytes = bytes()
        self._audio_chunks_fed = 0
        self._total_samples_processed = 0
        self._transcription_count = 0
        self._last_transcribed_text = ""  # Reset dedup
        self._last_phrase_start_time = None  # Reset phrase timing
        self._last_emitted_segment_index = -1  # Reset segment tracking for new session
        self._last_emitted_text = ""  # Reset text tracking for new session
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="AccumulatingTranscriptionProcessor"
        )
        self._processing_thread.start()
        print("DEBUG: Accumulating transcription processor started")
        print(f"DEBUG: Window size: {self.window_size}s, Update frequency: {self.update_frequency}s, Silence timeout: {self.silence_timeout}s")
    
    def stop(self) -> None:
        """Stop the transcription processor."""
        if not self._is_running:
            return
        
        print("DEBUG: Stopping accumulating transcription processor...")
        self._is_running = False
        self._stop_event.set()
        
        # Wait for processing thread to finish
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        # Don't transcribe remaining audio here - it's unsafe and can cause GGML_ASSERT failure
        # The processing loop will handle any remaining audio or we accept that the last phrase is lost
        
        print(f"DEBUG: Processor stopped. Total transcriptions: {self._transcription_count}, Total audio chunks: {self._audio_chunks_fed}")
    
    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Feed audio data to be accumulated.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array (mono, 16kHz)
        """
        # Calculate audio energy (RMS) for speech detection
        if len(audio_chunk) > 0:
            energy = np.sqrt(np.mean(audio_chunk ** 2))
        else:
            energy = 0
        
        # Threshold for speech detection (tune this value based on testing)
        # Lowered to 0.005 for better sensitivity with quiet microphones
        SPEECH_THRESHOLD = 0.005
        is_speech = energy > SPEECH_THRESHOLD
        
        # Only update last_audio_time if speech detected (fixes silence timeout bug)
        if is_speech:
            self._last_audio_time = datetime.utcnow()
            if not hasattr(self, '_last_speech_debug'):
                self._last_speech_debug = False
            if not self._last_speech_debug:
                print(f"DEBUG: Speech detected (energy: {energy:.4f})")
            self._last_speech_debug = True
        else:
            if not hasattr(self, '_last_speech_debug'):
                self._last_speech_debug = True
            if self._last_speech_debug:
                print(f"DEBUG: Silence detected (energy: {energy:.4f})")
            self._last_speech_debug = False
        
        # Convert float32 to int16 bytes (what whisper.cpp expects)
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        elif audio_chunk.dtype == np.int16:
            audio_int16 = audio_chunk
        else:
            audio_int16 = audio_chunk.astype(np.int16)
        
        # Add to accumulated buffer (always, for continuous transcription)
        chunk_bytes = audio_int16.tobytes()
        self._phrase_bytes += chunk_bytes
        self._audio_chunks_fed += 1
        self._total_samples_processed += len(audio_chunk)
        
        # Debug: log every 50 chunks and every 10 seconds of audio
        if self._audio_chunks_fed % 50 == 0:
            buffer_duration = len(self._phrase_bytes) / (16000 * 2)  # bytes / (samples/sec * bytes/sample)
            print(f"DEBUG: Fed audio chunk #{self._audio_chunks_fed}: {len(audio_chunk)} samples, buffer: {buffer_duration:.1f}s")
        
        # Trim buffer if it exceeds window size (keep most recent audio)
        if len(self._phrase_bytes) > self._max_buffer_bytes:
            excess = len(self._phrase_bytes) - self._max_buffer_bytes
            self._phrase_bytes = self._phrase_bytes[excess:]
            print(f"DEBUG: Trimmed buffer to maintain {self.window_size}s window")
    
    def _processing_loop(self) -> None:
        """Main processing loop - runs in background thread."""
        print("DEBUG: Processing loop started")
        silence_debug_counter = 0
        
        while self._is_running and not self._stop_event.is_set():
            try:
                now = datetime.utcnow()
                
                # Check if we should transcribe
                should_transcribe = False
                phrase_complete = False
                
                if self._phrase_bytes and self._last_audio_time:
                    time_since_audio = (now - self._last_audio_time).total_seconds()
                    time_since_update = (now - self._last_update_time).total_seconds() if self._last_update_time else float('inf')
                    buffer_duration = len(self._phrase_bytes) / (16000 * 2)  # seconds of audio in buffer
                    
                    # Debug silence detection every 10 iterations (~1 second)
                    silence_debug_counter += 1
                    if silence_debug_counter >= 10:
                        silence_debug_counter = 0
                        silence_detected = time_since_audio >= self.silence_timeout
                        print(f"DEBUG: Silence check - {time_since_audio:.1f}s since audio, {buffer_duration:.1f}s buffer")
                    
                    # Transcribe if:
                    # 1. Silence timeout reached AND we have enough audio (phrase complete)
                    # 2. Update frequency reached and we have enough audio (> min_phrase_duration)
                    if time_since_audio >= self.silence_timeout:
                        # Silence detected - phrase is complete
                        # Only transcribe if we have enough audio (prevents BLANK_AUDIO)
                        if buffer_duration >= self._min_phrase_duration:
                            should_transcribe = True
                            phrase_complete = True
                            print(f"DEBUG: Silence detected ({time_since_audio:.1f}s >= {self.silence_timeout}s), finalizing phrase ({buffer_duration:.1f}s buffer)")
                        else:
                            print(f"DEBUG: Silence detected but buffer too small ({buffer_duration:.1f}s < {self._min_phrase_duration}s), skipping")
                    elif time_since_update >= self.update_frequency and buffer_duration >= self._min_phrase_duration:
                        # Update frequency reached - transcribe but continue phrase
                        should_transcribe = True
                        phrase_complete = False
                        print(f"DEBUG: Update frequency reached ({time_since_update:.1f}s), transcribing {buffer_duration:.1f}s buffer")
                
                if should_transcribe and self._engine:
                    transcribe_start = time.time()
                    self._transcribe_accumulated(phrase_complete)
                    transcribe_time = time.time() - transcribe_start
                    self._last_update_time = now

                    # CRITICAL FIX: Reset timing state when phrase is complete
                    # This prevents duplicate transcriptions after silence
                    if phrase_complete:
                        print(f"DEBUG: === PHRASE COMPLETE ({transcribe_time:.2f}s for transcription) ===")
                        print("DEBUG: Starting new phrase after silence")
                        self._phrase_bytes = bytes()  # Clear buffer
                        self._last_audio_time = None  # CRITICAL: Reset to prevent duplicate transcriptions
                        self._last_transcribed_text = ""  # Reset dedup
                        self._last_phrase_start_time = None  # Reset phrase timing
                        self._new_phrase_started = True  # Flag: next transcription starts new phrase
                        self._last_emitted_segment_index = -1  # Reset segment tracking for new phrase
                        self._last_emitted_text = ""  # Reset text tracking for new phrase
                        print("DEBUG: State reset - waiting for new audio")
                
                # Sleep to prevent CPU spinning (check every 100ms)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"ERROR: Transcription loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        print("DEBUG: Processing loop ended")
    
    def _transcribe_accumulated(self, force_complete: bool = False) -> None:
        """
        Transcribe the accumulated audio buffer.
        
        Outputs each segment individually so the UI can color them by confidence
        and update them in-place as the model refines its transcription.
        
        Args:
            force_complete: If True, this phrase is complete (3s silence reached)
        """
        if not self._phrase_bytes or not self._engine:
            return
        
        try:
            buffer_duration = len(self._phrase_bytes) / (16000 * 2)
            print(f"DEBUG: Transcribing {buffer_duration:.1f}s accumulated audio...")
            
            # Check if this is the start of a new phrase
            phrase_start = self._new_phrase_started
            self._new_phrase_started = False  # Reset flag after use
            
            # Convert bytes to numpy array
            audio_np = np.frombuffer(self._phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with thread safety
            start_time = time.time()
            with self._model_lock:
                segments = self._engine.transcribe_chunk(audio_np)
            transcribe_time = time.time() - start_time

            self._transcription_count += 1
            
            if segments:
                # Emit all segments for display
                # The panel's update_segment method handles in-place updates
                # Whisper re-transcribes the same content, but panel handles displaying it
                print(f"DEBUG: Processing {len(segments)} segments")

                for i, seg in enumerate(segments):
                    seg_text = seg.text.strip()
                    print(f"DEBUG:   Segment {i}: '{seg_text[:30]}...'")

                    # Skip blank audio markers only
                    if seg_text == "[BLANK_AUDIO]":
                        print(f"DEBUG:     Skipping [BLANK_AUDIO]")
                        continue

                    # Emit all non-blank segments
                    # The panel will display/replace as needed based on segment_index
                    print(f"DEBUG:     Emitting segment {i}")
                    self._result_queue.put(SegmentResult(
                        text=seg_text,
                        confidence=int(seg.confidence),
                        start_time=seg.start,
                        end_time=seg.end,
                        segment_index=i,
                        is_final=force_complete,
                        phrase_start=(i == 0 and self._new_phrase_started)
                    ))

                # Track last state (for logging only)
                if segments:
                    last_seg_text = segments[-1].text.strip()
                    self._last_emitted_text = last_seg_text
                    print(f"DEBUG:   Last emitted text: '{self._last_emitted_text[:30]}...'")


                    # Callback
                    if self.on_result:
                        try:
                            print(f"DEBUG: Calling on_result callback")
                            self.on_result(result)
                            print(f"DEBUG: on_result callback returned successfully")
                        except Exception as e:
                            print(f"ERROR: on_result callback failed: {e}")
                    else:
                        print(f"DEBUG: No on_result callback registered!")
                    
                    print(f"DEBUG: Segment {i}: '{segment_text}' [conf: {seg.confidence}%, final: {force_complete}]")
                
                # Update last emitted index to the highest index we just processed
                if segments:
                    self._last_emitted_segment_index = len(segments) - 1
                    print(f"DEBUG: Updated _last_emitted_segment_index to {self._last_emitted_segment_index}")
                
                print(f"DEBUG: Transcribed {len(segments)} total segments, {len(new_segments)} new, in {transcribe_time:.2f}s")
            else:
                print(f"DEBUG: No transcription result for {buffer_duration:.1f}s of audio")
                
        except Exception as e:
            print(f"ERROR: Transcription error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_results(self) -> List[SegmentResult]:
        """Get all pending results (non-blocking)."""
        results = []
        try:
            while True:
                result = self._result_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        return results
    
    def get_stats(self) -> dict:
        """Get processor statistics."""
        buffer_duration = len(self._phrase_bytes) / (16000 * 2) if self._phrase_bytes else 0
        return {
            "is_running": self._is_running,
            "model_size": self.model_size,
            "window_size": self.window_size,
            "buffer_duration": buffer_duration,
            "audio_chunks_fed": self._audio_chunks_fed,
            "total_samples": self._total_samples_processed,
            "transcription_count": self._transcription_count,
            "pending_results": self._result_queue.qsize(),
        }


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = AccumulatingTranscriptionProcessor(
        model_size="tiny",
        window_size=60.0,
        update_frequency=2.0,
        silence_timeout=3.0
    )
    
    # Set up result handler
    def on_segment(result: SegmentResult):
        status = "✓" if result.is_final else "→"
        print(f"{status} [{result.confidence}%]: {result.text}")
    
    processor.on_result = on_segment
    
    # Load model
    processor.load_model()
    
    # Start
    processor.start()
    
    print("\nTranscription active - speak into microphone")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        # Simulate feeding audio (in real app, this comes from AudioSession)
        import time
        time.sleep(30)  # Run for 30 seconds
    except KeyboardInterrupt:
        pass
    
    # Stop
    processor.stop()
    
    print("\nFinal results:")
    for result in processor.get_results():
        print(f"  [{result.start_time:.1f}s - {result.end_time:.1f}s] {result.text}")
