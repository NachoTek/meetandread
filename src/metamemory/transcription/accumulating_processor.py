"""
Proper real-time transcription implementation matching whisper_real_time reference.

Key differences from our current approach:
1. Accumulate audio over time (not chunk-by-chunk isolation)
2. Re-transcribe accumulated buffer for context continuity
3. Detect phrase breaks via timeout (3s silence)
4. Update display in-place (edit current line, don't add new items)
5. Confidence calculated per phrase (not per chunk)
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
class PhraseResult:
    """Result of transcribing a phrase (accumulated audio)."""
    text: str
    confidence: int
    start_time: float
    end_time: float
    is_complete: bool  # True if phrase ended (3s silence detected)


class AccumulatingTranscriptionProcessor:
    """
    Real-time transcription that accumulates audio and re-transcribes for context.
    
    Architecture (matching whisper_real_time reference):
    1. Audio captured continuously
    2. Accumulated in buffer (phrase_bytes)
    3. Every N seconds or on silence timeout, transcribe accumulated audio
    4. Display updates in-place (current phrase edited, not new items added)
    5. After 3s silence, start new phrase
    
    This provides:
    - Better accuracy (context from accumulated audio)
    - Lower latency (only transcribe when needed, not every chunk)
    - Natural phrase breaks (based on silence detection)
    """
    
    def __init__(
        self,
        model_size: str = "tiny",
        record_timeout: float = 2.0,  # How often to check/transcribe
        phrase_timeout: float = 3.0,  # Silence duration to consider phrase complete
    ):
        """
        Initialize the accumulating transcription processor.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small")
            record_timeout: How often to run transcription (seconds)
            phrase_timeout: Silence duration before considering phrase complete (seconds)
        """
        self.model_size = model_size
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        
        # Accumulated audio buffer (raw bytes)
        self._phrase_bytes = bytes()
        
        # Transcription engine
        self._engine = None
        
        # Threading
        self._is_running = False
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        
        # Result queue for UI
        self._result_queue: Queue[PhraseResult] = Queue()
        
        # Timing
        self._last_audio_time: Optional[datetime] = None
        self._recording_start_time: Optional[datetime] = None
        
        # Callbacks
        self.on_result: Optional[Callable[[PhraseResult], None]] = None
    
    def load_model(self, progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Load the Whisper model."""
        from metamemory.transcription.engine import WhisperTranscriptionEngine
        
        print(f"Loading {self.model_size} model...")
        self._engine = WhisperTranscriptionEngine(
            model_size=self.model_size,
            device="cpu",
            compute_type="int8"
        )
        self._engine.load_model(progress_callback=progress_callback)
        print(f"✓ Model loaded")
    
    def start(self) -> None:
        """Start the transcription processing loop."""
        if self._is_running:
            return
        
        self._is_running = True
        self._stop_event.clear()
        self._recording_start_time = datetime.utcnow()
        self._last_audio_time = None
        self._phrase_bytes = bytes()
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="TranscriptionProcessor"
        )
        self._processing_thread.start()
        print("✓ Transcription processor started")
    
    def stop(self) -> None:
        """Stop the transcription processor."""
        if not self._is_running:
            return
        
        self._is_running = False
        self._stop_event.set()
        
        # Process any remaining audio
        if self._phrase_bytes:
            print("Processing final phrase...")
            self._transcribe_accumulated(force_complete=True)
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        print("✓ Transcription processor stopped")
    
    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Feed audio data to be accumulated.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array (mono, 16kHz)
        """
        # Convert float32 to int16 bytes (what whisper.cpp expects)
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)
        
        # Add to accumulated buffer
        self._phrase_bytes += audio_int16.tobytes()
        self._last_audio_time = datetime.utcnow()
    
    def _processing_loop(self) -> None:
        """Main processing loop - runs in background thread."""
        while self._is_running and not self._stop_event.is_set():
            try:
                now = datetime.utcnow()
                
                # Check if we should transcribe
                should_transcribe = False
                phrase_complete = False
                
                if self._phrase_bytes and self._last_audio_time:
                    time_since_audio = (now - self._last_audio_time).total_seconds()
                    
                    # Transcribe if:
                    # 1. We have audio and record_timeout has passed, OR
                    # 2. Phrase timeout (3s silence) has passed
                    if time_since_audio >= self.phrase_timeout:
                        # Silence detected - phrase is complete
                        should_transcribe = True
                        phrase_complete = True
                    elif len(self._phrase_bytes) >= int(self.record_timeout * 16000 * 2):  # 2 bytes per int16
                        # Record timeout reached - transcribe but continue phrase
                        should_transcribe = True
                        phrase_complete = False
                
                if should_transcribe and self._engine:
                    self._transcribe_accumulated(phrase_complete)
                    
                    if phrase_complete:
                        # Clear buffer for next phrase
                        self._phrase_bytes = bytes()
                
                # Sleep to prevent CPU spinning
                time.sleep(0.25)
                
            except Exception as e:
                print(f"Transcription loop error: {e}")
                time.sleep(0.5)
    
    def _transcribe_accumulated(self, force_complete: bool = False) -> None:
        """
        Transcribe the accumulated audio buffer.
        
        Args:
            force_complete: If True, mark this as a completed phrase
        """
        if not self._phrase_bytes or not self._engine:
            return
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(self._phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            start_time = time.time()
            segments = self._engine.transcribe_chunk(audio_np)
            transcribe_time = time.time() - start_time
            
            if segments:
                # Combine all segments
                full_text = " ".join([seg.text for seg in segments]).strip()
                
                # Calculate average confidence
                avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
                
                # Calculate timing
                elapsed = (datetime.utcnow() - self._recording_start_time).total_seconds() if self._recording_start_time else 0
                chunk_duration = len(audio_np) / 16000
                
                result = PhraseResult(
                    text=full_text,
                    confidence=int(avg_confidence),
                    start_time=elapsed - chunk_duration,
                    end_time=elapsed,
                    is_complete=force_complete
                )
                
                # Queue for UI
                self._result_queue.put(result)
                
                # Callback
                if self.on_result:
                    self.on_result(result)
                
                print(f"Transcribed ({transcribe_time:.2f}s): '{full_text[:50]}...' [conf: {result.confidence}]")
                
        except Exception as e:
            print(f"Transcription error: {e}")
    
    def get_results(self) -> List[PhraseResult]:
        """Get all pending results (non-blocking)."""
        results = []
        try:
            while True:
                result = self._result_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        return results


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = AccumulatingTranscriptionProcessor(
        model_size="tiny",
        record_timeout=2.0,
        phrase_timeout=3.0
    )
    
    # Set up result handler
    def on_phrase(result: PhraseResult):
        status = "✓ Complete" if result.is_complete else "→ Continuing"
        print(f"{status}: {result.text}")
    
    processor.on_result = on_phrase
    
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
