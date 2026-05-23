"""Audio ring buffer for efficient audio chunk management.

Provides thread-safe audio buffering with automatic trimming and timestamp tracking.
Designed for real-time transcription pipelines.
"""

import threading
import numpy as np


class AudioRingBuffer:
    """Fixed-size ring buffer for audio chunks with automatic trimming.
    
    Stores audio as float32 mono at 16kHz (Whisper requirement).
    Thread-safe for concurrent access from audio capture thread.
    
    Example:
        buffer = AudioRingBuffer(max_seconds=30)
        buffer.append(audio_chunk)  # Add audio
        recent = buffer.get_recent(5.0)  # Get last 5 seconds
        buffer.trim_committed(16000)  # Remove 1 second of processed audio
    """
    
    def __init__(self, max_seconds: int = 30, sample_rate: int = 16000):
        """Initialize the audio ring buffer.
        
        Args:
            max_seconds: Maximum seconds of audio to retain
            sample_rate: Sample rate in Hz (default 16000 for Whisper)
        """
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_samples = max_seconds * sample_rate
        
        # Audio storage as float32 array
        self._buffer = np.array([], dtype=np.float32)
        
        # Track total samples seen for timestamp calculation
        self._total_samples_seen = 0
        
        # Thread safety
        self._lock = threading.Lock()
    
    def append(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer, trimming old audio if exceeds max.
        
        Args:
            chunk: Audio samples as float32 array (mono)
        """
        with self._lock:
            # Concatenate new audio
            self._buffer = np.concatenate([self._buffer, chunk])
            self._total_samples_seen += len(chunk)
            
            # Trim from left if exceeds max size (keep most recent)
            if len(self._buffer) > self.max_samples:
                self._buffer = self._buffer[-self.max_samples:]
    
    def get_recent(self, seconds: float) -> np.ndarray:
        """Get the most recent N seconds of audio.
        
        Args:
            seconds: Number of seconds to retrieve
            
        Returns:
            Numpy array of audio samples (may be shorter if buffer doesn't have enough)
        """
        with self._lock:
            samples = int(seconds * self.sample_rate)
            if len(self._buffer) >= samples:
                return self._buffer[-samples:].copy()
            return self._buffer.copy()
    
    def get_samples(self, n_samples: int) -> np.ndarray:
        """Get exact number of samples from the end of buffer.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            Numpy array of audio samples (may be shorter if buffer doesn't have enough)
        """
        with self._lock:
            if len(self._buffer) >= n_samples:
                return self._buffer[-n_samples:].copy()
            return self._buffer.copy()
    
    def trim_committed(self, committed_samples: int) -> None:
        """Remove audio that has been committed/processed from the buffer.
        
        Args:
            committed_samples: Number of samples to remove from the start
        """
        with self._lock:
            if committed_samples > 0 and len(self._buffer) > 0:
                self._buffer = self._buffer[committed_samples:]
    
    def get_total_duration(self) -> float:
        """Get total duration of audio currently in buffer.
        
        Returns:
            Duration in seconds
        """
        with self._lock:
            return len(self._buffer) / self.sample_rate
    
    def get_total_samples_seen(self) -> int:
        """Get total number of samples seen since initialization.
        
        Returns:
            Total sample count (useful for timestamp tracking)
        """
        with self._lock:
            return self._total_samples_seen
    
    def is_empty(self) -> bool:
        """Check if buffer is empty.
        
        Returns:
            True if buffer contains no audio
        """
        with self._lock:
            return len(self._buffer) == 0
    
    def clear(self) -> None:
        """Clear all audio from buffer."""
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._total_samples_seen = 0
