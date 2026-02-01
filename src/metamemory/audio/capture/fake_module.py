"""Fake audio module for testing without real audio devices.

Provides file-driven audio source that emits PCM frames for deterministic tests.
"""

import wave
import numpy as np
import queue
import threading
from typing import Optional, Dict, Any
import time


class FakeAudioModule:
    """
    Fake audio source for testing that reads WAV files and emits PCM frames.
    
    This provides a deterministic audio source for automated testing without
    requiring real audio hardware. It mimics the API of SoundDeviceSource.
    """
    
    def __init__(
        self,
        wav_path: str,
        blocksize: int = 1024,
        queue_size: int = 10,
        loop: bool = False,
    ):
        """
        Initialize fake audio source from a WAV file.
        
        Args:
            wav_path: Path to WAV file (mono/stereo int16 PCM)
            blocksize: Number of frames per block
            queue_size: Maximum size of internal queue
            loop: Whether to loop the file when it ends
        """
        self.wav_path = wav_path
        self.blocksize = blocksize
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop = loop
        
        # Read WAV file metadata
        with wave.open(wav_path, 'rb') as wf:
            self._channels = wf.getnchannels()
            self._samplerate = wf.getframerate()
            self._sampwidth = wf.getsampwidth()
            self._nframes = wf.getnframes()
            
        # Validate format
        if self._sampwidth != 2:
            raise ValueError(f"FakeAudioModule only supports 16-bit PCM, got {self._sampwidth * 8}-bit")
        
        # Calculate duration
        self._duration = self._nframes / self._samplerate
    
    def _read_loop(self) -> None:
        """Background thread that reads WAV and pushes frames to queue."""
        while self._running:
            with wave.open(self.wav_path, 'rb') as wf:
                while self._running:
                    # Read blocksize frames
                    frames_to_read = self.blocksize
                    raw_data = wf.readframes(frames_to_read)
                    
                    if not raw_data:
                        # End of file
                        if self._loop:
                            wf.rewind()
                            continue
                        else:
                            break
                    
                    # Convert to numpy array (int16 -> float32)
                    n_frames_read = len(raw_data) // (self._sampwidth * self._channels)
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    audio_data = audio_data.reshape(-1, self._channels)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Push to queue (block if full to simulate real-time)
                    try:
                        self._queue.put(audio_data, timeout=1.0)
                    except queue.Full:
                        # If queue is full and we're stopping, exit
                        if not self._running:
                            break
            
            if not self._loop:
                break
    
    def start(self) -> None:
        """Start emitting audio frames."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
    
    def stop(self) -> None:
        """Stop emitting audio frames."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._thread:
                self._thread.join(timeout=2.0)
                self._thread = None
            
            # Clear the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
    
    def read_frames(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read audio frames from the queue.
        
        Args:
            timeout: Maximum time to wait for frames (None = block forever)
        
        Returns:
            Numpy array of audio frames, or None if timeout/stopped
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_running(self) -> bool:
        """Check if the source is currently emitting."""
        with self._lock:
            return self._running
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return source metadata (sample_rate, channels, etc.)."""
        return {
            'sample_rate': self._samplerate,
            'channels': self._channels,
            'dtype': 'float32',
            'source': 'fake',
            'wav_path': self.wav_path,
            'duration': self._duration,
            'total_frames': self._nframes,
        }


# Compatibility alias for older call sites
FakeAudioSource = FakeAudioModule
