"""Fake audio module for testing without real audio devices.

Provides file-driven audio source that emits PCM frames for deterministic tests.
Includes confidence variation support and ground truth tracking for accuracy testing.
"""

import wave
import numpy as np
import queue
import threading
from typing import Optional, Dict, Any, List, Tuple
import time
import math
from dataclasses import dataclass, field


@dataclass
class GroundTruth:
    """Ground truth for accuracy measurement of test audio."""
    text: str  # Expected transcription text
    words: List[str] = field(default_factory=list)  # Individual words for WER calculation
    confidence_range: Tuple[float, float] = (0.0, 100.0)  # Expected confidence range
    duration: float = 0.0  # Expected duration in seconds
    
    def __post_init__(self):
        if not self.words and self.text:
            self.words = self.text.split()


@dataclass
class TestAudioPattern:
    """Configuration for a test audio pattern with specific characteristics."""
    name: str  # Pattern identifier
    description: str  # Human-readable description
    confidence_level: str  # "high", "medium", "low", "mixed", "varying"
    duration_seconds: float = 5.0  # Pattern duration
    complexity: str = "medium"  # "simple", "medium", "complex"
    ground_truth: Optional[GroundTruth] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestAudioConfig:
    """Configuration for test audio generation."""
    patterns: List[TestAudioPattern] = field(default_factory=list)
    loop_patterns: bool = True  # Cycle through patterns when testing
    randomize_order: bool = False  # Shuffle pattern order
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeAudioModule:
    """
    Fake audio source for testing that reads WAV files and emits PCM frames.
    
    This provides a deterministic audio source for automated testing without
    requiring real audio hardware. It mimics the API of SoundDeviceSource.
    
    Testing Features:
    - Confidence-based audio generation with varying levels
    - Test audio with known ground truth for accuracy measurement
    - Configurable test audio patterns for different scenarios
    - Metadata for validation and WER calculation
    """
    
    def __init__(
        self,
        wav_path: str,
        blocksize: int = 1024,
        queue_size: int = 10,
        loop: bool = False,
        noise_level: float = 0.0,  # 0.0 = no noise, 0.5 = moderate noise, 1.0 = high noise
        # Enhancement testing parameters
        test_config: Optional[TestAudioConfig] = None,
        confidence_variation: bool = False,  # Enable confidence variation mode
        confidence_pattern: str = "uniform",  # "uniform", "sine", "step", "random"
        confidence_min: float = 0.3,  # Minimum confidence (0.0-1.0)
        confidence_max: float = 0.9,  # Maximum confidence (0.0-1.0)
    ):
        """
        Initialize fake audio source from a WAV file.
        
        Args:
            wav_path: Path to WAV file (mono/stereo int16 PCM)
            blocksize: Number of frames per block
            queue_size: Maximum size of internal queue
            loop: Whether to loop the file when it ends
            noise_level: Amount of noise to add (0.0-1.0). Higher = lower confidence.
            test_config: Optional TestAudioConfig for testing
            confidence_variation: Enable confidence variation for testing
            confidence_pattern: Pattern for confidence variation
            confidence_min: Minimum confidence level for variation
            confidence_max: Maximum confidence level for variation
        """
        self.wav_path = wav_path
        self.blocksize = blocksize
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop = loop
        self._noise_level = noise_level
        
        # Enhancement testing state
        self._test_config = test_config
        self._confidence_variation = confidence_variation
        self._confidence_pattern = confidence_pattern
        self._confidence_min = confidence_min
        self._confidence_max = confidence_max
        self._current_pattern_index = 0
        self._frame_count = 0
        self._ground_truth_segments: List[Dict[str, Any]] = []
        self._current_ground_truth: Optional[GroundTruth] = None
        
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
        
        # Initialize test patterns if config provided
        if self._test_config and self._test_config.patterns:
            self._initialize_test_patterns()
    
    def _initialize_test_patterns(self) -> None:
        """Initialize test patterns with ground truth for accuracy validation."""
        if not self._test_config:
            return
        
        import random
        patterns = self._test_config.patterns.copy()
        
        if self._test_config.randomize_order:
            random.shuffle(patterns)
        
        self._test_config.patterns = patterns
        self._current_pattern_index = 0
        
        # Set first ground truth if available
        if patterns:
            self._current_ground_truth = patterns[0].ground_truth
    
    def _get_current_confidence(self, frame_offset: int = 0) -> float:
        """
        Calculate current confidence level based on pattern and position.
        
        Args:
            frame_offset: Offset from current frame for calculation
            
        Returns:
            float: Confidence level (0.0-1.0)
        """
        if not self._confidence_variation:
            # Map noise level to confidence (higher noise = lower confidence)
            return max(0.1, 1.0 - self._noise_level)
        
        position = (self._frame_count + frame_offset) / self._samplerate  # in seconds
        
        if self._confidence_pattern == "uniform":
            # Constant confidence
            return (self._confidence_min + self._confidence_max) / 2
        
        elif self._confidence_pattern == "sine":
            # Sine wave variation
            period = 5.0  # 5 second period
            phase = (position / period) * 2 * math.pi
            amplitude = (self._confidence_max - self._confidence_min) / 2
            center = (self._confidence_max + self._confidence_min) / 2
            return center + amplitude * math.sin(phase)
        
        elif self._confidence_pattern == "step":
            # Step function (changes every 2 seconds)
            step_period = 2.0
            step_index = int(position / step_period)
            return self._confidence_min if step_index % 2 == 0 else self._confidence_max
        
        elif self._confidence_pattern == "random":
            # Random confidence within range
            import random
            return random.uniform(self._confidence_min, self._confidence_max)
        
        else:
            return (self._confidence_min + self._confidence_max) / 2
    
    def _apply_confidence_based_modification(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply audio modifications based on current confidence level.
        Lower confidence = more noise, distortion, or other artifacts.
        
        Args:
            audio_data: Input audio data (float32)
            
        Returns:
            np.ndarray: Modified audio data
        """
        if not self._confidence_variation:
            # Use legacy noise-based approach
            if self._noise_level > 0:
                noise = np.random.normal(0, self._noise_level * 0.1, audio_data.shape).astype(np.float32)
                audio_data = audio_data + noise
                audio_data = np.clip(audio_data, -1.0, 1.0)
            return audio_data
        
        confidence = self._get_current_confidence()
        
        # Calculate degradation factor (1.0 = perfect, 0.0 = worst)
        degradation = 1.0 - (confidence * 0.8)  # Max 80% degradation
        
        # Add noise scaled by degradation
        if degradation > 0.1:
            noise = np.random.normal(0, degradation * 0.2, audio_data.shape).astype(np.float32)
            audio_data = audio_data + noise
        
        # Add slight distortion for very low confidence
        if confidence < 0.4:
            # Clipping distortion
            clip_threshold = 0.7 + (confidence * 0.3)
            audio_data = np.clip(audio_data, -clip_threshold, clip_threshold)
        
        # Final clip
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def _record_ground_truth_segment(
        self,
        start_time: float,
        end_time: float,
        text: str,
        confidence: float
    ) -> None:
        """
        Record a ground truth segment for accuracy validation.
        
        Args:
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            text: Expected transcription text
            confidence: Expected confidence score
        """
        segment = {
            'start': start_time,
            'end': end_time,
            'text': text,
            'confidence': confidence * 100,  # Convert to percentage
            'words': text.split() if text else []
        }
        self._ground_truth_segments.append(segment)
    
    def get_ground_truth(self) -> List[Dict[str, Any]]:
        """
        Get recorded ground truth segments for accuracy validation.
        
        Returns:
            List[Dict[str, Any]]: List of ground truth segments
        """
        return self._ground_truth_segments.copy()
    
    def clear_ground_truth(self) -> None:
        """Clear recorded ground truth segments."""
        self._ground_truth_segments.clear()
    
    def get_test_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for test validation and reporting.
        
        Returns:
            Dict[str, Any]: Test metadata including configuration and ground truth
        """
        return {
            'test_config': {
                'confidence_variation': self._confidence_variation,
                'confidence_pattern': self._confidence_pattern,
                'confidence_min': self._confidence_min,
                'confidence_max': self._confidence_max,
                'has_test_patterns': self._test_config is not None,
                'pattern_count': len(self._test_config.patterns) if self._test_config else 0,
            },
            'ground_truth_segments': len(self._ground_truth_segments),
            'current_ground_truth': self._current_ground_truth,
            'frame_count': self._frame_count,
        }
    
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
                            self._frame_count = 0  # Reset frame count on loop
                            continue
                        else:
                            break
                    
                    # Convert to numpy array (int16 -> float32)
                    n_frames_read = len(raw_data) // (self._sampwidth * self._channels)
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    audio_data = audio_data.reshape(-1, self._channels)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Apply confidence-based modifications for testing
                    audio_data = self._apply_confidence_based_modification(audio_data)
                    
                    # Update frame counter for confidence calculation
                    self._frame_count += n_frames_read
                    
                    # Record ground truth if testing with patterns
                    if self._current_ground_truth and self._confidence_variation:
                        current_time = self._frame_count / self._samplerate
                        # Record periodically (every ~1 second)
                        if int(current_time) > len(self._ground_truth_segments):
                            self._record_ground_truth_segment(
                                start_time=current_time - 1.0,
                                end_time=current_time,
                                text=self._current_ground_truth.text,
                                confidence=self._get_current_confidence()
                            )
                    
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
        metadata = {
            'sample_rate': self._samplerate,
            'channels': self._channels,
            'dtype': 'float32',
            'source': 'fake',
            'wav_path': self.wav_path,
            'duration': self._duration,
            'total_frames': self._nframes,
            'noise_level': self._noise_level,
            # Confidence testing metadata
            'confidence_variation': self._confidence_variation,
            'confidence_pattern': self._confidence_pattern,
            'has_ground_truth': len(self._ground_truth_segments) > 0,
            'ground_truth_count': len(self._ground_truth_segments),
        }
        
        # Add test-specific metadata if available
        if self._test_config:
            metadata['test_config'] = True
            metadata['test_patterns'] = len(self._test_config.patterns)
        
        return metadata


# Compatibility alias for older call sites
FakeAudioSource = FakeAudioModule
