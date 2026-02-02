"""Audio session manager for recording from multiple sources.

Provides a high-level API for starting/stopping recording sessions that can
capture from microphone, system audio, or both simultaneously. Handles
resampling, mixing, and streaming to disk.

Example:
    # Single source recording
    config = SessionConfig(
        sources=[SourceConfig(type='mic')],
        output_dir=Path('/tmp/test'),
    )
    session = AudioSession()
    session.start(config)
    # ... wait for recording duration ...
    wav_path = session.stop()
    print(f"Saved to: {wav_path}")

    # Dual source recording (mic + system)
    config = SessionConfig(
        sources=[
            SourceConfig(type='mic', gain=1.0),
            SourceConfig(type='system', gain=0.8),
        ],
    )
    session = AudioSession()
    session.start(config)
    wav_path = session.stop()
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import soxr
import queue

from metamemory.audio.storage import (
    PcmPartWriter,
    finalize_part_to_wav,
    finalize_stem,
    new_recording_stem,
    get_recordings_dir,
    get_wav_filename,
)
from metamemory.audio.capture import (
    MicSource,
    SystemSource,
    FakeAudioModule,
    AudioSourceError,
)


class SessionState(Enum):
    """Recording session states."""
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    STOPPING = auto()
    FINALIZED = auto()
    ERROR = auto()


class SessionError(Exception):
    """Base exception for session errors."""
    pass


class NoSourcesError(SessionError):
    """Raised when no valid sources are configured."""
    pass


@dataclass
class SourceConfig:
    """Configuration for a single audio source in a session.

    Attributes:
        type: Source type - 'mic', 'system', or 'fake'
        device_id: Optional device ID (None for auto-select)
        gain: Gain multiplier (1.0 = unity, 0.5 = half, 2.0 = double)
        fake_path: Path to WAV file (only for type='fake')
        loop: Whether to loop fake audio source (only for type='fake', default: False)
    """
    type: str  # 'mic', 'system', 'fake'
    device_id: Optional[int] = None
    gain: float = 1.0
    fake_path: Optional[str] = None
    loop: bool = False


@dataclass
class SessionConfig:
    """Configuration for a recording session.

    Attributes:
        sources: List of source configurations to record from
        output_dir: Optional override for output directory
        sample_rate: Target sample rate in Hz (default: 16000)
        channels: Target channel count (default: 1 for mono)
        max_frames: Optional hard cap on frames to write to disk. Once this
            many frames are recorded, the consumer continues consuming frames
            but discards them (does not write). This ensures deterministic
            bounded recordings even if sources emit faster than real-time.
            Calculated as: int(round(seconds * sample_rate))
        on_audio_frame: Optional callback for mixed audio frames (float32).
            Called from the consumer thread with each mixed audio chunk.
    """
    sources: List[SourceConfig] = field(default_factory=list)
    output_dir: Optional[Path] = None
    sample_rate: int = 16000
    channels: int = 1
    max_frames: Optional[int] = None
    on_audio_frame: Optional[Callable[[np.ndarray], None]] = None


@dataclass
class SessionStats:
    """Statistics from a recording session.
    
    Attributes:
        frames_recorded: Total frames written to disk
        frames_dropped: Frames dropped due to queue overflow
        duration_seconds: Actual recording duration
        source_stats: Per-source statistics
    """
    frames_recorded: int = 0
    frames_dropped: int = 0
    duration_seconds: float = 0.0
    source_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class AudioSourceWrapper:
    """Wraps an audio source and handles resampling/mixing."""
    
    def __init__(
        self,
        source: Any,
        config: SourceConfig,
        target_rate: int = 16000,
        target_channels: int = 1,
    ):
        self.source = source
        self.config = config
        self.target_rate = target_rate
        self.target_channels = target_channels
        self.frames_dropped = 0
        
        # Get source metadata
        metadata = source.get_metadata()
        self.source_rate = metadata.get('sample_rate', 48000)
        self.source_channels = metadata.get('channels', 2)
        
        # Create resampler if needed
        if self.source_rate != self.target_rate:
            self._resampler = soxr.ResampleStream(
                in_rate=self.source_rate,
                out_rate=self.target_rate,
                num_channels=target_channels,
                dtype='float32',
            )
        else:
            self._resampler = None
    
    def read_and_process(self, timeout: Optional[float] = 0.1) -> Optional[np.ndarray]:
        """Read frames from source and process them.
        
        Returns resampled mono float32 array, or None if no frames available.
        """
        frames = self.source.read_frames(timeout=timeout)
        if frames is None:
            return None
        
        # Apply gain
        if self.config.gain != 1.0:
            frames = frames * self.config.gain
        
        # Downmix to mono if needed
        if frames.ndim > 1 and frames.shape[1] > 1 and self.target_channels == 1:
            # Average channels: stereo -> mono
            frames = frames.mean(axis=1, keepdims=True)
        elif frames.ndim == 1 and self.target_channels == 1:
            # Already mono, reshape to column vector
            frames = frames.reshape(-1, 1)
        
        # Resample if needed
        if self._resampler is not None:
            # soxr expects (samples, channels) shape
            if frames.ndim == 1:
                frames = frames.reshape(-1, 1)
            # Use resample_chunk for streaming resampler
            frames = self._resampler.resample_chunk(frames)
        
        return frames
    
    def start(self) -> None:
        """Start the underlying source."""
        self.source.start()
    
    def stop(self) -> None:
        """Stop the underlying source."""
        self.source.stop()
    
    def is_running(self) -> bool:
        """Check if source is running."""
        return self.source.is_running()


class AudioSession:
    """Manages a recording session from one or more audio sources.
    
    This is the main API for recording audio. It handles:
    - Starting/stopping multiple sources
    - Resampling to target rate (default 16kHz)
    - Mixing multiple sources together
    - Converting to int16 and streaming to disk
    - Finalizing to WAV format
    
    Thread-safety: This class is designed to be used from a single thread.
    The internal consumer thread handles all source reading and disk writes.
    
    Example:
        session = AudioSession()
        config = SessionConfig(sources=[SourceConfig(type='mic')])
        session.start(config)
        time.sleep(5)
        wav_path = session.stop()
    """
    
    def __init__(self):
        self._state = SessionState.IDLE
        self._config: Optional[SessionConfig] = None
        self._sources: List[AudioSourceWrapper] = []
        self._writer: Optional[PcmPartWriter] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stem: Optional[str] = None
        self._start_time: Optional[float] = None
        self._stats = SessionStats()
        self._error: Optional[Exception] = None
    
    def start(self, config: SessionConfig) -> None:
        """Start a recording session.
        
        Args:
            config: Session configuration including sources and settings
        
        Raises:
            SessionError: If session is already active or no valid sources
            AudioSourceError: If a source fails to initialize
        """
        if self._state not in (SessionState.IDLE, SessionState.ERROR, SessionState.FINALIZED):
            raise SessionError(f"Cannot start from state {self._state.name}")
        
        if not config.sources:
            raise NoSourcesError("At least one source must be configured")
        
        self._config = config
        self._state = SessionState.STARTING
        self._stats = SessionStats()
        self._error = None
        
        try:
            # Create sources
            self._sources = self._create_sources(config)
            
            # Create writer
            self._stem = new_recording_stem()
            self._writer = PcmPartWriter.create(
                stem=self._stem,
                sample_rate=config.sample_rate,
                channels=config.channels,
                sample_width_bytes=2,
                recordings_dir=config.output_dir,
            )
            
            # Start all sources
            for wrapper in self._sources:
                wrapper.start()
            
            # Start consumer thread
            self._stop_event.clear()
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                daemon=True,
            )
            self._consumer_thread.start()
            
            self._start_time = time.time()
            self._state = SessionState.RECORDING
            
        except Exception as e:
            self._state = SessionState.ERROR
            self._error = e
            self._cleanup()
            raise
    
    def stop(self) -> Path:
        """Stop the recording session and finalize to WAV.
        
        Returns:
            Path to the finalized WAV file
        
        Raises:
            SessionError: If session is not recording
        """
        if self._state != SessionState.RECORDING:
            raise SessionError(f"Cannot stop from state {self._state.name}")
        
        self._state = SessionState.STOPPING
        self._stop_event.set()

        # Stop all sources first (prevents new frames from being added)
        for wrapper in self._sources:
            wrapper.stop()

        # Wait for consumer thread to finish (drains existing frames)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=5.0)

        # Calculate final stats
        if self._start_time:
            self._stats.duration_seconds = time.time() - self._start_time

        # Close writer
        if self._writer:
            self._writer.close()
        
        # Finalize to WAV
        if not self._stem:
            raise SessionError("No stem available for finalization")
        
        output_dir = self._config.output_dir if self._config else None
        wav_path = finalize_stem(
            stem=self._stem,
            recordings_dir=output_dir or get_recordings_dir(),
        )
        
        self._state = SessionState.FINALIZED
        
        return wav_path
    
    def get_state(self) -> SessionState:
        """Get current session state."""
        return self._state
    
    def get_stats(self) -> SessionStats:
        """Get current recording statistics."""
        return self._stats
    
    def _create_sources(self, config: SessionConfig) -> List[AudioSourceWrapper]:
        """Create source wrappers from configuration."""
        wrappers = []
        
        for source_config in config.sources:
            if source_config.type == 'mic':
                source = MicSource(
                    device_id=source_config.device_id,
                    blocksize=1024,
                    queue_size=10,
                )
            elif source_config.type == 'system':
                source = SystemSource(
                    device_id=source_config.device_id,
                    blocksize=1024,
                    queue_size=10,
                )
            elif source_config.type == 'fake':
                if not source_config.fake_path:
                    raise SessionError("fake_path required for type='fake'")
                source = FakeAudioModule(
                    wav_path=source_config.fake_path,
                    blocksize=1024,
                    queue_size=10,
                    loop=source_config.loop,
                )
            else:
                raise SessionError(f"Unknown source type: {source_config.type}")
            
            wrapper = AudioSourceWrapper(
                source=source,
                config=source_config,
                target_rate=config.sample_rate,
                target_channels=config.channels,
            )
            wrappers.append(wrapper)
        
        return wrappers
    
    def _consumer_loop(self) -> None:
        """Background thread that reads from sources and writes to disk."""
        discard_mode = False
        max_frames = self._config.max_frames if self._config else None

        while not self._stop_event.is_set():
            # Check writer is available
            if not self._writer:
                break

            # Read from all sources
            frames_list = []
            for wrapper in self._sources:
                frames = wrapper.read_and_process(timeout=0.05)
                if frames is not None:
                    frames_list.append(frames)

            if not frames_list:
                # No frames available, sleep briefly
                time.sleep(0.01)
                continue

            # Mix frames together
            mixed = self._mix_frames(frames_list)

            # Feed to transcription callback (float32 audio before int16 conversion)
            # Flatten to 1D array as transcription buffer expects (n_samples,)
            if self._config and self._config.on_audio_frame:
                audio_for_transcription = mixed.flatten() if mixed.ndim > 1 else mixed
                self._config.on_audio_frame(audio_for_transcription)

            # Check max_frames cap
            if max_frames is not None and not discard_mode:
                remaining = max_frames - self._stats.frames_recorded
                if remaining <= 0:
                    # Cap reached, switch to discard mode
                    discard_mode = True
                elif len(mixed) > remaining:
                    # Partial chunk would exceed cap - write only remaining frames
                    mixed = mixed[:remaining]
                    int16_bytes = self._float32_to_int16_bytes(mixed)
                    self._writer.write_frames_i16(int16_bytes)
                    self._stats.frames_recorded += len(mixed)
                    # Switch to discard mode after final write
                    discard_mode = True
                    continue

            if discard_mode:
                # In discard mode: consume frames but don't write
                continue

            # Convert to int16 and write
            int16_bytes = self._float32_to_int16_bytes(mixed)
            self._writer.write_frames_i16(int16_bytes)
            self._stats.frames_recorded += len(mixed)

        # Drain remaining frames (respecting max_frames cap)
        for _ in range(50):  # Brief drain period
            if not self._writer:
                break

            # Check if we've already hit the cap
            if max_frames is not None and self._stats.frames_recorded >= max_frames:
                # Consume but discard remaining frames to prevent queue blocking
                for wrapper in self._sources:
                    wrapper.read_and_process(timeout=0.01)
                continue

            frames_list = []
            for wrapper in self._sources:
                frames = wrapper.read_and_process(timeout=0.01)
                if frames is not None:
                    frames_list.append(frames)

            if not frames_list:
                break

            mixed = self._mix_frames(frames_list)

            # Check max_frames cap during drain
            if max_frames is not None:
                remaining = max_frames - self._stats.frames_recorded
                if remaining <= 0:
                    # Cap already reached, consume but don't write
                    continue
                elif len(mixed) > remaining:
                    # Partial chunk - write only remaining frames
                    mixed = mixed[:remaining]

            int16_bytes = self._float32_to_int16_bytes(mixed)
            self._writer.write_frames_i16(int16_bytes)
            self._stats.frames_recorded += len(mixed)
    
    def _mix_frames(self, frames_list: List[np.ndarray]) -> np.ndarray:
        """Mix multiple frame arrays together.
        
        All frames must be the same shape. Returns the sum, clipped to [-1, 1].
        """
        if len(frames_list) == 1:
            return np.clip(frames_list[0], -1.0, 1.0)
        
        # Find minimum length
        min_len = min(f.shape[0] for f in frames_list)
        
        # Trim all to same length
        trimmed = [f[:min_len] for f in frames_list]
        
        # Sum and clip
        mixed = np.sum(trimmed, axis=0)
        mixed = np.clip(mixed, -1.0, 1.0)
        
        return mixed
    
    def _float32_to_int16_bytes(self, frames: np.ndarray) -> bytes:
        """Convert float32 array to little-endian int16 bytes."""
        # Scale from [-1, 1] to int16 range
        int16_array = (frames * 32767.0).astype(np.int16)
        return int16_array.tobytes()
    
    def _cleanup(self) -> None:
        """Clean up resources after error."""
        for wrapper in self._sources:
            try:
                wrapper.stop()
            except Exception:
                pass
        
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                pass
