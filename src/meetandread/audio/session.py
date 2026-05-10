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

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import soxr
import queue

from meetandread.audio.storage import (
    PcmPartWriter,
    finalize_part_to_wav,
    finalize_stem,
    new_recording_stem,
    get_recordings_dir,
    get_wav_filename,
)
from meetandread.audio.capture import (
    MicSource,
    SystemSource,
    FakeAudioModule,
    AudioSourceError,
)
from meetandread.audio.denoising import (
    DenoisingProvider,
    DenoisingResult,
    create_provider,
)

_log = logging.getLogger(__name__)


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
        denoise: Per-source denoising override. None means "denoise only real mic
            sources" (i.e. type='mic'). True forces denoising on (for test fake
            sources simulating mic). False forces denoising off.
    """
    type: str  # 'mic', 'system', 'fake'
    device_id: Optional[int] = None
    gain: float = 1.0
    fake_path: Optional[str] = None
    loop: bool = False
    denoise: Optional[bool] = None


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
        on_frames_dropped: Optional callback invoked when frames are dropped
            due to queue overflow. Receives the aggregate frames_dropped count.
            Called from the audio callback thread — must be non-blocking.
        enable_microphone_denoising: Whether to denoise mic-like sources.
        denoising_provider_name: Provider name (e.g. 'spectral_gate').
        denoising_latency_budget_ms: Per-chunk latency budget in ms.
        denoising_provider_factory: Optional callable returning a
            DenoisingProvider. Used by tests to inject a mock/broken provider.
            If None, create_provider() is used.
    """
    sources: List[SourceConfig] = field(default_factory=list)
    output_dir: Optional[Path] = None
    sample_rate: int = 16000
    channels: int = 1
    max_frames: Optional[int] = None
    on_audio_frame: Optional[Callable[[np.ndarray], None]] = None
    on_frames_dropped: Optional[Callable[[int], None]] = None
    enable_microphone_denoising: bool = False
    denoising_provider_name: Optional[str] = None
    denoising_latency_budget_ms: float = 200.0
    denoising_provider_factory: Optional[Callable[[], DenoisingProvider]] = None


@dataclass
class DenoisingStats:
    """Per-session denoising diagnostics.

    All fields are sanitized — no raw audio content or secrets.
    """
    provider: str = ""
    enabled: bool = False
    active: bool = False
    fallback: bool = False
    processed_frame_count: int = 0
    fallback_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    budget_exceeded_count: int = 0
    last_error_class: str = ""
    last_error_message: str = ""

    def record_success(self, latency_ms: float, budget_ms: float) -> None:
        """Record a successful denoising pass."""
        self.processed_frame_count += 1
        total_ms = self.avg_latency_ms * (self.processed_frame_count - 1) + latency_ms
        self.avg_latency_ms = total_ms / self.processed_frame_count
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms
        if latency_ms > budget_ms:
            self.budget_exceeded_count += 1

    def record_fallback(self, latency_ms: float, error: Optional[str] = None) -> None:
        """Record a fallback event."""
        self.fallback_count += 1
        self.fallback = True
        if error:
            parts = error.split(": ", 1)
            self.last_error_class = parts[0] if parts else error
            self.last_error_message = parts[1] if len(parts) > 1 else ""
        # Still track latency for fallback frames
        if self.processed_frame_count > 0:
            total_ms = self.avg_latency_ms * self.processed_frame_count + latency_ms
            self.avg_latency_ms = total_ms / (self.processed_frame_count + 1)
        else:
            self.avg_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms


@dataclass
class SessionStats:
    """Statistics from a recording session.
    
    Attributes:
        frames_recorded: Total frames written to disk
        frames_dropped: Frames dropped due to queue overflow
        duration_seconds: Actual recording duration
        source_stats: Per-source statistics
        denoising: Denoising diagnostics (empty when disabled)
    """
    frames_recorded: int = 0
    frames_dropped: int = 0
    duration_seconds: float = 0.0
    source_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    denoising: DenoisingStats = field(default_factory=DenoisingStats)


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

    @property
    def should_denoise(self) -> bool:
        """Whether this source's frames should be denoised.

        Logic: if SourceConfig.denoise is explicitly set, use that.
        Otherwise denoise only real mic sources (type='mic').
        """
        if self.config.denoise is not None:
            return self.config.denoise
        return self.config.type == 'mic'
    
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
            if frames.shape[0] == 0:
                return frames
            # Use resample_chunk for streaming resampler
            try:
                frames = self._resampler.resample_chunk(frames)
            except Exception:
                # soxr can crash on malformed input — log and drop
                return None
        
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
        # Denoising state
        self._denoising_provider: Optional[DenoisingProvider] = None
        self._denoising_disabled: bool = False  # True after init/process failure
        # Lock protecting _stats.frames_dropped increments from audio callbacks
        self._stats_lock = threading.Lock()

    def _on_source_frame_dropped(self, source_type: str, source_count: int) -> None:
        """Thread-safe handler called from audio callback threads on queue overflow.

        Increments SessionStats.frames_dropped and fires the optional
        SessionConfig.on_frames_dropped callback with the aggregate count.
        Never raises — exceptions are logged and swallowed so the audio
        callback thread is never compromised.
        """
        try:
            with self._stats_lock:
                self._stats.frames_dropped += 1
                aggregate = self._stats.frames_dropped
            _log.info(
                "Session frame drop: source=%s, source_count=%d, total=%d",
                source_type,
                source_count,
                aggregate,
            )
            if self._config and self._config.on_frames_dropped:
                try:
                    self._config.on_frames_dropped(aggregate)
                except Exception:
                    _log.exception("on_frames_dropped callback error")
        except Exception:
            # Absolute safety net — never raise into audio callback
            _log.exception("Unexpected error in _on_source_frame_dropped")
    
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
        self._denoising_provider = None
        self._denoising_disabled = False
        
        try:
            # Create sources
            self._sources = self._create_sources(config)

            # Create denoising provider if enabled
            if config.enable_microphone_denoising:
                self._init_denoising_provider(config)
            
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
        import logging
        _log = logging.getLogger(__name__)
        wrappers = []
        
        for source_config in config.sources:
            if source_config.type == 'mic':
                source = MicSource(
                    device_id=source_config.device_id,
                    blocksize=1024,
                    queue_size=10,
                    on_frame_dropped=self._on_source_frame_dropped,
                )
            elif source_config.type == 'system':
                source = SystemSource(
                    device_id=source_config.device_id,
                    blocksize=1024,
                    queue_size=10,
                    on_frame_dropped=self._on_source_frame_dropped,
                )
                if not source.available:
                    _log.warning(
                        "SystemSource unavailable — skipping system audio. "
                        "Recording will continue with remaining sources.",
                    )
                    continue  # Skip this source, don't add to wrappers
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
        
        if not wrappers:
            raise NoSourcesError(
                "No usable audio sources available. "
                "Ensure at least one source (mic, fake) can be initialized."
            )
        
        return wrappers

    def _init_denoising_provider(self, config: SessionConfig) -> None:
        """Initialize the denoising provider, fail-open on error.

        Creates the provider and updates stats. On failure, logs a sanitized
        warning and sets _denoising_disabled so the consumer loop feeds raw audio.
        """
        try:
            if config.denoising_provider_factory:
                provider = config.denoising_provider_factory()
            else:
                provider = create_provider(config.denoising_provider_name)

            self._denoising_provider = provider
            self._stats.denoising.enabled = True
            self._stats.denoising.active = True
            self._stats.denoising.provider = provider.name

            _log.info(
                "Denoising provider initialized: name=%s, budget_ms=%.1f",
                provider.name,
                config.denoising_latency_budget_ms,
            )
        except Exception as exc:
            self._denoising_disabled = True
            self._stats.denoising.enabled = True
            self._stats.denoising.fallback = True
            self._stats.denoising.last_error_class = type(exc).__name__
            self._stats.denoising.last_error_message = str(exc)[:200]

            _log.warning(
                "Denoising provider init failed, continuing raw: error_class=%s",
                type(exc).__name__,
            )

    def _apply_denoising(
        self,
        frames: np.ndarray,
        wrapper: AudioSourceWrapper,
    ) -> np.ndarray:
        """Apply denoising to a single source's frames before mixing.

        Returns denoised frames or raw frames on fallback. Never raises.
        Updates self._stats.denoising diagnostics.
        """
        # Fast path: not a denoise-enabled source
        if not wrapper.should_denoise:
            return frames

        # Fast path: denoising disabled (init failure or already hard-disabled)
        if self._denoising_disabled or self._denoising_provider is None:
            return frames

        try:
            # Flatten to 1-D for provider (which expects mono float32)
            flat = frames.flatten().astype(np.float32)
            result: DenoisingResult = self._denoising_provider.process(flat)

            if result.fallback:
                self._stats.denoising.record_fallback(result.latency_ms, result.error)
            else:
                self._stats.denoising.record_success(
                    result.latency_ms, self._config.denoising_latency_budget_ms
                )

            # Budget warning (not a hard failure)
            if result.latency_ms > self._config.denoising_latency_budget_ms:
                _log.info(
                    "Denoising latency exceeded budget: %.1fms > %.1fms",
                    result.latency_ms,
                    self._config.denoising_latency_budget_ms,
                )

            # Validate output shape
            output = result.audio
            if output.shape != flat.shape:
                _log.warning(
                    "Denoising output shape mismatch: expected %s got %s, using raw",
                    flat.shape,
                    output.shape,
                )
                self._stats.denoising.record_fallback(
                    result.latency_ms, "OutputShapeMismatch"
                )
                return frames

            # Reshape back to match input ndim
            if frames.ndim > 1:
                output = output.reshape(frames.shape)
            return output

        except Exception as exc:
            # Hard-disable on exception — continue raw for rest of session
            self._denoising_disabled = True
            self._stats.denoising.active = False
            self._stats.denoising.record_fallback(0.0, f"{type(exc).__name__}: {exc}")

            _log.warning(
                "Denoising process error, hard-disabling for session: error_class=%s",
                type(exc).__name__,
            )
            return frames
    
    def _consumer_loop(self) -> None:
        """Background thread that reads from sources and writes to disk."""
        discard_mode = False
        max_frames = self._config.max_frames if self._config else None

        while not self._stop_event.is_set():
            # Check writer is available
            if not self._writer:
                break

            # Read from all sources, applying denoising per-source before mixing
            frames_list = []
            for wrapper in self._sources:
                frames = wrapper.read_and_process(timeout=0.05)
                if frames is not None:
                    # Apply denoising to denoise-enabled sources before mixing
                    frames = self._apply_denoising(frames, wrapper)
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
                    frames = self._apply_denoising(frames, wrapper)
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
