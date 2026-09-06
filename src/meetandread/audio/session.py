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
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
import soxr
from meetandread.audio.storage import (
    PcmPartWriter,
    finalize_stem,
    new_recording_stem,
    get_recordings_dir,
)
from meetandread.audio.capture import (
    MicSource,
    SystemSource,
    FakeAudioModule,
)
from meetandread.audio.denoising import (
    DenoisingProvider,
    DenoisingResult,
    create_provider,
)

_log = logging.getLogger(__name__)

# Larger capture blocks reduce audio callback pressure under transcription load.
DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE = 4096

# Seconds without delivery after which a multi-source source is treated as
# stalled and its timeline span is zero-filled to keep mixing aligned.
DEFAULT_MIX_STALL_TIMEOUT_S = 0.5


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
        microphone_denoising_auto_disable_on_frame_drops: Whether frame-drop
            thresholds can fail open to raw mic audio for the rest of the session.
        mix_stall_timeout_s: Seconds without delivery after which a source in a
            multi-source mix is treated as stalled and zero-filled to keep the
            shared mixing timeline aligned (default: 0.5).
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
    microphone_denoising_auto_disable_on_frame_drops: bool = True
    mix_stall_timeout_s: float = DEFAULT_MIX_STALL_TIMEOUT_S
    on_error: Optional[Callable[[Exception], None]] = None


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
    disabled_reason: str = ""
    disabled_at: float = 0.0
    disabled_count: int = 0
    auto_disable_on_frame_drops: bool = True

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
        drop_rate: Aggregate dropped-callback rate across active sources
        max_consecutive_frames_dropped: Largest source-level drop burst observed
        consecutive_frames_dropped: Largest currently active source-level drop burst
        capture_block_size: Default block size used for real capture sources
        denoising: Denoising diagnostics (empty when disabled)
        retry_attempts: Number of start-time retry attempts recorded by the controller
        retry_outcome: Sanitized final retry/fallback outcome ("none" before retry)
        failed_sources: Sanitized source types that failed during retry/fallback
        fallback_sources: Sanitized source types used after fallback confirmation
        mix_stall_episodes: Number of stall episodes detected by the aligned
            multi-source mixer
        mix_zero_filled_samples: Samples zero-filled in stalled sources' spans
            by the aligned multi-source mixer
    """
    frames_recorded: int = 0
    frames_dropped: int = 0
    duration_seconds: float = 0.0
    source_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    drop_rate: float = 0.0
    max_consecutive_frames_dropped: int = 0
    consecutive_frames_dropped: int = 0
    capture_block_size: int = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    denoising: DenoisingStats = field(default_factory=DenoisingStats)
    retry_attempts: int = 0
    retry_outcome: str = "none"
    failed_sources: List[str] = field(default_factory=list)
    fallback_sources: List[str] = field(default_factory=list)
    mix_stall_episodes: int = 0
    mix_zero_filled_samples: int = 0


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
            _log.info(
                "AudioSourceWrapper: resampling active: source=%s, native=%dHz/%dch "
                "-> target=%dHz/%dch",
                config.type,
                self.source_rate,
                self.source_channels,
                self.target_rate,
                self.target_channels,
            )
        else:
            self._resampler = None
            _log.info(
                "AudioSourceWrapper: passthrough (no resample): source=%s, "
                "native=%dHz == target=%dHz",
                config.type,
                self.source_rate,
                self.target_rate,
            )

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
                _log.debug(
                    "zero-sample chunk before resampling, dropping: source=%s",
                    self.config.type,
                )
                return None
            # Use resample_chunk for streaming resampler
            try:
                frames = self._resampler.resample_chunk(frames)
            except Exception:
                # soxr can crash on malformed input — log and drop
                _log.warning(
                    "soxr resample_chunk failed (malformed input?), "
                    "dropping frame: error_class=%s",
                    type(frames).__name__ if frames is not None else "NoneType",
                )
                return None

        if frames.shape[0] == 0:
            _log.debug(
                "zero-sample chunk after processing, dropping: source=%s",
                self.config.type,
            )
            return None

        return frames
    
    def start(self) -> None:
        """Start the underlying source."""
        self.source.start()
    
    def stop(self) -> None:
        """Stop the underlying source and clean up resources."""
        self.source.stop()
        self._resampler = None
    
    def is_running(self) -> bool:
        """Check if source is running."""
        return self.source.is_running()


@dataclass
class _SourceMixState:
    """Per-source state for the position-aligned mixer.

    ``carry`` holds processed samples not yet emitted; ``source_pos`` is the
    total samples consumed from this source into carry. The session-scalar
    ``emitted_total`` is the shared timeline position.
    """

    carry: np.ndarray  # (n, 1) float32, non-empty while streaming
    source_pos: int
    last_delivery: float
    stalled: bool = False
    stall_episodes: int = 0
    zero_filled_samples: int = 0


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
        self._denoising_disabled: bool = False  # True after init/process failure or policy disable
        self._drop_rate_exceeded_since: Optional[float] = None
        # Lock protecting _stats.frames_dropped increments from audio callbacks
        self._stats_lock = threading.Lock()
        # Lock protecting _sources list for atomic swap mid-recording
        self._sources_lock = threading.Lock()
        # Hot-plug carry handoff: per-source-type stash of a departed
        # wrapper's un-emitted mix state, transferred to its replacement
        # on the next _sync_mix_states pass.
        self._mix_carry_handoff: Dict[str, tuple] = {}

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
        self._drop_rate_exceeded_since = None
        self._stats.denoising.auto_disable_on_frame_drops = (
            config.microphone_denoising_auto_disable_on_frame_drops
        )
        
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
            self._start_time = time.time()
            self._state = SessionState.RECORDING
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                daemon=True,
            )
            self._consumer_thread.start()
            
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
        if self._state not in (SessionState.RECORDING, SessionState.ERROR):
            raise SessionError(f"Cannot stop from state {self._state.name}")
        
        self._state = SessionState.STOPPING
        self._stop_event.set()

        # Stop all sources first (prevents new frames from being added)
        with self._sources_lock:
            sources_snapshot = list(self._sources)
        for wrapper in sources_snapshot:
            wrapper.stop()

        # Wait for consumer thread to finish (drains existing frames)
        # If consumer crashed, thread is already dead — join returns quickly
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
        """Get current recording statistics with sanitized source telemetry."""
        self._refresh_drop_stats()
        return self._stats

    def swap_source(self, source_type: str, new_wrapper: AudioSourceWrapper) -> Optional[AudioSourceWrapper]:
        """Atomically swap a source wrapper mid-recording.

        Finds the first wrapper whose config.type matches *source_type*,
        starts the new wrapper *before* stopping the old one, then
        replaces the entry in ``_sources`` — all under ``_sources_lock``.
        If the new wrapper fails to start, the old wrapper is left
        untouched and a :class:`SessionError` is raised.

        Args:
            source_type: Source type string to match (e.g. ``'mic'``).
            new_wrapper: The replacement :class:`AudioSourceWrapper`.

        Returns:
            The old wrapper that was replaced, or ``None`` if no
            wrapper matched *source_type*.

        Raises:
            SessionError: If the session is not in RECORDING state,
                or if the new wrapper fails to start.
        """
        if self._state != SessionState.RECORDING:
            raise SessionError("swap_source requires session must be RECORDING")

        # Find the target wrapper first (read-only, under lock)
        with self._sources_lock:
            old_wrapper = None
            old_index = None
            for i, wrapper in enumerate(self._sources):
                if wrapper.config.type == source_type:
                    old_wrapper = wrapper
                    old_index = i
                    break

        if old_wrapper is None:
            return None

        # Start the new source *before* removing the old one.
        # This ensures the consumer loop never sees a broken wrapper.
        try:
            new_wrapper.start()
        except Exception:
            raise SessionError(f"Failed to start replacement source: {source_type}")

        # Atomically swap under lock: the new wrapper is already running
        # so the consumer loop will immediately pick it up.
        with self._sources_lock:
            self._sources[old_index] = new_wrapper

        # Stop the old wrapper *after* the swap is committed
        old_wrapper.stop()

        return old_wrapper

    def _refresh_drop_stats(self) -> None:
        """Refresh aggregate sanitized frame-drop telemetry from sources.

        Called from the diagnostics path (``get_stats``).  Includes full
        source metadata which may involve blocking device queries on some
        backends — **do not** call this on the per-frame hot path; use
        :meth:`_refresh_drop_counters_only` instead.
        """
        source_stats: Dict[str, Dict[str, Any]] = {}
        total_callbacks = 0
        total_dropped = 0
        max_burst = 0
        current_burst = 0

        with self._sources_lock:
            sources_snapshot = list(self._sources)
        for wrapper in sources_snapshot:
            source = wrapper.source
            metadata = source.get_metadata() if hasattr(source, "get_metadata") else {}
            telemetry = (
                source.get_drop_telemetry()
                if hasattr(source, "get_drop_telemetry")
                else {}
            )
            stats = {**metadata, **telemetry}
            label = wrapper.config.type
            source_stats[label] = stats
            total_callbacks += int(stats.get("total_callbacks", 0) or 0)
            total_dropped += int(stats.get("frames_dropped", 0) or 0)
            max_burst = max(
                max_burst,
                int(stats.get("max_consecutive_frames_dropped", 0) or 0),
            )
            current_burst = max(
                current_burst,
                int(stats.get("consecutive_frames_dropped", 0) or 0),
            )

        with self._stats_lock:
            self._stats.source_stats = source_stats
            # Preserve callback-driven aggregate count if it is ahead of source reads.
            self._stats.frames_dropped = max(self._stats.frames_dropped, total_dropped)
            aggregate_dropped = self._stats.frames_dropped
            self._stats.drop_rate = (
                aggregate_dropped / total_callbacks
                if total_callbacks > 0
                else 0.0
            )
            self._stats.max_consecutive_frames_dropped = max_burst
            self._stats.consecutive_frames_dropped = current_burst
            self._stats.capture_block_size = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE

    def _refresh_drop_counters_only(self) -> None:
        """Lightweight drop-counter refresh for the per-frame hot path.

        Unlike :meth:`_refresh_drop_stats`, this skips ``get_metadata()``
        (which may trigger blocking WASAPI/PyAudio device queries) and
        reads only the in-memory telemetry counters needed for denoise
        auto-disable decisions.
        """
        total_callbacks = 0
        total_dropped = 0
        max_burst = 0
        current_burst = 0

        with self._sources_lock:
            sources_snapshot = list(self._sources)
        for wrapper in sources_snapshot:
            source = wrapper.source
            telemetry = (
                source.get_drop_telemetry()
                if hasattr(source, "get_drop_telemetry")
                else {}
            )
            total_callbacks += int(telemetry.get("total_callbacks", 0) or 0)
            total_dropped += int(telemetry.get("frames_dropped", 0) or 0)
            max_burst = max(
                max_burst,
                int(telemetry.get("max_consecutive_frames_dropped", 0) or 0),
            )
            current_burst = max(
                current_burst,
                int(telemetry.get("consecutive_frames_dropped", 0) or 0),
            )

        with self._stats_lock:
            self._stats.frames_dropped = max(self._stats.frames_dropped, total_dropped)
            aggregate_dropped = self._stats.frames_dropped
            self._stats.drop_rate = (
                aggregate_dropped / total_callbacks
                if total_callbacks > 0
                else 0.0
            )
            self._stats.max_consecutive_frames_dropped = max_burst
            self._stats.consecutive_frames_dropped = current_burst

    def get_error(self) -> Optional[Exception]:
        """Get the stored consumer thread error, if any.

        Returns the Exception that crashed the consumer loop, or None
        if no crash occurred.  This is the sanitized accessor for
        controller/UI diagnostics — never includes raw audio content.
        """
        return self._error
    
    def _create_sources(self, config: SessionConfig) -> List[AudioSourceWrapper]:
        """Create source wrappers from configuration."""
        import logging
        _log = logging.getLogger(__name__)
        wrappers = []
        
        for source_config in config.sources:
            if source_config.type == 'mic':
                source = MicSource(
                    device_id=source_config.device_id,
                    blocksize=DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
                    queue_size=10,
                    on_frame_dropped=self._on_source_frame_dropped,
                )
            elif source_config.type == 'system':
                source = SystemSource(
                    device_id=source_config.device_id,
                    blocksize=DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
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
            self._stats.denoising.enabled = True
            self._stats.denoising.fallback = True
            self._mark_denoising_disabled(
                "provider_init_error",
                error_class=type(exc).__name__,
                error_message=str(exc)[:200],
                log_level=logging.WARNING,
                log_message="Denoising provider init failed, continuing raw: error_class=%s",
            )

    def _mark_denoising_disabled(
        self,
        reason: str,
        *,
        error_class: str = "",
        error_message: str = "",
        log_level: int = logging.INFO,
        log_message: str = "Denoising disabled for session: reason=%s",
    ) -> None:
        """Fail open to raw mic audio and record sanitized disable metadata."""
        stats = self._stats.denoising
        if self._denoising_disabled and stats.disabled_reason:
            return

        self._denoising_disabled = True
        stats.active = False
        stats.fallback = True
        stats.disabled_reason = reason
        stats.disabled_at = time.time()
        stats.disabled_count += 1
        if error_class:
            stats.last_error_class = error_class
        if error_message:
            stats.last_error_message = error_message[:200]

        try:
            _log.log(log_level, log_message, error_class or reason)
        except TypeError:
            _log.log(log_level, "Denoising disabled for session: reason=%s", reason)

    def _maybe_disable_denoising_for_frame_drops(self) -> None:
        """Auto-disable denoising when sanitized frame-drop thresholds are exceeded."""
        if not self._config:
            return
        if not self._config.microphone_denoising_auto_disable_on_frame_drops:
            self._drop_rate_exceeded_since = None
            return
        if self._denoising_disabled or not self._stats.denoising.enabled:
            return

        self._refresh_drop_counters_only()
        stats = self._stats
        if stats.max_consecutive_frames_dropped > 10:
            self._mark_denoising_disabled(
                "frame_drop_burst",
                log_level=logging.WARNING,
                log_message="Denoising auto-disabled after frame-drop burst: reason=%s",
            )
            return

        now = time.time()
        if stats.drop_rate > 0.01:
            if self._drop_rate_exceeded_since is None:
                self._drop_rate_exceeded_since = now
            elif now - self._drop_rate_exceeded_since > 5.0:
                self._mark_denoising_disabled(
                    "frame_drop_rate_sustained",
                    log_level=logging.WARNING,
                    log_message="Denoising auto-disabled after sustained frame-drop rate: reason=%s",
                )
        else:
            self._drop_rate_exceeded_since = None

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

        self._maybe_disable_denoising_for_frame_drops()

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
            self._stats.denoising.record_fallback(0.0, f"{type(exc).__name__}: {exc}")
            self._mark_denoising_disabled(
                "provider_process_error",
                error_class=type(exc).__name__,
                error_message=str(exc)[:200],
                log_level=logging.WARNING,
                log_message="Denoising process error, hard-disabling for session: error_class=%s",
            )
            return frames
    
    def _consumer_loop(self) -> None:
        """Background thread that reads from sources and writes to disk."""
        try:
            self._consumer_loop_inner()
        except Exception as exc:
            # Guard: consumer thread crash must not die invisibly.
            # Log, store error, transition state, and invoke callback.
            _log.exception("Audio consumer thread crashed: %s", type(exc).__name__)
            self._error = exc
            # Transition to ERROR unless stop/finalize already in progress
            if self._state not in (SessionState.STOPPING, SessionState.FINALIZED):
                self._state = SessionState.ERROR
            # Signal stop to prevent further work
            self._stop_event.set()
            # Invoke on_error callback safely
            if self._config and self._config.on_error:
                try:
                    self._config.on_error(exc)
                except Exception:
                    _log.exception("on_error callback raised during consumer crash")

    def _consumer_loop_inner(self) -> None:
        """Inner consumer loop — separated for crash-guard wrapping."""
        discard_mode = False
        max_frames = self._config.max_frames if self._config else None

        # Shared-timeline position: total samples emitted (written OR discarded)
        emitted_total = 0
        # Per-wrapper mix state for the position-aligned mixer
        mix_states: Dict[AudioSourceWrapper, _SourceMixState] = {}
        stall_timeout = (
            self._config.mix_stall_timeout_s if self._config else DEFAULT_MIX_STALL_TIMEOUT_S
        )

        while not self._stop_event.is_set():
            # Check writer is available
            if not self._writer:
                break

            mix_states, sources_snapshot = self._sync_mix_states(
                mix_states, emitted_total
            )

            # Read from all sources, applying denoising per-source before mixing
            read_any = False
            for wrapper in sources_snapshot:
                frames = wrapper.read_and_process(timeout=0.05)
                if frames is not None:
                    read_any = True
                    # Apply denoising to denoise-enabled sources before mixing
                    frames = self._apply_denoising(frames, wrapper)
                    state = mix_states[wrapper]
                    state.carry = (
                        np.concatenate((state.carry, frames), axis=0)
                        if state.carry.shape[0]
                        else frames
                    )
                    state.source_pos += frames.shape[0]
                    state.last_delivery = time.monotonic()

            # Stall detection and stall-recovery backlog drop
            self._update_stall_states(
                mix_states, emitted_total, stall_timeout,
                detect_onset=not self._stop_event.is_set(),
            )

            # Position-aligned emission: min over non-stalled carries
            n = self._aligned_emit_length(mix_states)
            emitted = 0
            if n > 0:
                emitted, discard_now = self._emit_aligned(
                    mix_states, n, max_frames=max_frames, discard_mode=discard_mode,
                    final=False,
                )
                if discard_now:
                    discard_mode = True
                # Consumed is consumed — advance the shared timeline even
                # when the cap turned this round's samples into a discard.
                emitted_total += emitted

            if not emitted and not read_any:
                # No frames available, sleep briefly
                time.sleep(0.01)
                continue

            # discard_mode may have been set mid-emission; nothing else to do
            if discard_mode and emitted == 0 and read_any:
                # Frames were consumed but discarded under the cap — pace loop
                time.sleep(0.01)

        # Drain remaining frames (respecting max_frames cap)
        ended: set = set()
        empty_drain_rounds: Dict[AudioSourceWrapper, int] = {}
        for _ in range(50):  # Brief drain period
            if not self._writer:
                break

            mix_states, sources_snapshot = self._sync_mix_states(
                mix_states, emitted_total
            )

            # Check if we've already hit the cap
            if max_frames is not None and self._stats.frames_recorded >= max_frames:
                # Consume but discard remaining frames to prevent queue blocking
                for wrapper in sources_snapshot:
                    wrapper.read_and_process(timeout=0.01)
                continue

            read_any = False
            delivered: set = set()
            for wrapper in sources_snapshot:
                frames = wrapper.read_and_process(timeout=0.01)
                if frames is not None:
                    read_any = True
                    delivered.add(wrapper)
                    frames = self._apply_denoising(frames, wrapper)
                    state = mix_states[wrapper]
                    state.carry = (
                        np.concatenate((state.carry, frames), axis=0)
                        if state.carry.shape[0]
                        else frames
                    )
                    state.source_pos += frames.shape[0]
                    state.last_delivery = time.monotonic()

            # Track consecutive drain rounds where a source returns nothing
            # and has no carry: after 3, treat it as ENDED and exclude it
            # from the min-gate for the rest of the drain.
            for wrapper, state in mix_states.items():
                if wrapper in ended:
                    continue
                if state.carry.shape[0] == 0 and wrapper not in delivered:
                    empty_drain_rounds[wrapper] = empty_drain_rounds.get(wrapper, 0) + 1
                    if empty_drain_rounds[wrapper] >= 3:
                        ended.add(wrapper)
                        _log.info(
                            "Drain: source ended, span zero-filled: source=%s, "
                            "pos=%d",
                            wrapper.config.type,
                            state.source_pos,
                        )
                else:
                    empty_drain_rounds[wrapper] = 0

            # Drain never declares NEW stalls — it only processes recovery
            # drops for sources already stalled when draining began.
            self._update_stall_states(
                mix_states, emitted_total, stall_timeout, detect_onset=False,
            )

            gate_states = [
                s for w, s in mix_states.items()
                if w not in ended and not s.stalled
            ]
            n = min((s.carry.shape[0] for s in gate_states), default=0)
            if n <= 0:
                carry_pending = any(
                    s.carry.shape[0] > 0
                    for w, s in mix_states.items()
                    if w not in ended and not s.stalled
                )
                if not read_any and not carry_pending:
                    break
                continue

            emitted, _discard_now = self._emit_aligned(
                mix_states, n, max_frames=max_frames, discard_mode=False,
                final=True, ended=ended,
            )
            # Discard signal ignored here: the drain's own cap branch above
            # handles subsequent rounds once frames_recorded >= max_frames.
            emitted_total += emitted

    def _sync_mix_states(
        self,
        mix_states: Dict[AudioSourceWrapper, _SourceMixState],
        emitted_total: int,
    ) -> Tuple[Dict[AudioSourceWrapper, _SourceMixState], List[AudioSourceWrapper]]:
        """Sync per-source mix state with the current sources snapshot.

        Drops entries for wrappers no longer present (hotplug swap) and
        lazily creates state for new wrappers at the current timeline
        position (a new source starts "now").

        Carry handoff: a departed wrapper's un-emitted carry (plus its
        ``source_pos`` and stall counters) is stashed under its config
        type; a replacement wrapper of the same type receives that state
        (one-shot — the stash entry is popped on use) so a hot-plug swap
        neither drops nor re-timestamps the buffered tail. Transferring
        both carry and source_pos keeps ``source_pos - carry_len`` (the
        carry head's timeline position) consistent for the resync-drop
        arithmetic in ``_update_stall_states``. Departures with empty
        carry stash nothing.

        Returns the mix_states dict AND the exact sources snapshot the
        sync ran against. Callers must read from the returned snapshot —
        not re-snapshot ``self._sources`` — so a source added after sync
        cannot be read without a mix_states entry.
        """
        with self._sources_lock:
            sources_snapshot = list(self._sources)
        live = set(id(w) for w in sources_snapshot)
        for wrapper in list(mix_states.keys()):
            if id(wrapper) not in live:
                state = mix_states[wrapper]
                if state.carry.shape[0] > 0:
                    # Overwrite is correct: swap_source is the only
                    # mutation seam and replaces one wrapper per type.
                    self._mix_carry_handoff[wrapper.config.type] = (
                        state.carry,
                        state.source_pos,
                        state.stalled,
                        state.stall_episodes,
                        state.zero_filled_samples,
                    )
                del mix_states[wrapper]
        now = time.monotonic()
        for wrapper in sources_snapshot:
            if wrapper not in mix_states:
                handoff = self._mix_carry_handoff.pop(
                    wrapper.config.type, None
                )
                if handoff is not None:
                    carry, source_pos, stalled, stall_episodes, zero_filled = (
                        handoff
                    )
                    mix_states[wrapper] = _SourceMixState(
                        carry=carry,
                        source_pos=source_pos,
                        last_delivery=now,
                        stalled=stalled,
                        stall_episodes=stall_episodes,
                        zero_filled_samples=zero_filled,
                    )
                else:
                    mix_states[wrapper] = _SourceMixState(
                        carry=np.zeros((0, 1), dtype=np.float32),
                        source_pos=emitted_total,
                        last_delivery=now,
                    )
        return mix_states, sources_snapshot

    def _update_stall_states(
        self,
        mix_states: Dict[AudioSourceWrapper, _SourceMixState],
        emitted_total: int,
        stall_timeout: float,
        detect_onset: bool = True,
    ) -> None:
        """Detect stall onset/recovery and drop recovered sources' backlogs.

        A source is stalled iff its carry is empty and it has not delivered
        for longer than the stall timeout. A recovering source drops the
        carry-head samples whose timeline position was already zero-filled
        (``source_pos - len(carry) < emitted_total``). ``detect_onset``
        suppresses NEW stall declarations (after stop, during drain) while
        still processing recovery drops.
        """
        now = time.monotonic()
        # Pass 1 — recovery drops for stalled sources (so pass 2 sees the
        # post-drop carry of others when deciding stall materiality).
        for wrapper, state in mix_states.items():
            if not state.stalled:
                continue
            # Resync-drop: backlog samples whose span was already
            # zero-filled are dropped so the source stays position-locked.
            drop = min(
                state.carry.shape[0],
                max(emitted_total - (state.source_pos - state.carry.shape[0]), 0),
            )
            if drop > 0:
                state.carry = state.carry[drop:]
                _log.info(
                    "Mix source recovered, dropped zero-filled backlog: "
                    "source=%s, dropped=%d",
                    wrapper.config.type,
                    drop,
                )
            if state.carry.shape[0] > 0:
                state.stalled = False
        # Pass 2 — stall onset detection.
        for wrapper, state in mix_states.items():
            if state.stalled:
                continue
            if not (
                detect_onset
                and state.carry.shape[0] == 0
                and now - state.last_delivery > stall_timeout
            ):
                continue
            state.stalled = True
            # Only count/log episodes that materially gate the timeline:
            # the source is behind what has been emitted, nothing has been
            # emitted yet, or another source still has carry pending (so
            # this source's span will be zero-filled). A caught-up source
            # that simply has no more content is not a stall episode.
            others_have_carry = any(
                s.carry.shape[0] > 0
                for w2, s in mix_states.items()
                if w2 is not wrapper
            )
            materially_behind = (
                state.source_pos < emitted_total
                or emitted_total == 0
                or others_have_carry
            )
            if materially_behind:
                state.stall_episodes += 1
                self._stats.mix_stall_episodes += 1
                _log.warning(
                    "Mix source stalled, zero-filling its span: "
                    "source=%s, silent_for=%.2fs",
                    wrapper.config.type,
                    now - state.last_delivery,
                )

    def _aligned_emit_length(
        self,
        mix_states: Dict[AudioSourceWrapper, _SourceMixState],
    ) -> int:
        """Emission gate: min carry length over non-stalled sources.

        A non-stalled source with empty carry forces 0 (wait for it). If all
        sources are stalled, emit nothing this iteration.
        """
        gate_states = [s for s in mix_states.values() if not s.stalled]
        if not gate_states:
            return 0
        return min(s.carry.shape[0] for s in gate_states)

    def _emit_aligned(
        self,
        mix_states: Dict[AudioSourceWrapper, _SourceMixState],
        n: int,
        *,
        max_frames: Optional[int],
        discard_mode: bool,
        final: bool,
        ended: Optional[set] = None,
    ) -> Tuple[int, bool]:
        """Emit ``n`` position-aligned samples from every source's carry head.

        Stalled/ended sources contribute ``n`` zeros each (for mono mixing the
        live sources pass through). Runs the existing max_frames cap /
        discard_mode / partial-write logic and feeds ``on_audio_frame``
        exactly as the legacy loop did.

        Returns a tuple ``(consumed, discard_now)``: *consumed* is the
        number of samples removed from every carry this call (``n`` in
        every non-error path — written or discarded, including a partial
        cap write); *discard_now* is True when the caller must switch to
        discard mode (a partial write hit the cap or the cap was already
        reached).
        """
        take_states = [
            (w, s) for w, s in mix_states.items()
            if (ended is None or w not in ended) and s.carry.shape[0] >= n
        ]
        contributes = [
            s.carry[:n] for _, s in take_states
        ]
        zero_fill_states = [
            s for w, s in mix_states.items()
            if (ended is None or w not in ended) and s.carry.shape[0] < n
        ]
        parts = list(contributes) + [
            np.zeros((n, 1), dtype=np.float32) for _ in zero_fill_states
        ]
        for state in zero_fill_states:
            state.zero_filled_samples += n
            self._stats.mix_zero_filled_samples += n
        for _, state in take_states:
            state.carry = state.carry[n:]

        mixed = self._mix_frames(parts)

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
                # Partial chunk would exceed cap - write only remaining frames.
                # All n carry samples were consumed, written or not, so the
                # caller's timeline accounting must still advance by n.
                mixed = mixed[:remaining]
                int16_bytes = self._float32_to_int16_bytes(mixed)
                self._writer.write_frames_i16(int16_bytes)
                self._stats.frames_recorded += len(mixed)
                # Switch to discard mode after final write
                return n, True

        if discard_mode:
            # In discard mode: consume frames but don't write
            return n, True

        # Convert to int16 and write
        int16_bytes = self._float32_to_int16_bytes(mixed)
        self._writer.write_frames_i16(int16_bytes)
        self._stats.frames_recorded += len(mixed)
        return n, False
    
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
        with self._sources_lock:
            sources_snapshot = list(self._sources)
        for wrapper in sources_snapshot:
            try:
                wrapper.stop()
            except Exception:
                _log.debug(
                    "Cleanup: wrapper.stop() failed: error_class=%s",
                    type(wrapper).__name__,
                )
        
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                _log.debug("Cleanup: writer.close() failed")
