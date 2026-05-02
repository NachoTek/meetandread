"""Voice Activity Detection with WebRTC VAD and energy-based fallback.

Provides a reusable live VAD boundary for the transcription pipeline.
Follows the project's no-PyTorch constraint: lazily imports ``webrtcvad``
and degrades to an RMS/energy detector when the import fails or processing
errors occur.

Observability:
    VADResult and VADStats carry sanitized backend name, fallback/error
    status, latency, and decision counters. No raw audio samples, transcript
    text, or speaker embeddings are logged.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000  # Expected sample rate (Hz).
_FRAME_DURATION_MS = 30  # WebRTC VAD frame duration in milliseconds.
_FRAME_SAMPLES = int(_SAMPLE_RATE * _FRAME_DURATION_MS / 1000)  # 480 samples.
_DEFAULT_ENERGY_THRESHOLD = 0.005  # RMS threshold matching old SPEECH_THRESHOLD.
_DEFAULT_LATENCY_BUDGET_MS = 10.0  # VAD-specific latency budget (ms).


# ---------------------------------------------------------------------------
# Result / stats dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VADResult:
    """Immutable result of a single VAD decision.

    Attributes:
        is_speech: True if speech was detected in the frame.
        backend: Name of the backend that made the decision
            (``"webrtcvad"`` or ``"energy"``).
        fallback: True if the energy backend was used due to a WebRTC error
            or missing dependency.
        latency_ms: Wall-clock time for the decision in milliseconds.
        error: Sanitized error class/message if a fallback occurred.
    """
    is_speech: bool
    backend: str
    fallback: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class VADStats:
    """Cumulative diagnostics for a VoiceActivityDetector instance.

    Counters are mutable so callers can read them via
    ``VoiceActivityDetector.get_stats()`` at any time.
    """
    backend: str = "unknown"
    active_fallback: bool = False
    frames_processed: int = 0
    speech_decisions: int = 0
    silence_decisions: int = 0
    fallback_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_error_class: str = ""
    last_error_message: str = ""
    budget_exceeded_count: int = 0
    _total_latency_ms: float = field(default=0.0, repr=False)


# ---------------------------------------------------------------------------
# VoiceActivityDetector
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """WebRTC-backed VAD with energy fallback.

    Accepts float32 mono 16 kHz audio arrays of any chunk size. Internally
    buffers to exact 30 ms / 480-sample frames for WebRTC compatibility.
    Falls back to an RMS/energy detector when ``webrtcvad`` is unavailable
    or raises an error.

    Parameters:
        energy_threshold: RMS threshold for the energy fallback backend.
            Defaults to 0.005 (matching the old ``SPEECH_THRESHOLD``).
        latency_budget_ms: Per-frame latency budget in milliseconds.
            Decisions exceeding this are counted in ``budget_exceeded_count``.
        aggressiveness: WebRTC VAD aggressiveness (0–3). Higher values are
            more aggressive about filtering non-speech. Default 3.
    """

    def __init__(
        self,
        energy_threshold: float = _DEFAULT_ENERGY_THRESHOLD,
        latency_budget_ms: float = _DEFAULT_LATENCY_BUDGET_MS,
        aggressiveness: int = 3,
    ) -> None:
        self._energy_threshold = energy_threshold
        self._latency_budget_ms = latency_budget_ms
        self._aggressiveness = aggressiveness
        self._stats = VADStats()

        # Lazy WebRTC backend state
        self._vad: object = None
        self._webrtc_available: Optional[bool] = None  # None = not tried yet

        # Internal frame buffer for accumulating partial chunks to 480 samples
        self._buffer = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, audio: np.ndarray) -> VADResult:
        """Process an audio chunk and return a VAD decision.

        The chunk can be any length. Internally, the chunk is buffered to
        complete 480-sample (30 ms) frames. The decision reflects the
        *latest* complete frame in the chunk. If no complete frame is
        available, the last decision is based on whatever audio is present
        using the energy backend.

        Args:
            audio: Float32 mono 16 kHz numpy array. Other dtypes are
                converted; NaN/Inf are sanitized.

        Returns:
            VADResult with the speech/silence decision and diagnostics.
        """
        start = time.perf_counter()

        # Sanitize input
        audio = self._sanitize_input(audio)

        if audio.size == 0:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._update_stats(
                is_speech=False,
                backend="energy",
                fallback=True,
                latency_ms=latency_ms,
                error="empty_input",
            )
            return VADResult(
                is_speech=False,
                backend="energy",
                fallback=True,
                latency_ms=latency_ms,
                error="empty_input",
            )

        # Append to internal buffer
        self._buffer = np.concatenate([self._buffer, audio])

        # Process as many complete frames as possible; keep the last result
        result: Optional[VADResult] = None
        while self._buffer.size >= _FRAME_SAMPLES:
            frame = self._buffer[:_FRAME_SAMPLES]
            self._buffer = self._buffer[_FRAME_SAMPLES:]
            result = self._process_single_frame(frame, start)

        # If there were leftover partial samples (no complete frame),
        # use the energy backend on the partial buffer.
        if result is None:
            result = self._energy_decide(self._buffer, start, error="partial_frame")

        return result

    def get_stats(self) -> VADStats:
        """Return a snapshot of cumulative diagnostics."""
        return self._stats

    def reset(self) -> None:
        """Reset internal buffer and stats counters."""
        self._buffer = np.array([], dtype=np.float32)
        self._stats = VADStats()

    # ------------------------------------------------------------------
    # Single-frame processing
    # ------------------------------------------------------------------

    def _process_single_frame(self, frame: np.ndarray, chunk_start: float) -> VADResult:
        """Process exactly 480 samples through WebRTC or energy fallback."""
        # Try WebRTC first if available
        if self._webrtc_available is not False:
            try:
                vad = self._ensure_webrtc()
                if vad is not None:
                    return self._webrtc_decide(vad, frame, chunk_start)
            except Exception as exc:
                # Any error → fallback and log
                self._record_webrtc_error(exc)
                logger.warning(
                    "WebRTC VAD error, using energy fallback: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        # Energy fallback
        return self._energy_decide(frame, chunk_start, error=None)

    def _webrtc_decide(self, vad: object, frame: np.ndarray, chunk_start: float) -> VADResult:
        """Run WebRTC VAD on a single 480-sample frame."""
        # Clip to [-1, 1], then convert to int16 PCM
        clipped = np.clip(frame, -1.0, 1.0)
        pcm_int16 = (clipped * 32767.0).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        try:
            is_speech = bool(vad.is_speech(pcm_bytes, _SAMPLE_RATE))
        except Exception as exc:
            # WebRTC rejected the frame (e.g., wrong frame size)
            self._record_webrtc_error(exc)
            latency_ms = (time.perf_counter() - chunk_start) * 1000.0
            self._update_stats(
                is_speech=False,
                backend="energy",
                fallback=True,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
            return VADResult(
                is_speech=False,
                backend="energy",
                fallback=True,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = (time.perf_counter() - chunk_start) * 1000.0
        self._update_stats(
            is_speech=is_speech,
            backend="webrtcvad",
            fallback=False,
            latency_ms=latency_ms,
            error=None,
        )
        return VADResult(
            is_speech=is_speech,
            backend="webrtcvad",
            fallback=False,
            latency_ms=latency_ms,
        )

    def _energy_decide(self, frame: np.ndarray, chunk_start: float,
                       error: Optional[str]) -> VADResult:
        """RMS energy-based speech/silence decision."""
        if frame.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

        is_speech = rms > self._energy_threshold
        latency_ms = (time.perf_counter() - chunk_start) * 1000.0

        fallback = error is not None or self._webrtc_available is False
        err_str = error if error is not None else None

        self._update_stats(
            is_speech=is_speech,
            backend="energy",
            fallback=fallback,
            latency_ms=latency_ms,
            error=err_str,
        )
        return VADResult(
            is_speech=is_speech,
            backend="energy",
            fallback=fallback,
            latency_ms=latency_ms,
            error=err_str,
        )

    # ------------------------------------------------------------------
    # WebRTC lifecycle
    # ------------------------------------------------------------------

    def _ensure_webrtc(self) -> Optional[object]:
        """Lazily import and instantiate ``webrtcvad.Vad``.

        Returns None if the import fails (sets ``_webrtc_available`` to False).
        """
        if self._webrtc_available is not None:
            if self._webrtc_available and self._vad is not None:
                return self._vad
            return None

        try:
            import webrtcvad  # type: ignore[import-untyped]
            self._vad = webrtcvad.Vad(self._aggressiveness)
            self._webrtc_available = True
            self._stats.backend = "webrtcvad"
            logger.info(
                "VAD backend: webrtcvad (aggressiveness=%d, "
                "frame=%dms/%d samples)",
                self._aggressiveness,
                _FRAME_DURATION_MS,
                _FRAME_SAMPLES,
            )
            return self._vad
        except ImportError:
            self._webrtc_available = False
            self._stats.backend = "energy"
            self._stats.active_fallback = True
            logger.info(
                "VAD backend: energy fallback (webrtcvad not available, "
                "threshold=%.4f)",
                self._energy_threshold,
            )
            return None
        except Exception as exc:
            self._webrtc_available = False
            self._stats.backend = "energy"
            self._stats.active_fallback = True
            self._record_webrtc_error(exc)
            logger.warning(
                "VAD backend: energy fallback (webrtcvad init failed: %s: %s)",
                type(exc).__name__,
                exc,
            )
            return None

    def _record_webrtc_error(self, exc: Exception) -> None:
        """Record a WebRTC error in stats without exposing raw data."""
        self._stats.last_error_class = type(exc).__name__
        self._stats.last_error_message = str(exc)[:200]  # Truncate to avoid leaks

    # ------------------------------------------------------------------
    # Stats update
    # ------------------------------------------------------------------

    def _update_stats(
        self,
        is_speech: bool,
        backend: str,
        fallback: bool,
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        """Update cumulative stats counters after a decision."""
        s = self._stats
        s.frames_processed += 1
        if is_speech:
            s.speech_decisions += 1
        else:
            s.silence_decisions += 1
        if fallback:
            s.fallback_count += 1
            s.active_fallback = True
        s._total_latency_ms += latency_ms
        if latency_ms > s.max_latency_ms:
            s.max_latency_ms = latency_ms
        s.avg_latency_ms = (
            s._total_latency_ms / s.frames_processed
            if s.frames_processed > 0
            else 0.0
        )
        if latency_ms > self._latency_budget_ms:
            s.budget_exceeded_count += 1
        if error is not None and s.last_error_class == "":
            # Preserve first error; callers can check fallback_count for repeats
            s.last_error_class = "fallback"
            s.last_error_message = error[:200]

    # ------------------------------------------------------------------
    # Input sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_input(audio: np.ndarray) -> np.ndarray:
        """Sanitize input: convert dtype, replace NaN/Inf, clip to [-1, 1].

        Returns a 1-D float32 array. Does not raise.
        """
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio, dtype=np.float32)
            except Exception:
                return np.array([], dtype=np.float32)

        if audio.size == 0:
            return np.array([], dtype=np.float32)

        # Flatten 2D+ to 1-D (take first channel if multi-channel)
        if audio.ndim > 1:
            audio = audio.reshape(-1)

        # Convert to float32
        if audio.dtype != np.float32:
            try:
                audio = audio.astype(np.float32)
            except (ValueError, TypeError):
                return np.zeros(audio.size, dtype=np.float32)

        # Replace NaN → 0.0
        nan_mask = np.isnan(audio)
        if np.any(nan_mask):
            audio = np.where(nan_mask, 0.0, audio)

        # Replace Inf → sign-preserved ±1.0
        inf_mask = np.isinf(audio)
        if np.any(inf_mask):
            audio = np.where(inf_mask, np.sign(audio), audio)

        # Clip to [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        return audio
