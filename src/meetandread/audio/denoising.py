"""Audio denoising provider contract and lightweight implementation.

Provides a swappable provider boundary for microphone frame denoising
that avoids PyTorch/PyInstaller packaging risks (D015). All providers
accept 16 kHz mono float32 numpy frames and return same-shape float32
frames clipped to [-1, 1].

The default implementation is a numpy-only spectral gate inspired by
RNNoise-style noise suppression. It reduces stationary noise on synthetic
fixtures while preserving speech-like tones.

Observability:
    DenoisingResult carries sanitized provider name, fallback/error status,
    and latency_ms for downstream SessionStats/source_stats exposure.
    No raw audio or transcript content is logged.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER_NAME: str = "spectral_gate"
VALID_PROVIDER_NAMES: List[str] = [DEFAULT_PROVIDER_NAME]


# ---------------------------------------------------------------------------
# Result / stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class DenoisingResult:
    """Result of denoising a single audio frame.

    Attributes:
        audio: Denoised audio as float32 mono numpy array, clipped to [-1, 1].
        provider_name: Name of the provider that produced this result.
        latency_ms: Wall-clock time spent denoising in milliseconds.
        fallback: True if the provider fell back to raw/sanitized passthrough
            due to an error, malformed input, or unsupported shape.
        error: Sanitized error class + message if fallback occurred, else None.
    """
    audio: np.ndarray
    provider_name: str
    latency_ms: float = 0.0
    fallback: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider protocol (abstract base)
# ---------------------------------------------------------------------------

class DenoisingProvider(ABC):
    """Abstract base for audio denoising providers.

    Implementations must:
    - Accept 1-D float32 numpy arrays (16 kHz mono).
    - Return DenoisingResult with same-shape float32 audio clipped to [-1, 1].
    - Never raise on malformed input — return a fallback result instead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging and factory lookup."""
        ...

    @abstractmethod
    def process(self, frame: np.ndarray) -> DenoisingResult:
        """Denoise a single audio frame.

        Args:
            frame: Raw audio frame (ideally float32 mono, 16 kHz).

        Returns:
            DenoisingResult with cleaned audio and status metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Lightweight spectral-gate provider (numpy-only)
# ---------------------------------------------------------------------------

class SpectralGateProvider(DenoisingProvider):
    """Numpy-only spectral gate noise suppressor.

    Uses a short-time FFT to estimate a noise floor from the first few
    frames and applies a spectral gate that attenuates bins below a
    noise-threshold multiple. This is inspired by classic spectral
    subtraction / RNNoise-style gating but avoids any model download
    or PyTorch dependency.

    Parameters:
        fft_size: FFT window size (power of 2). Default 512 (~32ms at 16kHz).
        hop_size: Hop between successive analysis windows. Default 256.
        noise_floor_db: Noise floor estimate in dB below peak. Default -35.
        gate_threshold: Multiplicative threshold for the spectral gate.
            Bins with magnitude < noise_estimate * gate_threshold are
            attenuated. Default 1.8.
        attack_frames: Number of initial frames used to estimate noise.
            Default 4.
    """

    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 256,
        noise_floor_db: float = -35.0,
        gate_threshold: float = 1.8,
        attack_frames: int = 4,
    ) -> None:
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._noise_floor_db = noise_floor_db
        self._gate_threshold = gate_threshold
        self._attack_frames = attack_frames
        self._frame_count: int = 0
        # Noise spectrum estimate (accumulated during attack phase)
        self._noise_estimate: Optional[np.ndarray] = None
        # Cross-frame overlap tail: last fft_size samples from previous call,
        # prepended to the next frame so STFT windows span chunk boundaries.
        # Eliminates clicking artifacts from per-frame overlap-add discontinuities.
        self._overlap_tail: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "spectral_gate"

    def process(self, frame: np.ndarray) -> DenoisingResult:
        """Denoise a single audio frame with spectral gating.

        Handles malformed inputs gracefully by returning a fallback result.
        """
        start = time.perf_counter()

        # ---- Input validation / sanitization ----
        try:
            sanitized, validation_error = self._sanitize_input(frame)
        except Exception as exc:
            # Last-resort: return zeros with error
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DenoisingResult(
                audio=np.zeros_like(frame, dtype=np.float32).flatten(),
                provider_name=self.name,
                latency_ms=latency_ms,
                fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        if validation_error is not None:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DenoisingResult(
                audio=np.clip(sanitized, -1.0, 1.0),
                provider_name=self.name,
                latency_ms=latency_ms,
                fallback=True,
                error=validation_error,
            )

        # ---- Spectral gate processing ----
        try:
            output = self._spectral_gate(sanitized)
        except Exception as exc:
            logger.warning(
                "Denoising processing error, falling back to sanitized input: %s",
                exc,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DenoisingResult(
                audio=np.clip(sanitized, -1.0, 1.0),
                provider_name=self.name,
                latency_ms=latency_ms,
                fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Final clip to [-1, 1]
        output = np.clip(output, -1.0, 1.0)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._frame_count += 1

        return DenoisingResult(
            audio=output,
            provider_name=self.name,
            latency_ms=latency_ms,
            fallback=False,
            error=None,
        )

    def _sanitize_input(
        self, frame: np.ndarray
    ) -> tuple:
        """Validate and sanitize input frame.

        Returns:
            (sanitized_frame, error_string_or_None)
            If error is not None, the sanitized frame is a safe passthrough.
        """
        # Handle empty array — return empty with fallback marker
        if frame.size == 0:
            return np.array([], dtype=np.float32), "empty_input"

        # Reject 2D+ arrays
        if frame.ndim > 1:
            flat = np.zeros(frame.shape[0], dtype=np.float32)
            return flat, (
                f"Unsupported shape {frame.shape}: expected 1-D mono array"
            )

        # Convert to float32
        if frame.dtype != np.float32:
            try:
                frame = frame.astype(np.float32)
            except (ValueError, TypeError) as exc:
                return np.zeros(frame.shape[0], dtype=np.float32), (
                    f"Cannot convert dtype {frame.dtype} to float32: {exc}"
                )

        # Sanitize NaN → 0
        nan_count = int(np.sum(np.isnan(frame)))
        if nan_count > 0:
            frame = np.where(np.isnan(frame), 0.0, frame)

        # Sanitize Inf → ±1.0
        inf_mask = np.isinf(frame)
        if np.any(inf_mask):
            frame = np.where(inf_mask, np.sign(frame), frame)

        # Clip to a reasonable pre-processing range to avoid FFT blowups
        frame = np.clip(frame, -100.0, 100.0)

        return frame, None

    def _spectral_gate(self, frame: np.ndarray) -> np.ndarray:
        """Apply spectral gating to a 1-D float32 frame.

        Uses a scalar noise floor estimate derived from the global spectral
        median, which correctly handles pure tones, pure noise, and noisy
        speech without multi-frame accumulation. For frames shorter than
        fft_size, falls back to simple time-domain noise gating.

        Cross-frame overlap: prepends the tail of the previous frame so
        STFT windows span chunk boundaries, eliminating clicking artifacts
        from per-frame overlap-add discontinuities.
        """
        n = len(frame)

        if n < self._fft_size:
            # Too short for meaningful FFT — use simple energy gate
            return self._simple_gate(frame)

        # ---- Cross-frame overlap ----
        # Prepend the tail from the previous call so STFT windows bridge
        # the boundary smoothly. On the first call, use zeros.
        tail = self._overlap_tail
        if tail is None:
            tail = np.zeros(self._fft_size, dtype=np.float32)
        combined = np.concatenate([tail, frame])

        # ---- Overlap-add STFT on combined signal ----
        window = np.hanning(self._fft_size)
        combined_n = len(combined)
        remainder = (combined_n - self._fft_size) % self._hop_size
        pad_len = combined_n if remainder == 0 else combined_n + (self._hop_size - remainder)
        padded = np.zeros(pad_len, dtype=np.float32)
        padded[:combined_n] = combined

        num_windows = (pad_len - self._fft_size) // self._hop_size + 1

        # Collect magnitudes for noise estimation
        all_mags = []
        spectra = []

        for i in range(num_windows):
            start_idx = i * self._hop_size
            segment = padded[start_idx:start_idx + self._fft_size]
            windowed = segment * window
            spectrum = np.fft.rfft(windowed)
            mag = np.abs(spectrum)
            all_mags.append(mag)
            spectra.append(spectrum)

        # Scalar noise floor: median of ALL spectral magnitudes.
        # This works because:
        #   - Pure noise: all bins ≈ noise level → median ≈ noise ✓
        #   - Noisy speech: most bins are noise → median ≈ noise ✓
        #   - Pure tone: most bins ≈ 0 → median ≈ 0 → gate passes tone ✓
        noise_floor = float(np.median(np.concatenate(all_mags)))

        # Blend with accumulated estimate if available
        if self._noise_estimate is not None and self._noise_estimate.size == 1:
            noise_floor = float(0.6 * noise_floor + 0.4 * self._noise_estimate.item())

        # Store scalar noise estimate for next call
        self._noise_estimate = np.array([noise_floor])

        threshold = noise_floor * self._gate_threshold

        output = np.zeros(pad_len, dtype=np.float32)
        window_sum = np.zeros(pad_len, dtype=np.float32)

        for i in range(num_windows):
            start_idx = i * self._hop_size
            mag = all_mags[i]
            phase = np.angle(spectra[i])

            if threshold > 1e-8:
                # Compute gate gain using scalar noise threshold
                ratio = mag / threshold
                # Smooth gate: pass bins well above noise, suppress below
                gate_gain = np.clip((ratio - 0.5) / 1.0, 0.0, 1.0)
                # Strong suppression for very low energy bins
                low_mask = mag < threshold * 0.3
                gate_gain[low_mask] *= 0.05
            else:
                # Near-zero noise floor: pass everything
                gate_gain = np.ones_like(mag)

            # Apply gate and IFFT
            cleaned_mag = mag * gate_gain
            cleaned_spectrum = cleaned_mag * np.exp(1j * phase)
            cleaned_segment = np.fft.irfft(cleaned_spectrum, n=self._fft_size)

            # Overlap-add
            output[start_idx:start_idx + self._fft_size] += cleaned_segment * window
            window_sum[start_idx:start_idx + self._fft_size] += window ** 2

        # Normalize by window sum (avoid division by zero)
        window_sum = np.where(window_sum > 1e-8, window_sum, 1.0)
        output = output / window_sum

        # Save tail for next call (last fft_size samples of combined input)
        self._overlap_tail = combined[-self._fft_size:].copy()

        # Return only the output corresponding to the new frame
        # (skip the fft_size overlap portion from the previous frame)
        return output[self._fft_size:self._fft_size + n].astype(np.float32)

    def _simple_gate(self, frame: np.ndarray) -> np.ndarray:
        """Simple time-domain noise gate for very short frames.

        Attenuates samples below a threshold derived from frame energy.
        """
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 1e-8:
            # Near-silence: return as-is
            return frame

        # Gate threshold at 20% of RMS
        threshold = rms * 0.2
        gate = np.where(
            np.abs(frame) > threshold,
            1.0,
            (np.abs(frame) / (threshold + 1e-10)).astype(np.float32),
        )
        return (frame * gate).astype(np.float32)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "spectral_gate": SpectralGateProvider,
}


def create_provider(name: Optional[str] = None) -> DenoisingProvider:
    """Create a denoising provider by name.

    Args:
        name: Provider name. If None, uses DEFAULT_PROVIDER_NAME.

    Returns:
        Instantiated DenoisingProvider.

    Raises:
        ValueError: If name is not in VALID_PROVIDER_NAMES.
    """
    provider_name = name or DEFAULT_PROVIDER_NAME

    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown denoising provider '{provider_name}'. "
            f"Valid providers: {VALID_PROVIDER_NAMES}"
        )

    return _PROVIDERS[provider_name]()
