"""Tests for the audio denoising provider contract and lightweight implementation.

Covers:
- Provider shape/dtype/clipping contract
- Stationary-noise reduction on synthetic fixtures
- Speech-tone preservation
- Unsupported provider validation
- Malformed input fallback behavior
- Per-frame latency under configured budget
- Negative tests: empty, non-numeric, stereo, NaN/Inf, silence, clipped
"""

import time

import numpy as np
import pytest

from meetandread.audio.denoising import (
    DEFAULT_PROVIDER_NAME,
    VALID_PROVIDER_NAMES,
    DenoisingResult,
    SpectralGateProvider,
    create_provider,
)


# ============================================================================
# Helper fixtures
# ============================================================================

@pytest.fixture
def provider() -> SpectralGateProvider:
    """Default lightweight denoising provider."""
    return SpectralGateProvider()


def _make_noisy_speech(duration_s: float = 1.0, sr: int = 16000,
                       speech_freq: float = 440.0, noise_std: float = 0.15,
                       seed: int = 42) -> np.ndarray:
    """Create a synthetic noisy speech signal: sine tone + Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * speech_freq * t)
    noise = rng.normal(0, noise_std, size=t.shape).astype(np.float32)
    return (speech + noise).astype(np.float32)


def _make_pure_noise(duration_s: float = 1.0, sr: int = 16000,
                     noise_std: float = 0.2, seed: int = 7) -> np.ndarray:
    """Create a pure stationary noise signal."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, noise_std, size=int(sr * duration_s)).astype(np.float32)


def _make_speech_tone(duration_s: float = 1.0, sr: int = 16000,
                      freq: float = 300.0, amplitude: float = 0.3) -> np.ndarray:
    """Create a clean speech-like tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_clean_noise_and_noisy(
    duration_s: float = 2.0, sr: int = 16000,
    speech_freq: float = 440.0, speech_amplitude: float = 0.5,
    noise_std: float = 0.15, seed: int = 42,
) -> tuple:
    """Create deterministic clean speech, noise, and noisy speech arrays.

    Returns:
        (clean, noise, noisy) — all float32, same length.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    clean = (speech_amplitude * np.sin(2 * np.pi * speech_freq * t)).astype(np.float32)
    noise = rng.normal(0, noise_std, size=n_samples).astype(np.float32)
    noisy = (clean + noise).astype(np.float32)
    return clean, noise, noisy


def _compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB: 10 * log10(signal_power / noise_power).

    Returns -np.inf if noise power is zero.
    """
    sig_pow = float(np.mean(signal ** 2))
    noise_pow = float(np.mean(noise ** 2))
    if noise_pow < 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(sig_pow / noise_pow))


# ============================================================================
# Provider contract: shape, dtype, clipping
# ============================================================================

class TestProviderContract:
    """Tests for the DenoisingProvider interface contract."""

    def test_returns_same_shape_as_input(self, provider: SpectralGateProvider) -> None:
        """Output must have the same number of samples as input."""
        frame = np.zeros(16000, dtype=np.float32)
        result = provider.process(frame)
        assert result.audio.shape == frame.shape

    def test_output_is_float32(self, provider: SpectralGateProvider) -> None:
        """Output dtype must be float32."""
        frame = np.zeros(8000, dtype=np.float32)
        result = provider.process(frame)
        assert result.audio.dtype == np.float32

    def test_output_clipped_to_unit_range(self, provider: SpectralGateProvider) -> None:
        """Output must be clipped to [-1, 1]."""
        # Very high amplitude input that could cause numerical overflow
        frame = np.full(1024, 10.0, dtype=np.float32)
        result = provider.process(frame)
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)

    def test_result_has_stats(self, provider: SpectralGateProvider) -> None:
        """DenoisingResult must include timing and status stats."""
        frame = np.zeros(1024, dtype=np.float32)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert isinstance(result.fallback, bool)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0
        assert isinstance(result.provider_name, str)
        assert result.provider_name == "spectral_gate"

    def test_result_no_error_on_valid_input(self, provider: SpectralGateProvider) -> None:
        """No error should be reported on valid input."""
        frame = np.zeros(1024, dtype=np.float32)
        result = provider.process(frame)
        assert result.error is None
        assert result.fallback is False


# ============================================================================
# Noise reduction quality
# ============================================================================

class TestNoiseReduction:
    """Tests for measurable noise attenuation without brittle exact values."""

    def test_reduces_stationary_noise(self, provider: SpectralGateProvider) -> None:
        """Noisy signal should have lower energy after denoising."""
        noisy = _make_noisy_speech(noise_std=0.2)
        result = provider.process(noisy)

        # Energy of the residual (denoised signal minus the denoised signal's
        # low-frequency content) should be lower than the original noise floor
        # Simple: compare RMS of denoised output to RMS of noisy input
        rms_input = float(np.sqrt(np.mean(noisy ** 2)))
        rms_output = float(np.sqrt(np.mean(result.audio ** 2)))
        # Denoised should have measurably lower RMS than the noisy input
        assert rms_output < rms_input * 0.95, (
            f"Denoised RMS {rms_output:.4f} not meaningfully less than "
            f"input RMS {rms_input:.4f}"
        )

    def test_preserves_speech_tone(self, provider: SpectralGateProvider) -> None:
        """A clean speech-like tone should survive denoising."""
        tone = _make_speech_tone(amplitude=0.3)
        result = provider.process(tone)

        # The output should still contain significant energy at the tone frequency
        # Compare correlation between input and output
        correlation = float(np.corrcoef(tone, result.audio)[0, 1])
        assert correlation > 0.5, (
            f"Speech-tone correlation {correlation:.3f} too low after denoising"
        )

    def test_pure_noise_output_energy_lower(self, provider: SpectralGateProvider) -> None:
        """Pure noise should be significantly attenuated."""
        noise = _make_pure_noise(noise_std=0.2)
        result = provider.process(noise)

        rms_input = float(np.sqrt(np.mean(noise ** 2)))
        rms_output = float(np.sqrt(np.mean(result.audio ** 2)))
        # Pure noise should be strongly suppressed
        assert rms_output < rms_input * 0.7, (
            f"Pure noise RMS {rms_output:.4f} not suppressed enough vs {rms_input:.4f}"
        )

    def test_improves_noisy_speech_snr_by_at_least_5db(
        self, provider: SpectralGateProvider
    ) -> None:
        """Denoising should improve SNR by at least 5 dB on noisy speech.

        Uses a clean reference signal and deterministic noise to compute
        baseline SNR (clean vs injected noise) and denoised SNR (clean vs
        residual after denoising). Gain-aligns the denoised output to the
        clean signal to avoid penalizing harmless level changes.
        """
        clean, noise, noisy = _make_clean_noise_and_noisy(
            duration_s=2.0, speech_amplitude=0.5, noise_std=0.15, seed=42
        )

        # Baseline SNR: clean vs injected noise
        baseline_snr = _compute_snr(clean, noise)

        # Denoise the noisy signal
        result = provider.process(noisy)
        assert result.fallback is False, (
            f"Provider returned fallback on valid noisy input: {result.error}"
        )

        denoised = result.audio

        # Gain-align denoised output to clean to avoid penalizing level changes
        clean_energy = float(np.mean(clean ** 2))
        denoised_energy = float(np.mean(denoised ** 2))
        if denoised_energy > 1e-12:
            gain = float(np.sqrt(clean_energy / denoised_energy))
            denoised_aligned = denoised * gain
        else:
            denoised_aligned = denoised

        # Residual = what's left after subtracting clean from denoised
        residual = denoised_aligned - clean
        denoised_snr = _compute_snr(clean, residual)

        snr_improvement = denoised_snr - baseline_snr
        assert snr_improvement >= 5.0, (
            f"SNR improvement {snr_improvement:.1f} dB is below 5 dB target "
            f"(baseline={baseline_snr:.1f} dB, denoised={denoised_snr:.1f} dB)"
        )


# ============================================================================
# Provider factory
# ============================================================================

class TestProviderFactory:
    """Tests for the create_provider factory function."""

    def test_default_provider_name(self) -> None:
        """Default provider name should be 'spectral_gate'."""
        assert DEFAULT_PROVIDER_NAME == "spectral_gate"

    def test_create_default_provider(self) -> None:
        """Factory with default name returns SpectralGateProvider."""
        provider = create_provider()
        assert isinstance(provider, SpectralGateProvider)

    def test_create_spectral_gate_explicitly(self) -> None:
        """Factory with explicit 'spectral_gate' returns SpectralGateProvider."""
        provider = create_provider("spectral_gate")
        assert isinstance(provider, SpectralGateProvider)

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown denoising provider"):
            create_provider("nonexistent_provider")

    def test_valid_provider_names_includes_default(self) -> None:
        """VALID_PROVIDER_NAMES should contain the default."""
        assert DEFAULT_PROVIDER_NAME in VALID_PROVIDER_NAMES


# ============================================================================
# Negative tests: malformed inputs
# ============================================================================

class TestMalformedInputs:
    """Tests for malformed input handling — provider must never crash caller."""

    def test_empty_array(self, provider: SpectralGateProvider) -> None:
        """Empty array should return a fallback result without crashing."""
        frame = np.array([], dtype=np.float32)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert result.audio.shape == (0,)
        assert result.fallback is True

    def test_non_float32_input(self, provider: SpectralGateProvider) -> None:
        """Non-float32 numeric input should be converted and processed."""
        frame = np.zeros(1024, dtype=np.int16)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert result.audio.dtype == np.float32
        assert result.audio.shape == (1024,)

    def test_2d_stereo_rejected(self, provider: SpectralGateProvider) -> None:
        """2D stereo input should be rejected with a fallback result."""
        frame = np.zeros((1024, 2), dtype=np.float32)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert result.fallback is True
        assert result.error is not None

    def test_nan_input_sanitized(self, provider: SpectralGateProvider) -> None:
        """NaN samples should be sanitized without crashing."""
        frame = np.full(1024, np.nan, dtype=np.float32)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        # Output should not contain NaN
        assert not np.any(np.isnan(result.audio))

    def test_inf_input_sanitized(self, provider: SpectralGateProvider) -> None:
        """Inf samples should be sanitized without crashing."""
        frame = np.full(1024, np.inf, dtype=np.float32)
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert not np.any(np.isinf(result.audio))
        # Output should be clipped
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)


# ============================================================================
# Boundary conditions
# ============================================================================

class TestBoundaryConditions:
    """Tests for edge cases: silence, low-amplitude, clipped input."""

    def test_silence_only(self, provider: SpectralGateProvider) -> None:
        """Silent input should produce near-silent output."""
        frame = np.zeros(16000, dtype=np.float32)
        result = provider.process(frame)
        assert np.max(np.abs(result.audio)) < 1e-6

    def test_low_amplitude_tone(self, provider: SpectralGateProvider) -> None:
        """Low-amplitude speech-like tone should be preserved."""
        tone = _make_speech_tone(amplitude=0.05)
        result = provider.process(tone)
        # Should still have some energy
        assert np.max(np.abs(result.audio)) > 0.0

    def test_clipped_high_amplitude(self, provider: SpectralGateProvider) -> None:
        """High-amplitude input should be clipped to [-1, 1] in output."""
        frame = np.full(4096, 5.0, dtype=np.float32)
        result = provider.process(frame)
        assert np.all(result.audio >= -1.0)
        assert np.all(result.audio <= 1.0)

    def test_very_short_frame(self, provider: SpectralGateProvider) -> None:
        """Very short frames (below FFT size) should be handled gracefully."""
        frame = np.ones(32, dtype=np.float32) * 0.1
        result = provider.process(frame)
        assert isinstance(result, DenoisingResult)
        assert result.audio.shape == (32,)


# ============================================================================
# Latency budget
# ============================================================================

class TestLatencyBudget:
    """Tests for per-frame latency under configured budget."""

    def test_typical_chunk_under_200ms(self, provider: SpectralGateProvider) -> None:
        """Typical audio chunk (30ms at 16kHz = 480 samples) should process well under 200ms."""
        # 30ms chunk at 16kHz = 480 samples (typical for real-time audio callbacks)
        frame = _make_noisy_speech(duration_s=0.03, noise_std=0.15)
        result = provider.process(frame)
        assert result.latency_ms < 200.0, (
            f"Latency {result.latency_ms:.1f}ms exceeds 200ms budget"
        )

    def test_1s_chunk_under_200ms(self, provider: SpectralGateProvider) -> None:
        """1-second chunk should still process under the budget."""
        frame = _make_noisy_speech(duration_s=1.0, noise_std=0.15)
        result = provider.process(frame)
        assert result.latency_ms < 200.0, (
            f"Latency {result.latency_ms:.1f}ms exceeds 200ms budget for 1s chunk"
        )

    def test_measured_latency_matches_wall_clock(self, provider: SpectralGateProvider) -> None:
        """Reported latency should be reasonable vs wall clock time."""
        frame = _make_noisy_speech(duration_s=0.5, noise_std=0.15)
        start = time.perf_counter()
        result = provider.process(frame)
        wall_ms = (time.perf_counter() - start) * 1000
        # Reported should be close to wall clock (within 50% tolerance)
        assert result.latency_ms <= wall_ms * 2.0, (
            f"Reported latency {result.latency_ms:.1f}ms seems wrong vs "
            f"wall clock {wall_ms:.1f}ms"
        )
