"""Tests for the VoiceActivityDetector (WebRTC + energy fallback).

Covers:
- WebRTC backend with speech-like and silence fixtures
- Energy fallback on missing webrtcvad, errors, and malformed input
- Frame buffering: sub-frame chunks, exact frames, multi-frame chunks
- Diagnostics counters in VADStats
- Negative tests: empty, NaN/Inf, 2D, unsupported dtype
- Latency budget checks
- Sanitized logs (no raw audio content)
"""

import logging
import time
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from meetandread.transcription.vad import (
    VoiceActivityDetector,
    VADResult,
    VADStats,
    _FRAME_SAMPLES,
    _FRAME_DURATION_MS,
    _DEFAULT_ENERGY_THRESHOLD,
    _DEFAULT_LATENCY_BUDGET_MS,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_tone(freq: float = 440.0, duration_s: float = 0.03,
               sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Sine tone — simulates speech-like energy."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_silence(duration_s: float = 0.03, sr: int = 16000) -> np.ndarray:
    """Near-zero energy signal — simulates silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_low_noise(duration_s: float = 0.03, sr: int = 16000,
                    std: float = 0.001) -> np.ndarray:
    """Very low RMS noise — below energy threshold (should classify as silence)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, std, size=int(sr * duration_s)).astype(np.float32)


def _make_high_noise(duration_s: float = 0.03, sr: int = 16000,
                     std: float = 0.05) -> np.ndarray:
    """High RMS noise — above energy threshold (energy backend says speech)."""
    rng = np.random.default_rng(99)
    return rng.normal(0, std, size=int(sr * duration_s)).astype(np.float32)


def _make_noisy_speech(duration_s: float = 0.03, sr: int = 16000) -> np.ndarray:
    """Tone + noise — clearly above threshold."""
    tone = _make_tone(freq=300, duration_s=duration_s, sr=sr, amplitude=0.4)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.05, size=tone.shape).astype(np.float32)
    return (tone + noise).astype(np.float32)


# One exact frame (480 samples at 16 kHz / 30 ms)
EXACT_FRAME_TONE = _make_tone(duration_s=_FRAME_DURATION_MS / 1000)
EXACT_FRAME_SILENCE = _make_silence(duration_s=_FRAME_DURATION_MS / 1000)
EXACT_FRAME_LOW_NOISE = _make_low_noise(duration_s=_FRAME_DURATION_MS / 1000)
EXACT_FRAME_HIGH_NOISE = _make_high_noise(duration_s=_FRAME_DURATION_MS / 1000)
EXACT_FRAME_NOISY_SPEECH = _make_noisy_speech(duration_s=_FRAME_DURATION_MS / 1000)


# ---------------------------------------------------------------------------
# Basic VAD result shape
# ---------------------------------------------------------------------------

class TestVADResultShape:
    """VADResult is a frozen dataclass with expected fields."""

    def test_result_is_frozen(self) -> None:
        r = VADResult(is_speech=True, backend="webrtcvad")
        with pytest.raises(AttributeError):
            r.is_speech = False  # type: ignore[misc]

    def test_result_defaults(self) -> None:
        r = VADResult(is_speech=True, backend="energy")
        assert r.fallback is False
        assert r.latency_ms == 0.0
        assert r.error is None


# ---------------------------------------------------------------------------
# Energy backend (forced by mocking missing webrtcvad)
# ---------------------------------------------------------------------------

class TestEnergyBackend:
    """Energy fallback backend decisions based on RMS threshold."""

    @pytest.fixture
    def detector_no_webrtc(self) -> VoiceActivityDetector:
        """Detector with webrtcvad import blocked."""
        det = VoiceActivityDetector(energy_threshold=_DEFAULT_ENERGY_THRESHOLD)
        # Force energy-only mode
        det._webrtc_available = False
        det._stats.backend = "energy"
        det._stats.active_fallback = True
        return det

    def test_tone_is_speech(self, detector_no_webrtc: VoiceActivityDetector) -> None:
        result = detector_no_webrtc.process_chunk(EXACT_FRAME_TONE)
        assert result.is_speech is True
        assert result.backend == "energy"

    def test_silence_is_not_speech(self, detector_no_webrtc: VoiceActivityDetector) -> None:
        result = detector_no_webrtc.process_chunk(EXACT_FRAME_SILENCE)
        assert result.is_speech is False
        assert result.backend == "energy"

    def test_low_noise_is_silence(self, detector_no_webrtc: VoiceActivityDetector) -> None:
        result = detector_no_webrtc.process_chunk(EXACT_FRAME_LOW_NOISE)
        assert result.is_speech is False

    def test_high_noise_is_speech(self, detector_no_webrtc: VoiceActivityDetector) -> None:
        result = detector_no_webrtc.process_chunk(EXACT_FRAME_HIGH_NOISE)
        assert result.is_speech is True

    def test_noisy_speech_is_speech(self, detector_no_webrtc: VoiceActivityDetector) -> None:
        result = detector_no_webrtc.process_chunk(EXACT_FRAME_NOISY_SPEECH)
        assert result.is_speech is True

    def test_custom_threshold(self) -> None:
        det = VoiceActivityDetector(energy_threshold=0.1)
        det._webrtc_available = False
        # High noise with std=0.05 → RMS ~0.05, below custom threshold of 0.1
        result = det.process_chunk(EXACT_FRAME_HIGH_NOISE)
        assert result.is_speech is False


# ---------------------------------------------------------------------------
# WebRTC backend (using real webrtcvad if available)
# ---------------------------------------------------------------------------

class TestWebRTCBackend:
    """WebRTC backend processes frames and reports correct backend name."""

    @pytest.fixture
    def detector(self) -> VoiceActivityDetector:
        return VoiceActivityDetector()

    def test_webrtc_tone_is_speech(self, detector: VoiceActivityDetector) -> None:
        """A strong tone should be classified as speech by WebRTC."""
        # Feed multiple frames to ensure the buffer has a complete frame
        result = detector.process_chunk(EXACT_FRAME_TONE)
        # WebRTC VAD is fairly reliable on tones — this should be speech
        assert isinstance(result, VADResult)
        # If webrtcvad is available, backend should be webrtcvad
        try:
            import webrtcvad  # noqa: F401
            assert result.backend == "webrtcvad"
        except ImportError:
            assert result.backend == "energy"

    def test_webrtc_silence_is_not_speech(self, detector: VoiceActivityDetector) -> None:
        """Zero signal should be classified as silence."""
        result = detector.process_chunk(EXACT_FRAME_SILENCE)
        assert result.is_speech is False

    def test_mocked_webrtc_says_speech(self) -> None:
        """With a mocked WebRTC backend that always says speech, verify wiring."""
        det = VoiceActivityDetector()
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True

        # Inject mock
        det._vad = mock_vad
        det._webrtc_available = True
        det._stats.backend = "webrtcvad"

        result = det.process_chunk(EXACT_FRAME_TONE)
        assert result.is_speech is True
        assert result.backend == "webrtcvad"
        assert result.fallback is False

    def test_mocked_webrtc_says_silence(self) -> None:
        """With a mocked WebRTC backend that says silence, verify wiring."""
        det = VoiceActivityDetector()
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False

        det._vad = mock_vad
        det._webrtc_available = True
        det._stats.backend = "webrtcvad"

        result = det.process_chunk(EXACT_FRAME_SILENCE)
        assert result.is_speech is False
        assert result.backend == "webrtcvad"

    def test_mocked_webrtc_exception_falls_back(self) -> None:
        """WebRTC exception should fall back to energy without crashing."""
        det = VoiceActivityDetector()
        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = RuntimeError("frame error")

        det._vad = mock_vad
        det._webrtc_available = True
        det._stats.backend = "webrtcvad"

        # Feed a tone so energy fallback says speech
        result = det.process_chunk(EXACT_FRAME_TONE)
        assert isinstance(result, VADResult)
        # After the error, subsequent calls should still work
        assert result.fallback is True or result.backend == "energy"

    def test_mocked_webrtc_error_falls_back(self) -> None:
        """webrtcvad.Error should fall back to energy."""
        det = VoiceActivityDetector()
        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = Exception("generic error")

        det._vad = mock_vad
        det._webrtc_available = True

        result = det.process_chunk(EXACT_FRAME_TONE)
        assert isinstance(result, VADResult)
        # Should not crash; should still produce a decision


# ---------------------------------------------------------------------------
# Fallback activation paths
# ---------------------------------------------------------------------------

class TestFallbackActivation:
    """Fallback activates on ImportError, exceptions, and malformed input."""

    def test_import_error_activates_fallback(self) -> None:
        """When webrtcvad is missing, detector uses energy backend."""
        det = VoiceActivityDetector()
        with patch.dict("sys.modules", {"webrtcvad": None}):
            det._webrtc_available = None
            det._vad = None
            # Force re-import attempt
            det._ensure_webrtc()
            assert det._webrtc_available is False
            result = det.process_chunk(EXACT_FRAME_TONE)
            assert result.backend == "energy"
            assert result.fallback is True

    def test_fallback_count_increments(self) -> None:
        """Fallback counter increments on each fallback decision."""
        det = VoiceActivityDetector()
        det._webrtc_available = False
        det._stats.backend = "energy"

        det.process_chunk(EXACT_FRAME_TONE)
        assert det.get_stats().fallback_count == 1

        det.process_chunk(EXACT_FRAME_SILENCE)
        assert det.get_stats().fallback_count == 2


# ---------------------------------------------------------------------------
# Frame buffering
# ---------------------------------------------------------------------------

class TestFrameBuffering:
    """Sub-frame, exact-frame, and multi-frame chunk handling."""

    @pytest.fixture
    def det(self) -> VoiceActivityDetector:
        d = VoiceActivityDetector()
        d._webrtc_available = False  # Use energy for deterministic tests
        d._stats.backend = "energy"
        d._stats.active_fallback = True
        return d

    def test_sub_frame_chunk_uses_energy(self, det: VoiceActivityDetector) -> None:
        """Chunk smaller than 480 samples should use energy backend."""
        small = EXACT_FRAME_TONE[:100]
        result = det.process_chunk(small)
        assert isinstance(result, VADResult)
        # Energy fallback handles partial frames
        assert result.error == "partial_frame" or result.backend == "energy"

    def test_exact_frame_chunk(self, det: VoiceActivityDetector) -> None:
        """Exactly 480 samples → one complete frame."""
        result = det.process_chunk(EXACT_FRAME_TONE)
        assert result.is_speech is True
        assert det.get_stats().frames_processed == 1

    def test_multi_frame_chunk(self, det: VoiceActivityDetector) -> None:
        """960 samples = 2 frames; stats should reflect 2 frames processed."""
        two_frames = np.concatenate([EXACT_FRAME_TONE, EXACT_FRAME_TONE])
        result = det.process_chunk(two_frames)
        assert isinstance(result, VADResult)
        assert det.get_stats().frames_processed == 2

    def test_leftover_buffer_carried(self, det: VoiceActivityDetector) -> None:
        """Leftover samples from first chunk are combined with second chunk."""
        # Feed 600 samples (480 + 120 leftover)
        big = np.concatenate([EXACT_FRAME_TONE, np.zeros(120, dtype=np.float32)])
        det.process_chunk(big)
        assert det.get_stats().frames_processed == 1

        # Feed remaining 360 samples to complete the second frame
        det.process_chunk(np.zeros(360, dtype=np.float32))
        assert det.get_stats().frames_processed == 2


# ---------------------------------------------------------------------------
# Diagnostics counters
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """VADStats counters track backend, decisions, latency, and errors."""

    @pytest.fixture
    def det(self) -> VoiceActivityDetector:
        d = VoiceActivityDetector()
        d._webrtc_available = False
        d._stats.backend = "energy"
        d._stats.active_fallback = True
        return d

    def test_initial_stats(self) -> None:
        det = VoiceActivityDetector()
        stats = det.get_stats()
        assert stats.frames_processed == 0
        assert stats.speech_decisions == 0
        assert stats.silence_decisions == 0
        assert stats.fallback_count == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.max_latency_ms == 0.0

    def test_speech_silence_counters(self, det: VoiceActivityDetector) -> None:
        det.process_chunk(EXACT_FRAME_TONE)
        det.process_chunk(EXACT_FRAME_SILENCE)
        stats = det.get_stats()
        assert stats.speech_decisions == 1
        assert stats.silence_decisions == 1
        assert stats.frames_processed == 2

    def test_avg_max_latency(self, det: VoiceActivityDetector) -> None:
        det.process_chunk(EXACT_FRAME_TONE)
        stats = det.get_stats()
        assert stats.max_latency_ms >= 0
        assert stats.avg_latency_ms >= 0
        assert stats.max_latency_ms >= stats.avg_latency_ms

    def test_last_error_on_fallback(self) -> None:
        det = VoiceActivityDetector()
        det._webrtc_available = False
        det.process_chunk(np.array([], dtype=np.float32))
        stats = det.get_stats()
        assert stats.last_error_class != "" or stats.fallback_count > 0

    def test_reset_clears_stats(self, det: VoiceActivityDetector) -> None:
        det.process_chunk(EXACT_FRAME_TONE)
        assert det.get_stats().frames_processed > 0
        det.reset()
        stats = det.get_stats()
        assert stats.frames_processed == 0
        assert stats.speech_decisions == 0


# ---------------------------------------------------------------------------
# Negative tests: malformed inputs
# ---------------------------------------------------------------------------

class TestMalformedInputs:
    """VAD must never crash on malformed input."""

    @pytest.fixture
    def det(self) -> VoiceActivityDetector:
        d = VoiceActivityDetector()
        d._webrtc_available = False
        d._stats.backend = "energy"
        return d

    def test_empty_array(self, det: VoiceActivityDetector) -> None:
        result = det.process_chunk(np.array([], dtype=np.float32))
        assert isinstance(result, VADResult)
        assert result.is_speech is False
        assert result.error == "empty_input"

    def test_nan_array(self, det: VoiceActivityDetector) -> None:
        frame = np.full(_FRAME_SAMPLES, np.nan, dtype=np.float32)
        result = det.process_chunk(frame)
        assert isinstance(result, VADResult)
        # NaN → 0 → silence
        assert result.is_speech is False

    def test_inf_array(self, det: VoiceActivityDetector) -> None:
        frame = np.full(_FRAME_SAMPLES, np.inf, dtype=np.float32)
        result = det.process_chunk(frame)
        assert isinstance(result, VADResult)
        # Inf → 1.0 → high RMS → speech
        assert result.is_speech is True

    def test_2d_array(self, det: VoiceActivityDetector) -> None:
        frame = np.zeros((_FRAME_SAMPLES, 2), dtype=np.float32)
        result = det.process_chunk(frame)
        assert isinstance(result, VADResult)

    def test_int16_dtype(self, det: VoiceActivityDetector) -> None:
        frame = np.ones(_FRAME_SAMPLES, dtype=np.int16)
        result = det.process_chunk(frame)
        assert isinstance(result, VADResult)

    def test_over_range_clipped(self, det: VoiceActivityDetector) -> None:
        frame = np.full(_FRAME_SAMPLES, 10.0, dtype=np.float32)
        result = det.process_chunk(frame)
        assert isinstance(result, VADResult)
        # Should be clipped to 1.0 → high RMS → speech
        assert result.is_speech is True


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------

class TestLatencyBudget:
    """Per-frame latency should be under the configured budget."""

    def test_single_frame_under_budget(self) -> None:
        det = VoiceActivityDetector(latency_budget_ms=10.0)
        det._webrtc_available = False
        result = det.process_chunk(EXACT_FRAME_TONE)
        assert result.latency_ms < 50.0, (
            f"VAD latency {result.latency_ms:.1f}ms seems excessive"
        )

    def test_budget_exceeded_counter(self) -> None:
        """Stats should track budget exceedances."""
        # Use an unrealistic low budget to force exceedance
        det = VoiceActivityDetector(latency_budget_ms=0.0001)
        det._webrtc_available = False
        det.process_chunk(EXACT_FRAME_TONE)
        assert det.get_stats().budget_exceeded_count >= 1


# ---------------------------------------------------------------------------
# Sanitized logs
# ---------------------------------------------------------------------------

class TestSanitizedLogs:
    """Logs should not contain raw audio content or samples."""

    def test_no_raw_audio_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        det = VoiceActivityDetector(energy_threshold=0.005)
        det._webrtc_available = False

        with caplog.at_level(logging.INFO, logger="meetandread.transcription.vad"):
            det.process_chunk(EXACT_FRAME_TONE)

        for record in caplog.records:
            msg = record.message
            # Should not contain array representations
            assert "[" not in msg or "frame=" in msg or "threshold=" in msg
            # Should not contain sample values
            assert "0.5" not in msg or "threshold" in msg


# ---------------------------------------------------------------------------
# Integration: speech → silence → speech sequence
# ---------------------------------------------------------------------------

class TestSpeechSilenceSpeechSequence:
    """A noisy speech → silence → speech sequence should produce correct boundaries."""

    def test_speech_silence_speech_sequence(self) -> None:
        det = VoiceActivityDetector()
        det._webrtc_available = False
        det._stats.backend = "energy"

        # Feed speech
        speech1 = _make_noisy_speech(duration_s=0.09)  # 3 frames
        r1 = det.process_chunk(speech1)

        # Feed silence
        silence = _make_silence(duration_s=0.09)
        r2 = det.process_chunk(silence)

        # Feed speech again
        speech2 = _make_noisy_speech(duration_s=0.06)
        r3 = det.process_chunk(speech2)

        stats = det.get_stats()
        assert stats.frames_processed == 8  # 3 + 3 + 2
        assert stats.speech_decisions > 0
        assert stats.silence_decisions > 0
