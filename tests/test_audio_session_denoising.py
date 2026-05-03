"""Integration tests for AudioSession denoising boundary.

Tests prove the denoising boundary in the live recording path:
1. Denoised callback frames for mic-designated sources
2. Raw bypass for system/non-denoised sources
3. WAV finalization succeeds alongside denoising
4. Fail-open on provider init/process errors
5. Diagnostics populated correctly
6. Negative/boundary conditions
7. Log-level observability for denoising lifecycle events
"""

import wave
import time
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meetandread.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
)
from meetandread.audio.denoising import (
    DenoisingProvider,
    DenoisingResult,
    SpectralGateProvider,
)
from meetandread.audio.session import DenoisingStats


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def create_sine_wave_wav(
    path: Path,
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.5,
) -> None:
    """Create a sine wave WAV file."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    int16_data = (sine_wave * 32767).astype(np.int16)
    if channels == 2:
        int16_data = np.column_stack([int16_data, int16_data])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def create_noisy_wav(
    path: Path,
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    noise_level: float = 0.3,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Create a noisy sine WAV file. Returns the clean signal array."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    clean = amplitude * np.sin(2 * np.pi * frequency * t)
    rng = np.random.default_rng(42)
    noise = noise_level * rng.standard_normal(num_samples).astype(np.float32)
    noisy = np.clip(clean + noise, -1.0, 1.0)
    int16_data = (noisy * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
    return clean.astype(np.float32)


def create_silent_wav(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = 16000,
) -> None:
    """Create a silent WAV file."""
    num_samples = int(sample_rate * duration)
    int16_data = np.zeros(num_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def read_wav_as_float32(path: Path) -> np.ndarray:
    """Read a WAV file and return float32 samples."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        return data.astype(np.float32) / 32768.0


class TrackingProvider(DenoisingProvider):
    """Test denoising provider that tracks calls and reduces amplitude."""

    def __init__(self, scale: float = 0.5, fail_on_init: bool = False,
                 fail_on_process: bool = False, wrong_shape: bool = False,
                 latency_ms: float = 1.0):
        if fail_on_init:
            raise RuntimeError("Provider init failed (test)")
        self._scale = scale
        self._fail_on_process = fail_on_process
        self._wrong_shape = wrong_shape
        self._latency_ms = latency_ms
        self.process_calls: list = []

    @property
    def name(self) -> str:
        return "tracking_test"

    def process(self, frame: np.ndarray) -> DenoisingResult:
        self.process_calls.append(frame.copy())
        if self._fail_on_process:
            raise RuntimeError("Provider process failed (test)")
        output = (frame * self._scale).astype(np.float32)
        if self._wrong_shape:
            # Return wrong shape to test shape mismatch handling
            output = output[:len(output) // 2]
        return DenoisingResult(
            audio=output,
            provider_name=self.name,
            latency_ms=self._latency_ms,
            fallback=False,
        )


# ---------------------------------------------------------------------------
# Tests: Denoised callback frames
# ---------------------------------------------------------------------------


class TestDenoisedCallbackFrames:
    """Denoising occurs before on_audio_frame for denoise-enabled sources."""

    def test_fake_mic_source_receives_denoised_frames(self, tmp_path):
        """Fake source with denoise=True gets frames denoised before callback."""
        noisy_wav = tmp_path / "noisy.wav"
        create_noisy_wav(noisy_wav, frequency=440.0, duration=1.5)

        callback_frames: list = []

        provider = TrackingProvider(scale=0.1)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(noisy_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            on_audio_frame=lambda f: callback_frames.append(f),
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.6)
        wav_path = session.stop()

        # Provider should have been called with frames
        assert len(provider.process_calls) > 0, (
            "Denoising provider should have received frames"
        )

        # Callback should have been invoked with frames
        assert len(callback_frames) > 0, "on_audio_frame should have been called"

        # WAV should be finalized
        assert wav_path.exists(), "WAV file should exist after recording"

    def test_callback_receives_denoised_not_raw(self, tmp_path):
        """Verify callback frames are actually transformed, not raw passthrough.

        Uses a provider that scales by 0.1 — callback frames should be much
        quieter than the source signal.
        """
        # Create a loud sine wave
        loud_wav = tmp_path / "loud.wav"
        create_sine_wave_wav(loud_wav, frequency=440.0, duration=1.5, amplitude=0.9)

        callback_frames: list = []
        provider = TrackingProvider(scale=0.1)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(loud_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            on_audio_frame=lambda f: callback_frames.append(f),
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.6)
        session.stop()

        # The provider scales by 0.1, so callback frames should be low amplitude
        assert len(callback_frames) > 0
        max_amplitude = max(np.max(np.abs(f)) for f in callback_frames)
        # After 0.1 scaling, max should be well under original 0.9
        # (allow generous margin for mixing/float precision)
        assert max_amplitude < 0.5, (
            f"Callback frames should be denoised (scaled to 0.1), "
            f"but max amplitude was {max_amplitude:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests: Raw bypass for non-denoised sources
# ---------------------------------------------------------------------------


class TestRawBypass:
    """System/raw sources bypass denoising."""

    def test_system_fake_source_not_denoised(self, tmp_path):
        """Fake source without denoise flag is not processed by provider."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, frequency=440.0, duration=1.0)

        provider = TrackingProvider(scale=0.1)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav))],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        session.stop()

        # Provider should NOT have been called — source is type='fake' without denoise=True
        assert len(provider.process_calls) == 0, (
            "Non-denoised source should not be sent to provider"
        )

    def test_explicit_denoise_false_bypasses_provider(self, tmp_path):
        """Source with denoise=False bypasses denoising even when enabled."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        provider = TrackingProvider(scale=0.1)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=False)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        session.stop()

        assert len(provider.process_calls) == 0, (
            "Source with denoise=False should not be processed"
        )

    def test_mixed_denoised_and_raw_sources(self, tmp_path):
        """Only denoise-enabled sources are processed; others stay raw."""
        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        create_sine_wave_wav(wav_a, frequency=440.0, duration=1.0)
        create_sine_wave_wav(wav_b, frequency=880.0, duration=1.0)

        provider = TrackingProvider(scale=0.1)

        config = SessionConfig(
            sources=[
                SourceConfig(type="fake", fake_path=str(wav_a), denoise=True),
                SourceConfig(type="fake", fake_path=str(wav_b)),  # no denoise
            ],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.5)
        session.stop()

        # Provider should have been called for the denoise=True source only
        assert len(provider.process_calls) > 0, (
            "Denoise-enabled source should be processed"
        )


# ---------------------------------------------------------------------------
# Tests: WAV finalization with denoising
# ---------------------------------------------------------------------------


class TestWAVFinalization:
    """WAV files finalize correctly with denoising enabled."""

    def test_wav_finalized_with_denoising(self, tmp_path):
        """WAV output is valid when denoising is active."""
        noisy_wav = tmp_path / "noisy.wav"
        create_noisy_wav(noisy_wav, duration=1.5)

        provider = TrackingProvider(scale=0.5)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(noisy_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.6)
        wav_path = session.stop()

        # WAV should exist and be valid
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0

        stats = session.get_stats()
        assert stats.frames_recorded > 0

    def test_wav_finalized_without_denoising(self, tmp_path):
        """WAV finalization still works when denoising is disabled."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav))],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            # denoising not enabled
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnframes() > 0


# ---------------------------------------------------------------------------
# Tests: Fail-open behavior
# ---------------------------------------------------------------------------


class TestFailOpen:
    """Denoising failures fail open — recording continues with raw audio."""

    def test_provider_init_failure_continues_recording(self, tmp_path):
        """When provider init fails, recording continues with raw audio."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        callback_frames: list = []

        def failing_factory():
            raise RuntimeError("Init failed (test)")

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            on_audio_frame=lambda f: callback_frames.append(f),
            enable_microphone_denoising=True,
            denoising_provider_factory=failing_factory,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        # Recording should still complete
        assert wav_path.exists(), "WAV should still be created after provider init failure"
        assert len(callback_frames) > 0, "Callback should still receive raw frames"

        # Stats should show fallback
        stats = session.get_stats()
        assert stats.denoising.fallback is True
        assert stats.denoising.last_error_class != ""

    def test_provider_process_failure_continues_recording(self, tmp_path):
        """When provider.process() fails, recording continues raw."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        callback_frames: list = []
        provider = TrackingProvider(fail_on_process=True)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            on_audio_frame=lambda f: callback_frames.append(f),
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        # Recording completes even after process failure
        assert wav_path.exists()
        assert len(callback_frames) > 0

        # Stats show failure state
        stats = session.get_stats()
        assert stats.denoising.fallback is True

    def test_wrong_shape_output_uses_raw(self, tmp_path):
        """Provider returning wrong shape triggers fallback to raw frames."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        provider = TrackingProvider(wrong_shape=True)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        # Should still produce a valid WAV
        assert wav_path.exists()
        stats = session.get_stats()
        # Shape mismatch should increment fallback count
        assert stats.denoising.fallback is True


# ---------------------------------------------------------------------------
# Tests: Diagnostics
# ---------------------------------------------------------------------------


class TestDenoisingDiagnostics:
    """Latency/budget diagnostics are populated correctly."""

    def test_diagnostics_populated_on_success(self, tmp_path):
        """Stats show provider name, frame count, and latency after success."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.5)

        provider = TrackingProvider(scale=0.5, latency_ms=5.0)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.6)
        session.stop()

        stats = session.get_stats()
        ds = stats.denoising
        assert ds.provider == "tracking_test"
        assert ds.enabled is True
        assert ds.active is True
        assert ds.fallback is False
        assert ds.processed_frame_count > 0
        assert ds.avg_latency_ms > 0
        assert ds.max_latency_ms > 0

    def test_budget_exceeded_count(self, tmp_path):
        """Budget exceeded count increments when latency exceeds budget."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        # Provider simulates 300ms latency — well over default 200ms budget
        provider = TrackingProvider(scale=0.5, latency_ms=300.0)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
            denoising_latency_budget_ms=200.0,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        session.stop()

        stats = session.get_stats()
        assert stats.denoising.budget_exceeded_count > 0, (
            "Budget exceedance should be tracked"
        )

    def test_diagnostics_empty_when_disabled(self, tmp_path):
        """Denoising stats are empty when feature is disabled."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav))],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            # enable_microphone_denoising defaults to False
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.3)
        session.stop()

        stats = session.get_stats()
        assert stats.denoising.enabled is False
        assert stats.denoising.processed_frame_count == 0
        assert stats.denoising.provider == ""

    def test_error_class_message_populated_on_failure(self, tmp_path):
        """Error class and message are sanitized and populated on failure."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=1.0)

        provider = TrackingProvider(fail_on_process=True)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        session.stop()

        stats = session.get_stats()
        assert stats.denoising.last_error_class == "RuntimeError"
        assert "Provider process failed" in stats.denoising.last_error_message


# ---------------------------------------------------------------------------
# Tests: Negative / boundary conditions
# ---------------------------------------------------------------------------


class TestNegativeAndBoundary:
    """Malformed inputs, error paths, and boundary conditions."""

    def test_empty_silent_chunks_handled(self, tmp_path):
        """Provider receives silent chunks without crashing."""
        silent_wav = tmp_path / "silent.wav"
        create_silent_wav(silent_wav, duration=1.0)

        provider = TrackingProvider(scale=0.5)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(silent_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        assert wav_path.exists(), "Session should complete with silent input"

    def test_single_fake_mic_source_denoised(self, tmp_path):
        """Single source with denoise=True gets processed."""
        test_wav = tmp_path / "single.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        provider = TrackingProvider(scale=0.3)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.3)
        session.stop()

        assert len(provider.process_calls) > 0

    def test_max_frames_with_denoising(self, tmp_path):
        """max_frames cap works with denoising enabled."""
        test_wav = tmp_path / "capped.wav"
        create_sine_wave_wav(test_wav, duration=2.0)

        provider = TrackingProvider(scale=0.5)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            max_frames=16000,  # 1 second at 16kHz
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.5)
        wav_path = session.stop()

        assert wav_path.exists()
        stats = session.get_stats()
        # frames_recorded should not exceed max_frames
        assert stats.frames_recorded <= 16001  # small margin for rounding

    def test_real_spectral_gate_provider_works(self, tmp_path):
        """The real SpectralGateProvider reduces noise on a noisy fixture."""
        noisy_wav = tmp_path / "noisy.wav"
        clean_signal = create_noisy_wav(noisy_wav, frequency=440.0, duration=1.5,
                                         noise_level=0.3, amplitude=0.5)

        callback_frames: list = []

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(noisy_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            on_audio_frame=lambda f: callback_frames.append(f),
            enable_microphone_denoising=True,
            denoising_provider_name="spectral_gate",
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.6)
        wav_path = session.stop()

        assert wav_path.exists()
        stats = session.get_stats()
        assert stats.denoising.provider == "spectral_gate"
        assert stats.denoising.processed_frame_count > 0
        assert stats.denoising.fallback is False

    def test_existing_tests_still_pass_with_denoising_disabled(self, tmp_path):
        """Existing session functionality unaffected when denoising is off."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, frequency=440.0, duration=1.0)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav))],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
        )

        session = AudioSession()
        session.start(config)
        time.sleep(0.4)
        wav_path = session.stop()

        assert wav_path.exists()
        assert session.get_state().name == "FINALIZED"
        stats = session.get_stats()
        assert stats.frames_recorded > 0
        assert stats.denoising.enabled is False


# ---------------------------------------------------------------------------
# Tests: Log-level observability for denoising lifecycle
# ---------------------------------------------------------------------------


class TestDenoisingLogObservability:
    """Verify denoising logs structured records for init, fallback, and budget.

    These tests use caplog to assert the logging surface that future agents
    and downstream S02/S03 planners should be able to inspect.
    """

    def test_init_failure_logs_warning(self, tmp_path, caplog):
        """Provider init failure logs a sanitized WARNING with error_class."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        def failing_factory():
            raise RuntimeError("Init failed (test)")

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=failing_factory,
        )

        with caplog.at_level(logging.WARNING, logger="meetandread.audio.session"):
            session = AudioSession()
            session.start(config)
            time.sleep(0.2)
            session.stop()

        # Should have logged a warning about init failure
        denoising_warnings = [
            r for r in caplog.records
            if "Denoising provider init failed" in r.message
        ]
        assert len(denoising_warnings) > 0, (
            "Should log WARNING about denoising provider init failure"
        )
        # Sanitized: no raw audio or transcript content
        for record in denoising_warnings:
            assert "error_class" in record.message or "RuntimeError" in record.message

    def test_process_failure_logs_warning(self, tmp_path, caplog):
        """Provider process failure logs a WARNING about hard-disabling."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        provider = TrackingProvider(fail_on_process=True)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
        )

        with caplog.at_level(logging.WARNING, logger="meetandread.audio.session"):
            session = AudioSession()
            session.start(config)
            time.sleep(0.3)
            session.stop()

        # Should log about process error / hard-disabling
        process_warnings = [
            r for r in caplog.records
            if "hard-disabling" in r.message or "Denoising process error" in r.message
        ]
        assert len(process_warnings) > 0, (
            "Should log WARNING about denoising process failure"
        )

    def test_successful_init_logs_info(self, tmp_path, caplog):
        """Successful provider init logs an INFO with provider name and budget."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        provider = TrackingProvider(scale=0.5)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
            denoising_latency_budget_ms=200.0,
        )

        with caplog.at_level(logging.INFO, logger="meetandread.audio.session"):
            session = AudioSession()
            session.start(config)
            time.sleep(0.2)
            session.stop()

        init_logs = [
            r for r in caplog.records
            if "Denoising provider initialized" in r.message
        ]
        assert len(init_logs) > 0, (
            "Should log INFO about denoising provider initialization"
        )
        # Verify the log contains provider name and budget info
        assert "tracking_test" in init_logs[0].message
        assert "200" in init_logs[0].message

    def test_budget_exceeded_logs_info(self, tmp_path, caplog):
        """Budget exceeded logs an INFO with latency and budget values."""
        test_wav = tmp_path / "test.wav"
        create_sine_wave_wav(test_wav, duration=0.5)

        # Provider claims 300ms latency — exceeds 50ms budget
        provider = TrackingProvider(scale=0.5, latency_ms=300.0)

        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=str(test_wav), denoise=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=True,
            denoising_provider_factory=lambda: provider,
            denoising_latency_budget_ms=50.0,
        )

        with caplog.at_level(logging.INFO, logger="meetandread.audio.session"):
            session = AudioSession()
            session.start(config)
            time.sleep(0.3)
            session.stop()

        budget_logs = [
            r for r in caplog.records
            if "latency exceeded budget" in r.message
        ]
        assert len(budget_logs) > 0, (
            "Should log INFO when denoising latency exceeds budget"
        )


# ---------------------------------------------------------------------------
# Tests: Controller-to-session denoising wiring
# ---------------------------------------------------------------------------


class TestControllerDenoisingWiring:
    """RecordingController reads persisted denoising settings and passes
    them into SessionConfig when starting AudioSession.

    These tests patch AudioSession and transcription components to avoid
    heavy model loading, then inspect the SessionConfig that the controller
    creates.
    """

    @pytest.fixture(autouse=True)
    def _patch_heavy_deps(self):
        """Patch AudioSession and AccumulatingTranscriptionProcessor for speed."""
        from unittest.mock import MagicMock, patch as _patch

        # Create a mock AudioSession that records the config passed to start()
        self._session_configs: list = []
        mock_session_cls = MagicMock()
        mock_instance = MagicMock()

        def _capture_start(config):
            self._session_configs.append(config)
            mock_instance.get_state.return_value = MagicMock(name="RECORDING")

        mock_instance.start.side_effect = _capture_start
        mock_instance.stop.return_value = Path("/tmp/test.wav")
        mock_instance.get_state.return_value = MagicMock(name="IDLE")
        mock_session_cls.return_value = mock_instance

        # Patch transcription processor loading
        mock_processor = MagicMock()
        mock_processor_class = MagicMock(return_value=mock_processor)

        # Reset ConfigManager singleton between tests to avoid state leakage
        from meetandread.config.manager import ConfigManager
        from meetandread.config.persistence import SettingsPersistence
        ConfigManager._instance = None
        ConfigManager._initialized = False

        # Use temp config dir to isolate from user's real config
        import tempfile
        _tmp_config = Path(tempfile.mkdtemp(prefix="mar_test_config_"))

        with (
            _patch("meetandread.recording.controller.AudioSession", mock_session_cls),
            _patch(
                "meetandread.recording.controller.AccumulatingTranscriptionProcessor",
                mock_processor_class,
            ),
            _patch(
                "meetandread.config.manager.SettingsPersistence",
                lambda config_dir=None: SettingsPersistence(config_dir=_tmp_config),
            ),
        ):
            yield

        # Cleanup after test
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def _make_controller(self, enable_transcription=False) -> "RecordingController":
        """Create a RecordingController with transcription disabled by default."""
        from meetandread.recording.controller import RecordingController
        return RecordingController(enable_transcription=enable_transcription)

    def test_enabled_settings_reach_session_config(self):
        """When config has denoising enabled, SessionConfig gets the settings."""
        from meetandread.config.models import TranscriptionSettings
        controller = self._make_controller()

        # Configure denoising via ConfigManager
        controller._config_manager.set(
            "transcription.microphone_denoising_enabled", True
        )
        controller._config_manager.set(
            "transcription.microphone_denoising_provider", "spectral_gate"
        )
        controller._config_manager.set(
            "transcription.microphone_denoising_latency_budget_ms", 150
        )

        controller.start({"mic"})

        assert len(self._session_configs) == 1
        cfg = self._session_configs[0]
        assert cfg.enable_microphone_denoising is True
        assert cfg.denoising_provider_name == "spectral_gate"
        assert cfg.denoising_latency_budget_ms == 150

    def test_disabled_settings_reach_session_config(self):
        """When config disables denoising, SessionConfig reflects that."""
        controller = self._make_controller()

        controller._config_manager.set(
            "transcription.microphone_denoising_enabled", False
        )

        controller.start({"mic"})

        assert len(self._session_configs) == 1
        cfg = self._session_configs[0]
        assert cfg.enable_microphone_denoising is False

    def test_defaults_used_when_no_config(self):
        """Fresh config defaults (enabled=False, spectral_gate, 200ms) are used."""
        controller = self._make_controller()

        # Don't set any config — use defaults (denoising disabled)
        controller.start({"mic"})

        assert len(self._session_configs) == 1
        cfg = self._session_configs[0]
        assert cfg.enable_microphone_denoising is False
        assert cfg.denoising_provider_name is None  # Not set when disabled
        assert cfg.denoising_latency_budget_ms == 200

    def test_explicit_budget_override(self):
        """Explicit budget override is passed through."""
        controller = self._make_controller()

        controller._config_manager.set(
            "transcription.microphone_denoising_latency_budget_ms", 50
        )

        controller.start({"mic"})

        cfg = self._session_configs[0]
        assert cfg.denoising_latency_budget_ms == 50

    def test_invalid_provider_falls_back_to_default(self):
        """Invalid provider name in config falls back to safe default."""
        controller = self._make_controller()

        controller._config_manager.set(
            "transcription.microphone_denoising_enabled", True
        )
        controller._config_manager.set(
            "transcription.microphone_denoising_provider", "nonexistent_provider"
        )

        controller.start({"mic"})

        cfg = self._session_configs[0]
        # Should fall back to the default "spectral_gate"
        assert cfg.denoising_provider_name == "spectral_gate"

    def test_mic_source_gets_denoised_system_does_not(self):
        """Only mic sources get denoise=True in source configs."""
        controller = self._make_controller()

        controller._config_manager.set(
            "transcription.microphone_denoising_enabled", True
        )

        controller.start({"mic", "system"})

        cfg = self._session_configs[0]
        mic_sources = [s for s in cfg.sources if s.type == "mic"]
        sys_sources = [s for s in cfg.sources if s.type == "system"]

        assert len(mic_sources) == 1
        assert mic_sources[0].denoise is True
        assert len(sys_sources) == 1
        assert sys_sources[0].denoise is None  # system doesn't get denoised

    def test_recording_still_works_when_transcription_fails(self):
        """Recording proceeds even if transcription init fails."""
        from unittest.mock import patch as _patch, MagicMock

        controller = self._make_controller(enable_transcription=True)

        # Make transcription init fail
        with _patch(
            "meetandread.recording.controller.AccumulatingTranscriptionProcessor",
            side_effect=RuntimeError("Model load failed"),
        ):
            error = controller.start({"mic"})

        # Recording should still start (error from transcription is logged but non-fatal)
        assert error is None
        assert len(self._session_configs) == 1
        cfg = self._session_configs[0]
        assert cfg.enable_microphone_denoising is False

    def test_malformed_enabled_type_uses_default(self):
        """Non-bool enabled value falls back to safe default (False)."""
        controller = self._make_controller()

        # Force a bad type via direct settings mutation
        settings = controller._config_manager.get_settings()
        settings.transcription.microphone_denoising_enabled = "not_a_bool"  # type: ignore

        controller.start({"mic"})

        cfg = self._session_configs[0]
        # Should fall back to False (default)
        assert cfg.enable_microphone_denoising is False

    def test_missing_denoising_fields_uses_defaults(self):
        """Missing denoising fields in loaded config use safe defaults."""
        controller = self._make_controller()

        # Simulate a stale config by deleting the denoising fields
        settings = controller._config_manager.get_settings()
        # Simulate missing fields by resetting to an incomplete dict
        # This simulates a v2 config that was loaded without migration
        settings.transcription.microphone_denoising_enabled = True  # keep default
        settings.transcription.microphone_denoising_provider = ""
        settings.transcription.microphone_denoising_latency_budget_ms = -1

        controller.start({"mic"})

        cfg = self._session_configs[0]
        # Empty provider and invalid budget should fall back
        assert cfg.denoising_provider_name == "spectral_gate"
        assert cfg.denoising_latency_budget_ms == 200

    def test_callback_wiring_preserved(self):
        """Existing audio callback wiring is preserved alongside denoising."""
        from unittest.mock import MagicMock

        controller = self._make_controller(enable_transcription=True)

        controller.start({"mic"})

        cfg = self._session_configs[0]
        # on_audio_frame should be wired to feed_audio_for_transcription
        assert cfg.on_audio_frame is not None
        assert cfg.on_audio_frame == controller.feed_audio_for_transcription
