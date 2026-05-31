"""Tests for the shared audio utility functions."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from meetandread.audio.utils import load_wav_as_float32_mono


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(
    path: Path,
    samples: list[int] | np.ndarray,
    *,
    sample_rate: int = 16000,
    n_channels: int = 1,
    sample_width: int = 2,
) -> Path:
    """Write a minimal WAV file for testing."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if isinstance(samples, np.ndarray):
            samples = samples.tolist()
        raw = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw)
    return path


# ---------------------------------------------------------------------------
# Mono 16-bit WAV
# ---------------------------------------------------------------------------


class TestMono16Bit:
    def test_silence(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "silence.wav", [0] * 1600)
        audio = load_wav_as_float32_mono(wav)
        assert audio.dtype == np.float32
        assert len(audio) == 1600
        assert np.all(audio == 0.0)

    def test_nonzero_samples(self, tmp_path: Path) -> None:
        # 1600 samples of a sine-like pattern
        samples = [int(16000 * np.sin(2 * np.pi * i / 100)) for i in range(1600)]
        wav = _write_wav(tmp_path / "tone.wav", samples)
        audio = load_wav_as_float32_mono(wav)
        assert len(audio) == 1600
        assert audio.dtype == np.float32
        # First sample should match expected float value
        assert audio[0] == pytest.approx(samples[0] / 32768.0, abs=1e-6)

    def test_max_amplitude(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "max.wav", [32767])
        audio = load_wav_as_float32_mono(wav)
        assert audio[0] == pytest.approx(32767 / 32768.0)

    def test_min_amplitude(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "min.wav", [-32768])
        audio = load_wav_as_float32_mono(wav)
        assert audio[0] == pytest.approx(-32768 / 32768.0)


# ---------------------------------------------------------------------------
# Stereo downmix
# ---------------------------------------------------------------------------


class TestStereoDownmix:
    def test_stereo_averages_channels(self, tmp_path: Path) -> None:
        # Stereo: L=[1000, 2000], R=[3000, 4000] interleaved
        # Interleaved: L0, R0, L1, R1 = 1000, 3000, 2000, 4000
        samples = [1000, 3000, 2000, 4000]
        wav = _write_wav(
            tmp_path / "stereo.wav", samples, n_channels=2,
        )
        audio = load_wav_as_float32_mono(wav)
        # After downmix: (1000+3000)/2/32768, (2000+4000)/2/32768
        assert len(audio) == 2
        assert audio[0] == pytest.approx(2000 / 32768.0)
        assert audio[1] == pytest.approx(3000 / 32768.0)

    def test_stereo_silence(self, tmp_path: Path) -> None:
        samples = [0, 0, 0, 0]
        wav = _write_wav(
            tmp_path / "stereo_silence.wav", samples, n_channels=2,
        )
        audio = load_wav_as_float32_mono(wav)
        assert len(audio) == 2
        assert np.all(audio == 0.0)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    def test_8k_to_16k(self, tmp_path: Path) -> None:
        # 800 samples at 8 kHz = 0.1 s → 1600 samples at 16 kHz
        samples = [1000] * 800
        wav = _write_wav(
            tmp_path / "8k.wav", samples, sample_rate=8000,
        )
        audio = load_wav_as_float32_mono(wav)
        assert len(audio) == 1600
        # Constant input should produce constant output (within tolerance)
        assert np.all(np.abs(audio - audio[0]) < 0.01)

    def test_44100_to_16k(self, tmp_path: Path) -> None:
        # 441 samples at 44100 Hz ≈ 0.01 s → ~160 samples at 16 kHz
        samples = [5000] * 441
        wav = _write_wav(
            tmp_path / "44k.wav", samples, sample_rate=44100,
        )
        audio = load_wav_as_float32_mono(wav)
        expected_length = int(441 * 16000 / 44100)
        assert abs(len(audio) - expected_length) <= 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestUnsupportedSampleWidth:
    def test_8bit_raises(self, tmp_path: Path) -> None:
        wav = _write_wav(
            tmp_path / "8bit.wav",
            [0] * 100,
            sample_width=2,  # Must write with valid width first
        )
        # Rewrite with 8-bit
        with wave.open(str(wav), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes(bytes([128] * 100))
        with pytest.raises(ValueError, match="Unsupported sample width"):
            load_wav_as_float32_mono(wav)

    def test_24bit_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "24bit.wav"
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(16000)
            # 24-bit frames: 3 bytes per sample
            wf.writeframes(bytes([0, 0, 0] * 10))
        with pytest.raises(ValueError, match="Unsupported sample width"):
            load_wav_as_float32_mono(path)


class TestFileErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_wav_as_float32_mono(tmp_path / "nonexistent.wav")

    def test_non_wav_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_text("not a wav file")
        with pytest.raises(Exception):
            load_wav_as_float32_mono(bad)


# ---------------------------------------------------------------------------
# Integration: scrub and post_processor callers
# ---------------------------------------------------------------------------


class TestAudioUtilsIntegration:
    """Verify callers still work with the shared utility."""

    def test_scrub_runner_loads_wav(self, tmp_path: Path) -> None:
        from meetandread.transcription.scrub import ScrubRunner

        samples = [int(16000 * 0.5) for _ in range(1600)]
        wav = _write_wav(tmp_path / "test.wav", samples)
        audio = ScrubRunner._load_audio_file(wav)
        assert len(audio) == 1600
        assert audio.dtype == np.float32

    def test_post_processor_loads_wav(self, tmp_path: Path) -> None:
        from meetandread.transcription.post_processor import PostProcessingQueue
        from meetandread.config.models import AppSettings

        samples = [int(16000 * 0.5) for _ in range(1600)]
        wav = _write_wav(tmp_path / "test.wav", samples)

        queue = PostProcessingQueue(AppSettings())
        audio = queue._load_audio_file(wav)
        assert len(audio) == 1600
        assert audio.dtype == np.float32
