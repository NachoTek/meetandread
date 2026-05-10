"""Tests for RecordingController.get_live_audio_samples() waveform accessor.

Covers empty, short, odd-byte, normalization, duration capping, and
copy-independence using synthetic int16 PCM bytes.  No real audio hardware
is required — the controller is instantiated directly and its private
``_live_audio_buffer`` is populated with known byte sequences.
"""

import numpy as np
import pytest

from meetandread.recording.controller import RecordingController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_int16_bytes(values):
    """Convert a list/tuple of ints in [-32768, 32767] to int16 PCM bytes."""
    return np.array(values, dtype=np.int16).tobytes()


def _samples_per_second():
    """16 kHz int16 → samples per second."""
    return 16000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl():
    """A RecordingController in a neutral (IDLE) state."""
    return RecordingController(enable_transcription=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetLiveAudioSamplesEmpty:
    """Accessor must return an empty float32 array for empty / missing buffers."""

    def test_empty_buffer_returns_empty_array(self, ctrl):
        ctrl._live_audio_buffer = bytearray()
        result = ctrl.get_live_audio_samples()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.size == 0

    def test_zero_duration_returns_empty(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([100, -200, 300]))
        result = ctrl.get_live_audio_samples(duration_seconds=0)
        assert result.size == 0

    def test_negative_duration_returns_empty(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([100, -200, 300]))
        result = ctrl.get_live_audio_samples(duration_seconds=-1.5)
        assert result.size == 0


class TestGetLiveAudioSamplesShort:
    """Short buffers (less than requested duration) still return available samples."""

    def test_buffer_shorter_than_requested(self, ctrl):
        # 3 samples at 16 kHz ≈ 0.19 ms — far less than default 1.5 s
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([1000, -2000, 3000]))
        result = ctrl.get_live_audio_samples(duration_seconds=1.5)
        assert result.dtype == np.float32
        assert len(result) == 3

    def test_single_sample_buffer(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([16000]))
        result = ctrl.get_live_audio_samples()
        assert len(result) == 1
        # Normalized: 16000 / 32768 ≈ 0.488
        assert abs(result[0] - (16000 / 32768.0)) < 1e-5


class TestGetLiveAudioSamplesOddByte:
    """Odd-byte buffers must drop the trailing byte gracefully."""

    def test_odd_byte_count_drops_trailing_byte(self, ctrl):
        # 5 bytes → 2 full int16 samples + 1 trailing byte
        raw = _synth_int16_bytes([1000, -2000]) + b'\xAB'
        ctrl._live_audio_buffer = bytearray(raw)
        result = ctrl.get_live_audio_samples()
        assert len(result) == 2
        expected = np.array([1000, -2000], dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestGetLiveAudioSamplesNormalization:
    """Samples must be normalized float32 in approximately [-1, 1]."""

    def test_max_positive_normalizes_near_one(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([32767]))
        result = ctrl.get_live_audio_samples()
        assert result[0] == pytest.approx(32767 / 32768.0, abs=1e-5)

    def test_max_negative_normalizes_near_minus_one(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([-32768]))
        result = ctrl.get_live_audio_samples()
        assert result[0] == pytest.approx(-32768 / 32768.0, abs=1e-5)

    def test_zero_sample_normalizes_to_zero(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([0]))
        result = ctrl.get_live_audio_samples()
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_mixed_values_normalize_correctly(self, ctrl):
        values = [0, 16384, -16384, 32767, -32768]
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes(values))
        result = ctrl.get_live_audio_samples()
        expected = np.array(values, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestGetLiveAudioSamplesDurationCap:
    """Requested duration must be capped to the rolling max buffer window."""

    def test_overlong_duration_capped_to_buffer_contents(self, ctrl):
        # Fill with 2 seconds worth of samples (2 * 16000 = 32000 samples)
        n_samples = 2 * _samples_per_second()
        samples = np.arange(n_samples, dtype=np.int16)
        ctrl._live_audio_buffer = bytearray(samples.tobytes())

        # Request 100 seconds — should be capped by available data
        result = ctrl.get_live_audio_samples(duration_seconds=100.0)
        assert len(result) == n_samples

    def test_returns_most_recent_samples(self, ctrl):
        # Fill with 2 seconds of samples, then request only 1 second
        n_samples_2s = 2 * _samples_per_second()
        samples = np.arange(n_samples_2s, dtype=np.int16)
        ctrl._live_audio_buffer = bytearray(samples.tobytes())

        result = ctrl.get_live_audio_samples(duration_seconds=1.0)
        expected_n = _samples_per_second()
        assert len(result) == expected_n
        # Should be the LAST second worth (samples 16000..31999)
        expected_start = n_samples_2s - expected_n
        expected = samples[expected_start:].astype(np.float32) / 32768.0
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestGetLiveAudioSamplesCopyIndependence:
    """Returned array must not alias the internal buffer."""

    def test_returned_array_is_independent_of_buffer(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([1000, -2000, 3000]))
        result = ctrl.get_live_audio_samples()

        # Mutate internal buffer
        ctrl._live_audio_buffer.clear()
        # Result should be unchanged
        assert len(result) == 3
        assert result[0] != 0  # sanity: didn't get zeroed out

    def test_mutating_result_does_not_affect_buffer(self, ctrl):
        ctrl._live_audio_buffer = bytearray(_synth_int16_bytes([1000, -2000]))
        result = ctrl.get_live_audio_samples()

        result[0] = 999.0
        # Re-read from buffer should still produce original values
        result2 = ctrl.get_live_audio_samples()
        assert result2[0] != 999.0


class TestGetLiveAudioSamplesDefaults:
    """Default parameter behavior."""

    def test_default_duration_is_1_5_seconds(self, ctrl):
        # Fill with exactly 2 seconds
        n_samples_2s = 2 * _samples_per_second()
        samples = np.arange(n_samples_2s, dtype=np.int16)
        ctrl._live_audio_buffer = bytearray(samples.tobytes())

        result = ctrl.get_live_audio_samples()
        expected_n = int(1.5 * _samples_per_second())
        assert len(result) == expected_n
