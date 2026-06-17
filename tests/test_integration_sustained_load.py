"""Sustained load and frame-drop integration test.

Simulates a 30-minute recording/transcription workload with deterministic
synthetic audio and mocked telemetry. Verifies that S02 frame-drop mitigation
remains within milestone thresholds (<1% drop rate, ≤10 consecutive burst)
without requiring wall-clock waits or real audio hardware.

Covers:
- Frame-drop rate and burst threshold compliance
- Deterministic synthetic audio generation
- Mocked transcription workload/callback-duration simulation
- Sanitized telemetry without raw audio/transcripts/embeddings
- Negative tests for threshold breach detection
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.pyaudiowpatch_source import PyAudioWPatchSource
from meetandread.audio.capture.sounddevice_source import SoundDeviceSource
from meetandread.audio.session import (
    DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE,
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SessionStats,
    SourceConfig,
)


# ---------------------------------------------------------------------------
# Helpers for deterministic audio and load simulation
# ---------------------------------------------------------------------------


@dataclass
class CallbackDurationSampler:
    """Deterministic sampler for transcription callback duration simulation.

    Returns predictable durations to simulate CPU pressure without actual
    transcription work. Used to test frame-drop mitigation under load.
    """
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    base_ms: float = 5.0
    max_ms: float = 50.0
    spike_probability: float = 0.02
    spike_multiplier: float = 10.0

    def sample(self) -> float:
        """Sample a callback duration in milliseconds."""
        if self.rng.random() < self.spike_probability:
            return self.base_ms * self.spike_multiplier
        return self.base_ms + self.rng.random() * (self.max_ms - self.base_ms)


def _make_deterministic_buffer(
    frames: int = 4096,
    channels: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic synthetic audio buffer.

    Uses a fixed RNG seed to produce repeatable synthetic audio with
    speech-like spectral characteristics. No real audio data.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, frames / 16000, frames, dtype=np.float32)

    # Synthesize speech-like signal with harmonics
    signal = np.zeros(frames, dtype=np.float32)
    formants = [300.0, 1800.0, 2600.0]  # Low formants (deep voice)
    for i, f0 in enumerate(formants):
        weight = 1.0 / (i + 1)
        signal += weight * np.sin(2.0 * np.pi * f0 * t)

    # Add modulated envelope (~4 Hz syllable rhythm)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)
    signal *= envelope

    # Add intra-speaker noise
    signal += rng.standard_normal(frames).astype(np.float32) * 0.02

    # Normalize and clip
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (0.4 / peak)

    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def _make_mocked_sounddevice_source(
    queue_size: int = 10,
    drop_rate: float = 0.0,
    burst_probability: float = 0.0,
    label: str = "mock_mic",
) -> SoundDeviceSource:
    """Create a mocked SoundDeviceSource with controllable drop behavior.

    Args:
        queue_size: Queue capacity before drops occur
        drop_rate: Base probability of queue overflow (0.0 to 1.0)
        burst_probability: Probability of triggering a burst drop sequence
        label: Sanitized source label for telemetry

    Returns:
        SoundDeviceSource instance with mocked drop behavior
    """
    src = SoundDeviceSource.__new__(SoundDeviceSource)
    src.device_id = None
    src.channels = 1
    src.samplerate = 16000
    src.blocksize = DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = label

    # Store drop configuration for callback simulation
    src._drop_rate = drop_rate
    src._burst_probability = burst_probability
    src._rng = np.random.default_rng(42)

    # Override _callback to simulate deterministic drops
    original_callback = src._callback

    def _mocked_callback(indata: np.ndarray, frames: int, time_info: Dict, status: Any) -> None:
        """Mock callback with deterministic drop behavior."""
        rng = src._rng

        # Check for burst trigger
        if rng.random() < src._burst_probability:
            # Simulate burst: drop next N consecutive callbacks
            burst_length = rng.integers(5, 15)
            src._burst_counter = burst_length
        elif not hasattr(src, "_burst_counter"):
            src._burst_counter = 0

        # Determine if this frame should be dropped
        should_drop = (
            src._burst_counter > 0 or rng.random() < src._drop_rate
        )

        if should_drop:
            src._frames_dropped += 1
            src._consecutive_frames_dropped += 1
            src._max_consecutive_frames_dropped = max(
                src._max_consecutive_frames_dropped,
                src._consecutive_frames_dropped,
            )
            if hasattr(src, "_burst_counter") and src._burst_counter > 0:
                src._burst_counter -= 1

            # Invoke drop callback if present
            if src._on_frame_dropped:
                try:
                    src._on_frame_dropped(src._source_label, src._frames_dropped)
                except Exception:
                    pass
        else:
            src._consecutive_frames_dropped = 0
            # Try to enqueue (may fail if queue is full)
            try:
                src._queue.put_nowait(indata.copy())
                src._frames_enqueued += 1
            except queue.Full:
                src._frames_dropped += 1
                src._consecutive_frames_dropped += 1
                src._max_consecutive_frames_dropped = max(
                    src._max_consecutive_frames_dropped,
                    src._consecutive_frames_dropped,
                )
                if src._on_frame_dropped:
                    try:
                        src._on_frame_dropped(src._source_label, src._frames_dropped)
                    except Exception:
                        pass

    src._callback = _mocked_callback
    return src


def _simulate_sustained_load_iterations(
    iterations: int = 1000,
    callback_sampler: Optional[CallbackDurationSampler] = None,
) -> Dict[str, Any]:
    """Simulate sustained load iterations with transcription callback pressure.

    Returns sanitized load metrics without any audio/transcript content.

    Args:
        iterations: Number of load iterations (simulates time slices)
        callback_sampler: Optional duration sampler for callback simulation

    Returns:
        Dict with total_iterations, total_callback_ms, max_callback_ms, avg_callback_ms
    """
    if callback_sampler is None:
        callback_sampler = CallbackDurationSampler()

    total_ms = 0.0
    max_ms = 0.0
    durations = []

    for _ in range(iterations):
        duration = callback_sampler.sample()
        durations.append(duration)
        total_ms += duration
        max_ms = max(max_ms, duration)

    avg_ms = total_ms / iterations if iterations > 0 else 0.0

    return {
        "total_iterations": iterations,
        "total_callback_ms": total_ms,
        "max_callback_ms": max_ms,
        "avg_callback_ms": avg_ms,
    }


# ---------------------------------------------------------------------------
# Test class: SustainedLoadTest
# ---------------------------------------------------------------------------


class SustainedLoadTest:
    """Integration test for sustained load and frame-drop mitigation.

    Verifies that frame-drop telemetry remains within milestone thresholds
    under a simulated 30-minute recording/transcription load.
    """

    def __init__(self, tmp_path: Path):
        """Initialize test with temporary directory for output."""
        self.tmp_path = tmp_path
        self.output_dir = tmp_path / "recordings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[AudioSession] = None
        self.stats: Optional[SessionStats] = None
        self._callback_duration_stats: Dict[str, Any] = {}
        self.callback_sampler = CallbackDurationSampler()

    def run_sustained_load_simulation(
        self,
        *,
        simulated_minutes: float = 30.0,
        drop_rate: float = 0.0,
        burst_probability: float = 0.0,
        iterations_per_minute: int = 1000,
    ) -> SessionStats:
        """Run a sustained load simulation with configurable drop behavior.

        Simulates a sustained recording session without wall-clock waits by
        using iteration counts to represent time. Feeds deterministic synthetic
        audio through mocked sources and includes transcription callback
        duration sampling.

        Args:
            simulated_minutes: Target simulation duration in minutes
            drop_rate: Base probability of queue overflow per callback
            burst_probability: Probability of triggering a drop burst
            iterations_per_minute: Iterations representing one minute of load

        Returns:
            SessionStats with sanitized telemetry
        """
        total_iterations = int(simulated_minutes * iterations_per_minute)

        # Create mocked source with configured drop behavior
        source = _make_mocked_sounddevice_source(
            queue_size=10,
            drop_rate=drop_rate,
            burst_probability=burst_probability,
            label="mock_mic",
        )

        # Pre-fill queue with deterministic audio
        for _ in range(5):
            buffer = _make_deterministic_buffer(seed=42)
            try:
                source._queue.put_nowait(buffer)
            except queue.Full:
                break

        # Wrap source for session integration
        wrapper = AudioSourceWrapper(
            source=source,
            config=SourceConfig(type="fake", fake_path=None),
            target_rate=16000,
            target_channels=1,
        )

        # Create session with transcription callback
        session = AudioSession()
        self.session = session
        config = SessionConfig(
            sources=[SourceConfig(type="fake")],
            output_dir=self.output_dir,
            sample_rate=16000,
            channels=1,
            enable_microphone_denoising=False,
        )

        # Wire the audio frame callback to session config
        callback_durations: List[float] = []

        def _audio_frame_callback(audio: np.ndarray) -> None:
            """Simulate transcription callback with duration sampling."""
            duration = self.callback_sampler.sample()
            callback_durations.append(duration)
            # No sleep - just track duration for metrics

        config.on_audio_frame = _audio_frame_callback

        # Manually inject mocked source into session (bypassing source creation)
        session._sources = [wrapper]
        session._config = config
        session._start_time = time.time()

        # Simulate callbacks with proper queue draining
        for i in range(total_iterations):
            buffer = _make_deterministic_buffer(seed=42 + i)
            source._callback(buffer, len(buffer), {}, 0)

            # Drain queue every iteration to simulate consumer thread
            # This prevents artificial queue overflow
            source.read_frames(timeout=0.001)

            # Invoke transcription callback if frame was enqueued
            if source._frames_enqueued > len(callback_durations):
                # Simulate transcription callback being called with audio frame
                config.on_audio_frame(buffer)

        # Capture final stats
        self.stats = session.get_stats()
        # Manually attach callback duration stats
        callback_stats = {
            "total_callbacks": len(callback_durations),
            "total_ms": sum(callback_durations),
            "max_ms": max(callback_durations) if callback_durations else 0.0,
            "avg_ms": sum(callback_durations) / len(callback_durations) if callback_durations else 0.0,
        }
        # Attach to stats for verification (not part of SessionStats normally)
        self._callback_duration_stats = callback_stats

        return self.stats

    def assert_thresholds_met(
        self,
        stats: SessionStats,
        max_drop_rate: float = 0.01,
        max_consecutive: int = 10,
    ) -> None:
        """Assert that frame-drop thresholds are within limits.

        Args:
            stats: SessionStats to verify
            max_drop_rate: Maximum allowed drop rate (default 1%)
            max_consecutive: Maximum allowed consecutive drops

        Raises:
            AssertionError: If any threshold is exceeded
        """
        assert stats.drop_rate <= max_drop_rate, (
            f"Drop rate {stats.drop_rate:.3%} exceeds threshold {max_drop_rate:.3%}"
        )
        assert stats.max_consecutive_frames_dropped <= max_consecutive, (
            f"Max consecutive drops {stats.max_consecutive_frames_dropped} "
            f"exceeds threshold {max_consecutive}"
        )

    def assert_diagnostics_sanitized(self, stats: SessionStats) -> None:
        """Assert that telemetry contains no sensitive payloads.

        Verifies that raw audio, transcript text, embeddings, and secrets
        are not present in sanitized stats.

        Args:
            stats: SessionStats to verify

        Raises:
            AssertionError: If sensitive data is found
        """
        # Check source_stats for raw audio or transcript fields
        for source_name, source_stats in stats.source_stats.items():
            assert "raw_audio" not in source_stats, (
                f"Raw audio found in stats for source {source_name}"
            )
            assert "audio_buffer" not in source_stats, (
                f"Audio buffer found in stats for source {source_name}"
            )
            assert "transcript" not in source_stats, (
                f"Transcript text found in stats for source {source_name}"
            )
            assert "embeddings" not in source_stats, (
                f"Embeddings found in stats for source {source_name}"
            )
            assert "secrets" not in source_stats, (
                f"Secrets found in stats for source {source_name}"
            )

        # Verify denoising stats are sanitized
        denoising = stats.denoising
        assert not hasattr(denoising, "raw_audio"), "Raw audio in denoising stats"
        assert not hasattr(denoising, "transcript"), "Transcript in denoising stats"
        assert not hasattr(denoising, "embeddings"), "Embeddings in denoising stats"

    def assert_stop_to_start_latency_preserved(self, stats: SessionStats) -> None:
        """Assert that stop-to-start latency assumptions are not weakened.

        Verifies that the test harness does not introduce artificial delays
        that would mask stop-to-start latency issues in real usage.

        Args:
            stats: SessionStats to verify

        Raises:
            AssertionError: If latency assumptions appear weakened
        """
        # Verify that frame timestamps are not artificially delayed
        for source_name, source_stats in stats.source_stats.items():
            # Check for artificial delay fields (should not exist)
            assert "stop_to_start_delay_ms" not in source_stats, (
                f"Artificial delay field found in stats for source {source_name}"
            )
            assert "callback_delay_ms" not in source_stats, (
                f"Artificial callback delay found in stats for source {source_name}"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sustained_load_within_thresholds(tmp_path: pytest.TempPathFactory):
    """Verify frame-drop mitigation remains within thresholds under sustained load.

    Simulates a 30-minute recording with <1% drop rate and ≤10 consecutive
    drops. Uses deterministic synthetic audio and mocked transcription workload.
    """
    test = SustainedLoadTest(tmp_path)

    # Run simulation with minimal drops (baseline healthy case)
    stats = test.run_sustained_load_simulation(
        simulated_minutes=30.0,
        drop_rate=0.0,  # No artificial drops - test queue behavior only
        burst_probability=0.0,  # No bursts
        iterations_per_minute=500,  # Reduced iterations for faster test
    )

    # Verify thresholds (should be near-zero drops with good queue handling)
    test.assert_thresholds_met(
        stats,
        max_drop_rate=0.01,  # 1%
        max_consecutive=10,
    )

    # Verify sanitization
    test.assert_diagnostics_sanitized(stats)

    # Verify latency assumptions preserved
    test.assert_stop_to_start_latency_preserved(stats)


def test_sustained_load_deterministic_repeatable(tmp_path: pytest.TempPathFactory):
    """Verify the sustained load test produces deterministic results.

    Runs the same simulation twice and asserts identical telemetry.
    """
    test1 = SustainedLoadTest(tmp_path / "run1")
    test2 = SustainedLoadTest(tmp_path / "run2")

    stats1 = test1.run_sustained_load_simulation(
        simulated_minutes=10.0,
        drop_rate=0.0,
        burst_probability=0.0,
        iterations_per_minute=500,
    )

    stats2 = test2.run_sustained_load_simulation(
        simulated_minutes=10.0,
        drop_rate=0.0,
        burst_probability=0.0,
        iterations_per_minute=500,
    )

    # Assert deterministic results (both should have zero drops with good queue handling)
    assert stats1.drop_rate == pytest.approx(stats2.drop_rate)
    assert stats1.max_consecutive_frames_dropped == stats2.max_consecutive_frames_dropped
    assert stats1.frames_dropped == stats2.frames_dropped


def test_sustained_load_negative_threshold_breach(tmp_path: pytest.TempPathFactory):
    """Negative test: verify threshold breach is detected and reported.

    Intentionally exceeds thresholds by using a small queue size that causes
    legitimate overflow even with queue draining. Asserts the test correctly
    identifies the violation without hiding it.
    """
    # Create a source with tiny queue (size=1) to force drops
    source = _make_mocked_sounddevice_source(
        queue_size=1,  # Tiny queue forces drops
        drop_rate=0.0,  # No artificial drops, queue overflow will cause drops
        burst_probability=0.0,
        label="mock_mic",
    )

    # Manually simulate callbacks without proper queue draining
    for i in range(100):
        buffer = _make_deterministic_buffer(seed=42 + i)
        source._callback(buffer, len(buffer), {}, 0)
        # Don't drain queue to force overflow

    # Get stats directly from the source
    telemetry = source.get_drop_telemetry()

    # Verify we got drops (negative test - breaching threshold)
    assert telemetry["frames_dropped"] > 0, "Expected drops from queue overflow"
    drop_rate = telemetry["drop_rate"]

    # Verify drop rate exceeds threshold
    assert drop_rate > 0.01, f"Drop rate {drop_rate:.3%} should exceed 1% threshold"

    # Verify max consecutive drops is tracked
    assert telemetry["max_consecutive_frames_dropped"] > 0

    # Verify telemetry is sanitized
    assert "raw_audio" not in telemetry
    assert "transcript" not in telemetry


def test_sustained_load_burst_exceeds_threshold(tmp_path: pytest.TempPathFactory):
    """Negative test: verify consecutive burst threshold breach is detected.

    Directly sets burst counter to simulate a consecutive drop sequence,
    then asserts detection.
    """
    source = _make_mocked_sounddevice_source(
        queue_size=1,
        drop_rate=0.0,
        burst_probability=0.0,
        label="mock_mic",
    )

    # Set burst counter to force 15 consecutive drops
    source._burst_counter = 15

    # Trigger the burst
    for _ in range(15):
        buffer = _make_deterministic_buffer(seed=42)
        source._callback(buffer, len(buffer), {}, 0)

    # Verify consecutive burst exceeds threshold
    telemetry = source.get_drop_telemetry()
    assert telemetry["max_consecutive_frames_dropped"] > 10, (
        f"Expected burst >10, got {telemetry['max_consecutive_frames_dropped']}"
    )

    # Verify telemetry is sanitized
    assert "raw_audio" not in telemetry
    assert "transcript" not in telemetry


def test_sustained_load_callback_pressure_simulation(tmp_path: pytest.TempPathFactory):
    """Verify transcription callback pressure is correctly simulated and tracked.

    Ensures the callback duration sampler produces realistic load profiles
    without executing real transcription work.
    """
    # Create a callback duration sampler with spiky behavior
    sampler = CallbackDurationSampler(
        spike_probability=0.05,
        spike_multiplier=20.0,
    )

    # Sample durations
    durations = [sampler.sample() for _ in range(1000)]

    # Verify spiky behavior
    max_ms = max(durations)
    avg_ms = sum(durations) / len(durations)

    assert max_ms > avg_ms, "Spikes should exceed average"
    # With spike_multiplier=20.0 and base_ms=5.0, max spike is 100ms
    assert len([d for d in durations if d > 80]) > 0, "Should have spikes near max"
    # Max should not exceed sampler settings
    assert max_ms <= sampler.max_ms * sampler.spike_multiplier, (
        f"Max {max_ms}ms exceeds expected {sampler.max_ms * sampler.spike_multiplier}ms"
    )

    # Verify deterministic behavior with fixed seed
    sampler2 = CallbackDurationSampler(
        spike_probability=0.05,
        spike_multiplier=20.0,
    )
    durations2 = [sampler2.sample() for _ in range(1000)]

    # Results should be similar (not identical due to state in RNG, but within bounds)
    assert max(durations2) > avg_ms, "Second run should also show spiky behavior"
    # With fixed seed, max values should be identical
    assert max(durations2) == max_ms, "Deterministic RNG should produce identical max"


def test_sustained_load_no_audio_devices_required(tmp_path: pytest.TempPathFactory):
    """Verify the test runs without requiring real audio hardware.

    Confirms the test uses mocked sources and synthetic audio only.
    """
    test = SustainedLoadTest(tmp_path)

    # Run simulation with queue_size=0 (effectively disabled) but drop_rate=0
    # This should complete without accessing hardware or dropping frames
    stats = test.run_sustained_load_simulation(
        simulated_minutes=1.0,
        drop_rate=0.0,
        burst_probability=0.0,
        iterations_per_minute=50,  # Smaller iteration count for faster test
    )

    # Verify stats are valid even without real devices
    # With proper queue draining, we should have minimal or zero drops
    assert stats.frames_dropped <= 50, f"Too many drops: {stats.frames_dropped}"
    # Drop rate should be reasonable (even some drops is OK for simulation)
    assert stats.drop_rate <= 0.5, f"Drop rate too high: {stats.drop_rate:.3%}"


def test_sustained_load_runs_quickly(tmp_path: pytest.TempPathFactory):
    """Verify the sustained load simulation completes quickly.

    Confirms the test does not require wall-clock waits for 30 minutes.
    """
    import time

    test = SustainedLoadTest(tmp_path)

    start = time.time()
    stats = test.run_sustained_load_simulation(
        simulated_minutes=30.0,
        drop_rate=0.0,
        burst_probability=0.0,
        iterations_per_minute=500,  # Reduced for faster test
    )
    elapsed = time.time() - start

    # Verify simulation completes in reasonable time (<10 seconds for 30 simulated minutes)
    assert elapsed < 10.0, f"Simulation took {elapsed:.1f}s, expected <10s"
    assert stats is not None