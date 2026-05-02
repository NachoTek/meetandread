"""Synthetic audio fixture helpers for diarization and denoising tests.

Generates deterministic 16 kHz mono int16 WAV files under pytest's ``tmp_path``
using a fixed RNG seed. Each fixture simulates multiple speakers with distinct
modulated formant profiles, noisy silence gaps, and known ground-truth
boundaries — all inline, no committed binary WAVs.

Usage::

    from tests.audio_fixture_helpers import generate_noisy_multi_speaker_wav

    wav_path, ground_truth = generate_noisy_multi_speaker_wav(tmp_path / "test.wav")
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SAMPLE_WIDTH = 2  # bytes (16-bit)
DEFAULT_CHANNELS = 1

# ---------------------------------------------------------------------------
# Ground-truth data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthBoundary:
    """A known speaker-change boundary in a generated fixture.

    Attributes:
        time: Boundary time in seconds.
        speaker_before: Speaker label before the boundary (e.g. "A").
        speaker_after: Speaker label after the boundary (e.g. "B").
    """

    time: float
    speaker_before: str
    speaker_after: str


@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth metadata for a generated multi-speaker fixture.

    Attributes:
        duration: Total audio duration in seconds.
        sample_rate: Sample rate in Hz.
        speakers: Sorted list of unique speaker labels.
        boundaries: Ordered list of speaker-change boundaries.
        segments: Ordered list of (start, end, speaker) triples.
        noise_level: RMS noise level used during generation.
        seed: RNG seed used for deterministic generation.
    """

    duration: float
    sample_rate: int
    speakers: List[str]
    boundaries: List[GroundTruthBoundary]
    segments: List[Tuple[float, float, str]]
    noise_level: float
    seed: int


# ---------------------------------------------------------------------------
# Speaker profile synthesis
# ---------------------------------------------------------------------------

# Formant frequencies (Hz) that give each "speaker" a distinct spectral shape.
# These are deliberately far apart so a simplistic energy-based segmentation
# could distinguish them, while remaining within the normal speech F1 range.
_SPEAKER_FORMANTS: dict[str, tuple[float, float, float]] = {
    "A": (300.0, 1800.0, 2600.0),   # low formants — "deep voice"
    "B": (600.0, 2200.0, 3200.0),   # high formants — "higher voice"
}


def _synth_speaker_chunk(
    rng: np.random.Generator,
    duration: float,
    sample_rate: int,
    speaker: str,
    amplitude: float = 0.4,
) -> np.ndarray:
    """Synthesize a short speech-like signal chunk for one speaker.

    Uses multiple harmonics modulated by a slow amplitude envelope to create
    a signal with spectral energy concentrated around the speaker's formant
    frequencies. Not real speech, but spectrally distinguishable.
    """
    n = int(duration * sample_rate)
    if n <= 0:
        return np.array([], dtype=np.float32)

    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

    formants = _SPEAKER_FORMANTS.get(speaker, _SPEAKER_FORMANTS["A"])

    # Build signal from harmonics near formants with jitter
    signal = np.zeros(n, dtype=np.float32)
    for i, f0 in enumerate(formants):
        # Add slight frequency jitter for naturalness
        jitter = rng.uniform(-5.0, 5.0)
        freq = f0 + jitter
        # Weight formants: first is strongest
        weight = 1.0 / (i + 1)
        signal += weight * np.sin(2.0 * np.pi * freq * t)

    # Modulate with a slow envelope (simulates syllable rhythm ~4 Hz)
    envelope_freq = 3.5 + rng.uniform(-0.5, 0.5)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * envelope_freq * t)
    signal *= envelope

    # Add intra-speaker noise (breath/articulation noise)
    signal += rng.standard_normal(n).astype(np.float32) * 0.02

    # Normalize to target amplitude
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (amplitude / peak)

    return signal.astype(np.float32)


def _synth_noise_gap(
    rng: np.random.Generator,
    duration: float,
    sample_rate: int,
    noise_level: float = 0.15,
) -> np.ndarray:
    """Synthesize a noisy silence gap between speaker turns."""
    n = int(duration * sample_rate)
    if n <= 0:
        return np.array([], dtype=np.float32)
    noise = rng.standard_normal(n).astype(np.float32) * noise_level
    return noise


# ---------------------------------------------------------------------------
# Main fixture generator
# ---------------------------------------------------------------------------


def generate_noisy_multi_speaker_wav(
    output_path: Path,
    *,
    duration_per_speaker: float = 2.0,
    gap_duration: float = 0.3,
    noise_level: float = 0.15,
    seed: int = 42,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speaker_turns: list[tuple[str, float]] | None = None,
) -> tuple[Path, GroundTruth]:
    """Generate a deterministic noisy multi-speaker WAV file.

    Creates a 16 kHz mono 16-bit PCM WAV file with alternating speakers
    separated by noisy silence gaps. Returns the file path and ground-truth
    metadata including exact segment boundaries.

    Args:
        output_path: Where to write the WAV file (parent dirs must exist).
        duration_per_speaker: Default duration for each speaker turn (seconds).
        gap_duration: Duration of noisy silence gaps between turns (seconds).
        noise_level: RMS amplitude of background noise.
        seed: RNG seed for deterministic generation.
        sample_rate: Audio sample rate (default 16000).
        speaker_turns: Optional explicit turns as (speaker_label, duration)
            pairs. Defaults to A/B/A/B pattern.

    Returns:
        Tuple of (wav_path, ground_truth).

    Raises:
        ValueError: If parameters are invalid.
    """
    # --- Validate parameters ---
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if duration_per_speaker <= 0:
        raise ValueError(f"duration_per_speaker must be positive, got {duration_per_speaker}")
    if gap_duration < 0:
        raise ValueError(f"gap_duration must be non-negative, got {gap_duration}")
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    rng = np.random.default_rng(seed)

    if speaker_turns is None:
        speaker_turns = [
            ("A", duration_per_speaker),
            ("B", duration_per_speaker),
            ("A", duration_per_speaker * 0.8),
            ("B", duration_per_speaker),
        ]

    # --- Build audio signal ---
    audio_chunks: list[np.ndarray] = []
    segments: list[tuple[float, float, str]] = []
    boundaries: list[GroundTruthBoundary] = []
    speakers_seen: set[str] = set()
    current_time = 0.0

    for i, (speaker, dur) in enumerate(speaker_turns):
        speakers_seen.add(speaker)

        # Add noise gap before this turn (skip for first)
        if i > 0 and gap_duration > 0:
            gap_audio = _synth_noise_gap(rng, gap_duration, sample_rate, noise_level)
            audio_chunks.append(gap_audio)
            current_time += gap_duration

        # Synthesize speaker chunk
        chunk = _synth_speaker_chunk(rng, dur, sample_rate, speaker, amplitude=0.4)
        audio_chunks.append(chunk)

        # Add background noise throughout the speaker chunk
        bg_noise = rng.standard_normal(len(chunk)).astype(np.float32) * noise_level * 0.3
        chunk_with_noise = chunk + bg_noise
        audio_chunks[-1] = chunk_with_noise

        start_time = current_time
        end_time = current_time + dur

        segments.append((start_time, end_time, speaker))

        if i > 0:
            prev_speaker = speaker_turns[i - 1][0]
            boundaries.append(GroundTruthBoundary(
                time=start_time,
                speaker_before=prev_speaker,
                speaker_after=speaker,
            ))

        current_time = end_time

    total_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
    total_duration = current_time

    # Clip to valid range
    total_audio = np.clip(total_audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    pcm = (total_audio * 32767).astype(np.int16)

    # --- Write WAV ---
    output_path = Path(output_path)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(DEFAULT_CHANNELS)
        wf.setsampwidth(DEFAULT_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    ground_truth = GroundTruth(
        duration=total_duration,
        sample_rate=sample_rate,
        speakers=sorted(speakers_seen),
        boundaries=boundaries,
        segments=segments,
        noise_level=noise_level,
        seed=seed,
    )

    return output_path, ground_truth


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_fixture_wav(
    wav_path: Path,
    *,
    expected_sample_rate: int = DEFAULT_SAMPLE_RATE,
    expected_channels: int = DEFAULT_CHANNELS,
    expected_sample_width: int = DEFAULT_SAMPLE_WIDTH,
    min_duration: float = 0.0,
) -> dict:
    """Validate a generated WAV fixture and return metadata.

    Returns a dict with keys: sample_rate, channels, sample_width, num_frames,
    duration_seconds. Raises AssertionError on mismatch.
    """
    assert wav_path.exists(), f"WAV file does not exist: {wav_path}"

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nf = wf.getnframes()

    assert sr == expected_sample_rate, (
        f"Sample rate mismatch: expected {expected_sample_rate}, got {sr}"
    )
    assert ch == expected_channels, (
        f"Channel count mismatch: expected {expected_channels}, got {ch}"
    )
    assert sw == expected_sample_width, (
        f"Sample width mismatch: expected {expected_sample_width}, got {sw}"
    )

    duration = nf / sr
    assert duration >= min_duration, (
        f"Duration {duration:.3f}s shorter than minimum {min_duration:.3f}s"
    )

    return {
        "sample_rate": sr,
        "channels": ch,
        "sample_width": sw,
        "num_frames": nf,
        "duration_seconds": duration,
    }
