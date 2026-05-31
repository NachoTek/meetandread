"""Shared audio utility functions.

Centralizes WAV loading helpers used across transcription and scrub
pipelines so that sample-width validation, stereo downmix, and
resampling are tested and maintained in one place.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


def load_wav_as_float32_mono(path: Path | str) -> np.ndarray:
    """Load a WAV file as a float32 mono numpy array resampled to 16 kHz.

    Reads 16-bit PCM WAV files using the standard-library ``wave`` module.
    Stereo files are down-mixed to mono by averaging channels.

    Args:
        path: Path to the WAV file.

    Returns:
        1-D float32 numpy array with audio samples in the range [-1, 1],
        resampled to 16 000 Hz if the source uses a different sample rate.

    Raises:
        ValueError: If the sample width is not 16-bit (2 bytes).
        FileNotFoundError: If *path* does not exist.
        OSError: Propagated from ``wave.open`` for unreadable files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sample_width != 2:
            raise ValueError(
                f"Unsupported sample width: {sample_width} "
                "(only 16-bit PCM is supported)"
            )

        fmt = f"{n_frames * n_channels}h"
        samples = struct.unpack(fmt, raw_data)
        audio = np.array(samples, dtype=np.float32) / 32768.0

        # Stereo downmix: average left and right channels
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        elif n_channels > 2:
            # Multi-channel: average all channels
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(
                indices, np.arange(len(audio)), audio
            ).astype(np.float32)

        return audio
