"""WAV file finalization from PCM part files.

Converts raw PCM data with sidecar metadata into standard WAV files.
Uses the stdlib `wave` module for reliable WAV header generation.
"""

import wave
from pathlib import Path
from typing import Optional

from metamemory.audio.storage.pcm_part import load_metadata, PcmMetadata


def finalize_part_to_wav(
    part_path: Path,
    wav_path: Path,
    metadata: Optional[PcmMetadata] = None,
) -> Path:
    """Finalize a .pcm.part file into a standard WAV file.

    Reads the raw PCM data and metadata sidecar, then writes a properly
    formatted WAV file with a 44-byte header.

    Args:
        part_path: Path to the .pcm.part file containing raw PCM data.
        wav_path: Path where the .wav file should be written.
        metadata: Optional pre-loaded metadata. If None, loads from
            the sidecar file (part_path with .json extension).

    Returns:
        The path to the created WAV file.

    Raises:
        FileNotFoundError: If the part file or metadata doesn't exist.
        ValueError: If the metadata is invalid or incompatible.

    Example:
        >>> from pathlib import Path
        >>> wav_path = finalize_part_to_wav(
        ...     part_path=Path("recording.pcm.part"),
        ...     wav_path=Path("recording.wav"),
        ... )
        >>> print(f"Created: {wav_path}")
    """
    part_path = Path(part_path)
    wav_path = Path(wav_path)

    if not part_path.exists():
        raise FileNotFoundError(f"PCM part file not found: {part_path}")

    # Load metadata if not provided
    if metadata is None:
        metadata_path = part_path.with_suffix(".part.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        metadata = load_metadata(metadata_path)

    # Read raw PCM data
    with open(part_path, "rb") as f:
        pcm_data = f.read()

    # Calculate derived parameters
    n_frames = len(pcm_data) // (metadata.channels * metadata.sample_width_bytes)
    n_channels = metadata.channels
    sample_width = metadata.sample_width_bytes
    frame_rate = metadata.sample_rate

    # Write WAV file using stdlib wave module
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.setnframes(n_frames)
        wav_file.writeframes(pcm_data)

    return wav_path


def finalize_stem(
    stem: str,
    recordings_dir: Path,
    delete_part: bool = True,
) -> Path:
    """Finalize a recording by stem name.

    Convenience function that finds the .pcm.part and metadata files
    by stem name and converts them to a WAV file.

    Args:
        stem: The recording stem (e.g., "recording-2026-02-01-143045").
        recordings_dir: Directory containing the .pcm.part file.
        delete_part: If True (default), delete the .pcm.part and .json files
            after successful finalization. Set to False to preserve for debugging.

    Returns:
        Path to the created WAV file.

    Raises:
        FileNotFoundError: If the part file or metadata doesn't exist.
    """
    from metamemory.audio.storage.paths import get_part_filename, get_wav_filename

    part_path = recordings_dir / get_part_filename(stem)
    wav_path = recordings_dir / get_wav_filename(stem)

    result = finalize_part_to_wav(part_path, wav_path)

    if delete_part:
        metadata_path = part_path.with_suffix(".part.json")
        part_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    return result
