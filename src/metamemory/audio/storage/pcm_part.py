"""Streaming PCM audio writer with JSON sidecar metadata.

Writes raw PCM int16 frames to disk during recording with a sidecar metadata file.
This enables crash recovery - if the process dies, the metadata and partial frames
can be recovered into a playable WAV file.

Example:
    writer = PcmPartWriter.create(
        recordings_dir=Path("~/Documents/metamemory"),
        stem="recording-2026-02-01-143045",
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
    )
    
    # Write frames during recording
    writer.write_frames_i16(audio_bytes)
    writer.flush()
    
    # Close when done (or crash happens here - file is still recoverable)
    writer.close()
"""

import json
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from metamemory.audio.storage.paths import (
    get_recordings_dir,
    get_part_filename,
    get_part_metadata_filename,
)


@dataclass(frozen=True)
class PcmMetadata:
    """Metadata for a PCM recording.
    
    Stored as JSON sidecar alongside the .pcm.part file.
    """
    sample_rate: int
    channels: int
    sample_width_bytes: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PcmMetadata":
        return cls(
            sample_rate=data["sample_rate"],
            channels=data["channels"],
            sample_width_bytes=data["sample_width_bytes"],
        )


class PcmPartWriter:
    """Streaming writer for PCM audio data.
    
    Writes raw int16 PCM frames to a .pcm.part file with a JSON sidecar
    containing metadata. The sidecar enables recovery if the process crashes.
    
    Thread-safety: Not thread-safe. If used from multiple threads, the caller
    must synchronize access to write_frames_i16(), flush(), and close().
    """
    
    def __init__(
        self,
        part_path: Path,
        metadata_path: Path,
        metadata: PcmMetadata,
        file_handle,
    ):
        self._part_path = part_path
        self._metadata_path = metadata_path
        self._metadata = metadata
        self._file = file_handle
        self._closed = False
        self._frames_written = 0
    
    @classmethod
    def create(
        cls,
        stem: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        recordings_dir: Optional[Path] = None,
    ) -> "PcmPartWriter":
        """Create a new PcmPartWriter.
        
        Args:
            stem: The recording stem (e.g., "recording-2026-02-01-143045").
            sample_rate: Audio sample rate in Hz (default: 16000).
            channels: Number of channels (default: 1 for mono).
            sample_width_bytes: Bytes per sample (default: 2 for int16).
            recordings_dir: Optional override for recordings directory.
                If None, uses ~/Documents/metamemory.
        
        Returns:
            A new PcmPartWriter ready to receive audio frames.
        
        Raises:
            FileExistsError: If the .pcm.part file already exists.
        """
        base_dir = recordings_dir or get_recordings_dir()
        
        part_path = base_dir / get_part_filename(stem)
        metadata_path = base_dir / get_part_metadata_filename(stem)
        
        if part_path.exists():
            raise FileExistsError(f"PCM part file already exists: {part_path}")
        
        # Create metadata
        metadata = PcmMetadata(
            sample_rate=sample_rate,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
        )
        
        # Write metadata sidecar first (enables recovery from empty file)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Open PCM file for binary append
        file_handle = open(part_path, "wb")
        
        return cls(
            part_path=part_path,
            metadata_path=metadata_path,
            metadata=metadata,
            file_handle=file_handle,
        )
    
    def write_frames_i16(self, frames: bytes) -> None:
        """Write int16 PCM frames to the file.
        
        Args:
            frames: Raw PCM data as little-endian int16 bytes.
        
        Raises:
            ValueError: If the writer has been closed.
        """
        if self._closed:
            raise ValueError("Cannot write to closed PcmPartWriter")
        
        # Validate: int16 = 2 bytes per sample
        if len(frames) % 2 != 0:
            raise ValueError("Frames length must be even (int16 = 2 bytes)")
        
        self._file.write(frames)
        self._frames_written += len(frames) // self._metadata.sample_width_bytes
    
    def flush(self) -> None:
        """Flush written data to disk.
        
        Call this periodically during long recordings to ensure data
        is safely on disk in case of crash.
        """
        if not self._closed:
            self._file.flush()
    
    def close(self) -> None:
        """Close the writer and release resources.
        
        After closing, the .pcm.part file is ready to be finalized to WAV.
        """
        if not self._closed:
            self._file.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    @property
    def part_path(self) -> Path:
        """Path to the .pcm.part file."""
        return self._part_path
    
    @property
    def metadata_path(self) -> Path:
        """Path to the .pcm.part.json metadata file."""
        return self._metadata_path
    
    @property
    def metadata(self) -> PcmMetadata:
        """Metadata for this recording."""
        return self._metadata
    
    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._frames_written
    
    @property
    def is_closed(self) -> bool:
        """Whether the writer has been closed."""
        return self._closed


def load_metadata(metadata_path: Path) -> PcmMetadata:
    """Load metadata from a JSON sidecar file.
    
    Args:
        metadata_path: Path to the .pcm.part.json file.
    
    Returns:
        The loaded metadata.
    
    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
        ValueError: If the metadata file is invalid.
    """
    with open(metadata_path, "r") as f:
        data = json.load(f)
    
    return PcmMetadata.from_dict(data)
