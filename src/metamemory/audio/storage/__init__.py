"""Crash-safe audio storage primitives.

Provides on-disk recording with streaming writes, WAV finalization,
and recovery of partial recordings after crashes.

Key components:
- paths: Recording directory resolution and filename generation
- pcm_part: Streaming PCM writer with JSON sidecar metadata
- wav_finalize: Convert .pcm.part files to standard WAV format
- recovery: Detect and recover leftover partial recordings
"""

from metamemory.audio.storage.paths import (
    get_recordings_dir,
    new_recording_stem,
    get_part_filename,
    get_part_metadata_filename,
    get_wav_filename,
)

__all__ = [
    "get_recordings_dir",
    "new_recording_stem",
    "get_part_filename",
    "get_part_metadata_filename",
    "get_wav_filename",
]
