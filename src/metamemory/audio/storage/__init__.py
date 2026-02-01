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
from metamemory.audio.storage.pcm_part import (
    PcmPartWriter,
    PcmMetadata,
    load_metadata,
)
from metamemory.audio.storage.wav_finalize import (
    finalize_part_to_wav,
    finalize_stem,
)
from metamemory.audio.storage.recovery import (
    find_part_files,
    recover_part_file,
    recover_part_files,
    has_partial_recordings,
    get_recovery_summary,
)

__all__ = [
    # paths
    "get_recordings_dir",
    "new_recording_stem",
    "get_part_filename",
    "get_part_metadata_filename",
    "get_wav_filename",
    # pcm_part
    "PcmPartWriter",
    "PcmMetadata",
    "load_metadata",
    # wav_finalize
    "finalize_part_to_wav",
    "finalize_stem",
    # recovery
    "find_part_files",
    "recover_part_file",
    "recover_part_files",
    "has_partial_recordings",
    "get_recovery_summary",
]
