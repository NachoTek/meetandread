"""Audio recording path utilities.

Provides directory resolution and filename generation for audio recordings.
Uses platform-agnostic paths via pathlib.

Directory layout under ~/Documents/meetandread/:
    recordings/      — WAV files and PCM parts
    transcripts/     — Markdown transcript files
    speaker_signatures.db — Voice signature database (in recordings/)
    debug/           — Debug audio dumps

When custom storage paths are configured (StoragePaths in config), this module
resolves them with safe fallback to defaults on invalid/unset config.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


# Default subdirectory name within user's Documents folder
DEFAULT_RECORDINGS_SUBDIR = "meetandread"

# Subdirectory names within the base data directory
RECORDINGS_SUBDIR = "recordings"
TRANSCRIPTS_SUBDIR = "transcripts"


def _resolve_custom_path(field: str) -> Optional[Path]:
    """Resolve a custom storage path from config, returning None if unset/invalid.

    Uses lazy import to avoid circular imports between config and audio.storage.
    Only returns the path if it exists and is writable; otherwise None.
    """
    try:
        from meetandread.config.manager import get_config_manager
        cm = get_config_manager()
        settings = cm.get_settings()
        storage = getattr(settings, "storage_paths", None)
        if storage is None:
            return None
        raw = getattr(storage, field, None)
        if raw is None:
            return None
        resolved = Path(raw).expanduser().resolve()
        # Quick existence + write check
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    except (OSError, ValueError, AttributeError, ImportError, TypeError):
        return None


def get_data_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve and create the base data directory.

    Args:
        base_dir: Optional override for the base directory. If None, uses
            ~/Documents/{DEFAULT_RECORDINGS_SUBDIR}.

    Returns:
        Path to the base data directory (created if it didn't exist).
    """
    if base_dir is None:
        base_dir = Path.home() / "Documents"
    
    data_dir = base_dir / DEFAULT_RECORDINGS_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir


def get_recordings_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve and create the recordings subdirectory.

    When *base_dir* is None, checks for a custom recordings_path in config
    before falling back to the default ~/Documents/meetandread/recordings.

    Args:
        base_dir: Optional override for the base directory. If None, uses
            config custom path or default.

    Returns:
        Path to the recordings directory (created if it didn't exist).
    """
    if base_dir is None:
        custom = _resolve_custom_path("recordings_path")
        if custom is not None:
            return custom

    data_dir = get_data_dir(base_dir)
    recordings_dir = data_dir / RECORDINGS_SUBDIR
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir


def get_transcripts_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve and create the transcripts subdirectory.

    When *base_dir* is None, checks for a custom transcripts_path in config
    before falling back to the default ~/Documents/meetandread/transcripts.

    Args:
        base_dir: Optional override for the base directory. If None, uses
            config custom path or default.

    Returns:
        Path to the transcripts directory (created if it didn't exist).
    """
    if base_dir is None:
        custom = _resolve_custom_path("transcripts_path")
        if custom is not None:
            return custom

    data_dir = get_data_dir(base_dir)
    transcripts_dir = data_dir / TRANSCRIPTS_SUBDIR
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return transcripts_dir


def get_logs_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve and create the logs subdirectory.

    When *base_dir* is None, checks for a custom logs_path in config
    before falling back to the default ~/Documents/meetandread/logs.

    Args:
        base_dir: Optional override for the base directory. If None, uses
            config custom path or default.

    Returns:
        Path to the logs directory (created if it didn't exist).
    """
    if base_dir is None:
        custom = _resolve_custom_path("logs_path")
        if custom is not None:
            return custom

    data_dir = get_data_dir(base_dir)
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def new_recording_stem(now: Optional[datetime] = None) -> str:
    """Generate a timestamped filename stem for a new recording.

    Format: recording-YYYY-MM-DD-HHMMSS

    Args:
        now: Optional datetime to use. If None, uses current UTC time.

    Returns:
        Filename stem (without extension) for the recording.

    Examples:
        >>> from datetime import datetime
        >>> stem = new_recording_stem(datetime(2026, 2, 1, 14, 30, 45))
        >>> print(stem)
        recording-2026-02-01-143045
        >>>
        >>> # Default uses current time
        >>> stem = new_recording_stem()
    """
    if now is None:
        now = datetime.utcnow()
    
    return f"recording-{now.strftime('%Y-%m-%d-%H%M%S')}"


def get_part_filename(stem: str) -> str:
    """Get the .pcm.part filename for a recording stem.

    Args:
        stem: The recording stem (e.g., "recording-2026-02-01-143045").

    Returns:
        The .pcm.part filename.
    """
    return f"{stem}.pcm.part"


def get_part_metadata_filename(stem: str) -> str:
    """Get the .pcm.part.json metadata filename for a recording stem.

    Args:
        stem: The recording stem.

    Returns:
        The .pcm.part.json filename.
    """
    return f"{stem}.pcm.part.json"


def get_wav_filename(stem: str) -> str:
    """Get the .wav filename for a recording stem.

    Args:
        stem: The recording stem.

    Returns:
        The .wav filename.
    """
    return f"{stem}.wav"
