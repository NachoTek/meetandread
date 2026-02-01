"""Audio recording path utilities.

Provides directory resolution and filename generation for audio recordings.
Uses platform-agnostic paths via pathlib.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


# Default subdirectory name within user's Documents folder
DEFAULT_RECORDINGS_SUBDIR = "metamemory"


def get_recordings_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve and create the recordings directory.

    Args:
        base_dir: Optional override for the base directory. If None, uses
            ~/Documents/{DEFAULT_RECORDINGS_SUBDIR}.

    Returns:
        Path to the recordings directory (created if it didn't exist).

    Examples:
        >>> # Default behavior - creates ~/Documents/metamemory/
        >>> recordings_dir = get_recordings_dir()
        >>>
        >>> # Test override - use temporary directory
        >>> from pathlib import Path
        >>> recordings_dir = get_recordings_dir(base_dir=Path("/tmp/test"))
    """
    if base_dir is None:
        # Use user's Documents folder
        base_dir = Path.home() / "Documents"
    
    recordings_dir = base_dir / DEFAULT_RECORDINGS_SUBDIR
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    return recordings_dir


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
