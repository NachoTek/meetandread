"""Recovery of partial recordings after crashes.

Detects leftover .pcm.part files and converts them to playable WAV files.
Preserves originals by default for safety.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Callable

from metamemory.audio.storage.pcm_part import load_metadata
from metamemory.audio.storage.wav_finalize import finalize_part_to_wav


def find_part_files(recordings_dir: Path) -> List[Path]:
    """Find all .pcm.part files in the recordings directory.

    These are incomplete recordings that may need recovery.

    Args:
        recordings_dir: Directory to search for .pcm.part files.

    Returns:
        List of paths to .pcm.part files.

    Example:
        >>> from pathlib import Path
        >>> parts = find_part_files(Path("~/Documents/metamemory"))
        >>> for part in parts:
        ...     print(f"Found partial: {part.name}")
    """
    recordings_dir = Path(recordings_dir)
    if not recordings_dir.exists():
        return []

    return sorted(recordings_dir.glob("*.pcm.part"))


def recover_part_file(
    part_path: Path,
    recovered_wav_suffix: str = ".recovered.wav",
    backup_suffix: str = ".recovered.bak",
    delete_original: bool = False,
) -> Optional[Path]:
    """Recover a single .pcm.part file to a WAV file.

    Args:
        part_path: Path to the .pcm.part file.
        recovered_wav_suffix: Suffix for the recovered WAV file.
        backup_suffix: Suffix for backing up original files.
        delete_original: If True, delete the original .pcm.part and .json
            files after successful recovery. Default False (safer).

    Returns:
        Path to the recovered WAV file, or None if recovery failed.

    Raises:
        FileNotFoundError: If the part file or metadata doesn't exist.
    """
    part_path = Path(part_path)

    if not part_path.exists():
        raise FileNotFoundError(f"PCM part file not found: {part_path}")

    # Load metadata to validate the file
    metadata_path = part_path.with_suffix(".part.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = load_metadata(metadata_path)

    # Determine output path
    stem = part_path.stem.replace(".pcm", "")
    recordings_dir = part_path.parent
    wav_path = recordings_dir / f"{stem}{recovered_wav_suffix}"

    # Finalize to WAV
    finalize_part_to_wav(part_path, wav_path, metadata)

    if delete_original:
        # Delete original files
        part_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    else:
        # Backup original files
        part_backup = recordings_dir / f"{stem}.pcm.part{backup_suffix}"
        metadata_backup = recordings_dir / f"{stem}.pcm.part.json{backup_suffix}"

        shutil.move(str(part_path), str(part_backup))
        shutil.move(str(metadata_path), str(metadata_backup))

    return wav_path


def recover_part_files(
    recordings_dir: Path,
    recovered_wav_suffix: str = ".recovered.wav",
    backup_suffix: str = ".recovered.bak",
    delete_original: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[Path]:
    """Recover all .pcm.part files in the recordings directory.

    Args:
        recordings_dir: Directory to search for .pcm.part files.
        recovered_wav_suffix: Suffix for recovered WAV files.
        backup_suffix: Suffix for backing up original files.
        delete_original: If True, delete original files after recovery.
        progress_callback: Optional callback(current_file, current_index, total).

    Returns:
        List of paths to recovered WAV files.

    Example:
        >>> from pathlib import Path
        >>> def on_progress(file, i, total):
        ...     print(f"Recovering {i}/{total}: {file}")
        >>>
        >>> recovered = recover_part_files(
        ...     Path("~/Documents/metamemory"),
        ...     progress_callback=on_progress,
        ... )
        >>> print(f"Recovered {len(recovered)} files")
    """
    recordings_dir = Path(recordings_dir)
    part_files = find_part_files(recordings_dir)

    recovered: List[Path] = []
    total = len(part_files)

    for i, part_path in enumerate(part_files, 1):
        if progress_callback:
            progress_callback(part_path.name, i, total)

        try:
            wav_path = recover_part_file(
                part_path=part_path,
                recovered_wav_suffix=recovered_wav_suffix,
                backup_suffix=backup_suffix,
                delete_original=delete_original,
            )
            if wav_path:
                recovered.append(wav_path)
        except Exception as e:
            # Log error but continue with other files
            print(f"Failed to recover {part_path}: {e}")

    return recovered


def has_partial_recordings(recordings_dir: Path) -> bool:
    """Check if there are any partial recordings needing recovery.

    Args:
        recordings_dir: Directory to check.

    Returns:
        True if any .pcm.part files exist, False otherwise.
    """
    return len(find_part_files(recordings_dir)) > 0


def get_recovery_summary(recordings_dir: Path) -> dict:
    """Get a summary of recoverable partial recordings.

    Args:
        recordings_dir: Directory to check.

    Returns:
        Dictionary with recovery statistics.
    """
    recordings_dir = Path(recordings_dir)
    part_files = find_part_files(recordings_dir)

    total_size = sum(p.stat().st_size for p in part_files)

    return {
        "count": len(part_files),
        "total_bytes": total_size,
        "files": [p.name for p in part_files],
        "recordings_dir": str(recordings_dir),
    }
