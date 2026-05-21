"""Recording file enumeration, rename, and deletion utilities.

Provides stem-based file discovery and cleanup across the recordings/
and transcripts/ directories. Used by the History tab delete action,
by scrub file management, and by the cleanup queue service.

T02 extension: rollback-minded rename, structured deletion results,
and optional directory overrides for testability and custom path support.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from meetandread.audio.storage.paths import (
    get_recordings_dir,
    get_transcripts_dir,
)

logger = logging.getLogger(__name__)

# Stem validation: allow alphanumerics, hyphens, underscores, dots (no path separators)
_STEM_RE = re.compile(r'^[A-Za-z0-9._-]+$')


@dataclass
class DeletionResult:
    """Structured result of a recording file deletion attempt.

    Attributes:
        deleted: Paths that were successfully removed.
        failed: Paths that could not be removed, with the OSError reason.
        stem: The recording stem that was targeted.
    """
    stem: str
    deleted: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (path, reason)

    @property
    def success_count(self) -> int:
        return len(self.deleted)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def all_succeeded(self) -> bool:
        return len(self.failed) == 0


@dataclass
class RenameResult:
    """Structured result of a recording rename attempt.

    Attributes:
        old_stem: The original stem.
        new_stem: The target stem.
        renamed: List of (old_path, new_path) tuples for successfully renamed files.
        rolled_back: List of paths that were reverted during rollback.
        failed: List of (path, reason) tuples for files that could not be renamed.
        rolled_back_successfully: Whether rollback completed without further errors.
    """
    old_stem: str
    new_stem: str
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    rolled_back: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    rolled_back_successfully: bool = True


def _validate_stem(stem: str) -> None:
    """Validate that a stem is safe (no path traversal, no special chars).

    Raises:
        ValueError: If the stem contains disallowed characters.
    """
    if not stem:
        raise ValueError("Stem must not be empty")
    if not _STEM_RE.match(stem):
        raise ValueError(
            f"Invalid stem: {stem!r}. "
            "Only alphanumerics, hyphens, underscores, and dots are allowed."
        )


def _resolve_dirs(
    recordings_dir: Optional[Path] = None,
    transcripts_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve directory overrides, falling back to defaults."""
    rec = recordings_dir if recordings_dir is not None else get_recordings_dir()
    tra = transcripts_dir if transcripts_dir is not None else get_transcripts_dir()
    return rec, tra


def enumerate_recording_files(
    stem: str,
    *,
    recordings_dir: Optional[Path] = None,
    transcripts_dir: Optional[Path] = None,
) -> List[Path]:
    """Find all files associated with a recording stem.

    Searches both the recordings/ and transcripts/ directories for files
    whose names start with the given stem. Matches include:

    - ``recordings/{stem}.wav``
    - ``recordings/{stem}.pcm.part``
    - ``recordings/{stem}.pcm.part.json``
    - ``transcripts/{stem}.md``
    - ``transcripts/{stem}_scrub_*.md``  (sidecars from scrub operations)

    Files that do not exist on disk are silently skipped.

    Args:
        stem: Recording stem (e.g. ``"recording-2026-02-01-143045"``).
        recordings_dir: Optional override for recordings directory.
        transcripts_dir: Optional override for transcripts directory.

    Returns:
        List of Path objects for every matching file that exists on disk.
    """
    rec_dir, tra_dir = _resolve_dirs(recordings_dir, transcripts_dir)

    candidates: List[Path] = [
        # Recordings directory
        rec_dir / f"{stem}.wav",
        rec_dir / f"{stem}.pcm.part",
        rec_dir / f"{stem}.pcm.part.json",
        # Transcripts directory
        tra_dir / f"{stem}.md",
    ]

    # Scrub sidecars: transcripts/{stem}_scrub_*.md
    if tra_dir.exists():
        candidates.extend(tra_dir.glob(f"{stem}_scrub_*.md"))

    # Filter to files that actually exist
    found = [p for p in candidates if p.is_file()]

    logger.debug(
        "Enumerated %d files for stem %s: %s",
        len(found),
        stem,
        [p.name for p in found],
    )

    return found


def _enumerate_rename_pairs(
    old_stem: str,
    new_stem: str,
    recordings_dir: Path,
    transcripts_dir: Path,
) -> List[Tuple[Path, Path]]:
    """Build (old_path, new_path) pairs for all candidate files.

    Only includes files whose old_path exists on disk.
    """
    pairs: List[Tuple[Path, Path]] = [
        # Recordings directory
        (recordings_dir / f"{old_stem}.wav", recordings_dir / f"{new_stem}.wav"),
        (recordings_dir / f"{old_stem}.pcm.part", recordings_dir / f"{new_stem}.pcm.part"),
        (recordings_dir / f"{old_stem}.pcm.part.json", recordings_dir / f"{new_stem}.pcm.part.json"),
        # Transcripts directory
        (transcripts_dir / f"{old_stem}.md", transcripts_dir / f"{new_stem}.md"),
    ]

    # Scrub sidecars
    if transcripts_dir.exists():
        for old_sidecar in transcripts_dir.glob(f"{old_stem}_scrub_*.md"):
            suffix = old_sidecar.name[len(old_stem):]  # e.g. "_scrub_v1.md"
            new_sidecar = transcripts_dir / f"{new_stem}{suffix}"
            pairs.append((old_sidecar, new_sidecar))

    return [(old, new) for old, new in pairs if old.is_file()]


def rename_recording(
    old_stem: str,
    new_stem: str,
    *,
    recordings_dir: Optional[Path] = None,
    transcripts_dir: Optional[Path] = None,
) -> RenameResult:
    """Rename all files associated with a recording stem with rollback on failure.

    Validates both stems, checks every target file for conflicts before
    any mutation, renames all files, and rolls back already-renamed files
    if a later rename fails.

    Args:
        old_stem: Current recording stem.
        new_stem: Desired new stem.
        recordings_dir: Optional override for recordings directory.
        transcripts_dir: Optional override for transcripts directory.

    Returns:
        RenameResult with details of renamed files, failures, and rollback state.

    Raises:
        ValueError: If either stem fails validation.
    """
    _validate_stem(old_stem)
    _validate_stem(new_stem)

    rec_dir, tra_dir = _resolve_dirs(recordings_dir, transcripts_dir)

    result = RenameResult(old_stem=old_stem, new_stem=new_stem)

    # Build rename pairs
    pairs = _enumerate_rename_pairs(old_stem, new_stem, rec_dir, tra_dir)
    if not pairs:
        logger.info("No files found for stem %s — nothing to rename", old_stem)
        return result

    # Pre-check: all targets must not already exist
    for old_path, new_path in pairs:
        if new_path.exists():
            reason = f"Target already exists: {new_path.name}"
            logger.warning("Rename conflict: %s", reason)
            result.failed.append((str(new_path), reason))

    if result.failed:
        logger.error(
            "Rename aborted for %s -> %s: %d target conflicts",
            old_stem, new_stem, len(result.failed),
        )
        return result

    # Rename phase with rollback on failure
    logger.info(
        "Renaming recording %s -> %s (%d files)",
        old_stem, new_stem, len(pairs),
    )

    renamed_so_far: List[Tuple[Path, Path]] = []

    try:
        for old_path, new_path in pairs:
            try:
                old_path.rename(new_path)
                renamed_so_far.append((old_path, new_path))
                result.renamed.append((str(old_path), str(new_path)))
            except OSError as exc:
                reason = f"OSError: {exc}"
                logger.error("Failed to rename %s -> %s: %s", old_path, new_path, exc)
                result.failed.append((str(old_path), reason))
                # Trigger rollback
                raise
    except OSError:
        # Rollback: rename back all successfully renamed files
        logger.warning(
            "Rolling back %d already-renamed files for %s -> %s",
            len(renamed_so_far), old_stem, new_stem,
        )
        for rollback_old, rollback_new in renamed_so_far:
            try:
                rollback_new.rename(rollback_old)
                result.rolled_back.append(str(rollback_new))
            except OSError as rb_exc:
                logger.error(
                    "ROLLBACK FAILURE: could not restore %s from %s: %s",
                    rollback_old, rollback_new, rb_exc,
                )
                result.rolled_back_successfully = False

        result.renamed = []  # Rolled back = no successful renames

    if not result.failed:
        logger.info(
            "Successfully renamed %d files: %s -> %s",
            len(result.renamed), old_stem, new_stem,
        )

    return result


def delete_recording(
    stem: str,
    *,
    recordings_dir: Optional[Path] = None,
    transcripts_dir: Optional[Path] = None,
) -> Tuple[int, List[str]]:
    """Delete all files associated with a recording stem.

    Uses :func:`enumerate_recording_files` to discover files, then removes
    each one. Missing or already-deleted files are skipped without error.

    Args:
        stem: Recording stem (e.g. ``"recording-2026-02-01-143045"``).
        recordings_dir: Optional override for recordings directory.
        transcripts_dir: Optional override for transcripts directory.

    Returns:
        Tuple of ``(count_deleted, list_of_deleted_paths)`` where each path
        is a string representation of the deleted file.
    """
    files = enumerate_recording_files(
        stem,
        recordings_dir=recordings_dir,
        transcripts_dir=transcripts_dir,
    )
    deleted: List[str] = []

    for path in files:
        try:
            path.unlink()
            deleted.append(str(path))
            logger.info("Deleted recording file: %s", path)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", path, exc)

    logger.info(
        "Deleted %d/%d files for stem %s",
        len(deleted),
        len(files),
        stem,
    )

    return len(deleted), deleted


def delete_recording_structured(
    stem: str,
    *,
    recordings_dir: Optional[Path] = None,
    transcripts_dir: Optional[Path] = None,
) -> DeletionResult:
    """Delete all files associated with a recording stem with structured results.

    Unlike :func:`delete_recording`, this returns a :class:`DeletionResult`
    that distinguishes between successful deletions and failures, enabling
    UI dialogs to show partial-failure states.

    Args:
        stem: Recording stem.
        recordings_dir: Optional override for recordings directory.
        transcripts_dir: Optional override for transcripts directory.

    Returns:
        DeletionResult with deleted and failed paths.
    """
    files = enumerate_recording_files(
        stem,
        recordings_dir=recordings_dir,
        transcripts_dir=transcripts_dir,
    )
    result = DeletionResult(stem=stem)

    for path in files:
        try:
            path.unlink()
            result.deleted.append(str(path))
            logger.info("Deleted recording file: %s", path)
        except OSError as exc:
            reason = str(exc)
            result.failed.append((str(path), reason))
            logger.warning("Failed to delete %s: %s", path, exc)

    logger.info(
        "Deleted %d/%d files for stem %s (%d failures)",
        result.success_count,
        len(files),
        stem,
        result.failure_count,
    )

    return result
