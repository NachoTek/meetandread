"""Atomic file-write utility for crash-safe persistence.

Provides ``atomic_write`` — a single primitive that writes content to a
temporary file in the target directory and atomically replaces the
destination via ``os.replace``.  If anything goes wrong (disk full,
permission error, fsync failure), the original file is left untouched and
the temporary file is cleaned up.

This module must stay free of application-specific imports so it can be
used from transcript, bookmark, identity, UI, and controller modules
without creating circular dependencies.

Threat surface (Q3): only writes to paths supplied by trusted app code.
Never broadens accepted paths or logs file contents.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def atomic_write(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    fsync: bool = True,
) -> None:
    """Write *content* to *path* atomically.

    Strategy:
      1. Create a temp file in the **same directory** as *path* (so
         ``os.replace`` is a same-device rename, which is atomic on POSIX
         and near-atomic on Windows).
      2. Write *content* through ``os.fdopen``, flush, optionally
         ``os.fsync``.
      3. ``os.replace`` the temp file onto *path*.

    On **any** failure (write, fsync, or replace) the temp file is
    removed and the original *path* is left completely untouched — no
    truncation, no partial content.

    Args:
        path: Destination file path.  The parent directory **must**
            already exist (``FileNotFoundError`` is raised otherwise).
        content: Text to write.
        encoding: Text encoding (default ``utf-8``).
        fsync: If ``True`` (default), call ``os.fsync`` before replacing
            to ensure data reaches durable storage.

    Raises:
        FileNotFoundError: If the parent directory does not exist.
        OSError: Propagated from filesystem operations so callers can
            surface/log the failure.  Data is never silently dropped.
    """
    dest = Path(path)
    parent = dest.parent

    if not parent.is_dir():
        raise FileNotFoundError(
            f"Parent directory does not exist: {parent}"
        )

    temp_fd: int | None = None
    temp_path: str | None = None

    try:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(parent),
            prefix=f".{dest.name}.tmp.",
            suffix=".tmp",
        )

        with os.fdopen(temp_fd, "w", encoding=encoding) as f:
            temp_fd = None  # fd is now owned by the file object
            f.write(content)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

        os.replace(temp_path, str(dest))
        temp_path = None  # replace succeeded; nothing to clean up

    except BaseException:
        # Clean up the temp file without touching the destination.
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise
