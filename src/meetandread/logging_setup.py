"""Logging foundation for meetandread (issue #98).

Real log levels with root-level, all-or-nothing selection: a normal run
configures the root logger at INFO — a readable record of what the
application did — while the same codebase runs at full DEBUG when the
capture-mode flag is set (Issue Capture Mode, docs/specs/issue-reporting.md).
There is deliberately NO per-module override surface: users cannot know
which modules matter, and reproductions need maximum information.

The per-run timestamped log file under the user's Documents folder keeps
its existing shape. The stdout tee stays a console-mirroring convenience
only: stdout is never routed into the logging framework, so
transcript-bearing output may reach the console but never enters any
log stream, in any run mode.

Retention (decided 2026-09-05; user-selectable since issue #112): on
startup, normal-run log files older than the configured retention
period (default 30 days, ``logging.log_retention_days``) are deleted.
Capture-mode DEBUG logs and capture artifacts are never touched by
this cleanup at any age. Since issue #104, capture logs live inside
capture directories (``capture_mode`` module) and the retention scan
never targets a capture directory.

This module is structured for testability (ADR 0001 fast lane):
``normalize_level`` and ``select_expired_normal_logs`` are pure, and
``configure_logging`` accepts explicit ``logs_dir`` / ``now`` parameters
so tests never touch the real user tree or the wall clock.
"""

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Retention: normal-run logs older than this are deleted at startup.
LOG_RETENTION_DAYS = 30

# Filename prefixes distinguishing run kinds. Capture logs live inside
# capture directories (issue #104); the prefix keeps them excluded from
# any retention scan that ever sees them.
NORMAL_LOG_PREFIX = "meetandread_"
CAPTURE_LOG_PREFIX = "meetandread_capture_"

_LOG_FILENAME_FORMAT = "%Y%m%d_%H%M%S"

# Exact normal-run filename shape (only what our own writer produces):
# meetandread_YYYYMMDD_HHMMSS.log. The strict shape excludes capture
# logs (meetandread_capture_*) and foreign meetandread-prefixed
# artifacts by construction.
_NORMAL_LOG_NAME_RE = re.compile(r"^meetandread_\d{8}_\d{6}\.log$")

logger = logging.getLogger(__name__)


def normalize_level(capture_mode: bool) -> int:
    """Map the capture flag to the root logger level — all or nothing.

    Normal runs get INFO (quiet, readable); capture runs get full DEBUG
    app-wide. There are only these two outcomes by design: no per-module
    configuration exists anywhere.
    """
    return logging.DEBUG if capture_mode else logging.INFO


def resolve_logs_dir(base_dir: Optional[Path] = None) -> Path:
    """Resolve the logs directory through the storage-path seam.

    Defers to ``audio.storage.paths.get_logs_dir`` (imported at call
    time so test isolation that patches that module — PR #115's
    storage-isolation pattern — is honored here too). ``base_dir``
    parameterizes the home base for tests; None uses the configured or
    default user Documents location.
    """
    from meetandread.audio.storage.paths import get_logs_dir

    return get_logs_dir(base_dir=base_dir)


def select_expired_normal_logs(
    files: List[Path], cutoff: datetime
) -> List[Path]:
    """Pure selector: which files are expired normal-run logs.

    A file is selected when ALL hold:
    - its whole filename matches the exact normal-run shape
      ``meetandread_YYYYMMDD_HHMMSS.log`` (the only shape our writer
      produces, ``configure_logging``), AND its timestamp segment
      parses via the filename format — a shape-matching but invalid
      timestamp is a foreign artifact and survives, and
    - its modification time is strictly older than *cutoff*.

    The exact shape excludes capture logs (``meetandread_capture_*``)
    and any non-log or foreign meetandread-prefixed artifact by
    construction, at any age.

    Unreadable files (stat failure) are never selected — cleanup must
    not crash startup on a locked or vanished file.
    """
    expired = []
    for path in files:
        name = path.name
        if not _NORMAL_LOG_NAME_RE.match(name):
            continue
        try:
            datetime.strptime(
                name[len(NORMAL_LOG_PREFIX): -len(".log")], _LOG_FILENAME_FORMAT
            )
        except ValueError:
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        if mtime < cutoff:
            expired.append(path)
    return expired


def resolve_retention_days() -> int:
    """Resolve the user-selectable retention period (days) from config.

    Reads ``logging.log_retention_days`` through the ConfigManager
    seam. ANY failure (config unavailable, value missing or malformed)
    falls back to ``LOG_RETENTION_DAYS`` (30) — cleanup must never
    break startup, and the capture-exclusion contract holds at every
    value regardless.
    """
    try:
        from meetandread.config import get_config

        value = get_config("logging.log_retention_days")
    except Exception:
        return LOG_RETENTION_DAYS
    if isinstance(value, bool) or not isinstance(value, int):
        return LOG_RETENTION_DAYS
    if value < 1:
        return LOG_RETENTION_DAYS
    return value


def cleanup_expired_logs(
    logs_dir: Path,
    now: Optional[datetime] = None,
    retention_days: Optional[int] = None,
) -> List[Path]:
    """Delete expired normal-run logs from *logs_dir*; return what was deleted.

    The startup retention hook. The retention period is the
    user-selectable setting (``logging.log_retention_days``, default
    ``LOG_RETENTION_DAYS`` = 30) resolved when *retention_days* is
    None; tests inject the value explicitly. Capture-mode logs and any
    non-log files are never touched (see ``select_expired_normal_logs``
    — the exclusion holds at every retention value). A missing
    *logs_dir* is a no-op.

    Emits the named DEBUG event ``log_retention_cleanup`` with
    count deleted and the retention value (safe fields only, issue
    #112's named-event convention) whenever the scan runs.
    """
    if now is None:
        now = datetime.now()
    if retention_days is None:
        retention_days = resolve_retention_days()
    cutoff = now - timedelta(days=retention_days)
    try:
        entries = list(logs_dir.iterdir())
    except OSError:
        return []
    deleted = []
    for path in select_expired_normal_logs(entries, cutoff):
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            continue
    logger.debug(
        "log_retention_cleanup: deleted=%d retention_days=%d",
        len(deleted),
        retention_days,
    )
    return deleted


class TeeOutput:
    """Mirror stdout to the console only — never into the logging framework.

    Console-mirroring convenience: writes pass through to the real
    stdout and nowhere else. Stdout is never routed into the logging
    framework, so no run mode (normal INFO or capture DEBUG) can record
    transcript-bearing stdout in any log stream.
    """

    def __init__(self):
        self.stdout = sys.stdout

    def write(self, message):
        if self.stdout is not None:
            self.stdout.write(message)
            self.stdout.flush()

    def flush(self):
        if self.stdout is not None:
            self.stdout.flush()


def configure_logging(
    logs_dir: Optional[Path] = None,
    capture_mode: bool = False,
    now: Optional[datetime] = None,
    is_capture_dir: bool = False,
    handler_cls=logging.FileHandler,
) -> Path:
    """Configure root-level logging for one run and return the log file path.

    Level selection is all-or-nothing at the root (``normalize_level``).
    One timestamped file per run is created under *logs_dir* (resolved
    via the storage seam when None) — capture runs carry the capture
    prefix so retention never selects them. Expired normal logs are
    cleaned up alongside (the startup retention hook, honoring the
    user-selectable ``logging.log_retention_days`` setting since issue
    #112). Stdout is teed console-only: transcript-bearing output never
    enters the log stream in ANY mode (normal INFO or capture DEBUG).

    Issue Capture Mode (issue #104): when *logs_dir* is a capture
    directory (``is_capture_dir=True``, set by
    ``capture_mode.configure_capture_logging`` — the only supported
    caller shape), the capture log lives inside the capture directory
    and the retention scan is skipped for it: nothing inside a capture
    directory is ever a cleanup candidate at any age (retention
    decision 2026-09-05). *handler_cls* lets the capture path install
    its prompt-flush file handler (``capture_mode``) in the same call —
    default runs get the plain FileHandler, byte-identical to before.
    """
    if logs_dir is None:
        logs_dir = resolve_logs_dir()
    else:
        logs_dir.mkdir(parents=True, exist_ok=True)
    if now is None:
        now = datetime.now()

    prefix = CAPTURE_LOG_PREFIX if capture_mode else NORMAL_LOG_PREFIX
    log_file = logs_dir / f"{prefix}{now.strftime(_LOG_FILENAME_FORMAT)}.log"

    level = normalize_level(capture_mode)
    root = logging.getLogger()
    root.setLevel(level)
    # Idempotent across repeated calls (tests, future in-process flows):
    # drop any handler this module installed on a previous call before
    # adding the new run's file handler.
    for existing in list(root.handlers):
        if getattr(existing, "_meetandread_run_handler", False):
            root.removeHandler(existing)
            try:
                existing.close()
            except OSError:
                pass
    handler = handler_cls(log_file, mode='w', encoding='utf-8')
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    handler._meetandread_run_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Redirect stdout to the console only (console-mirroring tee; stdout
    # never enters the logging framework, so no mode can capture it).
    sys.stdout = TeeOutput()

    logger.info("Logging to: %s", log_file)
    logger.info(
        "Log level: %s (capture mode: %s)",
        logging.getLevelName(level),
        capture_mode,
    )

    if not is_capture_dir:
        # Retention scans the normal logs dir only — never inside a
        # capture directory (issue #104, retention decision 2026-09-05).
        cleanup_expired_logs(logs_dir, now=now)
    return log_file
