"""Issue Capture Mode core — the capture-directory contract (issue #104).

Spec: docs/specs/issue-reporting.md (capture-directory contract, amended
2026-09-05 per review #113). This module implements the artifact edge of
the one new test seam: what the app writes into a capture directory, and
when.

## The contract (consumed by #105-#110)

Entry — **process start ONLY** (ADR 0003). The app is launched with
``--issue-capture <DIR>``; the flag is parsed in ``main()`` before
logging is configured. There is deliberately NO API for entering capture
mode mid-process: diagnostics must cover the run from its first moment,
and a process that could switch modes could not guarantee that.

With the flag:

- ``<DIR>`` is created if missing; it is the single home for every
  capture artifact of the run.
- The run logs at full DEBUG app-wide (``logging_setup``), and the
  DEBUG log streams into ``<DIR>`` from process start.
- Every completed record is **prompt-flushed** on write: each log emit
  is flushed to the OS before the emitter returns (and fsynced — see
  ``_PromptFlushFileHandler``), so a forced kill loses at most the one
  record being written at that instant, never the run before it. No
  record is ever left held in a Python-side or OS buffer indefinitely.
- A **completion marker** — ``capture_complete.marker``, a one-line
  JSON object — is written on **clean exit only**, via the same
  prompt-flush discipline. Absence of the marker means a crashed run,
  by construction.
- Retention cleanup (``logging_setup``) never touches anything inside a
  capture directory at any age (decision 2026-09-05).

Without the flag, the run is a normal run, byte-identical in behavior
to pre-#104: no capture directory is created.

## Files in a capture directory (this ticket)

- ``meetandread_capture_YYYYMMDD_HHMMSS.log`` — the streaming DEBUG log.
- ``capture_complete.marker`` — the clean-exit marker (JSON object with
  ``finished_at`` ISO-8601 local time).

Later tickets add artifacts (JSONL event/snapshot streams, the
reporter's termination record) into the same directory; append-only
readers should tolerate at most one truncated final record per file
(``read_appendable_records`` below is that reader for line-shaped
files; the bundle assembler in #108 owns full assembly).

Everything here is structured for the two-layer test topology (ADR
0001): flag parsing, marker write/read, and the torn-record reader are
pure/fast-lane; launch-with-flag and kill-mid-run coverage lives in the
``windows``-marked subprocess tests.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from meetandread.logging_setup import CAPTURE_LOG_PREFIX  # noqa: F401 — re-export: single source of truth

# The launch flag naming the capture directory. The Issue Reporter (#107)
# passes exactly this flag; the name is contract, not style.
ISSUE_CAPTURE_FLAG = "--issue-capture"

# Filename of the clean-exit completion marker inside the capture dir.
# Tickets #105-#110 consume this name; treat it as stable contract.
COMPLETION_MARKER_NAME = "capture_complete.marker"

logger = logging.getLogger(__name__)


class CaptureModeError(Exception):
    """Capture mode cannot be entered as requested (fail before side effects).

    Raised only at process start (bad flag usage); the entry point turns
    it into exit code 2 with a stderr message — capture mode either
    starts with a usable directory from the first moment of the run, or
    it does not start at all.
    """


def parse_capture_flag(args: Optional[List[str]] = None) -> Optional[Path]:
    """Parse ``--issue-capture <DIR>`` from the process command line.

    Pure and side-effect-free: resolves nothing, creates nothing. The
    entry point calls this BEFORE logging is configured so capture mode
    owns the whole run (``main`` then configures logging with
    ``capture_dir``).

    Args:
        args: argument list; None uses ``sys.argv[1:]``.

    Returns:
        The capture directory path, or None for a normal run.

    Raises:
        CaptureModeError: the flag names an existing path that is not a
            directory. (Usage errors — e.g. the flag without a value —
            are handled by argparse itself: usage message on stderr,
            exit code 2, before the app starts.)
    """
    parser = argparse.ArgumentParser(
        prog="meetandread",
        add_help=False,
    )
    parser.add_argument(
        ISSUE_CAPTURE_FLAG,
        metavar="DIR",
        default=None,
        help="run in Issue Capture Mode, writing diagnostics into DIR",
    )
    known, _unknown = parser.parse_known_args(
        sys.argv[1:] if args is None else args
    )
    capture_dir: Optional[str] = getattr(
        known, "issue_capture", None
    )
    if capture_dir is None:
        return None
    path = Path(capture_dir)
    if path.exists() and not path.is_dir():
        raise CaptureModeError(
            f"{ISSUE_CAPTURE_FLAG}: '{capture_dir}' exists and is not a directory"
        )
    return path


class _PromptFlushFileHandler(logging.FileHandler):
    """FileHandler that flushes AND fsyncs on every emit.

    The capture-directory durability contract (spec, amended #113): a
    completed record must be on disk promptly — never parked in the
    Python-side stream buffer or the OS write-back cache — so a forced
    kill loses at most the record being written at that instant. Python
    ``flush()`` alone hands bytes to the OS; ``os.fsync`` forces them to
    the storage device. We do both, per record: capture runs are
    minutes long and write at most a few hundred KB/s of DEBUG records,
    so per-record fsync is cheap insurance against power-loss and
    hard-kill windows that flush-only leaves open.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        stream = self.stream
        if stream is None:
            return
        try:
            os.fsync(stream.fileno())
        except (OSError, ValueError, AttributeError):
            # Stream closed mid-shutdown or fd unavailable: the flush
            # above already preserved the record as far as the OS allows.
            pass


def configure_capture_logging(
    capture_dir: Path, now: Optional[datetime] = None
) -> Path:
    """Configure full-DEBUG logging streaming into *capture_dir*.

    Entry-discipline seam: called only from process start (``main``),
    never mid-process. Creates the capture directory if missing, then
    delegates to ``logging_setup.configure_logging`` with
    ``logs_dir=capture_dir, capture_mode=True`` — so the capture log
    lives INSIDE the capture directory with the shared
    ``meetandread_capture_`` prefix (issue #104: the #98 prefix
    anticipated this move), and retention, which only ever selects the
    exact normal-run shape from the normal logs dir, cannot touch it.
    The run handler is the prompt-flush one from the first record on:
    ``configure_logging`` installs ``_PromptFlushFileHandler`` for us
    (``handler_cls``), so no record ever passes through a flush-only
    handler.

    Returns the capture log file path.
    """
    from meetandread.logging_setup import configure_logging

    capture_dir.mkdir(parents=True, exist_ok=True)
    log_file = configure_logging(
        logs_dir=capture_dir,
        capture_mode=True,
        now=now,
        is_capture_dir=True,
        handler_cls=_PromptFlushFileHandler,
    )
    logger.info(
        "Issue Capture Mode: streaming diagnostics into %s", capture_dir
    )
    return log_file


def write_completion_marker(
    capture_dir: Path, now: Optional[datetime] = None
) -> Path:
    """Write the clean-exit completion marker; return its path.

    Called exactly once, on the clean-exit path only (the marker is the
    absence-proof of a crash: no marker == crashed run, by
    construction). The guarantee is deliberately one-directional — the
    marker says "the event loop returned normally with exit code 0",
    and how the run should be *interpreted* (clean stop vs user stop vs
    crash) is the reporter's termination record (#107), not this
    marker's job. One small JSON object — ``{"finished_at": ...}`` with
    local ISO-8601 time — written via the same prompt-flush discipline
    as the log: write, flush, fsync, close.
    """
    if now is None:
        now = datetime.now()
    payload = {"finished_at": now.isoformat(timespec="seconds")}
    marker = capture_dir / COMPLETION_MARKER_NAME
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return marker


def read_completion_marker(capture_dir: Path) -> Optional[dict]:
    """Read the completion marker; None when absent or unreadable.

    Pure read for the reporter side (#107) and tests: any state other
    than a parseable marker means "not a clean exit".
    """
    marker = capture_dir / COMPLETION_MARKER_NAME
    try:
        raw = marker.read_text(encoding="utf-8")
        return json.loads(raw)
    except (OSError, ValueError):
        return None


def read_appendable_records(path: Path) -> List[str]:
    """Read line-shaped records from an append-only file, tolerating one
    torn final record.

    The crash-tolerance read side of the durability contract: a forced
    kill can interrupt the single record being written, leaving a final
    line without its ``\\n``. Every completed record before it — the
    whole run up to the kill — is preserved and returned. A torn tail
    (present but unterminated) is dropped; it was never a completed
    record.

    Non-UTF-8 bytes in the torn tail are ignored (decode with errors
    replaced before splitting; the dropped tail cannot corrupt earlier
    lines because records are newline-delimited).

    This is the reader the kill-mid-run tests use; the Diagnostics
    Bundle assembler (#108) owns full assembly and is out of scope here.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not content:
        return []
    lines = content.split("\n")
    lines.pop()  # '' when the file ends with \n; else the torn record —
    # an in-flight write, never a completed one: dropped by contract.
    return lines
