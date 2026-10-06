"""Resource Snapshot series — the periodic resource record inside Issue
Capture Mode (issue #106, docs/specs/issue-reporting.md).

Spec: story 25 and the "Captured content" / cadence decisions; the
durability contract is the owner-approved amendment of spec review
#113. Built ON the #104 capture-directory contract (``capture_mode``):
same directory, same prompt-flush discipline, same torn-tail tolerance
— never altering it — and ON the existing ResourceMonitor building
block (``performance.monitor``): the series IS a ResourceMonitor whose
``on_snapshot`` callback persists each snapshot the moment it is
taken.

## The contract (consumed by the Diagnostics Bundle assembler #108)

File — exactly one per capture directory:

- ``resource_snapshots.jsonl`` — append-only JSONL, one line per
  snapshot, written as the snapshot is taken. Every completed record
  is written, flushed, AND fsynced to disk before the poll callback
  returns (the shared ``durable_jsonl.DurableJSONLWriter`` — the same
  writer the Interaction Trace uses), so a forced kill mid-run
  preserves every completed snapshot; at most the one record being
  written at the instant of the kill is lost (the ``windows``-marked
  subprocess tests are the authoritative proof). Assembly tolerates
  exactly one truncated final record (``read_snapshots`` reads through
  the #104 torn-record reader).

Cadence (owner-approved): the ResourceMonitor's existing 2-second
default poll interval — ``SNAPSHOT_INTERVAL_MS`` — with full history,
no ring buffer: capture runs are minutes long, and crash tolerance
beats truncation. The monitor also polls once immediately on start
(``ResourceMonitor.start``), so the series' first record lands at the
first moment of the run.

Record schema — one JSON object per line:

- ``ts`` — ISO-8601 local timestamp of the snapshot.
- ``ram_percent``, ``cpu_percent`` — the ResourceSnapshot percentages.
- ``available_ram_gb``, ``total_ram_gb`` — the ResourceSnapshot RAM
  figures (the building block's full shape, carried unchanged).

## Failure semantics

- A write failure (or a poll that cannot be persisted) is logged at
  ERROR and swallowed: diagnostics must not crash the run being
  diagnosed. The series stays append-only, so a failed record never
  corrupts the order of the records around it.
- Install refusals (a series file from another run, or a second
  conflicting directory) are startup failures:
  :class:`SnapshotSeriesError` at process start, the same refusal
  class as the #105 trace install — never a half-started run.

## Normal runs

Without capture mode there is no series: ``install_snapshot_series``
is called only from the capture-mode startup path (``main`` with a
``--issue-capture`` dir), and ``persist_snapshot`` without an
installed series is a silent no-op that creates nothing anywhere.

Test hygiene: the ResourceMonitor driving the series registers in the
monitor's class-level weakref registry (issue #86), so the test
suite's existing ``ResourceMonitor.stop_all()`` teardown covers
abandoned series exactly as it covers abandoned panels.

This module's capture-side surface (install/persist/read) is
stdlib-plus-psutil-free — it imports the monitor lazily so the module
stays importable (and its install contract testable) in any lane; the
timer-driven production wiring is exercised end to end under the
Windows venv (ADR 0001).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from meetandread.capture_mode import read_appendable_records
from meetandread.durable_jsonl import DurableJSONLWriter

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from meetandread.performance.monitor import ResourceMonitor

# Filename of the snapshot series inside the capture dir. One capture
# directory holds one run (#104) and one series; treat as stable
# contract (#108's assembler consumes it).
SNAPSHOT_FILE_NAME = "resource_snapshots.jsonl"

# The series cadence: the ResourceMonitor building block's existing
# default poll interval (owner-approved at the seam checkpoint). Kept
# as a named constant AND asserted equal to the monitor's default in
# the tests, so a future monitor change must consciously revisit the
# series cadence rather than silently drifting.
SNAPSHOT_INTERVAL_MS = 2000

logger = logging.getLogger(__name__)


class SnapshotSeriesError(Exception):
    """The snapshot series cannot be installed as requested.

    Raised only at process start (bad capture-dir state — a series
    file from another run, or a second conflicting directory); the
    entry point treats it as a capture-mode startup failure.
    """


class _SnapshotSeries:
    """One capture run's series: the monitor plus its durable sink.

    Holds the :class:`DurableJSONLWriter` (identical durability
    semantics to the Interaction Trace's writer — shared
    implementation, not a copy) for the process lifetime. The
    ResourceMonitor that drives it is created by
    :func:`start_snapshot_series` and registers in the monitor's
    weakref registry, so the suite's existing ``ResourceMonitor``
    teardown hygiene (``stop_all``, issue #86) covers abandoned
    series.
    """

    def __init__(self, writer: DurableJSONLWriter):
        self._writer = writer
        self._monitor: Optional["ResourceMonitor"] = None

    @property
    def writer(self) -> DurableJSONLWriter:
        return self._writer

    @property
    def path(self) -> Path:
        return self._writer.path

    def close(self) -> None:
        """Close the writer fd; stop the driving monitor if running.

        Best-effort and idempotent; called by the capture-mode exit
        path after the event loop returns.
        """
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception as exc:
                logger.debug(
                    "resource_snapshot_monitor_stop_failed: "
                    "error_class=%s",
                    type(exc).__name__,
                )
            self._monitor = None
        self._writer.close()


# The series THIS process has installed (None in a normal run).
# Process-global like the #104 capture claim and the #105 trace
# writer: one run records one series into one capture directory, from
# process start to the end of the event loop.
_series: Optional[_SnapshotSeries] = None


def install_snapshot_series(capture_dir: Path) -> Path:
    """Install the snapshot series for THIS capture run; return the path.

    Capture runs only — called once from the capture-mode startup path
    (``main`` with a ``--issue-capture`` dir), before the event loop
    starts. Idempotent for the same directory (the same run
    re-asking); a second, different directory is refused (one run, one
    series, one dir) as is a directory already holding a series file
    (another run's).

    Raises:
        SnapshotSeriesError: conflicting install (see above).
    """
    global _series
    if _series is not None:
        if Path(_series.path).parent == Path(capture_dir):
            return _series.path
        _series.close()
        raise SnapshotSeriesError(
            f"snapshot series already installed for "
            f"'{_series.path.parent}' — one run records one series"
        )
    series_path = Path(capture_dir) / SNAPSHOT_FILE_NAME
    try:
        writer = DurableJSONLWriter(series_path, "resource_snapshot")
    except FileExistsError as exc:
        raise SnapshotSeriesError(
            f"snapshot series file already exists: '{series_path}' "
            "— one capture run holds one series; pass a fresh capture "
            "directory"
        ) from exc
    _series = _SnapshotSeries(writer)
    logger.info(
        "Resource Snapshot series: persisting snapshots into %s",
        _series.path,
    )
    return _series.path


def series_installed() -> bool:
    """Is a snapshot series installed in this process?"""
    return _series is not None


def persist_snapshot(snapshot) -> bool:
    """Persist one ResourceSnapshot as a JSONL line; True when written.

    Fail-closed on shape, best-effort on I/O: the snapshot's four
    floats (the ResourceSnapshot building block's shape) plus the
    take-time timestamp are the whole record. A write failure is
    logged at ERROR and swallowed — diagnostics must not crash the run
    being diagnosed.

    Without an installed series (a normal run) this is a silent no-op
    returning False — no snapshot is recorded anywhere.
    """
    if _series is None:
        return False
    payload = {
        "ts": datetime.now().isoformat(),
        "ram_percent": float(snapshot.ram_percent),
        "cpu_percent": float(snapshot.cpu_percent),
        "available_ram_gb": float(snapshot.available_ram_gb),
        "total_ram_gb": float(snapshot.total_ram_gb),
    }
    try:
        return _series._writer.emit(payload)
    except OSError:
        # The shared writer already logs its own I/O failures; this
        # guard covers a writer that cannot even format/queue the
        # record (e.g. an fd closed mid-shutdown).
        logger.exception(
            "resource_snapshot_write_failed: file=%s",
            SNAPSHOT_FILE_NAME,
        )
        return False


def start_snapshot_series() -> bool:
    """Start the ResourceMonitor driving the installed series.

    Capture runs only, after ``install_snapshot_series``: creates the
    ResourceMonitor building block with its DEFAULT poll interval (the
    approved 2-second cadence — asserted equal to
    ``SNAPSHOT_INTERVAL_MS`` by the tests) and its threshold defaults,
    wires ``on_snapshot`` to :func:`persist_snapshot`, and starts it
    (which also takes the immediate first reading — the series' first
    record lands at the first moment of the run).

    Best-effort: a monitor that cannot start (no Qt timer available)
    logs and returns False; the run — the diagnosed one — continues
    with whatever artifacts it can still produce. Threshold warnings
    flow through the monitor's normal logging path into the capture
    DEBUG log.

    Returns:
        True when the monitor is polling.
    """
    if _series is None or _series._monitor is not None:
        return False
    from meetandread.performance.monitor import ResourceMonitor

    def _persist(snapshot) -> None:
        persist_snapshot(snapshot)

    monitor = ResourceMonitor(on_snapshot=_persist)
    monitor.start()
    if not monitor.is_running:
        return False
    _series._monitor = monitor
    return True


def stop_snapshot_series() -> bool:
    """Stop the driving monitor (leaves the writer to teardown close).

    Best-effort and idempotent; called by the capture-mode exit path
    BEFORE the writer close so no poll can race the closing fd.

    Returns:
        True when a monitor was stopped.
    """
    if _series is None or _series._monitor is None:
        return False
    monitor = _series._monitor
    _series._monitor = None
    try:
        monitor.stop()
        return True
    except Exception as exc:
        logger.debug(
            "resource_snapshot_monitor_stop_failed: error_class=%s",
            type(exc).__name__,
        )
        return False


def close_snapshot_series() -> None:
    """Close the installed series (process-exit teardown).

    Best-effort and idempotent: stops the driving monitor, closes the
    writer's fd, and clears the process-global install so every later
    persist is a no-op (``False``). Called by the capture-mode exit
    path AFTER the Qt event filter teardown and BEFORE the completion
    marker — same position in the exit sequence as the #105 trace
    close, keeping every diagnostic emission strictly inside the
    live-application window.
    """
    global _series
    if _series is not None:
        _series.close()
        _series = None
        logger.debug("resource_snapshot_series_closed")


def read_snapshots(capture_dir: Path) -> List[dict]:
    """Read the series as parsed records; [] when absent or unreadable.

    Reads through the #104 torn-record reader: a forced kill can tear
    at most the final JSONL record, and every complete record before
    it — the whole run up to the kill — is preserved. Non-JSON lines
    (a torn tail that still ends in ``\\n``, or foreign content) are
    dropped, not fatal: assembly (#108) owns full validation.
    """
    path = Path(capture_dir) / SNAPSHOT_FILE_NAME
    records: List[dict] = []
    for line in read_appendable_records(path):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records
