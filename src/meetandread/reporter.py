"""Issue Reporter supervisor core (issue #107, docs/specs/issue-reporting.md,
ADR 0003).

The Issue Reporter is a **standalone supervisor process** — never an
in-app dialog — that owns the user-facing wizard flow (describe →
launch → reproduce → stop; review/submit are #108/#109). It launches
the application in Issue Capture Mode pointed at a FRESH capture
directory, monitors it during reproduction, and — critically — if the
app crashes, the reporter SURVIVES, classifies the crash, writes the
termination record, and continues the flow anyway. The app is never
entrusted with reporting its own death.

## Defense discipline (the module's first constraint)

This module is stdlib-only BY DESIGN: importing it must never pull the
audio stack, the Qt widget tree, or any native backend — the reporter
must run even when the app's subsystems are the thing that is broken.
The structural test (fresh-interpreter import graph) enforces this. It
imports ONLY from sibling stdlib-only modules: ``capture_mode``
(the #104 contract readers) and ``durable_jsonl`` (the shared
durable-write helpers, #160) — and nothing else from the app.

## What this core owns

- **Fresh capture directory per run** (ADR 0005): timestamped
  ``run-YYYYMMDD_HHMMSS`` (with a same-second disambiguator) under a
  ``captures`` base; the reporter never reuses a directory.
- **Termination record** (owner-approved amendment, spec review #113):
  at run end the reporter writes ``termination.json`` into the capture
  directory — how the run ended (``clean_stop`` / ``user_stop`` /
  ``crash``), start/end timestamps, the app's process exit code (or
  Windows exception code — NTSTATUS values like 0xC0000005 arrive as
  the process returncode), and the completion marker's presence or
  absence. It is part of the bundle contract (consumed by #108).
- **Capture-state scan + standalone recovery**: on startup the
  reporter scans its captures base for resumable directories —
  ``incomplete`` (claimed, never finished, no termination record) and
  ``review_ready`` (termination record present, review/submission
  never completed) — and offers to resume them. Recovery must not
  depend on the app being able to launch (ADR 0003, amended).
- **Single-instance interplay** (decided 2026-09-05): before launch,
  if the app is already running, the reporter tells the user to close
  the existing instance and does not proceed until they have. The
  pure seam is ``app_is_running()`` (mutex probe); the wizard layer
  owns the asking.
- **Description capture**: the user's free-text description of the
  problem is written into the capture directory
  (``description.txt``) and carried forward for the later
  review/submission steps (#108/#109).
- **Manual cleanup of capture runs** (issue #112, the 2026-09-05
  retention decision): capture artifacts are excluded from the app's
  automatic retention cleanup ENTIRELY, so the wizard offers the
  deletion instead — ``list_capture_runs`` /
  ``delete_capture_runs`` are the pure listing/deletion seams.

## Launching the app (the supervise seam)

``launch_app()`` spawns the app (``[sys.executable, ``-m``,
``meetandread``, ``--issue-capture``, <dir>]`` — or the frozen exe
path when frozen) exactly as the capture-mode contract requires, with
the capture flag FIRST. ``supervise_run()`` waits, classifies the end
via the pure ``_classify_exit`` (exit code + marker presence), and
writes the termination record — a crashed app is a VALUE
(``RunOutcome.CRASH``), never a propagated exception.

The interactive wizard (the console front-end over this core) lives in
``reporter_wizard.py`` — also stdlib-only — so the core stays
UI-free and fast-lane testable (ADR 0001). The full
supervise-and-crash flow is driven by the ``windows``-marked
subprocess tests (tests/test_reporter_subprocess.py).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Sibling stdlib-only imports ONLY (defense discipline).
from meetandread.capture_mode import (
    CLAIM_FILE_NAME,
    COMPLETION_MARKER_NAME,
    ISSUE_CAPTURE_FLAG,
    read_claim,
    read_completion_marker,
)
from meetandread.durable_jsonl import (
    write_json_durable,
    write_text_durable,
)

# Directory (under the reporter's data base) holding all capture runs.
CAPTURES_DIRNAME = "captures"

# Filename of the reporter-written termination record inside a capture
# directory. Part of the bundle contract (#108 consumes it); treat as
# stable contract.
TERMINATION_FILE_NAME = "termination.json"

# Filename of the user's free-text problem description inside a capture
# directory (#108/#109 carry it into the prefilled issue body).
DESCRIPTION_FILE_NAME = "description.txt"

# Directory name stem for a fresh run: run-YYYYMMDD_HHMMSS[-k].
_RUN_DIR_PREFIX = "run-"
_RUN_DIR_FORMAT = "%Y%m%d_%H%M%S"

# Mutex name matching single_instance.py's default — the reporter probes
# the SAME named mutex the app's guard acquires.
_APP_MUTEX_NAME = "meetandread"

# How the app's single-instance mutex probe validates existence.
_ERROR_ALREADY_EXISTS = 183


class RunOutcome(str, Enum):
    """How a supervised capture run ended (termination-record vocabulary)."""

    CLEAN_STOP = "clean_stop"
    USER_STOP = "user_stop"
    CRASH = "crash"


@dataclass(frozen=True)
class CaptureState:
    """The state of one capture directory, from the reporter's view.

    Derived purely from the directory's contract artifacts (ADR 0005):
    the claim says the directory permanently belongs to exactly one
    run (never liveness); the completion marker alone says that run
    finished cleanly; the termination record says the reporter already
    saw the run end. Liveness is NEVER inferred from the claim.
    """

    path: Path
    claim_present: bool
    marker_present: bool
    termination_present: bool
    termination_outcome: Optional[str] = None

    @property
    def state_name(self) -> str:
        if not self.claim_present and not self.termination_present:
            return "absent"
        if self.termination_present:
            if self.termination_outcome in (
                RunOutcome.CLEAN_STOP.value,
                RunOutcome.USER_STOP.value,
            ) and self.marker_present:
                # The run ended cleanly AND the reporter recorded it —
                # the flow completed; only submission (#109) might
                # still be pending on it, which is #110's lane.
                return "done"
            # The reporter saw the run end (any crash, or a stop whose
            # record predates the marker): review/submission still
            # pending on this directory.
            return "review_ready"
        # No termination record yet — with or without a marker: a
        # marker whose record never arrived (the reporter died between
        # marker and record) is resumable exactly like a bare-claim
        # run, so both read "incomplete".
        return "incomplete"

    @property
    def is_resumable(self) -> bool:
        return self.state_name in ("incomplete", "review_ready")


# ---------------------------------------------------------------------------
# Fresh capture directory per run (ADR 0005)
# ---------------------------------------------------------------------------


def new_capture_dir(
    base: Path, now: Optional[datetime] = None
) -> Path:
    """Create a FRESH capture directory for one run; return its path.

    Shape: ``<base>/captures/run-YYYYMMDD_HHMMSS`` with a ``-2``,
    ``-3``, ... suffix when two runs start in the same wall-clock
    second. The reporter always passes a fresh directory to the app
    (one capture directory holds one run — ADR 0005); this function is
    where that guarantee is manufactured.
    """
    if now is None:
        now = datetime.now()
    captures = Path(base) / CAPTURES_DIRNAME
    captures.mkdir(parents=True, exist_ok=True)
    stem = f"{_RUN_DIR_PREFIX}{now.strftime(_RUN_DIR_FORMAT)}"
    candidate = captures / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = captures / f"{stem}-{suffix}"
    candidate.mkdir()
    return candidate


# ---------------------------------------------------------------------------
# Termination record (owner-approved amendment, spec review #113)
# ---------------------------------------------------------------------------


def write_termination_record(
    capture_dir: Path,
    outcome: RunOutcome,
    exit_code: int,
    started_at: datetime,
    ended_at: datetime,
    marker_present: bool,
) -> Path:
    """Write the termination record; return its path.

    Called exactly once at run end — after the app process is gone and
    the completion marker's presence has been observed. One JSON
    object flushed to disk promptly (write, flush, fsync — the same
    discipline as every capture artifact): the record must survive
    even if the reporter itself dies immediately after writing it.
    """
    payload = {
        "outcome": outcome.value,
        "exit_code": exit_code,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "marker_present": bool(marker_present),
    }
    path = Path(capture_dir) / TERMINATION_FILE_NAME
    write_json_durable(path, payload)
    return path


def read_termination_record(capture_dir: Path) -> Optional[dict]:
    """Read the termination record; None when absent or unreadable.

    Pure read for the recovery scan, the assembler (#108), and tests.
    """
    path = Path(capture_dir) / TERMINATION_FILE_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Description capture
# ---------------------------------------------------------------------------


def write_description(capture_dir: Path, text: str) -> Path:
    """Write the user's free-text problem description; return the path.

    Plain UTF-8 text inside the capture directory; #108/#109 carry it
    into the review screen and the prefilled issue body. Prompt-flush
    discipline so a reporter crash never loses it.
    """
    path = Path(capture_dir) / DESCRIPTION_FILE_NAME
    write_text_durable(path, text if text.endswith("\n") else text + "\n")
    return path


def read_description(capture_dir: Path) -> Optional[str]:
    """Read the description; None when absent (pure read)."""
    path = Path(capture_dir) / DESCRIPTION_FILE_NAME
    try:
        return path.read_text(encoding="utf-8").rstrip("\n")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Capture-state scan + standalone recovery (ADR 0003, amended)
# ---------------------------------------------------------------------------


def scan_capture_state(capture_dir: Path) -> CaptureState:
    """Classify one capture directory from its contract artifacts."""
    d = Path(capture_dir)
    record = read_termination_record(d)
    return CaptureState(
        path=d,
        claim_present=(d / CLAIM_FILE_NAME).is_file(),
        marker_present=(d / COMPLETION_MARKER_NAME).is_file(),
        termination_present=record is not None,
        termination_outcome=(
            record.get("outcome") if record else None
        ),
    )


def find_resumable_captures(base: Path) -> List[Path]:
    """Scan the captures base for resumable directories, newest first.

    A directory is resumable when its run never reached a
    reporter-written termination record (``incomplete``) or reached one
    but the review/submission flow never completed
    (``review_ready``). Non-capture entries (no claim, no termination
    record) are skipped; a missing base is an empty scan, never an
    error — the very first run has nothing to resume.
    """
    captures = Path(base) / CAPTURES_DIRNAME
    if not captures.is_dir():
        return []
    resumable: List[Path] = []
    try:
        entries = sorted(captures.iterdir(), reverse=True)
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        state = scan_capture_state(entry)
        if state.is_resumable:
            resumable.append(entry)
    return resumable


# ---------------------------------------------------------------------------
# Single-instance interplay (decided 2026-09-05)
# ---------------------------------------------------------------------------


def app_is_running(mutex_name: str = _APP_MUTEX_NAME) -> bool:
    """Is an instance of the app already running (mutex probe)?

    Probes the app's single-instance named mutex WITHOUT holding it:
    CreateMutexW reports ERROR_ALREADY_EXISTS exactly when another
    process holds it; the probe immediately closes its handle either
    way. The error is read through ``use_last_error=True`` capture —
    a plain ``GetLastError`` call could observe an intervening
    ctypes-runtime error instead of the mutex result (the classic
    ctypes pitfall). Non-Windows platforms have no guard — False
    (nothing blocks the launch).

    The wizard layer turns True into "please close the running
    instance first" and re-probes until False (the ticket's wait
    AC); this pure seam only answers the question.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    handle = k32.CreateMutexW(
        None, False, f"Global\\{mutex_name}_single_instance"
    )
    already = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
    if handle:
        k32.CloseHandle(handle)
    return already


# ---------------------------------------------------------------------------
# Launching + supervising the app (the supervise seam)
# ---------------------------------------------------------------------------


def _classify_exit(marker_present: bool, exit_code: int) -> RunOutcome:
    """Pure classification of how the supervised app process ended.

    - exit 0 WITH the completion marker → the app shut down cleanly
      (whether that was the user stopping it or the wizard asking for
      a stop is the wizard's knowledge, not the process's — the
      wizard layer refines ``clean_stop`` into ``user_stop``).
    - anything else — nonzero exit, or a zero exit that never wrote
      the marker — → ``crash``. The marker is the absence-proof of a
      crash (#104): no marker means the run did not end cleanly, by
      construction.
    """
    if exit_code == 0 and marker_present:
        return RunOutcome.CLEAN_STOP
    return RunOutcome.CRASH


def default_app_command() -> List[str]:
    """The supervised app command when the caller injects none.

    Frozen reporter (packaging #111): the sibling ``meetandread.exe``
    in the same onedir bundle (``sys._MEIPASS``-adjacent — for a
    onedir build ``sys.executable`` IS the reporter exe, so the app
    exe is its directory sibling). Development: the current
    interpreter running the app's lightweight bootstrap, with no
    ``-m`` under a frozen interpreter by construction.
    """
    if getattr(sys, "frozen", False):
        app_exe = (
            Path(sys.executable).parent / "meetandread.exe"
        )
        return [str(app_exe)]
    return [sys.executable, "-m", "meetandread"]


def launch_app(
    capture_dir: Path,
    app_command: Optional[List[str]] = None,
    extra_env: Optional[dict] = None,
) -> "subprocess.Popen":
    """Launch the app in Issue Capture Mode; return the Popen.

    The command defaults to the current interpreter running the
    lightweight bootstrap (``python -m meetandread``) — the same
    production entrypoint the frozen exe and console script use —
    with the capture flag FIRST. When the REPORTER itself is frozen
    (packaging #111), the default becomes the sibling ``meetandread.exe``
    in the same bundle — a frozen interpreter has no ``-m``; the
    packaged reporter must supervise the packaged app. ``app_command``
    overrides the program (tests inject a stub). The child is detached
    from the reporter's console process group so the wizard's Ctrl+C
    (user stop) does not kill the app out from under the supervised
    stop flow.

    The reporter does NOT suppress the app's stderr: startup failures
    must be visible to the user, and the capture log is the durable
    record either way.
    """
    if app_command is None:
        app_command = default_app_command()
    cmd = list(app_command) + [ISSUE_CAPTURE_FLAG, str(capture_dir)]
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    kwargs = {}
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP (the #104 pattern): the app gets
        # its own console group so the reporter can address CTRL_BREAK
        # to it specifically. Deliberately NOT DETACHED_PROCESS — the
        # user-stop signal must be deliverable.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd,
        env=env,
        **kwargs,
    )
    return proc


@dataclass(frozen=True)
class SupervisedRun:
    """The outcome of one supervised capture run (a value, never an
    exception — the reporter survives the app's death)."""

    outcome: RunOutcome
    exit_code: int
    marker_present: bool
    termination_record: Path
    capture_dir: Path
    started_at: datetime
    ended_at: datetime
    #: True when the run's stop was initiated from the reporter (the
    #: wizard's user-stop signal) rather than the app exiting on its
    #: own — the fact that refines clean_stop into user_stop.
    user_initiated_stop: bool = False

    def final_outcome(self) -> RunOutcome:
        """The termination-record outcome for this run.

        ``user_stop`` when the reporter initiated a graceful stop that
        the app answered with a clean exit; ``_classify_exit``'s
        verdict otherwise (an app that exited cleanly on its own is a
        ``clean_stop``, not a user stop).
        """
        if (
            self.user_initiated_stop
            and self.outcome == RunOutcome.CLEAN_STOP
        ):
            return RunOutcome.USER_STOP
        return self.outcome


def supervise_run(
    proc: "subprocess.Popen",
    capture_dir: Path,
    started_at: Optional[datetime] = None,
    user_initiated_stop: bool = False,
) -> SupervisedRun:
    """Wait for the supervised app process to end; record the end.

    The reporter's core loop: wait for the app (forever — reproduction
    takes as long as it takes), classify how it ended from the pure
    process facts (exit code + completion marker), and write the
    termination record into the capture directory. A crashed app is
    classified and recorded — NEVER a propagated exception; the
    wizard continues to its stop/review steps either way (the crash
    itself is always reportable).

    The graceful stop is the WIZARD's step, performed before this
    call (``_graceful_stop``): by the time supervision begins, the
    stop signal has already been sent exactly once — supervising is
    purely waiting, classifying, recording. ``user_initiated_stop``
    records whether THAT stop came from the reporter (the wizard's
    user-stop path), refining a clean exit into ``user_stop`` in the
    record and :attr:`final_outcome`.
    """
    if started_at is None:
        started_at = datetime.now()
    exit_code = proc.wait()
    ended_at = datetime.now()
    marker_present = read_completion_marker(capture_dir) is not None
    outcome = _classify_exit(marker_present, exit_code)
    record = write_termination_record(
        capture_dir,
        outcome=outcome,
        exit_code=exit_code,
        started_at=started_at,
        ended_at=ended_at,
        marker_present=marker_present,
    )
    run = SupervisedRun(
        outcome=outcome,
        exit_code=exit_code,
        marker_present=marker_present,
        termination_record=record,
        capture_dir=Path(capture_dir),
        started_at=started_at,
        ended_at=ended_at,
        user_initiated_stop=user_initiated_stop,
    )
    # The record must reflect the FINAL outcome vocabulary (user_stop
    # when the reporter initiated a clean stop): rewrite it once with
    # the refined verdict so the durable artifact and the returned
    # value agree (#108 consumes the record, not the object).
    if run.final_outcome() != run.outcome:
        write_termination_record(
            capture_dir,
            outcome=run.final_outcome(),
            exit_code=exit_code,
            started_at=started_at,
            ended_at=ended_at,
            marker_present=marker_present,
        )
    return run


def resume_capture(capture_dir: Path, data_base: Path) -> dict:
    """Resume an interrupted capture directory: fill in the termination
    record if the run's end was never recorded.

    The standalone-recovery path for ``incomplete`` directories: the
    run is over (its process is long gone), so the reporter
    reconstructs what it can from the durable artifacts — the claim's
    ``started_at``, the completion marker's presence — and writes the
    missing termination record with the crash vocabulary (a run whose
    end the reporter never witnessed did not end cleanly by
    construction). ``ended_at`` is recorded as unknown (``null``) —
    the resume time is NOT the run's end, and inventing one would
    corrupt the record; same for the exit code. A staged description
    (``pending-<run>.description`` written during the run, see the
    wizard) is reconciled into the capture directory here so a
    reporter crash can never orphan it. Returns the record payload.
    ``review_ready`` directories already hold their record and are
    returned unchanged.
    """
    d = Path(capture_dir)
    existing = read_termination_record(d)
    if existing is not None:
        return existing
    claim = read_claim(d)
    try:
        started_at = datetime.fromisoformat(claim["started_at"])  # type: ignore[index]
    except (TypeError, ValueError, KeyError):
        started_at = datetime.now()
    marker_present = read_completion_marker(d) is not None
    outcome = (
        RunOutcome.CLEAN_STOP if marker_present else RunOutcome.CRASH
    )
    payload = {
        "outcome": outcome.value,
        "exit_code": None,  # unwitnessed end: no code to report
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": None,  # unwitnessed end: never fabricate a time
        "marker_present": marker_present,
    }
    write_json_durable(d / TERMINATION_FILE_NAME, payload)

    # Reconcile a staged description (reporter died before the run-end
    # copy): the capture directory is the one place the later
    # review/submission steps (#108/#109) look.
    staged = Path(data_base) / f"pending-{d.name}.description"
    if staged.is_file() and not (d / DESCRIPTION_FILE_NAME).exists():
        try:
            write_description(d, staged.read_text(encoding="utf-8"))
            staged.unlink()
        except OSError:
            pass
    return payload


# ---------------------------------------------------------------------------
# Manual cleanup of capture runs (issue #112, the 2026-09-05 retention
# decision's other half: capture artifacts are excluded from automatic
# cleanup ENTIRELY, so the wizard offers the deletion instead)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureRunListing:
    """One capture run offered by the wizard's manual cleanup step.

    Pure value over the capture-state scan: the run's directory, its
    state name, whether a Diagnostics Bundle sits inside it, and the
    run's total on-disk size in bytes (rounded up to whole filesystem
    blocks would be overkill — the raw byte sum of ``stat().st_size``
    over the directory tree is plenty for a "how much space" display).
    """

    path: Path
    state_name: str
    has_bundle: bool
    size_bytes: int


def _dir_tree_size(path: Path) -> int:
    """Sum file sizes under *path* (0 on any error — display-only)."""
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def list_capture_runs(base: Path) -> List[CaptureRunListing]:
    """List every capture run under the data base, newest first.

    A directory qualifies when it is a CAPTURE directory — recognized
    by the contract artifacts the state scan already trusts (a claim
    or a termination record; ``scan_capture_state``'s ``absent``
    verdict skips foreign directories). EVERY state qualifies
    (incomplete, review_ready, done): the cleanup listing is the
    user's file manager for capture artifacts, not a recovery flow.
    A missing base is an empty list, never an error.
    """
    captures = Path(base) / CAPTURES_DIRNAME
    if not captures.is_dir():
        return []
    try:
        entries = sorted(captures.iterdir(), reverse=True)
    except OSError:
        return []
    listings: List[CaptureRunListing] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        state = scan_capture_state(entry)
        if state.state_name == "absent":
            continue
        listings.append(
            CaptureRunListing(
                path=entry,
                state_name=state.state_name,
                has_bundle=(entry / "diagnostics_bundle.txt").is_file(),
                size_bytes=_dir_tree_size(entry),
            )
        )
    return listings


def delete_capture_runs(runs: List[Path], base: Path) -> List[Path]:
    """Delete whole capture runs; return the directories removed.

    The wizard's manual-cleanup delete seam (issue #112). Deleting a
    run removes its capture directory — the Diagnostics Bundle lives
    INSIDE it, so directory and bundle go together by construction.

    Safety contract (the deletion boundary is narrow BY DESIGN):
    - Only directories inside ``<base>/captures`` are ever deleted —
      the path is resolved and checked against the captures root, so
      no absolute-path or traversal trickery can point the deletion
      at the user's Recordings, Audio, or Transcripts (which live
      under Documents/meetandread, a different tree entirely).
    - Only CAPTURE directories are deleted: recognized by the same
      contract artifacts as the listing (a claim or termination
      record); an empty or foreign directory inside captures/ is
      never touched.
    - Per-run errors (locked file, vanished mid-scan) are skipped,
      never raised — cleanup must not die on the first locked handle.
    """
    captures_root = (Path(base) / CAPTURES_DIRNAME).resolve()
    deleted: List[Path] = []
    for run in runs:
        d = Path(run)
        try:
            resolved = d.resolve()
            if not resolved.is_relative_to(captures_root):
                continue
            if resolved == captures_root:
                continue
        except OSError:
            continue
        if scan_capture_state(d).state_name == "absent":
            continue
        try:
            shutil.rmtree(d)
            deleted.append(d)
        except OSError:
            continue
    return deleted


# Re-exported for the wizard/tests: probing wait helper.
def wait_until(predicate, timeout_s: float, interval_s: float = 0.25):
    """Poll predicate until true or timeout; returns True/False."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False
