"""Resource Snapshot series subprocess tests (issue #106,
docs/specs/issue-reporting.md).

``windows``-marked (ADR 0001), same patterns as the #104/#105
subprocess tests (tests/test_capture_mode_subprocess.py,
tests/test_interaction_trace_subprocess.py):

1. Production wiring end to end: launch the REAL app with
   ``--issue-capture <DIR>`` through the lightweight bootstrap; the
   series file appears in the capture directory, snapshots land as
   JSONL records with the ResourceSnapshot fields + timestamps, at
   roughly the 2-second cadence (the monitor's immediate-first-poll
   behavior gives the first record from the first moment of the run).
2. Kill mid-run durability: launch the real app, let snapshots land,
   forcibly kill with ``taskkill /F``, and assert every completed
   snapshot record survived on disk (prompt-flush contract) with at
   most one torn final record.
3. Clean exit teardown: SIGBREAK-driven genuine clean exit; the run
   closes the series (monitor stopped, writer closed) and writes the
   completion marker; no snapshot lands after the teardown record.
4. Normal run: no capture flag — no snapshot series file is created
   anywhere in the sandbox.

Subprocess environment is fully sandboxed (APPDATA/USERPROFILE point
into pytest's tmp tree) and each run gets its own single-instance
mutex name, exactly like the sibling capture-mode tests.
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from meetandread.capture_mode import (
    CAPTURE_LOG_PREFIX,
    ISSUE_CAPTURE_FLAG,
    read_appendable_records,
    read_completion_marker,
)
from meetandread.resource_snapshots import (
    SNAPSHOT_FILE_NAME,
    read_snapshots,
)

# Real app subprocesses: Windows native stack required (ADR 0001).
pytestmark = pytest.mark.windows

REPO_ROOT = Path(__file__).resolve().parent.parent

STARTUP_TIMEOUT_S = 60.0
CLEAN_EXIT_TIMEOUT_S = 45.0

# How long to wait for the SECOND cadenced snapshot (first lands
# immediately on monitor start; the next is one poll interval — 2s —
# later, plus startup jitter).
SECOND_SNAPSHOT_TIMEOUT_S = 30.0


def _sandbox_env(tmp_path: Path, lock_name: str) -> dict:
    """Same sandbox shape as the #104/#105 subprocess tests."""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True, exist_ok=True)
    appdata = tmp_path / "appdata"
    appdata.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "APPDATA": str(appdata),
            "USERPROFILE": str(home),
            "MAR_TEST_LOCK_NAME": lock_name,
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return env


# The launch shim replays the REAL production entrypoint (the
# lightweight bootstrap: flag parse -> capture logging -> import
# meetandread.main -> main(), which installs the trace, installs +
# starts the snapshot series, builds the widget, and enters
# app.exec()), with only the single-instance lock name patched — the
# same shim as the #104 tests.
_LAUNCH_SHIM = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig_acquire = _si.acquire_single_instance_lock
def _acquire_with_test_name(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig_acquire(name)
_si.acquire_single_instance_lock = _acquire_with_test_name
import runpy
runpy.run_module("meetandread.__main__", run_name="__main__", alter_sys=True)
"""


def _launch_app(
    capture_dir, env: dict, stdout_file: Path
) -> subprocess.Popen:
    """Launch the real application in Issue Capture Mode (same helper
    contract as the #104 tests: stdout to a file, new process group)."""
    cmd = [sys.executable, "-c", _LAUNCH_SHIM]
    if capture_dir is not None:
        cmd += [ISSUE_CAPTURE_FLAG, str(capture_dir)]
    stdout_fh = open(stdout_file, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    proc._mar_stdout_fh = stdout_fh  # type: ignore[attr-defined]
    return proc


def _kill_hard(proc: subprocess.Popen) -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        capture_output=True,
        check=False,
    )
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pytest.fail("subprocess survived taskkill /F")


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    pytest.fail(f"timed out waiting for {what}")


def _find_capture_log(capture_dir: Path) -> Path:
    logs = sorted(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))
    assert logs, f"no capture log in {capture_dir}"
    return logs[-1]


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "<stdout file unavailable>"


def _records_so_far(capture_dir: Path) -> list:
    """Completed snapshot records so far; [] until the series file
    exists (poll-safe in wait loops)."""
    if not capture_dir.is_dir():
        return []
    path = capture_dir / SNAPSHOT_FILE_NAME
    if not path.exists():
        return []
    return read_snapshots(capture_dir)


class TestProductionSeriesEndToEnd:
    """AC: snapshots persist to the capture dir at the 2s cadence, one
    JSONL line per snapshot, written as taken; full history."""

    def test_capture_run_persists_snapshot_series_at_cadence(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "series-run"
        env = _sandbox_env(tmp_path, f"mar_rs_{uuid.uuid4().hex}")
        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")

        try:
            # The series file exists and the FIRST snapshot (the
            # monitor's immediate poll on start) is already on disk —
            # the series covers the run from its first moment.
            _wait_for(
                lambda: len(_records_so_far(capture_dir)) >= 1,
                STARTUP_TIMEOUT_S,
                "first snapshot record on disk",
            )

            # The cadence: a second record arrives within one more
            # poll interval (2s default) plus jitter. This is the
            # timer-driven wiring — the pure-lane tests cover the
            # interval constant, this proves the monitor is actually
            # polling and persisting over time.
            _wait_for(
                lambda: len(_records_so_far(capture_dir)) >= 2,
                SECOND_SNAPSHOT_TIMEOUT_S,
                "second (cadenced) snapshot record on disk",
            )

            # Clean exit via SIGBREAK (the graceful user-stop signal,
            # same as the #104 tests), with the same idempotent retry.
            exit_code = None
            for attempt in (1, 2):
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                try:
                    exit_code = proc.wait(timeout=CLEAN_EXIT_TIMEOUT_S)
                    break
                except subprocess.TimeoutExpired:
                    if attempt == 2:
                        _kill_hard(proc)
                        pytest.fail(
                            "app did not exit cleanly after SIGBREAK"
                        )
                    time.sleep(1.0)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        assert exit_code == 0, (
            "clean exit expected, got "
            f"{exit_code}: {_tail(tmp_path / 'app-stdout.txt')}"
        )

        records = read_snapshots(capture_dir)
        assert len(records) >= 2, f"expected >=2 snapshots, got {records}"

        # Record shape: the ResourceSnapshot fields + ISO timestamp.
        from datetime import datetime

        for record in records:
            assert set(record) == {
                "ts",
                "ram_percent",
                "cpu_percent",
                "available_ram_gb",
                "total_ram_gb",
            }
            assert 0.0 <= record["ram_percent"] <= 100.0
            assert 0.0 <= record["cpu_percent"] <= 100.0
            assert record["total_ram_gb"] > 0
            assert 0 < record["available_ram_gb"] <= record["total_ram_gb"]
            # A parseable ISO timestamp (the reader already parsed the
            # JSON; prove the ts field parses too).
            datetime.fromisoformat(record["ts"])

        # The cadence contract, read from the record timestamps: the
        # gap between consecutive records is the 2s poll interval
        # (immediate first poll, then one interval) within tolerance.
        # Tolerance is generous (timer jitter under load + psutil
        # scheduling) — it asserts CADENCE SHAPE (roughly-2s period,
        # not a burst), not clock precision.
        from datetime import datetime

        stamps = [
            datetime.fromisoformat(r["ts"]) for r in records[:3]
        ]
        for earlier, later in zip(stamps, stamps[1:]):
            gap = (later - earlier).total_seconds()
            assert 0.5 <= gap <= 15.0, (
                f"snapshot cadence broken: {gap}s between records "
                f"(expected ~2s poll interval): {records[:3]}"
            )

        # The teardown contract: clean exit closed the series before
        # the marker; the capture log holds the series-closed debug
        # record (prompt-flushed), and the marker exists (main
        # finished its full exit path — the series teardown did not
        # break it).
        log_records = read_appendable_records(
            _find_capture_log(capture_dir)
        )
        assert any(
            "resource_snapshot_series_closed" in r for r in log_records
        )
        assert read_completion_marker(capture_dir) is not None

        # Durability shape: every record is one complete JSONL line.
        raw = (capture_dir / SNAPSHOT_FILE_NAME).read_text(
            encoding="utf-8"
        )
        assert raw.endswith("\n")
        assert raw.count("\n") == len(records)


class TestKillMidRunDurability:
    """AC: a forced kill preserves every completed snapshot up to the
    kill; at most one torn final record."""

    def test_forced_kill_preserves_every_completed_snapshot(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "series-crash"
        env = _sandbox_env(tmp_path, f"mar_rs_{uuid.uuid4().hex}")
        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")

        try:
            # Let the series build history: first snapshot immediately,
            # then at least two cadenced records, so the kill lands
            # mid-series with real prior records to preserve.
            _wait_for(
                lambda: len(_records_so_far(capture_dir)) >= 3,
                STARTUP_TIMEOUT_S + 2 * SECOND_SNAPSHOT_TIMEOUT_S,
                "three snapshot records on disk pre-kill",
            )
            # Every completed record is ALREADY durable (each emit is
            # flushed+fsynced on write): snapshot what the contract
            # must preserve across the kill.
            before_kill = _records_so_far(capture_dir)
            _kill_hard(proc)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        # THE durability assertion: every completed snapshot survived
        # the forced kill, byte-identical, append-only.
        after_kill = read_snapshots(capture_dir)
        assert after_kill[: len(before_kill)] == before_kill, (
            f"kill lost completed snapshots: had {len(before_kill)}, "
            f"preserved prefix mismatch"
        )

        # At most one torn record: the raw tail after the last
        # preserved record is empty or exactly one unterminated
        # fragment that is NOT any completed record.
        raw = (capture_dir / SNAPSHOT_FILE_NAME).read_text(
            encoding="utf-8", errors="replace"
        )
        if not raw.endswith("\n"):
            torn_lines = raw.split("\n")
            assert torn_lines[:-1] == [
                json.dumps(e, ensure_ascii=True) for e in before_kill
            ], (
                "raw pre-torn content does not match the preserved "
                "completed snapshots"
            )

        # No completion marker: absence is the crash proof.
        assert read_completion_marker(capture_dir) is None

        # The surviving records are all well-formed (the torn tail,
        # if any, was dropped by the reader).
        for record in after_kill:
            assert set(record) == {
                "ts",
                "ram_percent",
                "cpu_percent",
                "available_ram_gb",
                "total_ram_gb",
            }


class TestNormalRunNoSeries:
    """AC: a normal run (no capture mode) persists no snapshot series."""

    def test_normal_run_creates_no_series_file(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_rs_{uuid.uuid4().hex}")
        proc = _launch_app(None, env, tmp_path / "app-stdout.txt")

        try:
            # Wait until the app is genuinely up (the normal-run log
            # in the sandboxed Documents tree) so the no-op assertion
            # is not vacuous — the app ran, and still wrote no series.
            logs_dir = (
                tmp_path / "home" / "Documents" / "meetandread" / "logs"
            )
            _wait_for(
                lambda: logs_dir.is_dir()
                and any(logs_dir.glob("meetandread_*.log")),
                STARTUP_TIMEOUT_S,
                "normal-run log in the normal logs dir",
            )
            # Give a would-be series time to (not) appear: several
            # poll intervals of the real app running.
            time.sleep(5.0)
        finally:
            if proc.poll() is None:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                try:
                    proc.wait(timeout=CLEAN_EXIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    _kill_hard(proc)
            else:
                proc.wait()

        # No snapshot series file anywhere in the sandbox this run
        # could touch.
        assert not list(tmp_path.rglob(SNAPSHOT_FILE_NAME)), (
            "Resource Snapshot series persisted in a normal run"
        )
