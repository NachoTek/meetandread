"""Issue Capture Mode subprocess tests (issue #104, docs/specs/issue-reporting.md).

``windows``-marked (ADR 0001): these tests launch the real application
as a subprocess — the full Qt widget, recording controller, and native
import surface — so they need the Windows interpreter and are skipped
off-Windows. They cover the LAUNCH edge of the one new seam (the
capture-directory contract):

1. Launch with ``--issue-capture <DIR>``: the run is in Issue Capture
   Mode from process start — full DEBUG streaming into the capture
   directory, recording subsystem initialized (AC5), completion marker
   written on a genuine clean exit driven by SIGBREAK (the reporter's
   user-stop signal, #107).
2. Kill mid-run (``taskkill /F``): every completed log record survives
   in the capture directory; at most the single in-flight record is
   torn; no completion marker exists (prompt-flush durability contract).
3. Launch WITHOUT the flag: a normal run — no capture directory is
   created anywhere (AC2), and the normal INFO log lands in the normal
   logs dir.

Subprocess environment is fully sandboxed (APPDATA/USERPROFILE point
into pytest's tmp tree) so the app never reads the real user's
recordings (recovery dialogs), config, or Documents tree, and
concurrent test runs cannot collide on the single-instance mutex.
"""

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
    COMPLETION_MARKER_NAME,
    ISSUE_CAPTURE_FLAG,
    read_appendable_records,
    read_completion_marker,
)

# Real app subprocesses: Windows native stack required (ADR 0001).
pytestmark = pytest.mark.windows

REPO_ROOT = Path(__file__).resolve().parent.parent

# How long to wait for the subprocess to reach a given milestone before
# failing. App startup (Qt + widget + post-processing queue) takes a few
# seconds under the Windows venv.
STARTUP_TIMEOUT_S = 60.0

# How long the clean-exit path may take after SIGBREAK (controller
# shutdown waits up to 10s for the finalizer).
CLEAN_EXIT_TIMEOUT_S = 45.0


def _sandbox_env(tmp_path: Path, lock_name: str) -> dict:
    """Build a subprocess environment whose every user-tree landing spot
    (config via APPDATA, home via USERPROFILE) is inside the test sandbox."""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True, exist_ok=True)
    appdata = tmp_path / "appdata"
    appdata.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "APPDATA": str(appdata),
            "USERPROFILE": str(home),
            # Distinct single-instance mutex per test run: the mutex name
            # is fixed ("meetandread"), so parallel pytest-xdist workers
            # would otherwise see each other as duplicate instances.
            "MAR_TEST_LOCK_NAME": lock_name,
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return env


def _launch_app(
    capture_dir: Path, env: dict, stdout_file: Path
) -> subprocess.Popen:
    """Launch the real application (``meetandread.main.main``) as a
    subprocess, optionally in Issue Capture Mode via the launch flag.

    The child runs a ``python -c`` shim only so each parallel test run
    gets its own single-instance mutex name (the guard hardcodes
    ``"meetandread"`` as its default); the shim patches that default and
    hands straight to ``main()`` with the flag arguments appended, so
    the production startup path (flag parse → logging → Qt app → widget
    → event loop → marker) runs unmodified.

    stdout/stderr go to *stdout_file*: a capture run streams DEBUG
    chatter that would fill an undrained PIPE's OS buffer (64 KB) and
    block the child mid-log-write — the exact opposite of the
    prompt-flush behavior under test.
    """
    cmd = [sys.executable, "-c", _LAUNCH_SHIM]
    if capture_dir is not None:
        cmd += [ISSUE_CAPTURE_FLAG, str(capture_dir)]
    # Keep the stdout file handle alive for the child's lifetime: if it
    # were garbage-collected here the child would inherit a closed
    # handle and die on its first write (Windows). Stored on the Popen
    # object so callers need not manage it.
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
from meetandread.main import main
main()
"""


def _find_capture_log(capture_dir: Path) -> Path:
    logs = sorted(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))
    assert logs, (
        f"no capture log in {capture_dir}: "
        f"{list(capture_dir.iterdir()) if capture_dir.is_dir() else '<dir absent>'}"
    )
    return logs[-1]


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    """Poll until predicate() is truthy; fail with *what* on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    pytest.fail(f"timed out waiting for {what}")


def _kill_hard(proc: subprocess.Popen) -> None:
    """Force-kill the app the way the demo does: taskkill /F (hard kill,
    no Qt shutdown, no Python atexit — the crash-equivalent)."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        capture_output=True,
        check=False,
    )
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pytest.fail("subprocess survived taskkill /F")


def _read_log_records(capture_dir: Path) -> list:
    """Completed records so far; [] until the capture log exists.

    Poll-safe: called in wait loops before the app has created the
    capture directory (startup takes a few seconds)."""
    if not capture_dir.is_dir():
        return []
    if not any(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log")):
        return []
    return read_appendable_records(_find_capture_log(capture_dir))


def _tail(path: Path, limit: int = 2000) -> str:
    """Last *limit* characters of a subprocess stdout file, for failure
    messages."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "<stdout file unavailable>"


class TestLaunchWithFlagCleanExit:
    """AC1, AC3, AC4 (marker on clean exit), AC5 (recording works)."""

    def test_capture_run_streams_debug_and_writes_marker_on_clean_exit(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "run-42"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")

        try:
            # From process start: the capture dir exists and the DEBUG
            # log streams into it (not buffered in memory) — real lines
            # appear while the app is still running.
            _wait_for(
                lambda: capture_dir.is_dir()
                and any(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log")),
                STARTUP_TIMEOUT_S,
                "capture log to appear in the capture directory",
            )
            _wait_for(
                lambda: "Starting meetandread (Issue Capture Mode)"
                in "\n".join(_read_log_records(capture_dir)),
                STARTUP_TIMEOUT_S,
                "capture-mode startup record on disk",
            )
            # AC5: recording works inside capture mode. Per the brief's
            # guidance ("if a full audio recording test is impractical
            # in-lane, a capture-mode subprocess smoke test that reaches
            # the recording subsystem's initialization plus existing
            # recording tests passing unmodified is acceptable; state
            # what you did"), this smoke test drives the real app — the
            # main widget (which owns the RecordingController and its
            # AudioSession) initializes, and the controller's
            # post-processing queue worker starts under capture DEBUG.
            # This machine has no audio input devices, so an actual
            # record-start cannot be driven here; the controller/audio
            # path is covered by the existing (unmodified, passing)
            # recording and audio-session test suites.
            _wait_for(
                lambda: "Main widget initialized" in "\n".join(
                    _read_log_records(capture_dir)
                ),
                STARTUP_TIMEOUT_S,
                "main widget + recording controller initialization (AC5)",
            )
            _wait_for(
                lambda: "PostProcessingQueue worker started"
                in "\n".join(_read_log_records(capture_dir)),
                STARTUP_TIMEOUT_S,
                "recording subsystem post-processing worker start (AC5)",
            )

            # Clean exit: SIGBREAK is the graceful user-stop signal (the
            # same path the Issue Reporter will drive, #107). The app
            # shuts its controller down and the event loop returns 0.
            # Under load the first signal can race the startup sequence
            # (handler not yet installed); _exit_application is
            # idempotent, so send once more before giving up.
            exit_code = None
            for attempt in (1, 2):
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                try:
                    exit_code = proc.wait(timeout=CLEAN_EXIT_TIMEOUT_S)
                    break
                except subprocess.TimeoutExpired:
                    if attempt == 2:
                        _kill_hard(proc)
                        pytest.fail("app did not exit cleanly after SIGBREAK")
                    time.sleep(1.0)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        assert exit_code == 0, (
            "clean exit expected, got "
            f"{exit_code}: {_tail(tmp_path / 'app-stdout.txt')}"
        )

        # Full DEBUG level from process start.
        records = _read_log_records(capture_dir)
        assert any("Log level: DEBUG (capture mode: True)" in r for r in records)

        # The completion marker: clean exit only.
        marker_data = read_completion_marker(capture_dir)
        assert marker_data is not None, "completion marker missing after clean exit"
        assert "finished_at" in marker_data
        assert (capture_dir / COMPLETION_MARKER_NAME).exists()


class TestKillMidRun:
    """AC4 (kill mid-run): prompt-flush durability."""

    def test_forced_kill_preserves_every_completed_record_and_no_marker(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "crash-run"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")

        try:
            # Wait until the app is fully up: recording subsystem running
            # (the bug class being reproduced is often mid-Recording).
            _wait_for(
                lambda: "PostProcessingQueue worker started"
                in "\n".join(_read_log_records(capture_dir)),
                STARTUP_TIMEOUT_S,
                "app fully started before the kill",
            )
            # And let a few more records land (tray wiring, banner, ...)
            # so the kill lands mid-stream, not at file creation.
            time.sleep(2.0)

            # Snapshot the completed records the contract must preserve.
            before_kill = _read_log_records(capture_dir)
            assert len(before_kill) >= 5, "expected a healthy log stream pre-kill"

            # Hard kill — no shutdown, no atexit, no flush hooks.
            _kill_hard(proc)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        # Every completed record before the kill is preserved on disk.
        after_kill = _read_log_records(capture_dir)
        assert after_kill[: len(before_kill)] == before_kill

        # At most the single in-flight record may be torn: the file's
        # tail after the last preserved record is either empty (all
        # records intact) or one unterminated fragment.
        raw = _find_capture_log(capture_dir).read_text(encoding="utf-8")
        torn_tail = raw.split("\n")
        if torn_tail and torn_tail[-1] != "":
            # A torn final record exists: everything before it must be
            # exactly the preserved records — no loss, no reordering.
            assert torn_tail[:-1][-len(before_kill):] == before_kill

        # No completion marker: absence is the crash proof by construction.
        assert not (capture_dir / COMPLETION_MARKER_NAME).exists()
        assert read_completion_marker(capture_dir) is None


class TestNormalRunWithoutFlag:
    """AC2: no flag, no capture directory — behavior exactly a normal run."""

    def test_normal_run_creates_no_capture_directory(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        proc = _launch_app(None, env, tmp_path / "app-stdout.txt")

        try:
            # The normal INFO log lands in the normal logs dir under the
            # sandboxed home; no capture directory is created anywhere.
            logs_dir = (
                tmp_path / "home" / "Documents" / "meetandread" / "logs"
            )
            _wait_for(
                lambda: logs_dir.is_dir()
                and any(logs_dir.glob("meetandread_*.log")),
                STARTUP_TIMEOUT_S,
                "normal-run log in the normal logs dir",
            )

            normal_logs = list(logs_dir.glob("meetandread_*.log"))
            assert all(
                not name.name.startswith(CAPTURE_LOG_PREFIX)
                for name in normal_logs
            ), "capture-prefixed log in a normal run"

            # No capture-prefixed directory or marker anywhere in the
            # sandbox (the only tree this run could touch).
            for marker in tmp_path.rglob(COMPLETION_MARKER_NAME):
                pytest.fail(f"completion marker created in a normal run: {marker}")
            assert not list(tmp_path.rglob(f"{CAPTURE_LOG_PREFIX}*")), (
                "capture log created in a normal run"
            )

            # INFO level: the normal log is readable and not DEBUG-wide.
            content = normal_logs[0].read_text(encoding="utf-8")
            assert "Log level: INFO (capture mode: False)" in content
        finally:
            if proc.poll() is None:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                try:
                    proc.wait(timeout=CLEAN_EXIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    _kill_hard(proc)
            else:
                proc.wait()
