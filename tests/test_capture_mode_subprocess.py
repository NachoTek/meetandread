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
    CLAIM_FILE_NAME,
    COMPLETION_MARKER_NAME,
    ISSUE_CAPTURE_FLAG,
    read_appendable_records,
    read_completion_marker,
)
from meetandread.logging_setup import NORMAL_LOG_PREFIX

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
import runpy
# Replay the REAL production entrypoint (src/meetandread/__main__.py —
# the lightweight bootstrap: flag parse -> capture logging -> import
# meetandread.main -> main()), not a hand-rolled shortcut, so the tests
# cover the exact startup path users and the frozen exe run.
runpy.run_module("meetandread.__main__", run_name="__main__", alter_sys=True)
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
                lambda: "main_widget_init:" in "\n".join(
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


class TestDeterministicHardExit:
    """Issue #124: the post-event-loop exit is a deterministic hard exit.

    PR #123's filter-removal ordering narrowed but cannot eliminate the
    0xC0000005 interpreter/Qt teardown race, so main() now ends in
    os._exit after the marker write. This test proves the deterministic
    path is TAKEN on a clean exit (via the pre-exit debug record); that
    the crash itself is gone is AC3 stress evidence, not automatable.
    """

    def test_clean_exit_uses_hard_exit_path(self, tmp_path):
        capture_dir = tmp_path / "capture" / "hard-exit-run"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")

        try:
            # Wait until the app is fully up (same milestone as the
            # clean-exit test: the recording subsystem is running).
            _wait_for(
                lambda: "PostProcessingQueue worker started"
                in "\n".join(_read_log_records(capture_dir)),
                STARTUP_TIMEOUT_S,
                "app fully started before the clean exit",
            )

            # Clean exit via SIGBREAK — the graceful user-stop signal,
            # with the same idempotent retry as TestLaunchWithFlagCleanExit.
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

        # The completion marker: written before the hard exit.
        marker_data = read_completion_marker(capture_dir)
        assert marker_data is not None, (
            "completion marker missing after clean hard exit: "
            f"{_tail(tmp_path / 'app-stdout.txt')}"
        )
        assert (capture_dir / COMPLETION_MARKER_NAME).exists()

        # THE assertion: main() logged the hard-exit record immediately
        # before os._exit. The capture log is prompt-flushed per record,
        # so the record survives even though the process never runs
        # interpreter finalization.
        records = _read_log_records(capture_dir)
        assert "hard_exit: code=0" in "\n".join(records), (
            "hard_exit debug record missing from capture log — "
            "the deterministic os._exit path was not taken: "
            f"{_tail(tmp_path / 'app-stdout.txt')}"
        )


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


class TestNormalRunExitPath:
    """Issue #124, PR #125 fix round 2: the deterministic hard exit is
    GATED on capture mode. A normal (no-flag) run keeps its pre-#124
    exit — sys.exit with normal interpreter finalization — because the
    hard exit's safety arguments (prompt-flushed capture log, marker
    written first) only exist on the capture path (#104). This test
    proves the two paths are distinct by asserting the hard-exit debug
    record is ABSENT from a normal run's log (its mirror,
    TestDeterministicHardExit, asserts the record is PRESENT in a
    capture run); the record is emitted immediately before os._exit, so
    its absence proves the hard-exit branch was not taken on a clean
    no-flag exit."""

    def test_no_flag_clean_exit_takes_normal_sys_exit_path(
        self, tmp_path
    ):
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        # Force meetandread.main's logger to DEBUG in the child (via a
        # sitecustomize pre-hook, earliest on PYTHONPATH): the hard_exit
        # record is DEBUG and a normal run logs at INFO, so without this
        # the absence assertion below would pass vacuously even if the
        # ungated branch fired.
        debug_dir = tmp_path / "debug_main_logger"
        debug_dir.mkdir()
        (debug_dir / "sitecustomize.py").write_text(
            _DEBUG_MAIN_LOGGER_SITECUSTOMIZE, encoding="utf-8"
        )
        env["PYTHONPATH"] = f"{debug_dir}{os.pathsep}{env['PYTHONPATH']}"

        proc = _launch_app(None, env, tmp_path / "app-stdout.txt")

        try:
            # The normal run's log lands under the sandboxed home
            # (INFO level; the hard-exit record is DEBUG, so this test
            # must not rely on level filtering alone — it forces DEBUG
            # for meetandread.main in the child below so a record
            # logged by a buggy ungated hard exit WOULD land in the
            # file. Absence then proves the branch was not taken).
            logs_dir = (
                tmp_path / "home" / "Documents" / "meetandread" / "logs"
            )
            _wait_for(
                lambda: logs_dir.is_dir()
                and any(logs_dir.glob(f"{NORMAL_LOG_PREFIX}*.log")),
                STARTUP_TIMEOUT_S,
                "normal-run log in the normal logs dir",
            )

            # App fully up (same milestone the capture tests use).
            def _normal_records():
                logs = list(logs_dir.glob(f"{NORMAL_LOG_PREFIX}*.log"))
                if not logs:
                    return []
                return read_appendable_records(logs[0])

            _wait_for(
                lambda: "PostProcessingQueue worker started"
                in "\n".join(_normal_records()),
                STARTUP_TIMEOUT_S,
                "app fully started before the clean exit",
            )

            # Clean exit via SIGBREAK, with the same idempotent retry
            # as the capture-mode tests.
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

        # THE assertion: the hard-exit debug record is ABSENT from the
        # normal run's log — the sys.exit branch was taken, not the
        # os._exit branch (the mirrored capture test proves presence).
        joined = "\n".join(_normal_records())
        assert "hard_exit: code=" not in joined, (
            "hard_exit debug record present in a NO-FLAG run's log — "
            "the capture-gated hard exit fired on a normal run "
            f"(path separation broken): {_tail(tmp_path / 'app-stdout.txt')}"
        )


class TestStartupFailureIsCaptured:
    """Fix round 1, finding 1: the bootstrap configures capture logging
    BEFORE any app subsystem is imported, so an import-time startup
    failure still produces a capture log holding the failure — the
    startup-crash reporting case the review flagged as uncovered."""

    def test_import_time_failure_lands_in_capture_log(self, tmp_path):
        capture_dir = tmp_path / "capture" / "import-crash"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        # Poison the import of a module meetandread.main needs at import
        # time (PyQt6.QtWidgets arrives before anything else heavy): a
        # fake PyQt6 package raising at import is placed EARLIEST on
        # sys.path via PYTHONPATH, ahead of the real site-packages.
        poison_dir = tmp_path / "poisoned_pyqt"
        (poison_dir / "PyQt6").mkdir(parents=True)
        (poison_dir / "PyQt6" / "__init__.py").write_text(
            "raise ImportError('poisoned PyQt6 import for capture-mode "
            "startup-failure test')\n",
            encoding="utf-8",
        )
        # PYTHONPATH order: poison first, then src (the sandbox helper
        # set src; prepend the poison dir).
        env["PYTHONPATH"] = f"{poison_dir}{os.pathsep}{env['PYTHONPATH']}"

        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")
        # The child dies on its own (import failure); reap it.
        try:
            exit_code = proc.wait(timeout=STARTUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_hard(proc)
            pytest.fail(
                "app did not exit after import-time startup failure: "
                f"{_tail(tmp_path / 'app-stdout.txt')}"
            )

        # The process failed (nonzero exit — the poisoned import).
        assert exit_code != 0

        # THE assertion: the capture dir holds a DEBUG log (created
        # exclusively by the bootstrap before any app import) that
        # records the startup failure.
        records = _read_log_records(capture_dir)
        assert records, (
            "capture log missing after import-time startup failure: "
            f"{_tail(tmp_path / 'app-stdout.txt')}"
        )
        joined = "\n".join(records)
        # The bootstrap logged the failure with traceback through the
        # prompt-flush handler BEFORE re-raising.
        assert "Startup failure: meetandread.main could not be imported" in joined
        assert "poisoned PyQt6 import" in joined
        # Full DEBUG from process start even for this doomed run.
        assert any("Log level: DEBUG (capture mode: True)" in r for r in records)
        # No clean-exit marker for a crashed startup, by construction.
        assert read_completion_marker(capture_dir) is None

    def test_console_script_entry_path_captures_import_failure(
        self, tmp_path
    ):
        """Fix round 2, finding 1a: the pip-installed console script
        points at the bootstrap's public run() (pyproject
        [project.scripts] -> meetandread.__main__:run). Driving THE SAME
        function the console script invokes — importing the module must
        NOT start the app (guarded module-level execution) — proves the
        installed command gets capture-before-heavy-imports too."""
        capture_dir = tmp_path / "capture" / "console-script-crash"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        # Same PyQt6 poison as the python -m test above.
        poison_dir = tmp_path / "poisoned_pyqt"
        (poison_dir / "PyQt6").mkdir(parents=True)
        (poison_dir / "PyQt6" / "__init__.py").write_text(
            "raise ImportError('poisoned PyQt6 import for console-script "
            "capture test')\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = f"{poison_dir}{os.pathsep}{env['PYTHONPATH']}"

        # EXACTLY what the installed console script runs: import the
        # bootstrap module (as the entry-point loader does), call run()
        # with the flag in sys.argv.
        shim = (
            "import os, sys\n"
            'sys.path.insert(0, os.environ["PYTHONPATH"])\n'
            "from meetandread.__main__ import run\n"
            "run()\n"
        )
        stdout_file = tmp_path / "app-stdout.txt"
        with open(stdout_file, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    shim,
                    ISSUE_CAPTURE_FLAG,
                    str(capture_dir),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        try:
            exit_code = proc.wait(timeout=STARTUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_hard(proc)
            pytest.fail(
                "console-script-path app did not exit: "
                f"{_tail(stdout_file)}"
            )

        assert exit_code != 0
        records = _read_log_records(capture_dir)
        assert records, (
            "capture log missing after console-script-path import "
            f"failure: {_tail(stdout_file)}"
        )
        joined = "\n".join(records)
        assert "Startup failure:" in joined
        assert "poisoned PyQt6 import" in joined
        assert any("Log level: DEBUG (capture mode: True)" in r for r in records)
        assert read_completion_marker(capture_dir) is None

    def test_main_startup_exception_lands_in_capture_log(self, tmp_path):
        """Fix round 2, finding 1b: an exception raised INSIDE main()
        (after the bootstrap configured capture logging and handed off)
        is recorded into the capture log before the nonzero exit — the
        guard now covers the main() call, not just its import.

        Mechanism: a sitecustomize pre-hook (loaded by the interpreter
        from PYTHONPATH before any app code) patches
        QApplication.__init__ to raise once main() constructs the Qt
        application — an unguarded main()-body statement (main.py:
        ``app = QApplication(sys.argv)``), deep after the handoff."""
        capture_dir = tmp_path / "capture" / "main-crash"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        poison_dir = tmp_path / "qapp_raiser"
        poison_dir.mkdir()
        (poison_dir / "sitecustomize.py").write_text(
            "from PyQt6.QtWidgets import QApplication\n"
            "def _raising_init(self, *a, **k):\n"
            "    raise RuntimeError(\n"
            "        'injected QApplication failure for '\n"
            "        'main-startup-exception test'\n"
            "    )\n"
            "QApplication.__init__ = _raising_init\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = f"{poison_dir}{os.pathsep}{env['PYTHONPATH']}"

        proc = _launch_app(capture_dir, env, tmp_path / "app-stdout.txt")
        try:
            exit_code = proc.wait(timeout=STARTUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_hard(proc)
            pytest.fail(
                "app did not exit after main() startup exception: "
                f"{_tail(tmp_path / 'app-stdout.txt')}"
            )

        # Nonzero exit (the injected failure propagates).
        assert exit_code != 0, (
            "expected nonzero exit after injected QApplication failure: "
            f"{_tail(tmp_path / 'app-stdout.txt')}"
        )

        # THE assertion: the capture log holds the main()-phase startup
        # failure with traceback — the guard covered the main() call.
        records = _read_log_records(capture_dir)
        joined = "\n".join(records)
        assert "Startup failure: meetandread failed during startup" in joined
        assert "injected QApplication failure" in joined
        # main() got far enough to prove the handoff happened (the
        # failure is INSIDE main, not at its import: the startup banner
        # is the first main()-body record after configure).
        assert "Starting meetandread (Issue Capture Mode)" in joined
        # Crashed run: no marker, by construction.
        assert read_completion_marker(capture_dir) is None


class TestReusedCaptureDirectoryRejected:
    """Fix round 1, finding 2: one capture directory == one run.

    Negative control (a): a REUSED directory holding a stale completion
    marker from a previous run is rejected at entry — before any
    logging starts — so a killed reused run can never be read as a
    clean exit through the stale marker. (The pure fast-lane twins —
    parse-time rejection, empty-dir acceptance, in-process same-second
    collision — live in tests/test_logging_foundation.py per ADR 0001;
    this file carries the end-to-end subprocess controls.)
    """

    def test_reused_dir_launch_exits_2_without_logging(self, tmp_path):
        """End-to-end negative control: launching the real app against
        a non-empty stale directory exits 2 (usage-error class) and
        starts NO capture stream — the old artifacts survive intact."""
        reused = tmp_path / "capture" / "stale-run"
        reused.mkdir(parents=True)
        old_marker = reused / COMPLETION_MARKER_NAME
        old_marker.write_text(
            '{"finished_at": "2026-09-06T10:00:00"}', encoding="utf-8"
        )
        old_log = reused / f"{CAPTURE_LOG_PREFIX}20260906_100000.log"
        old_log.write_text("old run line\n", encoding="utf-8")

        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        proc = _launch_app(reused, env, tmp_path / "app-stdout.txt")
        try:
            exit_code = proc.wait(timeout=STARTUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_hard(proc)
            pytest.fail(
                "app did not exit after reused-dir rejection: "
                f"{_tail(tmp_path / 'app-stdout.txt')}"
            )

        assert exit_code == 2, (
            f"reused non-empty capture dir must exit 2, got {exit_code}: "
            f"{_tail(tmp_path / 'app-stdout.txt')}"
        )
        # Nothing new was logged into the reused dir; the old run's
        # artifacts are byte-identical.
        assert old_log.read_text(encoding="utf-8") == "old run line\n"
        assert read_completion_marker(reused) == {
            "finished_at": "2026-09-06T10:00:00"
        }
        assert sorted(p.name for p in reused.iterdir()) == sorted(
            [old_log.name, old_marker.name]
        )


class TestSameSecondCollision:
    """Fix round 1, finding 2 (collision half): two starts resolving to
    the same second-resolution log filename — the second start must
    fail loudly (exclusive creation), and the first run's log must
    survive intact. (The in-process pure twin lives in
    tests/test_logging_foundation.py; this is the end-to-end control.)"""

    def test_two_subprocesses_same_second_second_exits_2(self, tmp_path):
        """End-to-end: two real app launches into the SAME (empty at
        first launch) directory, forced to the same filename second via
        a frozen clock shim, assert the loser exits 2 and the winner's
        log is intact."""
        capture_dir = tmp_path / "capture" / "same-second"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")
        # Freeze both children's capture filename clock to the same
        # second via the shim below.
        stdout_a = open(tmp_path / "a-stdout.txt", "w", encoding="utf-8")
        stdout_b = open(tmp_path / "b-stdout.txt", "w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-c",
            _FROZEN_CLOCK_SHIM.replace("_MAR_TS_", "12, 0, 0"),
            ISSUE_CAPTURE_FLAG,
            str(capture_dir),
        ]
        procs = []
        try:
            for fh in (stdout_a, stdout_b):
                procs.append(
                    subprocess.Popen(
                        cmd,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                )
            # Both children race; exactly one wins the exclusive log
            # creation, the other exits 2 quickly (rejected at entry).
            # The WINNER runs the full app (no clean exit required —
            # the collision is decided at log-file creation), so: wait
            # for the loser to exit, then hard-kill the winner.
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            exit_codes = {}
            while len(exit_codes) < 2 and time.monotonic() < deadline:
                for idx, proc in enumerate(procs):
                    if idx in exit_codes:
                        continue
                    if proc.poll() is not None:
                        exit_codes[idx] = proc.returncode
                if exit_codes:
                    break
                time.sleep(0.25)
            assert exit_codes, (
                "neither collision child exited in time: "
                f"a={_tail(tmp_path / 'a-stdout.txt')} "
                f"b={_tail(tmp_path / 'b-stdout.txt')}"
            )
        finally:
            for proc in procs:
                if proc.poll() is None:
                    _kill_hard(proc)
            for fh in (stdout_a, stdout_b):
                fh.close()

        # The loser exits 2 with the clear stderr message.
        codes = list(exit_codes.values())
        assert codes == [2], (
            f"expected exactly the collision loser (exit 2) to have "
            f"exited, got {codes}: "
            f"a={_tail(tmp_path / 'a-stdout.txt')} b={_tail(tmp_path / 'b-stdout.txt')}"
        )
        # Exactly one capture log exists, and it is intact (has records
        # from exactly one run — no interleaved truncation).
        log = _find_capture_log(capture_dir)
        records = read_appendable_records(log)
        assert "Log level: DEBUG (capture mode: True)" in "\n".join(records)
        # Exactly one log file — the loser never created/truncated one.
        assert len(list(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))) == 1


# Frozen-clock launch shim: identical to the production bootstrap path
# except the filename clocks (datetime.now) are pinned in BOTH
# capture_mode and logging_setup to _MAR_TS_ (h:m:s substituted per
# child by the test), controlling the second-resolution log filename.
_FROZEN_CLOCK_SHIM = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig_acquire = _si.acquire_single_instance_lock
def _acquire_with_test_name(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig_acquire(name)
_si.acquire_single_instance_lock = _acquire_with_test_name
import datetime as _dt
class _FrozenDatetime:
    @staticmethod
    def now(tz=None):
        return _dt.datetime(2026, 9, 7, _MAR_TS_)
import meetandread.capture_mode as _cm
import meetandread.logging_setup as _ls
_cm.datetime = _FrozenDatetime
_ls.datetime = _FrozenDatetime
import runpy
runpy.run_module("meetandread.__main__", run_name="__main__", alter_sys=True)
"""


class TestConcurrentDifferentTimestampStarts:
    """Fix round 2, finding 2: the capture-directory CLAIM is atomic.

    The reviewer's reproduced race: two processes pass the (cheap)
    emptiness check and start in DIFFERENT seconds — distinct
    timestamped log names, so the round-1 exclusive log creation never
    fires — and interleave their records in one directory. The
    O_CREAT|O_EXCL claim (capture_run.claim) is the authoritative
    gate: exactly one process wins; the loser exits 2 with the
    rejection message and writes NOTHING into the directory."""

    def test_two_different_timestamp_starts_one_wins_one_exits_2(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "concurrent"
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        # Distinct frozen clocks: child A at 12:00:00, child B at
        # 12:00:05 — different log filenames, defeating the round-1
        # same-second guard so the CLAIM is the only gate.
        # A start-line barrier (in the shim, OUTSIDE the claim code)
        # holds both children AFTER their emptiness check passed and
        # BEFORE either runs the real claim — the deterministic form
        # of the reviewer's near-simultaneous race.
        stdout_a = open(tmp_path / "a-stdout.txt", "w", encoding="utf-8")
        stdout_b = open(tmp_path / "b-stdout.txt", "w", encoding="utf-8")
        barrier = tmp_path / "start-line"
        barrier.mkdir()
        procs = []
        try:
            for fh, ts, who in (
                (stdout_a, "12, 0, 0", "a"),
                (stdout_b, "12, 0, 5", "b"),
            ):
                shim = _BARRIER_CLOCK_SHIM.replace("_MAR_TS_", ts).replace(
                    "_MAR_WHO_", who
                ).replace("_MAR_BARRIER_", barrier.as_posix())
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable, "-c", shim,
                            ISSUE_CAPTURE_FLAG, str(capture_dir),
                        ],
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                )
            # Exactly one child loses the claim race and exits 2
            # quickly; the winner runs the full app (killed in
            # finally — the race is decided at claim time).
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            loser_codes = []
            while time.monotonic() < deadline:
                for proc in procs:
                    if proc.poll() is not None and proc.returncode == 2:
                        loser_codes.append(proc.returncode)
                if loser_codes:
                    break
                time.sleep(0.25)
            assert loser_codes, (
                "no collision loser exited 2 — claim race not decided: "
                f"a={_tail(tmp_path / 'a-stdout.txt')} "
                f"b={_tail(tmp_path / 'b-stdout.txt')}"
            )
        finally:
            for proc in procs:
                if proc.poll() is None:
                    _kill_hard(proc)
            for fh in (stdout_a, stdout_b):
                fh.close()

        # Exactly one claim and one log; the loser wrote NOTHING.
        claims = list(capture_dir.glob(CLAIM_FILE_NAME))
        assert len(claims) == 1
        logs = list(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))
        assert len(logs) == 1
        # The winner's records are unmixed: the log carries exactly one
        # run-header sequence (one capture-mode banner).
        joined = "\n".join(read_appendable_records(logs[0]))
        assert joined.count("Issue Capture Mode: streaming diagnostics into") == 1
        assert "Log level: DEBUG (capture mode: True)" in joined
        # The rejection message named the CLAIM (the loser passed the
        # emptiness check — both parsed the empty dir before the
        # barrier — so only the exclusive claim could refuse it).
        loser_tail = _tail(tmp_path / "a-stdout.txt") + _tail(
            tmp_path / "b-stdout.txt"
        )
        assert "already claimed by another run" in loser_tail


# Barrier + frozen-clock launch shim: like _FROZEN_CLOCK_SHIM, but the
# REAL claim_capture_dir is wrapped with a start-line barrier — each
# child signals arrival and waits for its peer AFTER the bootstrap
# parsed the (still-empty) dir and BEFORE the claim executes — making
# the simultaneous emptiness-pass race deterministic. The claim itself
# is the production code, called unmodified inside the wrapper.
_BARRIER_CLOCK_SHIM = """
import os, sys, time
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig_acquire = _si.acquire_single_instance_lock
def _acquire_with_test_name(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig_acquire(name)
_si.acquire_single_instance_lock = _acquire_with_test_name
import datetime as _dt
class _FrozenDatetime:
    @staticmethod
    def now(tz=None):
        return _dt.datetime(2026, 9, 7, _MAR_TS_)
import meetandread.capture_mode as _cm
import meetandread.logging_setup as _ls
_cm.datetime = _FrozenDatetime
_ls.datetime = _FrozenDatetime
# Start-line barrier AROUND the real claim: both children have already
# parsed the empty dir when they reach the claim; hold each until the
# peer arrives, then run the PRODUCTION claim unmodified.
_real_claim = _cm.claim_capture_dir
def _barrier_claim(capture_dir, now=None):
    _me = os.path.join("_MAR_BARRIER_", "arrived_" + "_MAR_WHO_")
    _peer = os.path.join(
        "_MAR_BARRIER_", "arrived_" + ("b" if "_MAR_WHO_" == "a" else "a")
    )
    open(_me, "w").close()
    _deadline = time.monotonic() + 30.0
    while not os.path.exists(_peer) and time.monotonic() < _deadline:
        time.sleep(0.02)
    return _real_claim(capture_dir, now=now)
_cm.claim_capture_dir = _barrier_claim
import runpy
runpy.run_module("meetandread.__main__", run_name="__main__", alter_sys=True)
"""


class TestRecordingLifecycleInsideCaptureMode:
    """AC5 (fix round 1, finding 3): an actual Recording start->stop
    inside Issue Capture Mode, driven through the existing CLI
    fake-duration seam (tests/test_cli_fake_duration.py — it exists
    precisely to drive recording without real audio). The capture
    DEBUG log must show the recording lifecycle events."""
    def test_record_start_stop_lifecycle_lands_in_capture_log(
        self, tmp_path
    ):
        import wave

        import numpy as np

        # Fake audio input (same generator as test_cli_fake_duration).
        input_wav = tmp_path / "input_3s.wav"
        rate = 16000
        t = np.linspace(0, 3.0, int(rate * 3.0), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(input_wav), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm.tobytes())

        capture_dir = tmp_path / "capture" / "recording-run"
        output_dir = tmp_path / "recordings"
        output_dir.mkdir()
        env = _sandbox_env(tmp_path, f"mar_cap_{uuid.uuid4().hex}")

        # Capture-mode recording shim: configure capture logging FIRST
        # (the production bootstrap discipline — logging live before any
        # audio subsystem import), then run the REAL recording pipeline
        # through the CLI's record command (fake source, no real audio
        # device needed): session.start -> frames -> session.stop ->
        # WAV finalization.
        cmd = [
            sys.executable,
            "-c",
            _RECORDING_CAPTURE_SHIM,
            "record",
            "--fake", str(input_wav),
            "--seconds", "3",
            "--output-dir", str(output_dir),
            "--capture-dir", str(capture_dir),
        ]
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            "capture-mode recording run failed "
            f"(rc={result.returncode})\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

        # The recording produced a real WAV through the full pipeline.
        wav_files = list(output_dir.glob("*.wav"))
        assert wav_files, "no WAV produced by the capture-mode recording"

        # THE AC5 assertion: the capture DEBUG log holds the recording
        # lifecycle events of the run — session start, source start,
        # stop, and finalization.
        records = _read_log_records(capture_dir)
        joined = "\n".join(records)
        assert "Issue Capture Mode: streaming diagnostics into" in joined
        assert "session_start: sources=[fake]" in joined, (
            "recording start lifecycle missing from capture log"
        )
        assert "source_start: source=fake" in joined
        assert "session_stop: state=stopping" in joined, (
            "recording stop lifecycle missing from capture log"
        )
        assert "session_stop: state=finalized" in joined, (
            "recording finalization missing from capture log"
        )
        # Ordering: start precedes stop in the stream.
        assert joined.index("session_start: sources=[fake]") < joined.index(
            "session_stop: state=finalized"
        )
        # Full DEBUG level was live for the recording run.
        assert any(
            "Log level: DEBUG (capture mode: True)" in r for r in records
        )


# sitecustomize pre-hook for the no-flag exit-path test: forces the
# meetandread.main logger to DEBUG (the normal run configures the root
# at INFO, which would filter a hard_exit record by level and make the
# test's absence assertion vacuous). The child interpreter loads this
# from PYTHONPATH before any app code; NOTSET on the module logger
# means the level applies without touching the run's root INFO level.
_DEBUG_MAIN_LOGGER_SITECUSTOMIZE = """
import logging
logging.getLogger("meetandread.main").setLevel(logging.DEBUG)
"""


# Capture-mode recording shim: capture logging configured BEFORE the
# audio subsystem import (bootstrap discipline), then the real CLI
# record command runs inside it (fake source — the existing recording
# seam, no real audio devices required).
_RECORDING_CAPTURE_SHIM = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
_args = sys.argv[1:]
_i = _args.index("--capture-dir")
_capture_dir = _args[_i + 1]
_rest = _args[:_i] + _args[_i + 2:]
import meetandread.capture_mode as _cm
_cm.configure_capture_logging(__import__("pathlib").Path(_capture_dir))
from meetandread.audio.cli import main as _cli_main
sys.argv = ["meetandread.audio.cli"] + _rest
sys.exit(_cli_main())
"""
