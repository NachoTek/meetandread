"""Issue Reporter subprocess tests (issue #107,
docs/specs/issue-reporting.md, ADR 0003).

``windows``-marked (ADR 0001): these tests drive the REAL app under
the REAL reporter supervisor — full Qt widget, recording controller,
native import surface — so they need the Windows interpreter and are
skipped off-Windows. They cover the supervise-and-crash flow end to
end:

1. **Supervised clean run**: the wizard core launches the real app in
   Issue Capture Mode pointed at a fresh reporter-created capture
   directory; the app starts, the reporter signals the graceful user
   stop (CTRL_BREAK), the app exits 0 with the completion marker, and
   the reporter writes a ``user_stop`` termination record with
   start/end timestamps and the marker's presence.
2. **Supervised crash run**: the app is killed hard mid-run
   (``taskkill /F`` — the crash equivalent) BEFORE the user
   finishes; the reporter SURVIVES, classifies ``crash`` (nonzero
   exit, no marker), writes the crash termination record, and the
   wizard flow completes — the capture directory holds the claim,
   the capture log, and the termination record.
3. **Standalone recovery**: with a real interrupted capture directory
   on disk (claim, streaming log, no marker, no termination record),
   reporter startup's recovery scan finds it as ``incomplete`` and
   resuming fills in the crash termination record — no app launch
   needed.
4. **Single-instance gate**: with the real app's mutex held by a
   live process, the reporter's mutex probe returns True; after the
   holder exits, False.

Plumbing notes:

- The reporter driver runs the REAL ``run_wizard`` in a child process
  with injectable seams: scripted input answers come from a
  parent-controlled sentinel-file protocol (the parent writes each
  next answer when it chooses, so it can time the kill against the
  wizard's prompts), and output milestones land in a stdout file the
  parent polls.
- The app child (``python -m meetandread``) gets its single-instance
  lock-name override via a ``sitecustomize`` pre-hook earliest on
  PYTHONPATH — the driver process cannot patch a child it spawns
  fresh.
"""

import os
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
)
from meetandread.reporter import (
    CAPTURES_DIRNAME,
    TERMINATION_FILE_NAME,
    read_termination_record,
)

# Real app subprocesses: Windows native stack required (ADR 0001).
pytestmark = pytest.mark.windows

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

STARTUP_TIMEOUT_S = 90.0
CLEAN_EXIT_TIMEOUT_S = 45.0


def _sandbox_env(tmp_path: Path, lock_name: str) -> dict:
    """Same sandbox shape as the #104/#105/#106 subprocess tests, plus
    the reporter's data base pointed at the sandbox."""
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
            "MAR_REPORTER_DATA_BASE": str(tmp_path),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(SRC),
        }
    )
    return env


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    pytest.fail(f"timed out waiting for {what}")


def _kill_hard(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# The reporter driver
# ---------------------------------------------------------------------------

# Runs the REAL run_wizard in a child. Input answers arrive through a
# request/answer sentinel protocol: the driver writes
# <inbox>/<n>.ask, the parent (when it chooses) writes
# <inbox>/<n>.ans — giving the PARENT control over WHEN each answer
# lands (essential for timing the mid-reproduction kill against the
# wizard's "Press Enter" prompt). Milestones print to stdout.
# launch_app is wrapped ONLY to echo the spawned app's pid (the
# parent needs it for the mid-run kill; no production behavior
# changes).
_REPORTER_DRIVER = '''
import os
import sys
import time

sys.path.insert(0, sys.argv[1])

inbox = sys.argv[2]

import meetandread.reporter as _reporter
from meetandread.reporter_wizard import run_wizard

_launch_app_real = _reporter.launch_app


def _launch_app_echoing_pid(capture_dir, app_command=None, extra_env=None):
    proc = _launch_app_real(capture_dir, app_command=app_command,
                            extra_env=extra_env)
    print(f"MAR_APP_PID: {proc.pid}", flush=True)
    return proc


_reporter.launch_app = _launch_app_echoing_pid


def _next_answer(n):
    ask = os.path.join(inbox, f"{n}.ask")
    ans = os.path.join(inbox, f"{n}.ans")
    with open(ask, "w", encoding="utf-8") as fh:
        fh.write("1")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if os.path.exists(ans):
            with open(ans, "r", encoding="utf-8") as fh:
                return fh.read()
        time.sleep(0.1)
    raise TimeoutError(f"no answer for prompt {n}")


class SentinelInput:
    def __init__(self):
        self.n = 0

    def __call__(self, prompt=""):
        self.n += 1
        val = _next_answer(self.n)
        print(f"MAR_WIZARD_ANSWERED#{self.n}", flush=True)
        return val


def print_fn(*args, **kwargs):
    print(*args, flush=True)


run = run_wizard(
    app_command=[sys.executable, "-m", "meetandread"],
    input_fn=SentinelInput(),
    print_fn=print_fn,
)
if run is not None:
    print(f"MAR_WIZARD_RUN_DIR: {run.capture_dir}", flush=True)
    print(f"MAR_WIZARD_OUTCOME: {run.outcome.value}", flush=True)
    print(f"MAR_WIZARD_EXIT: {run.exit_code}", flush=True)
else:
    print("MAR_WIZARD_RESUMED_ONLY", flush=True)
print("MAR_WIZARD_DONE", flush=True)
'''

# The app child's lock-name override: sitecustomize runs before any
# app import in the spawned `python -m meetandread`, patching the
# default mutex name to the per-test one (the same override the
# sibling capture tests inject through their launch shims — here it
# must ride along in the environment because launch_app spawns the
# child itself).
_SITECUSTOMIZE = '''
import os
import sys

if os.environ.get("MAR_TEST_LOCK_NAME"):
    try:
        import meetandread.single_instance as _si
    except Exception:
        _si = None
    if _si is not None:
        _orig = _si.acquire_single_instance_lock

        def _acquire_with_test_name(name=None):
            if name is None or name == "meetandread":
                name = os.environ["MAR_TEST_LOCK_NAME"]
            return _orig(name)

        _si.acquire_single_instance_lock = _acquire_with_test_name
'''


def _prepare_lock_sitecustomize(tmp_path: Path, env: dict) -> None:
    """Make the app child (fresh `python -m meetandread`) pick up the
    per-test mutex name via a sitecustomize dir prepended to
    PYTHONPATH — earliest import hook there is."""
    hook_dir = tmp_path / "lockhook"
    hook_dir.mkdir(exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(
        _SITECUSTOMIZE, encoding="utf-8"
    )
    env["PYTHONPATH"] = f"{hook_dir}{os.pathsep}{env['PYTHONPATH']}"


class SentinelAnswers:
    """Parent-side half of the sentinel input protocol."""

    def __init__(self, tmp_path: Path):
        self.inbox = tmp_path / "inbox"
        self.inbox.mkdir(exist_ok=True)
        self._answered = 0

    def wait_prompt(self, timeout_s: float = STARTUP_TIMEOUT_S) -> int:
        """Wait until the driver asked for answer #_answered+1."""
        want = self._answered + 1
        _wait_for(
            lambda: (self.inbox / f"{want}.ask").exists(),
            timeout_s,
            f"wizard to ask for input #{want}",
        )
        return want

    def answer(self, text: str) -> None:
        n = self._answered + 1
        assert (self.inbox / f"{n}.ask").exists(), (
            f"answering prompt {n} that was never asked"
        )
        (self.inbox / f"{n}.ans").write_text(text, encoding="utf-8")
        self._answered = n


def _launch_reporter_driver(
    tmp_path: Path, env: dict
) -> subprocess.Popen:
    stdout_file = tmp_path / "reporter-stdout.txt"
    stdout_fh = open(stdout_file, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _REPORTER_DRIVER,
            str(SRC),
            str(tmp_path / "inbox"),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    proc._mar_stdout_fh = stdout_fh  # type: ignore[attr-defined]
    return proc


def _stdout_text(tmp_path: Path, limit: int = 6000) -> str:
    try:
        return (tmp_path / "reporter-stdout.txt").read_text(
            encoding="utf-8", errors="replace"
        )[-limit:]
    except OSError:
        return ""


def _capture_dirs(tmp_path: Path) -> list:
    # The reporter's data base IS tmp_path (MAR_REPORTER_DATA_BASE),
    # so runs land at tmp_path/captures/run-*.
    captures = tmp_path / CAPTURES_DIRNAME
    if not captures.is_dir():
        return []
    return sorted(d for d in captures.iterdir() if d.is_dir())


def _capture_log_has(capture_dir: Path, needle: str) -> bool:
    logs = list(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))
    if not logs:
        return False
    try:
        content = logs[0].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return needle in content


def _find_app_child_pid(
    reporter_pid: int, capture_dir: Path, tmp_path: Path
):
    """Find the `python -m meetandread` app child of the driver.

    Preferred source: the driver's own stdout — ``launch_app`` echoes
    ``MAR_APP_PID: <pid>`` when the environment asks for it (exact,
    no process-table scraping, immune to venv shim process layers).
    Fallback: the PowerShell/CIM two-generation child scan (direct
    child or grandchild matching the capture-dir command line).
    """
    try:
        stdout_text = (tmp_path / "reporter-stdout.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        stdout_text = ""
    for line in stdout_text.splitlines():
        if line.startswith("MAR_APP_PID: "):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    helper = Path(__file__).resolve().parent / "_find_children.ps1"
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(helper),
                str(reporter_pid),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    needle = str(capture_dir).replace("\\", "/")
    for line in out.stdout.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) < 4 or parts[0] not in ("CHILD", "GRAND"):
            continue
        _, pid_str, _name, cmdline = parts
        if (
            needle in cmdline.replace("\\", "/")
            and "meetandread" in cmdline
        ):
            try:
                return int(pid_str)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# 1. Supervised clean run (user stop via the wizard)
# ---------------------------------------------------------------------------


class TestSupervisedCleanRun:
    def test_wizard_supervises_real_app_to_user_stop(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_rep_{uuid.uuid4().hex}")
        _prepare_lock_sitecustomize(tmp_path, env)
        answers = SentinelAnswers(tmp_path)
        proc = _launch_reporter_driver(tmp_path, env)
        try:
            # Describe step: answer the description prompt.
            answers.wait_prompt()
            answers.answer("Real app user-stop description")

            # The wizard gates, creates the capture dir, launches the
            # app; wait for the app to be fully up (its capture log
            # reaches the post-processing milestone — the #104 tests'
            # fully-started milestone).
            _wait_for(
                lambda: len(_capture_dirs(tmp_path)) == 1
                and _capture_log_has(
                    _capture_dirs(tmp_path)[0],
                    "PostProcessingQueue worker started",
                ),
                STARTUP_TIMEOUT_S,
                "reporter-launched app fully started",
            )
            capture_dir = _capture_dirs(tmp_path)[0]

            # Reproduce done: answer the "Press Enter" prompt; the
            # wizard sends the graceful stop and waits for exit 0 +
            # marker, then writes the termination record.
            answers.wait_prompt()
            answers.answer("")

            # Review (#108) then the submit offer (#109): decline the
            # browser open (the test harness must not launch a real
            # browser; the wizard prints the not-opened message and
            # the flow completes).
            answers.wait_prompt()
            answers.answer("n")

            _wait_for(
                lambda: "MAR_WIZARD_DONE" in _stdout_text(tmp_path),
                STARTUP_TIMEOUT_S + CLEAN_EXIT_TIMEOUT_S,
                "wizard flow to complete after the user stop",
            )
            exit_code = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)

        assert exit_code == 0, (
            f"reporter driver failed: {_stdout_text(tmp_path)}"
        )
        out = _stdout_text(tmp_path)
        assert "MAR_WIZARD_OUTCOME: clean_stop" in out

        # THE termination record: user stop (wizard-refined), exit 0,
        # marker present, real timestamps.
        record = read_termination_record(capture_dir)
        assert record is not None
        assert record["outcome"] == "user_stop"
        assert record["exit_code"] == 0
        assert record["marker_present"] is True
        assert (capture_dir / COMPLETION_MARKER_NAME).is_file()
        assert record["started_at"] <= record["ended_at"]

        # The fresh-directory discipline: the run dir is under the
        # reporter's captures base, timestamped, with the claim and
        # the description carried forward.
        assert capture_dir.name.startswith("run-")
        assert (capture_dir / CLAIM_FILE_NAME).is_file()
        desc = (capture_dir / "description.txt").read_text(
            encoding="utf-8"
        )
        assert "user-stop description" in desc

        # The #108 review step ran in the wizard: the assembled,
        # redacted bundle was written into the capture directory (the
        # review screen shows exactly this artifact), the wizard's
        # output named it, and the bundle itself contains no username
        # from the sandboxed profile.
        bundle = capture_dir / "diagnostics_bundle.txt"
        assert bundle.is_file(), "review step wrote no bundle artifact"
        bundle_text = bundle.read_text(encoding="utf-8")
        assert "diagnostics bundle v1" in bundle_text
        assert "== environment ==" in bundle_text
        assert "== termination ==" in bundle_text
        assert "user_stop" in bundle_text
        assert env["USERNAME"] not in bundle_text, (
            "sandbox username leaked into the bundle"
        )
        assert str(Path(env["USERPROFILE"])) not in bundle_text, (
            "sandbox home path leaked into the bundle"
        )
        out_full = (tmp_path / "reporter-stdout.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "what will be sent" in out_full


# ---------------------------------------------------------------------------
# 2. Supervised crash run (hard kill mid-run; reporter survives)
# ---------------------------------------------------------------------------


class TestSupervisedCrashRun:
    def test_reporter_survives_and_records_hard_kill_crash(
        self, tmp_path
    ):
        env = _sandbox_env(tmp_path, f"mar_rep_{uuid.uuid4().hex}")
        _prepare_lock_sitecustomize(tmp_path, env)
        answers = SentinelAnswers(tmp_path)
        proc = _launch_reporter_driver(tmp_path, env)
        try:
            answers.wait_prompt()
            answers.answer("Crash during reproduction description")

            _wait_for(
                lambda: len(_capture_dirs(tmp_path)) == 1
                and _capture_log_has(
                    _capture_dirs(tmp_path)[0],
                    "PostProcessingQueue worker started",
                ),
                STARTUP_TIMEOUT_S,
                "reporter-launched app fully started pre-kill",
            )
            capture_dir = _capture_dirs(tmp_path)[0]

            # THE crash: hard-kill the app child mid-reproduction,
            # while the wizard still waits at the Enter prompt.
            app_pid = _find_app_child_pid(proc.pid, capture_dir, tmp_path)
            assert app_pid is not None, "app child process not found"
            _kill_hard(app_pid)

            # Now the user comes back and presses Enter; the wizard
            # observes the dead app, classifies the crash, records it,
            # and completes — the reporter survived. The submit offer
            # (#109) follows the review: decline it (no browser).
            answers.wait_prompt()
            answers.answer("")
            answers.wait_prompt()
            answers.answer("n")
            _wait_for(
                lambda: "MAR_WIZARD_DONE" in _stdout_text(tmp_path),
                STARTUP_TIMEOUT_S,
                "wizard flow to complete after the crash",
            )
            exit_code = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)

        assert exit_code == 0, (
            f"reporter driver failed: {_stdout_text(tmp_path)}"
        )
        out = _stdout_text(tmp_path)
        assert "MAR_WIZARD_OUTCOME: crash" in out

        record = read_termination_record(capture_dir)
        assert record is not None
        assert record["outcome"] == "crash"
        assert record["exit_code"] != 0
        assert record["marker_present"] is False
        assert not (capture_dir / COMPLETION_MARKER_NAME).exists()
        # The run's artifacts survived the crash for the bundle.
        assert (capture_dir / CLAIM_FILE_NAME).is_file()
        assert list(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))

        # The #108 review step assembled the CRASHED run's directory
        # too — incompleteness is a value, never an assembly failure:
        # the bundle exists and records the crash.
        bundle = capture_dir / "diagnostics_bundle.txt"
        assert bundle.is_file(), (
            "review step wrote no bundle for the crashed run"
        )
        crash_text = bundle.read_text(encoding="utf-8")
        assert "crash" in crash_text
        assert "completion marker: absent" in crash_text
        assert env["USERNAME"] not in crash_text
        # The wizard TOLD the user the crash was captured.
        assert "exited unexpectedly" in out


# ---------------------------------------------------------------------------
# 3. Standalone recovery (real interrupted capture directory on disk)
# ---------------------------------------------------------------------------


class TestStandaloneRecovery:
    def test_startup_scan_resumes_real_interrupted_capture(
        self, tmp_path
    ):
        env = _sandbox_env(tmp_path, f"mar_rep_{uuid.uuid4().hex}")

        # Stage a REAL interrupted run: launch the real app under
        # capture (through the production bootstrap), wait until its
        # log streams, kill it hard — the directory then holds claim +
        # partial log, no marker, no termination record. Staged inside
        # the reporter's captures base (where the wizard's recovery
        # scan looks), as an interrupted previous session's run.
        capture_dir = tmp_path / CAPTURES_DIRNAME / "run-staged"
        capture_dir.mkdir(parents=True)
        app = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, "
                    + repr(str(SRC))
                    + "); import runpy; runpy.run_module("
                    "'meetandread.__main__', run_name='__main__', "
                    "alter_sys=True)"
                ),
                "--issue-capture",
                str(capture_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            _wait_for(
                lambda: _capture_log_has(capture_dir, "app_startup"),
                STARTUP_TIMEOUT_S,
                "staged app run to start streaming",
            )
            _kill_hard(app.pid)
            app.wait(timeout=15)
        finally:
            if app.poll() is None:
                _kill_hard(app.pid)
                app.wait(timeout=15)

        assert (capture_dir / CLAIM_FILE_NAME).is_file()
        assert read_termination_record(capture_dir) is None

        # Reporter startup: the wizard's recovery offer finds the
        # incomplete run and resuming fills in the crash record —
        # WITHOUT launching the app (no description answer needed).
        answers = SentinelAnswers(tmp_path)
        proc = _launch_reporter_driver(tmp_path, env)
        try:
            answers.wait_prompt()
            answers.answer("y")
            # The resumed run's review passes and the submit offer
            # (#109) appears: decline it (no browser in tests).
            answers.wait_prompt()
            answers.answer("n")
            _wait_for(
                lambda: "MAR_WIZARD_DONE" in _stdout_text(tmp_path),
                STARTUP_TIMEOUT_S,
                "recovery flow to complete",
            )
            exit_code = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)

        assert exit_code == 0, (
            f"recovery driver failed: {_stdout_text(tmp_path)}"
        )
        out = _stdout_text(tmp_path)
        assert "state: incomplete" in out
        assert "Resumed" in out
        assert "MAR_WIZARD_RESUMED_ONLY" in out

        record = read_termination_record(capture_dir)
        assert record is not None
        assert record["outcome"] == "crash"
        assert record["marker_present"] is False


# ---------------------------------------------------------------------------
# 4. Single-instance gate (real mutex, cross-process)
# ---------------------------------------------------------------------------


class TestSingleInstanceGate:
    def test_probe_sees_real_mutex_holder(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_rep_{uuid.uuid4().hex}")
        lock = env["MAR_TEST_LOCK_NAME"]

        # A live process holds the app's mutex (the real guard, the
        # real Global\ name) and announces via a sentinel file.
        held = tmp_path / "mutex-held.sentinel"
        holder_src = (
            "import sys, time, os\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "import meetandread.single_instance as si\n"
            f"assert si.acquire_single_instance_lock({lock!r})\n"
            f"open({str(held)!r}, 'w').write('held')\n"
            "time.sleep(300)\n"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", holder_src],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for(
                lambda: held.exists(),
                STARTUP_TIMEOUT_S,
                "mutex holder to announce",
            )
            from meetandread.reporter import app_is_running

            # The gate's probe sees the running app.
            assert app_is_running(lock) is True
        finally:
            _kill_hard(holder.pid)
            holder.wait(timeout=15)

        from meetandread.reporter import app_is_running

        # Released after exit: the gate passes.
        assert app_is_running(lock) is False
