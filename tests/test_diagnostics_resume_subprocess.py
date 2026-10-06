"""App-launch diagnostics-resume subprocess tests (issue #110,
docs/specs/issue-reporting.md).

``windows``-marked (ADR 0001): these tests launch the REAL application
as a subprocess — the full Qt widget, recording controller, and native
import surface — so they need the Windows interpreter and are skipped
off-Windows. They cover the #110 reporter-death resilience edge end to
end:

1. **Detection + resume**: with a real assembled-but-unsubmitted
   bundle staged in the reporter's data base (the state a dead
   Issue Reporter leaves behind), a NORMAL app launch shows the
   resume offer, and accepting it runs the same Manual Submission
   flow (browser seam fires, clipboard receives the bundle path) and
   writes the ``submitted`` state — no re-reproduction, the saved
   bundle is consumed as-is.
2. **Decline**: answering "Not now" leaves normal startup otherwise
   unaffected — the app comes up, nothing opens, no state lands, the
   bundle stays offerable.

The dialog reply and the IO seams are scripted via a sitecustomize
pre-hook in the child (the established #104/#107 pattern): the offer's
QMessageBox.exec returns the scripted answer, and the browser /
clipboard seams record their calls to sentinel files the parent polls.
The sitecustomize also points the offer at the sandboxed reporter data
base via MAR_REPORTER_DATA_BASE (the same env the reporter program
honors).

Subprocess environment is fully sandboxed (APPDATA/USERPROFILE point
into pytest's tmp tree) so the app never reads the real user's
recordings, config, or Documents tree.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from meetandread.capture_mode import CLAIM_FILE_NAME
from meetandread.diagnostics_bundle import BUNDLE_FILE_NAME
from meetandread.logging_setup import NORMAL_LOG_PREFIX
from meetandread.manual_submission import (
    SUBMISSION_STATE_FILE_NAME,
    read_submission_state,
)
from meetandread.reporter import (
    CAPTURES_DIRNAME,
    RunOutcome,
    write_termination_record,
)

# Real app subprocesses: Windows native stack required (ADR 0001).
pytestmark = pytest.mark.windows

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

STARTUP_TIMEOUT_S = 90.0
CLEAN_EXIT_TIMEOUT_S = 45.0


def _sandbox_env(tmp_path: Path, lock_name: str) -> dict:
    """Same sandbox shape as the #104 subprocess tests, plus the
    reporter's data base pointed at the sandbox."""
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
            "MAR_REPORTER_DATA_BASE": str(tmp_path / "reporter-data"),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(SRC),
        }
    )
    return env


# The child-side pre-hook (sitecustomize, earliest on PYTHONPATH):
# scripts the offer dialog's reply and stubs the Manual Submission IO
# seams, recording their calls to sentinel files the parent polls.
# The reply is keyed on the dialog's window TITLE (not call order) so
# unrelated startup dialogs (hardware notices, recording recovery)
# answer Ok and never consume the scripted reply. The offer itself
# runs UNMODIFIED production logic in between.
_RESUME_SITECUSTOMIZE = '''
import os

if os.environ.get("MAR_TEST_REPLY"):
    from PyQt6.QtWidgets import QMessageBox as _QB

    _reply = {
        "yes": _QB.StandardButton.Yes.value,
        "no": _QB.StandardButton.No.value,
        "discard": _QB.StandardButton.Discard.value,
    }[os.environ["MAR_TEST_REPLY"]]

    import meetandread.main as _main

    class _ScriptedMsgBox(_QB):
        def exec(self, *a, **k):
            if "Unsubmitted Diagnostics Bundle" in self.windowTitle():
                return _reply
            return _QB.StandardButton.Ok.value

    _main.QMessageBox = _ScriptedMsgBox

    import meetandread.manual_submission as _ms

    def _record_open(url):
        with open(
            os.environ["MAR_TEST_SENTINELS"] + "/browser.txt", "w",
            encoding="utf-8",
        ) as fh:
            fh.write(url)
        return True

    def _record_clip(text):
        with open(
            os.environ["MAR_TEST_SENTINELS"] + "/clipboard.txt", "w",
            encoding="utf-8",
        ) as fh:
            fh.write(text)
        return True

    _ms.open_new_issue_form = _record_open
    _ms.copy_to_clipboard = _record_clip
'''


def _prepare_resume_hook(tmp_path: Path, env: dict) -> Path:
    sentinels = tmp_path / "sentinels"
    sentinels.mkdir(exist_ok=True)
    hook_dir = tmp_path / "resumehook"
    hook_dir.mkdir(exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(
        _RESUME_SITECUSTOMIZE, encoding="utf-8"
    )
    env["PYTHONPATH"] = f"{hook_dir}{os.pathsep}{env['PYTHONPATH']}"
    env["MAR_TEST_SENTINELS"] = str(sentinels)
    return sentinels


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


def _launch_app(env: dict, stdout_file: Path) -> subprocess.Popen:
    stdout_fh = open(stdout_file, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-c", _LAUNCH_SHIM],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    proc._mar_stdout_fh = stdout_fh  # type: ignore[attr-defined]
    return proc


def _stage_unsubmitted(tmp_path: Path) -> Path:
    """Stage a REAL assembled-but-unsubmitted capture run: a capture
    directory with contract artifacts, a termination record, and a
    bundle artifact on disk — exactly what a dead Issue Reporter
    leaves behind after review passed but submit never resolved."""
    base = tmp_path / "reporter-data"
    d = base / CAPTURES_DIRNAME / "run-staged"
    d.mkdir(parents=True)
    (d / CLAIM_FILE_NAME).write_text(
        '{"started_at": "2026-09-10T10:00:00"}\n', encoding="utf-8"
    )
    write_termination_record(
        d,
        outcome=RunOutcome.USER_STOP,
        exit_code=0,
        started_at=datetime(2026, 9, 10, 10, 0, 0),
        ended_at=datetime(2026, 9, 10, 10, 2, 30),
        marker_present=True,
    )
    (d / "description.txt").write_text(
        "Settings panel froze the app\n", encoding="utf-8"
    )
    # The bundle artifact review wrote (a minimal valid v1 bundle).
    (d / BUNDLE_FILE_NAME).write_text(
        "diagnostics bundle v1\n", encoding="utf-8"
    )
    return d


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


def _normal_log_text(tmp_path: Path) -> str:
    logs_dir = tmp_path / "home" / "Documents" / "meetandread" / "logs"
    logs = (
        list(logs_dir.glob(f"{NORMAL_LOG_PREFIX}*.log"))
        if logs_dir.is_dir()
        else []
    )
    if not logs:
        return ""
    try:
        return logs[-1].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class TestAppLaunchOffersResume:
    def test_accepting_resume_runs_manual_submission(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_110_{uuid.uuid4().hex}")
        d = _stage_unsubmitted(tmp_path)
        sentinels = _prepare_resume_hook(tmp_path, env)
        env["MAR_TEST_REPLY"] = "yes"

        proc = _launch_app(env, tmp_path / "app-stdout.txt")
        try:
            # The offer resolved before the event loop starts: the
            # browser seam fired and the submitted state landed.
            _wait_for(
                lambda: (sentinels / "browser.txt").is_file()
                and read_submission_state(d) == "submitted",
                STARTUP_TIMEOUT_S,
                "resume offer to run Manual Submission",
            )
            browser = (sentinels / "browser.txt").read_text(
                encoding="utf-8"
            )
            assert browser.startswith(
                "https://github.com/NachoTek/meetandread/issues/new?"
            )
            clipboard = (sentinels / "clipboard.txt").read_text(
                encoding="utf-8"
            )
            assert clipboard == str(d / BUNDLE_FILE_NAME)
            # The startup log names the resume.
            _wait_for(
                lambda: "diagnostics_resume" in _normal_log_text(tmp_path),
                STARTUP_TIMEOUT_S,
                "diagnostics_resume record in the normal log",
            )
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)

        log = _normal_log_text(tmp_path)
        assert "diagnostics_resume: action=submitted" in log

    def test_declining_leaves_startup_unaffected(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_110_{uuid.uuid4().hex}")
        d = _stage_unsubmitted(tmp_path)
        sentinels = _prepare_resume_hook(tmp_path, env)
        env["MAR_TEST_REPLY"] = "no"

        proc = _launch_app(env, tmp_path / "app-stdout.txt")
        try:
            # The app comes up fully (the offer did not block or break
            # startup) and nothing left the machine.
            _wait_for(
                lambda: "diagnostics_resume: action=postponed"
                in _normal_log_text(tmp_path),
                STARTUP_TIMEOUT_S,
                "postponed record in the normal log",
            )
            _wait_for(
                lambda: "PostProcessingQueue worker started"
                in _normal_log_text(tmp_path),
                STARTUP_TIMEOUT_S,
                "app fully started after declining",
            )
            assert not (sentinels / "browser.txt").exists()
            assert read_submission_state(d) is None
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)

    def test_discarding_resolves_state(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_110_{uuid.uuid4().hex}")
        d = _stage_unsubmitted(tmp_path)
        sentinels = _prepare_resume_hook(tmp_path, env)
        env["MAR_TEST_REPLY"] = "discard"

        proc = _launch_app(env, tmp_path / "app-stdout.txt")
        try:
            _wait_for(
                lambda: read_submission_state(d) == "discarded",
                STARTUP_TIMEOUT_S,
                "discarded state to land",
            )
            assert not (sentinels / "browser.txt").exists()
            # The capture data survives (retention is #112's lane).
            assert (d / BUNDLE_FILE_NAME).is_file()
            assert (d / CLAIM_FILE_NAME).is_file()
        finally:
            if proc.poll() is None:
                _kill_hard(proc.pid)
                proc.wait(timeout=15)
