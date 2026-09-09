"""Interaction Trace subprocess tests (issue #105, docs/specs/issue-reporting.md).

``windows``-marked (ADR 0001), same patterns as the #104
capture-mode subprocess tests (tests/test_capture_mode_subprocess.py):

1. Kill mid-run durability: launch the real app with
   ``--issue-capture <DIR>``, drive it to emit several trace events
   (an in-child driver drives the REAL widget funnels — record button,
   lobes, panel toggles, text field with a canary), forcibly kill with
   ``taskkill /F``, and assert every completed trace event survived on
   disk (prompt-flush contract) while the typed canary content appears
   NOWHERE — not in the trace and not in the capture DEBUG log.
2. Normal run: no capture flag — no trace file is created anywhere in
   the sandbox.

The in-child driver drives real Qt widgets with the trace installed
through the production startup path (the lightweight bootstrap), so
these tests cover the launch edge + artifact edge end to end.
"""

import json
import os
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
)
from meetandread.interaction_trace import (
    EVENT_VOCABULARY,
    TRACE_FILE_NAME,
    read_trace_events,
)

# Real app subprocesses: Windows native stack required (ADR 0001).
pytestmark = pytest.mark.windows

REPO_ROOT = Path(__file__).resolve().parent.parent

STARTUP_TIMEOUT_S = 60.0

# The canary the child types into a real text field. It must appear in
# NEITHER the trace file NOR the capture DEBUG log — only its length.
TYPED_CANARY = "SUBPROCESS-canary-Sebastopol-9931-typed-text"


def _sandbox_env(tmp_path: Path, lock_name: str) -> dict:
    """Same sandbox shape as the #104 subprocess tests."""
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
            "MAR_TRACE_CANARY": TYPED_CANARY,
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return env


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


# The in-child event driver: patches the single-instance lock name (as
# the #104 tests do), runs the REAL production bootstrap, and — instead
# of the interactive event loop — drives the real widget funnels with
# real Qt events once the app is up, sleeps to hold the run open for
# the kill, and prints the milestone line the parent waits on. The
# trace + filter are installed by main() itself (the production path).
_EVENT_DRIVER_SHIM = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig_acquire = _si.acquire_single_instance_lock
def _acquire_with_test_name(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig_acquire(name)
_si.acquire_single_instance_lock = _acquire_with_test_name

# Import the bootstrap pieces directly so we can drive the app's
# startup sequence and then the widget funnels, without exec().
import meetandread.__main__ as _boot
_boot._bootstrap()

# The trace installs at exactly the production point (main() after the
# bootstrap configured capture logging): replicate that sequence, then
# the Qt filter after the QApplication exists — as main() does.
from meetandread.interaction_trace import install_interaction_trace
install_interaction_trace(_boot._CAPTURE_FLAG_PARSED)

from PyQt6.QtWidgets import QApplication, QLineEdit
from PyQt6.QtCore import Qt, QEvent

app = QApplication.instance() or QApplication(sys.argv)
from meetandread.interaction_trace_qt import install_interaction_trace_qt_filter
install_interaction_trace_qt_filter(app)

# MeetAndReadWidget owns the funnels; build it exactly as main() does
# (post-processing init is the controller's, not the widget's concern
# for this drive).
from meetandread.widgets.main_widget import MeetAndReadWidget
widget = MeetAndReadWidget()
widget.show()
app.processEvents()

# Click-vs-drag state is normally set by the widget's own press event;
# the driver synthesizes releases only, so preset the short-click state
# exactly as a real press would leave it.
widget.is_dragging = False
widget._click_consumed = False

canary = os.environ["MAR_TRACE_CANARY"]

# --- Drive the REAL funnels with REAL Qt events ----------------------
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QMouseEvent

def _release():
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(10, 10), QPointF(10, 10),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

# 1. device selection: mic lobe on, then off (real mouse release path)
widget.mic_lobe.mouseReleaseEvent(_release())
widget.mic_lobe.mouseReleaseEvent(_release())
# 2. settings panel open, then close (real panel funnel)
widget._toggle_settings_panel()
widget._toggle_settings_panel()
# 3. record button (real mouse release path; recording start will fail
#    harmlessly with no sources selected — the button event stands)
widget.record_button.mouseReleaseEvent(_release())
# 4. free text into a real line edit through the app-level filter
edit = QLineEdit()
edit.show()
edit.setFocus()
app.processEvents()
edit.insert(canary)
app.processEvents()
QApplication.sendEvent(edit, QEvent(QEvent.Type.FocusOut))
app.processEvents()
edit.hide()

print("MAR_TRACE_DRIVE_COMPLETE", flush=True)

# Hold the run open mid-stream so the forced kill lands mid-run.
import time as _t
_t.sleep(300)
"""


class TestKillMidRunDurability:
    """AC: a forced kill preserves every completed trace event."""

    def test_forced_kill_preserves_every_completed_trace_event(
        self, tmp_path
    ):
        capture_dir = tmp_path / "capture" / "trace-crash"
        env = _sandbox_env(tmp_path, f"mar_tr_{uuid.uuid4().hex}")
        stdout_file = tmp_path / "app-stdout.txt"
        with open(stdout_file, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen(
                # The flag args after -c land in the child's sys.argv —
                # the production bootstrap parses them from there.
                [
                    sys.executable,
                    "-c",
                    _EVENT_DRIVER_SHIM,
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
            # Wait for the driver to finish driving the funnels: the
            # trace must hold every event by the time the milestone is
            # printed (each emit is flushed+fsynced on write).
            _wait_for(
                lambda: "MAR_TRACE_DRIVE_COMPLETE"
                in _stdout_text(stdout_file),
                STARTUP_TIMEOUT_S,
                "in-child trace event drive to complete",
            )
            # Every event emitted so far is ALREADY durable: snapshot
            # what the contract must preserve across the kill.
            before_kill = read_trace_events(capture_dir)
            assert len(before_kill) >= 6, (
                f"expected the driven events on disk pre-kill, got "
                f"{before_kill}"
            )
            _kill_hard(proc)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        # THE durability assertion: every completed event survived the
        # forced kill, byte-identical, append-only.
        after_kill = read_trace_events(capture_dir)
        assert after_kill[: len(before_kill)] == before_kill

        # At most one torn record: the raw tail after the last preserved
        # event is empty or exactly one unterminated fragment — and
        # that fragment is not any completed event (it was in-flight).
        raw = (capture_dir / TRACE_FILE_NAME).read_text(
            encoding="utf-8", errors="replace"
        )
        if not raw.endswith("\n"):
            torn_lines = raw.split("\n")
            # Everything before the torn fragment is exactly the
            # preserved completed events.
            assert torn_lines[:-1] == [
                json.dumps(e, ensure_ascii=True) for e in before_kill
            ], (
                "raw pre-torn content does not match the preserved "
                "completed events"
            )

        # The named vocabulary arrived: device selection, panel
        # open/close, button, text — all present with timestamps.
        kinds = [e["event"] for e in after_kill]
        assert "device_selected" in kinds
        assert "panel_opened" in kinds and "panel_closed" in kinds
        assert "button_pressed" in kinds
        # Typed free text is EXACTLY one text_edited char count.
        text_events = [e for e in after_kill if e["event"] == "text_edited"]
        assert len(text_events) == 1, (
            f"expected exactly one text_edited event, got {text_events}"
        )
        assert text_events[0]["chars"] == len(TYPED_CANARY)

        # THE privacy assertion: the canary is nowhere — not in the
        # trace, not in the capture DEBUG log (batch C instrumentation).
        raw_trace = (capture_dir / TRACE_FILE_NAME).read_text(
            encoding="utf-8", errors="replace"
        )
        assert TYPED_CANARY not in raw_trace
        assert "Sebastopol" not in raw_trace
        capture_log_raw = _find_capture_log(capture_dir).read_text(
            encoding="utf-8", errors="replace"
        )
        assert TYPED_CANARY not in capture_log_raw, (
            "typed content leaked into the capture DEBUG log"
        )
        assert "Sebastopol" not in capture_log_raw, (
            "typed content fragment leaked into the capture DEBUG log"
        )

        # Every trace record is a member of the closed vocabulary.
        from meetandread.interaction_trace import EVENT_VOCABULARY

        assert set(kinds) <= set(EVENT_VOCABULARY)


# ---------------------------------------------------------------------------
# Production-wiring end-to-end: the REAL startup path installs the trace
# (PR #123 re-review finding 3)
# ---------------------------------------------------------------------------

# In-child driver for the production-path test. Unlike the _EVENT_DRIVER_SHIM
# (which patches the lock, calls _bootstrap(), and hand-installs the trace),
# this child ONLY patches the single-instance lock name — everything else is
# the unmodified production startup: meetandread.__main__ parses the flag,
# configures capture logging, main() installs the Interaction Trace and the
# Qt filter, builds the widget, and enters app.exec(). The injected
# QApplication subclass replaces the event loop with a scripted drive of a
# REAL QLineEdit (typed canary, same test discipline as the Qt seam tests),
# then quits; the trace events land through the production wiring alone.
_PRODUCTION_WIRING_SHIM = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig_acquire = _si.acquire_single_instance_lock
def _acquire_with_test_name(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig_acquire(name)
_si.acquire_single_instance_lock = _acquire_with_test_name

# Inject a QApplication subclass BEFORE meetandread.main is imported: main()
# constructs QApplication(sys.argv) from its own imported name, so the drive
# rides inside the production main() call itself (flag parse -> logging ->
# trace install -> Qt filter -> widget -> "event loop").
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

class _DrivenQApplication(QApplication):
    def exec(self):
        try:
            from PyQt6.QtCore import QEvent
            from PyQt6.QtWidgets import QLineEdit
            edit = QLineEdit()
            edit.show()
            edit.setFocus()
            self.processEvents()
            edit.insert(os.environ["MAR_TRACE_CANARY"])
            self.processEvents()
            self.sendEvent(edit, QEvent(QEvent.Type.FocusOut))
            self.processEvents()
            edit.hide()
            edit.deleteLater()
            # A lobe toggle through the REAL funnel: the main widget owns it.
            from PyQt6.QtCore import QPointF, Qt
            from PyQt6.QtGui import QMouseEvent
            widget = self.property("mar_main_widget")
            if widget is not None:
                widget.is_dragging = False
                widget._click_consumed = False
                release = QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    QPointF(10, 10), QPointF(10, 10),
                    Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                widget.mic_lobe.mouseReleaseEvent(release)
            self.processEvents()
            print("MAR_E2E_DRIVE_COMPLETE", flush=True)
        except BaseException:
            import traceback
            traceback.print_exc()
            print("MAR_E2E_DRIVE_FAILED", flush=True)
        finally:
            QTimer.singleShot(0, self.quit)
        return super().exec()

import meetandread.main as _main_mod
_main_mod.QApplication = _DrivenQApplication

# The drive needs the main widget: wrap MeetAndReadWidget construction the
# way main() sees nothing of it, by exposing the instance after creation.
_orig_init = _main_mod.MeetAndReadWidget.__init__
def _init_exposing(self, *a, **k):
    _orig_init(self, *a, **k)
    app = QApplication.instance()
    if app is not None:
        app.setProperty("mar_main_widget", self)
_main_mod.MeetAndReadWidget.__init__ = _init_exposing

# THE production entrypoint — module execution of the real bootstrap, with
# the capture flag in argv exactly as the Issue Reporter passes it.
import runpy
runpy.run_module(
    "meetandread.__main__", run_name="__main__", alter_sys=True
)
"""


class TestProductionWiringEndToEnd:
    """AC: the production startup wiring (main.py) installs the trace;
    a real user action lands as a named JSONL event in the capture dir.

    Complements TestKillMidRunDurability (which uses the hand-installed
    _EVENT_DRIVER_SHIM for kill-timing control): this test exercises the
    wiring path only — flag parse -> capture logging -> main() ->
    install_interaction_trace -> Qt filter -> widget -> driven actions.
    """

    def test_production_main_wiring_lands_events_in_trace(self, tmp_path):
        capture_dir = tmp_path / "capture" / "e2e-wiring"
        env = _sandbox_env(tmp_path, f"mar_e2e_{uuid.uuid4().hex}")
        stdout_file = tmp_path / "app-stdout.txt"
        with open(stdout_file, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _PRODUCTION_WIRING_SHIM,
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
            _wait_for(
                lambda: "MAR_E2E_DRIVE_COMPLETE"
                in _stdout_text(stdout_file),
                STARTUP_TIMEOUT_S,
                "production-wiring in-app drive to complete",
            )
            exit_code = proc.wait(timeout=STARTUP_TIMEOUT_S)
        finally:
            if proc.poll() is None:
                _kill_hard(proc)
            proc.wait()

        # Exit discipline: the production app must exit 0 — same
        # contract as the #104 capture-mode subprocess siblings. The
        # teardown-ordering fix (filter removed + writer closed BEFORE
        # app destruction) makes the historical 0xC0000005 teardown
        # crash a real failure again, guarding the fix (PR #123 fix
        # round 3 unmasked this assertion).
        assert exit_code == 0, (
            f"production run failed to exit cleanly (teardown crash? "
            f"exit={exit_code}): {_stdout_text(stdout_file)}"
        )

        events = read_trace_events(capture_dir)
        kinds = [e["event"] for e in events]

        # The trace EXISTS and holds closed-vocabulary events — the
        # production wiring installed it (main.py:375-409 path).
        assert events, (
            "no trace events: production startup never installed the "
            f"trace — stdout: {_stdout_text(stdout_file)}"
        )
        assert set(kinds) <= set(EVENT_VOCABULARY)

        # A real text edit through the production-installed Qt filter:
        # exactly one text_edited char count for the canary.
        text_events = [
            e for e in events if e["event"] == "text_edited"
        ]
        assert len(text_events) == 1, (
            f"expected exactly one text_edited event, got: {events}"
        )
        assert text_events[0]["chars"] == len(TYPED_CANARY)

        # The lobe toggle through the real widget funnel.
        assert "device_selected" in kinds

        # Privacy: the canary is in neither the trace nor the log.
        raw_trace = (capture_dir / TRACE_FILE_NAME).read_text(
            encoding="utf-8", errors="replace"
        )
        assert TYPED_CANARY not in raw_trace
        assert "Sebastopol" not in raw_trace
        capture_log_raw = _find_capture_log(capture_dir).read_text(
            encoding="utf-8", errors="replace"
        )
        assert TYPED_CANARY not in capture_log_raw
        assert "Sebastopol" not in capture_log_raw

        # Clean exit reached the marker contract (production path end);
        # the marker is written by main() itself after app.exec()
        # returns 0 — its presence proves the run finished main().
        from meetandread.capture_mode import read_completion_marker

        assert read_completion_marker(capture_dir) is not None


class TestNormalRunNoTrace:
    """AC: a normal run (no capture mode) records no trace anywhere."""

    def test_normal_run_creates_no_trace_file(self, tmp_path):
        env = _sandbox_env(tmp_path, f"mar_tr_{uuid.uuid4().hex}")
        # Minimal real-widget drive WITHOUT the capture flag: the app
        # code path (widget construction, lobe release, panel toggle)
        # runs with no trace installed.
        shim = """
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
import meetandread.single_instance as _si
_orig = _si.acquire_single_instance_lock
def _acquire(name=None):
    if name is None or name == "meetandread":
        name = os.environ.get("MAR_TEST_LOCK_NAME", name)
    return _orig(name)
_si.acquire_single_instance_lock = _acquire
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF, QEvent
from PyQt6.QtGui import QMouseEvent
app = QApplication.instance() or QApplication(sys.argv)
from meetandread.widgets.main_widget import MeetAndReadWidget
widget = MeetAndReadWidget()
widget.show()
app.processEvents()
widget.is_dragging = False
widget._click_consumed = False
release = QMouseEvent(
    QMouseEvent.Type.MouseButtonRelease,
    QPointF(10, 10), QPointF(10, 10),
    Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
    Qt.KeyboardModifier.NoModifier,
)
widget.mic_lobe.mouseReleaseEvent(release)
widget._toggle_settings_panel()
widget._toggle_settings_panel()
app.processEvents()
print("MAR_NORMAL_DRIVE_COMPLETE", flush=True)
"""
        stdout_file = tmp_path / "app-stdout.txt"
        result = subprocess.run(
            [sys.executable, "-c", shim],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=STARTUP_TIMEOUT_S,
        )
        assert result.returncode == 0, (
            f"normal-run drive failed: {result.stdout[-1500:]} "
            f"{result.stderr[-1500:]}"
        )
        assert "MAR_NORMAL_DRIVE_COMPLETE" in result.stdout

        # No trace file anywhere in the sandbox this run could touch.
        assert not list(tmp_path.rglob(TRACE_FILE_NAME)), (
            "Interaction Trace recorded in a normal run"
        )


def _stdout_text(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""
