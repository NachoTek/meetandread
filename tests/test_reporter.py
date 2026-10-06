"""Issue Reporter supervisor core tests (issue #107,
docs/specs/issue-reporting.md, ADR 0003).

Pure-logic, fast-lane compatible (ADR 0001): the module under test is
stdlib-only — the reporter is deliberately small and defensive (no
audio stack, no Qt widget tree), and these tests enforce that boundary
structurally (no heavy module may be imported transitively).

Covers the supervisor core:

- Termination record: how the run ended (clean stop / user stop /
  crash), start/end timestamps, the app's exit code, and the
  completion marker's presence — written into the capture directory at
  run end, prompt-flushed.
- Run supervision: launching the app with ``--issue-capture <DIR>``,
  waiting, classifying the end (clean stop via exit 0 + marker; crash
  via nonzero exit or death by signal), and the reporter SURVIVING a
  crashed app (the classification is a value, never a propagated
  exception).
- Fresh-capture-directory discipline (ADR 0005): the reporter always
  creates a fresh directory per run, timestamped, never reusing one.
- Standalone recovery scan: startup finds incomplete (claim, no
  marker, no termination record) and review-ready (termination record
  present) capture directories and offers to resume them.
- Single-instance guard interplay: an already-running app blocks the
  launch step until closed (the reporter asks the user; the pure seam
  is a boolean check).
- Description capture: the user's free-text description is carried
  forward (stored alongside the run for the later review/submission
  steps — #108/#109).

The ``windows``-marked subprocess tests (tests/test_reporter_subprocess.py)
drive the full supervise-and-crash flow with real processes.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import meetandread.reporter as reporter
from meetandread.capture_mode import (
    CLAIM_FILE_NAME,
    COMPLETION_MARKER_NAME,
)
from meetandread.reporter import (
    CAPTURES_DIRNAME,
    DESCRIPTION_FILE_NAME,
    TERMINATION_FILE_NAME,
    RunOutcome,
    find_resumable_captures,
    new_capture_dir,
    read_description,
    read_termination_record,
    scan_capture_state,
    write_description,
    write_termination_record,
)


# ---------------------------------------------------------------------------
# The defensive boundary: stdlib-only module, no heavy imports
# ---------------------------------------------------------------------------

_HEAVY_TOP_LEVELS = {
    "PyQt6",
    "sounddevice",
    "pyaudiowpatch",
    "sherpa_onnx",
    "numpy",
    "scipy",
    "pywhispercpp",
    "comtypes",
    "webrtcvad",
    "psutil",
}


class TestDefensiveBoundary:
    def test_importing_reporter_pulls_no_audio_or_qt_stack(self):
        import os as _os
        import subprocess as sp

        # Fresh interpreter: prove the import graph itself is clean,
        # not merely that this pytest process happened to be clean.
        code = (
            "import sys, meetandread.reporter; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {"
            + ",".join(repr(m) for m in sorted(_HEAVY_TOP_LEVELS))
            + "}); "
            "print(heavy)"
        )
        env = dict(_os.environ)
        env["PYTHONPATH"] = str(
            Path(reporter.__file__).resolve().parent.parent
        )
        out = sp.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", (
            f"reporter imports heavy modules: {out.stdout.strip()}"
        )

    def test_reporter_module_has_no_qt_imports_in_source(self):
        src = Path(reporter.__file__).read_text(encoding="utf-8")
        for banned in ("PyQt6", "sounddevice", "pyaudiowpatch"):
            assert banned not in src


# ---------------------------------------------------------------------------
# Fresh capture directory per run (ADR 0005)
# ---------------------------------------------------------------------------


class TestFreshCaptureDir:
    def test_new_capture_dir_is_fresh_and_timestamped(self, tmp_path):
        d1 = new_capture_dir(tmp_path, now=datetime(2026, 9, 10, 10, 0, 0))
        assert d1.parent == tmp_path / CAPTURES_DIRNAME
        assert d1.exists()
        assert "20260910_100000" in d1.name

    def test_two_runs_in_same_second_get_distinct_dirs(self, tmp_path):
        now = datetime(2026, 9, 10, 10, 0, 0)
        d1 = new_capture_dir(tmp_path, now=now)
        d2 = new_capture_dir(tmp_path, now=now)
        assert d1 != d2
        assert d1.exists() and d2.exists()

    def test_dir_name_carries_run_kind(self, tmp_path):
        d = new_capture_dir(tmp_path, now=datetime(2026, 9, 10, 10, 0, 0))
        assert d.name.startswith("run-")


# ---------------------------------------------------------------------------
# Termination record
# ---------------------------------------------------------------------------


class TestTerminationRecord:
    def test_clean_stop_record_shape(self, tmp_path):
        rec = write_termination_record(
            tmp_path,
            outcome=RunOutcome.CLEAN_STOP,
            exit_code=0,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 2, 30),
            marker_present=True,
        )
        assert rec == tmp_path / TERMINATION_FILE_NAME
        data = read_termination_record(tmp_path)
        assert data["outcome"] == "clean_stop"
        assert data["exit_code"] == 0
        assert data["marker_present"] is True
        assert data["started_at"] == "2026-09-10T10:00:00"
        assert data["ended_at"] == "2026-09-10T10:02:30"

    def test_crash_record_carries_exit_code(self, tmp_path):
        write_termination_record(
            tmp_path,
            outcome=RunOutcome.CRASH,
            exit_code=0xC0000005,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 0, 5),
            marker_present=False,
        )
        data = read_termination_record(tmp_path)
        assert data["outcome"] == "crash"
        assert data["exit_code"] == 0xC0000005
        assert data["marker_present"] is False

    def test_user_stop_record(self, tmp_path):
        write_termination_record(
            tmp_path,
            outcome=RunOutcome.USER_STOP,
            exit_code=0,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 1, 0),
            marker_present=True,
        )
        assert read_termination_record(tmp_path)["outcome"] == "user_stop"

    def test_read_returns_none_when_absent(self, tmp_path):
        assert read_termination_record(tmp_path) is None

    def test_outcome_vocabulary_is_the_documented_set(self):
        assert {o.value for o in RunOutcome} == {
            "clean_stop",
            "user_stop",
            "crash",
        }


# ---------------------------------------------------------------------------
# Capture-directory state scan + standalone recovery
# ---------------------------------------------------------------------------


def _make_capture(
    base: Path,
    name: str,
    *,
    claim: bool = True,
    marker: bool = False,
    termination=None,
) -> Path:
    d = base / CAPTURES_DIRNAME / name
    d.mkdir(parents=True)
    if claim:
        (d / CLAIM_FILE_NAME).write_text(
            '{"started_at": "2026-09-10T10:00:00"}\n', encoding="utf-8"
        )
    if marker:
        (d / COMPLETION_MARKER_NAME).write_text(
            '{"finished_at": "2026-09-10T10:02:30"}\n', encoding="utf-8"
        )
    if termination is not None:
        outcome, exit_code = termination
        write_termination_record(
            d,
            outcome=outcome,
            exit_code=exit_code,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 2, 30),
            marker_present=marker,
        )
    return d


class TestCaptureStateScan:
    def test_incomplete_run_state(self, tmp_path):
        d = _make_capture(tmp_path, "run-a")  # claim only
        state = scan_capture_state(d)
        assert state.claim_present is True
        assert state.marker_present is False
        assert state.termination_present is False
        assert state.is_resumable is True
        assert state.state_name == "incomplete"

    def test_review_ready_state(self, tmp_path):
        d = _make_capture(
            tmp_path,
            "run-b",
            marker=False,
            termination=(RunOutcome.CRASH, 1),
        )
        state = scan_capture_state(d)
        assert state.termination_present is True
        assert state.is_resumable is True
        assert state.state_name == "review_ready"

    def test_completed_run_not_resumable(self, tmp_path):
        d = _make_capture(
            tmp_path,
            "run-c",
            marker=True,
            termination=(RunOutcome.CLEAN_STOP, 0),
        )
        state = scan_capture_state(d)
        assert state.is_resumable is False
        assert state.state_name == "done"

    def test_scan_tolerates_missing_dir(self, tmp_path):
        state = scan_capture_state(tmp_path / "nowhere")
        assert state.claim_present is False
        assert state.state_name == "absent"
        assert state.is_resumable is False


class TestRecoveryScan:
    def test_find_resumable_captures_across_states(self, tmp_path):
        _make_capture(tmp_path, "run-incomplete")
        _make_capture(
            tmp_path,
            "run-review",
            termination=(RunOutcome.CRASH, 1),
        )
        _make_capture(
            tmp_path,
            "run-done",
            marker=True,
            termination=(RunOutcome.CLEAN_STOP, 0),
        )
        resumable = find_resumable_captures(tmp_path)
        names = [p.name for p in resumable]
        assert "run-incomplete" in names
        assert "run-review" in names
        assert "run-done" not in names

    def test_find_skips_non_capture_dirs(self, tmp_path):
        (tmp_path / CAPTURES_DIRNAME / "not-a-run").mkdir(parents=True)
        (tmp_path / CAPTURES_DIRNAME / "stray.txt").write_text(
            "x", encoding="utf-8"
        )
        assert find_resumable_captures(tmp_path) == []

    def test_find_tolerates_missing_base(self, tmp_path):
        assert find_resumable_captures(tmp_path / "none") == []


# ---------------------------------------------------------------------------
# Description capture (carried forward for review/submission)
# ---------------------------------------------------------------------------


class TestDescriptionCapture:
    def test_write_and_read_description_roundtrip(self, tmp_path):
        text = "App froze when I opened the settings panel"
        path = write_description(tmp_path, text)
        assert path == tmp_path / DESCRIPTION_FILE_NAME
        assert read_description(tmp_path) == text

    def test_description_file_is_utf8_text(self, tmp_path):
        write_description(tmp_path, "déscription — with accents")
        raw = (tmp_path / DESCRIPTION_FILE_NAME).read_bytes()
        assert "déscription".encode("utf-8") in raw

    def test_read_returns_none_when_absent(self, tmp_path):
        assert read_description(tmp_path) is None


# ---------------------------------------------------------------------------
# End classification (pure function of exit status + marker)
# ---------------------------------------------------------------------------


class TestOutcomeClassification:
    def test_exit_zero_with_marker_is_clean_or_user_stop(self):
        # The clean/user distinction is the STOP SIGNAL, not the exit
        # status: both exit 0 with the marker written. classify() maps
        # the process facts; the wizard layer refines clean into user
        # when IT initiated the stop (SupervisedRun.final_outcome).
        assert (
            reporter._classify_exit(marker_present=True, exit_code=0)
            in (RunOutcome.CLEAN_STOP, RunOutcome.USER_STOP)
        )

    def test_nonzero_exit_is_crash(self):
        assert (
            reporter._classify_exit(marker_present=False, exit_code=1)
            == RunOutcome.CRASH
        )
        assert (
            reporter._classify_exit(marker_present=False, exit_code=0)
            == RunOutcome.CRASH
        )  # died without marker: crash-equivalent

    def test_negative_exit_is_crash_with_signal_encoding(self):
        # subprocess returncode is negative when killed by a signal
        # (POSIX shape; Windows taskkill surfaces as a code like 1 or
        # 0xC0000005 instead, but the reporter must handle both).
        assert (
            reporter._classify_exit(marker_present=False, exit_code=-1)
            == RunOutcome.CRASH
        )


class TestResumeCapture:
    def _make_incomplete(self, base: Path, name: str = "run-x") -> Path:
        d = base / "captures" / name
        d.mkdir(parents=True)
        (d / "capture_run.claim").write_text(
            '{"started_at": "2026-09-10T10:00:00"}\n', encoding="utf-8"
        )
        return d

    def test_resume_fills_crash_record_without_fabricating_times(
        self, tmp_path
    ):
        d = self._make_incomplete(tmp_path)
        record = reporter.resume_capture(d, data_base=tmp_path)
        assert record["outcome"] == "crash"
        assert record["marker_present"] is False
        # Unwitnessed end: the resume never invents an exit code or an
        # end time (the resume time is NOT the run's end).
        assert record["exit_code"] is None
        assert record["ended_at"] is None
        assert record["started_at"] == "2026-09-10T10:00:00"

    def test_resume_reconciles_staged_description(self, tmp_path):
        d = self._make_incomplete(tmp_path)
        staged = tmp_path / f"pending-{d.name}.description"
        staged.write_text(
            "The app froze when opening settings\n", encoding="utf-8"
        )
        reporter.resume_capture(d, data_base=tmp_path)
        assert read_description(d) == "The app froze when opening settings"
        assert not staged.exists()  # reconciled, not duplicated

    def test_resume_leaves_existing_record_untouched(self, tmp_path):
        d = self._make_incomplete(tmp_path)
        write_termination_record(
            d,
            outcome=RunOutcome.USER_STOP,
            exit_code=0,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 1, 0),
            marker_present=True,
        )
        record = reporter.resume_capture(d, data_base=tmp_path)
        assert record["outcome"] == "user_stop"

    def test_review_ready_dirs_not_reresumed(self, tmp_path):
        # review_ready = a termination record exists: resume_capture
        # returns it unchanged (the flow continues at review).
        d = self._make_incomplete(tmp_path)
        write_termination_record(
            d,
            outcome=RunOutcome.CRASH,
            exit_code=1,
            started_at=datetime(2026, 9, 10, 10, 0, 0),
            ended_at=datetime(2026, 9, 10, 10, 0, 5),
            marker_present=False,
        )
        record = reporter.resume_capture(d, data_base=tmp_path)
        assert record["exit_code"] == 1  # untouched, not refilled
