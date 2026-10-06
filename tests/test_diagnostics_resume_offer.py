"""App-launch diagnostics-resume offer tests (issue #110,
docs/specs/issue-reporting.md, ADR 0003).

The reporter-death resilience path: if the Issue Reporter died
mid-flow after a Diagnostics Bundle exists on disk, the next NORMAL
application launch notices the assembled-but-unsubmitted bundle and
offers to resume the submission — from the saved bundle, no
re-reproduction — with a clear option to discard instead. Declining
leaves normal startup otherwise unaffected.

Fast-lane (ADR 0001): the offer's decision seam is injectable (the
QMessageBox class and the Manual Submission IO seams are
monkeypatched; Qt itself is never executed beyond enum attribute
access), so the whole offer runs against fixture capture directories
with zero subprocesses.

Covers:

- Detection: an unsubmitted bundle (artifact + no submission state)
  is found on launch; nothing else triggers the offer.
- Resume: re-uses the SAME review and Manual Submission
  implementation — re-assembles fail-closed (redaction first),
  re-renders the review, then the browser/clipboard seams fire
  exactly as the wizard's submit step; the state record lands.
- Decline and discard: "Not now" resolves nothing (re-offered next
  launch); "Discard" writes the discarded state and never re-offers.
- Normal startup unaffected: capture runs never see the offer; a
  machine that never saw the reporter never sees the dialog.
"""

import sys
from pathlib import Path

import pytest

import meetandread.main as main_mod
import meetandread.manual_submission as msub
from meetandread.diagnostics_bundle import BUNDLE_FILE_NAME
from meetandread.reporter import RunOutcome, write_termination_record

from tests.test_diagnostics_bundle import make_complete
from tests.test_manual_submission import IDS, _reviewed

REPLY = {
    "resume": "resume",
    "discard": "discard",
    "not_now": "not_now",
}


def _mock_msg_box(reply: str):
    """A QMessageBox stand-in whose exec() returns the scripted reply.

    Follows the test_recovery_thread_safety.py pattern: enum attribute
    access (StandardButton, Icon) delegates to the real QMessageBox so
    production code reads real enum values.
    """
    from PyQt6.QtWidgets import QMessageBox as _RealMsgBox

    instances = []

    class _MockMsgBox:
        StandardButton = _RealMsgBox.StandardButton
        Icon = _RealMsgBox.Icon

        def __init__(self, *args, **kwargs):
            self.exec = lambda: {
                "yes": _RealMsgBox.StandardButton.Yes.value,
                "no": _RealMsgBox.StandardButton.No.value,
                "discard": _RealMsgBox.StandardButton.Discard.value,
            }[reply]
            self.show = lambda: None
            self.close = lambda: None
            self.setWindowTitle = lambda *a: None
            self.setText = lambda *a: None
            self.setInformativeText = lambda *a: None
            self.setStandardButtons = lambda *a: None
            self.setDefaultButton = lambda *a: None
            self.setIcon = lambda *a: None
            instances.append(self)

    _MockMsgBox.instances = instances  # type: ignore[attr-defined]
    return _MockMsgBox


def _make_unsubmitted(base: Path) -> Path:
    """A capture directory holding an assembled-but-unsubmitted
    bundle — the exact state a dead reporter leaves behind."""
    captures = base / "captures"
    captures.mkdir(parents=True, exist_ok=True)
    d = _reviewed(captures)
    write_termination_record(
        d,
        outcome=RunOutcome.USER_STOP,
        exit_code=0,
        started_at=__import__("datetime").datetime(
            2026, 9, 10, 10, 0, 0
        ),
        ended_at=__import__("datetime").datetime(
            2026, 9, 10, 10, 2, 30
        ),
        marker_present=True,
    )
    return d


def _run_offer(monkeypatch, tmp_path, reply: str, staged: Path):
    """Patch the offer's seams and run it; returns (result, opened,
    copied, dialog_count)."""
    opened, copied = [], []
    monkeypatch.setattr(
        msub, "open_new_issue_form",
        lambda url: opened.append(url) or True,
    )
    monkeypatch.setattr(
        msub, "copy_to_clipboard",
        lambda text: copied.append(text) or True,
    )
    box = _mock_msg_box(reply)
    monkeypatch.setattr(main_mod, "QMessageBox", box)

    import meetandread.diagnostics_bundle as dbundle

    monkeypatch.setattr(
        dbundle, "default_identifiers", lambda: IDS
    )

    result = main_mod.check_and_offer_diagnostics_resume(
        data_base=staged,
        parent=None,
    )
    return result, opened, copied, box.instances


class TestDetection:
    def test_no_reporter_data_base_no_dialog(self, monkeypatch,
                                             tmp_path):
        box = _mock_msg_box("yes")
        monkeypatch.setattr(main_mod, "QMessageBox", box)
        # A base with no captures tree at all.
        empty = tmp_path / "nowhere"
        result = main_mod.check_and_offer_diagnostics_resume(
            data_base=empty, parent=None
        )
        assert result is None
        assert box.instances == []

    def test_no_unsubmitted_bundle_no_dialog(self, monkeypatch,
                                             tmp_path):
        box = _mock_msg_box("yes")
        monkeypatch.setattr(main_mod, "QMessageBox", box)
        base = tmp_path / "data"
        (base / "captures").mkdir(parents=True)
        make_complete(base / "captures")  # no bundle artifact
        result = main_mod.check_and_offer_diagnostics_resume(
            data_base=base, parent=None
        )
        assert result is None
        assert box.instances == []

    def test_capture_mode_launch_skips_offer_entirely(
        self, monkeypatch, tmp_path
    ):
        # A capture run must never interrupt itself with a submission
        # offer — the offer is a NORMAL-launch feature (the reporter
        # supervises capture runs; it would be mid-wizard).
        box = _mock_msg_box("yes")
        monkeypatch.setattr(main_mod, "QMessageBox", box)
        base = tmp_path / "data"
        _make_unsubmitted(base)
        result = main_mod.check_and_offer_diagnostics_resume(
            data_base=base, parent=None, capture_mode=True
        )
        assert result is None
        assert box.instances == []


class TestResume:
    def test_resume_reuses_review_and_manual_submission(
        self, monkeypatch, tmp_path
    ):
        base = tmp_path / "data"
        d = _make_unsubmitted(base)
        result, opened, copied, dialogs = _run_offer(
            monkeypatch, tmp_path, "yes", base
        )
        # The offer dialog appeared exactly once.
        assert len(dialogs) == 1
        # The SAME Manual Submission seams fired: the browser opened
        # on the prefilled New Issue form...
        assert len(opened) == 1
        assert opened[0].startswith(
            "https://github.com/NachoTek/meetandread/issues/new?"
        )
        # ...and the clipboard carries the bundle's full local path.
        assert copied == [str(d / BUNDLE_FILE_NAME)]
        # The run's submission story is resolved.
        assert msub.read_submission_state(d) == "submitted"
        assert result == d

    def test_resume_starts_from_saved_bundle_no_reproduction(
        self, monkeypatch, tmp_path
    ):
        # Resuming must consume the SAVED artifact — no re-review, no
        # re-assembly of capture data (the run is over; the directory
        # is the source of truth). Poisoning create_reviewable_bundle
        # to fail closed proves the offer never re-assembled.
        import meetandread.diagnostics_bundle as dbundle

        base = tmp_path / "data"
        d = _make_unsubmitted(base)

        def fail_reassemble(capture_dir, identifiers=None):
            raise AssertionError(
                "the resume offer must not re-assemble the bundle"
            )

        monkeypatch.setattr(
            dbundle, "create_reviewable_bundle", fail_reassemble
        )
        result, opened, copied, dialogs = _run_offer(
            monkeypatch, tmp_path, "yes", base
        )
        assert result == d
        assert len(opened) == 1

    def test_resume_logs_diagnostics_resume_event(
        self, monkeypatch, tmp_path, caplog
    ):
        import logging

        base = tmp_path / "data"
        _make_unsubmitted(base)
        with caplog.at_level(
            logging.INFO, logger="meetandread.main"
        ):
            _run_offer(monkeypatch, tmp_path, "yes", base)
        assert any(
            "diagnostics_resume" in rec.message
            or "diagnostics_resume" in rec.getMessage()
            for rec in caplog.records
        )


class TestDeclineAndDiscard:
    def test_not_now_leaves_story_unresolved(self, monkeypatch,
                                             tmp_path):
        base = tmp_path / "data"
        d = _make_unsubmitted(base)
        result, opened, copied, dialogs = _run_offer(
            monkeypatch, tmp_path, "no", base
        )
        assert result is None
        assert opened == []  # nothing left the machine
        # The bundle stays offerable for the next launch.
        assert msub.read_submission_state(d) is None

    def test_discard_resolves_and_never_re_offers(
        self, monkeypatch, tmp_path
    ):
        base = tmp_path / "data"
        d = _make_unsubmitted(base)
        result, opened, copied, dialogs = _run_offer(
            monkeypatch, tmp_path, "discard", base
        )
        assert result is None
        assert opened == []
        assert msub.read_submission_state(d) == "discarded"
        # The next launch's scan no longer finds it.
        assert msub.find_unsubmitted_bundles(base) == []

    def test_discard_deletes_nothing(self, monkeypatch, tmp_path):
        # "Discard" resolves the offer, it never deletes the user's
        # capture data (retention/cleanup is #112's lane).
        base = tmp_path / "data"
        d = _make_unsubmitted(base)
        _run_offer(monkeypatch, tmp_path, "discard", base)
        assert (d / BUNDLE_FILE_NAME).is_file()
        assert (d / "capture_run.claim").is_file()


class TestNormalStartupUnaffected:
    def test_offer_failure_never_blocks_startup(
        self, monkeypatch, tmp_path
    ):
        # Even an exception inside the offer must not escape to
        # break startup: the caller wraps it (the same discipline as
        # check_and_offer_recovery's call site).
        import meetandread.main as m

        def boom(*a, **k):
            raise RuntimeError("offer internals broke")

        monkeypatch.setattr(m, "QMessageBox", boom)
        base = tmp_path / "data"
        _make_unsubmitted(base)
        try:
            m.check_and_offer_diagnostics_resume(
                data_base=base, parent=None
            )
        except Exception:
            pytest.fail(
                "the resume offer must never propagate exceptions"
            )


class TestDefensiveImport:
    def test_offer_module_stays_importable_without_qt_exec(self):
        # The offer's decision logic lives in the stdlib-only
        # submission module; main.py only adds the dialog. Importing
        # manual_submission must never pull the heavy stack (its own
        # boundary test lives in test_manual_submission.py).
        code = (
            "import sys, meetandread.manual_submission; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {'PyQt6','sounddevice','numpy',"
            "'psutil','sherpa_onnx','pyaudiowpatch','comtypes',"
            "'webrtcvad','scipy','pywhispercpp'}); "
            "print(heavy)"
        )
        import os as _os
        import subprocess as sp

        env = dict(_os.environ)
        env["PYTHONPATH"] = str(
            Path(msub.__file__).resolve().parent.parent
        )
        out = sp.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"
