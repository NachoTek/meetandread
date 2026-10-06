"""Issue Reporter wizard tests (issue #107, docs/specs/issue-reporting.md).

Fast-lane (ADR 0001): the wizard's interactive seams are injectable
(``input_fn`` / ``print_fn`` / ``app_command`` / ``data_base``), so
the full flow is testable with a stub app process and scripted input
— no real Qt app, no console.

Covers:

- The wizard walks describe → launch → reproduce → stop with a stub
  app, writing the description into the capture directory and
  producing a termination record.
- The user stop is classified ``user_stop`` (exit 0 + marker, stop
  was user-initiated).
- A crashed stub app (nonzero exit, no marker) does NOT fail the
  wizard: it completes, records ``crash``, and the capture directory
  survives.
- Startup recovery: a resumable capture directory is offered and
  resuming it skips the fresh-run flow; declining proceeds to a fresh
  run.
- The single-instance gate blocks while the mutex probe says running
  (probed via monkeypatched ``app_is_running``).
- ``main``'s internal-error path returns 1 and never raises.
"""

import sys
from pathlib import Path

import pytest

import meetandread.reporter_wizard as wizard
from meetandread.reporter import (
    RunOutcome,
    read_termination_record,
)


class ScriptedInput:
    """input_fn that replays scripted answers, recording prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(
                f"unexpected extra input prompt: {prompt!r} "
                f"(prompts so far: {self.prompts})"
            )
        return self.answers.pop(0)


class Lines:
    """print_fn collecting output lines."""

    def __init__(self):
        self.lines = []

    def __call__(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def text(self) -> str:
        return "\n".join(self.lines)


# Stub app: a well-behaved capture run that lives until stopped. It
# installs a SIGINT/SIGBREAK handler (as the real app does — the
# graceful user-stop path), writes the completion marker when the
# signal arrives, and exits 0 — so the wizard's stop IS
# user-initiated and the run classifies as user_stop.
STUB_APP_CLEAN = (
    "import signal, sys, time;\n"
    "from pathlib import Path;\n"
    "d = Path(sys.argv[2]);\n"
    "def _stop(sig, frame):\n"
    "    (d / 'capture_complete.marker').write_text("
    "'{\"finished_at\": \"2026-09-10T10:02:30\"}\\n', "
    "encoding='utf-8');\n"
    "    sys.exit(0)\n"
    "signal.signal(signal.SIGINT, _stop);\n"
    "signal.signal(signal.SIGBREAK, _stop);\n"
    "while True:\n"
    "    time.sleep(0.2)\n"
)

# Stub app: crashes immediately (no marker). It exits on its own
# before any stop signal, so the wizard observes the genuine exit
# code 3 (0xC0000142-style signal artifacts would mean the stop path
# fabricated the outcome).
STUB_APP_CRASH = "import sys; sys.exit(3)"


def _stub_command(code: str):
    return [sys.executable, "-c", code]


class TestWizardHappyPath:
    def test_user_stop_flow_records_user_stop_and_description(
        self, tmp_path
    ):
        inp = ScriptedInput(
            [
                "",  # empty description → re-ask
                "Settings panel froze the app",
                "",  # Press Enter to finish reproducing
                "n",  # decline the submit step (#109)
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert run.outcome == RunOutcome.CLEAN_STOP
        # The stop was user-initiated (the wizard signaled the live
        # stub): the FINAL outcome — and the durable record — say
        # user_stop.
        assert run.final_outcome() == RunOutcome.USER_STOP
        record = read_termination_record(run.capture_dir)
        assert record is not None
        assert record["outcome"] == "user_stop"
        assert record["marker_present"] is True
        # Description captured into the capture directory.
        desc = run.capture_dir / "description.txt"
        assert desc.read_text(encoding="utf-8").strip() == (
            "Settings panel froze the app"
        )
        # The flow text walked the wizard stages.
        text = out.text()
        assert "describe the problem" in text.lower()
        assert "reproduce the problem" in text.lower()
        assert "what will be sent" in text  # the #108 review step

    def test_crashed_app_wizard_completes_and_records_crash(
        self, tmp_path
    ):
        inp = ScriptedInput(
            [
                "It crashes on startup",
                "",  # finish reproducing
                "n",  # decline the submit step (#109)
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CRASH),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        # THE survival assertion: the crashed app did not fail the
        # wizard; the run was classified and recorded.
        assert run.outcome == RunOutcome.CRASH
        assert run.exit_code == 3
        record = read_termination_record(run.capture_dir)
        assert record["outcome"] == "crash"
        assert record["exit_code"] == 3
        assert "exited unexpectedly" in out.text()
        # The capture directory (with description) survives.
        assert (run.capture_dir / "description.txt").exists()


class TestWizardRecovery:
    def _make_incomplete(self, base: Path) -> Path:
        captures = base / "captures"
        captures.mkdir(parents=True)
        d = captures / "run-old"
        d.mkdir()
        (d / "capture_run.claim").write_text(
            '{"started_at": "2026-09-09T09:00:00"}\n', encoding="utf-8"
        )
        return d

    def test_startup_offers_resume_and_resuming_skips_fresh_run(
        self, tmp_path
    ):
        d = self._make_incomplete(tmp_path)
        inp = ScriptedInput(["y", "n"])  # resume; decline submit
        out = Lines()
        result = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert result is None  # no fresh run was launched
        # The incomplete run's termination record was filled in.
        record = read_termination_record(d)
        assert record is not None
        assert record["outcome"] == "crash"  # unwitnessed end
        assert "Resumed" in out.text()

    def test_declining_resume_proceeds_to_fresh_run(self, tmp_path):
        self._make_incomplete(tmp_path)
        inp = ScriptedInput(
            [
                "n",  # decline resume
                "Fresh description",
                "",  # finish reproducing
                "n",  # decline the submit step (#109)
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert "Skipped" in out.text()
        assert read_termination_record(run.capture_dir) is not None

    def test_no_resumable_runs_straight_to_describe(self, tmp_path):
        inp = ScriptedInput(["A problem", "", "n"])
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert "Found an interrupted" not in out.text()


class TestWizardSingleInstanceGate:
    def test_gate_blocks_until_app_closed(self, tmp_path, monkeypatch):
        states = iter([True, False])
        monkeypatch.setattr(
            wizard, "app_is_running", lambda: next(states, False)
        )
        monkeypatch.setattr(wizard, "_SINGLE_INSTANCE_POLL_S", 0.01)
        inp = ScriptedInput(["desc", "", "n"])
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        text = out.text()
        assert "already running" in text
        assert "Closed. Continuing." in text

    def test_no_running_app_skips_gate_silently(self, tmp_path):
        inp = ScriptedInput(["desc", "", "n"])
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert "already running" not in out.text()


class TestMainCrashHandling:
    def test_main_returns_1_on_internal_error(self, monkeypatch, capsys):
        def boom(**kwargs):
            raise RuntimeError("reporter internal bug")

        monkeypatch.setattr(wizard, "run_wizard", boom)
        code = wizard.main([])
        assert code == 1
        err = capsys.readouterr().err
        assert "safe on disk" in err
        assert "RuntimeError" in err

    def test_main_keyboard_interrupt_returns_130(self, monkeypatch):
        def interrupted(**kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(wizard, "run_wizard", interrupted)
        assert wizard.main([]) == 130

    def test_main_happy_path_returns_0(self, monkeypatch):
        monkeypatch.setattr(wizard, "run_wizard", lambda **k: None)
        assert wizard.main([]) == 0


class TestReviewStep:
    """The #108 review step: after the run (or a resume), the wizard
    assembles the Diagnostics Bundle and shows the user exactly what
    will be sent — the redacted artifact itself, or the fail-closed
    "submission unavailable" verdict."""

    def test_review_step_shows_bundle_and_writes_artifact(
        self, tmp_path
    ):
        inp = ScriptedInput(["desc", "", "n"])  # n: decline submit
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        text = out.text()
        assert "what will be sent" in text
        assert "NO audio and NO transcript" in text
        assert "diagnostics_bundle.txt" in text
        bundle = run.capture_dir / "diagnostics_bundle.txt"
        assert bundle.exists()
        written = bundle.read_text(encoding="utf-8")
        # The review shows the redacted artifact itself.
        assert "== environment ==" in written
        assert "clean_stop" in written or "user_stop" in written

    def test_review_step_on_crash_still_reviews(self, tmp_path):
        inp = ScriptedInput(["desc", "", "n"])  # n: decline submit
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CRASH),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert run.outcome == RunOutcome.CRASH
        assert (run.capture_dir / "diagnostics_bundle.txt").exists()
        assert "what will be sent" in out.text()
        assert "crash" in (
            run.capture_dir / "diagnostics_bundle.txt"
        ).read_text(encoding="utf-8")

    def test_review_step_fail_closed_blocks_submission(self, tmp_path):
        # Seed a poisoned canary leak into a capture log that appears
        # during the run: assembly must fail closed, the wizard tells
        # the user submission is unavailable, and NO artifact exists.
        import json as _json

        import meetandread.diagnostics_bundle
        from meetandread.transcript_canary import canary_ngrams

        secret = "Sebastopol canary spoken words recorded"
        # No third answer needed: the fail-closed review never asks.
        inp = ScriptedInput(["desc", ""])
        out = Lines()

        real_create = (
            meetandread.diagnostics_bundle.create_reviewable_bundle
        )

        def leaky_create(capture_dir, identifiers=None):
            log = capture_dir / "meetandread_capture_20260910_100000.log"
            if not log.exists():
                log.write_text(
                    f"leak: {secret}\n", encoding="utf-8"
                )
                body = "".join(
                    _json.dumps({"gram": g}) + "\n"
                    for g in sorted(canary_ngrams(secret))
                )
                (capture_dir / "transcript_canary.jsonl").write_text(
                    body, encoding="utf-8"
                )
            return real_create(capture_dir, identifiers=identifiers)

        meetandread.diagnostics_bundle.create_reviewable_bundle = (
            leaky_create
        )
        try:
            run = wizard.run_wizard(
                data_base=tmp_path,
                app_command=_stub_command(STUB_APP_CLEAN),
                input_fn=inp,
                print_fn=out,
            )
        finally:
            meetandread.diagnostics_bundle.create_reviewable_bundle = (
                real_create
            )
        assert run is not None
        assert not (run.capture_dir / "diagnostics_bundle.txt").exists()
        text = out.text()
        assert "UNAVAILABLE" in text
        assert "canary_leak" in text

    def test_resume_path_reviews_too(self, tmp_path):
        captures = tmp_path / "captures"
        d = captures / "run-old"
        d.mkdir(parents=True)
        (d / "capture_run.claim").write_text(
            '{"started_at": "2026-09-09T09:00:00"}\n', encoding="utf-8"
        )
        inp = ScriptedInput(["y", "n"])  # resume; decline submit
        out = Lines()
        result = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert result is None
        assert "what will be sent" in out.text()
        assert (d / "diagnostics_bundle.txt").exists()
