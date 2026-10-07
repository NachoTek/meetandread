"""Issue Reporter wizard — the console front-end over the supervisor
core (issues #107/#108/#109, docs/specs/issue-reporting.md,
ADR 0003/0004).

Stdlib-only like ``reporter.py`` (the defense discipline binds the
whole reporter program): no audio stack, no Qt. The wizard owns the
user-facing flow — describe → launch → reproduce → stop → review →
submit — end to end.

## Flow

1. **Recovery offer** (standalone recovery, amended ADR 0003): on
   startup, scan for resumable capture directories and offer to
   resume the newest; declining proceeds to a fresh run.
2. **Describe**: read the user's free-text description of the
   problem (carried into the capture directory, and forward to
   review/submission).
3. **Single-instance gate** (decided 2026-09-05): if the app is
   already running, tell the user to close it and wait until they
   have — capture must start from a clean single instance.
4. **Launch + reproduce**: create a FRESH capture directory
   (``new_capture_dir``, ADR 0005), write the description, launch the
   app in Issue Capture Mode, and supervise. The user reproduces the
   bug in the app; the wizard waits.
5. **Stop**: the user presses Enter in the wizard to stop the run —
   the reporter signals the app (CTRL_BREAK, the graceful user-stop
   path the capture-mode tests drive) and waits for the clean exit.
   If the app crashed on its own, the wizard reports the crash and
   CONTINUES — the crash itself is always reportable.
6. **Review** (issue #108): assemble the Diagnostics Bundle from the
   capture directory alone (redaction first — nothing unredacted is
   ever shown), and show the user "here is what will be sent". On
   any assembly/redaction failure the user is told submission is
   unavailable and NO artifact is written (fail-closed, ADR 0004).
7. **Submit** (issue #109, the review approval): after the review
   screen, ask to open the prefilled GitHub New Issue form in the
   user's default browser — the ONLY outbound action of the whole
   feature (ADR 0004) — and copy the bundle's full local path to
   the clipboard. The public prefilled body carries only the neutral
   bundle filename and attach-by-hand instructions; the local path
   stays on the clipboard and the screen.

## Crash handling (the reporter must be hard to kill)

``run_wizard`` wraps the flow in a try/except that catches everything,
prints the internal error, and STILL names the capture directory on
disk (when one exists) — an internal reporter error never loses the
run's artifacts. ``main`` installs ``sys.excepthook`` so an uncaught
PYTHON-level error still prints where the data lives before exiting;
a native crash cannot be caught this way, but nothing ever deletes
the capture directory, so the next reporter startup's recovery scan
finds it regardless.

## Testability

Every interactive seam is injectable: ``input_fn`` / ``print_fn``
(defaults: ``input`` / ``print``), ``app_command`` (the stub-app seam
the subprocess tests drive), and ``data_base`` (where capture
directories live — ``~/.meetandread-reporter`` by default; tests and
packaging (#111) point it elsewhere). The wizard is thin: every
decision lives in the pure ``reporter.py`` core and is tested there.
"""

import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from meetandread.reporter import (
    RunOutcome,
    SupervisedRun,
    app_is_running,
    find_resumable_captures,
    new_capture_dir,
    resume_capture,
    scan_capture_state,
    supervise_run,
    wait_until,
    write_description,
)

# Where the reporter keeps capture runs (and finds resumable ones).
# Kept OUTSIDE the app's Documents tree: the reporter must not depend
# on the app's storage configuration, and capture artifacts are
# excluded from the app's retention cleanup by living apart.
DEFAULT_DATA_BASE = Path.home() / ".meetandread-reporter"


def resolve_data_base(data_base: Optional[Path] = None) -> Path:
    """Resolve the reporter's data base: explicit argument, else the
    ``MAR_REPORTER_DATA_BASE`` env override (packaging #111 / tests),
    else the default under the user's home. The app's #110
    resume-on-launch offer resolves the SAME way — the app and the
    reporter must agree on where capture runs live."""
    if data_base is not None:
        return Path(data_base)
    env_base = os.environ.get("MAR_REPORTER_DATA_BASE")
    if env_base:
        return Path(env_base)
    return DEFAULT_DATA_BASE


# How long the single-instance gate waits between re-probes.
_SINGLE_INSTANCE_POLL_S = 1.0

InputFn = Callable[[str], str]
PrintFn = Callable[..., None]


def _ask_yes(input_fn: InputFn, prompt: str) -> bool:
    """A [Y/n] prompt: empty or yes-family answers are Yes (the
    wizard's recovery, submit — every step whose default is proceed).
    """
    answer = input_fn(prompt).strip().lower()
    return answer in ("", "y", "yes")


def _install_excepthook() -> None:
    """Reporter-level crash handling: an uncaught Python exception
    prints where the capture data lives before the process dies, so
    the artifacts are discoverable even after a reporter crash. (A
    NATIVE crash bypasses Python entirely; there is nothing to hook —
    the data is safe on disk either way, and the next startup's
    recovery scan finds it.)"""

    def hook(exc_type, exc, tb):
        sys.stderr.write(
            "\nIssue Reporter internal error — your capture data is "
            "safe on disk.\nRun the Issue Reporter again to resume "
            "the interrupted capture run.\n\n"
        )
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = hook


def offer_recovery(
    data_base: Path,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> Optional[Path]:
    """Startup recovery: offer EVERY resumable capture, newest first.

    The spec's recovery scan is plural ("offers to resume incomplete
    or review-ready capture directories"): each interrupted run is
    offered in turn — resuming one returns it immediately; declining
    moves to the next; declining all proceeds to a fresh run. Resuming
    an ``incomplete`` directory fills in its missing termination
    record (``resume_capture``, which also reconciles a staged
    description) so the flow can continue to review exactly as if the
    run had been supervised to its end.

    Runs whose submission story is already resolved (#110's
    ``submission_state.json`` — submitted or discarded) are never
    re-offered: their capture is done AND their submission is done,
    so there is nothing left to resume.
    """
    from meetandread.manual_submission import read_submission_state

    resumable = [
        d
        for d in find_resumable_captures(data_base)
        if read_submission_state(d) is None
    ]
    if not resumable:
        return None
    from meetandread.diagnostics_bundle import AssemblyFailure

    for candidate in resumable:
        state = scan_capture_state(candidate)
        print_fn(
            f"\nFound an interrupted capture run from a previous "
            f"session:\n  {candidate}\n  (state: {state.state_name})"
        )
        if _ask_yes(input_fn, "Resume it? [Y/n] "):
            resume_capture(candidate, data_base)
            print_fn(
                "Resumed. The captured diagnostics are ready for "
                "review."
            )
            result = review_capture(candidate, print_fn)
            # The resumed run's submission (#109) continues from the
            # re-reviewed bundle exactly like a fresh run's.
            if not isinstance(result, AssemblyFailure):
                submit_capture(
                    result, candidate, input_fn, print_fn
                )
            return candidate
        print_fn("Skipped.")
    print_fn("No more interrupted runs — starting a fresh flow.")
    return None


def gate_on_single_instance(
    input_fn: InputFn, print_fn: PrintFn
) -> None:
    """The single-instance gate (decided 2026-09-05): if the app is
    already running, tell the user to close it and wait until they
    have. The wizard does not proceed while the mutex is held."""
    if not app_is_running():
        return
    print_fn(
        "\nmeetandread is already running. Please close the open "
        "meetandread window (and its tray icon) first — the capture "
        "must start from a clean single instance."
    )
    while not wait_until(
        lambda: not app_is_running(),
        timeout_s=_SINGLE_INSTANCE_POLL_S,
        interval_s=_SINGLE_INSTANCE_POLL_S,
    ):
        print_fn("Still running — waiting for it to close...")
    print_fn("Closed. Continuing.")


def ask_description(input_fn: InputFn, print_fn: PrintFn) -> str:
    """The describe step: the user's problem description in their own
    words. Non-empty (re-asked while empty) — the human context is
    the one thing the machine diagnostics cannot supply. (Named
    ``ask_`` to distinguish from ``reporter.read_description``, the
    capture-directory reader.)"""
    print_fn(
        "\nFirst, describe the problem in your own words.\n"
        "(What did you do, what did you expect, and what happened?)"
    )
    while True:
        text = input_fn("Description> ").strip()
        if text:
            return text
        print_fn("Please describe the problem (a few words are enough).")


def _signal_user_stop(proc) -> None:
    """Ask the app to stop gracefully: CTRL_BREAK on Windows (the
    same signal the capture-mode clean-exit tests drive, and the
    Issue Reporter's designated user-stop signal), SIGINT elsewhere.
    Idempotent: safe to call repeatedly while the app winds down.
    """
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        # Process died between the poll and the signal: nothing to
        # stop — supervise_run will observe the exit.
        pass


def _graceful_stop(
    proc, deadline_s: float = 30.0, interval_s: float = 1.0
) -> bool:
    """Signal the user stop and give the app time to exit cleanly.

    Returns True when a stop signal was delivered to a LIVE app (the
    run's stop was user-initiated); False when the app had already
    exited on its own (its own outcome — clean or crash — stands).

    The stop must be graceful (the app writes its completion marker
    on the clean-exit path): send CTRL_BREAK, wait, re-send while the
    process lives — up to the deadline. A hard kill is NEVER used:
    the wizard's stop must not fabricate a crash out of a healthy
    app. If the app outlives the deadline it is left running;
    ``supervise_run`` keeps waiting (the user can still close it by
    hand, and a genuinely hung app is exactly the bug being
    reported).

    Before the FIRST signal, give the app a short grace window to
    exit on its own: the user pressing "Enter to finish" right after
    the app crashed on its own is the core scenario — the recorded
    exit must be the app's own crash code, not a stop-signal
    artifact (CTRL_BREAK to a process without a handler is fatal).
    """
    import time as _time

    # Grace window: an app that died during reproduction exits here,
    # keeping its genuine exit code.
    try:
        proc.wait(timeout=2.0)
        return False
    except subprocess.TimeoutExpired:
        pass

    signaled = False
    deadline = _time.monotonic() + deadline_s
    while _time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        _signal_user_stop(proc)
        signaled = True
        try:
            proc.wait(timeout=interval_s)
            break
        except subprocess.TimeoutExpired:
            continue
    return signaled


def run_wizard(
    data_base: Optional[Path] = None,
    app_command: Optional[List[str]] = None,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> Optional[SupervisedRun]:
    """Drive the full wizard flow; returns the supervised run (when
    one was launched), or None when only a recovery resume happened.

    Every failure path prints and returns/raises cleanly — the
    capture directory on disk is the crash-safe source of truth, and
    the next reporter startup can always resume from it.
    """
    if data_base is None:
        data_base = resolve_data_base()
    data_base = Path(data_base)
    data_base.mkdir(parents=True, exist_ok=True)

    print_fn(
        "=" * 60
        + "\nmeetandread Issue Reporter\n"
        + "Report a bug by reproducing it under diagnostics capture.\n"
        + "=" * 60
    )

    # 1. Standalone recovery (amended ADR 0003): reporter startup is
    #    the designated recovery path.
    resumed = offer_recovery(data_base, input_fn, print_fn)
    if resumed is not None:
        return None

    # 2. Describe (carried forward for review/submission, #108/#109).
    description = ask_description(input_fn, print_fn)

    # 3. Single-instance gate: capture starts from a clean instance.
    gate_on_single_instance(input_fn, print_fn)

    # 4. Launch + reproduce: a FRESH capture directory (ADR 0005) —
    #    handed to the app EMPTY: the capture-directory contract
    #    rejects a non-empty directory at entry (exit 2), so the
    #    description is staged OUTSIDE (in the data base) during the
    #    run and copied into the capture directory at run end, when
    #    no entry check can race it.
    capture_dir = new_capture_dir(data_base)
    staged_desc = data_base / f"pending-{capture_dir.name}.description"
    staged_desc.write_text(description + "\n", encoding="utf-8")
    print_fn(
        f"\nCapture run: {capture_dir}\n"
        "Starting meetandread in Issue Capture Mode..."
    )
    from meetandread.reporter import launch_app

    proc = launch_app(capture_dir, app_command=app_command)
    started_at = datetime.now()

    print_fn(
        "\nmeetandread is starting with full diagnostics.\n"
        "NOW REPRODUCE THE PROBLEM in the meetandread window.\n"
        "When you are done (or the app has crashed/closed), come back\n"
        "here and press Enter to finish the run."
    )
    input_fn("Press Enter when you have finished reproducing... ")
    print_fn("Stopping the capture run...")

    # The graceful stop: signal CTRL_BREAK and give the app time to
    # wind down and write its completion marker. Blocking by design —
    # the marker is what separates a user stop from a crash. A stop
    # signal actually delivered to a LIVE app makes this run's stop
    # user-initiated (refining a clean exit into ``user_stop``); an
    # app that already exited on its own keeps its own outcome.
    user_initiated = _graceful_stop(proc)

    run = supervise_run(
        proc,
        capture_dir,
        started_at=started_at,
        stop_signal=_signal_user_stop,
        user_initiated_stop=user_initiated,
    )

    # 5. End: the staged description joins the capture directory now
    #    that the run is over (no entry check can race it), and the
    #    staged copy is removed. Best-effort: the run's diagnostics
    #    exist regardless.
    try:
        write_description(capture_dir, description)
        staged_desc.unlink(missing_ok=True)
    except OSError:
        print_fn("(Could not save the description alongside the run.)")

    # 6. Report the outcome; review continues from the directory
    #    alone (#108; submission is #109).
    if run.final_outcome() == RunOutcome.CRASH:
        print_fn(
            f"\nmeetandread exited unexpectedly (code {run.exit_code}).\n"
            "The crash WAS captured — that is the most valuable part\n"
            "of the report."
        )
    else:
        print_fn("\nCapture run finished cleanly.")

    # 7. Review screen (issue #108): assemble the Diagnostics Bundle
    #    from the directory alone (any state), show the user exactly
    #    what will be sent. Fail-closed: on any assembly/redaction
    #    failure NO artifact is written and the user is told
    #    submission is unavailable — never a raw fallback.
    result = review_capture(
        capture_dir, print_fn, identifiers=None
    )
    # 8. Submit (issue #109): with the review passed and the bundle
    #    on disk, offer the manual submission (browser + clipboard).
    #    Unreachable on the fail-closed path (no artifact → no
    #    submission offer).
    submit_capture(result, capture_dir, input_fn, print_fn)
    return run


def review_capture(
    capture_dir: Path,
    print_fn: PrintFn,
    identifiers=None,
):
    """Assemble the bundle and render the review screen.

    The #108 seam in the wizard: the pure core
    (``diagnostics_bundle``) does everything; this layer only prints.
    On success the review screen shows the REDACTED artifact itself
    ("here's what will be sent") — nothing unredacted is ever
    displayed. On failure the user is told submission is unavailable
    and why (component named, no leaked content in the message).
    Returns the assembly result for tests.
    """
    from meetandread.diagnostics_bundle import (
        AssemblyFailure,
        create_reviewable_bundle,
        default_identifiers,
        render_review,
    )

    ids = identifiers if identifiers is not None else (
        default_identifiers()
    )
    print_fn("\nAssembling the Diagnostics Bundle...")
    result = create_reviewable_bundle(capture_dir, identifiers=ids)
    if isinstance(result, AssemblyFailure):
        print_fn(
            "\nSubmission is UNAVAILABLE for this capture run.\n"
            f"Reason: {result.reason}"
            + (
                f" (component: {result.component})"
                if result.component
                else ""
            )
            + f"\n{result.detail}\n"
            "No report file was created — nothing unreviewed or\n"
            "unredacted will leave your machine. The capture data\n"
            "remains on disk for a manual inspection."
        )
        return result
    print_fn(render_review(result))
    print_fn(
        f"\nDiagnostics Bundle (redacted, ready to submit):\n"
        f"  {result.path}"
    )
    return result


def submit_capture(
    review_result,
    capture_dir: Path,
    input_fn: InputFn,
    print_fn: PrintFn,
    identifiers=None,
) -> "object":
    """The #109 Manual Submission step: after the review screen (the
    user's approval of "here is what will be sent"), copy the
    bundle's full local path to the clipboard and offer to open the
    prefilled GitHub New Issue form in the default browser.

    The browser open is the single outbound action of the whole
    reporting feature (ADR 0004) — and it happens only on the
    user's Yes. The clipboard copy (local, private) runs for every
    answered prompt so the path is on the clipboard whether or not
    the browser opens: attaching the bundle by hand must remain one
    paste away on every branch. Returns the submission draft, the
    fail-closed ``SubmissionUnavailable``, or None when review
    itself failed closed (for tests).
    """
    from meetandread import manual_submission as ms
    from meetandread.diagnostics_bundle import AssemblyFailure

    if isinstance(review_result, AssemblyFailure):
        # Review failed closed: there is nothing submittable. The
        # review step already told the user why; nothing opens, the
        # clipboard stays untouched.
        return None
    draft = ms.build_submission(
        capture_dir,
        identifiers=identifiers,
    )
    if isinstance(draft, ms.SubmissionUnavailable):
        # The artifact vanished between review and submit (or the
        # review result was not from this directory) — fail closed,
        # same verdict shape as the review step's.
        print_fn(
            "\nSubmission is UNAVAILABLE: the reviewed Diagnostics\n"
            f"Bundle artifact is missing. {draft.detail}"
        )
        return draft
    print_fn(
        "\nReady to file the issue.\n"
        "The browser will open on the GitHub New Issue form with\n"
        "the title and description filled in — filed from YOUR\n"
        "GitHub account (which also subscribes you to answers)."
    )
    proceed = _ask_yes(input_fn, "Open the issue form now? [Y/n] ")
    # The path is on the clipboard on EVERY branch from here (the
    # clipboard is local and private — unlike the browser open, it
    # needs no approval): attach-by-hand stays one paste away.
    if ms.copy_to_clipboard(draft.clipboard_text):
        print_fn(f"Bundle path (on your clipboard):\n"
                 f"  {draft.clipboard_text}")
    else:
        print_fn(
            "(could not reach the clipboard — copy this path by\n"
            f" hand: {draft.clipboard_text})"
        )
    if not proceed:
        print_fn(
            "Browser not opened — nothing has left your machine.\n"
            f"The bundle stays ready on disk:\n  {draft.bundle_path}"
        )
        return draft
    if not ms.open_new_issue_form(draft.url):
        print_fn(
            "\nCould not open a browser. Copy this address by hand:\n"
            f"  {draft.url}"
        )
    else:
        # The form opened (#110): the run's submission story is
        # resolved — record it so no future offer (reporter recovery
        # or the app's next-launch resume) re-offers a filed report.
        ms.write_submission_state(
            capture_dir, ms.SubmissionState.SUBMITTED
        )
        print_fn(
            "\nThe New Issue form is open in your browser.\n"
            "IMPORTANT: GitHub cannot attach files automatically —\n"
            "ATTACH THE BUNDLE FILE BY HAND before submitting:\n"
            f"  file to attach: {ms.BUNDLE_FILE_NAME}\n"
            "  (drag it into the GitHub form; its full path is on\n"
            "  your clipboard and shown above)"
        )
    return draft


def main(argv: Optional[List[str]] = None) -> int:
    """Reporter entry point (console script / ``python -m``)."""
    _install_excepthook()
    try:
        run_wizard()
    except KeyboardInterrupt:
        sys.stderr.write(
            "\nCancelled. If a capture run was in flight, its data is "
            "safe; run the Issue Reporter again to resume it.\n"
        )
        return 130
    except Exception:
        # The reporter's own crash handling: the capture directory
        # remains on disk and the next startup's recovery scan finds
        # it — never lose the run to a reporter bug.
        sys.stderr.write(
            "\nThe Issue Reporter hit an internal error. Any capture "
            "data is safe on disk; run the Issue Reporter again to "
            "resume.\n"
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
