"""Lightweight executable bootstrap for ``python -m meetandread``, the
pip-installed ``meetandread`` console script, and the frozen exe
(``meetandread.spec`` analyzes THIS file as the entry).

Fix round 1 (PR #121 review, finding 1): capture mode must own the
process from its FIRST moment — including startup failures. Parsing
the flag inside ``main()`` was too late: importing
``meetandread.main`` pulls in PyQt/widgets/native audio at import
time, so an import-time or native-startup failure produced NO capture
log, and lock-path diagnostics (acquired before logging) were equally
lost.

This module therefore imports ONLY argparse/pathlib/stdlib plus
``meetandread.capture_mode`` (pure stdlib itself). It parses
``--issue-capture``, configures capture logging if flagged — BEFORE
any app subsystem import and before the single-instance lock — then
imports ``meetandread.main`` and delegates to ``main()``. If that
import OR THE ``main()`` CALL raises while capture logging is
configured, the exception (with traceback) is recorded into the
capture log via the prompt-flush handler (fix round 2, finding 1b),
then re-raised so exit code/stderr behavior is preserved — the
capture directory still holds the startup-crash evidence.

NORMAL runs (no flag) are byte-identical to the pre-bootstrap import
order — ``meetandread.main`` first, straight to ``main()`` — with no
new logging side effects. ``main()`` parses the flag again for its own
decisions but never reconfigures capture logging the bootstrap already
configured (``capture_mode.capture_logging_configured``).

Fix round 2, finding 1a: the pip-installed console script points HERE
(``meetandread.__main__:run`` — see pyproject ``[project.scripts]``),
so the installed ``meetandread`` command gets the identical
parse-and-configure-before-heavy-imports discipline. ``run`` is the
public entry; module-level execution is guarded so importing this
module (as the console-script loader does) does not start the app.
"""

import sys

_CAPTURE_FLAG_PARSED = None  # Optional[Path] once parsed below


def _bootstrap() -> None:
    """Parse the capture flag; configure capture logging if requested."""
    from meetandread.capture_mode import (
        ISSUE_CAPTURE_FLAG,
        CaptureModeError,
        configure_capture_logging,
        parse_capture_flag,
    )

    global _CAPTURE_FLAG_PARSED
    try:
        capture_dir = parse_capture_flag()
        _CAPTURE_FLAG_PARSED = capture_dir
        if capture_dir is not None:
            configure_capture_logging(capture_dir)
    except CaptureModeError as exc:
        # Pre-logging path: the root logger's lastResort handler routes
        # ERROR to stderr (same discipline as main()'s refusals).
        import logging

        logging.getLogger(__name__).error("%s %s", ISSUE_CAPTURE_FLAG, exc)
        sys.exit(2)


def _run() -> None:
    _bootstrap()
    if _CAPTURE_FLAG_PARSED is not None:
        try:
            from meetandread.main import main
        except BaseException:
            # Import-time failure of the application subsystems
            # (PyQt/widgets/native audio) WITH capture logging live:
            # record it, then re-raise so the process still fails with
            # its normal traceback/exit behavior.
            import logging

            logging.getLogger(__name__).exception(
                "Startup failure: meetandread.main could not be imported"
            )
            raise
        try:
            main(capture_dir=_CAPTURE_FLAG_PARSED)
        except SystemExit:
            # main()'s normal exit path (app.exec() -> sys.exit(code)):
            # not a startup failure — pass through untouched.
            raise
        except BaseException:
            # Startup failure INSIDE main() (fix round 2, finding 1b)
            # with capture logging live: record it with traceback, then
            # re-raise so the process still fails normally. The capture
            # dir holds the startup-crash record.
            import logging

            logging.getLogger(__name__).exception(
                "Startup failure: meetandread failed during startup"
            )
            raise
    else:
        # NORMAL run: same import order as before the bootstrap existed.
        from meetandread.main import main

        main()


def run() -> None:
    """Public executable entry (console script, ``python -m``)."""
    _run()


if __name__ == "__main__":
    _run()
