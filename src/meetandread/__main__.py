"""Lightweight executable bootstrap for ``python -m meetandread`` (and
the frozen exe — ``meetandread.spec`` analyzes THIS file as the entry).

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
import or startup raises while capture logging is configured, the
exception (with traceback) is recorded into the capture log via the
prompt-flush handler, so the capture directory still holds the
startup-crash evidence.

NORMAL runs (no flag) are byte-identical to the pre-bootstrap import
order — ``meetandread.main`` first, straight to ``main()`` — with no
new logging side effects. ``main()`` parses the flag again for its own
decisions but never reconfigures capture logging the bootstrap already
configured (``capture_mode.capture_logging_configured``).
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
            # its normal traceback/exit behavior. The capture dir now
            # holds the startup-crash record (the case finding 1 of the
            # PR #121 review closed).
            import logging

            logging.getLogger(__name__).exception(
                "Startup failure: meetandread.main could not be imported"
            )
            raise
        main(capture_dir=_CAPTURE_FLAG_PARSED)
    else:
        # NORMAL run: same import order as before the bootstrap existed.
        from meetandread.main import main

        main()


_run()
