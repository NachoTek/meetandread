"""Lightweight executable bootstrap for the Issue Reporter (issue
#111, ADR 0003) — the reporter's twin of ``meetandread.__main__``.

This module exists so the frozen build's ``issue-reporter.exe`` (and
any console script) enters the reporter through the same deliberate
boundary the app's bootstrap gives ``meetandread.exe``:

- A NAMED module (not ``-c``) whose import graph is exactly the
  wizard's stdlib-only graph — the structural defense test
  (tests/test_reporter.py) proves the chain stays free of the audio
  stack and the Qt widget tree, and this bootstrap adds nothing to it.
- A ``run()`` entry for the pip console script and the PyInstaller
  spec's second EXE analysis target.
- A frozen-mode default for the supervised app command (below) so the
  packaged reporter supervises the packaged app.

The frozen default lives in ``reporter.launch_app`` — packaging (#111)
injects the frozen exe path; tests inject a stub. When NOT frozen, the
reporter runs from a normal interpreter and ``launch_app`` keeps its
``[sys.executable, ``-m``, ``meetandread``]`` default, exactly as
before this module existed.
"""

import sys


def run() -> int:
    """Public executable entry: the wizard's ``main()`` owns the flow
    (recovery scan, describe, launch, reproduce, stop, review, submit)
    and its crash handling; this bootstrap only delegates."""
    from meetandread.reporter_wizard import main

    return main(argv=sys.argv[1:])


if __name__ == "__main__":
    sys.exit(run())
