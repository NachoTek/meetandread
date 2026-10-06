"""Manual Submission — the final wizard step (issue #109,
docs/specs/issue-reporting.md "Submission", ADR 0004).

The only path by which diagnostics leave the machine. After the user
approves the review screen (#108 wrote the redacted bundle artifact),
this step opens the user's default browser on the repository's New
Issue form prefilled via query params — title and body built from the
user's own problem description — and copies the bundle's full local
path to the clipboard.

## The privacy line: the body is public, the clipboard is private

GitHub query params cannot attach files, so the user attaches the
bundle by hand. The prefilled issue body therefore carries only the
NEUTRAL bundle filename plus attach-by-hand instructions — NEVER a
local filesystem path: the issue body is public, and an unredacted
path would disclose the user's home directory (amended spec). The
full local path goes to the CLIPBOARD and the review screen instead,
where attaching is a paste away.

## The outbound-action budget (ADR 0004): exactly one

No token ships inside the app and no report endpoint exists. The
browser open is the single outbound action this module ever
performs, and it happens only after the user says yes at the submit
prompt. The clipboard copy is local. Filing from the user's own
GitHub account subscribes them to their own issue for free.

## Fail-closed tail of the fail-closed chain

Submission consumes the artifact the review step WROTE
(``diagnostics_bundle.txt``, a stable contract since #108). If that
artifact is absent — review failed closed, or the directory was
never reviewed — there is nothing submittable on disk and
:func:`build_submission` returns :class:`SubmissionUnavailable`
rather than a URL pointing at a nonexistent file. The submittable
file on disk is never rewritten here: what the user reviewed is
byte-for-byte what they attach.

## Defense discipline

Stdlib-only like every reporter module (ADR 0003): ``webbrowser``,
``urllib.parse``, ``subprocess`` (the Windows clipboard helper
``clip.exe``). The pure core — title/body/URL/clipboard-text
construction from a capture directory — is fast-lane testable
against fixture capture directories with zero subprocesses (spec,
Testing Decisions §3; ADR 0001).
"""

import subprocess
import sys
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from meetandread.diagnostics_bundle import (
    BUNDLE_FILE_NAME,
    IdentifierSet,
    default_identifiers,
    redact_text,
)
from meetandread.reporter import read_description

# The repository's New Issue form (spec, Repo facts: NachoTek/
# meetandread, public — the reason the body carries no local path).
NEW_ISSUE_URL = "https://github.com/NachoTek/meetandread/issues/new"

# Title/body budget: GitHub truncates or rejects over-long query
# params, and a URL of unbounded length never reaches the form. The
# title keeps the head of the user's words; the body's fixed
# instruction block leaves the description less room than the body
# cap itself.
FALLBACK_TITLE = "meetandread issue report"
MAX_TITLE_LEN = 100
MAX_BODY_LEN = 4000
TRUNCATION_MARKER = " (...)"

# The attach-by-hand instruction block. GitHub query params cannot
# attach files; the user must attach the bundle themselves. The
# filename is the NEUTRAL one — never a local path (amended spec).
_BODY_INSTRUCTIONS = (
    "## Diagnostics Bundle\n\n"
    f"Please attach the file `{BUNDLE_FILE_NAME}` to this issue.\n"
    "(GitHub cannot attach files automatically: attach it by hand —\n"
    "drag it into the form or use the attach controls. Its full path\n"
    "is on your clipboard and was shown by the Issue Reporter.)\n\n"
    "The bundle contains a DEBUG log, an interaction trace, resource\n"
    "snapshots, environment info, and the termination record — with\n"
    "usernames, home paths, email addresses, and machine identifiers\n"
    "redacted, and no audio and no transcript content."
)


@dataclass(frozen=True)
class SubmissionDraft:
    """A complete, ready-to-file submission built from one capture
    directory: the prefilled New Issue URL (title/body query params),
    the clipboard text (the bundle's FULL local path — private), and
    the neutral attach-by-hand instructions' subject (public)."""

    title: str
    body: str
    url: str
    clipboard_text: str
    bundle_path: Path


@dataclass(frozen=True)
class SubmissionUnavailable:
    """The fail-closed verdict: nothing submittable exists on disk.

    ``detail`` is human-safe (leak-free by construction — it names
    no capture-directory path).
    """

    detail: str


SubmissionResult = Union[SubmissionDraft, SubmissionUnavailable]


def _truncate_marked(text: str, limit: int) -> str:
    """Trim *text* to *limit* chars, marking the cut. Word-safe when
    possible (the marker makes an over-long description visibly
    truncated, never silently shorn)."""
    if len(text) <= limit:
        return text
    room = limit - len(TRUNCATION_MARKER)
    cut = text[:room].rstrip()
    return cut + TRUNCATION_MARKER


def build_submission(
    capture_dir: Path,
    identifiers: Optional[IdentifierSet] = None,
) -> SubmissionResult:
    """Build the prefilled New Issue URL and clipboard text.

    A PURE function of the capture directory (any state): title and
    body from the user's own problem description — redacted with the
    same identifier rewriting as the bundle, because the description
    is typed by the user and may contain their own paths — the body
    carrying the NEUTRAL bundle filename and attach-by-hand
    instructions and never a local filesystem path. Returns
    :class:`SubmissionDraft` or, when no reviewed bundle artifact
    exists on disk, :class:`SubmissionUnavailable`.
    """
    d = Path(capture_dir)
    bundle_path = d / BUNDLE_FILE_NAME
    if not bundle_path.is_file():
        return SubmissionUnavailable(
            detail=(
                "no reviewed Diagnostics Bundle artifact exists for "
                "this capture run (review must pass before anything "
                "can be submitted)"
            )
        )

    ids = identifiers if identifiers is not None else (
        default_identifiers()
    )
    description = read_description(d)
    if description:
        # The description enters the PUBLIC issue: redact it exactly
        # like the bundle redacts its description line.
        description = redact_text(description, ids)

    # The title: the first line of the user's own words, flattened
    # to a single line (a query-param title has no newlines).
    if description:
        first_line = description.strip().splitlines()[0].strip()
        title = _truncate_marked(first_line, MAX_TITLE_LEN)
    else:
        title = FALLBACK_TITLE

    # The body: description (full, redacted) + attach instructions.
    # The instructions are budgeted FIRST — an over-long description
    # shrinks (markedly truncated), but the neutral filename and the
    # attach-by-hand instructions ALWAYS survive into the prefilled
    # body (AC: the body states plainly that the user must attach
    # the bundle by hand — a cut tail must never delete them).
    if description:
        budget = MAX_BODY_LEN - (
            len(_BODY_INSTRUCTIONS) + 2
        )
        trimmed = _truncate_marked(
            description.strip(), max(budget, 0)
        )
        body = trimmed + "\n\n" + _BODY_INSTRUCTIONS
    else:
        body = _BODY_INSTRUCTIONS

    query = urllib.parse.urlencode({"title": title, "body": body})
    url = f"{NEW_ISSUE_URL}?{query}"
    return SubmissionDraft(
        title=title,
        body=body,
        url=url,
        clipboard_text=str(bundle_path),
        bundle_path=bundle_path,
    )


# ---------------------------------------------------------------------------
# IO seams (the wizard's injectable edges)
# ---------------------------------------------------------------------------


def open_new_issue_form(url: str) -> bool:
    """Open the user's default browser on the prefilled form.

    THE single outbound action of the whole reporting feature (ADR
    0004): stdlib ``webbrowser``, no token, no endpoint, no payload
    beyond the URL itself. Returns whether a browser was opened.
    """
    return bool(webbrowser.open(url, new=2))


def copy_to_clipboard(text: str) -> bool:
    """Copy the bundle's full local path to the clipboard (local
    only — nothing leaves the machine). Windows pipes through
    ``clip.exe`` (stdlib subprocess discipline, ADR 0003); elsewhere
    best-effort through the terminal, with the caller always having
    the path on screen as the fallback."""
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["clip.exe"],
                input=text,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return False
