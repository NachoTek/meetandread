"""Diagnostics Bundle assembly, Redaction, and the review screen
(issue #108, docs/specs/issue-reporting.md, ADR 0004).

The assembly edge of the capture-directory seam: a PURE function of a
capture directory in any state — complete, or crashed mid-run —
producing the single submittable artifact. Assembly combines the
redacted DEBUG log, the Interaction Trace, the Resource Snapshot
series, environment info, and the reporter-written termination record
into ``diagnostics_bundle.txt`` (predictable format: same constituent
parts, same order, so any two reports read the same way).

## Redaction runs before the review screen, ALWAYS

Usernames and home-directory paths, email addresses, and machine
identifiers are rewritten across EVERY component — log, trace,
snapshots, environment info, termination record — never only the log
(:func:`redact_text` + the per-component pass in :func:`assemble_bundle`).
Transcript text and Recording titles are excluded outright: exclusion
is enforced at the capture boundary, and PROVEN here — every
component is scanned against the capture run's canary registry
(:mod:`meetandread.transcript_canary`); any hit means the boundary
failed, and assembly fails closed.

## Fail-closed (owner-approved amendment, spec review #113)

If assembly or redaction fails for ANY component, creation of the
submittable artifact is BLOCKED (:class:`AssemblyFailure`) and the
user is told submission is unavailable — the system NEVER falls back
to raw or partially redacted capture data. An error path must not be
able to bypass the spec's strongest privacy guarantee:

- ``canary_leak``      — transcript/title content detected in a
  component; the failure detail carries NO fragment of the leak.
- ``redaction_incomplete`` — the identifier rewrite is demonstrably
  not idempotent (a poisoned identifier set); scrubbing cannot be
  trusted for this run.
- ``assembly_error``   — the read side itself failed.

## Crash tolerance (durable-append contract)

Assembly tolerates at most one truncated final record per append-only
file (log lines and JSONL records read through
``capture_mode.read_appendable_records``): a crashed run keeps every
valid record before the truncation. Missing artifacts (a startup
crash wrote nothing) assemble as "(not captured)" sections —
incompleteness is a VALUE, never an assembly failure.

## Review screen

:func:`render_review` renders "here's what will be sent" FROM THE
REDACTED ARTIFACT ITSELF — the user reviews exactly the bytes that
would leave the machine. Nothing unredacted is ever displayed.

## Defense discipline

Stdlib-only BY DESIGN (the reporter program's constraint, ADR 0003):
imports ONLY sibling stdlib-only modules (``capture_mode``,
``interaction_trace``, ``resource_snapshots``, ``environment_info``,
``transcript_canary``, ``reporter``). The pure assembly edge is
fast-lane testable against fixture capture directories with zero
subprocesses (spec, Testing Decisions; ADR 0001).
"""

import json
import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from meetandread.capture_mode import (
    CAPTURE_LOG_PREFIX,
    read_appendable_records,
    read_completion_marker,
)
from meetandread.environment_info import read_environment_info
from meetandread.interaction_trace import read_trace_events
from meetandread.reporter import read_description, read_termination_record
from meetandread.resource_snapshots import read_snapshots
from meetandread.transcript_canary import (
    read_canary_ngrams,
    scan_text_for_canary_hits,
)

# Filename of the submittable bundle artifact inside the capture dir.
# #109 (Manual Submission) attaches this file by hand; treat as
# stable contract.
BUNDLE_FILE_NAME = "diagnostics_bundle.txt"

# The bundle's format marker (version of the layout itself).
BUNDLE_FORMAT = "diagnostics bundle v1"

# The constituent order — the "predictable format" contract: every
# bundle carries the same parts in the same order (story 28).
COMPONENT_ORDER = (
    "environment",
    "termination",
    "interaction_trace",
    "resource_snapshots",
    "debug_log",
)

# Redaction tokens (stable vocabulary, readable in a public report).
_HOME_TOKEN = "<redacted:home>"
_EMAIL_TOKEN = "<redacted:email>"
_MACHINE_TOKEN = "<redacted:machine>"
_USER_TOKEN = "<redacted:user>"

# Identifier values shorter than this are too weak to rewrite safely
# (they would shred ordinary words); they are skipped.
_MIN_IDENTIFIER_LEN = 3

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

# POSIX home trees: /home/<name>/..., /Users/<name>/...
_POSIX_HOME_RE = re.compile(r"/(?:home|users|Users)/[^/\s]+")


@dataclass(frozen=True)
class IdentifierSet:
    """The identifiers Redaction rewrites across every component.

    Gathered reporter-side from the environment
    (:func:`default_identifiers`) or injected by tests. ``None`` /
    empty / too-short values are ignored — a 1-2 char "username"
    would rewrite ordinary words.
    """

    home_dir: Optional[str] = None
    username: Optional[str] = None
    machine_name: Optional[str] = None


@dataclass(frozen=True)
class AssemblyFailure:
    """The fail-closed verdict: NO submittable artifact is produced.

    ``reason`` is one of ``canary_leak``, ``redaction_incomplete``,
    ``assembly_error``; ``component`` names the constituent that
    failed; ``detail`` is human-safe (never contains identifier
    values or leaked content — the failure itself must not leak).
    """

    reason: str
    component: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class AssembledBundle:
    """A successfully assembled, fully redacted bundle."""

    text: str
    path: Optional[Path] = None
    component_summaries: dict = field(default_factory=dict)


AssemblyResult = Union[AssembledBundle, AssemblyFailure]


def default_identifiers() -> IdentifierSet:
    """Gather the identifier set from the reporter's environment.

    Stdlib-only: the username/machine name come from the environment
    variables (Windows: USERNAME/COMPUTERNAME; POSIX: USER/HOSTNAME
    fallbacks), the home dir from ``Path.home()``. Every value is
    optional — redaction runs with whatever is actually known.
    """
    username = _safe_identifier(
        os.environ.get("USERNAME") or os.environ.get("USER")
    )
    machine = _safe_identifier(
        os.environ.get("COMPUTERNAME")
        or os.environ.get("HOSTNAME")
        or platform.node()
    )
    try:
        home = str(Path.home())
    except OSError:
        home = ""
    return IdentifierSet(
        home_dir=home or None,
        username=username,
        machine_name=machine,
    )


def _safe_identifier(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value if len(value) >= _MIN_IDENTIFIER_LEN else None


# ---------------------------------------------------------------------------
# Redaction (pure text rewriting)
# ---------------------------------------------------------------------------


def redact_text(text: str, ids: IdentifierSet) -> str:
    """Rewrite identifiers in *text*; pure and order-stable.

    Passes (each a plain substitution — no recursion, so a token can
    never be re-rewritten by a later pass):

    1. emails (before usernames: bob@example.com must become ONE
       email token, not a mangled user token inside it);
    2. the exact home directory (both slash orientations,
       case-insensitive — Windows paths appear both ways in logs);
    3. generic home trees (any ``C:\\Users\\<name>`` / ``/home/<name>``
       path — a path is a home-directory path whoever it belongs to);
    4. the machine name (case-insensitive word);
    5. the username (word-boundary, case-insensitive — never a
       substring: "Bobby" is a different word).
    """
    out = text
    out = _EMAIL_RE.sub(_EMAIL_TOKEN, out)

    home = _safe_identifier(ids.home_dir)
    if home:
        escaped = re.escape(home)
        # The exact home dir matches with either slash orientation
        # (Windows paths appear both ways in logs); keep the path's
        # own separators by matching each separator permissively.
        pattern = re.sub(
            r"[/\\]+", r"[/\\\\]+", escaped
        )
        out = re.sub(pattern, _HOME_TOKEN, out, flags=re.IGNORECASE)
    # Generic home trees (any user's, in either slash orientation or
    # case): C:\Users\<name>\..., C:/users/<name>/..., /home/<name>/.
    # The trailing separator is preserved so the rewrite stays a
    # path-shaped token.
    out = re.sub(
        r"(?i)[a-z]:[/\\]+users[/\\]+[^/\\\s]+[/\\]",
        _HOME_TOKEN + "/",
        out,
    )
    out = re.sub(
        r"(?i)[a-z]:[/\\]+users[/\\]+[^/\\\s]+$",
        _HOME_TOKEN,
        out,
    )
    out = _POSIX_HOME_RE.sub(_HOME_TOKEN, out)

    machine = _safe_identifier(ids.machine_name)
    if machine:
        out = re.sub(
            r"\b" + re.escape(machine) + r"\b",
            _MACHINE_TOKEN,
            out,
            flags=re.IGNORECASE,
        )

    username = _safe_identifier(ids.username)
    if username:
        out = re.sub(
            r"\b" + re.escape(username) + r"\b",
            _USER_TOKEN,
            out,
            flags=re.IGNORECASE,
        )
    return out


def _redaction_is_idempotent(ids: IdentifierSet) -> bool:
    """Can this identifier set actually be scrubbed?

    The fail-closed pre-check for poisoned sets: rewriting must be
    idempotent — applying it twice yields the same text as once. A
    username like ``redacted`` embeds itself in every redaction token
    (``<redacted:home>`` contains it), so a second pass would rewrite
    the tokens again — proof the scrubber cannot converge for this
    set. Such a run produces NO artifact, never a partial one.
    """
    probe = "user %s on %s in %s mail a@b.co" % (
        ids.username or "xx",
        ids.machine_name or "yy",
        ids.home_dir or "zz",
    )
    once = redact_text(probe, ids)
    twice = redact_text(once, ids)
    return once == twice


# ---------------------------------------------------------------------------
# Component readers (each: raw text + torn-tail-tolerant)
# ---------------------------------------------------------------------------


def _find_capture_log(capture_dir: Path) -> Optional[Path]:
    matches = sorted(
        p
        for p in capture_dir.iterdir()
        if p.is_file()
        and p.name.startswith(CAPTURE_LOG_PREFIX)
        and p.suffix == ".log"
    )
    return matches[0] if matches else None


def _read_log_lines(capture_dir: Path) -> List[str]:
    log = _find_capture_log(capture_dir)
    if log is None:
        return []
    return read_appendable_records(log)


def _format_jsonl(events: List[dict]) -> str:
    return "".join(
        json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n"
        for event in events
    )


# ---------------------------------------------------------------------------
# Assembly (pure: capture dir -> redacted bundle text)
# ---------------------------------------------------------------------------


def assemble_bundle(
    capture_dir: Path, identifiers: Optional[IdentifierSet] = None
) -> AssemblyResult:
    """Assemble the Diagnostics Bundle from *capture_dir*.

    Pure function of the directory in ANY state (complete, crashed,
    or nearly empty): reads every constituent through the torn-tail
    tolerant readers, redacts each, scans each against the canary
    registry, and renders the fixed-format bundle text. Returns
    :class:`AssembledBundle` or the fail-closed
    :class:`AssemblyFailure` — never raises for content reasons, and
    never falls back to raw data.
    """
    d = Path(capture_dir)
    ids = identifiers or default_identifiers()

    if not _redaction_is_idempotent(ids):
        return AssemblyFailure(
            reason="redaction_incomplete",
            component=None,
            detail=(
                "the identifier set cannot be scrubbed reliably "
                "for this run"
            ),
        )

    try:
        environment = read_environment_info(d)
        termination = read_termination_record(d)
        trace_events = read_trace_events(d)
        snapshots = read_snapshots(d)
        log_lines = _read_log_lines(d)
        canary = read_canary_ngrams(d)
        marker = read_completion_marker(d)
        description = read_description(d)
    except OSError as exc:
        return AssemblyFailure(
            reason="assembly_error",
            component=None,
            detail=f"capture directory read failed: {type(exc).__name__}",
        )

    # Raw (pre-redaction) component texts — the canary scan runs over
    # RAW content (a leak is a leak whether or not redaction would
    # happen to rewrite it; and identifier rewriting must not be able
    # to mask a transcript leak). The user-authored description and
    # the run name are checked like every other component: a pasted
    # transcript fragment in the description must fail assembly too.
    raw_components = {
        "environment": (
            json.dumps(environment, sort_keys=True)
            if environment is not None
            else ""
        ),
        "termination": (
            json.dumps(termination, sort_keys=True)
            if termination is not None
            else ""
        ),
        "interaction_trace": _format_jsonl(trace_events),
        "resource_snapshots": _format_jsonl(snapshots),
        "debug_log": "\n".join(log_lines),
        "description": description or "",
        "run_name": d.name,
    }

    # Canary scan: every component (the five constituents plus the
    # description and run-name headers), fail-closed, leak-free
    # detail.
    scan_targets = tuple(COMPONENT_ORDER) + ("description", "run_name")
    for component in scan_targets:
        hits = scan_text_for_canary_hits(
            raw_components[component], known=canary
        )
        if hits:
            return AssemblyFailure(
                reason="canary_leak",
                component=component,
                detail=(
                    "transcript or recording-title content was "
                    f"detected in the {component} component; the "
                    "capture boundary failed for this run"
                ),
            )

    # Redact every component (identifiers only — the scan above is
    # the transcript boundary's proof layer, not a scrubber).
    def r(component: str) -> str:
        return redact_text(raw_components[component], ids)

    def rl(line: str) -> str:
        return redact_text(line, ids)

    run_name = d.name
    parts: List[str] = [
        f"{BUNDLE_FORMAT} — meetandread issue report",
        f"run: {run_name}",
    ]
    if description:
        parts.append(f"description: {rl(description)}")

    env_lines = (
        [f"  {k}: {v}" for k, v in sorted(environment.items())]
        if environment is not None
        else ["  (not captured)"]
    )
    parts.append("== environment ==")
    parts.extend(
        line if line.strip() == "(not captured)" else rl(line)
        for line in env_lines
    )

    if termination is not None:
        term_lines = [
            f"  outcome: {termination.get('outcome')}",
            f"  exit_code: {termination.get('exit_code')}",
            f"  started_at: {termination.get('started_at')}",
            f"  ended_at: {termination.get('ended_at') or '(unknown)'}",
        ]
    else:
        term_lines = ["  (not captured)"]
    parts.append("== termination ==")
    parts.extend(
        line
        if line.strip() == "(not captured)"
        else rl(line)
        for line in term_lines
    )

    if trace_events:
        parts.append(
            f"== interaction trace ({len(trace_events)} events) =="
        )
        parts.append(rl(_format_jsonl(trace_events)).rstrip("\n"))
    else:
        parts.append("== interaction trace ==")
        parts.append("  (not captured)")

    if snapshots:
        parts.append(
            f"== resource snapshots ({len(snapshots)} samples) =="
        )
        parts.append(rl(_format_jsonl(snapshots)).rstrip("\n"))
    else:
        parts.append("== resource snapshots ==")
        parts.append("  (not captured)")

    marker_state = "present" if marker is not None else "absent"
    parts.append(f"== debug log (completion marker: {marker_state}) ==")
    if log_lines:
        parts.append(rl("\n".join(log_lines)))
    else:
        parts.append("  (not captured)")
    parts.append("")

    text = "\n".join(parts)
    summaries = {
        "environment": environment is not None,
        "termination": termination is not None,
        "interaction_trace": len(trace_events),
        "resource_snapshots": len(snapshots),
        "debug_log": len(log_lines),
    }
    return AssembledBundle(
        text=text, path=None, component_summaries=summaries
    )


def create_reviewable_bundle(
    capture_dir: Path, identifiers: Optional[IdentifierSet] = None
) -> AssemblyResult:
    """Assemble and WRITE the submittable artifact; fail-closed.

    The wizard's review step calls this. On success the redacted
    bundle text is written to ``<capture_dir>/diagnostics_bundle.txt``
    (prompt-flushed like every artifact) and the result carries its
    path — what the user reviews is exactly what would be attached.
    On ANY failure nothing is written: the caller tells the user
    submission is unavailable.
    """
    result = assemble_bundle(capture_dir, identifiers=identifiers)
    if isinstance(result, AssemblyFailure):
        return result
    path = Path(capture_dir) / BUNDLE_FILE_NAME
    # newline="" pins LF: the artifact on disk is byte-identical to
    # the redacted text shown at review (#109 attaches exactly those
    # bytes; text-mode CRLF translation would diverge them).
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(result.text)
        fh.flush()
        os.fsync(fh.fileno())
    return AssembledBundle(
        text=result.text,
        path=path,
        component_summaries=result.component_summaries,
    )


# ---------------------------------------------------------------------------
# Review screen ("here's what will be sent" — the redacted artifact)
# ---------------------------------------------------------------------------


# Review-screen debug-log preview: the bundle FILE carries the full
# log; the SCREEN preview caps it so the review stays readable (and
# a multi-megabyte capture log cannot bury the wizard's console).
REVIEW_LOG_PREVIEW_LINES = 40
# Trace/snapshot preview caps for the same reason (the JSONL sections
# of a minutes-long run can dwarf the log).
REVIEW_JSONL_PREVIEW_LINES = 12


def render_review(bundle: AssembledBundle) -> str:
    """Render the review screen from the REDACTED artifact itself.

    The user sees the bundle's own contents — never anything
    unredacted — plus the plain-language privacy statement (no audio,
    no transcript content, identifiers redacted) that lets a user
    report with confidence (stories 11-14). The debug-log section is
    previewed (its final lines, with the elided count noted); the
    written artifact carries the complete log.
    """
    text = bundle.text
    for marker in (
        "== debug log",
        "== interaction trace",
        "== resource snapshots",
    ):
        head, sep, tail = text.partition(marker)
        if not sep:
            continue
        lines = tail.splitlines()
        # Re-attach the header line (lines[0] is the rest of it).
        header, body = lines[0], lines[1:]
        cap = (
            REVIEW_LOG_PREVIEW_LINES
            if marker == "== debug log"
            else REVIEW_JSONL_PREVIEW_LINES
        )
        if len(body) > cap:
            preview = [header] + [
                f"  (... {len(body) - cap} further lines elided — "
                "the full section is in the bundle file ...)"
            ] + body[-cap:]
            text = head + marker + "\n".join(preview)
    header = [
        "=" * 60,
        "REVIEW — here is what will be sent",
        "=" * 60,
        "",
        "This Diagnostics Bundle contains:",
        "- the DEBUG log (identifiers redacted)",
        "- the Interaction Trace (named user actions, no keystrokes)",
        "- the Resource Snapshot series (RAM/CPU over the run)",
        "- environment info (app version, OS, hardware class)",
        "- the termination record (how the run ended)",
        "",
        "It contains NO audio and NO transcript content.",
        "Usernames, home-directory paths, email addresses, and machine",
        "identifiers have been redacted everywhere they appeared.",
        "(The bundle file below carries the COMPLETE contents; long",
        "sections are previewed here for readability.)",
        "",
        "-" * 60,
        "",
    ]
    return "\n".join(header) + text
