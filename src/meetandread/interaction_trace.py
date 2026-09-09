"""Interaction Trace core — the ordered record of semantic user actions
inside Issue Capture Mode (issue #105, docs/specs/issue-reporting.md).

Spec: story 24 and the "Captured content" section; the durability
contract is the owner-approved amendment of spec review #113 (durable
append semantics). Built ON the #104 capture-directory contract
(``capture_mode``): same directory, same prompt-flush discipline, same
torn-tail tolerance — never altering it.

## The contract (consumed by the Diagnostics Bundle assembler #108)

File — exactly one per capture directory:

- ``interaction_trace.jsonl`` — append-only JSONL, one line per event,
  created exclusively by ``install_interaction_trace`` at process start
  (capture runs only). Every completed event is written, flushed, AND
  fsynced to disk before the emit call returns, so a forced kill mid-run
  preserves every completed event; at most the one record being written
  at the instant of the kill is lost (the ``windows``-marked subprocess
  tests are the authoritative proof).

Assembly tolerates exactly one truncated final JSONL record:
``read_trace_events`` reads through the #104 torn-record reader
(``capture_mode.read_appendable_records``), so every complete record
before a torn tail survives.

## The closed event vocabulary (the whole set — nothing else is a trace event)

- ``button_pressed``        — a button/lobe/toggle was clicked
  (``target`` names it semantically).
- ``menu_item_selected``    — a menu or context-menu entry was chosen.
- ``shortcut_triggered``    — a keyboard shortcut fired.
- ``panel_opened``          — a floating panel became visible.
- ``panel_closed``          — a floating panel was hidden.
- ``panel_moved``           — a panel finished a drag (``x``/``y`` ints).
- ``panel_resized``         — a panel finished a resize (``w``/``h`` ints).
- ``device_selected``       — an audio source was (de)selected
  (``target``: ``microphone`` | ``system``; ``selected``: bool).
- ``window_focus_changed``  — app window focus gained/lost (``focused``).
- ``text_edited``           — free text was edited; carries ONLY
  ``{"chars": <int>}`` — the typed content NEVER enters the trace, any
  event payload, or any log line.

Event schema — one JSON object per line:

- ``ts``    — ISO-8601 local timestamp of the event.
- ``event`` — one of the closed vocabulary above.
- ``target`` — semantic name of the acted-on widget (enum-like, stable;
  never user free text beyond short semantic names).
- zero or more scalar extras (bool/int/float/str) specific to the event
  kind, as documented per name above.

Emission is fail-closed on shape (unknown event names or non-scalar
extras are refused and logged at ERROR — the trace must stay a valid
member of the closed vocabulary because a future relay's log-format
validation gate checks exactly this set), and best-effort on I/O (a
capture run must not die because a diagnostic write failed).

## Normal runs

Without capture mode there is no trace: ``install_interaction_trace``
is called only from the capture-mode startup path (``main`` with a
``--issue-capture`` dir), and ``emit_interaction_event`` /
``record_text_edited`` without an installed writer are silent no-ops
that create nothing anywhere.

This module is stdlib-only by design (like ``capture_mode``): it is
imported before/independently of any Qt or native-audio subsystem and
stays fast-lane testable (ADR 0001).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from meetandread.capture_mode import read_appendable_records

# Filename of the Interaction Trace inside the capture dir. One capture
# directory holds one run (#104) and one trace; treat as stable contract.
TRACE_FILE_NAME = "interaction_trace.jsonl"

# The closed set of named user actions (see module docstring). This is
# the vocabulary a future relay's log-format validation gate would check
# against — extending it is a spec-level decision, never ad hoc.
EVENT_VOCABULARY = frozenset(
    {
        "button_pressed",
        "menu_item_selected",
        "shortcut_triggered",
        "panel_opened",
        "panel_closed",
        "panel_moved",
        "panel_resized",
        "device_selected",
        "window_focus_changed",
        "text_edited",
    }
)

# The one payload extra ``text_edited`` permits — a char count, nothing
# else (the typed content must never enter the trace or any log line).
_TEXT_EDITED_EXTRAS = frozenset({"chars"})

logger = logging.getLogger(__name__)


class InteractionTraceError(Exception):
    """The Interaction Trace cannot be installed as requested.

    Raised only at process start (bad capture-dir state — a trace file
    from another run, or a second conflicting directory); the entry
    point treats it as a capture-mode startup failure.
    """


class _TraceWriter:
    """Durable append-only JSONL writer: one event, one line, fsynced.

    The file is opened once with ``O_APPEND | O_CREAT | O_EXCL`` (one
    trace per capture run — same exclusivity discipline as the #104
    capture log, never truncating or double-owning) and held for the
    process lifetime. Each event is formatted, persisted by a loop of
    ``os.write`` calls that continues across short writes until the
    full line is on disk (a zero-byte write is an error condition),
    and fsynced BEFORE :meth:`emit` returns — the durability
    contract's write side (a forced kill loses at most the one record
    being written at that instant). There is no Python-side buffer
    that could hold a completed record.

    Write-path failure semantics (documented contract, PR #123
    review):

    - ``os.write`` short-write → keep writing the remainder (loop).
    - zero-byte write → error condition (``False``, no fsync); the
      record boundary is intact (nothing landed), so the writer stays
      usable for later emits.
    - error after a PARTIAL write (≥1 byte of the record landed, then
      the write failed) → the file now ends in an unterminated
      fragment; the writer is PERMANENTLY POISONED — every later
      emit returns ``False`` without writing, so no later record can
      append to the fragment and claim success for a line that would
      read back as one malformed merged record. The fragment stays as
      the single torn tail the reader already tolerates. Chosen over
      best-effort boundary repair (writing a lone newline): after a
      partial-write failure the storage itself is failing, so the
      honest, fail-closed move is to stop claiming writes.
    - error at ``fsync`` AFTER the full line reached the OS → the
      record boundary is intact, so the writer is NOT poisoned;
      ``False`` is returned (the durability promise was not met) and
      later emits append cleanly.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._closed = False
        self._poisoned = False
        try:
            self._fd = os.open(
                str(self._path), os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as exc:
            raise InteractionTraceError(
                f"interaction trace file already exists: '{self._path}' "
                "— one capture run holds one trace; pass a fresh capture "
                "directory"
            ) from exc

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, payload: dict) -> bool:
        """Append one JSON line durably; True when it reached the disk.

        ``os.write`` may legally short-write (or write zero bytes), so
        the loop persists the FULL buffer — a partial count re-writes
        the remainder, a zero-byte write is an error condition — and
        only then fsyncs. True means the whole line is on disk.

        An error after a PARTIAL write poisons the writer (see the
        class docstring): later emits return False without writing,
        so nothing can append to the unterminated fragment.
        """
        if self._closed or self._poisoned:
            return False
        line = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
        view = memoryview(line)
        try:
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    raise OSError(
                        f"trace write wrote {written} of {len(view)} "
                        "remaining bytes"
                    )
                view = view[written:]
            os.fsync(self._fd)
            return True
        except OSError:
            if 0 < len(view) < len(line):
                # PARTIAL write then failure: the file now ends in an
                # unterminated fragment. Poison — never append to it.
                # (A full write's later fsync failure leaves the view
                # empty and the record boundary intact — not poisoned.)
                self._poisoned = True
            logger.exception(
                "interaction_trace_write_failed: file=%s", self._path.name
            )
            return False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                os.close(self._fd)
            except OSError:
                pass


# The trace writer THIS process has installed (None in a normal run).
# Process-global like the #104 capture claim: one run records one trace
# into one capture directory, from process start to the end of the
# event loop (close_interaction_trace ends the window at app teardown).
_writer: Optional[_TraceWriter] = None


def install_interaction_trace(capture_dir: Path) -> Path:
    """Install the trace writer for THIS capture run; return the path.

    Capture runs only — called once from the capture-mode startup path
    (``main`` with a ``--issue-capture`` dir), before the widget tree
    exists. Idempotent for the same directory (the same run re-asking);
    a second, different directory is refused (one run, one trace, one
    dir) as is a directory already holding a trace file (another run's).

    Raises:
        InteractionTraceError: conflicting install (see above).
    """
    global _writer
    if _writer is not None:
        if Path(_writer.path).parent == Path(capture_dir):
            return _writer.path
        _writer.close()
        raise InteractionTraceError(
            f"interaction trace already installed for "
            f"'{_writer.path.parent}' — one run records one trace"
        )
    writer = _TraceWriter(Path(capture_dir) / TRACE_FILE_NAME)
    _writer = writer
    logger.info(
        "Interaction Trace: recording user actions into %s", _writer.path
    )
    return _writer.path


def trace_installed() -> bool:
    """Is an Interaction Trace writer installed in this process?"""
    return _writer is not None


def close_interaction_trace() -> None:
    """Close the installed trace writer (process-exit teardown).

    Best-effort and idempotent: closes the writer's fd and clears the
    process-global install so every later emit is a no-op (``False``).
    Called by the capture-mode exit path AFTER the Qt event filter is
    removed and BEFORE ``QApplication`` destruction — the writer is
    stdlib-only (no Qt object), but the ordering keeps every trace
    emission strictly inside the live-application window (the trace
    runs from process start to the end of the event loop, not to
    interpreter death) so teardown can never race a half-dead emitter
    (PR #123 fix round 3).
    """
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None
        logger.debug("interaction_trace_closed")


def emit_interaction_event(event: str, target: str, **extras) -> bool:
    """Append one named event to the trace; True when written.

    Fail-closed on shape, best-effort on I/O: an event outside the
    closed vocabulary, a non-scalar extra, or a ``text_edited`` payload
    that is not exactly ``{"chars": int}`` is refused (ERROR log) — the
    trace must remain a valid member of the documented vocabulary. An
    I/O failure logs but never propagates: diagnostics must not crash
    the run being diagnosed.

    Without an installed writer (a normal run) this is a silent no-op
    returning False — no trace is recorded anywhere.
    """
    if _writer is None:
        return False
    if event not in EVENT_VOCABULARY:
        logger.error(
            "interaction_trace_refused: reason=unknown_event event=%s", event
        )
        return False
    if event == "text_edited":
        if set(extras) != set(_TEXT_EDITED_EXTRAS) or not isinstance(
            extras["chars"], int
        ) or isinstance(extras["chars"], bool):
            logger.error(
                "interaction_trace_refused: reason=text_edited_payload "
                "event=text_edited"
            )
            return False
    for value in extras.values():
        if not isinstance(value, (bool, int, float, str)):
            logger.error(
                "interaction_trace_refused: reason=non_scalar_extra "
                "event=%s",
                event,
            )
            return False
    payload = {
        "ts": datetime.now().isoformat(),
        "event": event,
        "target": target,
    }
    payload.update(extras)
    return _writer.emit(payload)


def record_text_edited(text: str) -> bool:
    """Record a free-text edit as exactly ``text edited (N chars)``.

    THE privacy boundary of the trace: the length is the only thing
    that leaves this call. The content never enters the trace, any
    event payload, or any log line — this function never logs and never
    formats the text anywhere. The Qt layer calls
    :func:`record_text_edited_length` instead so it never even holds
    the text; this entry point serves callers that already have the
    string (and must be the ONLY thing they do with it).
    """
    return record_text_edited_length(len(text))


def record_text_edited_length(length: int) -> bool:
    """Record one ``text edited (N chars)`` event from a length alone.

    The content-free twin of :func:`record_text_edited`: the Qt filter
    computes the length at the call site and passes the int, so typed
    content never leaves the widget at all.
    """
    return emit_interaction_event(
        "text_edited", target="text_field", chars=length
    )


def read_trace_events(capture_dir: Path) -> List[dict]:
    """Read the trace as parsed events; [] when absent or unreadable.

    Reads through the #104 torn-record reader: a forced kill can tear
    at most the final JSONL record, and every complete record before it
    — the whole run up to the kill — is preserved. Non-JSON lines (a
    torn tail that still ends in ``\\n``, or foreign content) are
    dropped, not fatal: assembly (#108) owns full validation.
    """
    path = Path(capture_dir) / TRACE_FILE_NAME
    events: List[dict] = []
    for line in read_appendable_records(path):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events
