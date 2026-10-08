"""Shared durable-write home for capture artifacts (issues
#105/#106/#160, docs/specs/issue-reporting.md): the append-only JSONL
series writer below, plus the single-object ``write_text_durable`` /
``write_json_durable`` helpers at the bottom (the extracted
prompt-flush sequence eight capture-artifact writers used to
hand-roll).

The durability contract's write side, extracted verbatim from the
Interaction Trace's ``_TraceWriter`` (PR #123, including its automated
review's short-write/partial-write findings) so every capture-directory
JSONL series — the Interaction Trace, the Resource Snapshot series —
has the SAME semantics by construction rather than by copy-paste drift:

- The file is opened once with ``O_APPEND | O_CREAT | O_EXCL`` (one
  series file per capture run — the #104 exclusivity discipline; never
  truncate or double-own another run's file) and held for the process
  lifetime.
- Each record is one JSON object per line, persisted by a loop of
  ``os.write`` calls that continues across short writes until the full
  line is on disk, and fsynced BEFORE :meth:`emit` returns — so a
  forced kill loses at most the one record being written at that
  instant. No completed record is ever held in a Python-side buffer or
  the OS write-back cache.
- Readers tolerate exactly one truncated final record (see
  ``capture_mode.read_appendable_records`` — the read side of the same
  contract, owned by #104).

Write-path failure semantics (documented contract, PR #123 review):

- ``os.write`` short-write → keep writing the remainder (loop).
- zero-byte write → error condition (``False``, no fsync); the record
  boundary is intact (nothing landed), so the writer stays usable for
  later emits.
- error after a PARTIAL write (≥1 byte of the record landed, then the
  write failed) → the file now ends in an unterminated fragment; the
  writer is PERMANENTLY POISONED — every later emit returns ``False``
  without writing, so no later record can append to the fragment and
  claim success for a line that would read back as one malformed
  merged record. The fragment stays as the single torn tail the reader
  already tolerates. (Chosen over best-effort boundary repair: after a
  partial-write failure the storage itself is failing, so the honest,
  fail-closed move is to stop claiming writes.)
- error at ``fsync`` AFTER the full line reached the OS → the record
  boundary is intact, so the writer is NOT poisoned; ``False`` is
  returned (the durability promise was not met) and later emits append
  cleanly.

This module is stdlib-only by design (like ``capture_mode`` and
``interaction_trace``): capture artifacts are written from before/independently
of any Qt or native-audio subsystem and stay fast-lane testable (ADR
0001).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DurableJSONLWriter:
    """Durable append-only JSONL writer: one record, one line, fsynced.

    The file is created exclusively (one series file per capture run)
    and held open for the process lifetime; :meth:`emit` persists each
    record fully before returning. See the module docstring for the
    full failure-semantics contract.
    """

    def __init__(self, path: Path, kind: str):
        """Open *path* exclusively for appending.

        Args:
            path: the series file inside the capture directory.
            kind: short human label for log lines (e.g.
                ``"interaction trace"``); used only in the failure log.

        Raises:
            FileExistsError: *path* already exists — one capture run
                holds one series file; callers turn this into their
                capture-mode startup refusal (exit 2).
        """
        self._path = Path(path)
        self._kind = kind
        self._closed = False
        self._poisoned = False
        self._fd = os.open(
            str(self._path), os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )

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
        module docstring): later emits return False without writing,
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
                        f"{self._kind} write wrote {written} of "
                        f"{len(view)} remaining bytes"
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
                "%s_write_failed: file=%s", self._kind, self._path.name
            )
            return False

    def close(self) -> None:
        """Close the writer's fd (best-effort, idempotent)."""
        if not self._closed:
            self._closed = True
            try:
                os.close(self._fd)
            except OSError:
                pass


def open_series_writer(
    path: Path, kind: str
) -> Optional[DurableJSONLWriter]:
    """Open a series writer, translating an existing file to None.

    Convenience seam for install functions: returns the writer, or
    None when the file already exists (another run's series — the
    caller refuses the install with its capture-mode startup error
    instead of writing a single byte into it).
    """
    try:
        return DurableJSONLWriter(path, kind)
    except FileExistsError:
        logger.error(
            "%s_write_refused: reason=file_exists file=%s", kind, path.name
        )
        return None


def write_text_durable(
    path: Path, text: str, *, newline: Optional[str] = None
) -> Path:
    """Write *text* as UTF-8 with the capture-artifact prompt-flush
    discipline: write, flush, fsync, close — the durability half of the
    open() mode contract (exclusive ``"x"``, truncate ``"w"``) stays
    the caller's choice, exactly as each site hand-rolled it (#160).

    The write is NOT atomic (no temp-file + replace — parity, not
    improvement); ``newline`` is forwarded to :func:`open` verbatim
    (``""`` pins LF; the default ``None`` keeps the text-mode platform
    newline translation the plain-text sites relied on).

    Raises: whatever :func:`open`, ``write``/``flush``, or
        ``os.fsync`` raise — OS errors propagate to the caller, the
        same contract every hand-rolled site had.
    """
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def write_json_durable(
    path: Path,
    obj: object,
    *,
    mode: str = "w",
    append_newline: bool = True,
) -> Path:
    """Write ONE JSON object as UTF-8 durably: serialize, newline,
    flush, fsync, close — one shared definition of the sequence eight
    capture-artifact writers used to copy-paste (#160's parity
    extraction; byte-identical output, error contract unchanged: OS
    errors propagate).

    ``json.dump``'s default separators/``ensure_ascii=True`` — the
    exact bytes every hand-rolled site put on disk (compact one-line
    JSON + trailing newline). ``mode="x"`` keeps a caller's exclusive
    creation (one run, one artifact — ADR 0005); the default ``"w"``
    truncates like the rewrite paths. ``append_newline`` suppresses
    the trailing newline for the (currently hypothetical) caller that
    does not want it.
    """
    with open(path, mode, encoding="utf-8") as fh:
        json.dump(obj, fh)
        if append_newline:
            fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path
