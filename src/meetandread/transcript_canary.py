"""Transcript canary registry — the capture boundary's leak detector
(issue #108, docs/specs/issue-reporting.md, ADR 0004).

Transcript text and Recording titles never enter the Diagnostics
Bundle. The exclusion is enforced at the capture boundary
(transcript-bearing output never enters the log stream) — but the
spec demands PROOF: "negative/canary tests prove no transcript text
reaches the assembled bundle" (ADR 0004). This module is that proof
mechanism, in two halves:

## Capture side (this module, app process)

During a capture run the app, at every point where transcript text or
a recording title exists in memory, samples fragments into a
REGISTRY: the text is reduced to hashed word n-grams (SHA-256,
truncated to 16 hex chars) and appended to ``transcript_canary.jsonl`` in the capture directory —
one JSON line per batch of grams, prompt-flushed like every capture
artifact. PLAINTEXT NEVER TOUCHES THE REGISTRY (or any diagnostics
artifact): the file holds only opaque hex digests, so the registry
itself cannot leak meeting content — it is safe to keep next to the
capture artifacts and safe for the user to attach by hand with the
bundle.

Hashing is deliberately SALT-FREE (a pure function of the n-gram
text): assembly must recompute the same digests over bundle text
years later, in a different process, with no stored salt. The
n-grammer normalizes whitespace and hashes the lowercased n-gram —
deterministic across processes and platforms.

## Assembly side (the bundle, reporter process)

:func:`read_canary_ngrams` loads the registry (tolerating one torn
final record, like every append-only capture artifact), and the
Diagnostics Bundle assembler runs EVERY component's text through
:func:`canary_ngrams` — any intersection with the known set is a
LEAK, and assembly fails closed (no submittable artifact, "submission
unavailable"), because a detected leak means the capture boundary
already failed and the raw data cannot be trusted to redact.

## Why n-grams and not whole-text matching

A leak into a log line is rarely the whole sentence: formatters
truncate, wrap, quote, and interleave. Word n-grams (default 5 —
long enough that ordinary logging prose essentially never collides
with spoken-meeting prose, short enough to survive truncation and
reformatting) match fragments. Collisions of digest-vs-prose are
computationally impossible in practice (SHA-256 over 5-word
sequences); false negatives (leak too short to contain a full n-gram)
are the accepted residual risk — the capture boundary remains the
primary defense, the canary is the proof layer.

## Normal runs

Without capture mode there is no registry: ``record_canary_text``
without an installed writer is a silent no-op creating nothing
anywhere.

Stdlib-only by design (like every capture-side module): imported at
process start before any app subsystem; fast-lane testable (ADR
0001). The writer is process-global with the same idempotence
discipline as the Interaction Trace install.
"""

import hashlib
import json
import logging
import os
import string
from pathlib import Path
from typing import List, Optional, Set

from meetandread.capture_mode import read_appendable_records

# Filename of the canary registry inside the capture dir. The bundle
# assembler consumes this name; treat it as stable contract.
CANARY_FILE_NAME = "transcript_canary.jsonl"

# Punctuation stripped from token EDGES during tokenization (leaks
# arrive wrapped in quotes, parens, commas — see _tokenize).
_PUNCT = string.punctuation

# Word n-gram size for canary matching. 5 words: long enough that
# incidental overlap with ordinary log prose is negligible, short
# enough to survive log truncation/wrapping.
CANARY_NGRAM_SIZE = 5

# Digest length (hex chars) of one canary n-gram: sha256 truncated.
# 16 hex chars = 64 bits — collision-proof for registry-sized sets.
_GRAM_HEX_CHARS = 16

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Whitespace-split with edge punctuation stripped per token.

    Leaks arrive embedded in formatting — JSON quotes, parentheses,
    trailing commas (``"died transcribing Sebastopol canary``,
    ``(chunk 3),``). Stripping each token's surrounding punctuation
    lets the n-grams survive that wrapping; interior characters are
    untouched (a punctuation-insensitive mash would over-match).
    """
    return [
        token.strip(_PUNCT)
        for token in text.split()
        if token.strip(_PUNCT)
    ]


def canary_ngrams(text: str, size: int = CANARY_NGRAM_SIZE) -> Set[str]:
    """Reduce *text* to its set of hashed word n-grams.

    Pure and deterministic (no salt — see module docstring):
    punctuation-stripped, lowercased words; every window of *size*
    consecutive words is hashed to a truncated SHA-256 hex digest.
    Text with fewer than *size* words yields the empty set.
    """
    words = _tokenize(text)
    grams: Set[str] = set()
    for start in range(0, len(words) - size + 1):
        window = " ".join(words[start : start + size]).lower()
        digest = hashlib.sha256(window.encode("utf-8")).hexdigest()
        grams.add(digest[:_GRAM_HEX_CHARS])
    return grams


# The registry writer THIS process has installed (None in a normal
# run). Process-global like the trace writer: one capture run holds
# one registry, closed at app teardown.
_writer: Optional["_CanaryWriter"] = None


class _CanaryWriter:
    """Append-only, prompt-flushed writer for hashed canary n-grams.

    One JSON line per recorded batch: ``{"gram": "<hex16>"}`` —
    deliberately not the batch (many grams per line would couple the
    durability granularity to the sampling size; one gram per line
    keeps every completed gram preserved independently, and a torn
    tail drops at most one).

    The file is created with ``O_CREAT | O_EXCL``: one capture
    directory holds one run (ADR 0005), one registry.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = os.open(
            str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )

    def emit(self, gram: str) -> bool:
        """Append one gram; True when durably written."""
        if self._fd is None:
            return False
        payload = json.dumps({"gram": gram}) + "\n"
        data = payload.encode("utf-8")
        try:
            written = 0
            while written < len(data):
                chunk = os.write(self._fd, data[written:])
                if chunk <= 0:
                    return False
                written += chunk
            os.fsync(self._fd)
        except OSError:
            logger.error("transcript_canary_write_failed: error_class=OSError")
            return False
        return True

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def install_canary_registry(capture_dir: Path) -> Path:
    """Install the registry writer for THIS capture run; return path.

    Capture runs only — called once from the capture-mode startup
    path, before any transcript-bearing subsystem runs. Idempotent
    for the same directory; a second, different directory is refused
    (one run, one registry, one dir) with :class:`FileExistsError`-
    style discipline via ``O_EXCL``.
    """
    global _writer
    if _writer is not None:
        if _writer.path.parent == Path(capture_dir):
            return _writer.path
        previous = _writer.path
        _writer.close()
        _writer = None
        raise RuntimeError(
            f"transcript canary registry already installed for "
            f"'{previous.parent}' — one run records one registry"
        )
    writer = _CanaryWriter(Path(capture_dir) / CANARY_FILE_NAME)
    _writer = writer
    logger.info(
        "Transcript canary registry: sampling hashed fragments into %s",
        _writer.path,
    )
    return _writer.path


def canary_installed() -> bool:
    """Is a canary registry installed in this process?"""
    return _writer is not None


def record_canary_text(text: str) -> int:
    """Sample *text* (transcript fragment or recording title) into the
    registry; return the number of NEW grams written.

    THE capture-side privacy rule: the text is reduced to hashed
    n-grams BEFORE anything is written — plaintext never enters this
    function's outputs, the registry file, or any log line (this
    function never logs and never formats the text anywhere). Without
    an installed writer (a normal run) this is a silent no-op
    returning 0. I/O failures are logged (class only) and swallowed:
    the canary is best-effort diagnostics on top of a boundary that
    must hold anyway.
    """
    if _writer is None or not text or not text.strip():
        return 0
    grams = canary_ngrams(text)
    written = 0
    for gram in sorted(grams):
        if _writer.emit(gram):
            written += 1
    return written


def close_canary_registry() -> None:
    """Close the installed registry writer (process-exit teardown).

    Best-effort and idempotent; mirrors the trace writer's teardown
    discipline.
    """
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None
        logger.debug("transcript_canary_closed")


def read_canary_ngrams(capture_dir: Path) -> Set[str]:
    """Read the registry as the set of known canary grams; empty when
    absent or unreadable.

    Reads through the #104 torn-record reader: a forced kill can tear
    at most the final line; every completed gram before it survives.
    Corrupt (non-JSON / wrong-shape) lines are dropped, not fatal —
    assembly owns the fail-closed verdict, and a dropped line only
    weakens the check by one gram, never fabricating a leak.
    """
    path = Path(capture_dir) / CANARY_FILE_NAME
    grams: Set[str] = set()
    for line in read_appendable_records(path):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("gram"), str)
        ):
            grams.add(payload["gram"])
    return grams


def scan_text_for_canary_hits(
    text: str, known: Set[str]
) -> Set[str]:
    """Which canary grams appear inside *text*? (assembly-side scan)

    Pure: reduces *text* with the same n-grammer and returns the
    intersection with the known set. Empty intersection = clean. The
    bundle runs this over EVERY component; any hit fails assembly.
    """
    if not known or not text:
        return set()
    return canary_ngrams(text) & known
