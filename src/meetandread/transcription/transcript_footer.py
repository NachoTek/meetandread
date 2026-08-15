"""The Transcript Footer: the sole owner of the Transcript Footer format.

A Transcript is a human-readable Markdown body followed by a machine-readable
Transcript Footer — a JSON object carrying the Recording's structured data
(words with timing, segments, speaker matches, recording start time).  This
module is the single source of truth for that format.

The public interface is exactly four operations:

``parse``
    Decode the Transcript Footer metadata of a complete Transcript.  Returns
    the metadata, or ``None`` when there is no usable Transcript Footer.

``split``
    Separate a complete Transcript into its Markdown body and decoded
    metadata.  Returns ``(body, metadata)``, or ``None`` when there is no
    usable Transcript Footer.

``join``
    Build a complete Transcript from a Markdown body and metadata, using the
    one canonical Transcript Footer representation.  Every write goes through
    here so the format drifts toward one shape over time.

``strip``
    Return the Markdown body without decoding the Transcript Footer JSON.
    Used by renderers that only want readable text; it keeps working when the
    Transcript Footer JSON is malformed.

This module also owns the format of the Post-processing Outcome block — the
``post_process`` object a footer may carry — and exposes it through
``PostProcessOutcome`` plus three Outcome operations:

``outcome_from_block`` / ``read_post_process_outcome``
    Decode the Outcome from a metadata block / a complete Transcript.

``write_post_process_outcome``
    Write or replace the Outcome block in a Transcript file, preserving the
    Markdown body and every other metadata field.

``clear_post_process_outcome``
    Remove the Outcome block from a Transcript file, preserving the Markdown
    body and every other metadata field.

The Transcript Footer is introduced by a Markdown horizontal rule on its own
line, followed by a blank line and an HTML comment labelled ``METADATA``.
Parsing selects the **last** such Transcript Footer, so body text that merely
quotes the format cannot shadow the real Transcript Footer.  Parsing tolerates
the known whitespace variant following the metadata label.  All framing
literals, closing syntax, and whitespace rules are private to this module;
callers express intent through the four operations alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from meetandread.utils.file_utils import atomic_write

__all__ = [
    "parse",
    "split",
    "join",
    "strip",
    "PostProcessOutcome",
    "outcome_from_block",
    "read_post_process_outcome",
    "write_post_process_outcome",
    "clear_post_process_outcome",
]


# ---------------------------------------------------------------------------
# Private framing details
#
# The Transcript Footer is a Markdown horizontal rule (``---``) on its own
# line, a blank line, then an HTML comment whose label is ``METADATA`` and
# whose payload is a JSON object.  The opener carries two leading newlines:
# the blank line before the rule guarantees valid Markdown framing regardless
# of the body's trailing newlines, and ``rfind`` on this exact opener consumes
# exactly those two newlines so the body round-trips byte-for-byte (see
# ``join`` / ``split``).
# ---------------------------------------------------------------------------

_HORIZONTAL_RULE = "---"
_LABEL = "<!-- METADATA:"
_CLOSER = " -->"

# The full Transcript Footer opener, including its two leading newlines.  The
# space after the label is *not* part of this constant: parsing tolerates its
# presence or absence, while ``join`` always emits the canonical spaced form.
_OPENER = "\n\n" + _HORIZONTAL_RULE + "\n\n" + _LABEL

# Parsing form of the opener.  Each newline may carry a carriage return, so a
# Transcript saved with Windows CRLF line endings is still recognised; ``join``
# continues to emit the canonical LF form, so LF round-trips stay exact.
_OPENER_RE = re.compile(
    r"(?:\r?\n){2}" + re.escape(_HORIZONTAL_RULE) + r"(?:\r?\n){2}" + re.escape(_LABEL)
)

# The canonical body and Transcript Footer separator plus opening emitted by
# ``join``.
_FRAMING = _OPENER + " "


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _locate_opener(content: str) -> Optional[re.Match]:
    """Return the last Transcript Footer opener match, or ``None`` if none.

    Selecting the last opener means an earlier marker-like string in the body
    (e.g. a Transcript that discusses the format) cannot shadow the real
    Transcript Footer written at the end.
    """
    last = None
    for match in _OPENER_RE.finditer(content):
        last = match
    return last


def _payload(content: str, opener: re.Match) -> Optional[str]:
    """Extract the raw Transcript Footer payload text, or ``None`` if unusable.

    Tolerates optional horizontal whitespace immediately following the metadata
    label (the canonical spaced form and the tolerated no-space form).  The
    payload runs from after that whitespace to the **last** closing marker, so
    JSON string values that happen to contain the closing syntax do not
    truncate the parse.  Returns ``None`` when no closing marker is present.
    """
    after_label = content[opener.end():]
    after_label = after_label.lstrip(" \t")
    end = after_label.rfind(_CLOSER)
    if end == -1:
        return None
    return after_label[:end]


def _decode(content: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Locate and decode the last Transcript Footer.

    Returns ``(opener_index, metadata)`` for the last Transcript Footer, or
    ``None`` when the Transcript Footer is missing, lacks its closing syntax,
    or holds malformed or non-object JSON.
    """
    opener = _locate_opener(content)
    if opener is None:
        return None
    payload = _payload(content, opener)
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return opener.start(), data


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def parse(content: str) -> Optional[Dict[str, Any]]:
    """Decode the Transcript Footer metadata of a complete Transcript.

    Selects the last Transcript Footer and returns its decoded JSON object, or
    ``None`` when the Transcript Footer is missing, lacks its closing syntax,
    or holds malformed JSON.
    """
    decoded = _decode(content)
    if decoded is None:
        return None
    _opener_index, data = decoded
    return data


def split(content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Separate a complete Transcript into ``(markdown_body, metadata)``.

    Selects the last Transcript Footer.  Returns ``None`` when the Transcript
    Footer is missing, lacks its closing syntax, or holds malformed JSON.  The
    returned body is exactly the text before the Transcript Footer opener, so
    ``split(join(body, metadata))`` returns the original body and metadata.
    """
    decoded = _decode(content)
    if decoded is None:
        return None
    opener_index, data = decoded
    return content[:opener_index], data


def join(body: str, metadata: Dict[str, Any]) -> str:
    """Build a complete Transcript from a Markdown body and metadata.

    Uses the one canonical Transcript Footer representation.  The body is
    appended verbatim — it is not mutated — so a subsequent ``split`` recovers
    it exactly, regardless of its trailing-newline state.  The separator's
    blank line before the horizontal rule keeps the framing valid Markdown.
    """
    return (
        body
        + _FRAMING
        + json.dumps(metadata, indent=2)
        + _CLOSER
        + "\n"
    )


def strip(content: str) -> str:
    """Return the Markdown body without decoding the Transcript Footer JSON.

    Locates the last Transcript Footer opener and returns everything before
    it.  Because the Transcript Footer JSON is never decoded, a malformed
    Transcript Footer still yields the readable Markdown body.  When no
    Transcript Footer opener is present the whole content is the body and is
    returned unchanged.
    """
    opener = _locate_opener(content)
    if opener is None:
        return content
    return content[:opener.start()]


# ---------------------------------------------------------------------------
# Post-processing Outcome block
#
# The Outcome is the durable terminal result of Post-processing for a
# Recording — Completed (including zero-speaker results) or Failed (with the
# failing stage and reason).  It is carried in the Transcript Footer under
# the ``post_process`` key so it lives and dies with the Transcript file.
# ---------------------------------------------------------------------------

#: Metadata key carrying the Outcome inside a Transcript Footer.
OUTCOME_KEY = "post_process"

#: Outcome status values.
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

#: The failing Post-processing stage of a Failed Outcome.
STAGE_ENGINE_LOAD = "engine-load"
STAGE_TRANSCRIBE = "transcribe"
STAGE_DIARIZE = "diarize"
STAGE_DEPENDENCY = "dependency"
STAGE_AUDIO_MISSING = "audio-missing"

OUTCOME_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED})
OUTCOME_STAGES = frozenset(
    {
        STAGE_ENGINE_LOAD,
        STAGE_TRANSCRIBE,
        STAGE_DIARIZE,
        STAGE_DEPENDENCY,
        STAGE_AUDIO_MISSING,
    }
)


@dataclass(frozen=True)
class PostProcessOutcome:
    """The durable terminal result of Post-processing for a Recording.

    Attributes:
        status: ``completed`` or ``failed``.  A zero-speaker completion is a
            legitimate ``completed`` result, not a failure.
        attempted_at: ISO timestamp of the Post-processing attempt.
        stage: The failing stage — required for ``failed`` Outcomes, absent
            for ``completed`` ones.
        error: The failure message — present for ``failed`` Outcomes only.
    """

    status: str
    attempted_at: str
    stage: Optional[str] = None
    error: Optional[str] = None

    def to_block(self) -> Dict[str, Any]:
        """Encode as the canonical ``post_process`` metadata block."""
        block: Dict[str, Any] = {
            "status": self.status,
            "attempted_at": self.attempted_at,
        }
        if self.stage is not None:
            block["stage"] = self.stage
        if self.error is not None:
            block["error"] = self.error
        return block


def outcome_from_block(block: Any) -> Optional[PostProcessOutcome]:
    """Decode a ``post_process`` metadata block into a PostProcessOutcome.

    Returns ``None`` when the block is missing, not an object, or does not
    carry a recognised status/stage — a footer written by another version
    must never crash a reader.
    """
    if not isinstance(block, dict):
        return None
    status = block.get("status")
    if status not in OUTCOME_STATUSES:
        return None
    stage = block.get("stage")
    if stage is not None and stage not in OUTCOME_STAGES:
        return None
    attempted_at = block.get("attempted_at")
    if not isinstance(attempted_at, str):
        attempted_at = ""
    error = block.get("error")
    if not isinstance(error, str):
        error = None
    return PostProcessOutcome(
        status=status,
        attempted_at=attempted_at,
        stage=stage,
        error=error,
    )


def read_post_process_outcome(content: str) -> Optional[PostProcessOutcome]:
    """Decode the Outcome carried by a complete Transcript.

    Returns ``None`` when the Transcript Footer is missing or carries no
    usable Outcome.
    """
    data = parse(content)
    if data is None:
        return None
    return outcome_from_block(data.get(OUTCOME_KEY))


def write_post_process_outcome(path: Path, outcome: PostProcessOutcome) -> bool:
    """Write or replace the Outcome block in a Transcript file.

    The Markdown body and every other metadata field are preserved
    byte-for-byte; only the ``post_process`` block changes.  Returns ``True``
    on success, ``False`` when the file is missing or has no usable
    Transcript Footer (an Outcome cannot live outside the footer).
    """
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    parts = split(content)
    if parts is None:
        return False
    body, metadata = parts
    metadata[OUTCOME_KEY] = outcome.to_block()
    try:
        atomic_write(Path(path), join(body, metadata))
    except OSError:
        return False
    return True


def clear_post_process_outcome(path: Path) -> bool:
    """Remove the Outcome block from a Transcript file.

    The Markdown body and every other metadata field are preserved
    byte-for-byte; only the ``post_process`` block is removed.  Returns
    ``True`` when an Outcome was removed, ``False`` when the file is
    missing, has no usable Transcript Footer, or carried no Outcome.
    """
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    parts = split(content)
    if parts is None:
        return False
    body, metadata = parts
    if metadata.pop(OUTCOME_KEY, None) is None:
        return False
    try:
        atomic_write(Path(path), join(body, metadata))
    except OSError:
        return False
    return True
