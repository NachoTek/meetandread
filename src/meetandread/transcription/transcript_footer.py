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
from typing import Any, Dict, Optional, Tuple

__all__ = ["parse", "split", "join", "strip"]


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

# The canonical body and Transcript Footer separator plus opening emitted by
# ``join``.
_FRAMING = _OPENER + " "


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _locate_opener(content: str) -> int:
    """Return the index of the last Transcript Footer opener, or ``-1`` if none.

    Selecting the last opener means an earlier marker-like string in the body
    (e.g. a Transcript that discusses the format) cannot shadow the real
    Transcript Footer written at the end.
    """
    return content.rfind(_OPENER)


def _payload(content: str, opener_index: int) -> Optional[str]:
    """Extract the raw Transcript Footer payload text, or ``None`` if unusable.

    Tolerates optional horizontal whitespace immediately following the metadata
    label (the canonical spaced form and the tolerated no-space form).  The
    payload runs from after that whitespace to the **last** closing marker, so
    JSON string values that happen to contain the closing syntax do not
    truncate the parse.  Returns ``None`` when no closing marker is present.
    """
    after_label = content[opener_index + len(_OPENER):]
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
    opener_index = _locate_opener(content)
    if opener_index == -1:
        return None
    payload = _payload(content, opener_index)
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return opener_index, data


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
    opener_index = _locate_opener(content)
    if opener_index == -1:
        return content
    return content[:opener_index]
