"""Backward-compat shim for the legacy ``scrub`` naming.

All implementation now lives in :mod:`meetandread.transcription.retranscribe`.
This module re-exports the new names so existing imports such as::

    from meetandread.transcription.scrub import ScrubRunner

continue to work unchanged.

The legacy :class:`ScrubRunner` is preserved as a subclass of
:class:`RetranscribeRunner` that overrides only the sidecar filename tag, so
sidecars written by older code (``{stem}_scrub_{model}.md``) keep their
original naming. New code should prefer :class:`RetranscribeRunner`.
"""

from meetandread.transcription.retranscribe import RetranscribeRunner

__all__ = ["ScrubRunner", "RetranscribeRunner"]


class ScrubRunner(RetranscribeRunner):
    """Legacy alias for :class:`RetranscribeRunner`.

    Produces sidecar files using the historical ``_scrub_`` filename tag so
    existing callers and on-disk artifacts continue to behave as before.
    """

    SIDECAR_TAG = "scrub"
