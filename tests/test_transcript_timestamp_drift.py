"""Tests for transcript word-timestamp production (issue #21).

The playback highlighter advances the word highlight using per-word
``[start_time, end_time)`` intervals from the Transcript Footer and already
holds the last spoken word during gaps (see ``test_history_transcript_highlighting.py``).
For that gap-hold to work during silence, the per-word timestamps fed into the
footer must reflect the **real spoken timing** produced by Whisper — they must
NOT be linearly stretched to fill the full audio duration. Stretching embeds
silence into the word spans, so the highlight marches ahead of the audio
during pauses.

These tests pin the contract for the two producers that build the final
playback transcript:

* ``PostProcessingQueue._create_post_processed_transcript``  (final transcript)
* ``RetranscribeRunner._create_transcript_from_segments``     (re-transcribe sidecar)

They must preserve Whisper's real segment timestamps and keep inter-segment
gaps intact. (Per-word timestamps are unavailable from the ``pywhispercpp``
binding, which exposes only segment-level ``t0``/``t1``; within a segment the
real ``[start, end]`` is therefore divided evenly across its words. That
intra-segment spacing is an approximation, but the segment boundaries — and
thus the silences between them — stay real, which is what lets the highlighter
hold during pauses.)
"""

import pytest

from meetandread.transcription.engine import TranscriptionSegment, WordInfo
from meetandread.transcription.post_processor import PostProcessingQueue
from meetandread.transcription.retranscribe import RetranscribeRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(text: str, start: float, end: float, confidence: int = 90) -> TranscriptionSegment:
    """Build a Whisper-style segment: one multi-word WordInfo spanning [start, end]."""
    return TranscriptionSegment(
        text=text,
        confidence=confidence,
        start=start,
        end=end,
        words=[WordInfo(text=text, start=start, end=end, confidence=confidence)],
    )


def _post_process(segments):
    """Call the instance method without running PostProcessingQueue.__init__.

    ``_create_post_processed_transcript`` does not read any instance state, so
    a bare ``__new__`` instance is sufficient and avoids spinning up threads.
    """
    queue = PostProcessingQueue.__new__(PostProcessingQueue)
    return queue._create_post_processed_transcript(segments)


def _retranscribe(segments):
    return RetranscribeRunner._create_transcript_from_segments(segments)


# ---------------------------------------------------------------------------
# Timestamps must be real, not stretched to fill a duration (#21)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_post_process, _retranscribe])
class TestTimestampsAreRealNotStretched:
    """Per-word timestamps come verbatim from Whisper's segment ``[start, end]``.

    They must never be linearly stretched to fill a longer duration. Previously
    the producers multiplied every timestamp by ``audio_duration / whisper_end``
    when Whisper under-reported the duration; that embedded silence into the
    word spans and made the highlight drift during pauses (issue #21). These
    producers no longer accept a duration input at all, so there is nothing to
    stretch by — the assertions below pin that the segment's real end is kept.
    """

    def test_multi_word_segment_end_is_real_segment_end(self, build):
        seg = _seg("hello world", start=0.0, end=2.0)
        words = build([seg]).get_all_words()

        assert [w.text for w in words] == ["hello", "world"]
        assert words[0].start_time == pytest.approx(0.0)
        # Last word ends at the real segment end — NOT stretched toward any
        # longer "audio duration".
        assert words[-1].end_time == pytest.approx(2.0), (
            "Timestamps must reflect Whisper's real segment end, not a "
            "duration-based stretch (issue #21)."
        )

    def test_output_span_matches_union_of_segments(self, build):
        # Word timing must span exactly the union of the input segments' real
        # times — nothing padded before the first start or after the last end.
        segments = [
            _seg("a b", start=1.0, end=2.0),
            _seg("c d", start=5.0, end=7.0),
        ]
        words = build(segments).get_all_words()

        assert len(words) == 4
        assert min(w.start_time for w in words) == pytest.approx(1.0)
        assert max(w.end_time for w in words) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Inter-segment gaps must be preserved (so gap-hold can fire)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_post_process, _retranscribe])
class TestInterSegmentGapsPreserved:
    """Real silences between segments must remain gaps, not be smeared together."""

    def test_gap_between_segments_kept(self, build):
        # Segment A ends at 1.0s, segment B starts at 3.0s -> 2s gap.
        segments = [
            _seg("hi there", start=0.0, end=1.0),
            _seg("you all", start=3.0, end=4.0),
        ]
        words = build(segments).get_all_words()

        assert [w.text for w in words] == ["hi", "there", "you", "all"]
        # End of segment A's last word == real segment A end.
        assert words[1].end_time == pytest.approx(1.0)
        # Start of segment B's first word == real segment B start (the gap).
        assert words[2].start_time == pytest.approx(3.0)
        # The gap between them is preserved (2s), not collapsed.
        assert words[2].start_time - words[1].end_time == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Within-segment even distribution is the documented approximation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_post_process, _retranscribe])
class TestWithinSegmentDistribution:
    """The segment's real ``[start, end]`` is divided evenly across its words.

    This pins that behaviour: the segment's own boundaries stay real (so
    inter-segment gaps survive), even though intra-word spacing is an
    approximation (the binding exposes no per-word timestamps).
    """

    def test_even_split_within_segment(self, build):
        seg = _seg("one two three four", start=10.0, end=14.0)
        words = build([seg]).get_all_words()

        assert [w.text for w in words] == ["one", "two", "three", "four"]
        # 4 words over 4s -> 1s each, anchored at the real segment start.
        assert [w.start_time for w in words] == pytest.approx([10.0, 11.0, 12.0, 13.0])
        assert words[-1].end_time == pytest.approx(14.0)

    def test_single_word_uses_real_segment_bounds(self, build):
        seg = _seg("solitary", start=5.0, end=6.5)
        words = build([seg]).get_all_words()
        assert words[0].start_time == pytest.approx(5.0)
        assert words[0].end_time == pytest.approx(6.5)
