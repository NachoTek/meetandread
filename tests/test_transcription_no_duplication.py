"""Test that live transcription re-transcription does not duplicate words.

Reproduces and verifies the fix for GitHub Issue #2:
Live transcription emits ALL segments on each re-transcription pass
(sliding window overlap). Without dedup, transcript_store.add_words()
appends the same text multiple times.

The fix uses a two-buffer approach: _words (permanent) + _live_phrase_words
(replaced on each pass). On phrase_start, the live buffer is committed and
a new one starts. On is_final, the live buffer is committed. Re-transcription
simply replaces the live buffer.
"""
import pytest
from unittest.mock import MagicMock
from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.transcription.accumulating_processor import SegmentResult


class TestTranscriptionNoDuplication:
    """Verify re-transcription does not duplicate text in transcript store."""

    def _make_word(self, text: str, start: float, end: float) -> Word:
        return Word(text=text, start_time=start, end_time=end,
                    confidence=80, speaker_id=None)

    def _make_result(self, text: str, start: float, end: float,
                     segment_index: int = 0, is_final: bool = False,
                     phrase_start: bool = False) -> SegmentResult:
        return SegmentResult(
            text=text,
            confidence=80,
            start_time=start,
            end_time=end,
            segment_index=segment_index,
            is_final=is_final,
            phrase_start=phrase_start,
        )

    # --- TranscriptStore live buffer tests ---

    def test_live_buffer_replaced_not_appended(self):
        """set_live_phrase_words replaces, doesn't append."""
        store = TranscriptStore()

        # First pass
        store.set_live_phrase_words([
            self._make_word("hello", 0.0, 0.5),
            self._make_word("world", 0.5, 1.0),
        ])
        assert store.get_word_count() == 2

        # Second pass (re-transcription) — replaces, not appends
        store.set_live_phrase_words([
            self._make_word("hello", 0.0, 0.5),
            self._make_word("world", 0.5, 1.0),
            self._make_word("foo", 1.0, 1.5),
        ])
        assert store.get_word_count() == 3  # not 5

    def test_commit_moves_live_to_permanent(self):
        """commit_live_phrase moves buffer into permanent storage."""
        store = TranscriptStore()
        store.set_live_phrase_words([self._make_word("hello", 0.0, 0.5)])
        assert store.get_word_count() == 1

        store.commit_live_phrase()
        # Same count, but now in _words not _live
        assert store.get_word_count() == 1
        assert len(store._words) == 1
        assert len(store._live_phrase_words) == 0

    def test_multiple_passes_no_duplication(self):
        """5 re-transcription passes of the same audio = same word count."""
        store = TranscriptStore()
        for _ in range(5):
            store.set_live_phrase_words([
                self._make_word(f"word{j}", float(j), float(j + 1))
                for j in range(10)
            ])
        assert store.get_word_count() == 10

    def test_long_phrase_single_buffer(self):
        """Long phrases use a single live buffer, replaced each pass.

        The 12s sliding window only sees the tail, but Whisper returns
        segments for the full window. The live buffer always contains
        ONLY the latest pass's words — no accumulation needed because
        each pass re-transcribes the visible window.
        """
        store = TranscriptStore()

        # Pass 1: window 0-12s → 12 words
        store.set_live_phrase_words([
            self._make_word(f"w{j}", float(j), float(j + 1))
            for j in range(12)
        ])
        assert store.get_word_count() == 12

        # Pass 5: window 10-22s → 12 words (replaces, not appends)
        store.set_live_phrase_words([
            self._make_word(f"w{j}", float(j), float(j + 1))
            for j in range(10, 22)
        ])
        assert store.get_word_count() == 12

        # Commit when phrase finalizes
        store.commit_live_phrase()
        assert store.get_word_count() == 12
        assert len(store._words) == 12

    def test_clear_resets_both_buffers(self):
        """clear() resets both permanent and live buffers."""
        store = TranscriptStore()
        store.set_live_phrase_words([self._make_word("a", 0.0, 0.5)])
        store.commit_live_phrase()
        store.set_live_phrase_words([self._make_word("b", 1.0, 1.5)])
        assert store.get_word_count() == 2

        store.clear()
        assert store.get_word_count() == 0
        assert len(store._live_phrase_words) == 0

    def test_empty_live_words_does_not_corrupt(self):
        """Setting empty live words clears the buffer without error."""
        store = TranscriptStore()
        store.set_live_phrase_words([self._make_word("a", 0.0, 0.5)])
        assert store.get_word_count() == 1

        store.set_live_phrase_words([])
        assert store.get_word_count() == 0

    # --- Controller integration tests ---

    def test_controller_no_duplication_on_retranscription(self):
        """Controller does not duplicate on re-transcription passes."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        # New phrase starts
        ctrl._on_phrase_result(
            self._make_result("Hello world", 0.0, 2.0, 0, phrase_start=True))
        assert ctrl._transcript_store.get_word_count() == 2

        # Re-transcription (same audio, refined text)
        ctrl._on_phrase_result(
            self._make_result("Hello world foo", 0.0, 3.0, 0))
        # Live buffer replaced: 3 words, not 5
        assert ctrl._transcript_store.get_word_count() == 3

        # Another re-transcription
        ctrl._on_phrase_result(
            self._make_result("Hello world foo bar", 0.0, 4.0, 0))
        # Still replaced: 4 words, not 7
        assert ctrl._transcript_store.get_word_count() == 4

        # Verify the actual words
        words = ctrl._transcript_store.get_all_words()
        texts = [w.text for w in words]
        assert texts == ["Hello", "world", "foo", "bar"]

    def test_controller_multiple_phrases(self):
        """Multiple phrases commit separately without cross-contamination."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        # Phrase 1: new phrase
        ctrl._on_phrase_result(
            self._make_result("First phrase", 0.0, 2.0, 0, phrase_start=True))
        assert ctrl._transcript_store.get_word_count() == 2

        # Phrase 1: finalized
        ctrl._on_phrase_result(
            self._make_result("First phrase", 0.0, 2.0, 0, is_final=True))
        # Committed: 2 permanent words
        assert ctrl._transcript_store.get_word_count() == 2
        assert len(ctrl._transcript_store._words) == 2
        assert len(ctrl._transcript_store._live_phrase_words) == 0

        # Phrase 2: new phrase (commits phrase 1, starts fresh)
        ctrl._on_phrase_result(
            self._make_result("Second phrase", 3.0, 5.0, 0, phrase_start=True))
        # 2 permanent + 2 live = 4
        assert ctrl._transcript_store.get_word_count() == 4

        # Phrase 2: re-transcription
        ctrl._on_phrase_result(
            self._make_result("Second phrase extended", 3.0, 6.0, 0))
        # 2 permanent + 3 live = 5
        assert ctrl._transcript_store.get_word_count() == 5

        # Phrase 2: finalized
        ctrl._on_phrase_result(
            self._make_result("Second phrase extended", 3.0, 6.0, 0,
                              is_final=True))
        # All committed: 2 + 3 = 5 permanent
        assert ctrl._transcript_store.get_word_count() == 5
        assert len(ctrl._transcript_store._words) == 5
        assert len(ctrl._transcript_store._live_phrase_words) == 0

        # Verify phrase 1 words intact
        words = ctrl._transcript_store.get_all_words()
        assert words[0].text == "First"
        assert words[1].text == "phrase"

    def test_controller_empty_result_no_corruption(self):
        """Empty text results don't corrupt the store."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        ctrl._on_phrase_result(
            self._make_result("Hello", 0.0, 1.0, 0, phrase_start=True))
        assert ctrl._transcript_store.get_word_count() == 1

        # Empty result — no words created, store unchanged
        ctrl._on_phrase_result(self._make_result("", 0.0, 0.0, 0))
        assert ctrl._transcript_store.get_word_count() == 1
