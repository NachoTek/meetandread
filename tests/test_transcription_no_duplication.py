"""Test that live transcription re-transcription does not duplicate words.

Reproduces and verifies the fix for GitHub Issue #2:
Live transcription emits ALL segments on each re-transcription pass
(sliding window overlap). Without dedup, transcript_store.add_words()
appends the same text multiple times.

The fix uses replace_current_phrase_words() to replace the current
phrase on re-transcription instead of blindly appending.
"""
import pytest
from unittest.mock import MagicMock, patch
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

    # --- TranscriptStore phrase boundary tests ---

    def test_mark_phrase_boundary_splits_store(self):
        """mark_phrase_boundary freezes words before the boundary."""
        store = TranscriptStore()
        store.add_words([self._make_word("hello", 0.0, 0.5)])
        store.add_words([self._make_word("world", 0.5, 1.0)])
        assert store.get_word_count() == 2

        store.mark_phrase_boundary()
        # Words before boundary are preserved
        assert store.get_word_count() == 2

    def test_replace_current_phrase_words_replaces_after_boundary(self):
        """replace_current_phrase_words keeps committed words, replaces live."""
        store = TranscriptStore()
        store.add_words([self._make_word("frozen", 0.0, 0.5)])
        store.mark_phrase_boundary()

        # First re-transcription pass (window covers 1.0-2.0)
        store.replace_current_phrase_words([
            self._make_word("hello", 1.0, 1.5),
            self._make_word("world", 1.5, 2.0),
        ])
        words = store.get_all_words()
        assert len(words) == 3
        assert words[0].text == "frozen"
        assert words[1].text == "hello"

        # Second re-transcription pass (window slides to 2.0-3.0)
        # "hello" (1.0-1.5) ends before 2.0, committed
        # "world" (1.5-2.0) ends at 2.0, also committed (<=)
        store.replace_current_phrase_words([
            self._make_word("world", 2.0, 2.5),
            self._make_word("foo", 2.5, 3.0),
        ])
        words = store.get_all_words()
        # frozen + hello + world (all committed) + new world + foo = 5
        assert len(words) == 5
        assert words[0].text == "frozen"
        assert words[1].text == "hello"
        assert words[2].text == "world"
        assert words[3].text == "world"
        assert words[4].text == "foo"

    def test_replace_without_duplication(self):
        """Multiple re-transcription passes of same window do not grow words."""
        store = TranscriptStore()
        store.mark_phrase_boundary()

        for i in range(5):
            store.replace_current_phrase_words([
                self._make_word(f"word{j}", float(j), float(j + 1))
                for j in range(10)
            ])

        # Same time range each time: should stay at 10
        assert store.get_word_count() == 10

    def test_long_phrase_accumulates_beyond_window(self):
        """Words from before the sliding window are preserved in a long phrase.

        Simulates a 30-second phrase with a 12-second window sliding by 2s:
        - Pass 1: window 0-12s, emits words for 0-12s
        - Pass 2: window 2-14s, emits words for 2-14s
        - Pass 5: window 10-22s, emits words for 10-22s
        - Pass 9: window 18-30s, emits words for 18-30s

        Words from 0-18s should be preserved despite falling out of the window.
        """
        store = TranscriptStore()
        store.mark_phrase_boundary()

        # Pass 1: window 0-12s
        pass1_words = [self._make_word(f"w{j}", float(j), float(j + 1))
                       for j in range(0, 12)]
        store.replace_current_phrase_words(pass1_words)
        assert store.get_word_count() == 12

        # Pass 2: window 2-14s — words 0-2 are committed (end <= 2.0)
        pass2_words = [self._make_word(f"w{j}", float(j), float(j + 1))
                       for j in range(2, 14)]
        store.replace_current_phrase_words(pass2_words)
        assert store.get_word_count() == 14  # 0-13

        # Pass 5: window 10-22s — words 0-10 are committed (end <= 10.0)
        pass5_words = [self._make_word(f"w{j}", float(j), float(j + 1))
                       for j in range(10, 22)]
        store.replace_current_phrase_words(pass5_words)
        assert store.get_word_count() == 22  # 0-21

        # Pass 9: window 18-30s — words 0-18 are committed (end <= 18.0)
        pass9_words = [self._make_word(f"w{j}", float(j), float(j + 1))
                       for j in range(18, 30)]
        store.replace_current_phrase_words(pass9_words)
        assert store.get_word_count() == 30  # 0-29

        # Verify all words are present and in order
        words = store.get_all_words()
        texts = [w.text for w in words]
        assert texts == [f"w{j}" for j in range(30)]

    def test_clear_resets_boundary(self):
        """clear() resets the phrase boundary index."""
        store = TranscriptStore()
        store.add_words([self._make_word("a", 0.0, 0.5)])
        store.mark_phrase_boundary()
        store.replace_current_phrase_words([self._make_word("b", 1.0, 1.5)])
        assert store.get_word_count() == 2

        store.clear()
        assert store.get_word_count() == 0
        # After clear, replace should still work (no stale boundary)
        store.mark_phrase_boundary()
        store.replace_current_phrase_words([self._make_word("c", 0.0, 0.5)])
        assert store.get_word_count() == 1

    # --- Controller integration tests ---

    def test_controller_no_duplication_on_retranscription(self):
        """Controller's _on_phrase_result does not duplicate on re-transcription."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        # Simulate: new phrase starts
        result1 = self._make_result("Hello world", 0.0, 2.0, 0,
                                    phrase_start=True)
        ctrl._on_phrase_result(result1)
        assert ctrl._transcript_store.get_word_count() == 2

        # Simulate: re-transcription of same audio (no phrase_start, no is_final)
        result2 = self._make_result("Hello world foo", 0.0, 3.0, 0)
        ctrl._on_phrase_result(result2)
        # Should REPLACE, not append: 3 words, not 5
        assert ctrl._transcript_store.get_word_count() == 3

        # Simulate: another re-transcription with refined text
        result3 = self._make_result("Hello world foo bar", 0.0, 4.0, 0)
        ctrl._on_phrase_result(result3)
        # Should REPLACE again: 4 words, not 7
        assert ctrl._transcript_store.get_word_count() == 4

        # Verify the actual words
        words = ctrl._transcript_store.get_all_words()
        texts = [w.text for w in words]
        assert texts == ["Hello", "world", "foo", "bar"]

    def test_controller_multiple_phrases_no_cross_contamination(self):
        """Words from completed phrases are not replaced by new phrases."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        # Phrase 1: new phrase (window 0-2s)
        ctrl._on_phrase_result(
            self._make_result("First phrase", 0.0, 2.0, 0, phrase_start=True))
        assert ctrl._transcript_store.get_word_count() == 2

        # Phrase 1: re-transcription (window 1-3s)
        # "First" (0.0-1.0) is committed (end=1.0 <= new_start=1.0)
        # "phrase" (1.0-2.0) is live (end=2.0 > 1.0), gets replaced
        # New words: "First"(1.0-1.67), "phrase"(1.67-2.33), "refined"(2.33-3.0)
        ctrl._on_phrase_result(
            self._make_result("First phrase refined", 1.0, 3.0, 0))
        # committed "First" + new "First" "phrase" "refined" = 4
        assert ctrl._transcript_store.get_word_count() == 4

        # Phrase 1: finalized (same window 1-3s) — replaces same range
        ctrl._on_phrase_result(
            self._make_result("First phrase finalized", 1.0, 3.0, 0,
                              is_final=True))
        # Same timing as previous, so same committed/live split = 4
        assert ctrl._transcript_store.get_word_count() == 4

        # Phrase 2: new phrase (marks new boundary, window 5-7s)
        ctrl._on_phrase_result(
            self._make_result("Second phrase", 5.0, 7.0, 0, phrase_start=True))
        # All phrase 1 words frozen at boundary (4), + 2 new = 6
        assert ctrl._transcript_store.get_word_count() == 6

        # Phrase 2: re-transcription should not touch phrase 1
        ctrl._on_phrase_result(
            self._make_result("Second phrase more", 6.0, 8.0, 0))
        # "Second"(5.0-6.0) committed, "phrase"(6.0-7.0) live, new 3 words
        # = 4 (phrase 1) + 1 (committed "Second") + 3 (new) = 8
        assert ctrl._transcript_store.get_word_count() == 8

        # Verify phrase 1 words are intact
        words = ctrl._transcript_store.get_all_words()
        phrase1_texts = [w.text for w in words[:4]]
        assert "First" in phrase1_texts

    def test_empty_result_does_not_corrupt_store(self):
        """Empty text results don't corrupt the store."""
        from meetandread.recording.controller import RecordingController

        ctrl = RecordingController.__new__(RecordingController)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._try_live_speaker_match = MagicMock(return_value=None)
        ctrl.on_phrase_result = MagicMock()

        # Add initial words
        ctrl._on_phrase_result(
            self._make_result("Hello", 0.0, 1.0, 0, phrase_start=True))
        assert ctrl._transcript_store.get_word_count() == 1

        # Empty result — no words created, store unchanged
        ctrl._on_phrase_result(self._make_result("", 0.0, 0.0, 0))
        assert ctrl._transcript_store.get_word_count() == 1
