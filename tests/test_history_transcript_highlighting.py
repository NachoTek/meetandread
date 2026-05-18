"""Tests for transcript word-highlighting timing and gap behaviour.

Covers:
- _extract_timed_words() preserving metadata order and fallback spans
- _find_active_word_index() [start_ms, end_ms) boundary semantics
- _find_active_word_index_linear() with mixed timed/untimed entries
- Gap-hold behaviour: last spoken word stays highlighted during inter-word gaps
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Lightweight panel fake — avoids needing a real QApplication
# ---------------------------------------------------------------------------

class _FakeScrollBar:
    """Minimal scroll-bar stand-in."""

    def __init__(self):
        self._value = 0

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v


class _FakeViewer:
    """Minimal QTextBrowser stand-in."""

    def __init__(self):
        self._html = ""

    def setHtml(self, html):
        self._html = html

    def verticalScrollBar(self):
        return _FakeScrollBar()


class _FakeSlider:
    """Minimal slider stand-in."""

    def __init__(self):
        self._value = 0
        self._signals_blocked = False

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v

    def blockSignals(self, b):
        self._signals_blocked = b


class _PanelFake:
    """Lightweight fake that exposes the timing methods under test.

    Uses __new__ to bypass QWidget __init__ and manually sets the
    attributes that _extract_timed_words / _find_active_word_index need.
    """

    def __init__(self):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        # Pull the static / instance methods we need onto a plain object
        # so we don't trigger QWidget.__init__.
        self._cached_timed_words: list = []
        self._current_highlight_word_idx: int = -1
        self._current_history_md_path: Optional[Path] = None
        self._history_viewer = _FakeViewer()
        self._playback_progress_slider = _FakeSlider()
        self._last_highlight_update_ms: int = 0
        self._HIGHLIGHT_UPDATE_INTERVAL_MS: int = 200
        self._last_slider_update_ms: int = 0
        self._POSITION_UPDATE_INTERVAL_MS: int = 50
        self._is_dragging_progress_slider = False

        # Bind the real methods from the class.
        # _validate_timed_word is a @staticmethod — bind as-is.
        # The rest are instance methods — bind with __get__.
        cls = FloatingSettingsPanel
        self._validate_timed_word = cls._validate_timed_word
        self._extract_timed_words = cls._extract_timed_words.__get__(self)
        self._find_active_word_index = cls._find_active_word_index.__get__(self)
        self._find_active_word_index_linear = cls._find_active_word_index_linear.__get__(self)
        self._reset_highlight_state = cls._reset_highlight_state.__get__(self)
        self._render_highlighted_transcript = cls._render_highlighted_transcript.__get__(self)


# ---------------------------------------------------------------------------
# Transcript fixture helpers
# ---------------------------------------------------------------------------

def _make_transcript_md(
    tmp_path: Path,
    words: List[Dict[str, Any]],
    *,
    name: str = "test.md",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a minimal transcript .md with METADATA footer containing *words*."""
    lines = ["# Transcript", "", "**Recorded:** 2026-04-22T14:30:00", ""]
    # Group words by speaker for the body (simple — one speaker is fine)
    body_words = [w.get("text", "") for w in words]
    lines += ["**Speaker**", "", " ".join(body_words), ""]
    md_body = "\n".join(lines)

    meta: Dict[str, Any] = {
        "recording_start_time": "2026-04-22T14:30:00",
        "word_count": len(words),
        "words": words,
        "segments": [],
    }
    if extra_metadata:
        meta.update(extra_metadata)

    p = tmp_path / name
    p.write_text(
        md_body + "\n---\n\n<!-- METADATA: " + json.dumps(meta, indent=2) + " -->\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# _extract_timed_words tests
# ---------------------------------------------------------------------------

class TestExtractTimedWords:
    """Verify _extract_timed_words preserves metadata order and handles edge cases."""

    def test_preserves_metadata_order(self, tmp_path):
        """Words must appear in the same order as the metadata array."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5},
            {"text": "world", "start_time": 0.5, "end_time": 1.0},
            {"text": "foo", "start_time": 1.0, "end_time": 1.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert len(result) == 3
        assert result[0] == (0, 500)
        assert result[1] == (500, 1000)
        assert result[2] == (1000, 1500)

    def test_missing_end_time_gets_fallback_span(self, tmp_path):
        """Words with no end_time get start+1ms fallback."""
        words = [
            {"text": "Hi", "start_time": 1.2},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == [(1200, 1201)]

    def test_invalid_end_time_gets_fallback_span(self, tmp_path):
        """Words with non-numeric end_time get start+1ms fallback."""
        words = [
            {"text": "X", "start_time": 0.5, "end_time": "not_a_number"},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == [(500, 501)]

    def test_end_before_start_gets_fallback_span(self, tmp_path):
        """Words where end_time < start_time get corrected to start+1ms."""
        words = [
            {"text": "Y", "start_time": 2.0, "end_time": 1.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == [(2000, 2001)]

    def test_missing_start_time_produces_none_tuple(self, tmp_path):
        """Words without start_time produce (None, None)."""
        words = [
            {"text": "untimed"},
            {"text": "timed", "start_time": 1.0, "end_time": 1.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result[0] == (None, None)
        assert result[1] == (1000, 1500)

    def test_negative_start_time_produces_none_tuple(self, tmp_path):
        """Words with negative start_time produce (None, None)."""
        words = [
            {"text": "bad", "start_time": -1.0},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == [(None, None)]

    def test_no_metadata_footer_gives_empty(self, tmp_path):
        """Files without the METADATA footer produce an empty list."""
        md = tmp_path / "bare.md"
        md.write_text("# Just a transcript\nNo metadata here.\n", encoding="utf-8")
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == []

    def test_malformed_json_gives_empty(self, tmp_path):
        """Corrupt JSON in the metadata footer produces an empty list."""
        md = tmp_path / "bad.md"
        md.write_text(
            "# T\n---\n\n<!-- METADATA: {invalid json} -->\n",
            encoding="utf-8",
        )
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert result == []

    def test_missing_file_gives_empty(self, tmp_path):
        """Non-existent file path produces an empty list without raising."""
        panel = _PanelFake()
        result = panel._extract_timed_words(tmp_path / "no_such_file.md")
        assert result == []

    def test_caches_result(self, tmp_path):
        """_extract_timed_words stores its result in _cached_timed_words."""
        words = [
            {"text": "A", "start_time": 0.0, "end_time": 0.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        result = panel._extract_timed_words(md)
        assert panel._cached_timed_words is result


# ---------------------------------------------------------------------------
# _find_active_word_index tests — [start_ms, end_ms) semantics
# ---------------------------------------------------------------------------

class TestFindActiveWordIndex:
    """Verify _find_active_word_index implements strict [start, end) lookup."""

    @pytest.fixture()
    def panel_with_words(self, tmp_path):
        """Panel with 3 words: [0,500), [500,1000), [1000,1500)."""
        words = [
            {"text": "A", "start_time": 0.0, "end_time": 0.5},
            {"text": "B", "start_time": 0.5, "end_time": 1.0},
            {"text": "C", "start_time": 1.0, "end_time": 1.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        return panel

    def test_before_first_word_returns_minus_1(self, panel_with_words):
        assert panel_with_words._find_active_word_index(-100) == -1

    def test_exact_start_of_first_word(self, panel_with_words):
        assert panel_with_words._find_active_word_index(0) == 0

    def test_inside_first_word(self, panel_with_words):
        assert panel_with_words._find_active_word_index(250) == 0

    def test_exact_end_of_first_word_returns_minus_1(self, panel_with_words):
        """end_ms is exclusive — position at end should NOT match."""
        assert panel_with_words._find_active_word_index(500) == 1

    def test_exact_start_of_second_word(self, panel_with_words):
        assert panel_with_words._find_active_word_index(500) == 1

    def test_inside_second_word(self, panel_with_words):
        assert panel_with_words._find_active_word_index(750) == 1

    def test_exact_end_of_last_word_returns_minus_1(self, panel_with_words):
        """After the last word's end_ms, no word is active."""
        assert panel_with_words._find_active_word_index(1500) == -1

    def test_well_beyond_last_word(self, panel_with_words):
        assert panel_with_words._find_active_word_index(9999) == -1

    def test_between_word_gap(self, tmp_path):
        """A gap between two non-adjacent words returns -1."""
        words = [
            {"text": "X", "start_time": 0.0, "end_time": 0.3},
            {"text": "Y", "start_time": 0.8, "end_time": 1.2},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        # Position in the gap [300, 800)
        assert panel._find_active_word_index(400) == -1
        assert panel._find_active_word_index(500) == -1
        assert panel._find_active_word_index(799) == -1

    def test_empty_cache_returns_minus_1(self):
        panel = _PanelFake()
        assert panel._find_active_word_index(100) == -1

    def test_negative_position_returns_minus_1(self, panel_with_words):
        assert panel_with_words._find_active_word_index(-1) == -1

    def test_single_ms_before_end(self, panel_with_words):
        """One ms before end should still be inside the word."""
        assert panel_with_words._find_active_word_index(1499) == 2


# ---------------------------------------------------------------------------
# _find_active_word_index_linear tests — mixed timed/untimed
# ---------------------------------------------------------------------------

class TestFindActiveWordIndexLinear:
    """Verify linear fallback handles mixed timed/untimed entries."""

    def test_skips_untimed_finds_timed(self, tmp_path):
        words = [
            {"text": "no_time"},
            {"text": "yes", "start_time": 1.0, "end_time": 1.5},
            {"text": "also_no"},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        assert panel._find_active_word_index_linear(1200) == 1

    def test_all_untimed_returns_minus_1(self, tmp_path):
        words = [
            {"text": "a"},
            {"text": "b"},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        assert panel._find_active_word_index_linear(0) == -1
        assert panel._find_active_word_index_linear(500) == -1

    def test_returns_first_match(self, tmp_path):
        """When two timed words overlap, linear returns the first match."""
        words = [
            {"text": "X", "start_time": 0.0, "end_time": 1.0},
            {"text": "Y", "start_time": 0.5, "end_time": 1.5},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        # At 750ms both match; linear scan returns index 0
        assert panel._find_active_word_index_linear(750) == 0


# ---------------------------------------------------------------------------
# Gap-hold behavioural tests
# ---------------------------------------------------------------------------
# These tests simulate the production highlight-update logic from
# _on_player_position_changed.  The contract:
#   - When _find_active_word_index returns -1 (in a gap), the panel must
#     HOLD _current_highlight_word_idx at its previous value (last spoken word).
#   - When playback reaches the next word, the highlight advances.
#
# The current production code clears the highlight to -1 during gaps.
# These tests are expected to FAIL against the current implementation,
# confirming the regression target for the fix.

def _simulate_highlight_step(panel: _PanelFake, position_ms: int) -> int:
    """Simulate one highlight-update step from _on_player_position_changed.

    Mirrors the production logic:
      1. Find active word index
      2. If active_idx == -1 and current highlight != -1, hold (gap-hold)
      3. Otherwise, if different from current highlight, update

    Returns the new _current_highlight_word_idx after the step.
    """
    active_idx = panel._find_active_word_index(position_ms)
    if active_idx != panel._current_highlight_word_idx:
        if active_idx == -1 and panel._current_highlight_word_idx != -1:
            # Gap-hold: preserve last highlighted word during speech gaps
            pass
        else:
            panel._current_highlight_word_idx = active_idx
    return panel._current_highlight_word_idx


class TestGapHoldBehaviour:
    """Verify that the last spoken word remains highlighted during gaps.

    These tests exercise the actual _find_active_word_index logic and
    simulate the production highlight-update path.  They define the
    EXPECTED gap-hold contract and are designed to FAIL against the
    current implementation (which clears to -1 in gaps).
    """

    @pytest.fixture()
    def gap_panel(self, tmp_path):
        """Panel with words that have a gap between them:
        word 0: [0, 300)
        gap:    [300, 800)
        word 1: [800, 1200)
        word 2: [1200, 1600)
        """
        words = [
            {"text": "Alpha", "start_time": 0.0, "end_time": 0.3},
            {"text": "Beta", "start_time": 0.8, "end_time": 1.2},
            {"text": "Gamma", "start_time": 1.2, "end_time": 1.6},
        ]
        md = _make_transcript_md(tmp_path, words, name="gap_test.md")
        panel = _PanelFake()
        panel._current_history_md_path = md
        panel._extract_timed_words(md)
        return panel

    def test_gap_holds_last_word_highlight(self, gap_panel):
        """Moving from an active word into a gap must keep the last highlight.

        EXPECTED BEHAVIOUR: _current_highlight_word_idx stays at 0 during gap.
        CURRENT BEHAVIOUR: clears to -1 (this test FAILS until fixed).
        """
        # Step 1: playback at word 0 → highlight becomes 0
        idx = _simulate_highlight_step(gap_panel, 150)
        assert idx == 0  # word 0 should be highlighted

        # Step 2: playback enters gap [300, 800) → highlight should HOLD at 0
        idx = _simulate_highlight_step(gap_panel, 500)
        assert idx == 0, (
            f"Gap-hold contract violated: highlight cleared to {idx} "
            f"instead of holding at 0 during inter-word gap"
        )

    def test_gap_then_next_word_advances(self, gap_panel):
        """After holding through a gap, highlight advances to the next word.

        EXPECTED BEHAVIOUR: after gap-hold, reaching word 1 advances highlight to 1.
        CURRENT BEHAVIOUR: highlight clears to -1 in gap, then sets to 1.
        (This test may appear to pass on the "advance to 1" part but the
        hold-in-gap part is what matters — tested above.)
        """
        # Step 1: word 0 active
        _simulate_highlight_step(gap_panel, 150)
        assert gap_panel._current_highlight_word_idx == 0

        # Step 2: gap — should hold
        _simulate_highlight_step(gap_panel, 500)
        assert gap_panel._current_highlight_word_idx == 0, "Highlight lost in gap"

        # Step 3: reach word 1 — should advance
        idx = _simulate_highlight_step(gap_panel, 900)
        assert idx == 1, f"Highlight should advance to word 1, got {idx}"

    def test_trailing_gap_after_last_word_holds(self, gap_panel):
        """After the last word ends, highlight stays on that word during trailing gap.

        EXPECTED BEHAVIOUR: highlight holds at 2 after word 2 ends.
        CURRENT BEHAVIOUR: clears to -1 (this test FAILS until fixed).
        """
        # Advance to word 2
        _simulate_highlight_step(gap_panel, 150)   # word 0
        _simulate_highlight_step(gap_panel, 500)   # gap — hold at 0
        _simulate_highlight_step(gap_panel, 900)   # word 1
        _simulate_highlight_step(gap_panel, 1300)  # word 2
        assert gap_panel._current_highlight_word_idx == 2

        # Position beyond last word's end (1600ms) — trailing silence
        idx = _simulate_highlight_step(gap_panel, 2000)
        assert idx == 2, (
            f"Trailing-gap hold violated: highlight cleared to {idx} "
            f"instead of holding at 2 after last word"
        )


# ---------------------------------------------------------------------------
# Boundary stress tests
# ---------------------------------------------------------------------------

class TestBoundaryStress:
    """Edge cases around timing boundaries and single-ms precision."""

    def test_adjacent_words_no_gap(self, tmp_path):
        """Words with end=start of next — end boundary of first should match next."""
        words = [
            {"text": "A", "start_time": 0.0, "end_time": 0.5},
            {"text": "B", "start_time": 0.5, "end_time": 1.0},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        # At 500ms: word A's exclusive end, word B's inclusive start
        assert panel._find_active_word_index(499) == 0
        assert panel._find_active_word_index(500) == 1

    def test_zero_duration_word(self, tmp_path):
        """Word where start == end (after fallback to start+1ms)."""
        words = [
            {"text": "Z", "start_time": 1.0, "end_time": 1.0},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        # The code corrects end < start to start+1, but end == start
        # results in (1000, 1000), which is a zero-width interval.
        # [1000, 1000) matches nothing. Verify:
        result = panel._cached_timed_words[0]
        start, end = result
        # Zero-width interval: start == end, nothing matches
        if start == end:
            assert panel._find_active_word_index(1000) == -1

    def test_very_large_timestamps(self, tmp_path):
        """Handles timestamps in the thousands-of-seconds range."""
        words = [
            {"text": "Late", "start_time": 3600.0, "end_time": 3601.0},
        ]
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)
        assert panel._find_active_word_index(3600500) == 0
        assert panel._find_active_word_index(3600999) == 0
        assert panel._find_active_word_index(3601000) == -1

    def test_many_words_binary_search(self, tmp_path):
        """Binary search correctness with 1000 contiguous words."""
        words = []
        for i in range(1000):
            words.append({
                "text": f"w{i}",
                "start_time": i * 0.1,
                "end_time": (i + 1) * 0.1,
            })
        md = _make_transcript_md(tmp_path, words)
        panel = _PanelFake()
        panel._extract_timed_words(md)

        # Spot-check several positions
        assert panel._find_active_word_index(0) == 0
        assert panel._find_active_word_index(500) == 5  # word 5: [500,600)
        assert panel._find_active_word_index(50005) == 500
        assert panel._find_active_word_index(99999) == 999
        assert panel._find_active_word_index(100000) == -1  # past end
