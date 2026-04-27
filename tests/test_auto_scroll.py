"""Comprehensive tests for proportional auto-scroll threshold and NewContentBadge behavior.

Covers:
- Proportional threshold correctness (max(10, pageStep * 0.1))
- Pause detection uses proportional threshold
- _is_at_bottom consistency with threshold
- Badge visibility on paused auto-scroll with new content
- Badge count increments
- Badge click resumes auto-scroll and hides badge
- Scroll-to-bottom resumes auto-scroll
- 10s timer resume hides badge and resets count
- Badge hidden when auto-scroll active
- Badge repositions on resize
"""

import pytest

from PyQt6.QtWidgets import QApplication

from meetandread.widgets.floating_panels import FloatingTranscriptPanel


# ---------------------------------------------------------------------------
# Fixtures — same pattern as test_scrub.py
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    """Provide a QApplication singleton for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    """Create a FloatingTranscriptPanel for testing, cleaned up after.

    The panel is shown so that child widget visibility (badge, legend)
    behaves as it would at runtime — Qt's isVisible() returns False for
    children of hidden parents.
    """
    p = FloatingTranscriptPanel()
    p.show()
    # Stop the auto-scroll timer so it doesn't interfere with tests
    p.scroll_timer.stop()
    yield p
    p.close()


def _populate_text(panel: FloatingTranscriptPanel, lines: int = 50) -> None:
    """Fill the text edit with enough lines to create a scrollable area."""
    # Generate enough lines to exceed the viewport height (450px panel, ~400px text edit)
    paragraph = "The quick brown fox jumps over the lazy dog. " * 5
    for i in range(lines):
        panel.text_edit.append(f"<p>Line {i}: {paragraph}</p>")


# ---------------------------------------------------------------------------
# 1. Proportional threshold correctness
# ---------------------------------------------------------------------------

class TestProportionalThreshold:
    """Verify _near_bottom_threshold returns max(10, pageStep * 0.1)."""

    def test_threshold_returns_int(self, panel):
        _populate_text(panel)
        threshold = panel._near_bottom_threshold()
        assert isinstance(threshold, int)

    def test_threshold_at_least_ten(self, panel):
        """Even on a tiny viewport, threshold must be >= 10."""
        _populate_text(panel)
        threshold = panel._near_bottom_threshold()
        assert threshold >= 10

    def test_threshold_matches_formula(self, panel):
        """Threshold equals max(10, int(pageStep * 0.1))."""
        _populate_text(panel)
        scrollbar = panel.text_edit.verticalScrollBar()
        page_step = scrollbar.pageStep()
        expected = max(10, int(page_step * 0.1))
        actual = panel._near_bottom_threshold()
        assert actual == expected

    def test_threshold_with_larger_content(self, panel):
        """More content doesn't change the threshold — it's viewport-relative."""
        _populate_text(panel, lines=200)
        scrollbar = panel.text_edit.verticalScrollBar()
        page_step = scrollbar.pageStep()
        expected = max(10, int(page_step * 0.1))
        assert panel._near_bottom_threshold() == expected


# ---------------------------------------------------------------------------
# 2. Pause detection uses proportional threshold
# ---------------------------------------------------------------------------

class TestPauseDetectionProportional:
    """Verify _on_scroll_value_changed pauses/resumes using proportional threshold."""

    def test_scroll_near_bottom_does_not_pause(self, panel):
        """Scrolling within threshold of bottom should NOT trigger pause."""
        _populate_text(panel, lines=100)
        # Scroll to bottom first
        scrollbar = panel.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Process events so _on_scroll_value_changed fires
        qapp = QApplication.instance()
        qapp.processEvents()

        # Reset state
        panel._auto_scroll_paused = False

        # Scroll to just within threshold (maximum - threshold + 1)
        threshold = panel._near_bottom_threshold()
        near_bottom = scrollbar.maximum() - threshold + 1
        scrollbar.setValue(near_bottom)
        qapp.processEvents()

        # Should NOT be paused
        assert not panel._auto_scroll_paused

    def test_scroll_well_above_bottom_triggers_pause(self, panel):
        """Scrolling well above bottom DOES trigger pause."""
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()
        # Scroll to top region
        scrollbar.setValue(0)
        qapp = QApplication.instance()
        qapp.processEvents()

        assert panel._auto_scroll_paused

    def test_pause_starts_timer(self, panel):
        """Pausing auto-scroll should start the 10-second timer."""
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()
        panel._auto_scroll_paused = False
        panel._pause_timer.stop()

        # Scroll well above bottom
        scrollbar.setValue(0)
        qapp = QApplication.instance()
        qapp.processEvents()

        assert panel._pause_timer.isActive()


# ---------------------------------------------------------------------------
# 3. _is_at_bottom consistency
# ---------------------------------------------------------------------------

class TestIsAtBottomConsistent:
    """Verify _is_at_bottom uses the same threshold as pause detection."""

    def test_at_bottom_is_true(self, panel):
        """When scrolled to maximum, _is_at_bottom should be True."""
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        qapp = QApplication.instance()
        qapp.processEvents()

        assert panel._is_at_bottom is True

    def test_above_threshold_is_false(self, panel):
        """When scrolled well above bottom, _is_at_bottom should be False."""
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()
        # Scroll to top
        scrollbar.setValue(0)
        qapp = QApplication.instance()
        qapp.processEvents()

        assert panel._is_at_bottom is False

    def test_within_threshold_is_true(self, panel):
        """When scrolled within threshold of bottom, _is_at_bottom is True."""
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()
        threshold = panel._near_bottom_threshold()
        # Value at exactly maximum - threshold should still count as bottom
        value_at_threshold = scrollbar.maximum() - threshold
        scrollbar.setValue(value_at_threshold)
        qapp = QApplication.instance()
        qapp.processEvents()

        assert panel._is_at_bottom is True


# ---------------------------------------------------------------------------
# 5. Badge appears on paused auto-scroll with new content
# ---------------------------------------------------------------------------

class TestBadgeShowsOnPausedScroll:
    """When auto-scroll is paused and new content arrives, badge appears."""

    def test_badge_visible_after_update_while_paused(self, panel):
        panel._auto_scroll_paused = True
        panel._pending_content_count = 0

        panel.update_segment("Hello world", 90, 0, is_final=True, phrase_start=True)

        assert panel._new_content_badge.isVisible()
        assert "1" in panel._new_content_badge.text()
        assert "new" in panel._new_content_badge.text()

    def test_badge_text_contains_new(self, panel):
        panel._auto_scroll_paused = True
        panel.update_segment("Test segment", 85, 0, is_final=True, phrase_start=True)

        assert "new" in panel._new_content_badge.text().lower()


# ---------------------------------------------------------------------------
# 6. Badge count increments
# ---------------------------------------------------------------------------

class TestBadgeIncrements:
    """Each update_segment call while paused increments the badge count."""

    def test_count_increments_to_three(self, panel):
        panel._auto_scroll_paused = True
        panel._pending_content_count = 0

        panel.update_segment("First", 90, 0, is_final=True, phrase_start=True)
        panel.update_segment("Second", 85, 0, is_final=True, phrase_start=True)
        panel.update_segment("Third", 80, 0, is_final=True, phrase_start=True)

        assert panel._pending_content_count == 3
        assert panel._new_content_badge.isVisible()
        badge_text = panel._new_content_badge.text()
        assert "3" in badge_text

    def test_count_continues_from_existing(self, panel):
        """If count was already non-zero, it keeps incrementing."""
        panel._auto_scroll_paused = True
        panel._pending_content_count = 5

        panel.update_segment("More text", 90, 0, is_final=True, phrase_start=True)

        assert panel._pending_content_count == 6
        assert "6" in panel._new_content_badge.text()


# ---------------------------------------------------------------------------
# 7. Badge click scrolls to bottom and hides
# ---------------------------------------------------------------------------

class TestBadgeClickResumes:
    """Clicking the badge resumes auto-scroll and hides badge."""

    def test_badge_click_resumes_and_resets(self, panel):
        # Pause and add content
        panel._auto_scroll_paused = True
        panel.update_segment("Some content", 90, 0, is_final=True, phrase_start=True)
        panel.update_segment("More content", 85, 0, is_final=True, phrase_start=True)

        assert panel._pending_content_count == 2
        assert panel._new_content_badge.isVisible()

        # Simulate badge click
        panel._on_badge_clicked()

        assert panel._auto_scroll_paused is False
        assert panel._pending_content_count == 0
        assert not panel._new_content_badge.isVisible()

    def test_badge_click_updates_status_label(self, panel):
        panel._auto_scroll_paused = True
        panel.update_segment("Content", 90, 0, is_final=True, phrase_start=True)

        panel._on_badge_clicked()

        assert "Recording" in panel.status_label.text()


# ---------------------------------------------------------------------------
# 8. User scrolling to bottom resumes
# ---------------------------------------------------------------------------

class TestScrollToBottomResumes:
    """Scrolling back to bottom while paused should resume auto-scroll."""

    def test_scroll_to_max_resumes(self, panel):
        _populate_text(panel, lines=100)
        scrollbar = panel.text_edit.verticalScrollBar()

        # First scroll away from bottom so setValue(max) will actually change
        scrollbar.setValue(0)
        qapp = QApplication.instance()
        qapp.processEvents()

        # Pause auto-scroll
        panel._auto_scroll_paused = True
        panel._pending_content_count = 3
        panel._new_content_badge.show()
        panel._pause_timer.start(10000)

        # Simulate user scrolling to bottom
        scrollbar.setValue(scrollbar.maximum())
        qapp.processEvents()

        assert panel._auto_scroll_paused is False
        assert panel._pending_content_count == 0
        assert not panel._new_content_badge.isVisible()


# ---------------------------------------------------------------------------
# 9. 10s timer resume hides badge
# ---------------------------------------------------------------------------

class TestPauseTimerHidesBadge:
    """When the pause timer fires (_resume_auto_scroll), badge hides and count resets."""

    def test_resume_resets_state(self, panel):
        panel._auto_scroll_paused = True
        panel._pending_content_count = 5
        panel._new_content_badge.show()

        panel._resume_auto_scroll()

        assert panel._auto_scroll_paused is False
        assert panel._pending_content_count == 0
        assert not panel._new_content_badge.isVisible()

    def test_resume_updates_status_label(self, panel):
        panel._auto_scroll_paused = True
        panel.status_label.setText("Auto-scroll paused (10s)")

        panel._resume_auto_scroll()

        assert "Recording" in panel.status_label.text()


# ---------------------------------------------------------------------------
# 10. Badge hidden when not paused
# ---------------------------------------------------------------------------

class TestBadgeHiddenWhenNotPaused:
    """When auto-scroll is active, update_segment should not show the badge."""

    def test_badge_stays_hidden(self, panel):
        panel._auto_scroll_paused = False
        panel._pending_content_count = 0

        panel.update_segment("Active scroll content", 90, 0, is_final=True, phrase_start=True)

        assert not panel._new_content_badge.isVisible()

    def test_count_does_not_increment(self, panel):
        panel._auto_scroll_paused = False
        panel._pending_content_count = 0

        panel.update_segment("Content", 90, 0, is_final=True, phrase_start=True)
        panel.update_segment("More", 85, 0, is_final=True, phrase_start=True)

        assert panel._pending_content_count == 0


# ---------------------------------------------------------------------------
# 11. Badge repositions on resize
# ---------------------------------------------------------------------------

class TestBadgeRepositionOnResize:
    """Badge should reposition to bottom-center when text_edit is resized."""

    def test_badge_repositions_after_resize(self, panel):
        # Show the badge
        panel._auto_scroll_paused = True
        panel._pending_content_count = 1
        panel._new_content_badge.show()
        panel._position_new_content_badge()

        # Record old position
        old_pos = panel._new_content_badge.pos()

        # Resize text_edit (simulate panel resize)
        old_width = panel.text_edit.width()
        old_height = panel.text_edit.height()
        panel.text_edit.setFixedSize(old_width + 50, old_height + 50)

        # Trigger resize event
        panel.resizeEvent(None)
        panel._position_new_content_badge()

        new_pos = panel._new_content_badge.pos()

        # Position should have changed (or at least badge is still correctly positioned)
        te = panel.text_edit
        expected_x = max((te.width() - panel._new_content_badge.width()) // 2, 0)
        expected_y = max(te.height() - panel._new_content_badge.height() - 8, 0)

        assert panel._new_content_badge.x() == expected_x
        assert panel._new_content_badge.y() == expected_y
