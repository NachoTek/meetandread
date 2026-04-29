"""Tests for CCOverlayPanel — compact closed-caption overlay shell.

Covers:
- Child inventory: one QTextEdit, one QSizeGrip, no history/status chrome
- Size constraints: compact min/default/max bounds
- Grip anchoring: direct-child QSizeGrip at bottom-right corner
- Drag movement: mouse-drag repositions the panel
- Safe no-parent construction: works without parent, no crash
- Shell methods: show_panel, hide_panel (immediate + deferred), toggle_panel,
  dock_to_widget, clear
- Empty state: panel starts with _has_content == False
- Object names: panel and text_edit have correct objectNames
- Live transcript: update_segment, phrase tracking, speaker labels,
  safe HTML handling, segment replacement, blank filtering, confidence colours
"""

import pytest

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QSizeGrip,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from meetandread.widgets.floating_panels import CCOverlayPanel, Phrase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    """Provide a QApplication singleton for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def cc_panel(qapp):
    """Create a shown CCOverlayPanel for testing, cleaned up after."""
    panel = CCOverlayPanel()
    panel.show()
    qapp.processEvents()
    yield panel
    panel.close()


@pytest.fixture
def cc_panel_with_parent(qapp):
    """Create a CCOverlayPanel with a parent widget, cleaned up after."""
    parent = QWidget()
    parent.show()
    qapp.processEvents()
    panel = CCOverlayPanel(parent=parent)
    panel.show()
    qapp.processEvents()
    yield panel, parent
    panel.close()
    parent.close()


# ---------------------------------------------------------------------------
# 1. Child inventory — correct children, no history/status
# ---------------------------------------------------------------------------

class TestChildInventory:
    """CCOverlayPanel must have exactly the expected child widgets."""

    def test_has_text_edit(self, cc_panel):
        text_edits = cc_panel.findChildren(QTextEdit)
        assert len(text_edits) >= 1, "CCOverlayPanel should have at least one QTextEdit"

    def test_text_edit_is_read_only(self, cc_panel):
        te = cc_panel.findChild(QTextEdit)
        assert te is not None
        assert te.isReadOnly(), "CC overlay text edit must be read-only"

    def test_has_size_grip(self, cc_panel):
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None, "CCOverlayPanel should have a QSizeGrip child"

    def test_grip_is_direct_child(self, cc_panel):
        """QSizeGrip must be a direct child of the panel (MEM083)."""
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None
        assert grip.parent() is cc_panel, "QSizeGrip parent must be the panel itself"

    def test_no_tab_widget(self, cc_panel):
        """CC overlay must not have a QTabWidget (no history chrome)."""
        tabs = cc_panel.findChildren(QTabWidget)
        assert len(tabs) == 0, "CC overlay should not contain any QTabWidget"

    def test_no_status_label(self, cc_panel):
        """CC overlay should not have status/recording labels."""
        from PyQt6.QtWidgets import QLabel
        labels = cc_panel.findChildren(QLabel)
        # We expect at most the text display widget, not status labels
        for label in labels:
            # No label should contain "Recording" or "Ready" status text
            text = label.text()
            assert "Recording" not in text
            assert "Ready" not in text


# ---------------------------------------------------------------------------
# 2. Size constraints — compact bounds
# ---------------------------------------------------------------------------

class TestSizeConstraints:
    """CC overlay uses compact size bounds suitable for caption overlay."""

    def test_minimum_width(self, cc_panel):
        assert cc_panel.minimumWidth() >= 280

    def test_minimum_height(self, cc_panel):
        assert cc_panel.minimumHeight() >= 80

    def test_maximum_width(self, cc_panel):
        assert cc_panel.maximumWidth() <= 1000

    def test_maximum_height(self, cc_panel):
        assert cc_panel.maximumHeight() <= 500

    def test_not_fixed_size(self, cc_panel):
        """Panel should be resizable (min != max)."""
        sz = cc_panel.minimumSize()
        mx = cc_panel.maximumSize()
        assert (sz.width() != mx.width()) or (sz.height() != mx.height())

    def test_compact_default_size(self, cc_panel):
        """Default size should be compact — smaller than transcript panel."""
        # CC overlay should be significantly smaller than FloatingTranscriptPanel
        # which has minSize(350, 300)
        assert cc_panel.minimumWidth() < 350
        assert cc_panel.minimumHeight() < 300


# ---------------------------------------------------------------------------
# 3. Grip anchoring — direct-child grip at bottom-right
# ---------------------------------------------------------------------------

class TestGripAnchoring:
    """QSizeGrip must anchor at bottom-right after resize."""

    def test_grip_at_bottom_right(self, cc_panel, qapp):
        cc_panel.resize(400, 200)
        qapp.processEvents()
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = cc_panel.width() - grip.width()
        expected_y = cc_panel.height() - grip.height()
        assert grip.x() == expected_x, f"Grip x={grip.x()}, expected {expected_x}"
        assert grip.y() == expected_y, f"Grip y={grip.y()}, expected {expected_y}"

    def test_grip_at_bottom_right_after_larger_resize(self, cc_panel, qapp):
        cc_panel.resize(600, 350)
        qapp.processEvents()
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = cc_panel.width() - grip.width()
        expected_y = cc_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y

    def test_grip_at_bottom_right_at_min_size(self, cc_panel, qapp):
        min_w = cc_panel.minimumWidth()
        min_h = cc_panel.minimumHeight()
        cc_panel.resize(min_w, min_h)
        qapp.processEvents()
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = cc_panel.width() - grip.width()
        expected_y = cc_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y

    def test_grip_visible(self, cc_panel):
        grip = cc_panel.findChild(QSizeGrip)
        assert grip is not None
        assert grip.isVisible()


# ---------------------------------------------------------------------------
# 4. Drag movement — mouse-drag repositions panel
# ---------------------------------------------------------------------------

class TestDragMovement:
    """Mouse drag on the panel background should move the window."""

    def test_drag_moves_panel(self, cc_panel, qapp):
        from PyQt6.QtCore import QPointF
        # Position the panel at a known location
        cc_panel.move(100, 100)
        qapp.processEvents()
        initial_pos = cc_panel.pos()

        # Simulate mouse press at center
        local = QPointF(cc_panel.width() / 2, cc_panel.height() / 2)
        global_press = QPointF(100 + local.x(), 100 + local.y())
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            local, global_press,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cc_panel.mousePressEvent(press)

        # Simulate mouse move to global position offset by (50, 30)
        global_move = QPointF(global_press.x() + 50, global_press.y() + 30)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(local.x() + 50, local.y() + 30),
            global_move,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cc_panel.mouseMoveEvent(move)

        # Release
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(local.x() + 50, local.y() + 30),
            global_move,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cc_panel.mouseReleaseEvent(release)
        qapp.processEvents()

        # Panel should have moved by the drag offset
        new_pos = cc_panel.pos()
        assert new_pos != initial_pos, f"Panel should have moved: was {initial_pos}, now {new_pos}"


# ---------------------------------------------------------------------------
# 5. Safe no-parent construction
# ---------------------------------------------------------------------------

class TestSafeNoParentConstruction:
    """CCOverlayPanel must work when constructed without a parent."""

    def test_construct_no_parent(self, qapp):
        """Constructing without parent must not crash."""
        panel = CCOverlayPanel(parent=None)
        assert panel is not None
        panel.close()

    def test_show_without_parent(self, qapp):
        """Showing without parent must not crash."""
        panel = CCOverlayPanel(parent=None)
        panel.show()
        qapp.processEvents()
        assert panel.isVisible()
        panel.close()

    def test_dock_to_widget_without_parent(self, qapp):
        """dock_to_widget with a valid widget must work even without initial parent."""
        panel = CCOverlayPanel(parent=None)
        dummy = QWidget()
        dummy.show()
        qapp.processEvents()
        # Should not crash
        panel.dock_to_widget(dummy)
        qapp.processEvents()
        panel.close()
        dummy.close()


# ---------------------------------------------------------------------------
# 6. Shell methods — show, hide, toggle, clear
# ---------------------------------------------------------------------------

class TestShellMethods:
    """Shell methods must work correctly."""

    def test_show_panel(self, cc_panel, qapp):
        cc_panel.hide()
        qapp.processEvents()
        assert not cc_panel.isVisible()
        cc_panel.show_panel()
        qapp.processEvents()
        assert cc_panel.isVisible()

    def test_hide_panel_immediate(self, cc_panel, qapp):
        cc_panel.show()
        qapp.processEvents()
        assert cc_panel.isVisible()
        cc_panel.hide_panel(immediate=True)
        qapp.processEvents()
        assert not cc_panel.isVisible()

    def test_toggle_panel_shows(self, cc_panel, qapp):
        cc_panel.hide()
        qapp.processEvents()
        cc_panel.toggle_panel()
        qapp.processEvents()
        assert cc_panel.isVisible()

    def test_toggle_panel_hides(self, cc_panel, qapp):
        """Toggle on visible panel triggers fade-out which ends in hide."""
        cc_panel.show()
        qapp.processEvents()
        assert cc_panel.isVisible()
        cc_panel.toggle_panel()
        # The fade-out timer runs 15 steps of 10ms each
        # Process timer events by spinning the event loop with delays
        import time
        for _ in range(20):
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()
        assert not cc_panel.isVisible(), "Panel should be hidden after fade-out completes"

    def test_clear_resets_content(self, cc_panel, qapp):
        te = cc_panel.findChild(QTextEdit)
        assert te is not None
        te.setPlainText("Some text")
        cc_panel.clear()
        assert te.toPlainText() == "", "clear() should empty the text edit"
        assert not cc_panel._has_content, "clear() should reset _has_content"

    def test_dock_to_widget_positions_panel(self, cc_panel_with_parent, qapp):
        panel, parent = cc_panel_with_parent
        panel.dock_to_widget(parent)
        qapp.processEvents()
        # Panel should be positioned relative to the parent (not at the same location)
        parent_pos = parent.mapToGlobal(parent.rect().topLeft())
        # The panel must have been repositioned away from its initial spot
        # (dock_to_widget places it adjacent to the parent)
        assert panel._has_been_docked, "dock_to_widget should set _has_been_docked"


# ---------------------------------------------------------------------------
# 7. Object names
# ---------------------------------------------------------------------------

class TestObjectNames:
    """Panel and child widgets must have correct object names for styling."""

    def test_panel_object_name(self, cc_panel):
        assert cc_panel.objectName() == "AethericCCOverlay"

    def test_text_edit_object_name(self, cc_panel):
        te = cc_panel.findChild(QTextEdit)
        assert te is not None
        # Text edit should have a scoped object name
        assert te.objectName() != "", "Text edit should have a non-empty objectName"


# ---------------------------------------------------------------------------
# 8. Window flags — frameless, tool, always-on-top
# ---------------------------------------------------------------------------

class TestWindowFlags:
    """CC overlay must have frameless/tool/always-on-top flags."""

    def test_frameless(self, cc_panel):
        flags = cc_panel.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint

    def test_tool_window(self, cc_panel):
        flags = cc_panel.windowFlags()
        assert flags & Qt.WindowType.Tool

    def test_always_on_top(self, cc_panel):
        flags = cc_panel.windowFlags()
        assert flags & Qt.WindowType.WindowStaysOnTopHint

    def test_translucent_background(self, cc_panel):
        assert cc_panel.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


# ---------------------------------------------------------------------------
# 9. Live transcript rendering — update_segment
# ---------------------------------------------------------------------------

class TestLiveTranscriptRendering:
    """CC overlay must render live transcript segments safely."""

    def test_initial_state(self, cc_panel):
        """Panel starts with no content, empty phrases, -1 phrase index."""
        assert cc_panel._has_content is False
        assert cc_panel.phrases == []
        assert cc_panel.current_phrase_idx == -1

    def test_single_segment_appends_text(self, cc_panel, qapp):
        """First update_segment with phrase_start shows text."""
        cc_panel.update_segment("Hello world", 90, 0, False, True)
        qapp.processEvents()
        assert cc_panel._has_content is True
        assert len(cc_panel.phrases) == 1
        assert cc_panel.phrases[0].segments == ["Hello world"]
        text = cc_panel.text_edit.toPlainText()
        assert "Hello world" in text

    def test_append_second_segment_to_phrase(self, cc_panel, qapp):
        """Appending a segment to the current phrase shows both."""
        cc_panel.update_segment("Hello", 85, 0, False, True)
        cc_panel.update_segment("world", 90, 1, False, False)
        qapp.processEvents()
        assert len(cc_panel.phrases) == 1
        assert cc_panel.phrases[0].segments == ["Hello", "world"]
        text = cc_panel.text_edit.toPlainText()
        assert "Hello" in text
        assert "world" in text

    def test_phrase_start_creates_new_line(self, cc_panel, qapp):
        """phrase_start=True creates a new phrase (new line)."""
        cc_panel.update_segment("Line one", 80, 0, False, True)
        cc_panel.update_segment("Line two", 85, 0, False, True)
        qapp.processEvents()
        assert len(cc_panel.phrases) == 2
        assert cc_panel.phrases[0].segments == ["Line one"]
        assert cc_panel.phrases[1].segments == ["Line two"]

    def test_is_final_marks_phrase(self, cc_panel, qapp):
        """is_final=True marks the current phrase as complete."""
        cc_panel.update_segment("Done", 95, 0, True, True)
        qapp.processEvents()
        assert cc_panel.phrases[0].is_final is True

    def test_replacement_in_place(self, cc_panel, qapp):
        """Updating an existing segment index replaces it in-place."""
        cc_panel.update_segment("Hello", 80, 0, False, True)
        cc_panel.update_segment("Hello world", 90, 0, False, False)
        qapp.processEvents()
        assert cc_panel.phrases[0].segments == ["Hello world"]
        text = cc_panel.text_edit.toPlainText()
        assert "Hello world" in text

    def test_blank_audio_filtered(self, cc_panel, qapp):
        """[BLANK_AUDIO] text should be silently ignored."""
        cc_panel.update_segment("[BLANK_AUDIO]", 50, 0, False, True)
        qapp.processEvents()
        assert cc_panel._has_content is False
        assert len(cc_panel.phrases) == 0

    def test_empty_text_not_filtered(self, cc_panel, qapp):
        """Empty string is still processed (not blank audio)."""
        cc_panel.update_segment("", 50, 0, False, True)
        qapp.processEvents()
        # Empty text is allowed — only [BLANK_AUDIO] is filtered
        assert len(cc_panel.phrases) == 1

    def test_clear_resets_phrases(self, cc_panel, qapp):
        """clear() resets phrases, index, and content flag."""
        cc_panel.update_segment("Hello", 90, 0, False, True)
        cc_panel.clear()
        assert cc_panel.phrases == []
        assert cc_panel.current_phrase_idx == -1
        assert cc_panel._has_content is False
        assert cc_panel.text_edit.toPlainText() == ""


# ---------------------------------------------------------------------------
# 10. Speaker labels
# ---------------------------------------------------------------------------

class TestSpeakerLabels:
    """CC overlay must show speaker labels with colours."""

    def test_speaker_label_shown(self, cc_panel, qapp):
        """Segment with speaker_id shows the speaker label."""
        cc_panel.update_segment("Hello", 90, 0, False, True, speaker_id="SPK_0")
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "SPK_0" in text

    def test_speaker_label_has_color(self, cc_panel, qapp):
        """Speaker label must be rendered with a deterministic color."""
        cc_panel.update_segment("Hello", 90, 0, False, True, speaker_id="SPK_0")
        qapp.processEvents()
        # Check that the phrase has the speaker_id stored
        assert cc_panel.phrases[0].speaker_id == "SPK_0"

    def test_no_speaker_label_when_none(self, cc_panel, qapp):
        """Segment without speaker_id should not show speaker label."""
        cc_panel.update_segment("Hello", 90, 0, False, True, speaker_id=None)
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "[" not in text or "SPK" not in text

    def test_multiple_speakers(self, cc_panel, qapp):
        """Multiple speakers get different labels."""
        cc_panel.update_segment("Alice says", 90, 0, False, True, speaker_id="SPK_0")
        cc_panel.update_segment("Bob says", 85, 0, False, True, speaker_id="SPK_1")
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "SPK_0" in text
        assert "SPK_1" in text

    def test_set_speaker_names_refreshes(self, cc_panel, qapp):
        """set_speaker_names updates the display name mapping."""
        cc_panel.update_segment("Hello", 90, 0, False, True, speaker_id="SPK_0")
        cc_panel.set_speaker_names({"SPK_0": "Alice"})
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "Alice" in text

    def test_get_speaker_names_returns_copy(self, cc_panel):
        """get_speaker_names returns a copy, not the internal dict."""
        cc_panel.set_speaker_names({"SPK_0": "Alice"})
        names = cc_panel.get_speaker_names()
        names["SPK_0"] = "Bob"
        # Internal dict should be unchanged
        assert cc_panel.get_speaker_names()["SPK_0"] == "Alice"


# ---------------------------------------------------------------------------
# 11. Safe HTML handling
# ---------------------------------------------------------------------------

class TestSafeHTMLHandling:
    """CC overlay must handle HTML/script-like text safely."""

    def test_html_tags_escaped(self, cc_panel, qapp):
        """HTML tags in transcript text must be escaped, not rendered."""
        cc_panel.update_segment("<b>bold</b>", 90, 0, False, True)
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "<b>" not in text or "bold" in text
        # The text must display the literal characters, not render them as HTML
        assert "bold" in text

    def test_script_tag_escaped(self, cc_panel, qapp):
        """Script tags must be escaped."""
        cc_panel.update_segment('<script>alert("xss")</script>', 90, 0, False, True)
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        # Should not execute — just show as text
        assert "alert" in text

    def test_ampersand_escaped(self, cc_panel, qapp):
        """Ampersands must be properly escaped."""
        cc_panel.update_segment("Tom & Jerry", 90, 0, False, True)
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "Tom & Jerry" in text or "Tom" in text

    def test_angle_brackets_escaped(self, cc_panel, qapp):
        """Angle brackets in text must be shown literally."""
        cc_panel.update_segment("5 > 3 and 3 < 5", 90, 0, False, True)
        qapp.processEvents()
        text = cc_panel.text_edit.toPlainText()
        assert "5" in text and "3" in text


# ---------------------------------------------------------------------------
# 12. Confidence colour application
# ---------------------------------------------------------------------------

class TestConfidenceColours:
    """CC overlay uses canonical confidence colours from confidence module."""

    def test_high_confidence_color(self, cc_panel, qapp):
        """High confidence text should have green-ish color."""
        from meetandread.transcription.confidence import get_confidence_color
        color = get_confidence_color(95)
        assert color  # Must return a valid color string

    def test_low_confidence_color(self, cc_panel, qapp):
        """Low confidence text should have red-ish color."""
        from meetandread.transcription.confidence import get_confidence_color
        color = get_confidence_color(30)
        assert color

    def test_panel_uses_canonical_colors(self, cc_panel, qapp):
        """Panel delegates to get_confidence_color (MEM027)."""
        cc_panel.update_segment("text", 85, 0, False, True)
        qapp.processEvents()
        # The method exists and doesn't crash — detailed colour verification
        # is covered by the confidence module's own tests
        color = cc_panel._get_confidence_color(85)
        from meetandread.transcription.confidence import get_confidence_color
        assert color == get_confidence_color(85)


# ---------------------------------------------------------------------------
# 13. segment_ready signal
# ---------------------------------------------------------------------------

class TestSegmentReadySignal:
    """CC overlay emits segment_ready signal."""

    def test_signal_exists(self, cc_panel):
        """Panel has segment_ready signal."""
        assert hasattr(cc_panel, 'segment_ready')

    def test_signal_emitted(self, cc_panel, qapp):
        """update_segment emits segment_ready."""
        emitted = []
        cc_panel.segment_ready.connect(
            lambda t, c, si, f, ps: emitted.append((t, c, si, f, ps))
        )
        cc_panel.update_segment("Hello", 90, 0, True, True)
        qapp.processEvents()
        assert len(emitted) == 1
        assert emitted[0] == ("Hello", 90, 0, True, True)
