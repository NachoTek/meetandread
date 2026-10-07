"""Tests for expandable error help panels and auto-hide duration."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QFontMetrics, QFont


# ── get_error_help_text tests ──

def test_help_for_no_source_selected():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("No audio source selected")
    assert result is not None
    assert "lobe" in result.lower()
    assert "microphone" in result.lower()


def test_help_for_no_mic_selected():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("No microphone selected")
    assert result is not None


def test_help_for_transcription_failure():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Transcription failed unexpectedly")
    assert result is not None
    assert "Whisper" in result
    assert "Settings" in result


def test_help_for_transcription_error():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Transcription error occurred")
    assert result is not None
    assert "model" in result.lower()


def test_help_for_device_not_found():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Microphone not detected")
    assert result is not None
    assert "connected" in result.lower()


def test_help_for_speaker_not_available():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Speaker not available")
    assert result is not None


def test_help_for_memory_low():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Memory low on system")
    assert result is not None
    assert "Close" in result


def test_help_for_out_of_memory():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Out of memory during transcription")
    assert result is not None


def test_help_for_unknown_message():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("Something completely unrelated happened")
    assert result is None


def test_help_case_insensitive():
    from meetandread.widgets.main_widget import get_error_help_text
    result = get_error_help_text("TRANSCRIPTION FAILED")
    assert result is not None
    result2 = get_error_help_text("No Source Selected")
    assert result2 is not None


# ── ErrorIndicatorItem help integration tests ──

@pytest.fixture
def indicator(qtbot):
    """Create an ErrorIndicatorItem for testing."""
    from meetandread.widgets.main_widget import ErrorIndicatorItem
    parent = MagicMock()
    item = ErrorIndicatorItem(parent)
    return item


def test_indicator_help_text_set_for_matching_message(indicator):
    """Matching error message sets _help_text."""
    indicator.set_text("No audio source selected", is_recoverable=True)
    assert indicator._help_text is not None
    assert "lobe" in indicator._help_text.lower()


def test_indicator_no_help_for_unmatching_message(indicator):
    """Non-matching error message leaves _help_text as None."""
    indicator.set_text("Random unknown error", is_recoverable=True)
    assert indicator._help_text is None


def test_indicator_no_help_for_non_recoverable(indicator):
    """Non-recoverable errors never get help text, even with matching messages."""
    indicator.set_text("No audio source selected", is_recoverable=False)
    assert indicator._help_text is None


def test_indicator_help_text_resets_on_new_message(indicator):
    """Setting new text resets expansion state."""
    indicator.set_text("No audio source selected", is_recoverable=True)
    indicator._help_expanded = True
    indicator.set_text("Device not found", is_recoverable=True)
    assert indicator._help_expanded is False


def test_indicator_initial_no_help(indicator):
    """Fresh indicator has no help text."""
    assert indicator._help_text is None
    assert indicator._help_expanded is False


def test_indicator_width_is_190(indicator):
    """Width should be 190 to accommodate help button."""
    assert indicator.rect().width() == 190


# ── Help button toggle via mousePressEvent ──

def test_click_help_button_toggles_expansion(indicator):
    """Clicking the '?' button toggles _help_expanded."""
    indicator.set_text("Transcription failed", is_recoverable=True)
    assert indicator._help_text is not None
    assert not indicator._help_expanded

    # Simulate click in help button area
    btn = indicator._help_button_rect()
    event = MagicMock()
    event.pos.return_value = btn.center()
    indicator.mousePressEvent(event)
    assert indicator._help_expanded is True
    event.accept.assert_called()

    # Click again to collapse
    event2 = MagicMock()
    event2.pos.return_value = btn.center()
    indicator.mousePressEvent(event2)
    assert indicator._help_expanded is False


def test_click_outside_help_button_does_not_toggle(indicator):
    """Clicking outside the '?' area does not toggle help."""
    indicator.set_text("Transcription failed", is_recoverable=True)
    event = MagicMock()
    event.pos.return_value = indicator.rect().center()
    indicator.mousePressEvent(event)
    assert indicator._help_expanded is False


def test_click_no_help_button_when_no_help(indicator):
    """No toggle when _help_text is None."""
    indicator.set_text("Random error", is_recoverable=True)
    event = MagicMock()
    event.pos.return_value = indicator.rect().center()
    indicator.mousePressEvent(event)
    assert indicator._help_expanded is False


def test_help_expansion_notifies_parent(indicator):
    """Expanding help calls _on_error_help_toggled on parent widget."""
    indicator.set_text("Transcription failed", is_recoverable=True)
    btn = indicator._help_button_rect()
    event = MagicMock()
    event.pos.return_value = btn.center()
    indicator.mousePressEvent(event)
    indicator.parent_widget._on_error_help_toggled.assert_called_once_with(True)


# ── Auto-hide timer tests ──

def test_show_error_uses_8s_timer(qtbot):
    """_show_error should use 8000ms auto-hide timer."""
    from meetandread.widgets.main_widget import MeetAndReadWidget
    with patch('meetandread.widgets.main_widget.MeetAndReadWidget.__init__', return_value=None):
        # We need a minimal widget — just test the timer mechanism
        pass

    # Directly test the timer creation mechanism
    from PyQt6.QtCore import QTimer
    from unittest.mock import call

    # Verify 8000ms value by testing _show_error behavior
    from meetandread.widgets.main_widget import MeetAndReadWidget

    # Mock __init__ to avoid full widget construction
    orig_init = MeetAndReadWidget.__init__
    try:
        MeetAndReadWidget.__init__ = lambda self: None  # pyright: ignore[reportAttributeAccessIssue]  # intentional mock seam
        widget = MeetAndReadWidget()
        widget._error_indicator = MagicMock()
        widget._error_hide_timer = None

        # Patch QTimer to track calls
        with patch('meetandread.widgets.main_widget.QTimer') as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            widget._show_error("Test error")

            # Should have started timer with 8000ms
            mock_timer_instance.start.assert_called_once_with(8000)
    finally:
        MeetAndReadWidget.__init__ = orig_init


def test_expanding_help_cancels_timer():
    """Expanding help should stop the auto-hide timer."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    orig_init = MeetAndReadWidget.__init__
    try:
        MeetAndReadWidget.__init__ = lambda self: None  # pyright: ignore[reportAttributeAccessIssue]  # intentional mock seam
        widget = MeetAndReadWidget()
        widget._error_indicator = MagicMock()
        widget._error_hide_timer = MagicMock()

        widget._on_error_help_toggled(expanded=True)

        # Should have stopped the timer
        widget._error_hide_timer.stop.assert_called_once()
    finally:
        MeetAndReadWidget.__init__ = orig_init


def test_collapsing_help_restarts_timer():
    """Collapsing help should restart the auto-hide timer with 8s."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    orig_init = MeetAndReadWidget.__init__
    try:
        MeetAndReadWidget.__init__ = lambda self: None  # pyright: ignore[reportAttributeAccessIssue]  # intentional mock seam
        widget = MeetAndReadWidget()
        widget._error_indicator = MagicMock()
        widget._error_hide_timer = MagicMock()

        with patch('meetandread.widgets.main_widget.QTimer') as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            widget._on_error_help_toggled(expanded=False)

            # Should have restarted timer with 8000ms
            mock_timer_instance.start.assert_called_once_with(8000)
    finally:
        MeetAndReadWidget.__init__ = orig_init


def test_on_controller_error_passes_recoverable():
    """_on_controller_error should pass is_recoverable to _show_error."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    orig_init = MeetAndReadWidget.__init__
    try:
        MeetAndReadWidget.__init__ = lambda self: None  # pyright: ignore[reportAttributeAccessIssue]  # intentional mock seam
        widget = MeetAndReadWidget()

        error = MagicMock()
        error.message = "No source selected"
        error.is_recoverable = True

        with patch.object(widget, '_show_error') as mock_show:
            widget._on_controller_error(error)
            mock_show.assert_called_once_with("No source selected", is_recoverable=True)

        # Test non-recoverable
        error2 = MagicMock()
        error2.message = "Unexpected error"
        error2.is_recoverable = False

        with patch.object(widget, '_show_error') as mock_show:
            widget._on_controller_error(error2)
            mock_show.assert_called_once_with("Unexpected error", is_recoverable=False)
    finally:
        MeetAndReadWidget.__init__ = orig_init
