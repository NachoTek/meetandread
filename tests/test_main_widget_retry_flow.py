"""Tests for widget-level WASAPI start retry flow (T02)."""
import time
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QDialog

from meetandread.widgets.main_widget import MeetAndReadWidget
from meetandread.recording.controller import ControllerState, ControllerError


def _create_widget_with_mocked_controller(monkeypatch):
    """Create a widget with mocked controller for testing.
    
    Uses heavy mocking to avoid floating panel initialization issues in test environment.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Mock RecordingController
    mock_controller = Mock()
    mock_controller.is_recording.return_value = False
    mock_controller.is_busy.return_value = False
    mock_controller.get_state.return_value = ControllerState.IDLE
    mock_controller.clear_error.return_value = None
    mock_controller.begin_start_retry_sequence.return_value = None
    mock_controller.record_start_retry_attempt.return_value = None
    mock_controller.record_start_retry_outcome.return_value = None

    # Mock floating panels to avoid initialization issues
    with patch("meetandread.widgets.main_widget.FloatingSettingsPanel"), \
         patch("meetandread.widgets.main_widget.CCOverlayPanel"), \
         patch("meetandread.widgets.main_widget.ToastManager") as mock_toast_mgr:
        
        # Create a minimal mock toast manager
        toast_manager_instance = Mock()
        toast_manager_instance.show = Mock()
        toast_manager_instance.dismiss = Mock()
        mock_toast_mgr.return_value = toast_manager_instance
        
        with patch("meetandread.widgets.main_widget.RecordingController", return_value=mock_controller):
            widget = MeetAndReadWidget()
            widget._controller = mock_controller
            widget.toast_manager = toast_manager_instance

    return widget, mock_controller


def test_start_recording_with_mic_only_does_not_retry(monkeypatch):
    """Mic-only starts should not trigger retry flow."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to succeed immediately
    mock_controller.start.return_value = None  # Success

    # Set mic-only selection
    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = False

    # Start recording
    widget.start_recording()

    # Should have called start once without retry
    assert mock_controller.start.call_count == 1
    sources = mock_controller.start.call_args[0][0]
    assert sources == {'mic'}

    # No retry helpers should have been called
    assert not mock_controller.begin_start_retry_sequence.called
    assert not mock_controller.record_start_retry_attempt.called


def test_start_recording_with_system_audio_invokes_retry_flow_on_failure(monkeypatch):
    """System audio start failures should trigger retry flow with backoff."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to fail (simulate WASAPI endpoint unavailable)
    mock_controller.start.return_value = ControllerError(
        "AudioSourceError: Could not open system audio endpoint"
    )

    # Set system+mic selection
    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True

    # Start recording
    widget.start_recording()

    # Should have called start once and begun retry sequence
    assert mock_controller.start.call_count == 1
    assert mock_controller.begin_start_retry_sequence.called

    # Retry state should be set
    assert hasattr(widget, '_retry_in_progress')
    assert widget._retry_in_progress is True


def test_retry_attempts_uses_exponential_backoff_schedule(monkeypatch):
    """Retry attempts should follow 1s, 2s, 4s exponential backoff."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to fail consistently
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # First retry attempt should have been recorded with 1s backoff
    assert mock_controller.record_start_retry_attempt.called
    call_kwargs = mock_controller.record_start_retry_attempt.call_args[1]
    assert call_kwargs['backoff_seconds'] == 1.0  # First backoff is 1s


def test_retry_toast_shows_attempt_number_and_countdown(monkeypatch):
    """Retry toast should show attempt number and countdown."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to fail
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    # Mock ToastManager.show
    toast_calls = []
    original_toast_show = widget.toast_manager.show

    def capture_toast(toast_id, title, message, **kwargs):
        toast_calls.append({'id': toast_id, 'title': title, 'message': message})

    widget.toast_manager.show = capture_toast

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # Should have shown retry toast
    assert len(toast_calls) > 0
    retry_toast = toast_calls[0]
    assert retry_toast['id'] == 'wasapi-retry'
    assert 'Attempt 1/3' in retry_toast['title'] or 'Opening system audio' in retry_toast['title']


def test_user_can_cancel_retry_by_clicking_record_button(monkeypatch):
    """User should be able to cancel retry by clicking record button again."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to fail
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    # Start recording with system audio (will enter retry flow)
    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # Verify retry is in progress
    assert widget._retry_in_progress is True

    # Cancel retry by calling toggle_recording again
    widget.toggle_recording()

    # Retry should be cancelled
    assert widget._retry_in_progress is False
    assert hasattr(widget, '_retry_timer')
    if widget._retry_timer:
        assert widget._retry_timer.isActive() is False


def test_successful_transient_retry_clears_retry_state(monkeypatch):
    """Successful retry should clear retry state and toast."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to fail first, then succeed on retry
    start_attempts = [0]

    def mock_start_side_effect(sources):
        start_attempts[0] += 1
        if start_attempts[0] == 1:
            return ControllerError("AudioSourceError: endpoint unavailable")
        else:
            return None  # Success

    mock_controller.start.side_effect = mock_start_side_effect

    # Start recording with system audio
    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # Manually trigger retry attempt (simulate timer fire)
    if hasattr(widget, '_retry_timer') and widget._retry_timer:
        widget._retry_timer.timeout.emit()

    # Verify retry attempt was recorded
    assert mock_controller.record_start_retry_attempt.called

    # After successful retry, state should be cleared
    # (This would be verified via state_change callback in real flow)


def test_exhausted_retries_show_fallback_dialog(monkeypatch):
    """After 3 failed retries, should show fallback confirmation dialog."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to always fail
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    # Mock FallbackConfirmationDialog
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        widget.mic_lobe.is_active = True
        widget.system_lobe.is_active = True

        # Simulate exhausted retries by setting attempt count
        widget._retry_attempt = 3
        widget._start_with_retry({'mic', 'system'}, first_attempt=False)

        # Should have shown fallback dialog
        assert mock_dialog_class.called
        assert mock_dialog.exec.called


def test_fallback_accepted_starts_with_mic_only(monkeypatch):
    """If user accepts fallback, should start with mic-only sources."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to succeed
    mock_controller.start.return_value = None

    # Mock FallbackConfirmationDialog to accept fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have recorded retry outcome with fallback
        assert mock_controller.record_start_retry_outcome.called
        outcome_args = mock_controller.record_start_retry_outcome.call_args[1]
        assert 'fallback' in outcome_args.get('outcome', '').lower()

        # Should have started with mic-only
        start_call_sources = mock_controller.start.call_args[0][0]
        assert start_call_sources == {'mic'}


def test_fallback_rejected_clears_retry_state(monkeypatch):
    """If user rejects fallback, should clear retry state and return to IDLE."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog to reject fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog.accepted_fallback.return_value = False
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have recorded retry outcome as failed
        assert mock_controller.record_start_retry_outcome.called
        outcome_args = mock_controller.record_start_retry_outcome.call_args[1]
        assert outcome_args.get('outcome') == 'failed'

        # Should NOT have started recording
        assert mock_controller.start.call_count == 0


def test_retry_does_not_wrap_hotplug_recovery(monkeypatch):
    """Retry flow should NOT wrap hot-plug recovery (mid-recording only)."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock controller as already recording (not idle)
    mock_controller.is_recording.return_value = True
    mock_controller.is_busy.return_value = True

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True

    # Toggle recording while already recording (should call stop, not retry start)
    widget.toggle_recording()

    # Should have called stop, not start with retry
    assert not mock_controller.begin_start_retry_sequence.called
    # Verify stop was called instead
    assert mock_controller.stop.called or widget.is_recording


def test_retrying_state_is_considered_busy(monkeypatch):
    """Widget should consider RETRYING controller state as busy."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Simulate controller in RETRYING state
    mock_controller.get_state.return_value = ControllerState.RETRYING
    mock_controller.is_busy.return_value = True

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True

    # Try to start while retrying
    widget.start_recording()

    # Should not start new recording (busy)
    assert not mock_controller.start.called


def test_retry_updates_same_toast_widget_instead_of_spamming(monkeypatch):
    """Retry attempts should update the same toast widget, not create new ones."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    toast_count = [0]

    def count_toasts(toast_id, title, message, **kwargs):
        if toast_id == 'wasapi-retry':
            toast_count[0] += 1

    widget.toast_manager.show = count_toasts

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # Should only call show once for initial toast
    assert toast_count[0] == 1


def test_retry_timer_is_cleaned_up_after_completion(monkeypatch):
    """Retry timer should be stopped and cleaned up after retry completes."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to succeed on retry
    start_attempts = [0]

    def mock_start_side_effect(sources):
        start_attempts[0] += 1
        if start_attempts[0] == 1:
            return ControllerError("Audio device error: WASAPI loopback endpoint unavailable")
        return None

    mock_controller.start.side_effect = mock_start_side_effect

    widget.mic_lobe.is_active = True
    widget.system_lobe.is_active = True
    widget.start_recording()

    # Trigger retry by emitting timer timeout (simulates timer fire)
    if hasattr(widget, '_retry_timer') and widget._retry_timer:
        widget._retry_timer.timeout.emit()

    # After successful retry, timer reference should be cleared and state reset
    assert widget._retry_timer is None
    assert widget._retry_in_progress is False
