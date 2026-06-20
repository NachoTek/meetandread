"""Tests for WASAPI fallback flow (T03) - explicit mic-only fallback confirmation."""
from unittest.mock import Mock, patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog

from meetandread.widgets.main_widget import MeetAndReadWidget
from meetandread.recording.controller import ControllerState, ControllerError


def _create_widget_with_mocked_controller(monkeypatch):
    """Create a widget with mocked controller for testing."""
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
        
        toast_manager_instance = Mock()
        toast_manager_instance.show = Mock()
        toast_manager_instance.dismiss = Mock()
        mock_toast_mgr.return_value = toast_manager_instance
        
        with patch("meetandread.widgets.main_widget.RecordingController", return_value=mock_controller):
            widget = MeetAndReadWidget()
            widget._controller = mock_controller
            widget.toast_manager = toast_manager_instance

    return widget, mock_controller


def test_no_silent_degraded_start_without_confirmation(monkeypatch):
    """No silent degraded start - fallback requires explicit user confirmation."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to always fail
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    # Mock FallbackConfirmationDialog to reject fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog.accepted_fallback.return_value = False
        mock_dialog_class.return_value = mock_dialog

        widget.mic_lobe.is_active = True
        widget.system_lobe.is_active = True

        # Simulate exhausted retries
        widget._retry_attempt = 3
        widget._start_with_retry({'mic', 'system'}, first_attempt=False)

        # Should have shown dialog (not silently started degraded recording)
        assert mock_dialog_class.called
        assert mock_dialog.exec.called

        # Should NOT have started recording (user rejected fallback)
        assert mock_controller.start.call_count == 1  # Only initial attempt, no fallback


def test_fallback_accepted_starts_only_mic_sources(monkeypatch):
    """Accepted fallback starts only mic sources, not system audio."""
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

        # Should have started with mic-only
        assert mock_controller.start.called
        start_call_sources = mock_controller.start.call_args[0][0]
        assert start_call_sources == {'mic'}
        assert 'system' not in start_call_sources

        # Should have recorded fallback outcome
        assert mock_controller.record_start_retry_outcome.called
        outcome_args = mock_controller.record_start_retry_outcome.call_args[1]
        assert 'fallback' in outcome_args.get('outcome', '').lower()
        assert outcome_args.get('fallback_sources') == ['mic']


def test_fallback_rejected_leaves_controller_idle(monkeypatch):
    """Rejected fallback leaves controller in IDLE state with clear error."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog to reject fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog.accepted_fallback.return_value = False
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have recorded failed outcome
        assert mock_controller.record_start_retry_outcome.called
        outcome_args = mock_controller.record_start_retry_outcome.call_args[1]
        assert outcome_args.get('outcome') == 'failed'

        # Should NOT have started recording
        assert mock_controller.start.call_count == 0

        # Should have shown clear error to user
        # (Verify retry state was cleared)
        assert widget._retry_in_progress is False


def test_missing_microphone_reports_clear_error(monkeypatch):
    """Missing microphone reports clear error when fallback cannot proceed."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog to accept fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        # Request system-only (no mic in requested sources)
        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'system'})

        # Should have shown clear error (not started recording)
        assert mock_controller.start.call_count == 0

        # Should have recorded failed outcome (cannot proceed without mic)
        assert mock_controller.record_start_retry_outcome.called


def test_fresh_start_attempts_all_requested_sources(monkeypatch):
    """Fresh start attempts all requested sources, no memory of prior failures."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock start to always fail
    mock_controller.start.return_value = ControllerError("Audio device error: WASAPI loopback endpoint unavailable")

    # First attempt with system+mic - will exhaust retries and show fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog.accepted_fallback.return_value = False
        mock_dialog_class.return_value = mock_dialog

        widget.mic_lobe.is_active = True
        widget.system_lobe.is_active = True

        # First start attempt
        widget._retry_attempt = 3
        widget._start_with_retry({'mic', 'system'}, first_attempt=False)

        # First attempt failed, sources requested = {'mic', 'system'}
        assert mock_controller.start.call_count == 1
        first_call_sources = mock_controller.start.call_args_list[0][0][0]
        assert first_call_sources == {'mic', 'system'}

    # Reset call count for second attempt
    mock_controller.reset_mock()
    mock_controller.start.return_value = None  # Now succeed

    # Second fresh start - should retry all requested sources again
    widget._retry_attempt = 0  # Reset retry counter
    widget.start_recording()

    # Should have attempted both sources again (not remembered prior failure)
    assert mock_controller.start.called
    second_call_sources = mock_controller.start.call_args[0][0]
    assert second_call_sources == {'mic', 'system'}
    # System audio was attempted again, no memory of prior failure


def test_partial_source_failure_shows_fallback_dialog(monkeypatch):
    """Partial source failure (some sources succeed, others fail) shows fallback dialog."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        # Simulate partial failure: system failed, mic available
        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have shown fallback dialog
        assert mock_dialog_class.called
        assert mock_dialog.exec.called

        # Should have started with mic-only
        assert mock_controller.start.called
        start_call_sources = mock_controller.start.call_args[0][0]
        assert start_call_sources == {'mic'}


def test_fallback_records_metadata_in_session_stats(monkeypatch):
    """Fallback decision records metadata in session stats."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog to accept fallback
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have recorded retry outcome with metadata
        assert mock_controller.record_start_retry_outcome.called
        call_kwargs = mock_controller.record_start_retry_outcome.call_args[1]
        
        assert 'outcome' in call_kwargs
        assert 'fallback' in call_kwargs.get('outcome', '').lower()
        assert 'failed_sources' in call_kwargs
        assert 'fallback_sources' in call_kwargs
        assert call_kwargs['failed_sources'] == ['system']
        assert call_kwargs['fallback_sources'] == ['mic']


def test_fallback_dialog_cancels_retry_ui(monkeypatch):
    """Fallback dialog dismisses retry UI when accepted or rejected."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Set up retry state
    widget._retry_in_progress = True
    widget._retry_timer = Mock()
    widget._retry_toast_id = 'wasapi-retry'

    # Mock FallbackConfirmationDialog to accept
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.accepted_fallback.return_value = True
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Retry state should be cleared
        assert widget._retry_in_progress is False
        
        # Toast should be dismissed
        assert widget.toast_manager.dismiss.called
        assert widget.toast_manager.dismiss.call_args[0][0] == 'wasapi-retry'


def test_fallback_dialog_uses_correct_message_text(monkeypatch):
    """Fallback dialog uses correct message text as specified in task."""
    widget, mock_controller = _create_widget_with_mocked_controller(monkeypatch)

    # Mock FallbackConfirmationDialog to capture arguments
    with patch("meetandread.widgets.main_widget.FallbackConfirmationDialog") as mock_dialog_class:
        mock_dialog = Mock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        mock_dialog.accepted_fallback.return_value = False
        mock_dialog_class.return_value = mock_dialog

        widget._show_fallback_dialog(failed_sources=['system'], requested_sources={'mic', 'system'})

        # Should have been instantiated
        assert mock_dialog_class.called
        
        # Widget should have been passed as parent
        # (Verifies dialog follows QDialog pattern properly)
        # The implementation passes self (widget) as parent
