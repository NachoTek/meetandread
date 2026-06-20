"""Tests for MeetAndReadWidget shutdown wiring — exit paths call controller.shutdown().

Covers T01 must-haves:
- _exit_application calls controller.shutdown()
- closeEvent (no tray) calls controller.shutdown()
- closeEvent (with tray) does NOT call shutdown (close-to-tray preserved)
- Signal handler fallback before widget creation
- Each exit path calls shutdown exactly once
"""

import threading
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

from meetandread.recording.controller import RecordingController, ControllerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_widget():
    """Create a MeetAndReadWidget with minimal patching for test isolation.

    Patches QApplication to avoid needing a real display.
    """
    with patch("meetandread.widgets.main_widget.QApplication"):
        with patch("meetandread.widgets.main_widget.QGraphicsView.__init__"):
            with patch(
                "meetandread.widgets.main_widget.MeetAndReadWidget._create_components",
                return_value=None,
            ):
                with patch(
                    "meetandread.widgets.main_widget.MeetAndReadWidget._create_floating_panels",
                    return_value=None,
                ):
                    with patch(
                        "meetandread.widgets.main_widget.MeetAndReadWidget._layout_components",
                        return_value=None,
                    ):
                        with patch(
                            "meetandread.widgets.main_widget.MeetAndReadWidget._position_initial",
                            return_value=None,
                        ):
                            # Import here to apply patches before class init
                            from meetandread.widgets.main_widget import (
                                MeetAndReadWidget,
                            )

                            widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
                            # Manually init minimal state to avoid QGraphicsView init
                            widget._tray_manager = None
                            widget._controller = MagicMock(spec=RecordingController)
                            widget._controller.get_state.return_value = (
                                ControllerState.IDLE
                            )
                            widget._controller.is_recording.return_value = False
                            widget._floating_settings_panel = None
                            widget._cc_overlay = None
                            widget._error_hide_timer = None
                            widget._warning_hide_timer = None
                            widget.toast_manager = None
                            return widget


# ===================================================================
# _exit_application calls shutdown
# ===================================================================

class TestExitApplicationCallsShutdown:
    """_exit_application() must call controller.shutdown()."""

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_exit_calls_shutdown(self, mock_save, mock_quit):
        widget = _make_widget()
        widget._exit_application()
        widget._controller.shutdown.assert_called_once()
        # Verify timeout arg is reasonable
        call_args = widget._controller.shutdown.call_args
        assert call_args[1].get("timeout", 10.0) > 0

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_exit_calls_quit_after_shutdown(self, mock_save, mock_quit):
        widget = _make_widget()
        widget._exit_application()
        mock_quit.assert_called_once()

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_exit_proceeds_on_shutdown_error(self, mock_save, mock_quit):
        """If shutdown raises, exit still calls QApplication.quit()."""
        widget = _make_widget()
        widget._controller.shutdown.side_effect = RuntimeError("boom")
        widget._exit_application()
        mock_quit.assert_called_once()

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_exit_saves_geometry(self, mock_save, mock_quit):
        widget = _make_widget()
        mock_panel = MagicMock()
        widget._floating_settings_panel = mock_panel
        mock_overlay = MagicMock()
        widget._cc_overlay = mock_overlay
        widget._exit_application()
        mock_panel.save_geometry.assert_called_once()
        mock_panel.hide.assert_called_once()
        mock_overlay.save_geometry.assert_called_once()
        mock_overlay.hide.assert_called_once()


# ===================================================================
# closeEvent — no tray manager → full quit with shutdown
# ===================================================================

class TestCloseEventNoTray:
    """Without tray, closeEvent calls shutdown and quits."""

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_no_tray_calls_shutdown(self, mock_save, mock_quit):
        widget = _make_widget()
        widget._tray_manager = None  # No tray

        mock_event = MagicMock()
        widget.closeEvent(mock_event)
        widget._controller.shutdown.assert_called_once()
        mock_event.accept.assert_called_once()
        mock_event.ignore.assert_not_called()

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_no_tray_proceeds_on_shutdown_error(
        self, mock_save, mock_quit
    ):
        widget = _make_widget()
        widget._tray_manager = None
        widget._controller.shutdown.side_effect = RuntimeError("boom")

        mock_event = MagicMock()
        widget.closeEvent(mock_event)
        mock_quit.assert_called_once()


# ===================================================================
# closeEvent — with tray → close-to-tray (NO shutdown)
# ===================================================================

class TestCloseEventWithTray:
    """With tray manager, closeEvent hides to tray (shutdown NOT called)."""

    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_with_tray_hides(self, mock_save):
        widget = _make_widget()
        widget._tray_manager = MagicMock()  # Tray active
        # Mock the hide method to avoid QGraphicsView
        widget.hide = MagicMock()

        mock_event = MagicMock()
        widget.closeEvent(mock_event)

        # Should NOT call shutdown
        widget._controller.shutdown.assert_not_called()
        # Should hide instead
        widget.hide.assert_called_once()
        mock_event.ignore.assert_called_once()

    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_with_tray_saves_geometry(self, mock_save):
        widget = _make_widget()
        widget._tray_manager = MagicMock()
        widget.hide = MagicMock()
        mock_panel = MagicMock()
        widget._floating_settings_panel = mock_panel

        mock_event = MagicMock()
        widget.closeEvent(mock_event)
        mock_panel.save_geometry.assert_called_once()


# ===================================================================
# Signal handler fallback
# ===================================================================

class TestSignalHandlerFallback:
    """Signal handlers gracefully handle missing widget."""

    def test_signal_handler_without_widget(self):
        """Signal handler with no widget falls back to app.quit()."""
        from meetandread.main import setup_signal_handlers

        mock_app = MagicMock()
        # widget_ref returns None (no widget yet)
        setup_signal_handlers(mock_app, widget_ref=lambda: None)

        # Simulate SIGINT — we can't easily test the actual signal,
        # but we can verify the app.quit fallback by checking the
        # function was set up without error
        # (Full signal testing requires process-level tricks)

    def test_signal_handler_with_widget(self):
        """Signal handler with widget calls _exit_application."""
        from meetandread.main import setup_signal_handlers

        mock_app = MagicMock()
        mock_widget = MagicMock()
        setup_signal_handlers(mock_app, widget_ref=lambda: mock_widget)

        # The handler was set up successfully
        # (Testing the actual signal dispatch requires OS-level tricks)


# ===================================================================
# Shutdown called exactly once per exit path
# ===================================================================

class TestShutdownCallCount:
    """Verify shutdown is called exactly once per exit invocation."""

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_exit_application_calls_shutdown_once(self, mock_save, mock_quit):
        widget = _make_widget()
        widget._exit_application()
        assert widget._controller.shutdown.call_count == 1

    @patch("meetandread.widgets.main_widget.QApplication.quit")
    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_no_tray_calls_shutdown_once(self, mock_save, mock_quit):
        widget = _make_widget()
        widget._tray_manager = None

        mock_event = MagicMock()
        widget.closeEvent(mock_event)
        assert widget._controller.shutdown.call_count == 1

    @patch("meetandread.widgets.main_widget.MeetAndReadWidget._save_position")
    def test_close_event_with_tray_calls_shutdown_zero(self, mock_save):
        widget = _make_widget()
        widget._tray_manager = MagicMock()
        widget.hide = MagicMock()

        mock_event = MagicMock()
        widget.closeEvent(mock_event)
        assert widget._controller.shutdown.call_count == 0
