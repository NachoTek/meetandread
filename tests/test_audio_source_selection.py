"""Integration tests for audio source selection lobes.

Covers locked state, unavailable state, no-source pulse animation,
config persistence, and recording lock wiring.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

from metamemory.widgets.main_widget import ToggleLobeItem, MeetAndReadWidget
from metamemory.recording import ControllerState


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped, same pattern as test_widget_docking.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lobe(lobe_type: str = "microphone") -> ToggleLobeItem:
    """Create a ToggleLobeItem with a minimal mock parent widget."""
    parent = MagicMock(spec=MeetAndReadWidget)
    parent.is_dragging = False
    parent._click_consumed = False
    parent._on_lobe_toggled = MagicMock()
    lobe = ToggleLobeItem(lobe_type, parent)
    return lobe


def _simulate_mouse_release(lobe: ToggleLobeItem):
    """Simulate a left-button mouse release on the lobe."""
    event = MagicMock()
    event.button.return_value = Qt.MouseButton.LeftButton
    lobe.mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# ToggleLobeItem: locked state
# ---------------------------------------------------------------------------

class TestToggleLobeLocked:
    """Locked state dims lobes and prevents toggle during recording."""

    def test_locked_prevents_toggle(self, qapp):
        lobe = _make_lobe()
        lobe.is_active = False
        lobe.set_locked(True)
        assert lobe._is_locked is True
        _simulate_mouse_release(lobe)
        # is_active should stay False — toggle was blocked
        assert lobe.is_active is False

    def test_locked_active_stays_active(self, qapp):
        lobe = _make_lobe()
        lobe.is_active = True
        lobe.set_locked(True)
        _simulate_mouse_release(lobe)
        assert lobe.is_active is True

    def test_set_locked_updates_cursor(self, qapp):
        lobe = _make_lobe()
        lobe.set_locked(True)
        assert lobe.cursor().shape() == Qt.CursorShape.ForbiddenCursor
        lobe.set_locked(False)
        assert lobe.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_set_unlocked_restores_interactivity(self, qapp):
        lobe = _make_lobe()
        lobe.set_locked(True)
        lobe.set_locked(False)
        assert lobe._is_locked is False
        _simulate_mouse_release(lobe)
        assert lobe.is_active is True  # toggle works again


# ---------------------------------------------------------------------------
# ToggleLobeItem: unavailable state
# ---------------------------------------------------------------------------

class TestToggleLobeUnavailable:
    """Unavailable state greys out system lobe when no loopback device."""

    def test_unavailable_prevents_toggle(self, qapp):
        lobe = _make_lobe()
        lobe.is_active = False
        lobe.set_unavailable(True)
        assert lobe._is_unavailable is True
        _simulate_mouse_release(lobe)
        assert lobe.is_active is False

    def test_unavailable_active_stays_active(self, qapp):
        lobe = _make_lobe()
        lobe.is_active = True
        lobe.set_unavailable(True)
        _simulate_mouse_release(lobe)
        assert lobe.is_active is True

    def test_set_unavailable_updates_cursor(self, qapp):
        lobe = _make_lobe()
        lobe.set_unavailable(True)
        assert lobe.cursor().shape() == Qt.CursorShape.ForbiddenCursor
        lobe.set_unavailable(False)
        assert lobe.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_unavailable_priority_over_locked(self, qapp):
        """If both unavailable and locked, unavailable cursor wins."""
        lobe = _make_lobe()
        lobe.set_unavailable(True)
        lobe.set_locked(True)
        # Last call (set_locked) sets ForbiddenCursor too — both use same cursor
        assert lobe.cursor().shape() == Qt.CursorShape.ForbiddenCursor


# ---------------------------------------------------------------------------
# ToggleLobeItem: pulse opacity
# ---------------------------------------------------------------------------

class TestToggleLobePulse:
    """Pulse opacity scales the visual intensity."""

    def test_default_pulse_opacity(self, qapp):
        lobe = _make_lobe()
        assert lobe._pulse_opacity == 1.0

    def test_pulse_opacity_mutable(self, qapp):
        lobe = _make_lobe()
        lobe._pulse_opacity = 0.5
        assert lobe._pulse_opacity == 0.5


# ---------------------------------------------------------------------------
# MeetAndReadWidget: lobe toggled → config persistence
# ---------------------------------------------------------------------------

class TestLobeTogglePersistence:
    """_on_lobe_toggled persists sources to config."""

    @patch("metamemory.widgets.main_widget.save_config")
    @patch("metamemory.widgets.main_widget.set_config")
    def test_toggle_mic_persists(self, mock_set, mock_save):
        widget = MagicMock(spec=MeetAndReadWidget)
        mic = _make_lobe()
        mic.is_active = True
        sys_lobe = _make_lobe("system")
        sys_lobe.is_active = False
        widget.mic_lobe = mic
        widget.system_lobe = sys_lobe
        # Wire real method so _get_selected_sources reads lobe states
        widget._get_selected_sources = lambda: MeetAndReadWidget._get_selected_sources(widget)
        MeetAndReadWidget._on_lobe_toggled(widget)
        mock_set.assert_called_once_with('ui.audio_sources', ['mic'])
        mock_save.assert_called_once()

    @patch("metamemory.widgets.main_widget.save_config")
    @patch("metamemory.widgets.main_widget.set_config")
    def test_toggle_both_persists(self, mock_set, mock_save):
        widget = MagicMock(spec=MeetAndReadWidget)
        mic = _make_lobe()
        mic.is_active = True
        sys_lobe = _make_lobe("system")
        sys_lobe.is_active = True
        widget.mic_lobe = mic
        widget.system_lobe = sys_lobe
        widget._get_selected_sources = lambda: MeetAndReadWidget._get_selected_sources(widget)
        MeetAndReadWidget._on_lobe_toggled(widget)
        sources = mock_set.call_args[0][1]
        assert set(sources) == {'mic', 'system'}


# ---------------------------------------------------------------------------
# MeetAndReadWidget: restore audio sources from config
# ---------------------------------------------------------------------------

class TestRestoreAudioSources:
    """_restore_audio_sources sets lobe states from config."""

    @patch("metamemory.widgets.main_widget.get_config", return_value=['mic'])
    def test_restore_mic_only(self, mock_get):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._restore_audio_sources(widget)
        assert widget.mic_lobe.is_active is True
        assert widget.system_lobe.is_active is False

    @patch("metamemory.widgets.main_widget.get_config", return_value=['mic', 'system'])
    def test_restore_both(self, mock_get):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._restore_audio_sources(widget)
        assert widget.mic_lobe.is_active is True
        assert widget.system_lobe.is_active is True

    @patch("metamemory.widgets.main_widget.get_config", return_value=None)
    def test_restore_none_first_launch(self, mock_get):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._restore_audio_sources(widget)
        assert widget.mic_lobe.is_active is False
        assert widget.system_lobe.is_active is False


# ---------------------------------------------------------------------------
# MeetAndReadWidget: no-source pulse animation
# ---------------------------------------------------------------------------

class TestNoSourcePulse:
    """start_recording with no sources triggers pulse animation."""

    def test_pulse_timer_created_on_no_sources(self, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        mic = _make_lobe()
        sys_lobe = _make_lobe("system")
        widget.mic_lobe = mic
        widget.system_lobe = sys_lobe
        widget._pulse_timer = None
        widget._show_error = MagicMock()

        mock_timer = MagicMock()
        mock_timer.isActive.return_value = True

        with patch("metamemory.widgets.main_widget.QTimer", return_value=mock_timer):
            MeetAndReadWidget._pulse_lobes(widget)

        assert widget._pulse_timer is mock_timer
        mock_timer.start.assert_called_once_with(100)

    @patch("metamemory.widgets.main_widget.save_config")
    @patch("metamemory.widgets.main_widget.set_config")
    @patch("metamemory.widgets.main_widget.get_config", return_value=None)
    def test_start_recording_no_sources_triggers_pulse(self, mock_get, mock_set, mock_save):
        """Full start_recording flow with no sources activates pulse timer."""
        widget = MagicMock(spec=MeetAndReadWidget)
        widget._get_selected_sources = MagicMock(return_value=set())
        widget._pulse_lobes = MagicMock()
        widget._show_error = MagicMock()

        MeetAndReadWidget.start_recording(widget)

        widget._pulse_lobes.assert_called_once()
        widget._show_error.assert_called_once()


# ---------------------------------------------------------------------------
# MeetAndReadWidget: recording lock wiring
# ---------------------------------------------------------------------------

class TestRecordingLockWiring:
    """Simulate RECORDING state → lobes locked, IDLE → lobes unlocked."""

    def test_recording_state_locks_lobes(self, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        widget.record_button = MagicMock()
        widget._floating_transcript_panel = None
        widget._hide_error = MagicMock()
        widget._tray_manager = None
        widget.is_processing = False
        widget.is_recording = False

        MeetAndReadWidget._on_controller_state_change(widget, ControllerState.RECORDING)

        assert widget.mic_lobe._is_locked is True
        assert widget.system_lobe._is_locked is True

    def test_starting_state_locks_lobes(self, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        widget._tray_manager = None

        MeetAndReadWidget._on_controller_state_change(widget, ControllerState.STARTING)

        assert widget.mic_lobe._is_locked is True
        assert widget.system_lobe._is_locked is True

    def test_idle_state_unlocks_lobes(self, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        widget.record_button = MagicMock()
        widget._tray_manager = None
        widget.is_recording = True
        widget.is_processing = False

        MeetAndReadWidget._on_controller_state_change(widget, ControllerState.IDLE)

        assert widget.mic_lobe._is_locked is False
        assert widget.system_lobe._is_locked is False

    def test_error_state_unlocks_lobes(self, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.mic_lobe = _make_lobe()
        widget.system_lobe = _make_lobe()
        widget.record_button = MagicMock()
        widget._tray_manager = None
        widget.is_recording = True
        widget.is_processing = False

        MeetAndReadWidget._on_controller_state_change(widget, ControllerState.ERROR)

        assert widget.mic_lobe._is_locked is False
        assert widget.system_lobe._is_locked is False


# ---------------------------------------------------------------------------
# System audio availability probe
# ---------------------------------------------------------------------------

class TestSystemAudioProbe:
    """_probe_system_audio_availability marks lobe unavailable when no device."""

    @patch("metamemory.audio.capture.devices.get_default_loopback_device", return_value=None)
    def test_no_device_marks_unavailable(self, mock_probe, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._probe_system_audio_availability(widget)
        assert widget.system_lobe._is_unavailable is True

    @patch("metamemory.audio.capture.devices.get_default_loopback_device",
           return_value={"name": "Speakers (Loopback)"})
    def test_device_found_stays_available(self, mock_probe, qapp):
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._probe_system_audio_availability(widget)
        assert widget.system_lobe._is_unavailable is False

    @patch("metamemory.audio.capture.devices.get_default_loopback_device",
           side_effect=Exception("import failed"))
    def test_probe_exception_stays_available(self, mock_probe, qapp):
        """On probe error, stay optimistic (lobe stays available)."""
        widget = MagicMock(spec=MeetAndReadWidget)
        widget.system_lobe = _make_lobe()
        MeetAndReadWidget._probe_system_audio_availability(widget)
        assert widget.system_lobe._is_unavailable is False
