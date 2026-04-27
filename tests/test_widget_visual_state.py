"""Tests for WidgetVisualState enum and _WidgetVisualStateMachine.

Covers T01 must-haves:
- Enum values IDLE / RECORDING / PROCESSING exist
- transition_to() changes state and resets progress to 0.0
- tick() advances progress toward 1.0 and clamps at 1.0
- current_opacity() returns eased (quadratic ease-out) values
- current_properties() returns correct dict shape
- State transitions are logged at debug level
- Integration: MeetAndReadWidget._visual_state initialises to IDLE
- Integration: _on_controller_state_change drives state machine transitions
"""

import logging

import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF

from meetandread.widgets.main_widget import (
    WidgetVisualState,
    _WidgetVisualStateMachine,
)


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Unit tests — pure state machine, no Qt widget needed
# ---------------------------------------------------------------------------

class TestWidgetVisualStateEnum:
    """Enum values exist and are distinct."""

    def test_enum_members_exist(self):
        assert WidgetVisualState.IDLE is not None
        assert WidgetVisualState.RECORDING is not None
        assert WidgetVisualState.PROCESSING is not None

    def test_enum_members_distinct(self):
        values = [WidgetVisualState.IDLE, WidgetVisualState.RECORDING,
                  WidgetVisualState.PROCESSING]
        assert len(set(values)) == 3


class TestWidgetVisualStateMachine:
    """State machine transition and easing behaviour."""

    def test_initial_state(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        assert sm.current == WidgetVisualState.IDLE
        assert sm.progress == 1.0  # settled

    def test_transition_to_resets_progress(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.RECORDING)
        assert sm.current == WidgetVisualState.RECORDING
        assert sm.previous == WidgetVisualState.IDLE
        assert sm.progress == 0.0

    def test_transition_to_same_settled_is_noop(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.IDLE)
        assert sm.progress == 1.0  # unchanged

    def test_tick_advances_progress(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.RECORDING)
        assert sm.progress == 0.0
        sm.tick()
        assert 0.0 < sm.progress < 1.0

    def test_tick_converges_to_one(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.RECORDING)
        for _ in range(20):  # well past 6-frame expected convergence
            sm.tick()
        assert sm.progress == 1.0

    def test_current_opacity_eased(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.RECORDING)
        # At progress=0.0, eased opacity should be 0.0
        assert sm.current_opacity() == pytest.approx(0.0)
        # After one tick (progress ≈ 1/6), ease-out(1/6) = 1-(5/6)^2 ≈ 0.3056
        sm.tick()
        expected = 1.0 - (1.0 - 1.0 / 6) ** 2
        assert sm.current_opacity() == pytest.approx(expected, abs=1e-4)
        # Fully settled → opacity 1.0
        for _ in range(20):
            sm.tick()
        assert sm.current_opacity() == pytest.approx(1.0)

    def test_current_properties_shape(self):
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.PROCESSING)
        props = sm.current_properties()
        assert props["state"] == WidgetVisualState.PROCESSING
        assert "opacity" in props
        assert "settled" in props
        assert props["settled"] is False
        # After convergence
        for _ in range(20):
            sm.tick()
        props = sm.current_properties()
        assert props["settled"] is True

    def test_ease_out_curve(self):
        """Quadratic ease-out: 1-(1-t)^2."""
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        assert sm._ease_out(0.0) == pytest.approx(0.0)
        assert sm._ease_out(0.5) == pytest.approx(0.75)
        assert sm._ease_out(1.0) == pytest.approx(1.0)

    def test_transition_log_message(self, caplog):
        """State transitions should emit a debug log."""
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        with caplog.at_level(logging.DEBUG, logger="root"):
            sm.transition_to(WidgetVisualState.RECORDING)
        assert any("WidgetVisualState" in r.message and "IDLE" in r.message
                    for r in caplog.records)

    def test_mid_transition_retarget(self):
        """Retargeting mid-transition should not jump visually."""
        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        sm.transition_to(WidgetVisualState.RECORDING)
        sm.tick()  # progress ≈ 1/6
        # Now retarget to PROCESSING while still transitioning
        sm.transition_to(WidgetVisualState.PROCESSING)
        assert sm.current == WidgetVisualState.PROCESSING
        assert sm.previous == WidgetVisualState.RECORDING
        assert sm.progress == 0.0  # reset for new transition


# ---------------------------------------------------------------------------
# Integration — widget initialises and drives state machine
# ---------------------------------------------------------------------------

class _FakeScreenGeometry:
    """Minimal stand-in for QScreen.geometry()."""

    def __init__(self, width=1920, height=1080):
        self._w = width
        self._h = height

    def width(self):
        return self._w

    def height(self):
        return self._h

    def contains(self, point):
        return 0 <= point.x() < self._w and 0 <= point.y() < self._h


class TestWidgetIntegration:
    """Verify the state machine is wired into MeetAndReadWidget."""

    @pytest.fixture
    def widget(self, qapp):
        from unittest.mock import patch, MagicMock
        from meetandread.widgets.main_widget import MeetAndReadWidget

        fake_geo = _FakeScreenGeometry()
        fake_screen = MagicMock()
        fake_screen.geometry.return_value = fake_geo
        with patch("meetandread.widgets.main_widget.QApplication.primaryScreen",
                    return_value=fake_screen), \
             patch("meetandread.widgets.main_widget.QApplication.screens",
                    return_value=[fake_screen]), \
             patch("meetandread.widgets.main_widget.get_config", return_value=None), \
             patch("meetandread.widgets.main_widget.save_config"):
            w = MeetAndReadWidget()
        # Stub floating panels to avoid side effects in integration tests
        w._floating_transcript_panel = MagicMock()
        w._floating_transcript_panel.isVisible.return_value = False
        w._floating_settings_panel = MagicMock()
        w._floating_settings_panel.isVisible.return_value = False
        yield w
        w.close()

    def test_widget_initial_state_is_idle(self, widget):
        assert widget._visual_state.current == WidgetVisualState.IDLE

    def test_controller_recording_transitions_state(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        assert widget._visual_state.current == WidgetVisualState.RECORDING

    def test_controller_stopping_transitions_state(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.STOPPING)
        assert widget._visual_state.current == WidgetVisualState.PROCESSING

    def test_controller_idle_transitions_state(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.IDLE)
        assert widget._visual_state.current == WidgetVisualState.IDLE

    def test_controller_error_transitions_to_idle(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.ERROR)
        assert widget._visual_state.current == WidgetVisualState.IDLE

    def test_update_animations_ticks_state_machine(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        assert widget._visual_state.progress == 0.0
        widget._update_animations()
        assert widget._visual_state.progress > 0.0


# ---------------------------------------------------------------------------
# T02: Glass opacity integration
# ---------------------------------------------------------------------------

class TestGlassOpacity:
    """Window opacity smoothly transitions between idle (0.87) and active (1.0)."""

    IDLE_OPACITY = 0.87
    ACTIVE_OPACITY = 1.0

    @pytest.fixture
    def widget(self, qapp):
        from unittest.mock import patch, MagicMock
        from meetandread.widgets.main_widget import MeetAndReadWidget

        fake_geo = _FakeScreenGeometry()
        fake_screen = MagicMock()
        fake_screen.geometry.return_value = fake_geo
        with patch("meetandread.widgets.main_widget.QApplication.primaryScreen",
                    return_value=fake_screen), \
             patch("meetandread.widgets.main_widget.QApplication.screens",
                    return_value=[fake_screen]), \
             patch("meetandread.widgets.main_widget.get_config", return_value=None), \
             patch("meetandread.widgets.main_widget.save_config"):
            w = MeetAndReadWidget()
        w._floating_transcript_panel = MagicMock()
        w._floating_transcript_panel.isVisible.return_value = False
        w._floating_settings_panel = MagicMock()
        w._floating_settings_panel.isVisible.return_value = False
        yield w
        w.close()

    # -- Initial state -------------------------------------------------------

    def test_initial_opacity_is_idle(self, widget):
        """Widget starts translucent (0.87) since initial state is IDLE."""
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    # -- Idle → Recording transition -----------------------------------------

    def test_idle_to_recording_opacity_starts_at_idle(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Before any animation ticks, opacity should still be at idle
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    def test_idle_to_recording_opacity_transitions_toward_active(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._update_animations()  # one tick
        opacity = widget.windowOpacity()
        assert self.IDLE_OPACITY < opacity < self.ACTIVE_OPACITY

    def test_idle_to_recording_opacity_converges_to_active(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(self.ACTIVE_OPACITY, abs=0.01)

    # -- Recording → Idle transition -----------------------------------------

    def test_recording_to_idle_opacity_converges_to_idle(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(20):
            widget._update_animations()
        # Now go back to idle
        widget._on_controller_state_change(ControllerState.IDLE)
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    # -- Processing state opacity --------------------------------------------

    def test_processing_opacity_is_active(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(20):
            widget._update_animations()
        widget._on_controller_state_change(ControllerState.STOPPING)
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(self.ACTIVE_OPACITY, abs=0.01)

    # -- showEvent / hideEvent opacity reset ---------------------------------

    def test_hide_event_resets_opacity_to_one(self, widget):
        """hideEvent sets opacity to 1.0 to avoid stale low-opacity on reappear."""
        from PyQt6.QtGui import QHideEvent
        widget.hideEvent(QHideEvent())
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)

    def test_show_event_resets_opacity_for_idle_state(self, widget):
        """showEvent sets correct opacity for current visual state."""
        from PyQt6.QtGui import QShowEvent
        # Widget is idle, opacity should be 0.87
        widget.setWindowOpacity(1.0)  # force wrong value
        widget.showEvent(QShowEvent())
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    def test_show_event_resets_opacity_for_recording_state(self, widget):
        """showEvent sets 1.0 opacity if currently recording."""
        from PyQt6.QtGui import QShowEvent
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(20):
            widget._update_animations()
        widget.setWindowOpacity(self.IDLE_OPACITY)  # force wrong value
        widget.showEvent(QShowEvent())
        assert widget.windowOpacity() == pytest.approx(self.ACTIVE_OPACITY, abs=0.01)

    # -- Debug logging during transition -------------------------------------

    def test_opacity_transition_logs_debug(self, widget, caplog):
        from meetandread.recording import ControllerState
        with caplog.at_level(logging.DEBUG, logger="root"):
            widget._on_controller_state_change(ControllerState.RECORDING)
            widget._update_animations()  # one tick, transition not settled
        assert any("Glass opacity" in r.message for r in caplog.records)
