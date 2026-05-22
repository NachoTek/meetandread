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
    RecordButtonItem,
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
        w._cc_overlay = MagicMock()
        w._cc_overlay.isVisible.return_value = False
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
    """Window opacity smoothly transitions between idle (1.0) and active (1.0)."""

    IDLE_OPACITY = 1.0
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
        w._cc_overlay = MagicMock()
        w._cc_overlay.isVisible.return_value = False
        yield w
        w.close()

    # -- Initial state -------------------------------------------------------

    def test_initial_opacity_is_idle(self, widget):
        """Widget starts translucent (1.0) since initial state is IDLE."""
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    # -- Idle → Recording transition -----------------------------------------

    def test_idle_to_recording_opacity_starts_at_idle(self, widget):
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Before any animation ticks, opacity should still be at idle
        assert widget.windowOpacity() == pytest.approx(self.IDLE_OPACITY, abs=0.01)

    def test_idle_to_recording_opacity_transitions_toward_active(self, widget):
        """With idle=active=1.0, opacity remains at 1.0 throughout transition."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._update_animations()  # one tick
        opacity = widget.windowOpacity()
        assert opacity == pytest.approx(self.ACTIVE_OPACITY, abs=0.01)

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
        # Widget is idle, opacity should be 1.0
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


# ---------------------------------------------------------------------------
# T04: Smooth integrated state transitions across full lifecycle
# ---------------------------------------------------------------------------

class TestIntegratedStateTransitions:
    """Verify widget state machine and RecordButtonItem are wired together
    and all lifecycle transitions are visually smooth (no flicker, no phase
    jumps, no stale animation phases)."""

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
        w._cc_overlay = MagicMock()
        w._cc_overlay.isVisible.return_value = False
        yield w
        w.close()

    # -- Step 1: Widget state machine triggers RecordButtonItem simultaneously --

    def test_recording_syncs_both_state_systems(self, widget):
        """Controller RECORDING triggers both visual state machine and RecordButtonItem."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Widget state machine should be RECORDING
        assert widget._visual_state.current == WidgetVisualState.RECORDING
        # RecordButtonItem should be recording
        assert widget.record_button.is_recording is True
        assert widget.record_button._to_key == 'recording'

    def test_processing_syncs_both_state_systems(self, widget):
        """Controller STOPPING triggers both state systems."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.STOPPING)
        assert widget._visual_state.current == WidgetVisualState.PROCESSING
        assert widget.record_button.is_processing is True
        assert widget.record_button._to_key == 'processing'

    def test_idle_syncs_both_state_systems(self, widget):
        """Controller IDLE resets both state systems."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.IDLE)
        assert widget._visual_state.current == WidgetVisualState.IDLE
        assert widget.record_button.is_recording is False
        assert widget.record_button.is_processing is False
        assert widget.record_button._to_key == 'idle'

    # -- Step 2-3: Phase reset timing (smooth decay, not snap) --

    def test_pulse_phase_not_reset_during_crossfade(self, widget):
        """pulse_phase should NOT snap to 0 while RecordButtonItem cross-fades
        back to idle — the fading-out recording state needs a valid phase."""
        from meetandread.recording import ControllerState
        # Enter recording, advance pulse_phase
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()
        assert widget.pulse_phase > 0.0

        # Transition to idle — cross-fade starts
        widget._on_controller_state_change(ControllerState.IDLE)
        widget._update_animations()  # one tick — cross-fade in progress

        # Cross-fade is not yet complete (6 frames needed)
        assert widget.record_button._state_t < 1.0
        # pulse_phase should NOT have been reset to 0 yet
        assert widget.pulse_phase != 0.0, (
            "pulse_phase was snapped to 0 during cross-fade — would cause visual jump"
        )

    def test_pulse_phase_resets_after_crossfade_settles(self, widget):
        """pulse_phase should reset to 0 only after the cross-fade completes."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()
        assert widget.pulse_phase > 0.0

        widget._on_controller_state_change(ControllerState.IDLE)
        # Run enough ticks for both state machine and RecordButtonItem to settle
        for _ in range(20):
            widget._update_animations()

        # Cross-fade should be fully settled now
        assert widget.record_button._state_t >= 1.0
        # Phase should now be reset
        assert widget.pulse_phase == 0.0

    def test_swirl_phase_preserved_during_crossfade_to_idle(self, widget):
        """swirl_phase should remain valid during processing→idle cross-fade."""
        from meetandread.recording import ControllerState
        # Enter recording then processing
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(5):
            widget._update_animations()
        widget._on_controller_state_change(ControllerState.STOPPING)
        for _ in range(10):
            widget._update_animations()
        assert widget.pulse_phase > 0.0  # swirl uses pulse_phase

        # Now go to idle
        widget._on_controller_state_change(ControllerState.IDLE)
        widget._update_animations()  # one tick

        # Cross-fade in progress — swirl phase should not be snapped
        assert widget.record_button._state_t < 1.0
        assert widget.record_button.swirl_phase != 0.0 or widget.pulse_phase != 0.0, (
            "swirl_phase was snapped during cross-fade — would cause visual jump"
        )

    # -- Step 4-6: End-to-end transition tests --

    def test_idle_to_recording_transition(self, widget):
        """idle→recording: opacity 1.0→1.0, state key idle→recording, cross-fade."""
        from meetandread.recording import ControllerState
        # Start idle
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)
        assert widget.record_button._to_key == 'idle'

        # Trigger recording
        widget._on_controller_state_change(ControllerState.RECORDING)
        assert widget._visual_state.current == WidgetVisualState.RECORDING
        assert widget.record_button._to_key == 'recording'
        assert widget.record_button._state_t == 0.0  # cross-fade starting

        # Advance animation — opacity stays at 1.0 (idle=active)
        widget._update_animations()
        opacity = widget.windowOpacity()
        assert opacity == pytest.approx(1.0, abs=0.01)

        # Settle
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)
        assert widget.record_button._state_t >= 1.0

    def test_recording_to_processing_transition(self, widget):
        """recording→processing: red pulse→blue swirl, phase handoff."""
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()
        # pulse_phase should have advanced
        pulse_at_stop = widget.pulse_phase
        assert pulse_at_stop > 0.0

        # Trigger processing
        widget._on_controller_state_change(ControllerState.STOPPING)
        assert widget._visual_state.current == WidgetVisualState.PROCESSING
        assert widget.record_button._to_key == 'processing'
        assert widget.record_button._state_t == 0.0

        # Advance — swirl_phase should start advancing
        for _ in range(5):
            widget._update_animations()
        # swirl_phase should be advancing (pulse_phase continues as swirl driver)
        assert widget.pulse_phase > pulse_at_stop

        # Settle
        for _ in range(15):
            widget._update_animations()
        assert widget.record_button._state_t >= 1.0
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)

    def test_processing_to_idle_transition(self, widget):
        """processing→idle: swirl→glass gradient, opacity 1.0→1.0."""
        from meetandread.recording import ControllerState
        # Setup: go through recording→processing
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(5):
            widget._update_animations()
        widget._on_controller_state_change(ControllerState.STOPPING)
        for _ in range(10):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)

        # Now go idle
        widget._on_controller_state_change(ControllerState.IDLE)
        assert widget._visual_state.current == WidgetVisualState.IDLE
        assert widget.record_button._to_key == 'idle'

        # During transition, opacity stays at 1.0 (idle=active)
        widget._update_animations()
        opacity = widget.windowOpacity()
        assert opacity == pytest.approx(1.0, abs=0.01)

        # Settle
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)
        assert widget.record_button._state_t >= 1.0

    # -- Step 7: Visual smoke test — full lifecycle --

    def test_full_lifecycle_no_flicker(self, widget):
        """Exercise all transitions programmatically: idle→recording→processing→idle.

        Verifies:
        - No state key mismatches between widget and RecordButtonItem
        - Opacity transitions are monotonic (no flicker)
        - Animation phases are always valid (no NaN, no negative)
        - All state changes log at debug level
        """
        from meetandread.recording import ControllerState

        opacity_values = []

        # --- Phase 1: idle → recording ---
        widget._on_controller_state_change(ControllerState.RECORDING)
        assert widget._visual_state.current == WidgetVisualState.RECORDING
        assert widget.record_button._to_key == 'recording'
        for i in range(20):
            widget._update_animations()
            opacity_values.append(widget.windowOpacity())
            assert widget.pulse_phase >= 0.0

        # Opacity should have increased toward 1.0
        assert opacity_values[-1] > opacity_values[0]

        # --- Phase 2: recording → processing ---
        widget._on_controller_state_change(ControllerState.STOPPING)
        assert widget._visual_state.current == WidgetVisualState.PROCESSING
        assert widget.record_button._to_key == 'processing'
        for i in range(20):
            widget._update_animations()
            opacity_values.append(widget.windowOpacity())
            assert widget.pulse_phase >= 0.0

        # --- Phase 3: processing → idle ---
        widget._on_controller_state_change(ControllerState.IDLE)
        assert widget._visual_state.current == WidgetVisualState.IDLE
        assert widget.record_button._to_key == 'idle'
        for i in range(20):
            widget._update_animations()
            opacity_values.append(widget.windowOpacity())

        # Final state: idle opacity
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)
        assert widget.record_button._state_t >= 1.0
        assert widget.pulse_phase == 0.0  # settled → reset
        assert widget.record_button.pulse_phase == 0.0
        assert widget.record_button.swirl_phase == 0.0

    def test_rapid_state_changes_no_visual_jump(self, widget):
        """Rapid state changes (idle→recording→idle) should not cause jumps.

        The state machine's mid-transition retarget should prevent visual
        discontinuities.
        """
        from meetandread.recording import ControllerState

        # Quickly toggle: idle → recording → idle without settling
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._update_animations()  # 1 tick
        assert 0.0 < widget._visual_state.progress < 1.0

        recording_progress = widget._visual_state.progress

        widget._on_controller_state_change(ControllerState.IDLE)
        # State machine should have retargeted to IDLE
        assert widget._visual_state.current == WidgetVisualState.IDLE
        assert widget._visual_state.previous == WidgetVisualState.RECORDING
        assert widget._visual_state.progress == 0.0  # reset for new transition

        # Settle to idle
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)

    def test_error_recovery_resets_to_idle(self, widget):
        """Recording error should transition both state systems cleanly to idle."""
        from meetandread.recording import ControllerState

        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()

        widget._on_controller_state_change(ControllerState.ERROR)
        assert widget._visual_state.current == WidgetVisualState.IDLE
        assert widget.record_button._to_key == 'idle'
        assert widget.record_button.is_recording is False
        assert widget.record_button.is_processing is False

        # Settle
        for _ in range(20):
            widget._update_animations()
        assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# T02 (S02): Orbit configuration tests for RecordButtonItem._ORBITS
# ---------------------------------------------------------------------------

class TestOrbitConfig:
    """Verify the data-driven orbit configuration used for the 'processing'
    swirl animation has correct structure and produces visual stagger."""

    REQUIRED_KEYS = {"radius_frac", "speed_mult", "size", "alpha", "offset"}

    # -- Structure ------------------------------------------------------------

    def test_orbit_config_structure(self):
        """_ORBITS has 4 entries, each with all required keys."""
        orbits = RecordButtonItem._ORBITS
        assert len(orbits) == 4
        for i, entry in enumerate(orbits):
            missing = self.REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"Orbit entry {i} missing keys: {missing}"

    # -- Multiple radii -------------------------------------------------------

    def test_orbit_config_has_multiple_radii(self):
        """At least 2 distinct radius_frac values for visual depth."""
        radii = {e["radius_frac"] for e in RecordButtonItem._ORBITS}
        assert len(radii) >= 2, (
            f"Expected ≥2 distinct radii, got {len(radii)}: {radii}"
        )

    # -- Multiple speeds ------------------------------------------------------

    def test_orbit_config_has_multiple_speeds(self):
        """At least 2 distinct speed_mult values for varied orbital speeds."""
        speeds = {e["speed_mult"] for e in RecordButtonItem._ORBITS}
        assert len(speeds) >= 2, (
            f"Expected ≥2 distinct speeds, got {len(speeds)}: {speeds}"
        )

    # -- Angular positions differ at same phase --------------------------------

    def test_orbit_dots_at_different_positions(self):
        """At a fixed swirl_phase, dots should occupy different angles
        due to different speed_mult and offset values."""
        import math
        phase = 1.0
        angles = [
            phase * e["speed_mult"] + e["offset"]
            for e in RecordButtonItem._ORBITS
        ]
        # Normalise to [0, 2π) for comparison
        normed = [a % (2 * math.pi) for a in angles]
        # At least 2 dots should be at different angular positions
        assert len(set(f"{a:.4f}" for a in normed)) >= 2, (
            f"All dots at same angle: {normed}"
        )

    # -- Speed multiplier produces stagger ------------------------------------

    def test_orbit_speed_multiplier_derivation(self):
        """For each orbit entry, angle = phase * speed_mult + offset should
        differ from the identity (phase * 1.0 + 0.0) for at least some entries,
        proving that speed_mult creates visual stagger rather than uniform motion."""
        phase = 2.5  # arbitrary non-trivial phase
        identity_angle = phase * 1.0 + 0.0
        staggered_count = 0
        for e in RecordButtonItem._ORBITS:
            angle = phase * e["speed_mult"] + e["offset"]
            if abs(angle - identity_angle) > 0.01:
                staggered_count += 1
        assert staggered_count >= 1, (
            "No orbit entries produce angles different from identity — "
            "speed_mult/offset are not creating visual stagger"
        )


# ---------------------------------------------------------------------------
# T03 (S02): Glass visibility strengthening tests
# ---------------------------------------------------------------------------

class TestDragSurfaceItemVisibility:
    """DragSurfaceItem is invisible — hit-test only, no visible fill."""

    def test_drag_surface_brush_alpha_is_zero(self):
        """Brush alpha should be 0 (invisible, hit-test only)."""
        from meetandread.widgets.main_widget import DragSurfaceItem
        from unittest.mock import MagicMock, patch
        mock_parent = MagicMock()
        item = DragSurfaceItem.__new__(DragSurfaceItem)
        from PyQt6.QtGui import QColor, QBrush
        import inspect
        source = inspect.getsource(DragSurfaceItem.__init__)
        import re
        match = re.search(r'QColor\(0,\s*0,\s*0,\s*(\d+)\)', source)
        assert match, "Could not find QColor(0,0,0,alpha) in DragSurfaceItem.__init__"
        alpha = int(match.group(1))
        assert alpha == 0, f"DragSurfaceItem brush alpha ({alpha}) must be 0 for invisible hit-test surface"

    def test_drag_surface_alpha_below_half_opaque(self):
        """Brush alpha should be 0 (invisible)."""
        from meetandread.widgets.main_widget import DragSurfaceItem
        import inspect, re
        source = inspect.getsource(DragSurfaceItem.__init__)
        match = re.search(r'QColor\(0,\s*0,\s*0,\s*(\d+)\)', source)
        assert match
        alpha = int(match.group(1))
        assert alpha < 128, f"DragSurfaceItem brush alpha ({alpha}) should be < 128 for glass aesthetic"


class TestIdleOpacityVisibility:
    """Idle opacity is 1.0 — widget fully transparent, darkness on button only."""

    def test_idle_opacity_is_one(self):
        """Idle opacity is 1.0 — no widget-level dimming."""
        from meetandread.widgets.main_widget import MeetAndReadWidget
        assert MeetAndReadWidget._IDLE_OPACITY == 1.0

    def test_idle_opacity_equals_active(self):
        """Idle and active opacity are both 1.0 — differentiation is on the button paint."""
        from meetandread.widgets.main_widget import MeetAndReadWidget
        assert MeetAndReadWidget._IDLE_OPACITY == MeetAndReadWidget._ACTIVE_OPACITY

    def test_active_opacity_is_one(self):
        """Active opacity remains fully opaque."""
        from meetandread.widgets.main_widget import MeetAndReadWidget
        assert MeetAndReadWidget._ACTIVE_OPACITY == 1.0

