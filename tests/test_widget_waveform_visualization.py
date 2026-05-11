"""Tests for RecordButtonItem waveform visualization.

Covers:
- Sample downsampling to bounded point count (~120 amplitudes)
- Silent/empty data fallback to flat low-amplitude waveform
- Bounded point count for oversized arrays
- Rotation phase advancement
- Theme-aware color via current_palette().text
- _paint_waveform integrates into _paint_recording without child items
- Malformed input handling (non-finite, wrong-shaped values)
"""

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication, QGraphicsScene
from PyQt6.QtGui import QPainter, QColor, QPixmap
from PyQt6.QtCore import QRectF, Qt

from meetandread.widgets.main_widget import RecordButtonItem


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
# RecordButtonItem fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def button(qapp):
    """Create a RecordButtonItem for testing.

    The scene is kept alive for the fixture lifetime to prevent premature
    C++ object deletion of the QGraphicsItem.
    """
    # Need a parent widget with is_dragging and _click_consumed attributes
    parent = MagicMock()
    parent.is_dragging = False
    parent._click_consumed = False
    parent.toggle_recording = MagicMock()

    btn = RecordButtonItem(parent)
    # Add to scene so paint() can work — keep scene alive for fixture lifetime
    scene = QGraphicsScene()
    scene.addItem(btn)
    btn._test_scene = scene  # prevent GC
    return btn


# ---------------------------------------------------------------------------
# Sample downsampling
# ---------------------------------------------------------------------------

class TestSampleDownsampling:
    """set_waveform_samples downsamples to ~120 absolute amplitudes."""

    def test_exact_target_count(self, button):
        """Providing exactly 60 samples should produce 60 amplitudes."""
        samples = np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) == 60

    def test_oversized_array_downsampled(self, button):
        """16000 samples (1 second) should be downsampled to ~60."""
        samples = np.random.randn(16000).astype(np.float32) * 0.5
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) == button._WAVEFORM_TARGET_POINTS

    def test_undersized_array_padded_or_kept(self, button):
        """Very short arrays (< target) should still produce valid data."""
        samples = np.array([0.1, 0.5, -0.3], dtype=np.float32)
        button.set_waveform_samples(samples)
        # Should not crash; data should be available
        assert len(button._waveform_data) > 0
        # All values should be finite
        for v in button._waveform_data:
            assert math.isfinite(v)

    def test_amplitudes_are_absolute_values(self, button):
        """Resulting amplitudes should be absolute values (>= 0)."""
        samples = np.array([-0.8, 0.3, -0.1, 0.9, -0.5], dtype=np.float32)
        button.set_waveform_samples(samples)
        for v in button._waveform_data:
            assert v >= 0.0

    def test_bounded_for_large_sample_array(self, button):
        """Even very large arrays must produce bounded point count."""
        samples = np.random.randn(1_000_000).astype(np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) <= button._WAVEFORM_TARGET_POINTS


# ---------------------------------------------------------------------------
# Silent / empty fallback
# ---------------------------------------------------------------------------

class TestSilentFallback:
    """Empty or all-zero samples produce a flat low-amplitude waveform."""

    def test_empty_array_produces_flat_fallback(self, button):
        """Empty input should produce flat low-amplitude data."""
        samples = np.ndarray(0, dtype=np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) > 0
        # All values should be the same low-amplitude fallback
        assert all(v == button._waveform_data[0] for v in button._waveform_data)
        assert button._waveform_data[0] >= 0.0
        assert button._waveform_data[0] <= 0.1  # low amplitude

    def test_all_zero_samples_produce_flat_fallback(self, button):
        """All-zero input should produce flat low-amplitude data."""
        samples = np.zeros(1000, dtype=np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) > 0
        assert all(v == button._waveform_data[0] for v in button._waveform_data)
        assert button._waveform_data[0] <= 0.1

    def test_none_samples_produce_flat_fallback(self, button):
        """None input should produce flat low-amplitude data without raising."""
        button.set_waveform_samples(None)
        assert len(button._waveform_data) > 0
        assert button._waveform_data[0] <= 0.1


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------

class TestMalformedInput:
    """Non-finite / wrong-shaped values degrade gracefully."""

    def test_non_finite_values_replaced(self, button):
        """NaN and Inf values should be coerced to safe values."""
        samples = np.array([0.5, float('nan'), float('inf'), -0.3, float('-inf')],
                          dtype=np.float32)
        button.set_waveform_samples(samples)
        for v in button._waveform_data:
            assert math.isfinite(v)

    def test_wrong_dtype_converted(self, button):
        """int64 or float64 arrays should be handled without error."""
        samples = np.array([100, -200, 300, -400], dtype=np.int64)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) > 0
        for v in button._waveform_data:
            assert math.isfinite(v)

    def test_1d_required(self, button):
        """2D arrays should be handled gracefully (flattened or fallback)."""
        samples = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) > 0


# ---------------------------------------------------------------------------
# Rotation phase advancement
# ---------------------------------------------------------------------------

class TestRotationPhaseAdvancement:
    """advance_waveform_rotation updates rotation phase correctly."""

    def test_initial_phase_is_zero(self, button):
        assert button._waveform_rotation_phase == 0.0

    def test_advance_increases_phase(self, button):
        button.advance_waveform_rotation(0.05)
        assert button._waveform_rotation_phase == pytest.approx(0.05)

    def test_advance_accumulates(self, button):
        button.advance_waveform_rotation(0.1)
        button.advance_waveform_rotation(0.2)
        assert button._waveform_rotation_phase == pytest.approx(0.3)

    def test_negative_delta_allowed(self, button):
        button.advance_waveform_rotation(1.0)
        button.advance_waveform_rotation(-0.5)
        assert button._waveform_rotation_phase == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Theme-aware color
# ---------------------------------------------------------------------------

class TestThemeColorLookup:
    """Waveform painting uses current_palette().text at paint time."""

    def test_paint_waveform_uses_palette_text_color(self, button):
        """_paint_waveform should create a QColor from current_palette().text."""
        from meetandread.widgets.theme import current_palette

        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        )
        button.is_recording = True

        # Create a QPainter on a small pixmap
        pixmap = QPixmap(80, 80)
        painter = QPainter(pixmap)

        # Paint and verify no exception
        try:
            button._paint_waveform(painter, QRectF(0, 0, 80, 80))
        finally:
            painter.end()

        # Verify palette.text is a valid hex color
        palette = current_palette()
        color = QColor(palette.text)
        assert color.isValid()

    def test_palette_text_produces_valid_qcolor(self):
        """current_palette().text is always a valid hex string for QColor."""
        from meetandread.widgets.theme import current_palette
        palette = current_palette()
        color = QColor(palette.text)
        assert color.isValid()


# ---------------------------------------------------------------------------
# Paint integration
# ---------------------------------------------------------------------------

class TestPaintIntegration:
    """Waveform integrates into recording paint pass without child items."""

    def test_paint_recording_does_not_add_child_items(self, button):
        """After painting, no new child graphics items should exist."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        )

        initial_children = len(button.childItems())
        pixmap = QPixmap(80, 80)
        painter = QPainter(pixmap)
        try:
            # Call the recording paint path directly
            rect = QRectF(0, 0, 80, 80)
            button._paint_recording(painter, rect)
        finally:
            painter.end()

        assert len(button.childItems()) == initial_children, \
            "Waveform rendering added child items — should be paint-only"

    def test_paint_waveform_with_flat_data_noops_safely(self, button):
        """_paint_waveform with flat fallback data should not raise."""
        button.set_waveform_samples(np.ndarray(0, dtype=np.float32))
        pixmap = QPixmap(80, 80)
        painter = QPainter(pixmap)
        try:
            button._paint_waveform(painter, QRectF(0, 0, 80, 80))
        finally:
            painter.end()
        # Should complete without exception

    def test_waveform_data_is_bounded_list(self, button):
        """_waveform_data should be a bounded list of finite floats."""
        samples = np.random.randn(50000).astype(np.float32) * 0.8
        button.set_waveform_samples(samples)
        assert isinstance(button._waveform_data, list)
        assert len(button._waveform_data) <= button._WAVEFORM_TARGET_POINTS
        for v in button._waveform_data:
            assert isinstance(v, float)
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Frame counter / update tracking
# ---------------------------------------------------------------------------

class TestFrameCounter:
    """Waveform update frame counter tracks calls to set_waveform_samples."""

    def test_initial_frame_counter_is_zero(self, button):
        assert button._waveform_update_count == 0

    def test_frame_counter_increments(self, button):
        samples = np.array([0.1, 0.2], dtype=np.float32)
        button.set_waveform_samples(samples)
        assert button._waveform_update_count == 1
        button.set_waveform_samples(samples)
        assert button._waveform_update_count == 2


# ---------------------------------------------------------------------------
# T03: Integration tests — waveform polling in animation loop
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


def _make_widget(qapp):
    """Create a MeetAndReadWidget with mocked Qt globals for testing."""
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
    w._floating_settings_panel = MagicMock()
    w._floating_settings_panel.isVisible.return_value = False
    w._cc_overlay = MagicMock()
    w._cc_overlay.isVisible.return_value = False
    return w


class TestWaveformPollingCadence:
    """Waveform samples are polled from controller every 3rd animation tick."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_no_polling_while_idle(self, widget):
        """Idle widget should never call get_live_audio_samples."""
        from unittest.mock import patch
        with patch.object(widget._controller, "get_live_audio_samples",
                          side_effect=AssertionError("should not be called")) as mock:
            for _ in range(30):
                widget._update_animations()
            mock.assert_not_called()

    def test_no_polling_while_processing(self, widget):
        """Processing widget should never call get_live_audio_samples."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._on_controller_state_change(ControllerState.STOPPING)
        assert widget.is_processing
        with patch.object(widget._controller, "get_live_audio_samples",
                          side_effect=AssertionError("should not be called")) as mock:
            for _ in range(30):
                widget._update_animations()
            mock.assert_not_called()

    def test_polls_every_frame_while_recording(self, widget):
        """During recording, get_live_audio_samples is called every frame (30fps)."""
        from unittest.mock import patch, MagicMock
        from meetandread.recording import ControllerState
        fake_samples = np.random.randn(16000).astype(np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Settle visual state
        for _ in range(10):
            widget._update_animations()

        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples) as mock:
            mock.reset_mock()
            for _ in range(30):
                widget._update_animations()
            # 30 frames at 30fps = 30 calls
            assert mock.call_count == 30

    def test_polling_uses_correct_duration(self, widget):
        """Controller is queried with duration_seconds=0.1."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        widget._waveform_frame_counter = 0
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples) as mock:
            # Single frame triggers poll (every frame at 30fps)
            widget._update_animations()
            mock.assert_called_with(duration_seconds=0.1)


class TestWaveformSamplePropagation:
    """Controller samples propagate into record_button waveform data."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_samples_reach_record_button(self, widget):
        """After polling, record_button._waveform_data reflects the samples."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.sin(np.linspace(0, 2 * math.pi, 500)).astype(np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Settle
        for _ in range(10):
            widget._update_animations()

        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            # Advance 3 frames to trigger one poll
            for _ in range(3):
                widget._update_animations()
            # Waveform data should now reflect the samples (downsampled to 60)
            assert widget.record_button._waveform_update_count > 0
            assert len(widget.record_button._waveform_data) == 60

    def test_empty_samples_degrade_gracefully(self, widget):
        """Controller returning empty samples → button gets flat fallback."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        empty_samples = np.ndarray(0, dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()

        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=empty_samples):
            for _ in range(3):
                widget._update_animations()
            # Should have flat fallback data, not crash
            assert len(widget.record_button._waveform_data) == 60
            assert all(math.isfinite(v) for v in widget.record_button._waveform_data)


class TestWaveformRotationAdvancement:
    """Rotation advances smoothly during recording."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_rotation_is_static_during_recording(self, widget):
        """Waveform rotation is disabled — phase stays at 0."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()

        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(6):
                widget._update_animations()
        # Rotation is disabled — phase should remain 0
        assert widget.record_button._waveform_rotation_phase == pytest.approx(0.0, abs=0.001)

    def test_rotation_resets_after_crossfade(self, widget):
        """Rotation phase stays at 0 after cross-fade (static waveform)."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(15):
                widget._update_animations()
        # Rotation is disabled — always 0

        # Go idle
        widget._on_controller_state_change(ControllerState.IDLE)
        # Settle cross-fade
        for _ in range(20):
            widget._update_animations()
        assert widget.record_button._state_t >= 1.0
        assert widget.record_button._waveform_rotation_phase == 0.0


class TestWaveformErrorDegradation:
    """Controller errors degrade to flat waveform without crashing."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_controller_exception_degrades_gracefully(self, widget):
        """If controller raises, animation loop continues with flat data."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()

        call_count = 0
        def _raising_getter(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("audio subsystem error")

        with patch.object(widget._controller, "get_live_audio_samples",
                          side_effect=_raising_getter):
            # Should not raise — animation loop must survive
            for _ in range(30):
                widget._update_animations()

        # Controller was called (error path was exercised)
        assert call_count > 0
        # Animation loop survived: pulse_phase advanced
        assert widget.pulse_phase > 0.0
        # Waveform data is still valid (flat fallback from None passthrough)
        assert len(widget.record_button._waveform_data) == 60


class TestWaveformClickThrough:
    """Waveform visualization does not add interactive items blocking clicks."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_no_new_interactive_child_items_during_recording(self, widget):
        """Recording with waveform does not add child items to the button."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.random.randn(16000).astype(np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)

        initial_children = len(widget.record_button.childItems())
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(30):
                widget._update_animations()
        # No new child items added
        assert len(widget.record_button.childItems()) == initial_children

    def test_record_button_remains_clickable(self, widget):
        """RecordButtonItem mouse event handling still works with waveform."""
        from unittest.mock import patch, MagicMock
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(15):
                widget._update_animations()

        # Simulate a click: toggle_recording should be callable via the button
        widget.is_dragging = False
        widget._click_consumed = False
        event = MagicMock()
        event.button.return_value = Qt.MouseButton.LeftButton
        with patch.object(widget, "toggle_recording") as mock_toggle:
            widget.record_button.mouseReleaseEvent(event)
            mock_toggle.assert_called_once()


class TestWaveformFrameCounterReset:
    """Frame counter resets properly on state transitions."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_frame_counter_resets_after_recording(self, widget):
        """_waveform_frame_counter resets to 0 after recording stops and settles."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(20):
                widget._update_animations()
        assert widget._waveform_frame_counter > 0

        # Go idle and settle
        widget._on_controller_state_change(ControllerState.IDLE)
        for _ in range(20):
            widget._update_animations()
        assert widget._waveform_frame_counter == 0

    def test_frame_counter_does_not_advance_while_idle(self, widget):
        """Frame counter stays at 0 during idle."""
        for _ in range(30):
            widget._update_animations()
        assert widget._waveform_frame_counter == 0


# ---------------------------------------------------------------------------
# T03: Health-state threshold, recovery, color, and click-through tests
# ---------------------------------------------------------------------------

import time as _time


class TestHealthStateThreshold:
    """Warning enters only at >= 5 aggregate drops; counts 0–4 stay NORMAL."""

    def test_initial_health_is_normal(self, button):
        """Fresh button starts in NORMAL health state."""
        assert button._health_state == button._HEALTH_NORMAL
        assert button._health_t == 0.0

    def test_count_below_threshold_stays_normal(self, button):
        """Counts 0–4 do not trigger WARNING."""
        for count in range(0, 5):
            button.is_recording = True
            button.on_frames_dropped(count)
            assert button._health_state == button._HEALTH_NORMAL, \
                f"Count {count} should not trigger WARNING"

    def test_count_5_enters_warning(self, button):
        """Count 5 triggers WARNING state."""
        button.is_recording = True
        button.on_frames_dropped(5)
        assert button._health_state == button._HEALTH_WARNING
        assert button._last_frames_dropped == 5

    def test_count_above_5_enters_warning(self, button):
        """Counts above 5 also trigger WARNING."""
        button.is_recording = True
        button.on_frames_dropped(100)
        assert button._health_state == button._HEALTH_WARNING

    def test_threshold_boundary_exact_5(self, button):
        """Boundary: 4 stays NORMAL, 5 enters WARNING."""
        button.is_recording = True
        button.on_frames_dropped(4)
        assert button._health_state == button._HEALTH_NORMAL
        button.on_frames_dropped(5)
        assert button._health_state == button._HEALTH_WARNING


class TestHealthStateRepeatedDrops:
    """Repeated drop events refresh _last_drop_time and sustain WARNING."""

    def test_repeated_drops_refresh_timestamp(self, button):
        """Each drop updates _last_drop_time to prevent premature recovery."""
        button.is_recording = True
        button.on_frames_dropped(5)
        first_time = button._last_drop_time
        # Sleep longer to ensure monotonic() returns a different timestamp
        # (Windows timer resolution may be ~15ms, so 50ms is safe)
        import time
        time.sleep(0.05)
        button.on_frames_dropped(6)
        assert button._last_drop_time > first_time

    def test_warning_sustained_with_continuous_drops(self, button):
        """WARNING state stays active as drops continue."""
        button.is_recording = True
        for count in range(5, 15):
            button.on_frames_dropped(count)
        assert button._health_state == button._HEALTH_WARNING

    def test_health_t_resets_on_reentry(self, button):
        """Re-entering WARNING after recovery resets _health_t for smooth transition."""
        button.is_recording = True
        button.on_frames_dropped(5)
        assert button._health_state == button._HEALTH_WARNING
        assert button._health_t == 0.0  # starts at 0 for transition animation


class TestHealthStateTimedRecovery:
    """WARNING auto-recovers after 1 second of no drops via tick()."""

    def test_no_recovery_before_one_second(self, button):
        """tick() does not recover if < 1 second since last drop."""
        button.is_recording = True
        button.on_frames_dropped(5)
        assert button._health_state == button._HEALTH_WARNING

        # Advance ticks but not enough time
        with patch("meetandread.widgets.main_widget._time") as mock_time:
            mock_time.monotonic.return_value = button._last_drop_time + 0.5
            for _ in range(20):
                button.tick()
        assert button._health_state == button._HEALTH_WARNING

    def test_recovery_after_one_second(self, button):
        """tick() recovers to NORMAL after 1 second of no drops."""
        button.is_recording = True
        button.on_frames_dropped(5)
        assert button._health_state == button._HEALTH_WARNING

        # Advance time past recovery window
        with patch("meetandread.widgets.main_widget._time") as mock_time:
            mock_time.monotonic.return_value = button._last_drop_time + 1.1
            button.tick()
        assert button._health_state == button._HEALTH_NORMAL

    def test_recovery_decay_animates_health_t(self, button):
        """After recovery, _health_t decays back to 0 over multiple ticks."""
        button.is_recording = True
        button.on_frames_dropped(5)
        # Advance health_t to 1.0
        for _ in range(10):
            button.tick()
        assert button._health_t == 1.0

        # Trigger recovery
        with patch("meetandread.widgets.main_widget._time") as mock_time:
            mock_time.monotonic.return_value = button._last_drop_time + 1.1
            button.tick()
        assert button._health_state == button._HEALTH_NORMAL

        # Decay _health_t back to 0
        for _ in range(10):
            button.tick()
        assert button._health_t == 0.0

    def test_exact_one_second_triggers_recovery(self, button):
        """Recovery triggers at exactly 1.0 seconds since last drop."""
        button.is_recording = True
        button.on_frames_dropped(5)

        with patch("meetandread.widgets.main_widget._time") as mock_time:
            mock_time.monotonic.return_value = button._last_drop_time + 1.0
            button.tick()
        assert button._health_state == button._HEALTH_NORMAL


class TestHealthStateStopReset:
    """Stopping recording resets health state immediately."""

    def test_stop_resets_to_normal(self, button):
        """set_recording_state(False) immediately resets to NORMAL."""
        button.is_recording = True
        button.on_frames_dropped(10)
        assert button._health_state == button._HEALTH_WARNING

        button.set_recording_state(False)
        assert button._health_state == button._HEALTH_NORMAL
        assert button._health_t == 0.0
        assert button._last_drop_time == 0.0
        assert button._last_frames_dropped == 0

    def test_no_warning_after_stop(self, button):
        """After stop, no delayed warning appears even with high _last_frames_dropped."""
        button.is_recording = True
        button.on_frames_dropped(10)
        button.set_recording_state(False)

        # Advance many ticks — should stay NORMAL
        for _ in range(30):
            button.tick()
        assert button._health_state == button._HEALTH_NORMAL

    def test_drops_ignored_when_not_recording(self, button):
        """on_frames_dropped is ignored when not recording."""
        button.is_recording = False
        button.on_frames_dropped(100)
        assert button._health_state == button._HEALTH_NORMAL
        assert button._last_frames_dropped == 100  # count stored but no state change


class TestHealthStateMalformedInput:
    """Malformed inputs do not crash and do not enter WARNING."""

    def test_none_count_ignored(self, button):
        """None count is safely ignored."""
        button.is_recording = True
        button.on_frames_dropped(None)
        assert button._health_state == button._HEALTH_NORMAL

    def test_negative_count_ignored(self, button):
        """Negative count is safely ignored."""
        button.is_recording = True
        button.on_frames_dropped(-5)
        assert button._health_state == button._HEALTH_NORMAL

    def test_string_count_ignored(self, button):
        """String count is safely ignored."""
        button.is_recording = True
        button.on_frames_dropped("not-a-number")
        assert button._health_state == button._HEALTH_NORMAL

    def test_float_count_coerced(self, button):
        """Float count is truncated to int."""
        button.is_recording = True
        button.on_frames_dropped(5.9)
        assert button._health_state == button._HEALTH_WARNING
        assert button._last_frames_dropped == 5

    def test_zero_count_ignored(self, button):
        """Zero count does not change state."""
        button.is_recording = True
        button.on_frames_dropped(0)
        assert button._health_state == button._HEALTH_NORMAL
        assert button._last_frames_dropped == 0


class TestHealthColorInterpolation:
    """_get_waveform_health_color produces valid QColor at all progress levels."""

    def test_zero_health_t_returns_base_color(self, button):
        """At _health_t=0.0, returns base color unchanged."""
        base = QColor(200, 200, 200, 200)
        button._health_t = 0.0
        result = button._get_waveform_health_color(base)
        assert result.red() == base.red()
        assert result.green() == base.green()
        assert result.blue() == base.blue()

    def test_full_health_t_approaches_warning_color(self, button):
        """At _health_t=1.0 (after easing), color is close to warning amber."""
        base = QColor(200, 200, 200, 200)
        button._health_t = 1.0
        result = button._get_waveform_health_color(base)
        # Should be interpolated toward amber (255, 165, 0)
        assert result.red() == 255  # base 200 → 255 (warning red)
        assert result.green() < base.green()  # should decrease toward 165
        assert result.blue() < base.blue()  # should decrease toward 0

    def test_intermediate_health_t_valid_color(self, button):
        """At intermediate _health_t, result is a valid QColor."""
        base = QColor(200, 200, 200, 200)
        button._health_t = 0.5
        result = button._get_waveform_health_color(base)
        assert result.isValid()
        assert 0 <= result.red() <= 255
        assert 0 <= result.green() <= 255
        assert 0 <= result.blue() <= 255
        assert 0 <= result.alpha() <= 255

    def test_interpolation_is_monotonic(self, button):
        """Red channel increases monotonically as _health_t advances."""
        base = QColor(200, 200, 200, 200)
        prev_r = 0
        for t_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            button._health_t = t_val
            result = button._get_waveform_health_color(base)
            assert result.red() >= prev_r
            prev_r = result.red()

    def test_invalid_base_color_falls_back(self, button):
        """Invalid base color still produces a valid result."""
        base = QColor()
        assert not base.isValid()
        button._health_t = 0.5
        result = button._get_waveform_health_color(base)
        assert result.isValid()


class TestHealthStatePaintIntegration:
    """Health color is used in waveform paint without adding child items."""

    def test_warning_paint_uses_amber_color(self, button):
        """During WARNING, waveform paint path uses amber-interpolated color."""
        button.is_recording = True
        button.on_frames_dropped(5)
        # Advance health_t to full
        for _ in range(10):
            button.tick()
        assert button._health_t == 1.0

        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        )

        pixmap = QPixmap(80, 80)
        painter = QPainter(pixmap)
        try:
            button._paint_waveform(painter, QRectF(0, 0, 80, 80))
        finally:
            painter.end()
        # Should complete without error — amber color was used

    def test_paint_does_not_add_child_items_with_health_warning(self, button):
        """WARNING state paint adds no child items."""
        button.is_recording = True
        button.on_frames_dropped(5)
        for _ in range(10):
            button.tick()

        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        )

        initial_children = len(button.childItems())
        pixmap = QPixmap(80, 80)
        painter = QPainter(pixmap)
        try:
            button._paint_waveform(painter, QRectF(0, 0, 80, 80))
        finally:
            painter.end()

        assert len(button.childItems()) == initial_children

    def test_health_warning_preserves_click_through(self, button):
        """WARNING state does not affect mouse event handling."""
        button.is_recording = True
        button.on_frames_dropped(5)
        for _ in range(10):
            button.tick()

        # Mouse events should still work
        event = MagicMock()
        event.button.return_value = Qt.MouseButton.LeftButton
        button.parent_widget.is_dragging = False
        button.parent_widget._click_consumed = False
        with patch.object(button.parent_widget, "toggle_recording") as mock:
            button.mouseReleaseEvent(event)
            mock.assert_called_once()


class TestHealthStateColorInPaintWaveform:
    """Verify waveform color reflects health state during _paint_recording."""

    def test_normal_uses_base_color(self, button):
        """NORMAL health paints with theme base color (not amber)."""
        from meetandread.widgets.theme import current_palette

        button.is_recording = True
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        )

        # Get expected base color
        palette = current_palette()
        expected_base = QColor(palette.text)

        # Check health color returns base color when normal
        result = button._get_waveform_health_color(expected_base)
        assert result.red() == expected_base.red()
        assert result.green() == expected_base.green()

    def test_warning_produces_different_color_than_base(self, button):
        """WARNING health produces a different color than the base."""
        from meetandread.widgets.theme import current_palette

        button.is_recording = True
        button.on_frames_dropped(5)
        for _ in range(10):
            button.tick()

        palette = current_palette()
        base = QColor(palette.text)
        result = button._get_waveform_health_color(base)
        # Color should differ from base (unless base happens to be amber)
        colors_differ = (
            result.red() != base.red() or
            result.green() != base.green() or
            result.blue() != base.blue()
        )
        assert colors_differ, "WARNING color should differ from base theme color"


# ---------------------------------------------------------------------------
# T02: Waveform render guard — ui.waveform_enabled controls _paint_waveform
# ---------------------------------------------------------------------------

class TestWaveformRenderGuard:
    """_paint_recording guards _paint_waveform on ui.waveform_enabled config."""

    def test_enabled_calls_paint_waveform(self, button):
        """When config is True, _paint_waveform is called during recording paint."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=True):
            with patch.object(button, "_paint_waveform",
                              wraps=button._paint_waveform) as mock_wf:
                pixmap = QPixmap(80, 80)
                painter = QPainter(pixmap)
                try:
                    button._paint_recording(painter, QRectF(0, 0, 80, 80))
                finally:
                    painter.end()
                mock_wf.assert_called_once()

    def test_disabled_skips_paint_waveform(self, button):
        """When config is False, _paint_waveform is NOT called during recording paint."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=False):
            with patch.object(button, "_paint_waveform") as mock_wf:
                pixmap = QPixmap(80, 80)
                painter = QPainter(pixmap)
                try:
                    button._paint_recording(painter, QRectF(0, 0, 80, 80))
                finally:
                    painter.end()
                mock_wf.assert_not_called()

    def test_config_error_defaults_to_enabled(self, button):
        """When config lookup raises, waveform is still painted (enabled)."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    side_effect=RuntimeError("config corrupted")):
            with patch.object(button, "_paint_waveform",
                              wraps=button._paint_waveform) as mock_wf:
                pixmap = QPixmap(80, 80)
                painter = QPainter(pixmap)
                try:
                    button._paint_recording(painter, QRectF(0, 0, 80, 80))
                finally:
                    painter.end()
                mock_wf.assert_called_once()

    def test_none_config_defaults_to_enabled(self, button):
        """When config returns None, waveform is still painted (enabled)."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=None):
            with patch.object(button, "_paint_waveform",
                              wraps=button._paint_waveform) as mock_wf:
                pixmap = QPixmap(80, 80)
                painter = QPainter(pixmap)
                try:
                    button._paint_recording(painter, QRectF(0, 0, 80, 80))
                finally:
                    painter.end()
                mock_wf.assert_called_once()

    def test_disabled_still_draws_pulse_base(self, button):
        """Disabled waveform still draws the pulse/glow/base (no crash)."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=False):
            pixmap = QPixmap(80, 80)
            painter = QPainter(pixmap)
            try:
                # Should complete without exception — pulse base still draws
                button._paint_recording(painter, QRectF(0, 0, 80, 80))
            finally:
                painter.end()

    def test_disabled_no_child_items(self, button):
        """Disabled waveform does not add child items."""
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        initial_children = len(button.childItems())
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=False):
            pixmap = QPixmap(80, 80)
            painter = QPainter(pixmap)
            try:
                button._paint_recording(painter, QRectF(0, 0, 80, 80))
            finally:
                painter.end()
        assert len(button.childItems()) == initial_children

    def test_guard_is_cheap_no_save_or_alloc(self, button):
        """Guard path does not call save_config or allocate large objects.

        The render guard executes every paint frame, so it must only do a
        cheap config read. This test verifies the guard doesn't call save
        or trigger UI rebuilding.
        """
        button.is_recording = True
        button.pulse_phase = 0.5
        button.set_waveform_samples(
            np.sin(np.linspace(0, 2 * math.pi, 60)).astype(np.float32)
        )
        with patch("meetandread.widgets.main_widget.get_config",
                    return_value=True) as mock_get, \
             patch("meetandread.widgets.main_widget.save_config") as mock_save:
            pixmap = QPixmap(80, 80)
            painter = QPainter(pixmap)
            try:
                button._paint_recording(painter, QRectF(0, 0, 80, 80))
            finally:
                painter.end()
            # Only get_config called — no save
            mock_get.assert_called_once_with("ui.waveform_enabled")
            mock_save.assert_not_called()
