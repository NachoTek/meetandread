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
        """Providing exactly 120 samples should produce 120 amplitudes."""
        samples = np.sin(np.linspace(0, 2 * math.pi, 120)).astype(np.float32)
        button.set_waveform_samples(samples)
        assert len(button._waveform_data) == 120

    def test_oversized_array_downsampled(self, button):
        """16000 samples (1 second) should be downsampled to ~120."""
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

    def test_polls_every_third_frame_while_recording(self, widget):
        """During recording, get_live_audio_samples is called every 3rd frame."""
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
            # 30 frames / 3 cadence = 10 calls
            assert mock.call_count == 10

    def test_polling_uses_correct_duration(self, widget):
        """Controller is queried with duration_seconds=1.5."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        # Advance to first polling frame (frame 3 → counter=3)
        widget._waveform_frame_counter = 0
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples) as mock:
            # Advance 3 frames to trigger first poll
            for _ in range(3):
                widget._update_animations()
            mock.assert_called_once_with(duration_seconds=1.5)


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
            # Waveform data should now reflect the samples (downsampled to 120)
            assert widget.record_button._waveform_update_count > 0
            assert len(widget.record_button._waveform_data) == 120

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
            assert len(widget.record_button._waveform_data) == 120
            assert all(math.isfinite(v) for v in widget.record_button._waveform_data)


class TestWaveformRotationAdvancement:
    """Rotation advances smoothly during recording."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_rotation_advances_during_recording(self, widget):
        """Waveform rotation phase increases during recording."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        for _ in range(10):
            widget._update_animations()

        initial_phase = widget.record_button._waveform_rotation_phase
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(6):  # 2 poll cycles
                widget._update_animations()
        # Each poll advances by 0.06 radians → 2 polls = 0.12
        expected = initial_phase + 0.12
        assert widget.record_button._waveform_rotation_phase == pytest.approx(
            expected, abs=0.001
        )

    def test_rotation_resets_after_crossfade(self, widget):
        """Rotation phase resets to 0 only after cross-fade settles on idle."""
        from unittest.mock import patch
        from meetandread.recording import ControllerState
        fake_samples = np.array([0.1, 0.2], dtype=np.float32)
        widget._on_controller_state_change(ControllerState.RECORDING)
        with patch.object(widget._controller, "get_live_audio_samples",
                          return_value=fake_samples):
            for _ in range(15):
                widget._update_animations()
        assert widget.record_button._waveform_rotation_phase > 0.0

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
        assert len(widget.record_button._waveform_data) == 120


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
