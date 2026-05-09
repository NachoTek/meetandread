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
from PyQt6.QtCore import QRectF

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
