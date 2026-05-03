"""Tests for panel resize constraints, QSizeGrip positioning, and content reflow.

Covers:
- Min/max constraints on FloatingTranscriptPanel and FloatingSettingsPanel
- No setFixedSize on the panels themselves (panels are resizable)
- QSizeGrip child exists on both panels
- Grip repositions to bottom-right corner after programmatic resize
- Content reflow: QTextEdit in Live tab expands with FloatingTranscriptPanel
- Minimum size enforced: resize below minimum stays at minimumSize
- Settings panel QTabWidget adjusts on resize
- Legend overlay repositions within text_edit bounds after resize
"""

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSizeGrip, QTabWidget, QTextEdit, QWidget, QStackedWidget

from meetandread.widgets.floating_panels import (
    FloatingTranscriptPanel,
    FloatingSettingsPanel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    """Provide a QApplication singleton for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def transcript_panel(qapp):
    """Create a shown FloatingTranscriptPanel for testing, cleaned up after.

    panel.show() is required so that child widget visibility and positions
    behave as at runtime (Qt isVisible() returns False for children of
    hidden parents — MEM082).
    """
    p = FloatingTranscriptPanel()
    p.show()
    qapp.processEvents()
    yield p
    p.close()


@pytest.fixture
def settings_panel(qapp):
    """Create a shown FloatingSettingsPanel for testing, cleaned up after."""
    p = FloatingSettingsPanel()
    p.show()
    qapp.processEvents()
    yield p
    p.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resize_and_settle(panel: QWidget, w: int, h: int, qapp: QApplication) -> None:
    """Resize a panel and let the layout settle before assertions."""
    panel.resize(w, h)
    qapp.processEvents()


# ---------------------------------------------------------------------------
# 1. Min/max constraints
# ---------------------------------------------------------------------------

class TestTranscriptPanelConstraints:
    """FloatingTranscriptPanel min/max size constraints."""

    def test_minimum_width(self, transcript_panel):
        assert transcript_panel.minimumWidth() == 350

    def test_minimum_height(self, transcript_panel):
        assert transcript_panel.minimumHeight() == 300

    def test_maximum_width(self, transcript_panel):
        assert transcript_panel.maximumWidth() == 800

    def test_maximum_height(self, transcript_panel):
        assert transcript_panel.maximumHeight() == 900

    def test_not_fixed_size(self, transcript_panel):
        """Panel should NOT have identical min/max (that would mean setFixedSize)."""
        sz = transcript_panel.minimumSize()
        mx = transcript_panel.maximumSize()
        # At least one dimension must differ between min and max
        assert (sz.width() != mx.width()) or (sz.height() != mx.height())


class TestSettingsPanelConstraints:
    """FloatingSettingsPanel min/max size constraints."""

    def test_minimum_width(self, settings_panel):
        assert settings_panel.minimumWidth() == 545

    def test_minimum_height(self, settings_panel):
        assert settings_panel.minimumHeight() == 425

    def test_maximum_width(self, settings_panel):
        assert settings_panel.maximumWidth() == 825

    def test_maximum_height(self, settings_panel):
        assert settings_panel.maximumHeight() == 825

    def test_not_fixed_size(self, settings_panel):
        sz = settings_panel.minimumSize()
        mx = settings_panel.maximumSize()
        assert (sz.width() != mx.width()) or (sz.height() != mx.height())


# ---------------------------------------------------------------------------
# 2. QSizeGrip exists
# ---------------------------------------------------------------------------

class TestGripExists:
    """Both panels must have a QSizeGrip child widget."""

    def test_transcript_panel_has_grip(self, transcript_panel):
        grip = transcript_panel.findChild(QSizeGrip)
        assert grip is not None, "FloatingTranscriptPanel should have a QSizeGrip child"

    def test_settings_panel_has_grip(self, settings_panel):
        grip = settings_panel.findChild(QSizeGrip)
        assert grip is not None, "FloatingSettingsPanel should have a QSizeGrip child"

    def test_transcript_grip_visible(self, transcript_panel):
        grip = transcript_panel.findChild(QSizeGrip)
        assert grip is not None
        # MEM082: panel must be shown for isVisible() to be meaningful
        assert grip.isVisible()

    def test_settings_grip_visible(self, settings_panel):
        grip = settings_panel.findChild(QSizeGrip)
        assert grip is not None
        assert grip.isVisible()


# ---------------------------------------------------------------------------
# 3. Grip repositions on resize
# ---------------------------------------------------------------------------

class TestGripReposition:
    """Grip should move to the bottom-right corner after resize."""

    def test_transcript_grip_position_after_resize(self, transcript_panel, qapp):
        _resize_and_settle(transcript_panel, 500, 500, qapp)
        grip = transcript_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = transcript_panel.width() - grip.width()
        expected_y = transcript_panel.height() - grip.height()
        assert grip.x() == expected_x, f"Grip x={grip.x()}, expected {expected_x}"
        assert grip.y() == expected_y, f"Grip y={grip.y()}, expected {expected_y}"

    def test_transcript_grip_position_after_larger_resize(self, transcript_panel, qapp):
        _resize_and_settle(transcript_panel, 700, 700, qapp)
        grip = transcript_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = transcript_panel.width() - grip.width()
        expected_y = transcript_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y

    def test_settings_grip_position_after_resize(self, settings_panel, qapp):
        _resize_and_settle(settings_panel, 400, 600, qapp)
        grip = settings_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = settings_panel.width() - grip.width()
        expected_y = settings_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y

    def test_settings_grip_position_after_min_size(self, settings_panel, qapp):
        min_w = settings_panel.minimumWidth()
        min_h = settings_panel.minimumHeight()
        _resize_and_settle(settings_panel, min_w, min_h, qapp)
        grip = settings_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = settings_panel.width() - grip.width()
        expected_y = settings_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y


# ---------------------------------------------------------------------------
# 4. Content reflow — QTextEdit expands with panel
# ---------------------------------------------------------------------------

class TestContentReflow:
    """Text edit and tab widget should expand when panel is resized."""

    def test_text_edit_grows_with_panel(self, transcript_panel, qapp):
        """QTextEdit width/height should increase when panel is enlarged."""
        _resize_and_settle(transcript_panel, 400, 400, qapp)
        initial_w = transcript_panel.text_edit.width()
        initial_h = transcript_panel.text_edit.height()

        _resize_and_settle(transcript_panel, 600, 600, qapp)
        qapp.processEvents()
        new_w = transcript_panel.text_edit.width()
        new_h = transcript_panel.text_edit.height()

        assert new_w > initial_w, (
            f"text_edit width did not grow: {initial_w} -> {new_w}"
        )
        assert new_h > initial_h, (
            f"text_edit height did not grow: {initial_h} -> {new_h}"
        )

    def test_text_edit_width_proportional(self, transcript_panel, qapp):
        """text_edit width should be roughly panel width minus margins."""
        _resize_and_settle(transcript_panel, 500, 500, qapp)
        qapp.processEvents()
        te_w = transcript_panel.text_edit.width()
        panel_w = transcript_panel.width()
        # Layout has 10px margins on each side = 20px total horizontal margin
        # Allow some slack for borders and spacing
        assert te_w <= panel_w, "text_edit wider than panel"
        assert te_w >= panel_w - 40, (
            f"text_edit too narrow: {te_w} vs panel {panel_w} (expected >= {panel_w - 40})"
        )


# ---------------------------------------------------------------------------
# 5. Minimum size enforced
# ---------------------------------------------------------------------------

class TestMinimumSizeEnforced:
    """Resizing below minimum should clamp to minimumSize."""

    def test_transcript_panel_min_enforced(self, transcript_panel, qapp):
        """Resizing below minimum stays at minimum."""
        min_w = transcript_panel.minimumWidth()
        min_h = transcript_panel.minimumHeight()
        _resize_and_settle(transcript_panel, min_w - 50, min_h - 50, qapp)
        assert transcript_panel.width() >= min_w
        assert transcript_panel.height() >= min_h

    def test_settings_panel_min_enforced(self, settings_panel, qapp):
        min_w = settings_panel.minimumWidth()
        min_h = settings_panel.minimumHeight()
        _resize_and_settle(settings_panel, min_w - 50, min_h - 50, qapp)
        assert settings_panel.width() >= min_w
        assert settings_panel.height() >= min_h

    def test_transcript_grip_still_at_corner_at_min_size(self, transcript_panel, qapp):
        """Even at minimum size, grip should be at the corner."""
        min_w = transcript_panel.minimumWidth()
        min_h = transcript_panel.minimumHeight()
        _resize_and_settle(transcript_panel, min_w, min_h, qapp)
        grip = transcript_panel.findChild(QSizeGrip)
        assert grip is not None
        expected_x = transcript_panel.width() - grip.width()
        expected_y = transcript_panel.height() - grip.height()
        assert grip.x() == expected_x
        assert grip.y() == expected_y


# ---------------------------------------------------------------------------
# 6. Settings panel content adjusts
# ---------------------------------------------------------------------------

class TestSettingsPanelContent:
    """Settings panel content stack should adjust on resize."""

    def test_content_stack_grows_with_panel(self, settings_panel, qapp):
        """QStackedWidget width should increase when panel is enlarged."""
        _resize_and_settle(settings_panel, 440, 450, qapp)
        stack = settings_panel.findChild(QStackedWidget)
        assert stack is not None, "Settings panel should have a QStackedWidget"
        initial_w = stack.width()

        _resize_and_settle(settings_panel, 600, 650, qapp)
        qapp.processEvents()
        new_w = stack.width()

        assert new_w > initial_w, (
            f"QStackedWidget width did not grow: {initial_w} -> {new_w}"
        )

    def test_content_stack_fits_within_panel(self, settings_panel, qapp):
        """QStackedWidget should never exceed panel dimensions."""
        _resize_and_settle(settings_panel, 500, 600, qapp)
        qapp.processEvents()
        stack = settings_panel.findChild(QStackedWidget)
        assert stack is not None
        assert stack.width() <= settings_panel.width()
        assert stack.height() <= settings_panel.height()


# ---------------------------------------------------------------------------
# 7. Legend overlay repositions after resize
# ---------------------------------------------------------------------------

class TestLegendOverlayReposition:
    """Legend overlay should stay within text_edit bounds after resize."""

    def test_legend_within_text_edit_after_resize(self, transcript_panel, qapp):
        _resize_and_settle(transcript_panel, 500, 500, qapp)
        # Show the legend
        transcript_panel._legend_btn.setChecked(True)
        transcript_panel._toggle_legend()
        qapp.processEvents()

        overlay = transcript_panel._legend_overlay
        te = transcript_panel.text_edit
        assert overlay is not None
        assert overlay.isVisible()

        # Overlay should be within text_edit bounds
        # overlay.geometry() is relative to text_edit (its parent)
        assert overlay.x() >= 0, f"Overlay x={overlay.x()} is negative"
        assert overlay.y() >= 0, f"Overlay y={overlay.y()} is negative"
        assert overlay.x() + overlay.width() <= te.width(), (
            f"Overlay right edge {overlay.x() + overlay.width()} exceeds text_edit width {te.width()}"
        )
        assert overlay.y() + overlay.height() <= te.height(), (
            f"Overlay bottom edge {overlay.y() + overlay.height()} exceeds text_edit height {te.height()}"
        )

    def test_legend_repositions_on_larger_resize(self, transcript_panel, qapp):
        """Legend should reposition correctly after a larger resize."""
        _resize_and_settle(transcript_panel, 450, 400, qapp)
        transcript_panel._legend_btn.setChecked(True)
        transcript_panel._toggle_legend()
        qapp.processEvents()

        # Now resize larger
        _resize_and_settle(transcript_panel, 600, 600, qapp)
        qapp.processEvents()

        overlay = transcript_panel._legend_overlay
        te = transcript_panel.text_edit
        assert overlay is not None
        # Check within bounds again at new size
        assert overlay.x() >= 0
        assert overlay.y() >= 0
        assert overlay.x() + overlay.width() <= te.width()
        assert overlay.y() + overlay.height() <= te.height()
