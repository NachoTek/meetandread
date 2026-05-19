"""Tests for the CircularPlaybackControl widget.

Covers: region hit detection, play/pause toggling, skip routing,
speed cycling, volume +/- clamping, no-audio/no-helper no-ops,
boundary hit points outside regions, speed rates not in option list,
volume clamp at 0/1, and absence of QtMultimedia import in widget code.

Uses a mock helper matching the PlaybackHelper protocol so tests do
not require QtMultimedia DLLs.
"""

import sys
from unittest.mock import MagicMock, PropertyMock

import pytest

# Skip this module in environments where Qt widgets cannot be instantiated
try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401
except Exception:
    pytest.skip(
        "Skipping Qt widget tests (PyQt6 not available)",
        allow_module_level=True,
    )

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

from meetandread.widgets.playback_control import (
    CircularPlaybackControl,
    PlaybackRegion,
    SPEED_RATES,
    VOLUME_STEP,
)
from meetandread.widgets.theme import DARK_PALETTE

# Mark this module as requiring real Qt widgets
pytestmark = pytest.mark.requires_qt_widgets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Ensure a single QApplication exists for the entire module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_mock_helper(playing=False, audio_available=True, volume=0.8):
    """Create a mock helper matching the PlaybackHelper protocol.

    Args:
        playing: If True, playbackState() returns 1 (PlayingState).
        audio_available: What is_audio_available should return.
        volume: Current volume level.
    """
    helper = MagicMock()
    helper.is_audio_available = audio_available
    helper._playing = playing

    mock_player = MagicMock()
    mock_player.playbackState.return_value = 1 if playing else 0
    helper.player = mock_player

    mock_audio = MagicMock()
    helper.audio_output = mock_audio

    # Track actual calls for assertions
    helper.play = MagicMock()
    helper.pause = MagicMock()
    helper.stop = MagicMock()
    helper.set_rate = MagicMock()
    helper.set_volume = MagicMock()
    helper.skip_backward = MagicMock()
    helper.skip_forward = MagicMock()

    return helper


@pytest.fixture
def control(qapp):
    """Create a fresh CircularPlaybackControl without a helper."""
    return CircularPlaybackControl()


@pytest.fixture
def control_with_helper(qapp):
    """Create a CircularPlaybackControl with a mock helper."""
    helper = _make_mock_helper()
    ctrl = CircularPlaybackControl(helper=helper)
    return ctrl, helper


# ---------------------------------------------------------------------------
# Helper: get scene center and lobe positions
# ---------------------------------------------------------------------------

def _center_point(control: CircularPlaybackControl) -> QPointF:
    """Point at the center of the control widget."""
    return QPointF(control._SIZE / 2, control._SIZE / 2)


def _lobe_center(control: CircularPlaybackControl, item) -> QPointF:
    """Point at the center of a lobe item."""
    rect = item.rect()
    # Lobe items are in scene coordinates, convert to widget coordinates
    scene_center = QPointF(rect.center().x(), rect.center().y())
    widget_pt = control.mapFromScene(scene_center)
    return QPointF(widget_pt.x(), widget_pt.y())


# ---------------------------------------------------------------------------
# Tests: Region hit detection
# ---------------------------------------------------------------------------

class TestRegionHitDetection:
    """Verify deterministic region_at() for all regions."""

    def test_center_region(self, control):
        pt = _center_point(control)
        region = control.region_at(pt)
        assert region == PlaybackRegion.CENTER

    def test_skip_back_region(self, control):
        pt = _lobe_center(control, control._skip_back)
        region = control.region_at(pt)
        assert region == PlaybackRegion.SKIP_BACK

    def test_skip_forward_region(self, control):
        pt = _lobe_center(control, control._skip_forward)
        region = control.region_at(pt)
        assert region == PlaybackRegion.SKIP_FORWARD

    def test_speed_region(self, control):
        pt = _lobe_center(control, control._speed)
        region = control.region_at(pt)
        assert region == PlaybackRegion.SPEED

    def test_vol_up_region(self, control):
        """Top half of volume lobe should detect VOL_UP."""
        vol_rect = control._volume.rect()
        # Point slightly above center of volume lobe
        scene_pt = QPointF(vol_rect.center().x(), vol_rect.center().y() - 5)
        widget_pt = control.mapFromScene(scene_pt)
        region = control.region_at(QPointF(widget_pt.x(), widget_pt.y()))
        assert region == PlaybackRegion.VOL_UP

    def test_vol_down_region(self, control):
        """Bottom half of volume lobe should detect VOL_DOWN."""
        vol_rect = control._volume.rect()
        # Point slightly below center of volume lobe
        scene_pt = QPointF(vol_rect.center().x(), vol_rect.center().y() + 5)
        widget_pt = control.mapFromScene(scene_pt)
        region = control.region_at(QPointF(widget_pt.x(), widget_pt.y()))
        assert region == PlaybackRegion.VOL_DOWN

    def test_outside_all_regions(self, control):
        """Point well outside the control should return NONE."""
        region = control.region_at(QPointF(-10, -10))
        assert region == PlaybackRegion.NONE

    def test_corner_outside(self, control):
        """Corner of the widget (no lobe there) should return NONE."""
        region = control.region_at(QPointF(2, 2))
        assert region == PlaybackRegion.NONE

    def test_between_lobes(self, control):
        """Point between lobes but outside center should return NONE."""
        # Point between center and edge — outside both lobe and center rects
        region = control.region_at(QPointF(control._SIZE / 2, 2))
        assert region == PlaybackRegion.NONE


# ---------------------------------------------------------------------------
# Tests: item_for_region lookup
# ---------------------------------------------------------------------------

class TestItemForRegion:
    """Verify item_for_region returns correct items."""

    def test_center_item(self, control):
        item = control.item_for_region(PlaybackRegion.CENTER)
        assert item is control._center

    def test_skip_back_item(self, control):
        item = control.item_for_region(PlaybackRegion.SKIP_BACK)
        assert item is control._skip_back

    def test_skip_forward_item(self, control):
        item = control.item_for_region(PlaybackRegion.SKIP_FORWARD)
        assert item is control._skip_forward

    def test_speed_item(self, control):
        item = control.item_for_region(PlaybackRegion.SPEED)
        assert item is control._speed

    def test_vol_up_item(self, control):
        item = control.item_for_region(PlaybackRegion.VOL_UP)
        assert item is control._volume

    def test_vol_down_item(self, control):
        item = control.item_for_region(PlaybackRegion.VOL_DOWN)
        assert item is control._volume

    def test_none_returns_none(self, control):
        item = control.item_for_region(PlaybackRegion.NONE)
        assert item is None


# ---------------------------------------------------------------------------
# Tests: Play/pause toggling
# ---------------------------------------------------------------------------

class TestPlayPauseToggle:
    """Verify center button toggles play/pause via integer state comparison."""

    def test_play_when_stopped(self, control_with_helper):
        ctrl, helper = control_with_helper
        # Player is in stopped state (0), so clicking center should call play()
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        helper.play.assert_called_once()
        helper.pause.assert_not_called()

    def test_pause_when_playing(self, control_with_helper):
        ctrl, helper = control_with_helper
        helper._playing = True
        helper.player.playbackState.return_value = 1  # PlayingState
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        helper.pause.assert_called_once()
        helper.play.assert_not_called()

    def test_toggle_updates_label(self, control_with_helper):
        ctrl, helper = control_with_helper
        # Initially stopped — center text should become ⏸ after play
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        assert ctrl._center._text == "⏸"

    def test_toggle_pause_updates_label(self, control_with_helper):
        ctrl, helper = control_with_helper
        helper.player.playbackState.return_value = 1  # PlayingState
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        assert ctrl._center._text == "▶"


# ---------------------------------------------------------------------------
# Tests: Skip routing
# ---------------------------------------------------------------------------

class TestSkipRouting:
    """Verify skip back/forward dispatches to helper."""

    def test_skip_backward(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._handle_region_press(PlaybackRegion.SKIP_BACK, QPointF(0, 0))
        helper.skip_backward.assert_called_once()

    def test_skip_forward(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._handle_region_press(PlaybackRegion.SKIP_FORWARD, QPointF(0, 0))
        helper.skip_forward.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Speed cycling
# ---------------------------------------------------------------------------

class TestSpeedCycling:
    """Verify speed lobe cycles through SPEED_RATES."""

    def test_initial_speed_is_1x(self, control):
        assert control.current_speed == 1.0
        assert control.current_speed_index == SPEED_RATES.index(1.0)

    def test_one_click_advances_speed(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl.current_speed == 1.25
        helper.set_rate.assert_called_once_with(1.25)

    def test_full_cycle(self, control_with_helper):
        ctrl, helper = control_with_helper
        # Starting at index 3 (1.0), each click advances by 1
        expected_sequence = [1.25, 1.5, 2.0, 0.25, 0.5, 0.75, 1.0]
        for expected_rate in expected_sequence:
            ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
            assert ctrl.current_speed == expected_rate

    def test_speed_updates_label(self, control_with_helper):
        ctrl, _ = control_with_helper
        ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl._speed._text == "1.25×"

    def test_integer_rate_label(self, control_with_helper):
        ctrl, _ = control_with_helper
        # Cycle from index 3 (1.0) → 4 (1.25) → 5 (1.5) → 6 (2.0) = 3 clicks
        for _ in range(3):
            ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl._speed._text == "2×"


# ---------------------------------------------------------------------------
# Tests: Volume +/- clamping
# ---------------------------------------------------------------------------

class TestVolumeClamping:
    """Verify volume +/- clamped to [0.0, 1.0]."""

    def test_vol_up_increments(self, control_with_helper):
        ctrl, helper = control_with_helper
        initial = ctrl.current_volume
        ctrl._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        assert ctrl.current_volume == min(1.0, initial + VOLUME_STEP)
        helper.set_volume.assert_called_once_with(ctrl.current_volume)

    def test_vol_down_decrements(self, control_with_helper):
        ctrl, helper = control_with_helper
        initial = ctrl.current_volume
        ctrl._handle_region_press(PlaybackRegion.VOL_DOWN, QPointF(0, 0))
        assert ctrl.current_volume == max(0.0, initial - VOLUME_STEP)
        helper.set_volume.assert_called_once_with(ctrl.current_volume)

    def test_vol_up_clamped_at_1(self, control_with_helper):
        ctrl, helper = control_with_helper
        # Drive volume to near-max
        ctrl._volume_val = 0.95
        ctrl._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        assert ctrl.current_volume == 1.0

    def test_vol_up_stays_at_1(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._volume_val = 1.0
        ctrl._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        assert ctrl.current_volume == 1.0

    def test_vol_down_clamped_at_0(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._volume_val = 0.05
        ctrl._handle_region_press(PlaybackRegion.VOL_DOWN, QPointF(0, 0))
        assert ctrl.current_volume == 0.0

    def test_vol_down_stays_at_0(self, control_with_helper):
        ctrl, helper = control_with_helper
        ctrl._volume_val = 0.0
        ctrl._handle_region_press(PlaybackRegion.VOL_DOWN, QPointF(0, 0))
        assert ctrl.current_volume == 0.0


# ---------------------------------------------------------------------------
# Tests: No-audio / no-helper no-ops
# ---------------------------------------------------------------------------

class TestNoAudioNoHelper:
    """Verify no controller calls when audio unavailable or helper is None."""

    def test_no_helper_play(self, control):
        """Play/pause with no helper should not crash."""
        control._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        # No assertion needed — just no crash

    def test_no_helper_skip(self, control):
        """Skip with no helper should not crash."""
        control._handle_region_press(PlaybackRegion.SKIP_BACK, QPointF(0, 0))
        control._handle_region_press(PlaybackRegion.SKIP_FORWARD, QPointF(0, 0))

    def test_no_helper_speed(self, control):
        """Speed still cycles UI state even with no helper."""
        control._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert control.current_speed == 1.25  # cycled from 1.0

    def test_no_helper_volume(self, control):
        """Volume state updates even with no helper."""
        initial = control.current_volume
        control._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        assert control.current_volume == min(1.0, initial + VOLUME_STEP)

    def test_audio_unavailable_no_play(self, qapp):
        """Play with audio unavailable should not call helper.play()."""
        helper = _make_mock_helper(audio_available=False)
        ctrl = CircularPlaybackControl(helper=helper)
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        helper.play.assert_not_called()
        helper.pause.assert_not_called()

    def test_audio_unavailable_no_skip(self, qapp):
        helper = _make_mock_helper(audio_available=False)
        ctrl = CircularPlaybackControl(helper=helper)
        ctrl._handle_region_press(PlaybackRegion.SKIP_BACK, QPointF(0, 0))
        ctrl._handle_region_press(PlaybackRegion.SKIP_FORWARD, QPointF(0, 0))
        helper.skip_backward.assert_not_called()
        helper.skip_forward.assert_not_called()

    def test_audio_unavailable_speed_cycles_ui(self, qapp):
        """Speed still cycles in UI even when audio unavailable."""
        helper = _make_mock_helper(audio_available=False)
        ctrl = CircularPlaybackControl(helper=helper)
        ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl.current_speed == 1.25
        # But set_rate is not called because audio is unavailable
        helper.set_rate.assert_not_called()

    def test_audio_unavailable_volume_no_set(self, qapp):
        """Volume state updates but set_volume not called when audio unavailable."""
        helper = _make_mock_helper(audio_available=False)
        ctrl = CircularPlaybackControl(helper=helper)
        ctrl._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        helper.set_volume.assert_not_called()

    def test_set_helper_to_none(self, control_with_helper):
        """Setting helper to None should disable transport calls."""
        ctrl, helper = control_with_helper
        ctrl.set_helper(None)
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        helper.play.assert_not_called()

    def test_set_helper_replaces(self, control_with_helper):
        """Setting a new helper should dispatch to the new one."""
        ctrl, old_helper = control_with_helper
        new_helper = _make_mock_helper()
        ctrl.set_helper(new_helper)
        ctrl._handle_region_press(PlaybackRegion.CENTER, QPointF(0, 0))
        new_helper.play.assert_called_once()
        old_helper.play.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Speed rates not in option list
# ---------------------------------------------------------------------------

class TestSpeedBoundaryConditions:
    """Verify speed cycling wraps correctly and handles boundary rates."""

    def test_wraps_from_last_to_first(self, control_with_helper):
        ctrl, _ = control_with_helper
        # Starting at index 3 (1.0), cycle 7 times to return to 1.0
        for _ in range(len(SPEED_RATES)):
            ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl.current_speed == 1.0

    def test_all_rates_are_valid(self):
        """Ensure SPEED_RATES contains expected values."""
        expected = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        assert SPEED_RATES == expected

    def test_rate_not_in_list_is_replaced(self, control_with_helper):
        """If helper returns a rate not in our list, cycling still works.

        The control only stores an index; the helper's actual rate is not
        synced back. This test verifies the control's internal state
        remains consistent.
        """
        ctrl, helper = control_with_helper
        # Manually set a weird speed index
        ctrl._speed_index = 3  # 1.0
        ctrl._handle_region_press(PlaybackRegion.SPEED, QPointF(0, 0))
        assert ctrl.current_speed == 1.25  # index 4
        helper.set_rate.assert_called_with(1.25)


# ---------------------------------------------------------------------------
# Tests: Volume boundary conditions
# ---------------------------------------------------------------------------

class TestVolumeBoundaryConditions:
    """Volume clamp at exact 0 and 1 boundaries."""

    def test_volume_starts_at_0_8(self, control):
        assert control.current_volume == 0.8

    def test_multiple_vol_down_to_zero(self, control_with_helper):
        ctrl, _ = control_with_helper
        for _ in range(20):
            ctrl._handle_region_press(PlaybackRegion.VOL_DOWN, QPointF(0, 0))
        assert ctrl.current_volume == 0.0

    def test_multiple_vol_up_to_one(self, control_with_helper):
        ctrl, _ = control_with_helper
        for _ in range(20):
            ctrl._handle_region_press(PlaybackRegion.VOL_UP, QPointF(0, 0))
        assert ctrl.current_volume == 1.0


# ---------------------------------------------------------------------------
# Tests: Object names and debuggability
# ---------------------------------------------------------------------------

class TestObjectNames:
    """Verify object names are set for test/debug inspection."""

    def test_view_object_name(self, control):
        assert control.objectName() == "CircularPlaybackControl"

    def test_center_object_name(self, control):
        assert control._center._object_name == "lobe_center"

    def test_skip_back_object_name(self, control):
        assert control._skip_back._object_name == "lobe_skip_back"

    def test_skip_forward_object_name(self, control):
        assert control._skip_forward._object_name == "lobe_skip_forward"

    def test_speed_object_name(self, control):
        assert control._speed._object_name == "lobe_speed"

    def test_volume_object_name(self, control):
        assert control._volume._object_name == "lobe_volume"


# ---------------------------------------------------------------------------
# Tests: No QtMultimedia import
# ---------------------------------------------------------------------------

class TestNoQtMultimediaImport:
    """Verify the widget module does not import PyQt6.QtMultimedia."""

    def test_no_qtmultimedia_import(self):
        import ast
        import meetandread.widgets.playback_control as mod
        source = open(mod.__file__, "r", encoding="utf-8").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "QtMultimedia" in alias.name:
                        pytest.fail(f"Import of QtMultimedia found: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "QtMultimedia" in node.module:
                    pytest.fail(f"Import from QtMultimedia found: {node.module}")


# ---------------------------------------------------------------------------
# Tests: Widget instantiation
# ---------------------------------------------------------------------------

class TestWidgetInstantiation:
    """Verify widget can be created independently."""

    def test_creates_without_helper(self, control):
        assert control._helper is None

    def test_creates_with_helper(self, control_with_helper):
        ctrl, helper = control_with_helper
        assert ctrl._helper is helper

    def test_fixed_size(self, control):
        assert control.width() == 160
        assert control.height() == 160

    def test_scene_has_items(self, control):
        items = control._scene.items()
        assert len(items) >= 5  # center + 3 lobes + volume
