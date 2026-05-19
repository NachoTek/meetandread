"""Circular playback control widget for the History panel.

A standalone ~160×160 widget with a center play/pause button and four
surrounding lobes (skip-back, skip-forward, speed, volume split +/-).
Uses child QGraphicsEllipseItem items for deterministic region hit detection
and an injected controller helper with a duck-typed interface matching
HistoryPlaybackController.

Design language mirrors RecordButtonItem / ToggleLobeItem from main_widget.py:
directional border cues, AETHERIC theme tokens, hover/pressed feedback,
and quadratic ease-out transitions.
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto
from typing import List, Optional, Protocol, runtime_checkable

from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QColor,
    QBrush,
    QPen,
    QFont,
    QPainter,
)

from meetandread.widgets.theme import (
    current_palette,
    AETHERIC_RED,
    AETHERIC_BORDER_LIGHT,
    AETHERIC_BORDER_DARK,
    AETHERIC_SETTINGS_BG,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed playback rates (must match History panel speed combo)
# ---------------------------------------------------------------------------
SPEED_RATES: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Volume step for +/- buttons
VOLUME_STEP: float = 0.1


# ---------------------------------------------------------------------------
# Controller helper protocol — duck-typed interface
# ---------------------------------------------------------------------------

@runtime_checkable
class PlaybackHelper(Protocol):
    """Minimal interface the circular control needs.

    Matches HistoryPlaybackController's public API via duck typing.
    """

    @property
    def is_audio_available(self) -> bool: ...

    @property
    def player(self) -> object: ...

    @property
    def audio_output(self) -> object: ...

    def play(self) -> None: ...

    def pause(self) -> None: ...

    def stop(self) -> None: ...

    def set_rate(self, rate: float) -> None: ...

    def set_volume(self, volume: float) -> None: ...

    def skip_backward(self, seconds: float = 5.0) -> None: ...

    def skip_forward(self, seconds: float = 5.0) -> None: ...


# ---------------------------------------------------------------------------
# Region identifiers
# ---------------------------------------------------------------------------

class PlaybackRegion(Enum):
    """Named clickable regions within the circular control."""
    NONE = auto()
    CENTER = auto()       # play/pause
    SKIP_BACK = auto()    # ← skip
    SKIP_FORWARD = auto() # → skip
    SPEED = auto()        # speed cycle
    VOL_UP = auto()       # volume + (top half of volume lobe)
    VOL_DOWN = auto()     # volume − (bottom half of volume lobe)


# ---------------------------------------------------------------------------
# Lobe child item — one button region on the circle
# ---------------------------------------------------------------------------

class _LobeItem(QGraphicsEllipseItem):
    """A circular lobe button with hover/pressed visual feedback.

    Positioned on the perimeter of the main control circle.  Uses
    deterministic boundingRect checks for hit testing instead of
    pixel-level screenshot comparison.
    """

    def __init__(
        self,
        region: PlaybackRegion,
        rect: QRectF,
        text: str = "",
        parent=None,
    ):
        super().__init__(rect, parent)
        self.region = region
        self._text = text
        self._hovered = False
        self._pressed = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, False)

        # Object name for test/debug inspection (stored as plain attribute
        # since QGraphicsEllipseItem does not inherit setObjectName)
        self._object_name = f"lobe_{region.name.lower()}"

    # -- Visual state -------------------------------------------------------

    def set_pressed(self, pressed: bool) -> None:
        self._pressed = pressed
        self.update()

    # -- QGraphicsItem overrides -------------------------------------------

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint lobe with AETHERIC glass design + hover/pressed feedback."""
        p = current_palette()
        rect = self.rect()

        # --- Background fill ---
        if self._pressed:
            fill = QColor(AETHERIC_RED)
            fill.setAlpha(60)
        elif self._hovered:
            fill = QColor(255, 255, 255, 20)
        else:
            fill = QColor(AETHERIC_SETTINGS_BG)
            # Parse hex to get RGB, then apply alpha
            fill = QColor(AETHERIC_SETTINGS_BG)
            fill.setAlpha(200)

        painter.setBrush(QBrush(fill))

        # --- Border (directional cues) ---
        border_light = QColor(AETHERIC_BORDER_LIGHT)
        border_dark = QColor(AETHERIC_BORDER_DARK)
        if self._hovered:
            pen = QPen(QColor(AETHERIC_RED), 1.5)
        else:
            pen = QPen(border_light, 1)
        painter.setPen(pen)
        painter.drawEllipse(rect)

        # --- Icon / text label ---
        painter.setPen(QPen(QColor(p.text_secondary)))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)


# ---------------------------------------------------------------------------
# SplitVolumeLobe — vertically split for + / −
# ---------------------------------------------------------------------------

class _SplitVolumeLobe(QGraphicsEllipseItem):
    """Volume lobe split vertically into top (+) and bottom (−) halves.

    Hit detection checks whether the click point is above or below the
    vertical centre of the ellipse bounding rect.
    """

    def __init__(self, rect: QRectF, parent=None):
        super().__init__(rect, parent)
        self._hovered = False
        self._pressed_up = False
        self._pressed_down = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._object_name = "lobe_volume"

    def region_at(self, point: QPointF) -> PlaybackRegion:
        """Return VOL_UP or VOL_DOWN depending on *point* position."""
        if not self.contains(point):
            return PlaybackRegion.NONE
        mid_y = self.rect().center().y()
        if point.y() < mid_y:
            return PlaybackRegion.VOL_UP
        return PlaybackRegion.VOL_DOWN

    def set_pressed_up(self, pressed: bool) -> None:
        self._pressed_up = pressed
        self.update()

    def set_pressed_down(self, pressed: bool) -> None:
        self._pressed_down = pressed
        self.update()

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None):
        """Paint split volume lobe with + on top and − on bottom."""
        p = current_palette()
        rect = self.rect()

        # Background
        fill = QColor(AETHERIC_SETTINGS_BG)
        fill.setAlpha(200)
        if self._hovered:
            fill = QColor(255, 255, 255, 20)
        painter.setBrush(QBrush(fill))

        # Border
        if self._hovered:
            pen = QPen(QColor(AETHERIC_RED), 1.5)
        else:
            pen = QPen(QColor(AETHERIC_BORDER_LIGHT), 1)
        painter.setPen(pen)
        painter.drawEllipse(rect)

        # Divider line
        mid_y = rect.center().y()
        painter.setPen(QPen(QColor(AETHERIC_BORDER_DARK), 0.5))
        painter.drawLine(
            QPointF(rect.left() + 4, mid_y),
            QPointF(rect.right() - 4, mid_y),
        )

        # + on top, − on bottom
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        color_up = QColor(AETHERIC_RED) if (self._pressed_up or self._hovered) else QColor(p.text_secondary)
        color_dn = QColor(AETHERIC_RED) if (self._pressed_down or self._hovered) else QColor(p.text_secondary)
        top_rect = QRectF(rect.left(), rect.top(), rect.width(), rect.height() / 2)
        bot_rect = QRectF(rect.left(), mid_y, rect.width(), rect.height() / 2)

        painter.setPen(QPen(color_up))
        painter.drawText(top_rect, Qt.AlignmentFlag.AlignCenter, "+")
        painter.setPen(QPen(color_dn))
        painter.drawText(bot_rect, Qt.AlignmentFlag.AlignCenter, "−")


# ---------------------------------------------------------------------------
# CircularPlaybackControl — main widget
# ---------------------------------------------------------------------------

class CircularPlaybackControl(QGraphicsView):
    """Modern circular playback control with center play/pause and lobes.

    Sized ~160×160 with transparent background.  Uses an injected
    controller helper (Protocol: PlaybackHelper) for all transport
    actions.  No QtMultimedia import — playback state checked via
    integer comparison (``state == 1`` means PlayingState).

    Layout (approximate radii from center):
        - Center:  80×80 ellipse — play/pause toggle
        - Skip-back lobe:    36×36 at ~135° (top-left)
        - Skip-forward lobe: 36×36 at ~45°  (top-right)
        - Speed lobe:        36×36 at ~225° (bottom-left)
        - Volume lobe:       36×36 at ~315° (bottom-right)

    Test surface:
        - ``region_at(point)`` — deterministic region lookup
        - ``item_for_region(region)`` — named item lookup
        - ``current_speed_index`` / ``current_volume`` — observable state

    Object names:
        - View: ``CircularPlaybackControl``
        - Lobes: ``lobe_center``, ``lobe_skip_back``, etc.
    """

    # Control dimensions
    _SIZE = 160
    _CENTER_SIZE = 80
    _LOBE_SIZE = 36
    _LOBE_ORBIT_RADIUS = 58  # distance from center to lobe center

    def __init__(self, parent=None, helper: Optional[PlaybackHelper] = None):
        super().__init__(parent)
        self.setObjectName("CircularPlaybackControl")
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet("background: transparent; border: none;")
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._helper: Optional[PlaybackHelper] = helper
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, self._SIZE, self._SIZE)
        self.setScene(self._scene)

        # Observable state
        self._speed_index: int = SPEED_RATES.index(1.0)  # start at 1.0×
        self._volume_val: float = 0.8  # initial volume

        # -- Build child items --
        self._center: _LobeItem
        self._skip_back: _LobeItem
        self._skip_forward: _LobeItem
        self._speed: _LobeItem
        self._volume: _SplitVolumeLobe

        self._build_items()

        logger.debug(
            "CircularPlaybackControl created (size=%d, helper=%s)",
            self._SIZE,
            type(helper).__name__ if helper else "None",
        )

    # -- Item construction --------------------------------------------------

    def _build_items(self) -> None:
        """Create and position all child items in the scene."""
        cx, cy = self._SIZE / 2, self._SIZE / 2
        half_c = self._CENTER_SIZE / 2
        half_l = self._LOBE_SIZE / 2

        # Center play/pause
        self._center = _LobeItem(
            PlaybackRegion.CENTER,
            QRectF(cx - half_c, cy - half_c, self._CENTER_SIZE, self._CENTER_SIZE),
            text="▶",
        )
        self._center._object_name = "lobe_center"
        self._scene.addItem(self._center)

        # Orbital lobes at 45°, 135°, 225°, 315°
        # Skip-forward: top-right (45°)
        # Skip-back:    top-left (135°)
        # Speed:        bottom-left (225°)
        # Volume:       bottom-right (315°)
        lobe_specs = [
            (PlaybackRegion.SKIP_BACK, 135, "⏪", "_skip_back"),
            (PlaybackRegion.SKIP_FORWARD, 45, "⏩", "_skip_forward"),
            (PlaybackRegion.SPEED, 225, "1×", "_speed"),
        ]

        for region, angle_deg, text, attr_name in lobe_specs:
            angle = math.radians(angle_deg)
            lx = cx + self._LOBE_ORBIT_RADIUS * math.cos(angle) - half_l
            ly = cy - self._LOBE_ORBIT_RADIUS * math.sin(angle) - half_l
            item = _LobeItem(
                region,
                QRectF(lx, ly, self._LOBE_SIZE, self._LOBE_SIZE),
                text=text,
            )
            setattr(self, attr_name, item)
            self._scene.addItem(item)

        # Volume lobe (split) at 315°
        angle = math.radians(315)
        lx = cx + self._LOBE_ORBIT_RADIUS * math.cos(angle) - half_l
        ly = cy - self._LOBE_ORBIT_RADIUS * math.sin(angle) - half_l
        self._volume = _SplitVolumeLobe(
            QRectF(lx, ly, self._LOBE_SIZE, self._LOBE_SIZE),
        )
        self._scene.addItem(self._volume)

        # Initialize speed label
        self._update_speed_label()

    # -- Public test helpers ------------------------------------------------

    def set_helper(self, helper: Optional[PlaybackHelper]) -> None:
        """Inject or replace the playback controller helper."""
        self._helper = helper

    def region_at(self, point: QPointF) -> PlaybackRegion:
        """Return the named region at *point* (in widget coordinates).

        Deterministic hit detection using child item bounding rects.
        Volume lobe uses vertical split logic.
        """
        # Map widget point to scene coordinates
        scene_pt = self.mapToScene(int(point.x()), int(point.y()))

        # Check lobes first (smaller, on top)
        if self._volume.contains(scene_pt):
            return self._volume.region_at(scene_pt)

        for item in (self._skip_back, self._skip_forward, self._speed):
            if item.contains(scene_pt):
                return item.region

        # Check center last (largest, behind lobes)
        if self._center.contains(scene_pt):
            return PlaybackRegion.CENTER

        return PlaybackRegion.NONE

    def item_for_region(self, region: PlaybackRegion):
        """Return the QGraphicsItem for a named region, or None."""
        mapping = {
            PlaybackRegion.CENTER: self._center,
            PlaybackRegion.SKIP_BACK: self._skip_back,
            PlaybackRegion.SKIP_FORWARD: self._skip_forward,
            PlaybackRegion.SPEED: self._speed,
            PlaybackRegion.VOL_UP: self._volume,
            PlaybackRegion.VOL_DOWN: self._volume,
        }
        return mapping.get(region)

    @property
    def current_speed_index(self) -> int:
        """Index into SPEED_RATES for the current playback speed."""
        return self._speed_index

    @property
    def current_speed(self) -> float:
        """Current playback rate value."""
        return SPEED_RATES[self._speed_index]

    @property
    def current_volume(self) -> float:
        """Current volume level (0.0 – 1.0)."""
        return self._volume_val

    # -- Event handling -----------------------------------------------------

    def mousePressEvent(self, event):
        """Handle clicks by mapping to regions and invoking actions."""
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        widget_pt = QPointF(event.position().x(), event.position().y())
        scene_pt = self.mapToScene(int(widget_pt.x()), int(widget_pt.y()))
        region = self.region_at(widget_pt)

        self._handle_region_press(region, scene_pt)
        event.accept()

    def mouseReleaseEvent(self, event):
        """Clear pressed state on release."""
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self._clear_all_pressed()
        event.accept()

    # -- Region action dispatch ---------------------------------------------

    def _handle_region_press(self, region: PlaybackRegion, scene_pt: QPointF) -> None:
        """Dispatch the action for a pressed region."""
        helper = self._helper

        if region == PlaybackRegion.CENTER:
            self._center.set_pressed(True)
            if helper is not None and helper.is_audio_available:
                player = helper.player
                state = player.playbackState()
                # PlayingState == 1 in QtMultimedia — integer comparison
                if state == 1:
                    helper.pause()
                    self._center._text = "▶"
                else:
                    helper.play()
                    self._center._text = "⏸"
                self._center.update()
            return

        if region == PlaybackRegion.SKIP_BACK:
            self._skip_back.set_pressed(True)
            if helper is not None and helper.is_audio_available:
                helper.skip_backward()
            return

        if region == PlaybackRegion.SKIP_FORWARD:
            self._skip_forward.set_pressed(True)
            if helper is not None and helper.is_audio_available:
                helper.skip_forward()
            return

        if region == PlaybackRegion.SPEED:
            self._speed.set_pressed(True)
            # Speed cycles regardless of audio availability (UI-only state)
            self._speed_index = (self._speed_index + 1) % len(SPEED_RATES)
            self._update_speed_label()
            if helper is not None and helper.is_audio_available:
                helper.set_rate(SPEED_RATES[self._speed_index])
            return

        if region == PlaybackRegion.VOL_UP:
            self._volume.set_pressed_up(True)
            self._volume_val = min(1.0, self._volume_val + VOLUME_STEP)
            if helper is not None and helper.is_audio_available:
                helper.set_volume(self._volume_val)
            return

        if region == PlaybackRegion.VOL_DOWN:
            self._volume.set_pressed_down(True)
            self._volume_val = max(0.0, self._volume_val - VOLUME_STEP)
            if helper is not None and helper.is_audio_available:
                helper.set_volume(self._volume_val)
            return

    def _clear_all_pressed(self) -> None:
        """Reset pressed state on all lobes."""
        for item in (self._center, self._skip_back, self._skip_forward, self._speed):
            item.set_pressed(False)
        self._volume.set_pressed_up(False)
        self._volume.set_pressed_down(False)

    def _update_speed_label(self) -> None:
        """Update the speed lobe text to reflect current rate."""
        rate = SPEED_RATES[self._speed_index]
        if rate == int(rate):
            self._speed._text = f"{int(rate)}×"
        else:
            self._speed._text = f"{rate}×"
        self._speed.update()
