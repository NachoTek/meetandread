"""Programmatic icon generation for meetandread.

Creates application icons using QPainter/QPixmap so no external image
files are needed. Two variants:
  - Default app icon: green circle with "M" letter
  - Recording overlay: red pulsing dot to indicate active recording
"""

import logging

from PyQt6.QtGui import (
    QIcon,
    QPixmap,
    QPainter,
    QColor,
    QBrush,
    QPen,
    QFont,
    QRadialGradient,
)
from PyQt6.QtCore import Qt, QRectF

logger = logging.getLogger(__name__)

# Icon size in pixels (square)
_ICON_SIZE = 64
_PLAYBACK_ICON_SIZE = 20


def create_play_icon(color: QColor | None = None) -> QIcon:
    """Create a play triangle icon (right-pointing).

    Args:
        color: Icon color. Defaults to white.
    """
    c = color or QColor(255, 255, 255)
    sz = _PLAYBACK_ICON_SIZE
    pixmap = QPixmap(sz, sz)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(c))

    # Triangle with slight inset for visual balance
    m = 2
    tri = QRectF(m, m, sz - 2 * m, sz - 2 * m)
    from PyQt6.QtGui import QPolygonF
    from PyQt6.QtCore import QPointF
    painter.drawPolygon(QPolygonF([
        QPointF(tri.left() + 1, tri.top()),
        QPointF(tri.right(), tri.center().y()),
        QPointF(tri.left() + 1, tri.bottom()),
    ]))

    painter.end()
    return QIcon(pixmap)


def create_pause_icon(color: QColor | None = None) -> QIcon:
    """Create a pause icon (two vertical bars).

    Args:
        color: Icon color. Defaults to white.
    """
    c = color or QColor(255, 255, 255)
    sz = _PLAYBACK_ICON_SIZE
    pixmap = QPixmap(sz, sz)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(c))

    m = 2
    bar_w = 4
    bar_h = sz - 2 * m
    # Left bar
    painter.drawRoundedRect(QRectF(m, m, bar_w, bar_h), 1, 1)
    # Right bar
    painter.drawRoundedRect(QRectF(sz - m - bar_w, m, bar_w, bar_h), 1, 1)

    painter.end()
    return QIcon(pixmap)


def create_speaker_icon(volume: int = 80, muted: bool = False,
                        color: QColor | None = None) -> QIcon:
    """Create a speaker icon showing approximate volume level.

    Args:
        volume: 0-100 volume level.
        muted: If True, draw X instead of waves.
        color: Icon color. Defaults to white.
    """
    c = color or QColor(255, 255, 255)
    sz = _PLAYBACK_ICON_SIZE
    pixmap = QPixmap(sz, sz)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(c, 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(QBrush(c))

    m = 2
    r = QRectF(m, m, sz - 2 * m, sz - 2 * m)

    # Speaker body (trapezoid shape)
    body_w = r.width() * 0.3
    body_left = r.left() + r.width() * 0.05
    body_top = r.top() + r.height() * 0.3
    body_bot = r.bottom() - r.height() * 0.3
    body_mid_y = r.center().y()
    body_right = body_left + body_w

    from PyQt6.QtGui import QPolygonF
    from PyQt6.QtCore import QPointF
    painter.drawPolygon(QPolygonF([
        QPointF(body_left, body_top),
        QPointF(body_right, body_top),
        QPointF(body_right + body_w * 0.4, body_top - r.height() * 0.15),
        QPointF(body_right + body_w * 0.4, body_bot + r.height() * 0.15),
        QPointF(body_right, body_bot),
        QPointF(body_left, body_bot),
    ]))

    # Sound waves or X
    wave_cx = body_right + body_w * 0.4 + r.width() * 0.08
    wave_cy = body_mid_y

    if muted or volume == 0:
        # Draw X
        xr = r.width() * 0.12
        painter.setPen(QPen(c, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(wave_cx - xr, wave_cy - xr),
                         QPointF(wave_cx + xr, wave_cy + xr))
        painter.drawLine(QPointF(wave_cx + xr, wave_cy - xr),
                         QPointF(wave_cx - xr, wave_cy + xr))
    else:
        # Draw 1-3 arcs based on volume
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen_arc = QPen(c, 1.5)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)

        n_waves = 1 if volume < 33 else (2 if volume < 66 else 3)
        for i in range(n_waves):
            arc_r = r.width() * (0.15 + i * 0.12)
            arc_rect = QRectF(wave_cx - arc_r, wave_cy - arc_r,
                              arc_r * 2, arc_r * 2)
            painter.drawArc(arc_rect, -45 * 16, 90 * 16)

    painter.end()
    return QIcon(pixmap)


def create_app_icon() -> QIcon:
    """Create the default green application icon.

    Returns a QIcon with a dark circle background, green gradient fill,
    and a white "M" letter in the center.
    """
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(QColor(0, 0, 0, 0))  # transparent background

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = 4
    rect = QRectF(margin, margin, _ICON_SIZE - 2 * margin, _ICON_SIZE - 2 * margin)

    # Dark outer ring
    painter.setPen(QPen(QColor(30, 30, 30), 2))
    painter.setBrush(QBrush(QColor(30, 30, 30)))
    painter.drawEllipse(rect)

    # Green gradient fill
    inner = rect.adjusted(3, 3, -3, -3)
    gradient = QRadialGradient(inner.center(), inner.width() / 2)
    gradient.setColorAt(0.0, QColor(80, 220, 120))
    gradient.setColorAt(1.0, QColor(40, 167, 69))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(gradient))
    painter.drawEllipse(inner)

    # White "M" letter
    font = QFont("Segoe UI", 28, QFont.Weight.Bold)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "M")

    painter.end()
    return QIcon(pixmap)


def create_recording_icon() -> QIcon:
    """Create a recording-state icon with a red dot overlay.

    Uses the default app icon as base and draws a pulsing red dot in
    the bottom-right corner.
    """
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = 4
    rect = QRectF(margin, margin, _ICON_SIZE - 2 * margin, _ICON_SIZE - 2 * margin)

    # Dark outer ring
    painter.setPen(QPen(QColor(30, 30, 30), 2))
    painter.setBrush(QBrush(QColor(30, 30, 30)))
    painter.drawEllipse(rect)

    # Muted green fill (darker than default to let red dot stand out)
    inner = rect.adjusted(3, 3, -3, -3)
    gradient = QRadialGradient(inner.center(), inner.width() / 2)
    gradient.setColorAt(0.0, QColor(60, 160, 90))
    gradient.setColorAt(1.0, QColor(30, 120, 50))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(gradient))
    painter.drawEllipse(inner)

    # White "M" letter
    font = QFont("Segoe UI", 28, QFont.Weight.Bold)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "M")

    # Red recording dot (bottom-right corner)
    dot_radius = 10
    dot_center_x = _ICON_SIZE - margin - dot_radius - 2
    dot_center_y = _ICON_SIZE - margin - dot_radius - 2
    dot_rect = QRectF(
        dot_center_x - dot_radius,
        dot_center_y - dot_radius,
        dot_radius * 2,
        dot_radius * 2,
    )

    # Red glow
    glow_gradient = QRadialGradient(dot_rect.center(), dot_radius * 1.4)
    glow_gradient.setColorAt(0.0, QColor(255, 50, 50, 180))
    glow_gradient.setColorAt(0.7, QColor(220, 30, 30, 100))
    glow_gradient.setColorAt(1.0, QColor(200, 20, 20, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(glow_gradient))
    painter.drawEllipse(dot_rect.adjusted(-4, -4, 4, 4))

    # Solid red dot
    red_gradient = QRadialGradient(dot_rect.center(), dot_radius)
    red_gradient.setColorAt(0.0, QColor(255, 80, 80))
    red_gradient.setColorAt(1.0, QColor(220, 30, 30))
    painter.setBrush(QBrush(red_gradient))
    painter.drawEllipse(dot_rect)

    painter.end()
    return QIcon(pixmap)
