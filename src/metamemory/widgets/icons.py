"""Programmatic icon generation for metamemory.

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
