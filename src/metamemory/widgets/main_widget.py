"""
Main metamemory widget implementation using QGraphicsView.

This widget provides a borderless, always-on-top interface with:
- Record button as main body
- Toggle lobes for audio input selection
- Transcript panel that expands from the widget
- Drag and snap-to-edge functionality
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsWidget, QApplication
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, QTimer, QTime, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QLinearGradient

from metamemory.recording import RecordingController, ControllerState, ControllerError


class DragSurfaceItem(QGraphicsRectItem):
    """Invisible hit-testable background surface for drag initiation.
    
    Covers the entire widget scene rect to:
    - Prevent click-through to underlying applications
    - Provide a surface to initiate dragging from empty areas
    - Stay behind all interactive controls (z-value -1000)
    """
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 200, 120)
        self.parent_widget = parent_widget
        
        # No visible border
        self.setPen(QPen(Qt.PenStyle.NoPen))
        
        # Near-invisible fill (alpha=1) - hit-testable but effectively transparent
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))
        
        # Stay behind all other items
        self.setZValue(-1000)
        
        # Accept left mouse button for hit-testing
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
    
    def paint(self, painter, option, widget):
        """Override paint to render the near-invisible background."""
        # Fill with near-transparent color to be hit-testable
        painter.fillRect(self.rect(), self.brush())


class MeetAndReadWidget(QGraphicsView):
    """
    Main application widget.
    
    Borderless, always-on-top widget with custom painted components
    living in a QGraphicsScene for smooth animations and complex visuals.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Window configuration
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # Graphics view setup
        self.setRenderHints(QPainter.RenderHint.Antialiasing | 
                           QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")
        
        # Create scene (use _scene to avoid conflict with scene() method)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # Widget state
        self.is_recording = False
        self.is_processing = False
        self.is_dragging = False
        self.drag_start_pos = QPoint()
        self.widget_start_pos = QPoint()
        self.press_time = QTime.currentTime()

        # Docking state
        self.is_docked = False
        self.dock_edge = None  # 'left', 'right', 'top', 'bottom'
        
        # Recording controller
        self._controller = RecordingController()
        self._controller.on_state_change = self._on_controller_state_change
        self._controller.on_error = self._on_controller_error
        self._error_indicator = None  # For showing errors
        
        # Create widget components
        self._create_components()
        self._layout_components()
        
        # Set initial size
        self.setFixedSize(200, 120)
        self._scene.setSceneRect(0, 0, 200, 120)
        
        # Position on screen
        self._position_initial()
        
        # Timer for animations
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animations)
        self.animation_timer.start(33)  # ~30fps
        
        self.pulse_phase = 0.0
    
    def _create_components(self):
        """Create all widget components."""
        # Background drag surface (must be first to be behind everything)
        self.drag_surface = DragSurfaceItem(self)
        self._scene.addItem(self.drag_surface)
        
        # Main record button
        self.record_button = RecordButtonItem(self)
        self._scene.addItem(self.record_button)
        
        # Audio input toggle lobes
        self.mic_lobe = ToggleLobeItem("microphone", self)
        self.system_lobe = ToggleLobeItem("system", self)
        self._scene.addItem(self.mic_lobe)
        self._scene.addItem(self.system_lobe)
        
        # Settings lobe
        self.settings_lobe = SettingsLobeItem(self)
        self._scene.addItem(self.settings_lobe)
        
        # Error indicator (hidden by default)
        self._error_indicator = ErrorIndicatorItem(self)
        self._error_indicator.hide()
        self._scene.addItem(self._error_indicator)
    
    def _layout_components(self):
        """Position all components."""
        # Center the record button
        self.record_button.setPos(60, 20)
        
        # Position lobes on top 1/3rd of record button
        self.mic_lobe.setPos(50, 10)
        self.system_lobe.setPos(110, 10)
        
        # Settings lobe on side
        self.settings_lobe.setPos(160, 50)
        
        # Error indicator at bottom
        self._error_indicator.setPos(10, 105)
    
    def _position_initial(self):
        """Position widget on screen initially."""
        screen = QApplication.primaryScreen().geometry()
        # Start in bottom-right corner
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 40
        self.move(x, y)
    
    def _update_animations(self):
        """Update animation states."""
        if self.is_recording and not self.is_processing:
            self.pulse_phase += 0.1
            if self.pulse_phase > 6.28:  # 2*PI
                self.pulse_phase = 0.0
            self.record_button.set_pulse_phase(self.pulse_phase)
        elif self.is_processing:
            self.pulse_phase += 0.2
            if self.pulse_phase > 6.28:
                self.pulse_phase = 0.0
            self.record_button.set_swirl_phase(self.pulse_phase)
    
    def mousePressEvent(self, event):
        """Record press position for click vs drag detection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.globalPosition().toPoint()
            self.widget_start_pos = self.pos()
            self.press_time = QTime.currentTime()
            
            # Determine if press started on the drag surface
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self._scene.itemAt(scene_pos, self.transform())
            self._press_on_drag_surface = (item is self.drag_surface)
            
            # DON'T accept - let events propagate to child items
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self.is_dragging:
            delta = event.globalPosition().toPoint() - self.drag_start_pos
            new_pos = self.widget_start_pos + delta
            self.move(new_pos)
            self.is_docked = False
            self.dock_edge = None
            self._update_docked_state()
            event.accept()
        elif event.buttons() == Qt.MouseButton.LeftButton and self._press_on_drag_surface:
            # Check if movement exceeds threshold to start dragging
            current_pos = event.globalPosition().toPoint()
            movement = (current_pos - self.drag_start_pos).manhattanLength()
            if movement >= 5:
                # Start dragging from drag surface
                self.is_dragging = True
                delta = current_pos - self.drag_start_pos
                new_pos = self.widget_start_pos + delta
                self.move(new_pos)
                self.is_docked = False
                self.dock_edge = None
                self._update_docked_state()
                event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle click (short duration, small movement) or end drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            release_pos = event.globalPosition().toPoint()
            release_time = QTime.currentTime()

            # Calculate movement distance and time elapsed
            movement = (release_pos - self.drag_start_pos).manhattanLength()
            elapsed_ms = self.press_time.msecsTo(release_time)

            # Threshold: less than 5 pixels = click, not drag
            if movement < 5:
                # This is a click
                if self._press_on_drag_surface:
                    # Click on drag surface - consume to prevent click-through
                    event.accept()
                else:
                    # Click on interactive item - let child items handle it
                    super().mouseReleaseEvent(event)
            else:
                # This is a drag - finalize drag operation
                self.is_dragging = False
                self._check_snap_to_edge()
                event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def _check_snap_to_edge(self):
        """Check if widget should snap to screen edge."""
        screen = QApplication.primaryScreen().geometry()
        pos = self.pos()
        snap_threshold = 20
        
        # Check each edge
        if pos.x() < snap_threshold:
            self.dock_edge = 'left'
            self.is_docked = True
        elif pos.x() + self.width() > screen.width() - snap_threshold:
            self.dock_edge = 'right'
            self.is_docked = True
        elif pos.y() < snap_threshold:
            self.dock_edge = 'top'
            self.is_docked = True
        elif pos.y() + self.height() > screen.height() - snap_threshold:
            self.dock_edge = 'bottom'
            self.is_docked = True
        else:
            self.is_docked = False
            self.dock_edge = None
        
        self._update_docked_state()
    
    def _update_docked_state(self):
        """Update widget appearance based on docked state."""
        if self.is_docked:
            # When docked, show 4/5ths of the button
            if self.dock_edge == 'right':
                self.move(QApplication.primaryScreen().geometry().width() - 
                         int(self.width() * 0.8), self.y())
            elif self.dock_edge == 'left':
                self.move(int(self.width() * -0.2), self.y())
    
    def _get_selected_sources(self):
        """Get set of selected audio sources based on lobe states."""
        sources = set()
        if self.mic_lobe.is_active:
            sources.add('mic')
        if self.system_lobe.is_active:
            sources.add('system')
        return sources
    
    def _show_error(self, message):
        """Show error indicator with message."""
        self._error_indicator.set_text(message)
        self._error_indicator.show()
        # Auto-hide after 3 seconds
        QTimer.singleShot(3000, self._hide_error)
    
    def _hide_error(self):
        """Hide error indicator."""
        self._error_indicator.hide()
    
    def _on_controller_state_change(self, state):
        """Handle controller state changes."""
        if state == ControllerState.RECORDING:
            self.is_recording = True
            self.is_processing = False
            self.record_button.set_recording_state(True)
            self._hide_error()
        elif state == ControllerState.STOPPING:
            self.is_recording = False
            self.is_processing = True
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(True)
        elif state == ControllerState.IDLE:
            self.is_recording = False
            self.is_processing = False
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(False)
        elif state == ControllerState.ERROR:
            self.is_recording = False
            self.is_processing = False
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(False)
    
    def _on_controller_error(self, error):
        """Handle controller errors."""
        self._show_error(error.message)
        print(f"Recording error: {error.message}")
    
    def _on_recording_complete(self, wav_path):
        """Handle recording completion."""
        self.is_processing = False
        self.record_button.set_processing_state(False)
        print(f"Recording saved to: {wav_path}")
    
    def toggle_recording(self):
        """Toggle recording state via controller."""
        if not self._controller.is_recording():
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording via controller."""
        # Check if any source is selected
        sources = self._get_selected_sources()
        
        if not sources:
            self._show_error("Select mic or system audio first")
            return
        
        # Clear any previous error
        self._controller.clear_error()
        
        # Start recording via controller
        error = self._controller.start(sources)
        if error:
            self._show_error(error.message)
    
    def stop_recording(self):
        """Stop recording via controller (non-blocking)."""
        error = self._controller.stop(on_complete=self._on_recording_complete)
        if error:
            self._show_error(error.message)


class RecordButtonItem(QGraphicsEllipseItem):
    """Main record button component."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 80, 80)
        self.parent_widget = parent_widget
        self.is_recording = False
        self.is_processing = False
        self.pulse_phase = 0.0
        self.swirl_phase = 0.0
        
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def set_recording_state(self, recording):
        """Update recording state."""
        self.is_recording = recording
        self.update()
    
    def set_processing_state(self, processing):
        """Update processing state."""
        self.is_processing = processing
        self.update()
    
    def set_pulse_phase(self, phase):
        """Set pulse animation phase."""
        self.pulse_phase = phase
        self.update()
    
    def set_swirl_phase(self, phase):
        """Set swirl animation phase."""
        self.swirl_phase = phase
        self.update()
    
    def paint(self, painter, option, widget):
        """Custom paint for glass effect and animations."""
        rect = self.rect()
        
        if self.is_processing:
            # Swirling animation
            self._paint_swirl(painter, rect)
        elif self.is_recording:
            # Glowing red pulse
            self._paint_recording(painter, rect)
        else:
            # Translucent glass
            self._paint_idle(painter, rect)
        
        # Draw record icon
        self._paint_icon(painter, rect)
    
    def _paint_idle(self, painter, rect):
        """Paint idle state - translucent glass."""
        # Glass gradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor(255, 255, 255, 40))
        gradient.setColorAt(0.5, QColor(255, 255, 255, 20))
        gradient.setColorAt(1, QColor(255, 255, 255, 40))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 60), 2))
        painter.drawEllipse(rect)
    
    def _paint_recording(self, painter, rect):
        """Paint recording state - glowing red pulse."""
        # Calculate pulse intensity
        pulse = (1 + 0.3 * abs(self.pulse_phase)) / 1.3
        
        # Outer glow
        for i in range(3, 0, -1):
            alpha = int(80 * pulse / i)
            radius = i * 4
            painter.setBrush(QBrush(QColor(255, 50, 50, alpha)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.adjusted(-radius, -radius, radius, radius))
        
        # Main button - solid red
        painter.setBrush(QBrush(QColor(255, 50, 50, 200)))
        painter.setPen(QPen(QColor(255, 100, 100, 255), 2))
        painter.drawEllipse(rect)
    
    def _paint_swirl(self, painter, rect):
        """Paint processing state - swirling animation."""
        # Background
        painter.setBrush(QBrush(QColor(100, 100, 255, 180)))
        painter.setPen(QPen(QColor(150, 150, 255, 255), 2))
        painter.drawEllipse(rect)
        
        # Swirl effect
        import math
        center = rect.center()
        radius = rect.width() / 2 - 5
        
        for i in range(8):
            angle = self.swirl_phase + (i * math.pi / 4)
            x = center.x() + radius * 0.7 * math.cos(angle)
            y = center.y() + radius * 0.7 * math.sin(angle)
            
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
    
    def _paint_icon(self, painter, rect):
        """Paint record/stop icon."""
        center = rect.center()
        
        if self.is_recording:
            # Stop icon (square)
            size = 20
            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(center.x() - size/2), int(center.y() - size/2), 
                           size, size)
        else:
            # Record icon (circle)
            size = 24
            painter.setBrush(QBrush(QColor(255, 50, 50, 255)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(center.x() - size/2), int(center.y() - size/2), 
                              size, size)
    
    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_widget.toggle_recording()
            event.accept()


class ToggleLobeItem(QGraphicsEllipseItem):
    """Audio input toggle lobe (microphone or system audio)."""
    
    def __init__(self, lobe_type, parent_widget):
        super().__init__(0, 0, 40, 40)
        self.lobe_type = lobe_type
        self.parent_widget = parent_widget
        self.is_active = False
        
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def paint(self, painter, option, widget):
        """Paint the lobe."""
        rect = self.rect()
        
        if self.is_active:
            # Active state - bright
            painter.setBrush(QBrush(QColor(100, 200, 100, 200)))
            painter.setPen(QPen(QColor(150, 255, 150, 255), 2))
        else:
            # Inactive state - dim
            painter.setBrush(QBrush(QColor(100, 100, 100, 150)))
            painter.setPen(QPen(QColor(150, 150, 150, 200), 2))
        
        painter.drawEllipse(rect)
        
        # Draw icon
        painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        center = rect.center()
        
        if self.lobe_type == "microphone":
            # Simple mic icon
            painter.drawLine(int(center.x()), int(center.y() - 8), 
                           int(center.x()), int(center.y() + 4))
            painter.drawArc(int(center.x() - 6), int(center.y() - 4), 12, 12, 
                          0, 180 * 16)
        else:
            # Simple speaker icon
            painter.drawPolygon([
                QPointF(center.x() - 4, center.y() - 6),
                QPointF(center.x() + 2, center.y() - 6),
                QPointF(center.x() + 6, center.y() - 10),
                QPointF(center.x() + 6, center.y() + 10),
                QPointF(center.x() + 2, center.y() + 6),
                QPointF(center.x() - 4, center.y() + 6)
            ])
    
    def mousePressEvent(self, event):
        """Toggle state."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_active = not self.is_active
            self.update()
            print(f"{self.lobe_type} toggled: {self.is_active}")
            event.accept()


class SettingsLobeItem(QGraphicsEllipseItem):
    """Settings lobe for accessing configuration."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 30, 30)
        self.parent_widget = parent_widget
        
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def paint(self, painter, option, widget):
        """Paint settings lobe."""
        rect = self.rect()
        
        painter.setBrush(QBrush(QColor(100, 100, 200, 180)))
        painter.setPen(QPen(QColor(150, 150, 255, 255), 2))
        painter.drawEllipse(rect)
        
        # Draw gear icon
        painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        center = rect.center()
        painter.drawEllipse(int(center.x() - 6), int(center.y() - 6), 12, 12)
        painter.drawPoint(int(center.x()), int(center.y()))
    
    def mousePressEvent(self, event):
        """Open settings."""
        if event.button() == Qt.MouseButton.LeftButton:
            print("Settings clicked - dialog to be implemented")
            event.accept()


class ErrorIndicatorItem(QGraphicsRectItem):
    """Error indicator displayed below the record button."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 180, 14)
        self.parent_widget = parent_widget
        self._text = ""
        self._visible = False
        
    def set_text(self, text):
        """Set the error message text."""
        self._text = text
        self.update()
    
    def show(self):
        """Show the error indicator."""
        self._visible = True
        self.update()
    
    def hide(self):
        """Hide the error indicator."""
        self._visible = False
        self.update()
    
    def paint(self, painter, option, widget=None):
        """Paint error indicator."""
        if not self._visible:
            return
        
        rect = self.rect()
        
        # Background - red translucent
        painter.setBrush(QBrush(QColor(255, 50, 50, 180)))
        painter.setPen(QPen(QColor(255, 100, 100, 255), 1))
        painter.drawRoundedRect(rect, 3, 3)
        
        # Text
        if self._text:
            painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
            font = QFont("Arial", 8)
            font.setBold(True)
            painter.setFont(font)
            
            # Center text
            text_rect = rect.adjusted(2, 1, -2, -1)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._text)