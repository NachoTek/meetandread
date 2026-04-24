"""
Main metamemory widget implementation using QGraphicsView with floating panels.

This widget provides a borderless, always-on-top interface with:
- Record button as main body
- Toggle lobes for audio input selection
- FloatingTranscriptPanel that appears outside the widget (not clipped)
- Drag and snap-to-edge functionality
- Real-time transcription display with confidence colors

Key improvement: Uses floating QWidget panels instead of QGraphicsItem panels
to avoid clipping issues and enable proper text editing.
"""

from pathlib import Path
from typing import Optional
import logging
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsWidget, QApplication, QGraphicsItemGroup, QWidget, QMenu
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, QTimer, QTime, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QLinearGradient

from metamemory.recording import RecordingController, ControllerState, ControllerError
from metamemory.transcription.confidence import get_confidence_color, get_distortion_intensity
from metamemory.transcription.transcript_store import Word
from metamemory.transcription.accumulating_processor import SegmentResult
from metamemory.config import get_config, set_config, save_config, AppSettings
from metamemory.hardware.recommender import ModelRecommender, get_model_info
from metamemory.widgets.floating_panels import FloatingTranscriptPanel, FloatingSettingsPanel
import time as _time


class _SlideState:
    """Tracks an in-progress slide animation from one position to another."""

    __slots__ = ('active', 'start_pos', 'target_pos', 'start_time_ms', 'duration_ms')

    def __init__(self):
        self.active: bool = False
        self.start_pos: QPoint = QPoint()
        self.target_pos: QPoint = QPoint()
        self.start_time_ms: int = 0
        self.duration_ms: int = 300

    def __repr__(self):
        if not self.active:
            return "<_SlideState inactive>"
        return (f"<_SlideState {self.start_pos}→{self.target_pos} "
                f"dur={self.duration_ms}ms>")


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
    
    def paint(self, painter, option, widget=None):
        """Override paint to render the near-invisible background."""
        # Fill with near-transparent color to be hit-testable
        painter.fillRect(self.rect(), self.brush())


class MeetAndReadWidget(QGraphicsView):
    """
    Main application widget.
    
    Borderless, always-on-top widget with custom painted components
    living in a QGraphicsScene for smooth animations and complex visuals.
    
    Uses FLOATING PANELS (separate QWidgets) for transcript and settings
to avoid clipping issues and enable proper text rendering.
    """
    
    def __init__(self, parent=None, tray_manager=None):
        super().__init__(parent)
        
        # System tray integration
        self._tray_manager = tray_manager
        
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
        self.setStyleSheet("""
            MeetAndReadWidget {
                background: transparent;
                border: none;
            }
            QMenu {
                background-color: #2a2a2a;
                color: #ddd;
                border: 1px solid #555;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #4CAF50;
                color: #fff;
            }
        """)
        
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
        self._slide_state = _SlideState()
        
        # Recording controller
        self._controller = RecordingController()
        self._controller.on_state_change = self._on_controller_state_change
        self._controller.on_error = self._on_controller_error
        self._controller.on_phrase_result = self._on_phrase_result  # For accumulating processor
        self._controller.on_recording_complete = self._on_recording_complete
        self._controller.on_post_process_complete = self._on_post_process_complete
        self._error_indicator = None  # For showing errors
        
        # Floating panels (separate windows, not QGraphicsItems)
        self._floating_transcript_panel: Optional[FloatingTranscriptPanel] = None
        self._floating_settings_panel: Optional[FloatingSettingsPanel] = None
        
        # Create widget components
        self._create_components()
        self._create_floating_panels()
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
        
        # Context menu setup
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        print("DEBUG: Main widget initialized with floating panels")
    
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
    
    def _create_floating_panels(self):
        """Create floating panels (separate QWidgets)."""
        # Floating transcript panel
        self._floating_transcript_panel = FloatingTranscriptPanel(self)
        self._floating_transcript_panel.hide_panel()
        # Connect segment signal (thread-safe, automatically queues to main thread)
        self._floating_transcript_panel.segment_ready.connect(self._on_panel_segment)
        # Connect speaker name pin signal
        self._floating_transcript_panel.speaker_name_pinned.connect(self._on_speaker_name_pinned)
        print("DEBUG: Created floating transcript panel")
        
        # Floating settings panel — pass controller and tray_manager for Performance tab wiring
        self._floating_settings_panel = FloatingSettingsPanel(
            self,
            controller=self._controller,
            tray_manager=self._tray_manager,
        )
        self._floating_settings_panel.hide_panel()

        # Connect model_changed signal to save config
        self._floating_settings_panel.model_changed.connect(save_config)

        print("DEBUG: Created floating settings panel")
    
    def _on_panel_segment(self, text: str, confidence: int, segment_index: int, is_final: bool, phrase_start: bool):
        """Handle segment signal from panel (runs on main thread)."""
        print(f"DEBUG Panel Signal: text='{text[:30]}...', idx={segment_index}, phrase_start={phrase_start}")
        try:
            self._floating_transcript_panel.update_segment(
                text=text,
                confidence=confidence,
                segment_index=segment_index,
                is_final=is_final,
                phrase_start=phrase_start
            )
            print(f"DEBUG Panel: Updated via signal successfully")
        except Exception as e:
            print(f"DEBUG Panel: Error via signal: {e}")
    
    def _layout_components(self):
        """Position all components."""
        # Center the record button
        self.record_button.setPos(60, 20)
        
        # Position lobes on top 1/3rd of record button
        self.mic_lobe.setPos(50, 10)
        self.system_lobe.setPos(110, 10)
        
        # Settings lobe overlapping bottom of record button (like input lobes on top)
        self.settings_lobe.setPos(85, 85)
        
        # Error indicator at bottom
        self._error_indicator.setPos(10, 105)
    
    def _update_floating_panels_position(self):
        """Update position of floating panels based on widget position."""
        if not self._floating_transcript_panel or not self._floating_settings_panel:
            return
        
        # Get widget position in global coordinates
        widget_global_pos = self.mapToGlobal(self.rect().topLeft())
        
        # Determine position based on dock edge or default
        if self.dock_edge == 'right':
            transcript_pos = "left"
            settings_pos = "left"
        elif self.dock_edge == 'left':
            transcript_pos = "right"
            settings_pos = "right"
        elif self.dock_edge == 'top':
            transcript_pos = "bottom"
            settings_pos = "right"
        elif self.dock_edge == 'bottom':
            transcript_pos = "top"
            settings_pos = "right"
        else:
            # Default: panel flows to the left
            transcript_pos = "left"
            settings_pos = "right"
        
        # Update panel positions
        if self._floating_transcript_panel.isVisible():
            self._floating_transcript_panel.dock_to_widget(self, transcript_pos)
        
        if self._floating_settings_panel.isVisible():
            self._floating_settings_panel.dock_to_widget(self, settings_pos)
    
    def moveEvent(self, event):
        """Handle widget move - update floating panel positions."""
        super().moveEvent(event)
        self._update_floating_panels_position()
    
    def _position_initial(self):
        """Position widget on screen initially."""
        try:
            settings = get_config()
            if settings.ui.widget_position:
                # Restore saved position
                x, y = settings.ui.widget_position
                self.move(x, y)
                print(f"DEBUG: Restored widget position: ({x}, {y})")
                
                # Restore dock state if applicable
                if settings.ui.widget_dock_edge:
                    self.dock_edge = settings.ui.widget_dock_edge
                    self.is_docked = True
                    self._update_docked_state()
                
                # Off-screen recovery: if saved position is no longer valid
                # (e.g. monitor disconnected), snap to nearest valid position
                self._recover_offscreen_position()
                return
        except Exception as e:
            print(f"DEBUG: Failed to restore position: {e}")
        
        # Default: Start in bottom-right corner
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 40
        self.move(x, y)
    
    def _recover_offscreen_position(self):
        """Recover widget position if it's off all available screens.

        Checks whether the widget's top-left corner is visible on any
        screen. If not, repositions to the bottom-right of the primary
        screen. Logs the recovery for diagnostics.
        """
        pos = self.pos()
        widget_point = QPoint(pos.x(), pos.y())
        
        # Check if the widget's position is on any available screen
        screens = QApplication.screens()
        on_screen = False
        for screen in screens:
            if screen.geometry().contains(widget_point):
                on_screen = True
                break
        
        if not on_screen:
            primary = QApplication.primaryScreen().geometry()
            new_x = primary.width() - self.width() - 20
            new_y = primary.height() - self.height() - 40
            logging.getLogger(__name__).info(
                "Widget position (%d, %d) is off-screen. "
                "Recovering to (%d, %d).",
                pos.x(), pos.y(), new_x, new_y,
            )
            self.move(new_x, new_y)
    
    def _update_animations(self):
        """Update animation states."""
        # Advance record button state transitions (~200ms eased cross-fade)
        self.record_button.tick()

        # --- Slide animation (edge docking) ---
        s = self._slide_state
        if s.active:
            elapsed = int(_time.monotonic() * 1000) - s.start_time_ms
            progress = min(elapsed / s.duration_ms, 1.0)
            # Ease-out deceleration: t = 1 - (1 - progress)^2
            t = 1.0 - (1.0 - progress) ** 2
            ix = int(s.start_pos.x() + (s.target_pos.x() - s.start_pos.x()) * t)
            iy = int(s.start_pos.y() + (s.target_pos.y() - s.start_pos.y()) * t)
            self.move(ix, iy)
            if progress >= 1.0:
                s.active = False
                self.move(s.target_pos)
                logging.getLogger(__name__).debug(
                    "Slide complete: widget at %s", s.target_pos
                )
                print(f"DEBUG: Slide complete: widget at {s.target_pos}")
        
        if self.is_recording and not self.is_processing:
            self.pulse_phase += 0.1
            if self.pulse_phase > 6.28:  # 2*PI
                self.pulse_phase = 0.0
            self.record_button.pulse_phase = self.pulse_phase
            self.record_button.update()
        elif self.is_processing:
            self.pulse_phase += 0.2
            if self.pulse_phase > 6.28:
                self.pulse_phase = 0.0
            self.record_button.swirl_phase = self.pulse_phase
            self.record_button.update()
        else:
            # Idle — reset animation phases so stale values don't leak
            if self.pulse_phase != 0.0:
                self.pulse_phase = 0.0
                self.record_button.pulse_phase = 0.0
                self.record_button.swirl_phase = 0.0
            # Force state to idle if somehow stuck
            if self.record_button._to_key != 'idle':
                self.record_button._from_key = self.record_button._to_key
                self.record_button._to_key = 'idle'
                self.record_button._state_t = 0.0
                self.record_button.update()

    def mousePressEvent(self, event):
        """Record press position for click vs drag detection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.globalPosition().toPoint()
            self.widget_start_pos = self.pos()
            self.press_time = QTime.currentTime()
            self._click_consumed = False
            # DON'T accept - let events propagate to child items
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle dragging from any component."""
        if self.is_dragging:
            delta = event.globalPosition().toPoint() - self.drag_start_pos
            new_pos = self.widget_start_pos + delta
            self.move(new_pos)
            self.is_docked = False
            self.dock_edge = None
            self._update_docked_state()
            self._update_floating_panels_position()
            event.accept()
        elif event.buttons() & Qt.MouseButton.LeftButton:
            current_pos = event.globalPosition().toPoint()
            movement = (current_pos - self.drag_start_pos).manhattanLength()
            if movement >= 5:
                # Drag threshold exceeded - start dragging, consume click
                self.is_dragging = True
                self._click_consumed = True
                delta = current_pos - self.drag_start_pos
                new_pos = self.widget_start_pos + delta
                self.move(new_pos)
                self.is_docked = False
                self.dock_edge = None
                self._update_docked_state()
                self._update_floating_panels_position()
                event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle click vs drag release."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_dragging:
                self.is_dragging = False
                self._check_snap_to_edge()
                event.accept()
            elif self._click_consumed:
                event.accept()
            else:
                # Short click - let child items handle it
                super().mouseReleaseEvent(event)
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
        self._update_floating_panels_position()
    
    @property
    def _peek_width(self) -> int:
        """Width of the visible peek strip when docked (1/5th of widget)."""
        return int(self.width() * 0.2)

    def _start_slide_to(self, target_pos: QPoint):
        """Begin a smooth slide animation from current position to *target_pos*.

        The animation is driven by the existing ``animation_timer`` at ~30 fps
        and completes in 300 ms with ease-out deceleration.
        """
        s = self._slide_state
        s.start_pos = self.pos()
        s.target_pos = target_pos
        s.start_time_ms = int(_time.monotonic() * 1000)
        s.duration_ms = 300
        s.active = True
        logging.getLogger(__name__).debug(
            "Slide start: %s → %s", s.start_pos, target_pos
        )
        print(f"DEBUG: Slide start: {s.start_pos} → {target_pos}")

    def _update_docked_state(self):
        """Update widget appearance based on docked state.

        When docked, slides the widget to a peek position where only 1/5th
        of its width is visible at the screen edge. Uses smooth 300 ms
        slide animation instead of instant positioning.
        """
        if self.is_docked:
            peek = self._peek_width
            if self.dock_edge == 'right':
                target_x = QApplication.primaryScreen().geometry().width() - peek
                self._start_slide_to(QPoint(target_x, self.y()))
            elif self.dock_edge == 'left':
                target_x = -(self.width() - peek)
                self._start_slide_to(QPoint(target_x, self.y()))
    
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
            
            # Show floating transcript panel when recording starts
            if self._floating_transcript_panel:
                print("DEBUG: Showing floating transcript panel")
                self._floating_transcript_panel.clear()
                # Only dock to widget on first show; preserve user position after that
                if not self._floating_transcript_panel._has_been_docked:
                    self._floating_transcript_panel.dock_to_widget(self, self._get_panel_position())
                self._floating_transcript_panel.show_panel()
                # Switch to Live tab
                self._floating_transcript_panel._tab_widget.setCurrentIndex(0)
            else:
                print("DEBUG: No floating transcript panel to show!")
            
        elif state == ControllerState.STARTING:
            # No visual change during startup — wait for RECORDING or ERROR
            pass
            
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
            
            # Keep transcript panel visible for review
            # User can manually close it or it will be cleared on next recording
            
        elif state == ControllerState.ERROR:
            self.is_recording = False
            self.is_processing = False
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(False)
        
        # Forward state to tray icon manager
        if self._tray_manager is not None:
            self._tray_manager.update_recording_state(state)
    
    def _get_panel_position(self):
        """Determine where to dock the panel based on widget position."""
        if self.dock_edge == 'right':
            return "left"
        elif self.dock_edge == 'left':
            return "right"
        elif self.dock_edge == 'top':
            return "bottom"
        elif self.dock_edge == 'bottom':
            return "top"
        else:
            return "left"  # Default
    
    def _on_controller_error(self, error):
        """Handle controller errors."""
        self._show_error(error.message)
        print(f"Recording error: {error.message}")
    
    def _on_phrase_result(self, result: SegmentResult):
        """Handle segment result from accumulating transcription processor.

        Thread-safe: emits signal which automatically queues to main thread.

        Args:
            result: SegmentResult with text, confidence, and completion status
        """
        phrase_start = getattr(result, 'phrase_start', False)

        print(f"DEBUG UI: Segment: '{result.text}' [conf: {result.confidence}%, final: {result.is_final}, phrase_start: {phrase_start}]")

        if self._floating_transcript_panel:
            # Emit signal (thread-safe, automatically queues to main thread)
            self._floating_transcript_panel.segment_ready.emit(
                result.text,
                result.confidence,
                result.segment_index,
                result.is_final,
                phrase_start
            )
            print(f"DEBUG UI: Emitted signal with phrase_start={phrase_start}")
        else:
            print("DEBUG UI: No floating transcript panel available!")
    
    def _on_recording_complete(self, wav_path, transcript_path):
        """Handle recording completion."""
        self.is_processing = False
        self.record_button.set_processing_state(False)
        print(f"DEBUG: Recording saved to: {wav_path}")
        if transcript_path:
            print(f"DEBUG: Transcript saved to: {transcript_path}")
        
        # Update panel status
        if self._floating_transcript_panel:
            self._floating_transcript_panel.status_label.setText("Recording complete - Post-processing...")
    
    def _on_post_process_complete(self, job_id, transcript_path):
        """Handle post-processing completion.

        Args:
            job_id: The post-processing job ID
            transcript_path: Path to the post-processed transcript file
        """
        print(f"DEBUG UI: Post-processing complete! Job: {job_id}")
        print(f"DEBUG UI: Post-processed transcript saved to: {transcript_path}")

        # Update panel status and switch to History tab
        if self._floating_transcript_panel:
            self._floating_transcript_panel.status_label.setText(f"Post-processed transcript saved!")
            # Switch to History tab to show the completed recording
            self._floating_transcript_panel._tab_widget.setCurrentIndex(1)
            # Refresh the history list to pick up the new transcript
            self._floating_transcript_panel._refresh_history()
            QTimer.singleShot(3000, lambda: self._floating_transcript_panel.status_label.setText("Ready"))

        # Update Performance tab WER display
        if self._floating_settings_panel:
            wer = self._controller.get_last_wer()
            self._floating_settings_panel.update_wer_display(wer)

    def _on_speaker_name_pinned(self, raw_label: str, name: str):
        """Handle user pinning a speaker name in the transcript panel.

        Saves the voice signature for the named speaker and refreshes
        all speaker labels in the transcript display.

        Args:
            raw_label: Raw speaker label from diarization (e.g. "spk0")
            name: User-chosen display name for this speaker
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info("Speaker name pinned: %s -> '%s'", raw_label, name)

        # Save the signature via controller
        self._controller.pin_speaker_name(raw_label, name)

        # Refresh the transcript panel with updated speaker names
        if self._floating_transcript_panel:
            # Get the updated speaker names from the controller
            speaker_names = self._controller.get_speaker_names()
            if speaker_names:
                self._floating_transcript_panel.set_speaker_names(speaker_names)

    def toggle_transcript_panel(self):
        """Toggle floating transcript panel visibility."""
        if self._floating_transcript_panel:
            if self._floating_transcript_panel.isVisible():
                self._floating_transcript_panel.hide_panel()
            else:
                self._floating_transcript_panel.dock_to_widget(self, self._get_panel_position())
                self._floating_transcript_panel.show_panel()
    
    def _toggle_settings_panel(self):
        """Toggle floating settings panel visibility."""
        if self._floating_settings_panel:
            if self._floating_settings_panel.isVisible():
                self._floating_settings_panel.hide_panel()
            else:
                self._floating_settings_panel.dock_to_widget(self, "right")
                self._floating_settings_panel.show_panel()
    
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
        error = self._controller.stop()
        if error:
            self._show_error(error.message)
    
    def _show_context_menu(self, position):
        """Show context menu with Exit action."""
        menu = QMenu(self)
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self._exit_application)
        menu.exec(self.mapToGlobal(position))
    
    def _exit_application(self):
        """Exit the application cleanly (full quit, even with tray)."""
        self._save_position()
        # Hide all floating panels first
        if self._floating_transcript_panel:
            self._floating_transcript_panel.hide()
        if self._floating_settings_panel:
            self._floating_settings_panel.hide()
        if self._tray_manager is not None:
            self._tray_manager.hide()
        QApplication.quit()
    
    def _save_position(self):
        """Save widget position to config."""
        try:
            set_config('ui.widget_position', (self.x(), self.y()))
            set_config('ui.widget_dock_edge', self.dock_edge)
            save_config()
            print(f"DEBUG: Saved widget position: ({self.x()}, {self.y()}), dock: {self.dock_edge}")
        except Exception as e:
            print(f"DEBUG: Failed to save position: {e}")
    
    def closeEvent(self, event):
        """Handle close event — close-to-tray if tray is active, else quit.

        When a TrayIconManager is wired in, closing the window hides it to
        the system tray instead of quitting the app. This lets users keep
        recording in the background. Without a tray manager, the app quits
        normally (ALT+F4, etc.).
        """
        self._save_position()
        
        if self._tray_manager is not None:
            # Close-to-tray: hide the widget instead of quitting
            logging.getLogger(__name__).info(
                "closeEvent: hiding widget to system tray"
            )
            self.hide()
            event.ignore()
        else:
            event.accept()
            QApplication.quit()


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
        
        # State transition animation (~200ms eased cross-fade)
        self._state_t = 1.0  # 0.0 (old state) → 1.0 (new state)
        self._transition_speed = 1.0 / 6  # ~200ms at 33ms/frame
        self._from_key = 'idle'
        self._to_key = 'idle'
    
    def _state_key(self):
        """Return current visual state key."""
        if self.is_processing:
            return 'processing'
        if self.is_recording:
            return 'recording'
        return 'idle'
    
    def tick(self):
        """Advance state transition animation by one frame (~33ms)."""
        if self._state_t < 1.0:
            self._state_t = min(1.0, self._state_t + self._transition_speed)
            self.update()
    
    @staticmethod
    def _ease_out(t):
        """Quadratic ease-out curve for smooth deceleration."""
        return 1 - (1 - t) ** 2
    
    def set_recording_state(self, recording):
        """Update recording state with eased transition."""
        self._from_key = self._to_key
        self.is_recording = recording
        self._to_key = self._state_key()
        self._state_t = 0.0
        self.update()
    
    def set_processing_state(self, processing):
        """Update processing state with eased transition."""
        self._from_key = self._to_key
        self.is_processing = processing
        self._to_key = self._state_key()
        self._state_t = 0.0
        self.update()
    
    def set_pulse_phase(self, phase):
        """Set pulse animation phase."""
        self.pulse_phase = phase

    def set_swirl_phase(self, phase):
        """Set swirl animation phase."""
        self.swirl_phase = phase
    
    def paint(self, painter, option, widget=None):
        """Custom paint with eased cross-fade between states."""
        rect = self.rect()
        
        if self._state_t < 1.0:
            t = self._ease_out(self._state_t)
            # Draw previous state fading out
            painter.save()
            painter.setOpacity(1.0 - t)
            self._paint_for_state(painter, rect, self._from_key)
            self._paint_icon_for_state(painter, rect, self._from_key)
            painter.restore()
            # Draw new state fading in
            painter.save()
            painter.setOpacity(t)
            self._paint_for_state(painter, rect, self._to_key)
            self._paint_icon_for_state(painter, rect, self._to_key)
            painter.restore()
        else:
            self._paint_for_state(painter, rect, self._to_key)
            self._paint_icon_for_state(painter, rect, self._to_key)
    
    def _paint_for_state(self, painter, rect, state_key):
        """Paint the visual appearance for a given state key."""
        if state_key == 'recording':
            self._paint_recording(painter, rect)
        elif state_key == 'processing':
            self._paint_swirl(painter, rect)
        else:
            self._paint_idle(painter, rect)
    
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
    
    def _paint_icon_for_state(self, painter, rect, state_key):
        """Paint record/stop icon for a given state key."""
        center = rect.center()
        
        if state_key == 'recording':
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
        """Accept press to get release event — action fires on release."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()

    def mouseReleaseEvent(self, event):
        """Fire action on release, only if this wasn't a drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.parent_widget.is_dragging and not self.parent_widget._click_consumed:
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
        self.setTransformOriginPoint(20, 20)  # Center of 40×40
        self._hovered = False
    
    def hoverEnterEvent(self, event):
        """Scale up and brighten on hover."""
        self._hovered = True
        self.setScale(1.05)
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Revert to normal on hover leave."""
        self._hovered = False
        self.setScale(1.0)
        self.update()
        super().hoverLeaveEvent(event)
    
    def paint(self, painter, option, widget=None):
        """Paint the lobe with hover glow effect."""
        rect = self.rect()
        
        # Hover glow (subtle outer ring)
        if self._hovered:
            for i in range(2, 0, -1):
                glow_alpha = 40 if self.is_active else 25
                painter.setBrush(QBrush(QColor(255, 255, 255, glow_alpha // i)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(rect.adjusted(-i * 3, -i * 3, i * 3, i * 3))
        
        if self.is_active:
            # Active state - bright, boosted on hover
            fill_alpha = 230 if self._hovered else 200
            painter.setBrush(QBrush(QColor(100, 200, 100, fill_alpha)))
            painter.setPen(QPen(QColor(150, 255, 150, 255), 2))
        else:
            # Inactive state - dim, brightened on hover
            fill_alpha = 190 if self._hovered else 150
            border_alpha = 240 if self._hovered else 200
            fill_val = 140 if self._hovered else 100
            border_val = 200 if self._hovered else 150
            painter.setBrush(QBrush(QColor(fill_val, fill_val, fill_val, fill_alpha)))
            painter.setPen(QPen(QColor(border_val, border_val, border_val, border_alpha), 2))
        
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
        """Accept press to get release event — action fires on release."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()

    def mouseReleaseEvent(self, event):
        """Toggle on release, only if this wasn't a drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.parent_widget.is_dragging and not self.parent_widget._click_consumed:
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
        self.setTransformOriginPoint(15, 15)  # Center of 30×30
        self._hovered = False
    
    def hoverEnterEvent(self, event):
        """Scale up and brighten on hover."""
        self._hovered = True
        self.setScale(1.05)
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Revert to normal on hover leave."""
        self._hovered = False
        self.setScale(1.0)
        self.update()
        super().hoverLeaveEvent(event)
    
    def paint(self, painter, option, widget=None):
        """Paint settings lobe with hover glow effect."""
        rect = self.rect()
        
        # Hover glow (subtle outer ring)
        if self._hovered:
            for i in range(2, 0, -1):
                painter.setBrush(QBrush(QColor(255, 255, 255, 30 // i)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(rect.adjusted(-i * 3, -i * 3, i * 3, i * 3))
        
        fill_alpha = 220 if self._hovered else 180
        fill_val = 130 if self._hovered else 100
        painter.setBrush(QBrush(QColor(fill_val, fill_val, 220 if self._hovered else 200, fill_alpha)))
        painter.setPen(QPen(QColor(150, 150, 255, 255), 2))
        painter.drawEllipse(rect)
        
        # Draw gear icon
        painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        center = rect.center()
        painter.drawEllipse(int(center.x() - 6), int(center.y() - 6), 12, 12)
        painter.drawPoint(int(center.x()), int(center.y()))
    
    def mousePressEvent(self, event):
        """Accept press to get release event — action fires on release."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()

    def mouseReleaseEvent(self, event):
        """Open settings on release, only if this wasn't a drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.parent_widget.is_dragging and not self.parent_widget._click_consumed:
                self.parent_widget._toggle_settings_panel()
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
