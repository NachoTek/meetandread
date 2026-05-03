"""
Main meetandread widget implementation using QGraphicsView with floating panels.

This widget provides a borderless, always-on-top interface with:
- Record button as main body
- Toggle lobes for audio input selection
- CCOverlayPanel for live closed-caption transcription
- FloatingSettingsPanel for configuration
- Drag clamped to visible desktop area
- Real-time transcription display with confidence colors

Key improvement: Uses floating QWidget panels instead of QGraphicsItem panels
to avoid clipping issues and enable proper text editing.
"""

from enum import Enum, auto
from pathlib import Path
from typing import Optional
import logging
import math as _math
import re
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsWidget, QApplication, QGraphicsItemGroup, QWidget, QMenu
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, QTimer, QTime, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QLinearGradient

from meetandread.recording import RecordingController, ControllerState, ControllerError
from meetandread.transcription.confidence import get_confidence_color, get_distortion_intensity
from meetandread.transcription.transcript_store import Word
from meetandread.transcription.accumulating_processor import SegmentResult
from meetandread.config import get_config, set_config, save_config, AppSettings
from meetandread.hardware.recommender import ModelRecommender, get_model_info
from meetandread.widgets.floating_panels import FloatingSettingsPanel, CCOverlayPanel, ensure_on_screen
from meetandread.widgets.theme import context_menu_css, current_palette


class _ControllerBridge(QObject):
    """Thread-safe signal bridge for controller callbacks.

    Controller callbacks fire from worker threads (stop worker, post-processor).
    Qt widgets must only be modified from the UI thread. This bridge emits
    signals that Qt automatically queues to the UI thread's event loop,
    preventing freezes, deadlocks, and undefined behavior.
    """
    state_changed = pyqtSignal(object)       # ControllerState
    error_occurred = pyqtSignal(object)      # ControllerError
    recording_complete = pyqtSignal(object, object)  # wav_path, transcript_path
    post_process_complete = pyqtSignal(str, object)   # job_id, transcript_path
    phrase_result = pyqtSignal(object)       # SegmentResult



class _WidgetVisualStateMachine:
    """Owns the widget's current visual state and eased transition progress.

    States: IDLE → RECORDING → PROCESSING → IDLE.
    Transitions animate over ~200 ms at 30 fps (6 frames).  Transition
    progress goes from 0.0 (old state fully visible) to 1.0 (new state
    fully visible), using quadratic ease-out deceleration matching
    RecordButtonItem._ease_out().
    """

    def __init__(self, initial_state: "WidgetVisualState"):
        self._current: WidgetVisualState = initial_state
        self._previous: WidgetVisualState = initial_state
        self._progress: float = 1.0  # 0.0 → 1.0; 1.0 means settled
        self._speed: float = 1.0 / 6  # ~200 ms at 33 ms/frame

    # -- public API --------------------------------------------------------

    @property
    def current(self) -> "WidgetVisualState":
        return self._current

    @property
    def previous(self) -> "WidgetVisualState":
        return self._previous

    @property
    def progress(self) -> float:
        """Transition progress: 0.0 (old state) → 1.0 (new state)."""
        return self._progress

    def transition_to(self, new_state: "WidgetVisualState") -> None:
        """Begin an eased transition to *new_state*.

        If already transitioning to the same state this is a no-op.
        If mid-transition, the current interpolated position becomes the
        new starting point so there is no visual jump.
        """
        if new_state == self._current and self._progress >= 1.0:
            return  # already settled at this state
        if new_state != self._current:
            logging.debug(
                "WidgetVisualState: %s → %s (was progress=%.2f)",
                self._current.name, new_state.name, self._progress,
            )
            self._previous = self._current
            self._current = new_state
            self._progress = 0.0

    def tick(self) -> None:
        """Advance one animation frame (~33 ms)."""
        if self._progress < 1.0:
            self._progress = min(1.0, self._progress + self._speed)

    @staticmethod
    def _ease_out(t: float) -> float:
        """Quadratic ease-out curve (same as RecordButtonItem._ease_out)."""
        return 1.0 - (1.0 - t) ** 2

    def current_opacity(self) -> float:
        """Return the eased opacity of the *new* state (0.0 → 1.0).

        The old state's implied opacity is ``1.0 - current_opacity()``.
        """
        return self._ease_out(self._progress)

    def current_properties(self) -> dict:
        """Return a dict describing the current interpolated visual state.

        Keys:
            state   – current WidgetVisualState
            opacity – eased opacity of the new state (0.0–1.0)
            settled – True when the transition is complete
        """
        return {
            "state": self._current,
            "opacity": self.current_opacity(),
            "settled": self._progress >= 1.0,
        }


class WidgetVisualState(Enum):
    """Named visual states for the main widget."""

    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


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

        # Visual state machine (idle → recording → processing → idle)
        self._visual_state = _WidgetVisualStateMachine(WidgetVisualState.IDLE)

        # Glass idle appearance — start translucent since initial state is IDLE
        self.setWindowOpacity(self._IDLE_OPACITY)
        
        # Recording controller
        self._controller = RecordingController()

        # Thread-safe signal bridge: controller callbacks fire from worker threads,
        # but Qt widgets must only be modified from the UI thread. The bridge emits
        # signals that Qt automatically queues to the UI thread's event loop.
        self._bridge = _ControllerBridge(self)
        self._bridge.state_changed.connect(self._on_controller_state_change)
        self._bridge.error_occurred.connect(self._on_controller_error)
        self._bridge.recording_complete.connect(self._on_recording_complete)
        self._bridge.post_process_complete.connect(self._on_post_process_complete)
        self._bridge.phrase_result.connect(self._on_phrase_result)

        # Controller callbacks emit bridge signals (thread-safe)
        self._controller.on_state_change = self._bridge.state_changed.emit
        self._controller.on_error = self._bridge.error_occurred.emit
        self._controller.on_phrase_result = lambda result: self._bridge.phrase_result.emit(result)
        self._controller.on_recording_complete = lambda wav, t: self._bridge.recording_complete.emit(wav, t)
        self._controller.on_post_process_complete = lambda jid, tp: self._bridge.post_process_complete.emit(jid, tp)
        self._error_indicator = None  # For showing errors
        self._warning_indicator = None  # For showing resource warnings
        self._warning_hide_timer: Optional[QTimer] = None  # Auto-hide timer for warnings
        self._error_hide_timer: Optional[QTimer] = None  # Auto-hide timer for errors
        
        # Floating panels (separate windows, not QGraphicsItems)
        self._floating_settings_panel: Optional[FloatingSettingsPanel] = None
        self._cc_overlay: Optional[CCOverlayPanel] = None
        
        # Create widget components
        self._create_components()
        self._create_floating_panels()
        self._layout_components()
        
        # Set initial size
        self.setFixedSize(200, 140)
        self._scene.setSceneRect(0, 0, 200, 140)
        
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
        
        logging.debug("Main widget initialized with floating panels")
    
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
        
        # Transcript lobe (toggle transcript panel)
        self.transcript_lobe = TranscriptLobeItem(self)
        self._scene.addItem(self.transcript_lobe)
        
        # Settings lobe
        self.settings_lobe = SettingsLobeItem(self)
        self._scene.addItem(self.settings_lobe)
        
        # Error indicator (hidden by default)
        self._error_indicator = ErrorIndicatorItem(self)
        self._error_indicator.hide()
        self._scene.addItem(self._error_indicator)

        # Warning indicator for resource warnings (hidden by default)
        self._warning_indicator = ErrorIndicatorItem(self)
        self._warning_indicator.set_color_mode("warning")
        self._warning_indicator.hide()
        self._scene.addItem(self._warning_indicator)
        
        # Restore persisted audio source selection
        self._restore_audio_sources()
        
        # Probe system audio loopback availability
        self._probe_system_audio_availability()
        
        # Pulse animation timer (created lazily by _pulse_lobes)
        self._pulse_timer: Optional[QTimer] = None
    
    def _create_floating_panels(self):
        """Create floating panels (separate QWidgets)."""
        # Floating settings panel — pass controller and tray_manager for Performance tab wiring
        self._floating_settings_panel = FloatingSettingsPanel(
            self,
            controller=self._controller,
            tray_manager=self._tray_manager,
            main_widget=self,
        )
        self._floating_settings_panel.hide_panel()

        # Connect model_changed signal to save config
        self._floating_settings_panel.model_changed.connect(save_config)

        # Connect CC font size signal to apply font size live
        self._floating_settings_panel.cc_font_size_changed.connect(
            self._on_cc_font_size_changed
        )

        logging.debug("Created floating settings panel")

        # CC overlay — compact caption panel for live transcript
        self._cc_overlay = CCOverlayPanel()
        self._cc_overlay.hide()
        # Apply saved CC font size
        try:
            from meetandread.config import get_config
            _saved_size = get_config("transcription.cc_font_size")
            if isinstance(_saved_size, (int, float)) and 16 <= _saved_size <= 96:
                self._cc_overlay.set_font_size(int(_saved_size))
        except Exception:
            pass
        # Connect segment signal for thread-safe delivery from _on_phrase_result
        self._cc_overlay.segment_ready.connect(self._on_cc_segment)
        logging.debug("Created CC overlay panel")

    def _on_panel_segment(self, text: str, confidence: int, segment_index: int, is_final: bool, phrase_start: bool):
        """Handle segment signal (kept for compatibility — CC overlay handles its own updates)."""
        pass

    def _on_cc_segment(self, text: str, confidence: int, segment_index: int, is_final: bool, phrase_start: bool):
        """Handle segment signal from CC overlay (runs on main thread).

        Re-emits to the CC overlay's update_segment for live display.
        The CC overlay is the primary live transcript surface during
        recording; the legacy FloatingTranscriptPanel is retained for
        tests until final cleanup.
        """
        try:
            if self._cc_overlay:
                self._cc_overlay.update_segment(
                    text=text,
                    confidence=confidence,
                    segment_index=segment_index,
                    is_final=is_final,
                    phrase_start=phrase_start,
                )
        except Exception as e:
            logging.error("CC overlay update failed: %s", e)

    def _on_cc_font_size_changed(self, size_px: int) -> None:
        """Apply CC overlay font size change immediately."""
        if self._cc_overlay:
            self._cc_overlay.set_font_size(size_px)
    
    def _layout_components(self):
        """Position all components."""
        # Center the record button
        self.record_button.setPos(60, 20)
        
        # Position lobes on top 1/3rd of record button
        self.mic_lobe.setPos(50, 10)
        self.system_lobe.setPos(110, 10)
        
        # Transcript lobe center on record button perimeter, lower-left
        self.transcript_lobe.setPos(54, 71)
        
        # Settings lobe mirrored around Y axis from transcript lobe (lower-right)
        self.settings_lobe.setPos(116, 71)
        
        # Error indicator at bottom
        self._error_indicator.setPos(10, 105)

        # Warning indicator below error indicator
        self._warning_indicator.setPos(10, 120)
    
    def moveEvent(self, event):
        """Handle widget move — no panel repositioning (panels are independent)."""
        super().moveEvent(event)
    
    def _position_initial(self):
        """Position widget on screen initially."""
        try:
            settings = get_config()
            if settings.ui.widget_position:
                # Restore saved position
                x, y = settings.ui.widget_position
                self.move(x, y)
                logging.debug("Restored widget position: (%d, %d)", x, y)
                
                # Off-screen recovery: if saved position is no longer valid
                # (e.g. monitor disconnected), snap to nearest valid position
                self._recover_offscreen_position()
                return
        except Exception as e:
            logging.warning("Failed to restore position: %s", e)
        
        # Default: Start in bottom-right corner
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 40
        self.move(x, y)
    
    def _recover_offscreen_position(self):
        """Recover widget position if it's off all available screens.

        Checks whether the widget's center point is visible on any
        screen. If not, repositions to the center of the primary
        monitor. Logs the recovery for diagnostics.
        """
        pos = self.pos()
        widget_center = QPoint(
            pos.x() + self.width() // 2,
            pos.y() + self.height() // 2,
        )

        # Check if the widget's center is on any available screen
        screens = QApplication.screens()
        on_screen = False
        for screen in screens:
            if screen.geometry().contains(widget_center):
                on_screen = True
                break

        if not on_screen:
            primary = QApplication.primaryScreen().geometry()
            new_x = primary.x() + (primary.width() - self.width()) // 2
            new_y = primary.y() + (primary.height() - self.height()) // 2
            logging.getLogger(__name__).info(
                "Widget position (%d, %d) is off-screen. "
                "Recovering to center of primary monitor (%d, %d).",
                pos.x(), pos.y(), new_x, new_y,
            )
            self.move(new_x, new_y)
    
    # -- Glass opacity constants for visual state ----------------------------
    _IDLE_OPACITY = 0.87      # translucent glass when idle
    _ACTIVE_OPACITY = 1.0     # fully opaque when recording/processing

    def _update_visual_state_opacity(self):
        """Interpolate window opacity between IDLE (0.87) and active (1.0).

        Uses the state machine's eased progress so the opacity change is
        smooth over ~200 ms (6 frames at 33 ms).  Called every frame from
        ``_update_animations()``.
        """
        props = self._visual_state.current_properties()
        target_state = props["state"]
        eased_t = props["opacity"]  # 0.0 → 1.0 (eased)

        # Determine which opacity each end maps to
        prev_state = self._visual_state.previous
        from_opacity = (self._IDLE_OPACITY
                        if prev_state == WidgetVisualState.IDLE
                        else self._ACTIVE_OPACITY)
        to_opacity = (self._IDLE_OPACITY
                      if target_state == WidgetVisualState.IDLE
                      else self._ACTIVE_OPACITY)

        interpolated = from_opacity + (to_opacity - from_opacity) * eased_t
        self.setWindowOpacity(interpolated)

        if not props["settled"]:
            logging.debug(
                "Glass opacity: %.3f (state=%s progress=%.2f)",
                interpolated, target_state.name, self._visual_state.progress,
            )

    def _update_animations(self):
        """Update animation states."""
        # Advance record button state transitions (~200ms eased cross-fade)
        self.record_button.tick()

        # Advance widget visual state machine (~200ms eased transition)
        self._visual_state.tick()

        # Interpolate window opacity for glass idle effect
        self._update_visual_state_opacity()

        if self.is_recording and not self.is_processing:
            self.pulse_phase += 0.05  # ~2s period at 30fps
            self.record_button.pulse_phase = self.pulse_phase
            self.record_button.update()
        elif self.is_processing:
            self.pulse_phase += 0.2
            if self.pulse_phase > 6.28:
                self.pulse_phase = 0.0
            self.record_button.swirl_phase = self.pulse_phase
            self.record_button.update()
        else:
            # Idle — let animation phases decay during cross-fade rather than
            # snapping to zero, which would cause a visual jump in the fading-
            # out state.  Only reset once the RecordButtonItem cross-fade has
            # fully settled so no stale painting references the phases.
            if self.record_button._state_t >= 1.0:
                # Cross-fade settled — safe to reset phases
                if self.pulse_phase != 0.0:
                    logging.debug(
                        "Phase decay complete: resetting pulse_phase=%.2f",
                        self.pulse_phase,
                    )
                    self.pulse_phase = 0.0
                    self.record_button.pulse_phase = 0.0
                    self.record_button.swirl_phase = 0.0
            else:
                # Still cross-fading — keep driving phases so the fading-out
                # state continues to paint smoothly (pulse decays naturally
                # since we stop incrementing; swirl wraps harmlessly).
                pass
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
        """Handle dragging from any component, clamped to visible screen area."""
        if self.is_dragging:
            delta = event.globalPosition().toPoint() - self.drag_start_pos
            new_pos = self.widget_start_pos + delta
            self.move(self._clamp_to_desktop(new_pos))
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
                self.move(self._clamp_to_desktop(new_pos))
                event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle click vs drag release."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_dragging:
                self.is_dragging = False
                event.accept()
            elif self._click_consumed:
                event.accept()
            else:
                # Short click - let child items handle it
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)
    

    def _clamp_to_desktop(self, pos: QPoint) -> QPoint:
        """Clamp *pos* so widget stays fully within the visible desktop."""
        screens = QApplication.screens()
        if not screens:
            return pos
        try:
            min_x = min(s.geometry().x() for s in screens)
            min_y = min(s.geometry().y() for s in screens)
            max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
            max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        except (AttributeError, TypeError):
            return pos
        w, h = self.width(), self.height()
        x = max(min_x, min(pos.x(), max_x - w))
        y = max(min_y, min(pos.y(), max_y - h))
        return QPoint(x, y)

    def _get_selected_sources(self):
        """Get set of selected audio sources based on lobe states."""
        sources = set()
        if self.mic_lobe.is_active:
            sources.add('mic')
        if self.system_lobe.is_active:
            sources.add('system')
        return sources
    
    def _on_lobe_toggled(self):
        """Persist the current audio source selection to config.

        Called by ToggleLobeItem after toggling its active state.
        """
        sources = list(self._get_selected_sources())
        set_config('ui.audio_sources', sources)
        save_config()
    
    def _restore_audio_sources(self):
        """Restore lobe active states from persisted config.

        On first launch (None), both lobes stay inactive. Otherwise
        sets each lobe active if its name appears in the stored list.
        """
        sources = get_config('ui.audio_sources')
        if sources is None or not isinstance(sources, list):
            return
        self.mic_lobe.is_active = ('mic' in sources)
        self.system_lobe.is_active = ('system' in sources)
        self.mic_lobe.update()
        self.system_lobe.update()
    
    def _probe_system_audio_availability(self):
        """Probe system audio loopback availability and mark lobe if absent.

        On non-Windows platforms or when pyaudiowpatch is not installed the
        probe gracefully degrades — the lobe stays available (optimistic
        default per failure-mode spec).
        """
        try:
            from meetandread.audio.capture.devices import get_default_loopback_device
            if get_default_loopback_device() is None:
                self.system_lobe.set_unavailable(True)
                logging.info("System audio lobe marked unavailable — no loopback device")
            else:
                logging.debug("System audio loopback device detected")
        except Exception as exc:
            # Optimistic: keep lobe available on probe errors
            logging.warning("System audio probe failed, keeping lobe available: %s", exc)
    
    def _pulse_lobes(self):
        """Animate both lobes with an opacity pulse for ~2 seconds.

        Oscillates ``_pulse_opacity`` between 0.3 and 1.0 over 4-5 cycles
        (20 ticks at 100 ms = 2 s), then resets to 1.0 and stops.
        """
        if self._pulse_timer is not None:
            self._pulse_timer.stop()
        
        tick_count = 0
        
        def _tick():
            nonlocal tick_count
            tick_count += 1
            # Sinusoidal oscillation: 0.3 → 1.0 → 0.3 …
            # 4 cycles over 20 ticks → period = 5 ticks
            import math
            phase = (tick_count % 5) / 5.0 * 2 * math.pi
            opacity = 0.65 + 0.35 * math.cos(phase)  # ranges [0.3, 1.0]
            
            self.mic_lobe._pulse_opacity = opacity
            self.system_lobe._pulse_opacity = opacity
            self.mic_lobe.update()
            self.system_lobe.update()
            
            if tick_count >= 20:
                # Reset and stop
                self.mic_lobe._pulse_opacity = 1.0
                self.system_lobe._pulse_opacity = 1.0
                self.mic_lobe.update()
                self.system_lobe.update()
                self._pulse_timer.stop()
        
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(_tick)
        self._pulse_timer.start(100)
    
    def _show_error(self, message, is_recoverable=True):
        """Show error indicator with message."""
        self._error_indicator.set_text(message, is_recoverable=is_recoverable)
        self._error_indicator.show()
        # Auto-hide after 8 seconds
        if self._error_hide_timer is not None:
            self._error_hide_timer.stop()
        self._error_hide_timer = QTimer(self)
        self._error_hide_timer.setSingleShot(True)
        self._error_hide_timer.timeout.connect(self._hide_error)
        self._error_hide_timer.start(8000)
    
    def _hide_error(self):
        """Hide error indicator."""
        self._error_indicator.hide()
    
    def _on_error_help_toggled(self, expanded):
        """Handle help panel expansion — cancel/restart auto-hide timer."""
        if self._error_hide_timer is not None:
            self._error_hide_timer.stop()
        if not expanded:
            # Restart auto-hide timer when collapsed
            self._error_hide_timer = QTimer(self)
            self._error_hide_timer.setSingleShot(True)
            self._error_hide_timer.timeout.connect(self._hide_error)
            self._error_hide_timer.start(8000)
    
    def _show_resource_warning(self, message):
        """Show resource warning indicator on the main widget scene."""
        self._warning_indicator.set_text(message)
        self._warning_indicator.show()
        # Auto-hide after 10 seconds
        if self._warning_hide_timer is not None:
            self._warning_hide_timer.stop()
        self._warning_hide_timer = QTimer(self)
        self._warning_hide_timer.setSingleShot(True)
        self._warning_hide_timer.timeout.connect(self._hide_resource_warning)
        self._warning_hide_timer.start(10000)
    
    def _hide_resource_warning(self):
        """Hide resource warning indicator."""
        self._warning_indicator.hide()
    
    def _on_controller_state_change(self, state):
        """Handle controller state changes.

        Routes CC overlay lifecycle:
        - RECORDING: clear, cancel any pending hide, show
        - STOPPING/IDLE/ERROR: start delayed hide (keeps final words)
        """
        if state == ControllerState.RECORDING:
            self.is_recording = True
            self.is_processing = False
            self._visual_state.transition_to(WidgetVisualState.RECORDING)
            self.record_button.set_recording_state(True)
            self._hide_error()
            # Lock lobes during recording
            self.mic_lobe.set_locked(True)
            self.system_lobe.set_locked(True)
            logging.debug("Lobes locked (RECORDING)")
            
            # Show CC overlay for live transcript
            if self._cc_overlay:
                logging.debug("Showing CC overlay for recording")
                self._cc_overlay.clear()
                # Position near widget on first show if never positioned
                if not self._cc_overlay.isVisible():
                    try:
                        offset_x = self.x() - self._cc_overlay.width() - 10
                        offset_y = self.y()
                        # Ensure it stays on screen
                        screen = QApplication.primaryScreen().geometry()
                        if offset_x < screen.left():
                            offset_x = self.x() + self.width() + 10
                        self._cc_overlay.move(offset_x, offset_y)
                        ensure_on_screen(self._cc_overlay)
                    except (TypeError, AttributeError):
                        pass  # Graceful fallback in test/mock environments
                self._cc_overlay.show_panel()

            # Show floating transcript panel when recording starts (legacy)
            # (Removed — CC overlay is the only live transcript surface)
            
        elif state == ControllerState.STARTING:
            # Lock lobes during startup phase too
            self.mic_lobe.set_locked(True)
            self.system_lobe.set_locked(True)
            logging.debug("Lobes locked (STARTING)")
            
        elif state == ControllerState.STOPPING:
            self.is_recording = False
            self.is_processing = True
            self._visual_state.transition_to(WidgetVisualState.PROCESSING)
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(True)
            # CC overlay: start delayed fade-out (keeps final words visible)
            if self._cc_overlay:
                self._cc_overlay.start_delayed_hide()
                logging.debug("CC overlay: delayed hide scheduled (STOPPING)")
            
        elif state == ControllerState.IDLE:
            self.is_recording = False
            self.is_processing = False
            self._visual_state.transition_to(WidgetVisualState.IDLE)
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(False)
            # Unlock lobes when idle
            self.mic_lobe.set_locked(False)
            self.system_lobe.set_locked(False)
            logging.debug("Lobes unlocked (IDLE)")
            # CC overlay: start delayed fade-out if still visible
            if self._cc_overlay and self._cc_overlay.isVisible():
                self._cc_overlay.start_delayed_hide()
                logging.debug("CC overlay: delayed hide scheduled (IDLE)")
            
        elif state == ControllerState.ERROR:
            self.is_recording = False
            self.is_processing = False
            self._visual_state.transition_to(WidgetVisualState.IDLE)
            self.record_button.set_recording_state(False)
            self.record_button.set_processing_state(False)
            # Unlock lobes on error so user can adjust
            self.mic_lobe.set_locked(False)
            self.system_lobe.set_locked(False)
            logging.debug("Lobes unlocked (ERROR)")
            # CC overlay: start delayed fade-out on error
            if self._cc_overlay and self._cc_overlay.isVisible():
                self._cc_overlay.start_delayed_hide()
                logging.debug("CC overlay: delayed hide scheduled (ERROR)")
        
        # Forward state to tray icon manager
        if self._tray_manager is not None:
            self._tray_manager.update_recording_state(state)
    
    def _on_controller_error(self, error):
        """Handle controller errors."""
        self._show_error(error.message, is_recoverable=error.is_recoverable)
        logging.error("Recording error: %s (recoverable: %s)", error.message, error.is_recoverable)
    
    def _on_phrase_result(self, result: SegmentResult):
        """Handle segment result from accumulating transcription processor.

        Thread-safe: emits signals which automatically queue to main thread.

        Args:
            result: SegmentResult with text, confidence, and completion status
        """
        phrase_start = getattr(result, 'phrase_start', False)

        logging.debug("Segment: '%s' [conf: %d%%, final: %s, phrase_start: %s]",
                       result.text[:40], result.confidence, result.is_final, phrase_start)

        if self._cc_overlay:
            # Emit signal (thread-safe, automatically queues to main thread)
            self._cc_overlay.segment_ready.emit(
                result.text,
                result.confidence,
                result.segment_index,
                result.is_final,
                phrase_start
            )
    
    def _on_recording_complete(self, wav_path, transcript_path):
        """Handle recording completion."""
        self.is_processing = False
        self.record_button.set_processing_state(False)
        logging.info("Recording saved to: %s", wav_path)
        if transcript_path:
            logging.info("Transcript saved to: %s", transcript_path)

    def _on_post_process_complete(self, job_id, transcript_path):
        """Handle post-processing completion.

        Updates WER display in settings panel.  CC overlay lifecycle
        is handled by STOPPING/IDLE delayed hide.
        """
        logging.info("Post-processing complete! Job: %s, transcript: %s", job_id, transcript_path)

        # Update Performance tab WER display (Settings panel)
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

        # Also update CC overlay speaker names
        if self._cc_overlay:
            speaker_names = self._controller.get_speaker_names()
            if speaker_names:
                self._cc_overlay.set_speaker_names(speaker_names)

    def toggle_transcript_panel(self):
        """Toggle CC overlay visibility via the transcript lobe.

        The transcript lobe controls the compact CC overlay (live captions).
        The legacy FloatingTranscriptPanel remains available but the primary
        user-facing toggle operates on the CC overlay.
        """
        if self._cc_overlay:
            if self._cc_overlay.isVisible():
                self._cc_overlay.hide_panel()
            else:
                ensure_on_screen(self._cc_overlay)
                self._cc_overlay.show_panel()
    
    def _toggle_settings_panel(self):
        """Toggle floating settings panel visibility.

        Opening: positions the settings panel with a default offset from
        the widget. Closing: simply hides the panel. Panels are
        independent free-floating windows — no docking or z-order sync.
        """
        if self._floating_settings_panel:
            if self._floating_settings_panel.isVisible():
                self._floating_settings_panel.hide_panel()
            else:
                # Position with offset to the right of the widget
                offset_x = self.x() + self.width() + 10
                offset_y = self.y()
                self._floating_settings_panel.move(offset_x, offset_y)
                ensure_on_screen(self._floating_settings_panel)
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
            self._pulse_lobes()
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
        """Show context menu with recording toggle, settings, and exit actions."""
        p = current_palette()
        menu = QMenu(self)
        menu.setStyleSheet(context_menu_css(p, accent_color='#4CAF50'))

        # Recording toggle
        toggle_text = "Stop Recording" if self.is_recording else "Start Recording"
        toggle_action = menu.addAction(toggle_text)
        toggle_action.triggered.connect(self.toggle_recording)

        menu.addSeparator()

        # Settings
        settings_action = menu.addAction("Settings")
        settings_action.triggered.connect(self._toggle_settings_panel)

        menu.addSeparator()

        # Exit
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self._exit_application)

        menu.exec(self.mapToGlobal(position))
    
    def _exit_application(self):
        """Exit the application cleanly (full quit, even with tray)."""
        self._save_position()
        # Hide all floating panels first
        if self._floating_settings_panel:
            self._floating_settings_panel.hide()
        if self._cc_overlay:
            self._cc_overlay.hide()
        if self._tray_manager is not None:
            self._tray_manager.hide()
        QApplication.quit()
    
    def _save_position(self):
        """Save widget position to config."""
        try:
            set_config('ui.widget_position', (self.x(), self.y()))
            save_config()
            logging.debug("Saved widget position: (%d, %d)", self.x(), self.y())
        except Exception as e:
            logging.warning("Failed to save position: %s", e)
    
    def showEvent(self, event):
        """Reset opacity to match current state when widget reappears."""
        super().showEvent(event)
        # Immediately set correct opacity for current state without animation
        if self._visual_state.current == WidgetVisualState.IDLE:
            self.setWindowOpacity(self._IDLE_OPACITY)
        else:
            self.setWindowOpacity(self._ACTIVE_OPACITY)
        logging.debug("showEvent: opacity reset for state=%s",
                      self._visual_state.current.name)

    def hideEvent(self, event):
        """Reset opacity to 1.0 on hide so reappear isn't stale at low opacity."""
        super().hideEvent(event)
        self.setWindowOpacity(1.0)

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
    
    _ORBITS = [
        {"radius_frac": 0.55, "speed_mult": 0.6, "size": 6, "alpha": 120, "offset": 0.0},
        {"radius_frac": 0.70, "speed_mult": 1.0, "size": 8, "alpha": 180, "offset": 2.1},
        {"radius_frac": 0.70, "speed_mult": 1.0, "size": 8, "alpha": 180, "offset": 4.2},
        {"radius_frac": 0.85, "speed_mult": 1.4, "size": 5, "alpha": 100, "offset": 1.0},
    ]

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
        """Paint recording state - smooth sinusoidal glowing red pulse."""
        # Sinusoidal intensity: smooth 0.0 → 1.0 → 0.0 oscillation
        pulse = 0.5 + 0.5 * _math.sin(self.pulse_phase)
        
        # Outer glow rings — 3 rings with alpha/radius spread modulated by pulse
        for i in range(3, 0, -1):
            alpha = int(30 + 60 * pulse / i)  # 30–90 based on pulse
            radius = 3 + i * (2 + int(pulse * 2))  # 3–8px spread
            painter.setBrush(QBrush(QColor(255, 50, 50, alpha)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.adjusted(-radius, -radius, radius, radius))
        
        # Main button — subtle breathing effect
        alpha = 180 + int(75 * pulse)
        painter.setBrush(QBrush(QColor(255, 50, 50, alpha)))
        painter.setPen(QPen(QColor(255, 100, 100, 255), 2))
        painter.drawEllipse(rect)
    
    def _paint_swirl(self, painter, rect):
        """Paint processing state - orbiting dots at varied radii and speeds."""
        # Background
        painter.setBrush(QBrush(QColor(100, 100, 255, 180)))
        painter.setPen(QPen(QColor(150, 150, 255, 255), 2))
        painter.drawEllipse(rect)
        
        # Multi-radius orbiting dots
        center = rect.center()
        max_radius = rect.width() / 2 - 5
        
        for orbit in self._ORBITS:
            angle = self.swirl_phase * orbit["speed_mult"] + orbit["offset"]
            radius = max_radius * orbit["radius_frac"]
            x = center.x() + radius * _math.cos(angle)
            y = center.y() + radius * _math.sin(angle)
            
            size = orbit["size"]
            painter.setBrush(QBrush(QColor(255, 255, 255, orbit["alpha"])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(x) - size // 2, int(y) - size // 2, size, size)
    
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
        self._is_locked: bool = False
        self._is_unavailable: bool = False
        self._pulse_opacity: float = 1.0
        
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
        """Paint the lobe with hover glow effect.

        Rendering priority: unavailable > locked > normal (active/inactive).
        When pulsing, all alphas are multiplied by ``_pulse_opacity``.
        """
        rect = self.rect()
        
        # Hover glow (subtle outer ring) — suppressed for unavailable
        if self._hovered and not self._is_unavailable:
            for i in range(2, 0, -1):
                glow_alpha = 40 if self.is_active else 25
                painter.setBrush(QBrush(QColor(255, 255, 255, glow_alpha // i)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(rect.adjusted(-i * 3, -i * 3, i * 3, i * 3))
        
        # --- State-dependent rendering (priority: unavailable > locked > normal) ---
        if self._is_unavailable:
            # Dim reddish tint with pulse scaling
            fill_alpha = int(100 * self._pulse_opacity)
            border_alpha = int(140 * self._pulse_opacity)
            painter.setBrush(QBrush(QColor(120, 80, 80, fill_alpha)))
            painter.setPen(QPen(QColor(160, 100, 100, border_alpha), 2))
            painter.drawEllipse(rect)
            # Diagonal line through icon area
            painter.setPen(QPen(QColor(255, 80, 80, int(180 * self._pulse_opacity)), 2))
            painter.drawLine(int(rect.left() + 10), int(rect.top() + 10),
                             int(rect.right() - 10), int(rect.bottom() - 10))
        elif self._is_locked:
            # Locked — dimmed via reduced opacity
            painter.save()
            painter.setOpacity(0.4)
            if self.is_active:
                painter.setBrush(QBrush(QColor(100, 200, 100, 200)))
                painter.setPen(QPen(QColor(150, 255, 150, 255), 2))
            else:
                painter.setBrush(QBrush(QColor(100, 100, 100, 150)))
                painter.setPen(QPen(QColor(150, 150, 150, 200), 2))
            painter.drawEllipse(rect)
            painter.restore()
        elif self.is_active:
            # Active state - bright, boosted on hover
            fill_alpha = int((230 if self._hovered else 200) * self._pulse_opacity)
            border_alpha = int(255 * self._pulse_opacity)
            painter.setBrush(QBrush(QColor(100, 200, 100, fill_alpha)))
            painter.setPen(QPen(QColor(150, 255, 150, border_alpha), 2))
            painter.drawEllipse(rect)
        else:
            # Inactive state - dim, brightened on hover
            fill_alpha = int((190 if self._hovered else 150) * self._pulse_opacity)
            border_alpha = int((240 if self._hovered else 200) * self._pulse_opacity)
            fill_val = 140 if self._hovered else 100
            border_val = 200 if self._hovered else 150
            painter.setBrush(QBrush(QColor(fill_val, fill_val, fill_val, fill_alpha)))
            painter.setPen(QPen(QColor(border_val, border_val, border_val, border_alpha), 2))
            painter.drawEllipse(rect)
        
        # Draw icon — suppressed for unavailable (diagonal line replaces it)
        if not self._is_unavailable:
            icon_alpha = int(255 * self._pulse_opacity) if not self._is_locked else int(255 * 0.4)
            painter.setPen(QPen(QColor(255, 255, 255, icon_alpha), 2))
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
            # Prevent toggle when locked or unavailable
            if self._is_locked or self._is_unavailable:
                event.accept()
                return
            if not self.parent_widget.is_dragging and not self.parent_widget._click_consumed:
                self.is_active = not self.is_active
                self.update()
                self.parent_widget._on_lobe_toggled()
            event.accept()

    def set_locked(self, locked: bool) -> None:
        """Set the locked state (dimmed, non-interactive during recording)."""
        self._is_locked = locked
        self.setCursor(
            Qt.CursorShape.ForbiddenCursor if locked
            else Qt.CursorShape.PointingHandCursor
        )
        self.update()

    def set_unavailable(self, unavailable: bool) -> None:
        """Set the unavailable state (reddish tint, no loopback device)."""
        self._is_unavailable = unavailable
        self.setCursor(
            Qt.CursorShape.ForbiddenCursor if unavailable
            else Qt.CursorShape.PointingHandCursor
        )
        self.update()


class SettingsLobeItem(QGraphicsEllipseItem):
    """Settings lobe for accessing configuration."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 22, 22)
        self.parent_widget = parent_widget
        
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setTransformOriginPoint(11, 11)  # Center of 22×22
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
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5))
        center = rect.center()
        painter.drawEllipse(int(center.x() - 4), int(center.y() - 4), 8, 8)
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


class TranscriptLobeItem(QGraphicsEllipseItem):
    """Transcript lobe for toggling the transcript panel visibility."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 22, 22)
        self.parent_widget = parent_widget
        
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setTransformOriginPoint(11, 11)  # Center of 22×22
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
        """Paint transcript lobe with document/text icon and hover glow."""
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
        
        # Draw document/text icon (rectangle with lines)
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5))
        center = rect.center()
        # Document body
        doc_left = int(center.x() - 4)
        doc_top = int(center.y() - 5)
        doc_width = 8
        doc_height = 10
        painter.drawRect(doc_left, doc_top, doc_width, doc_height)
        # Text lines inside document
        line_left = doc_left + 1
        line_right = doc_left + doc_width - 1
        painter.drawLine(line_left, int(center.y() - 2), line_right, int(center.y() - 2))
        painter.drawLine(line_left, int(center.y()), line_right, int(center.y()))
        painter.drawLine(line_left, int(center.y() + 2), line_right, int(center.y() + 2))
    
    def mousePressEvent(self, event):
        """Accept press to get release event — action fires on release."""
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()

    def mouseReleaseEvent(self, event):
        """Toggle transcript panel on release, only if this wasn't a drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.parent_widget.is_dragging and not self.parent_widget._click_consumed:
                self.parent_widget.toggle_transcript_panel()
            event.accept()


def get_error_help_text(message: str) -> Optional[str]:
    """Return context-specific help text for known error patterns, or None."""
    patterns = [
        (r'no.*(source|mic|system|audio).*select',
         "Click a lobe on the widget to enable microphone or system audio, then try recording again."),
        (r'transcription.*(fail|error|unable)',
         "Try selecting a smaller Whisper model (Tiny or Base) in Settings → Settings tab. Smaller models use less memory."),
        (r'(device|microphone|speaker).*not.*(found|available|detect)',
         "Check that your audio device is connected and recognized by the system. Try unplugging and reconnecting."),
        (r'(memory|ram|resource).*low|out of memory',
         "Close other applications to free memory. Try a smaller Whisper model in Settings."),
    ]
    lower = message.lower()
    for pattern, help_text in patterns:
        if re.search(pattern, lower):
            return help_text
    return None


class ErrorIndicatorItem(QGraphicsRectItem):
    """Error indicator displayed below the record button."""
    
    def __init__(self, parent_widget):
        super().__init__(0, 0, 190, 14)
        self.parent_widget = parent_widget
        self._text = ""
        self._visible = False
        self._color_mode = "error"  # "error" (red) or "warning" (orange)
        self._help_text: Optional[str] = None
        self._help_expanded: bool = False
        self._base_height: int = 14

    def set_color_mode(self, mode: str):
        """Set color mode: 'error' for red, 'warning' for orange."""
        self._color_mode = mode
    
    def set_text(self, text, is_recoverable=True):
        """Set the error message text. Only show help for recoverable errors."""
        self._text = text
        if is_recoverable:
            self._help_text = get_error_help_text(text)
        else:
            self._help_text = None
        self._help_expanded = False
        self._recalc_rect()
        self.update()
    
    def _recalc_rect(self):
        """Recalculate rect height based on expansion state."""
        w = 190
        h = self._base_height
        if self._help_expanded and self._help_text:
            font = QFont("Arial", 7)
            from PyQt6.QtGui import QFontMetrics
            from PyQt6.QtCore import QRect
            fm = QFontMetrics(font)
            # Wrap help text within the available width (190 - 8 margins)
            text_width = w - 8
            help_rect = fm.boundingRect(QRect(0, 0, text_width, 100),
                                        Qt.TextFlag.TextWordWrap, self._help_text)
            h += help_rect.height() + 4  # 4px gap between error text and help
        self.setRect(0, 0, w, h)
    
    def _help_button_rect(self):
        """Return the QRectF for the '?' help button area."""
        rect = self.rect()
        return QRectF(rect.width() - 16, 1, 14, 12)
    
    def show(self):
        """Show the error indicator."""
        self._visible = True
        self.update()
    
    def hide(self):
        """Hide the error indicator."""
        self._visible = False
        self._help_expanded = False
        self._recalc_rect()
        self.update()
    
    def mousePressEvent(self, event):
        """Handle clicks — toggle help panel if '?' button clicked."""
        if self._help_text is not None:
            btn_rect = self._help_button_rect()
            pos = event.pos()
            if btn_rect.contains(pos):
                self._help_expanded = not self._help_expanded
                self._recalc_rect()
                self.update()
                # Notify parent widget about expansion change
                if hasattr(self.parent_widget, '_on_error_help_toggled'):
                    self.parent_widget._on_error_help_toggled(self._help_expanded)
                event.accept()
                return
        try:
            super().mousePressEvent(event)
        except TypeError:
            pass
    
    def paint(self, painter, option, widget=None):
        """Paint error indicator."""
        if not self._visible:
            return
        
        rect = self.rect()
        
        if self._color_mode == "warning":
            painter.setBrush(QBrush(QColor(255, 152, 0, 200)))
            painter.setPen(QPen(QColor(255, 183, 77, 255), 1))
        else:
            painter.setBrush(QBrush(QColor(255, 50, 50, 180)))
            painter.setPen(QPen(QColor(255, 100, 100, 255), 1))
        painter.drawRoundedRect(rect, 3, 3)
        
        # Error text
        if self._text:
            painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
            font = QFont("Arial", 8)
            font.setBold(True)
            painter.setFont(font)
            
            text_rect = rect.adjusted(2, 1, -16 if self._help_text is not None else -2, -1)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._text)
        
        # Help '?' button
        if self._help_text is not None:
            btn = self._help_button_rect()
            painter.setBrush(QBrush(QColor(0, 188, 188, 220)))  # teal
            painter.setPen(QPen(QColor(0, 220, 220, 255), 1))
            painter.drawRoundedRect(btn, 3, 3)
            painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
            help_font = QFont("Arial", 8)
            help_font.setBold(True)
            painter.setFont(help_font)
            painter.drawText(btn, Qt.AlignmentFlag.AlignCenter, "?")
        
        # Expanded help text
        if self._help_expanded and self._help_text:
            painter.setPen(QPen(QColor(255, 255, 200, 230), 1))
            help_font = QFont("Arial", 7)
            help_font.setBold(False)
            painter.setFont(help_font)
            help_rect = QRectF(4, self._base_height + 2, rect.width() - 8, rect.height() - self._base_height - 2)
            painter.drawText(help_rect, Qt.TextFlag.TextWordWrap, self._help_text)
