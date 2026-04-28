"""
Floating transcript panel - separate window that docks to the main widget.

This solves the clipping issue by making the panel a separate QWidget
that floats outside the main widget bounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLabel, QFrame, QHBoxLayout, QPushButton,
    QInputDialog, QApplication, QTabWidget, QListWidget, QListWidgetItem,
    QSplitter, QTextBrowser, QProgressBar, QComboBox, QMenu, QMessageBox,
    QDialog, QDialogButtonBox, QSizePolicy, QSizeGrip, QStackedWidget,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QPainter, QPen
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import time

from meetandread.transcription.confidence import get_confidence_color, get_confidence_legend
from meetandread.hardware.detector import HardwareDetector
from meetandread.hardware.recommender import ModelRecommender
from meetandread.performance.monitor import ResourceMonitor, ResourceSnapshot
from meetandread.performance.benchmark import BenchmarkRunner, BenchmarkResult
from meetandread.performance.wer import calculate_wer
from meetandread.widgets.theme import (
    current_palette, DARK_PALETTE,
    panel_base_css, glass_panel_css, title_css, header_button_css, tab_widget_css,
    text_area_css, status_label_css, splitter_css, list_widget_css,
    detail_header_css, action_button_css, context_menu_css, dialog_css,
    badge_css, resize_grip_css, legend_overlay_css, info_label_css,
    progress_bar_css, separator_css, combo_box_css,
    aetheric_settings_shell_css, aetheric_sidebar_css, aetheric_nav_button_css,
    aetheric_dock_bay_css, aetheric_placeholder_css, aetheric_combo_box_css,
    AETHERIC_RED, AETHERIC_NAV_INACTIVE_TEXT,
)

import logging

logger = logging.getLogger(__name__)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe embedding in innerHTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# Speaker color palette — deterministic colors for up to 8 speakers
# ---------------------------------------------------------------------------
SPEAKER_COLORS: Dict[str, str] = {
    "SPK_0": "#4FC3F7",  # Light blue
    "SPK_1": "#FF8A65",  # Orange
    "SPK_2": "#AED581",  # Light green
    "SPK_3": "#CE93D8",  # Purple
    "SPK_4": "#FFD54F",  # Amber
    "SPK_5": "#F48FB1",  # Pink
    "SPK_6": "#4DD0E1",  # Cyan
    "SPK_7": "#FFB74D",  # Deep orange
}
_DEFAULT_SPEAKER_COLOR = "#90A4AE"  # Blue grey fallback


def speaker_color(label: str) -> str:
    """Return a deterministic color hex string for a speaker label."""
    return SPEAKER_COLORS.get(label, _DEFAULT_SPEAKER_COLOR)


@dataclass
class Phrase:
    """A phrase (line) of transcript with its segments."""
    segments: List[str]  # Text of each segment
    confidences: List[int]  # Confidence of each segment
    is_final: bool  # True if phrase is complete
    speaker_id: Optional[str] = None  # Speaker label for this phrase


class FloatingTranscriptPanel(QWidget):
    """
    Floating transcript panel that appears outside the main widget.
    
    Features:
    - Separate window (not clipped by main widget bounds)
    - Docks to main widget position
    - Shows transcript with confidence-based coloring
    - Auto-scrolls to show latest text
    - Can be manually toggled
    """
    
    # Signals
    closed = pyqtSignal()  # Emitted when user closes panel
    segment_ready = pyqtSignal(str, int, int, bool, bool)  # text, confidence, segment_index, is_final, phrase_start
    speaker_name_pinned = pyqtSignal(str, str)  # raw_speaker_label, user_chosen_name

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Speaker label display mapping: raw_label -> display_name
        # e.g. {"spk0": "Alice", "spk1": "SPK_1"}
        self._speaker_names: Dict[str, str] = {}
        # Raw speaker labels that have been pinned by the user
        self._pinned_speakers: set = set()
        
        # Track whether panel has been positioned at least once
        self._has_been_docked = False
        
        # Window settings (FloatingTranscriptPanel)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Don't show in taskbar
        )
        
        # Glass pair: translucent background matching the widget's glass aesthetic
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # Size — resizable with min/max bounds
        self.setMinimumSize(350, 300)
        self.setMaximumSize(800, 900)
        
        # Style — applied via _apply_theme() at end of __init__
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header with title and close button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(5)
        
        # Title bar (clickable for dragging)
        self._title_label = QLabel("Live Transcript")
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        
        # Legend toggle button (?)
        self._legend_btn = QPushButton("?")
        self._legend_btn.setFixedSize(24, 24)
        self._legend_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._legend_btn.setToolTip("Confidence legend")
        self._legend_btn.setCheckable(True)
        self._legend_btn.clicked.connect(self._toggle_legend)
        header_layout.addWidget(self._legend_btn)
        
        # Close button
        self._close_btn = QPushButton("×")
        self._close_btn.setFixedSize(24, 24)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setToolTip("Close panel")
        self._close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(self._close_btn)
        
        layout.addLayout(header_layout)
        
        # ------------------------------------------------------------------
        # Tab widget — Live and History tabs
        # ------------------------------------------------------------------
        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)

        # ------------------------------------------------------------------
        # Live tab — existing transcript display
        # ------------------------------------------------------------------
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(2)

        # Text edit for transcript
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        # Handle anchor clicks on speaker labels (signal only on QTextBrowser)
        self.text_edit.setMouseTracking(True)
        if hasattr(self.text_edit, "anchorClicked"):
            self.text_edit.anchorClicked.connect(self._on_anchor_clicked)
        live_layout.addWidget(self.text_edit)

        # Status label
        self.status_label = QLabel("Ready")
        live_layout.addWidget(self.status_label)

        self._tab_widget.addTab(live_tab, "Live")

        # ------------------------------------------------------------------
        # History tab — recording list and transcript viewer
        # ------------------------------------------------------------------
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: recording list
        self._history_list = QListWidget()
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        # Enable context menu on history items
        self._history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        self._splitter.addWidget(self._history_list)

        # Bottom section: detail header bar + transcript viewer
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        # Detail header bar with Delete button (hidden until selection)
        self._detail_header = QFrame()
        detail_header_layout = QHBoxLayout(self._detail_header)
        detail_header_layout.setContentsMargins(6, 2, 6, 2)
        detail_header_layout.setSpacing(4)

        detail_header_layout.addStretch()

        self._scrub_btn = QPushButton("🔄 Scrub")
        self._scrub_btn.setFixedHeight(26)
        self._scrub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scrub_btn.setToolTip("Re-transcribe with a different model")
        self._scrub_btn.clicked.connect(self._on_scrub_clicked)
        detail_header_layout.addWidget(self._scrub_btn)

        self._delete_btn = QPushButton("🗑 Delete")
        self._delete_btn.setFixedHeight(26)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Delete this recording")
        self._delete_btn.clicked.connect(self._on_delete_btn_clicked)
        detail_header_layout.addWidget(self._delete_btn)

        self._detail_header.hide()
        viewer_layout.addWidget(self._detail_header)

        # Transcript viewer (read-only, supports anchor clicks)
        self._history_viewer = QTextBrowser()
        self._history_viewer.setReadOnly(True)
        self._history_viewer.setFrameShape(QFrame.Shape.NoFrame)
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._history_viewer.setOpenExternalLinks(False)
        self._history_viewer.setOpenLinks(False)
        self._history_viewer.anchorClicked.connect(self._on_history_anchor_clicked)
        viewer_layout.addWidget(self._history_viewer)

        self._splitter.addWidget(viewer_container)

        # 40% list / 60% viewer
        self._splitter.setSizes([160, 240])

        history_layout.addWidget(self._splitter)
        self._tab_widget.addTab(history_tab, "History")

        # Connect tab change to refresh history when switching to it
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Track the currently-viewed history transcript path (for rename)
        self._current_history_md_path: Optional[Path] = None

        # Scrub state
        self._scrub_runner: Optional[object] = None  # ScrubRunner instance
        self._scrub_model_size: Optional[str] = None  # model being scrubbed
        self._scrub_sidecar_path: Optional[str] = None  # expected sidecar path
        self._scrub_original_html: Optional[str] = None  # original transcript HTML
        self._is_scrubbing: bool = False  # True while scrub is in progress
        self._is_comparison_mode: bool = False  # True when showing side-by-side
        
        # Dragging
        self._dragging = False
        self._drag_pos = None
        
        # Track phrases (each phrase is a line)
        self.phrases: List[Phrase] = []
        self.current_phrase_idx = -1
        
        # Auto-scroll timer
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self._scroll_to_bottom)
        
        # Auto-scroll pause mechanism
        self._auto_scroll_paused = False
        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.timeout.connect(self._resume_auto_scroll)
        self._last_scroll_value = 0
        self._is_at_bottom = True
        
        # Pending content count for badge when auto-scroll is paused
        self._pending_content_count: int = 0
        
        # Connect to scrollbar value changed signal to detect manual scroll
        self.text_edit.verticalScrollBar().valueChanged.connect(self._on_scroll_value_changed)
        
        # Confidence legend overlay (initially hidden)
        self._create_legend_overlay()
        
        # New-content badge (initially hidden)
        self._create_new_content_badge()
        
        # Recording duration tracking
        self._recording_start_time: Optional[float] = None
        self._duration_timer = QTimer(self)
        self._duration_timer.setInterval(1000)  # 1-second tick
        self._duration_timer.timeout.connect(self._update_duration)
        
        # Empty state tracking
        self._has_content: bool = False
        
        # Glass pair opacity — matches the widget's glass aesthetic
        # 0.87 = translucent idle (desktop visible behind), 1.0 = active/recording
        self._glass_idle_opacity = 0.87
        self._glass_active_opacity = 1.0
        self._is_glass_active = False
        self.setWindowOpacity(self._glass_idle_opacity)

        # Apply initial theme to all widgets
        self._apply_theme()

        # Connect to desktop theme changes for live re-theming
        try:
            from PyQt6.QtGui import QGuiApplication
            hints = QGuiApplication.styleHints()
            if hints is not None:
                hints.colorSchemeChanged.connect(lambda: self._apply_theme())
        except (ImportError, RuntimeError):
            pass

        self._show_empty_state()
    
    # ------------------------------------------------------------------
    # Confidence legend overlay
    # ------------------------------------------------------------------

    def _create_legend_overlay(self) -> None:
        """Build the confidence legend overlay positioned over the text edit area."""
        self._legend_overlay = QFrame(self.text_edit)
        self._legend_overlay.setFixedSize(220, 140)

        # Layout
        overlay_layout = QVBoxLayout(self._legend_overlay)
        overlay_layout.setContentsMargins(10, 8, 10, 8)
        overlay_layout.setSpacing(4)

        # Title
        self._legend_title = QLabel("Confidence Levels")
        overlay_layout.addWidget(self._legend_title)

        # Separator
        self._legend_sep = QFrame()
        self._legend_sep.setFrameShape(QFrame.Shape.HLine)
        self._legend_sep.setFixedHeight(1)
        overlay_layout.addWidget(self._legend_sep)

        # Legend rows from canonical source
        self._legend_range_labels: list = []
        self._legend_desc_labels: list = []
        for item in get_confidence_legend():
            row = QHBoxLayout()
            row.setSpacing(6)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {item.color}; border-radius: 3px; border: none;"
            )
            row.addWidget(swatch)

            # Range text
            range_label = QLabel(item.range_str)
            range_label.setFixedWidth(50)
            row.addWidget(range_label)
            self._legend_range_labels.append(range_label)

            # Description
            desc_label = QLabel(item.description)
            row.addWidget(desc_label)
            self._legend_desc_labels.append(desc_label)

            row.addStretch()
            overlay_layout.addLayout(row)

        # Position bottom-right of text_edit
        self._position_legend_overlay()
        self._legend_overlay.hide()

    # ------------------------------------------------------------------
    # New-content badge (auto-scroll pause indicator)
    # ------------------------------------------------------------------

    def _create_new_content_badge(self) -> None:
        """Build the '↓ N new' badge that appears when auto-scroll is paused."""
        self._new_content_badge = QPushButton("↓ 0 new", self.text_edit)
        self._new_content_badge.setFixedSize(120, 32)
        self._new_content_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_content_badge.clicked.connect(self._on_badge_clicked)
        self._position_new_content_badge()
        self._new_content_badge.hide()

        # Resize grip — direct child of panel (not in layout) so it stays at bottom-right
        self._resize_grip = QSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resize_grip.show()

    def _position_new_content_badge(self) -> None:
        """Position the badge at bottom-center of the text edit."""
        if not hasattr(self, '_new_content_badge'):
            return
        te = self.text_edit
        badge = self._new_content_badge
        x = (te.width() - badge.width()) // 2
        y = te.height() - badge.height() - 8
        badge.move(max(x, 0), max(y, 0))

    def _on_badge_clicked(self) -> None:
        """Handle badge click: resume auto-scroll and hide badge."""
        self._auto_scroll_paused = False
        self._pause_timer.stop()
        self._pending_content_count = 0
        self._new_content_badge.hide()
        self.status_label.setText("Recording...")
        self._scroll_to_bottom()

    def _position_legend_overlay(self) -> None:
        """Position the legend overlay in the bottom-right corner of the text edit."""
        if not hasattr(self, '_legend_overlay'):
            return
        te = self.text_edit
        x = te.width() - self._legend_overlay.width() - 8
        y = te.height() - self._legend_overlay.height() - 8
        self._legend_overlay.move(max(x, 0), max(y, 0))

    def _toggle_legend(self) -> None:
        """Toggle the confidence legend overlay visibility."""
        visible = self._legend_btn.isChecked()
        if visible:
            self._position_legend_overlay()
        self._legend_overlay.setVisible(visible)

    def resizeEvent(self, event) -> None:
        """Reposition overlays and resize grip on resize."""
        if hasattr(self, '_legend_overlay') and self._legend_overlay.isVisible():
            self._position_legend_overlay()
        if hasattr(self, '_new_content_badge') and self._new_content_badge.isVisible():
            self._position_new_content_badge()
        if hasattr(self, '_resize_grip'):
            self._resize_grip.move(
                self.width() - self._resize_grip.width(),
                self.height() - self._resize_grip.height(),
            )
        super().resizeEvent(event)
    
    def dock_to_widget(self, widget, position: str = "right") -> None:
        """Position panel next to a widget.
        
        Args:
            widget: The main widget to dock to
            position: "left", "right", "top", "bottom"
        """
        # Get widget position in screen coordinates
        widget_pos = widget.mapToGlobal(widget.rect().topLeft())
        widget_rect = widget.geometry()
        
        # Calculate panel position
        if position == "left":
            x = widget_pos.x() - self.width() - 10
            y = widget_pos.y()
        elif position == "right":
            x = widget_pos.x() + widget_rect.width() + 10
            y = widget_pos.y()
        elif position == "top":
            x = widget_pos.x()
            y = widget_pos.y() - self.height() - 10
        else:  # bottom
            x = widget_pos.x()
            y = widget_pos.y() + widget_rect.height() + 10
        
        self.move(x, y)
        self._has_been_docked = True
    
    def show_panel(self) -> None:
        """Show the panel with a 150ms fade-in and start auto-scroll."""
        self._set_glass_active(True)
        self._start_fade_in()
        self.scroll_timer.start(100)  # Scroll check every 100ms
        self._recording_start_time = time.time()
        self._duration_timer.start()
        self._update_duration()  # Show "Recording · 00:00" immediately
        # Reset badge state on panel show
        self._pending_content_count = 0
        if hasattr(self, '_new_content_badge'):
            self._new_content_badge.hide()
    
    def hide_panel(self) -> None:
        """Hide the panel with a 150ms fade-out."""
        self._set_glass_active(False)
        self.scroll_timer.stop()
        self._duration_timer.stop()
        self._recording_start_time = None
        self._start_fade_out()

    def _set_glass_active(self, active: bool) -> None:
        """Transition glass opacity between idle (0.87) and active (1.0).

        Args:
            active: True for recording/active state, False for idle.
        """
        self._is_glass_active = active
        target = self._glass_active_opacity if active else self._glass_idle_opacity
        self.setWindowOpacity(target)

    def _update_duration(self) -> None:
        """Update the status label with elapsed recording duration (mm:ss)."""
        if self._recording_start_time is not None:
            elapsed = int(time.time() - self._recording_start_time)
            mins = f"{elapsed // 60:02d}"
            secs = f"{elapsed % 60:02d}"
            self.status_label.setText(f"Recording · {mins}:{secs}")

    def _show_empty_state(self) -> None:
        """Show a friendly placeholder in the transcript area when no content exists."""
        if not self._has_content:
            p = current_palette()
            self.text_edit.setHtml(
                f'<div style="color: {p.text_tertiary}; text-align: center; margin-top: 80px;">'
                'Transcription will appear here...'
                '</div>'
            )

    # ------------------------------------------------------------------
    # Adaptive theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply theme-aware stylesheets to all panel widgets.

        Idempotent and cheap — just re-sets stylesheets from the current
        palette.  Called once at end of __init__ and on desktop theme change.
        """
        p = current_palette()
        self._current_palette = p

        # Panel base
        self.setStyleSheet(glass_panel_css(p, "FloatingTranscriptPanel"))

        # Header widgets
        self._title_label.setStyleSheet(title_css(p))
        self._legend_btn.setStyleSheet(header_button_css(p, "legend"))
        self._close_btn.setStyleSheet(header_button_css(p, "close"))

        # Tabs
        self._tab_widget.setStyleSheet(tab_widget_css(p))

        # Live tab — text area and status
        self.text_edit.setStyleSheet(text_area_css(p))
        self.status_label.setStyleSheet(status_label_css(p))

        # History tab — splitter, list, detail header, buttons, viewer
        self._splitter.setStyleSheet(splitter_css(p))
        self._history_list.setStyleSheet(list_widget_css(p))
        self._detail_header.setStyleSheet(detail_header_css(p))
        self._scrub_btn.setStyleSheet(action_button_css(p, "scrub"))
        self._delete_btn.setStyleSheet(action_button_css(p, "delete"))
        self._history_viewer.setStyleSheet(text_area_css(p))

        # Legend overlay
        legend_styles = legend_overlay_css(p)
        self._legend_overlay.setStyleSheet(legend_styles["overlay"])
        if hasattr(self, "_legend_title"):
            self._legend_title.setStyleSheet(legend_styles["title"])
        if hasattr(self, "_legend_sep"):
            self._legend_sep.setStyleSheet(legend_styles["separator"])
        for lbl in getattr(self, "_legend_range_labels", []):
            lbl.setStyleSheet(legend_styles["range_label"])
        for lbl in getattr(self, "_legend_desc_labels", []):
            lbl.setStyleSheet(legend_styles["desc_label"])

        # Badge
        self._new_content_badge.setStyleSheet(badge_css(p))

        # Resize grip
        self._resize_grip.setStyleSheet(resize_grip_css(p))

        # Re-render empty state with updated text colour
        if not self._has_content:
            self._show_empty_state()

        scheme_name = "dark" if p is DARK_PALETTE else "light"
        logger.info("Applied %s theme to FloatingTranscriptPanel", scheme_name)

    # ------------------------------------------------------------------
    # Fade transition helpers
    # ------------------------------------------------------------------

    _FADE_DURATION_MS = 150
    _FADE_STEP_MS = 10
    _FADE_STEPS = _FADE_DURATION_MS // _FADE_STEP_MS  # 15

    def _start_fade_in(self) -> None:
        """Animate window opacity from 0 → 1 over 150ms, then show."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        # Re-apply theme on show (picks up any desktop theme change while hidden)
        self._apply_theme()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade_step = 0
        self._fade_direction = 1  # 1 = fading in
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _start_fade_out(self) -> None:
        """Animate window opacity from 1 → 0 over 150ms, then hide."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self.setWindowOpacity(1.0)
        self._fade_step = 0
        self._fade_direction = -1  # -1 = fading out
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _on_fade_tick(self) -> None:
        """Process one step of a fade animation."""
        self._fade_step += 1
        progress = self._fade_step / self._FADE_STEPS
        if self._fade_direction == 1:
            self.setWindowOpacity(min(progress, 1.0))
        else:
            self.setWindowOpacity(max(1.0 - progress, 0.0))
        if self._fade_step >= self._FADE_STEPS:
            self._fade_timer.stop()
            if self._fade_direction == -1:
                self.hide()
                self.setWindowOpacity(1.0)  # Reset for next show
    
    def toggle_panel(self) -> None:
        """Toggle panel visibility."""
        if self.isVisible():
            self.hide_panel()
        else:
            self.show_panel()
    
    def clear(self) -> None:
        """Clear all transcript content."""
        self.text_edit.clear()
        self.phrases.clear()
        self.current_phrase_idx = -1
        self._has_content = False
        self._show_empty_state()

    def update_segment(self, text: str, confidence: int, segment_index: int, is_final: bool = False, phrase_start: bool = False, speaker_id: Optional[str] = None) -> None:
        """
        Update a single segment. Each segment is part of a phrase (line).

        Args:
            text: Transcribed text for this segment
            confidence: Confidence score (0-100)
            segment_index: Position of this segment in current phrase
            is_final: If True, this phrase is complete
            phrase_start: If True, start a new phrase (new line)
            speaker_id: Optional speaker label for this phrase
        """
        if text.strip() == "[BLANK_AUDIO]":
            return
        
        # Clear empty-state placeholder on first real content
        if not self._has_content:
            self._has_content = True
            self.text_edit.clear()
        
        # Start new phrase if needed
        if phrase_start or self.current_phrase_idx < 0:
            # Insert new block before creating phrase structure
            cursor = self.text_edit.textCursor()
            if self.current_phrase_idx >= 0:
                cursor.insertBlock()  # New paragraph after previous phrase
            
            self.phrases.append(Phrase(segments=[], confidences=[], is_final=False, speaker_id=speaker_id))
            self.current_phrase_idx = len(self.phrases) - 1
            
            # Insert speaker label if provided
            if speaker_id:
                display_name = self._display_speaker_for(speaker_id)
                self._insert_speaker_label(display_name)
        
        # Get current phrase
        phrase = self.phrases[self.current_phrase_idx]
        phrase.is_final = is_final
        
        # Update or add segment to internal tracking
        if segment_index < len(phrase.segments):
            # Update existing segment - REPLACE IN PLACE
            phrase.segments[segment_index] = text
            phrase.confidences[segment_index] = confidence

            # Find and replace just this segment in display
            self._replace_segment_in_display(self.current_phrase_idx, segment_index, text, confidence)
        else:
            # Add new segment - APPEND TO CURRENT LINE
            phrase.segments.append(text)
            phrase.confidences.append(confidence)

            # Append segment to display with proper formatting
            self._append_segment_to_display(text, confidence)
        
        # Auto-scroll
        self._scroll_to_bottom()
        
        # If auto-scroll is paused, increment pending badge
        if self._auto_scroll_paused:
            self._pending_content_count += 1
            if hasattr(self, '_new_content_badge'):
                self._new_content_badge.setText(f"↓ {self._pending_content_count} new")
                self._new_content_badge.show()
                self._position_new_content_badge()
    
    # ------------------------------------------------------------------
    # Speaker label management
    # ------------------------------------------------------------------

    def set_speaker_names(self, names: Dict[str, str]) -> None:
        """Set the speaker label display mapping and rebuild the transcript.

        Args:
            names: Mapping from raw speaker labels (e.g. "spk0") to display
                   names (e.g. "Alice" or "SPK_0").
        """
        self._speaker_names = dict(names)
        self._rebuild_display()

    def get_speaker_names(self) -> Dict[str, str]:
        """Return the current speaker label mapping."""
        return dict(self._speaker_names)

    def pin_speaker_name(self, raw_label: str, name: str) -> None:
        """Pin a user-chosen name to a speaker label and refresh display.

        Args:
            raw_label: Raw speaker label (e.g. "spk0").
            name: User-chosen display name.
        """
        self._speaker_names[raw_label] = name
        self._pinned_speakers.add(raw_label)
        self._rebuild_display()

    # ------------------------------------------------------------------
    # Speaker label helpers
    # ------------------------------------------------------------------

    def _display_speaker_for(self, raw_or_display_label: str) -> str:
        """Resolve a label to its display form.

        The transcript store may contain display labels like "Alice" or
        "SPK_0" that were set by _apply_speaker_labels. We need to find
        the original raw label to check for user pins.
        """
        # Direct hit in speaker names
        if raw_or_display_label in self._speaker_names:
            return self._speaker_names[raw_or_display_label]
        # Search by value — the store has display labels, we need to
        # check if any raw label maps to this display label
        for raw, display in self._speaker_names.items():
            if display == raw_or_display_label:
                return display
        return raw_or_display_label

    def _raw_label_for_display(self, display_label: str) -> Optional[str]:
        """Find the raw label that maps to a display label.

        Returns None if no mapping found (label may be a raw label itself).
        """
        for raw, display in self._speaker_names.items():
            if display == display_label:
                return raw
        # The display label might itself be a raw label
        if display_label in self._speaker_names:
            return display_label
        return None

    def _prompt_speaker_name(self, current_label: str) -> None:
        """Show an input dialog for the user to name a speaker.

        If the user provides a name, emits speaker_name_pinned signal.
        """
        parent = self.parent() if self.parent() else self
        name, ok = QInputDialog.getText(
            parent,
            "Name Speaker",
            f"Enter a name for {current_label}:",
            text=current_label if not current_label.startswith("SPK_") else "",
        )
        if ok and name.strip():
            raw_label = self._raw_label_for_display(current_label)
            if raw_label is None:
                raw_label = current_label
            self.speaker_name_pinned.emit(raw_label, name.strip())

    def _on_anchor_clicked(self, url) -> None:
        """Handle clicks on speaker label anchors in the transcript."""
        link = url.toString() if hasattr(url, 'toString') else str(url)
        if link.startswith("speaker://"):
            speaker_id = link[len("speaker://"):]
            self._prompt_speaker_name(speaker_id)

    # ------------------------------------------------------------------
    # History tab methods
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """Refresh history when switching to the History tab."""
        if index == 1:  # History tab
            self._refresh_history()

    def _refresh_history(self) -> None:
        """Re-scan recordings and repopulate the history list."""
        try:
            from meetandread.transcription.transcript_scanner import scan_recordings
        except ImportError:
            logger.warning("transcript_scanner not available — cannot populate history")
            return
        self._populate_history_list(scan_recordings())

    def _populate_history_list(self, recordings: list) -> None:
        """Populate the history QListWidget from a list of RecordingMeta.

        Args:
            recordings: List of RecordingMeta objects (expected sorted newest-first).
        """
        self._history_list.clear()
        for meta in recordings:
            # Format display date from ISO timestamp
            display_date = meta.recording_time
            if display_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(display_date)
                    display_date = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass  # keep raw string

            if meta.word_count == 0:
                display_text = f"{display_date} | (Empty recording)"
            else:
                display_text = (
                    f"{display_date} | {meta.word_count} words"
                    f" | {meta.speaker_count} speakers"
                )

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, str(meta.path))
            self._history_list.addItem(item)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Load and display the transcript for the clicked history item."""
        # Reset comparison mode when switching items
        if self._is_comparison_mode:
            self._hide_scrub_accept_reject()

        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        if not md_path.exists():
            self._current_history_md_path = None
            self._history_viewer.setPlainText(f"(File not found: {md_path})")
            self._detail_header.show()
            return

        self._current_history_md_path = md_path
        self._detail_header.show()
        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
        else:
            # Fallback: display raw markdown without anchors
            try:
                content = md_path.read_text(encoding="utf-8")
            except OSError as exc:
                self._history_viewer.setPlainText(f"(Error reading file: {exc})")
                return
            footer_marker = "\n---\n\n<!-- METADATA:"
            marker_idx = content.find(footer_marker)
            if marker_idx != -1:
                content = content[:marker_idx]
            self._history_viewer.setMarkdown(content)

    # ------------------------------------------------------------------
    # History delete functionality
    # ------------------------------------------------------------------

    def _on_history_context_menu(self, pos) -> None:
        """Show context menu on history list items.

        Args:
            pos: Position relative to the history list widget.
        """
        item = self._history_list.itemAt(pos)
        if item is None:
            return

        menu = QMenu(self._history_list)
        p = current_palette()
        menu.setStyleSheet(context_menu_css(p, accent_color=p.danger))

        scrub_action = menu.addAction("🔄  Scrub Recording")
        delete_action = menu.addAction("🗑  Delete Recording")
        scrub_action.triggered.connect(lambda: self._on_scrub_clicked())
        delete_action.triggered.connect(lambda: self._delete_recording(item))
        menu.exec(self._history_list.mapToGlobal(pos))

    def _on_delete_btn_clicked(self) -> None:
        """Handle Delete button click in the detail header."""
        current = self._history_list.currentItem()
        if current is None:
            return
        self._delete_recording(current)

    def _delete_recording(self, item: QListWidgetItem) -> None:
        """Delete a recording after user confirmation.

        Extracts the .md path from the item, enumerates associated files,
        shows a confirmation dialog, and performs the deletion.

        Args:
            item: The QListWidgetItem representing the recording.
        """
        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return

        md_path = Path(md_path_str)
        stem = md_path.stem  # filename without .md

        # Build a human-readable name from the item display text
        recording_name = item.text().split("|")[0].strip()

        # Enumerate files to show count in confirmation
        try:
            from meetandread.recording.management import enumerate_recording_files, delete_recording
            files = enumerate_recording_files(stem)
        except Exception as exc:
            logger.error("Failed to enumerate recording files: %s", exc)
            files = []

        file_count = len(files)

        # Show confirmation dialog
        parent = self.parent() if self.parent() else self
        reply = QMessageBox.question(
            parent,
            "Delete Recording",
            f"Delete '{recording_name}'?\n\n"
            f"This will permanently remove {file_count} file{'s' if file_count != 1 else ''}.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Perform deletion
        try:
            count, deleted = delete_recording(stem)
            logger.info(
                "Deleted recording '%s': %d files removed",
                recording_name, count,
            )
        except Exception as exc:
            logger.error("Failed to delete recording '%s': %s", recording_name, exc)
            QMessageBox.warning(
                parent,
                "Delete Failed",
                f"Could not delete recording '{recording_name}'.\n\n{exc}",
            )
            return

        # Clear viewer state
        self._current_history_md_path = None
        self._history_viewer.clear()
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._detail_header.hide()

        # Refresh the history list
        self._refresh_history()

    # ------------------------------------------------------------------
    # Scrub (re-transcribe) functionality
    # ------------------------------------------------------------------

    def _on_scrub_clicked(self) -> None:
        """Handle Scrub button / context-menu click.

        Validates that the selected recording has a WAV file, shows a model
        picker dialog, then starts ScrubRunner in a background thread.
        """
        if self._is_scrubbing:
            return

        current = self._history_list.currentItem()
        if current is None:
            return

        md_path_str = current.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        stem = md_path.stem

        # Check for WAV file
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            wav_path = get_recordings_dir() / f"{stem}.wav"
        except Exception:
            wav_path = md_path.parent.parent / "recordings" / f"{stem}.wav"

        if not wav_path.exists():
            parent = self.parent() if self.parent() else self
            QMessageBox.information(
                parent,
                "Cannot Scrub",
                "Cannot scrub — audio file missing.\n\n"
                "The original .wav recording file is required for re-transcription.",
            )
            return

        # Show model picker dialog
        dialog = self._create_scrub_dialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        model_size = dialog._model_combo.currentData()
        if not model_size:
            return

        # Start the scrub
        self._start_scrub(wav_path, md_path, model_size)

    def _create_scrub_dialog(self) -> QDialog:
        """Create the model picker dialog for scrub.

        Returns a QDialog with a QComboBox showing all 5 Whisper models
        with WER from benchmark_history. Default selection is the current
        post-process model from config.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Scrub Recording")
        dialog.setFixedSize(340, 180)
        p = current_palette()
        dialog.setStyleSheet(dialog_css(p))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title label
        title_label = QLabel("Re-transcribe with a different model:")
        title_label.setStyleSheet(f"font-weight: bold; color: {p.info}; font-size: 13px;")
        layout.addWidget(title_label)

        # Model combo
        combo = QComboBox()
        combo.setStyleSheet(combo_box_css(p, accent_color=p.info))

        # Populate with models + WER
        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _bench_history = _cfg.transcription.benchmark_history
            _default_model = _cfg.transcription.postprocess_model_size
        except Exception:
            _bench_history = {}
            _default_model = "base"

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            combo.addItem(_item_text, _mn)
            if _mn == _default_model:
                _select_idx = _i
        combo.setCurrentIndex(_select_idx)

        layout.addWidget(combo)
        dialog._model_combo = combo  # store reference for caller

        layout.addStretch()

        # OK / Cancel buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        btn_box.setStyleSheet(action_button_css(p, "dialog"))
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        return dialog

    def _start_scrub(self, wav_path: Path, md_path: Path, model_size: str) -> None:
        """Start a ScrubRunner background re-transcription.

        Args:
            wav_path: Path to the source .wav audio file.
            md_path: Path to the canonical transcript .md file.
            model_size: Whisper model size (e.g. "small").
        """
        from meetandread.transcription.scrub import ScrubRunner

        # Store state
        self._scrub_model_size = model_size
        self._is_scrubbing = True
        self._is_comparison_mode = False

        # Save original transcript HTML for comparison later
        self._scrub_original_html = self._history_viewer.toHtml()

        # Disable scrub button and update text
        self._scrub_btn.setEnabled(False)
        self._scrub_btn.setText("Scrubbing... 0%")

        # Create and start runner
        self._scrub_runner = ScrubRunner(
            settings=self._get_app_settings(),
            on_progress=self._on_scrub_progress,
            on_complete=self._on_scrub_complete,
        )
        self._scrub_sidecar_path = self._scrub_runner.scrub_recording(
            wav_path, md_path, model_size,
        )

    def _get_app_settings(self):
        """Get the current AppSettings from config."""
        try:
            from meetandread.config import get_config
            return get_config()
        except Exception:
            from meetandread.config.models import AppSettings
            return AppSettings()

    def _on_scrub_progress(self, pct: int) -> None:
        """Update scrub button text with progress percentage.

        Called from the ScrubRunner background thread — uses QMetaObject
        to marshal the update to the GUI thread.
        """
        # Schedule on GUI thread
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self._scrub_btn, "setText",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, f"Scrubbing... {pct}%"),
        )

    def _on_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Handle scrub completion — show comparison or error.

        Called from the ScrubRunner background thread — schedules the
        heavy UI work on the GUI thread via QTimer.singleShot.
        """
        # Use QTimer to run on GUI thread
        QTimer.singleShot(0, lambda: self._handle_scrub_complete(sidecar_path, error))

    def _handle_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Process scrub completion on the GUI thread."""
        self._is_scrubbing = False
        self._scrub_btn.setEnabled(True)
        self._scrub_btn.setText("🔄 Scrub")

        if error:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent,
                "Scrub Failed",
                f"Re-transcription failed:\n\n{error}",
            )
            logger.error("Scrub failed: %s", error)
            return

        # Show side-by-side comparison
        self._show_scrub_comparison(sidecar_path)

    def _show_scrub_comparison(self, sidecar_path: str) -> None:
        """Show side-by-side comparison of original vs scrubbed transcript.

        Renders both transcripts in a split view with Accept/Reject buttons.

        Args:
            sidecar_path: Path to the sidecar .md file with scrub result.
        """
        sidecar = Path(sidecar_path)
        if not sidecar.exists():
            logger.warning("Sidecar not found for comparison: %s", sidecar_path)
            return

        self._is_comparison_mode = True
        self._scrub_sidecar_path = sidecar_path

        # Build the comparison view as HTML in the history viewer
        original_text = self._extract_transcript_body(
            self._current_history_md_path
        )
        scrubbed_text = self._extract_transcript_body(sidecar)

        # Build HTML with two-column layout
        html = f"""
        <html>
        <head><style>
            body {{ margin: 0; padding: 4px; background-color: #2a2a2a; color: #fff; font-size: 12px; }}
            .comparison {{ display: flex; gap: 8px; }}
            .column {{ flex: 1; }}
            .column-header {{
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px 4px 0 0;
                font-size: 11px;
                text-align: center;
            }}
            .original .column-header {{ background-color: #37474F; color: #B0BEC5; }}
            .scrubbed .column-header {{ background-color: #1B5E20; color: #A5D6A7; }}
            .content {{
                padding: 6px 8px;
                background-color: #333;
                border-radius: 0 0 4px 4px;
                min-height: 50px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style></head>
        <body>
        <div class="comparison">
            <div class="column original">
                <div class="column-header">Original</div>
                <div class="content">{_escape_html(original_text)}</div>
            </div>
            <div class="column scrubbed">
                <div class="column-header">Scrubbed ({_escape_html(self._scrub_model_size or "?")})</div>
                <div class="content">{_escape_html(scrubbed_text)}</div>
            </div>
        </div>
        </body></html>
        """

        self._history_viewer.setHtml(html)

        # Show Accept/Reject buttons instead of normal header buttons
        self._show_scrub_accept_reject()

    def _show_scrub_accept_reject(self) -> None:
        """Replace the scrub button with Accept/Reject during comparison mode."""
        self._scrub_btn.hide()

        # Create Accept button
        if not hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn = QPushButton("✓ Accept")
            self._scrub_accept_btn.setFixedHeight(26)
            p = current_palette()
            self._scrub_accept_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {p.surface};
                    color: {p.accent};
                    border: 1px solid {p.accent};
                    border-radius: 4px;
                    padding: 2px 10px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {p.surface_hover};
                    border-color: {p.accent};
                }}
                QPushButton:pressed {{
                    background-color: {p.surface};
                }}
            """)
            self._scrub_accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_accept_btn.clicked.connect(self._on_scrub_accept)

            # Create Reject button
            self._scrub_reject_btn = QPushButton("✗ Reject")
            self._scrub_reject_btn.setFixedHeight(26)
            self._scrub_reject_btn.setStyleSheet(action_button_css(p, "delete"))
            self._scrub_reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_reject_btn.clicked.connect(self._on_scrub_reject)

            # Insert into detail header layout (before delete button)
            header_layout = self._detail_header.layout()
            delete_idx = header_layout.indexOf(self._delete_btn)
            header_layout.insertWidget(delete_idx, self._scrub_accept_btn)
            header_layout.insertWidget(delete_idx + 1, self._scrub_reject_btn)
        else:
            self._scrub_accept_btn.show()
            self._scrub_reject_btn.show()

    def _hide_scrub_accept_reject(self) -> None:
        """Hide Accept/Reject buttons and show the scrub button again."""
        if hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn.hide()
        if hasattr(self, '_scrub_reject_btn'):
            self._scrub_reject_btn.hide()
        self._scrub_btn.show()
        self._is_comparison_mode = False

    def _on_scrub_accept(self) -> None:
        """Accept the scrub result — promote sidecar to canonical transcript."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.accept_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Accepted scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except FileNotFoundError:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed",
                "Sidecar file not found. It may have been deleted.",
            )
            self._hide_scrub_accept_reject()
            return
        except Exception as exc:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed", f"Could not accept scrub result:\n\n{exc}",
            )
            self._hide_scrub_accept_reject()
            return

        # Refresh the viewer with the updated transcript
        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _on_scrub_reject(self) -> None:
        """Reject the scrub result — delete the sidecar file."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.reject_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Rejected scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except Exception as exc:
            logger.warning("Error rejecting scrub: %s", exc)

        # Restore original view
        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _refresh_after_scrub(self) -> None:
        """Refresh the history list and viewer after accept/reject.

        After accept the canonical transcript changes (word count may differ),
        so the recording list must be repopulated.  After reject the list is
        refreshed as well (harmless, ensures consistency).  The previously
        selected item is re-selected so the user stays on the same recording.
        """
        md_path = self._current_history_md_path

        # Refresh the history list (word count may have changed after accept)
        self._refresh_history()

        # Re-select the item that was being viewed
        if md_path is not None:
            self._reselect_history_item(md_path)

        # Refresh the viewer content
        if md_path is not None and md_path.exists():
            html = self._render_history_transcript(md_path)
            if html is not None:
                self._history_viewer.setHtml(html)
            else:
                try:
                    content = md_path.read_text(encoding="utf-8")
                except OSError:
                    content = ""
                footer_marker = "\n---\n\n<!-- METADATA:"
                marker_idx = content.find(footer_marker)
                if marker_idx != -1:
                    content = content[:marker_idx]
                self._history_viewer.setMarkdown(content)
        else:
            self._history_viewer.clear()
            self._history_viewer.setPlaceholderText(
                "Select a recording to view its transcript",
            )

    def _reselect_history_item(self, md_path: Path) -> None:
        """Re-select a history list item by its transcript path.

        Called after the list is repopulated so the user stays on the
        same recording they were viewing.

        Args:
            md_path: Path to the transcript .md file to re-select.
        """
        md_str = str(md_path)
        for i in range(self._history_list.count()):
            item = self._history_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == md_str:
                self._history_list.setCurrentItem(item)
                return
        logger.debug("Could not re-select history item for %s", md_path)

    @staticmethod
    def _extract_transcript_body(md_path: Optional[Path]) -> str:
        """Extract the markdown body (before METADATA footer) from a transcript.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            The markdown body text, or an error message string.
        """
        if md_path is None or not md_path.exists():
            return "(file not found)"
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"(error reading file: {exc})"

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx != -1:
            content = content[:marker_idx]
        return content.strip()

    # ------------------------------------------------------------------
    # History transcript rendering with clickable speaker anchors
    # ------------------------------------------------------------------

    def _render_history_transcript(self, md_path: Path) -> Optional[str]:
        """Render a transcript .md file as HTML with clickable speaker anchors.

        Reads the .md file, parses the JSON metadata footer to get speakers,
        and returns HTML where each speaker label is an anchor tag with
        format ``speaker://{speaker_id}``.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            HTML string for the viewer, or None if no metadata is found.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read transcript for rendering: %s: %s", md_path, exc)
            return None

        # Split markdown body from JSON footer
        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            return None

        md_body = content[:marker_idx]

        # Parse metadata to find speakers
        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed metadata in %s: %s", md_path, exc)
            return None

        # Collect unique speaker IDs from words
        speakers = []
        seen = set()
        for word in data.get("words", []):
            sid = word.get("speaker_id")
            if sid is not None and sid not in seen:
                seen.add(sid)
                speakers.append(sid)

        if not speakers:
            # No speakers — just return the markdown body as-is
            return None

        # Build HTML with clickable speaker anchors
        # The markdown body has lines like "**SPK_0**" — make them anchors
        html_lines = []
        for line in md_body.splitlines():
            # Match speaker label lines: **SpeakerName**
            match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
            if match:
                speaker_label = match.group(1)
                if speaker_label in seen:
                    color = speaker_color(speaker_label)
                    html_lines.append(
                        f'<p><a href="speaker:{speaker_label}" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{speaker_label}]</a></p>'
                    )
                else:
                    html_lines.append(f"<p><b>{speaker_label}</b></p>")
            else:
                # Regular line — escape HTML and preserve whitespace
                escaped = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                # Preserve leading spaces and convert newlines
                if escaped.strip():
                    # Convert markdown italic markers
                    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
                    html_lines.append(f"<p>{escaped}</p>")
                elif not escaped:
                    html_lines.append("<br>")

        return "\n".join(html_lines)

    # ------------------------------------------------------------------
    # Speaker rename in history transcripts
    # ------------------------------------------------------------------

    def _on_history_anchor_clicked(self, url: QUrl) -> None:
        """Handle clicks on speaker label anchors in the history viewer.

        Extracts the speaker_id from the ``speaker:{id}`` URL, prompts
        the user for a new name, and performs the rename.
        """
        link = url.toString()
        prefix = "speaker:"
        if not link.startswith(prefix):
            return

        old_name = link[len(prefix):]
        if not old_name:
            return

        parent = self.parent() if self.parent() else self
        name, ok = QInputDialog.getText(
            parent,
            "Rename Speaker",
            f"Enter a new name for '{old_name}':",
            text=old_name if not old_name.startswith("SPK_") else "",
        )
        if not ok or not name.strip():
            return

        new_name = name.strip()
        if new_name == old_name:
            return

        md_path = self._current_history_md_path
        if md_path is None or not md_path.exists():
            logger.warning("No transcript file selected for rename")
            return

        try:
            self._rename_speaker_in_file(md_path, old_name, new_name)
        except Exception as exc:
            logger.error(
                "Failed to rename speaker '%s' -> '%s' in %s: %s",
                old_name, new_name, md_path, exc,
            )
            return

        # Propagate to signature store (best-effort)
        try:
            self._propagate_rename_to_signatures(md_path, old_name, new_name)
        except Exception as exc:
            logger.error(
                "Failed to propagate rename to signature store for '%s' -> '%s': %s",
                old_name, new_name, exc,
            )

        # Refresh the viewer
        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
        else:
            self._history_viewer.setPlainText("(Error refreshing after rename)")

    def _rename_speaker_in_file(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Rename a speaker in a transcript .md file.

        Updates both the JSON metadata (words and segments arrays) and the
        markdown body speaker labels.

        Args:
            md_path: Path to the transcript .md file.
            old_name: Current speaker name to replace.
            new_name: New speaker name.
        """
        content = md_path.read_text(encoding="utf-8")

        # Split into markdown body and JSON footer
        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            raise ValueError(f"No metadata footer found in {md_path}")

        md_body = content[:marker_idx]
        # Capture the full prefix including the space before JSON
        # e.g. "\n---\n\n<!-- METADATA: "
        after_marker = content[marker_idx + len(footer_marker):]
        space_before_json = ""
        if after_marker.startswith(" "):
            space_before_json = " "
            after_marker = after_marker[1:]

        metadata_text = after_marker
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        data = json.loads(metadata_text)

        # Update words array
        words_updated = 0
        for word in data.get("words", []):
            if word.get("speaker_id") == old_name:
                word["speaker_id"] = new_name
                words_updated += 1

        # Update segments array
        segments_updated = 0
        for segment in data.get("segments", []):
            if segment.get("speaker_id") == old_name:
                segment["speaker_id"] = new_name
                segments_updated += 1

        # Rebuild markdown body: replace speaker labels
        # Speaker labels appear as **OldName** on their own line
        updated_body = re.sub(
            re.escape(f"**{old_name}**"),
            f"**{new_name}**",
            md_body,
        )

        # Rebuild the file
        updated_json = json.dumps(data, indent=2)
        new_content = (
            updated_body + footer_marker + space_before_json + updated_json + " -->\n"
        )

        md_path.write_text(new_content, encoding="utf-8")

        logger.info(
            "Renamed speaker '%s' -> '%s' in %s (%d words, %d segments updated)",
            old_name, new_name, md_path, words_updated, segments_updated,
        )

    def _propagate_rename_to_signatures(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Propagate a speaker rename to the VoiceSignatureStore.

        If the old speaker name has a saved embedding in the signature
        database (located in the same directory as the transcript file),
        saves the embedding under the new name and deletes the old entry.

        Args:
            md_path: Path to the transcript file (used to locate the DB).
            old_name: Current speaker name.
            new_name: New speaker name.
        """
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
        except ImportError:
            logger.warning(
                "VoiceSignatureStore not available — skipping rename propagation"
            )
            return

        db_path = md_path.parent / "speaker_signatures.db"
        if not db_path.exists():
            # Try the default data directory
            from meetandread.audio.storage.paths import get_recordings_dir
            default_db = get_recordings_dir() / "speaker_signatures.db"
            if default_db.exists():
                db_path = default_db
            else:
                logger.info(
                    "No signature database found — speaker '%s' not in store",
                    old_name,
                )
                return

        with VoiceSignatureStore(db_path=str(db_path)) as store:
            # Find the old speaker's profile
            profiles = store.load_signatures()
            old_profile = None
            for profile in profiles:
                if profile.name == old_name:
                    old_profile = profile
                    break

            if old_profile is None:
                logger.info(
                    "Speaker '%s' not found in signature store — no propagation needed",
                    old_name,
                )
                return

            # Save under new name, delete old
            store.save_signature(
                new_name,
                old_profile.embedding,
                averaged_from_segments=old_profile.num_samples,
            )
            store.delete_signature(old_name)

            logger.info(
                "Propagated rename '%s' -> '%s' to signature store at %s",
                old_name, new_name, db_path,
            )

    def _rebuild_display(self) -> None:
        """Rebuild the entire text display from stored phrases."""
        if self.text_edit is None:
            return
        self.text_edit.clear()
        for i, phrase in enumerate(self.phrases):
            if i > 0:
                cursor = self.text_edit.textCursor()
                cursor.insertBlock()

            # Write speaker label if known
            if phrase.speaker_id:
                display_name = self._display_speaker_for(phrase.speaker_id)
                self._insert_speaker_label(display_name)

            # Write phrase text with confidence coloring
            for seg_idx, (seg_text, seg_conf) in enumerate(zip(phrase.segments, phrase.confidences)):
                self._append_segment_to_display(seg_text, seg_conf, add_space=(seg_idx > 0))

    def _insert_speaker_label(self, speaker_id: str) -> None:
        """Insert a clickable speaker label at the current cursor position."""
        cursor = self.text_edit.textCursor()
        
        # Prepend elapsed timestamp if recording is active
        if self._recording_start_time is not None:
            elapsed = int(time.time() - self._recording_start_time)
            mins = f"{elapsed // 60:02d}"
            secs = f"{elapsed % 60:02d}"
            ts_fmt = QTextCharFormat()
            ts_fmt.setForeground(QColor("#666666"))
            ts_fmt.setFontWeight(QFont.Weight.Normal)
            cursor.insertText(f"[{mins}:{secs}] ", ts_fmt)
        
        color = speaker_color(speaker_id)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Bold)
        fmt.setAnchor(True)
        fmt.setAnchorHref(f"speaker://{speaker_id}")
        label_text = f"[{speaker_id}] "
        cursor.insertText(label_text, fmt)

    def _append_segment_to_display(self, text: str, confidence: int, add_space: bool = False) -> None:
        """Append a segment to the current line with proper formatting."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Add space between segments
        if add_space or (self.phrases and self.current_phrase_idx >= 0
                         and self.phrases[self.current_phrase_idx].segments
                         and len(self.phrases[self.current_phrase_idx].segments) > 0):
            cursor.insertText(" ")

        # Determine color based on confidence
        color = self._get_confidence_color(confidence)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Normal)

        cursor.insertText(text, fmt)

    def _replace_segment_in_display(self, phrase_idx: int, segment_idx: int, text: str, confidence: int) -> None:
        """Replace a specific segment in the display without rebuilding everything."""
        cursor = self.text_edit.textCursor()
        
        # Move to start of document
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        
        # Navigate to correct phrase block
        for _ in range(phrase_idx):
            cursor.movePosition(QTextCursor.MoveOperation.NextBlock)
        
        # Navigate to correct segment within phrase
        # Segments are separated by spaces, so we move by words
        for _ in range(segment_idx):
            # Move past text and space
            cursor.movePosition(QTextCursor.MoveOperation.NextWord)
            # Skip the space between segments (if not last)
            if _ < segment_idx - 1:
                cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
        
        # Select the segment text
        cursor.movePosition(QTextCursor.MoveOperation.StartOfWord)
        # Find end of this segment (either space or block end)
        if segment_idx < len(self.phrases[phrase_idx].segments) - 1:
            # Segment is followed by space
            cursor.movePosition(QTextCursor.MoveOperation.EndOfWord, QTextCursor.MoveMode.KeepAnchor)
        else:
            # Last segment - select to end of block
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        
        # Replace with new text and formatting
        color = self._get_confidence_color(confidence)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Normal)
        cursor.insertText(text, fmt)
    
    def _get_confidence_color(self, confidence: int) -> str:
        """Get color based on confidence score — delegates to canonical thresholds."""
        return get_confidence_color(confidence)
    
    def _near_bottom_threshold(self) -> int:
        """Return a proportional bottom-detection threshold in pixels.
        
        Uses 10% of the scrollbar page step, with a floor of 10px to
        avoid degenerate cases on very small viewports. This replaces
        all hardcoded pixel thresholds for bottom detection.
        """
        return max(10, int(self.text_edit.verticalScrollBar().pageStep() * 0.1))
    
    def _on_scroll_value_changed(self, value: int) -> None:
        """
        Detect manual scroll and pause/resume auto-scroll.
        
        Called when scrollbar value changes. If user scrolls up from bottom,
        pause auto-scroll for 10 seconds to allow reading. If the user scrolls
        back to the bottom while paused, resume auto-scroll immediately.
        """
        scrollbar = self.text_edit.verticalScrollBar()
        maximum = scrollbar.maximum()
        threshold = self._near_bottom_threshold()
        
        if maximum > 0 and value < maximum - threshold:
            # User scrolled up — pause auto-scroll
            if not self._auto_scroll_paused:
                self._auto_scroll_paused = True
                self._pause_timer.start(10000)  # 10 seconds
                self.status_label.setText("Auto-scroll paused (10s)")
        elif self._auto_scroll_paused and maximum > 0 and value >= maximum - threshold:
            # User scrolled back to bottom while paused — resume
            self._auto_scroll_paused = False
            self._pause_timer.stop()
            self._pending_content_count = 0
            if hasattr(self, '_new_content_badge'):
                self._new_content_badge.hide()
            self.status_label.setText("Recording...")
            self._scroll_to_bottom()
        
        # Update tracking
        self._last_scroll_value = value
        self._is_at_bottom = (value >= maximum - threshold)
    
    def _resume_auto_scroll(self) -> None:
        """Resume auto-scroll after pause timer expires."""
        self._auto_scroll_paused = False
        self._pending_content_count = 0
        if hasattr(self, '_new_content_badge'):
            self._new_content_badge.hide()
        self.status_label.setText("Recording...")
        # Immediately scroll to bottom to catch up
        self._scroll_to_bottom()
    
    def _scroll_to_bottom(self) -> None:
        """Auto-scroll to show latest text."""
        # Don't scroll if auto-scroll is paused
        if self._auto_scroll_paused:
            return
        
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def save_to_file(self, filepath: str) -> None:
        """Save transcript to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Transcription\n\n")
            for i, phrase in enumerate(self.phrases):
                text = " ".join(phrase.segments)
                avg_conf = sum(phrase.confidences) // len(phrase.confidences) if phrase.confidences else 0
                f.write(f"{i+1}. [{avg_conf}%] {text}\n")
    
    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.closed.emit()
        event.accept()
    
    def mousePressEvent(self, event):
        """Start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self._dragging and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()


# Settings panel (similar floating approach)


class BudgetProgressBar(QProgressBar):
    """Progress bar that draws a red vertical line at the budget threshold."""

    def __init__(self, budget_percent: float = 80.0, parent=None):
        super().__init__(parent)
        self._budget_percent = budget_percent

    def paintEvent(self, event):
        """Paint the progress bar then overlay a red budget marker."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 60, 60, 220), 2))
        x = int(self.width() * self._budget_percent / 100.0)
        painter.drawLine(x, 0, x, self.height())
        painter.end()


class FloatingSettingsPanel(QWidget):
    """Frameless Aetheric Glass settings shell with sidebar navigation.

    Hosts Settings, Performance, and History sections in a left sidebar +
    right content stack layout. No internal title bar or close button —
    the shell is closed via the widget's settings affordance or hide_panel().
    """

    closed = pyqtSignal()
    model_changed = pyqtSignal(str)  # Emit model name when changed

    # Nav page indices — correspond to QStackedWidget indices
    _NAV_SETTINGS = 0
    _NAV_PERFORMANCE = 1
    _NAV_HISTORY = 2

    def __init__(self, parent: Optional[QWidget] = None,
                 controller: object = None, tray_manager: object = None,
                 main_widget: object = None):
        super().__init__(parent)
        self.setObjectName("AethericSettingsShell")
        
        # Store optional references
        self._controller = controller
        self._tray_manager = tray_manager
        self._main_widget = main_widget

        # -- Performance backend instances (wired in T03) --
        self._resource_monitor = ResourceMonitor(
            poll_interval_ms=2000,
            cpu_warning_percent=80.0,
            ram_warning_percent=85.0,
            on_snapshot=self._on_resource_snapshot,
            on_warning=self._on_resource_warning,
        )

        # -- Metrics refresh timer (updates recording metrics every 2s) --
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(2000)
        self._metrics_timer.timeout.connect(self._refresh_recording_metrics)

        # -- Benchmark state --
        self._benchmark_runner: Optional[BenchmarkRunner] = None
        self._benchmark_history: List[dict] = []  # last 5 results as dicts

        # -- Track whether Performance page is active --
        self._perf_tab_active = False

        # Window settings
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        # Glass pair: translucent background matching the widget's glass aesthetic
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        # Glass opacity — matches transcript panel
        self.setWindowOpacity(0.87)

        # Size — wider to accommodate sidebar + content stack
        self.setMinimumSize(420, 400)
        self.setMaximumSize(700, 800)

        # ------------------------------------------------------------------
        # Root layout: horizontal sidebar + content stack
        # ------------------------------------------------------------------
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ------------------------------------------------------------------
        # Left sidebar
        # ------------------------------------------------------------------
        self._sidebar = QWidget()
        self._sidebar.setObjectName("AethericSidebar")
        self._sidebar.setFixedWidth(160)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 12)
        sidebar_layout.setSpacing(6)

        # Navigation buttons
        self._nav_buttons: List[QPushButton] = []

        # Settings nav
        self._nav_settings_btn = QPushButton("⚙  Settings")
        self._nav_settings_btn.setObjectName("AethericNavButton")
        self._nav_settings_btn.setCheckable(True)
        self._nav_settings_btn.setChecked(True)
        self._nav_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_settings_btn.setProperty("nav_id", "settings")
        self._nav_settings_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_SETTINGS))
        sidebar_layout.addWidget(self._nav_settings_btn)
        self._nav_buttons.append(self._nav_settings_btn)

        # Performance nav
        self._nav_performance_btn = QPushButton("📊  Performance")
        self._nav_performance_btn.setObjectName("AethericNavButton")
        self._nav_performance_btn.setCheckable(True)
        self._nav_performance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_performance_btn.setProperty("nav_id", "performance")
        self._nav_performance_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_PERFORMANCE))
        sidebar_layout.addWidget(self._nav_performance_btn)
        self._nav_buttons.append(self._nav_performance_btn)

        # History nav (placeholder — S02 builds the real list)
        self._nav_history_btn = QPushButton("🕐  History")
        self._nav_history_btn.setObjectName("AethericNavButton")
        self._nav_history_btn.setCheckable(True)
        self._nav_history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_history_btn.setProperty("nav_id", "history")
        self._nav_history_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_HISTORY))
        sidebar_layout.addWidget(self._nav_history_btn)
        self._nav_buttons.append(self._nav_history_btn)

        sidebar_layout.addStretch()

        # Dock bay placeholder — T03 will align the real widget over/into this
        self._dock_bay = QWidget()
        self._dock_bay.setObjectName("AethericDockBay")
        self._dock_bay.setFixedHeight(48)
        dock_bay_layout = QVBoxLayout(self._dock_bay)
        dock_bay_layout.setContentsMargins(4, 4, 4, 4)
        self._dock_bay_label = QLabel("Widget Dock")
        self._dock_bay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dock_bay_layout.addWidget(self._dock_bay_label)
        sidebar_layout.addWidget(self._dock_bay)

        root_layout.addWidget(self._sidebar)

        # ------------------------------------------------------------------
        # Right content stack (QStackedWidget replacing QTabWidget)
        # ------------------------------------------------------------------
        self._content_stack = QStackedWidget()
        self._content_stack.setObjectName("AethericContentStack")
        root_layout.addWidget(self._content_stack, 1)

        # ------------------------------------------------------------------
        # Page 0: Settings — model selection + hardware info
        # ------------------------------------------------------------------
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(12, 16, 12, 12)
        settings_layout.setSpacing(5)

        # Model selection — Live Model dropdown
        self._live_model_label = QLabel("Live Model (real-time display):")
        settings_layout.addWidget(self._live_model_label)

        self._live_model_combo = QComboBox()
        self._live_model_combo.setObjectName("AethericComboBox")
        self._populate_model_dropdown(self._live_model_combo, "realtime_model_size")
        self._live_model_combo.currentIndexChanged.connect(self._on_live_model_changed)
        settings_layout.addWidget(self._live_model_combo)

        # Model selection — Post Process Model dropdown
        self._postprocess_model_label = QLabel("Post Process Model (archive quality):")
        settings_layout.addWidget(self._postprocess_model_label)

        self._postprocess_model_combo = QComboBox()
        self._postprocess_model_combo.setObjectName("AethericComboBox")
        self._populate_model_dropdown(self._postprocess_model_combo, "postprocess_model_size")
        self._postprocess_model_combo.currentIndexChanged.connect(self._on_postprocess_model_changed)
        settings_layout.addWidget(self._postprocess_model_combo)

        # Hardware detection section
        self._hardware_label = QLabel("Hardware:")
        settings_layout.addWidget(self._hardware_label)

        self.hardware_detector = HardwareDetector()
        self.model_recommender = ModelRecommender()

        ram_value = self.hardware_detector.get_ram_gb()
        cpu_cores = self.hardware_detector.get_cpu_cores()
        cpu_freq = self.hardware_detector.get_cpu_frequency()
        recommended = self.model_recommender.get_recommendation()

        self._ram_label = QLabel(f"RAM: {ram_value:.1f} GB")
        self._cpu_info_label = QLabel(f"CPU: {cpu_cores} cores @ {cpu_freq:.1f} GHz")
        self._rec_label = QLabel(f"Recommended: {recommended}")

        settings_layout.addWidget(self._ram_label)
        settings_layout.addWidget(self._cpu_info_label)
        settings_layout.addWidget(self._rec_label)

        settings_layout.addStretch()
        self._content_stack.addWidget(settings_page)

        # ------------------------------------------------------------------
        # Page 1: Performance — live resource monitoring + benchmarks
        # ------------------------------------------------------------------
        perf_page = QWidget()
        perf_layout = QVBoxLayout(perf_page)
        perf_layout.setContentsMargins(12, 16, 12, 12)
        perf_layout.setSpacing(6)

        # --- Resource Usage Section ---
        self._resource_header = QLabel("Resource Usage")
        perf_layout.addWidget(self._resource_header)

        # RAM bar
        ram_row = QHBoxLayout()
        ram_row.setSpacing(6)
        self._ram_lbl = QLabel("RAM:")
        self._ram_lbl.setFixedWidth(36)
        ram_row.addWidget(self._ram_lbl)
        self._ram_bar = BudgetProgressBar(budget_percent=85.0)
        self._ram_bar.setRange(0, 100)
        self._ram_bar.setValue(0)
        self._ram_bar.setFormat("%v%")
        ram_row.addWidget(self._ram_bar)
        perf_layout.addLayout(ram_row)

        # CPU bar
        cpu_row = QHBoxLayout()
        cpu_row.setSpacing(6)
        self._cpu_lbl = QLabel("CPU:")
        self._cpu_lbl.setFixedWidth(36)
        cpu_row.addWidget(self._cpu_lbl)
        self._cpu_bar = BudgetProgressBar(budget_percent=80.0)
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setValue(0)
        self._cpu_bar.setFormat("%v%")
        cpu_row.addWidget(self._cpu_bar)
        perf_layout.addLayout(cpu_row)

        # Resource warning indicator (hidden by default)
        self._resource_warning = QLabel("⚠ Low Resource Warning")
        self._resource_warning.hide()
        perf_layout.addWidget(self._resource_warning)

        # Separator
        self._perf_sep = QFrame()
        self._perf_sep.setFrameShape(QFrame.Shape.HLine)
        perf_layout.addWidget(self._perf_sep)

        # --- Recording Metrics Section ---
        self._rec_metrics_header = QLabel("Recording Metrics")
        perf_layout.addWidget(self._rec_metrics_header)

        self._metric_model = QLabel("Model: Not recording")
        perf_layout.addWidget(self._metric_model)

        self._metric_buffer = QLabel("Buffer: Not recording")
        perf_layout.addWidget(self._metric_buffer)

        self._metric_count = QLabel("Transcriptions: Not recording")
        perf_layout.addWidget(self._metric_count)

        self._metric_throughput = QLabel("Throughput: Not recording")
        perf_layout.addWidget(self._metric_throughput)

        # Separator
        self._perf_sep2 = QFrame()
        self._perf_sep2.setFrameShape(QFrame.Shape.HLine)
        perf_layout.addWidget(self._perf_sep2)

        # --- WER Display ---
        self._wer_label = QLabel("Last recording WER: —")
        perf_layout.addWidget(self._wer_label)

        # --- Model Selector for Benchmark ---
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._bench_model_lbl = QLabel("Benchmark Model:")
        model_row.addWidget(self._bench_model_lbl)

        self._benchmark_model_combo = QComboBox()
        self._benchmark_model_combo.setObjectName("AethericComboBox")

        # Populate with all 5 models, default to current live model
        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _default_bench_model = _cfg.transcription.realtime_model_size
            _bench_history = _cfg.transcription.benchmark_history
        except Exception:
            _default_bench_model = "tiny"
            _bench_history = {}

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            self._benchmark_model_combo.addItem(_item_text, _mn)
            if _mn == _default_bench_model:
                _select_idx = _i
        self._benchmark_model_combo.setCurrentIndex(_select_idx)
        model_row.addWidget(self._benchmark_model_combo, 1)
        perf_layout.addLayout(model_row)

        # --- Benchmark Button ---
        self._benchmark_btn = QPushButton("Run Benchmark")
        self._benchmark_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        perf_layout.addWidget(self._benchmark_btn)

        # --- Benchmark History ---
        self._history_header = QLabel("Benchmark History")
        perf_layout.addWidget(self._history_header)

        self._benchmark_history_label = QLabel("No benchmarks yet")
        self._benchmark_history_label.setWordWrap(True)
        perf_layout.addWidget(self._benchmark_history_label)

        perf_layout.addStretch()
        self._content_stack.addWidget(perf_page)

        # ------------------------------------------------------------------
        # Page 2: History placeholder — S02 implements the real list
        # ------------------------------------------------------------------
        history_page = QWidget()
        history_page.setObjectName("AethericPlaceholderRow")
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(12, 16, 12, 12)
        history_layout.setSpacing(8)

        self._history_placeholder_title = QLabel("History")
        history_layout.addWidget(self._history_placeholder_title)

        self._history_placeholder_desc = QLabel(
            "Recording history will appear here.\n\n"
            "Browse, view, scrub, and delete past recordings."
        )
        self._history_placeholder_desc.setWordWrap(True)
        history_layout.addWidget(self._history_placeholder_desc)

        history_layout.addStretch()
        self._content_stack.addWidget(history_page)

        # ------------------------------------------------------------------
        # Resize grip — direct child of panel, positioned at bottom-right
        # ------------------------------------------------------------------
        self._resize_grip = QSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resize_grip.show()

        # Wire benchmark button
        self._benchmark_btn.clicked.connect(self._on_benchmark_clicked)

        # Dragging
        self._dragging = False
        self._drag_pos = None

        # Apply initial theme to all widgets
        self._apply_theme()

        # Connect to desktop theme changes for live re-theming
        try:
            from PyQt6.QtGui import QGuiApplication
            hints = QGuiApplication.styleHints()
            if hints is not None:
                hints.colorSchemeChanged.connect(lambda: self._apply_theme())
        except (ImportError, RuntimeError):
            pass

    def resizeEvent(self, event) -> None:
        """Reposition resize grip on resize."""
        if hasattr(self, '_resize_grip'):
            self._resize_grip.move(
                self.width() - self._resize_grip.width(),
                self.height() - self._resize_grip.height(),
            )
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Adaptive theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply Aetheric Glass theme to all settings panel widgets.

        Idempotent and cheap — just re-sets stylesheets from the current
        palette.  Called once at end of __init__ and on desktop theme change.
        """
        p = current_palette()
        self._current_palette = p

        # Panel shell — Aetheric Glass
        self.setStyleSheet(aetheric_settings_shell_css(p))

        # Sidebar
        self._sidebar.setStyleSheet(aetheric_sidebar_css(p))

        # Nav buttons
        nav_css = aetheric_nav_button_css(p)
        for btn in self._nav_buttons:
            btn.setStyleSheet(nav_css)

        # Dock bay
        self._dock_bay.setStyleSheet(aetheric_dock_bay_css(p))
        self._dock_bay_label.setStyleSheet(
            f"QLabel {{ color: {AETHERIC_NAV_INACTIVE_TEXT}; font-size: 10px; }}"
        )

        # Content stack — transparent so per-page styles show through
        self._content_stack.setStyleSheet(
            "QStackedWidget { background-color: transparent; border: none; }"
        )

        # Settings page — labels and combos
        self._live_model_label.setStyleSheet(status_label_css(p))
        self._live_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._postprocess_model_label.setStyleSheet(status_label_css(p))
        self._postprocess_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._hardware_label.setStyleSheet(status_label_css(p))
        self._ram_label.setStyleSheet(info_label_css(p))
        self._cpu_info_label.setStyleSheet(info_label_css(p))
        self._rec_label.setStyleSheet(
            f"QLabel {{ font-weight: bold; color: {p.accent}; }}"
        )

        # Performance page — section headers
        self._resource_header.setStyleSheet(
            f"QLabel {{ color: {p.accent}; font-weight: bold; font-size: 12px; padding: 2px; }}"
        )

        # RAM/CPU bar labels
        self._ram_lbl.setStyleSheet(info_label_css(p))
        self._cpu_lbl.setStyleSheet(info_label_css(p))

        # Progress bars — semantic chunk colors stay constant
        self._ram_bar.setStyleSheet(progress_bar_css(p, "#4CAF50"))
        self._cpu_bar.setStyleSheet(progress_bar_css(p, "#2196F3"))

        # Resource warning — semantic orange stays but text colour adapts
        self._resource_warning.setStyleSheet(
            f"QLabel {{ color: #FF9800; font-size: 11px; font-weight: bold; padding: 2px; }}"
        )

        # Separators
        self._perf_sep.setStyleSheet(separator_css(p))
        self._perf_sep2.setStyleSheet(separator_css(p))

        # Recording metrics header
        self._rec_metrics_header.setStyleSheet(
            f"QLabel {{ color: {p.accent}; font-weight: bold; font-size: 12px; padding: 2px; }}"
        )

        # Metric labels
        self._metric_model.setStyleSheet(info_label_css(p))
        self._metric_buffer.setStyleSheet(info_label_css(p))
        self._metric_count.setStyleSheet(info_label_css(p))
        self._metric_throughput.setStyleSheet(info_label_css(p))

        # WER label
        self._wer_label.setStyleSheet(
            f"QLabel {{ color: {p.text}; font-size: 12px; font-weight: bold; padding: 2px; }}"
        )

        # Benchmark section
        self._bench_model_lbl.setStyleSheet(info_label_css(p))
        self._benchmark_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._benchmark_btn.setStyleSheet(action_button_css(p, "benchmark"))
        self._history_header.setStyleSheet(status_label_css(p))
        self._benchmark_history_label.setStyleSheet(
            f"QLabel {{ color: {p.text_disabled}; font-size: 11px; padding: 2px; }}"
        )

        # History placeholder page
        history_css = aetheric_placeholder_css(p)
        history_page = self._content_stack.widget(self._NAV_HISTORY)
        if history_page is not None:
            history_page.setStyleSheet(history_css)
        self._history_placeholder_title.setStyleSheet(
            f"QLabel {{ color: {AETHERIC_RED}; font-weight: bold; font-size: 14px; }}"
        )
        self._history_placeholder_desc.setStyleSheet(
            f"QLabel {{ color: {AETHERIC_NAV_INACTIVE_TEXT}; font-size: 12px; }}"
        )

        # Resize grip
        self._resize_grip.setStyleSheet(resize_grip_css(p))

        scheme_name = "dark" if p is DARK_PALETTE else "light"
        logger.info("Applied %s Aetheric theme to FloatingSettingsPanel", scheme_name)

    def show_panel(self):
        """Show the panel with a 150ms fade-in and start monitoring if on Performance tab."""
        self._start_fade_in()
        # Activate monitoring if Performance tab is visible
        if self._perf_tab_active:
            self._start_resource_monitor()
            self._metrics_timer.start()
    
    def hide_panel(self):
        """Hide the panel with a 150ms fade-out and stop monitoring."""
        self._stop_resource_monitor()
        self._metrics_timer.stop()
        self._start_fade_out()

    # ------------------------------------------------------------------
    # Fade transition helpers
    # ------------------------------------------------------------------

    _FADE_DURATION_MS = 150
    _FADE_STEP_MS = 10
    _FADE_STEPS = _FADE_DURATION_MS // _FADE_STEP_MS  # 15

    def _start_fade_in(self) -> None:
        """Animate window opacity from 0 → 1 over 150ms, then show."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        # Re-apply theme on show (picks up any desktop theme change while hidden)
        self._apply_theme()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade_step = 0
        self._fade_direction = 1  # 1 = fading in
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _start_fade_out(self) -> None:
        """Animate window opacity from 1 → 0 over 150ms, then hide."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self.setWindowOpacity(1.0)
        self._fade_step = 0
        self._fade_direction = -1  # -1 = fading out
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _on_fade_tick(self) -> None:
        """Process one step of a fade animation."""
        self._fade_step += 1
        progress = self._fade_step / self._FADE_STEPS
        if self._fade_direction == 1:
            self.setWindowOpacity(min(progress, 1.0))
        else:
            self.setWindowOpacity(max(1.0 - progress, 0.0))
        if self._fade_step >= self._FADE_STEPS:
            self._fade_timer.stop()
            if self._fade_direction == -1:
                self.hide()
                self.setWindowOpacity(1.0)  # Reset for next show
    
    def dock_to_widget(self, widget: QWidget, position: str = "left") -> None:
        """
        Position panel next to a widget.
        
        Args:
            widget: The main widget to dock to
            position: "left", "right", "top", "bottom"
        """
        # Get widget position in screen coordinates
        widget_pos = widget.mapToGlobal(widget.rect().topLeft())
        widget_rect = widget.geometry()
        
        # Calculate panel position
        if position == "left":
            x = widget_pos.x() - self.width() - 10
            y = widget_pos.y()
        elif position == "right":
            x = widget_pos.x() + widget_rect.width() + 10
            y = widget_pos.y()
        elif position == "top":
            x = widget_pos.x()
            y = widget_pos.y() - self.height() - 10
        else:  # bottom
            x = widget_pos.x()
            y = widget_pos.y() + widget_rect.height() + 10
        
        self.move(x, y)
    
    # ------------------------------------------------------------------
    # Performance tab wiring (T03)
    # ------------------------------------------------------------------

    def _on_nav_clicked(self, page_index: int) -> None:
        """Handle sidebar nav clicks — switch page and manage ResourceMonitor.

        Args:
            page_index: QStackedWidget index (_NAV_SETTINGS, _NAV_PERFORMANCE, _NAV_HISTORY).
        """
        if page_index < 0 or page_index >= self._content_stack.count():
            logger.warning("Invalid nav index %d — ignoring", page_index)
            return

        # Update checked state on nav buttons (exclusive toggle)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_index)

        # Switch content stack
        self._content_stack.setCurrentIndex(page_index)

        # Track performance active state
        self._perf_tab_active = (page_index == self._NAV_PERFORMANCE)

        # Start/stop monitoring based on visibility
        if self._perf_tab_active and self.isVisible():
            self._start_resource_monitor()
            self._metrics_timer.start()
            self._refresh_recording_metrics()
        else:
            self._stop_resource_monitor()
            self._metrics_timer.stop()

        nav_id = self._nav_buttons[page_index].property("nav_id") if page_index < len(self._nav_buttons) else "?"
        logger.info("Settings nav changed to '%s' (index %d)", nav_id, page_index)

    def _start_resource_monitor(self) -> None:
        """Start the ResourceMonitor if not already running."""
        if not self._resource_monitor.is_running:
            self._resource_monitor.start()
            logger.info("ResourceMonitor started for Performance tab")

    def _stop_resource_monitor(self) -> None:
        """Stop the ResourceMonitor if running."""
        if self._resource_monitor.is_running:
            self._resource_monitor.stop()
            logger.info("ResourceMonitor stopped")

    def _on_resource_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Update RAM/CPU bars from a resource snapshot.

        Args:
            snapshot: ResourceSnapshot with current metrics.
        """
        self._ram_bar.setValue(int(snapshot.ram_percent))
        self._cpu_bar.setValue(int(snapshot.cpu_percent))

        # Get current palette for theme-aware progress bar styling
        p = current_palette()

        # Color-code RAM bar: green → orange → red (budget: 85% orange, 90% red)
        if snapshot.ram_percent >= 90:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#F44336"))
        elif snapshot.ram_percent >= 85:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#FF9800"))
        else:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#4CAF50"))

        # Color-code CPU bar: blue → orange → red (budget: 80% orange, 90% red)
        if snapshot.cpu_percent >= 90:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#F44336"))
        elif snapshot.cpu_percent >= 80:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#FF9800"))
        else:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#2196F3"))

    def _on_resource_warning(self, resource_name: str, value: float, threshold: float) -> None:
        """Show resource warning indicator and optionally send tray notification.

        Args:
            resource_name: 'ram' or 'cpu'.
            value: Current usage percentage.
            threshold: Warning threshold percentage.
        """
        self._resource_warning.setText(f"⚠ High {resource_name.upper()}: {value:.0f}% (threshold: {threshold:.0f}%)")
        self._resource_warning.show()

        # Auto-hide after 10 seconds if resource recovers
        QTimer.singleShot(10000, self._check_hide_resource_warning)

        # Send tray notification if available
        if self._tray_manager is not None:
            try:
                tray = self._tray_manager.tray_icon
                tray.showMessage(
                    "Resource Warning",
                    f"High {resource_name.upper()} usage: {value:.0f}% (threshold: {threshold:.0f}%)",
                )
            except Exception as exc:
                logger.debug("Failed to send tray notification: %s", exc)

        # Also show warning on the main widget scene
        if self._main_widget is not None:
            try:
                self._main_widget._show_resource_warning(
                    f"⚠ High {resource_name.upper()}: {value:.0f}%"
                )
            except Exception as exc:
                logger.debug("Failed to show main widget resource warning: %s", exc)

    def _check_hide_resource_warning(self) -> None:
        """Hide warning if resources are back to normal."""
        snap = self._resource_monitor.current_snapshot
        if snap is not None:
            if (snap.ram_percent < self._resource_monitor.ram_warning_percent and
                    snap.cpu_percent < self._resource_monitor.cpu_warning_percent):
                self._resource_warning.hide()

    def _refresh_recording_metrics(self) -> None:
        """Update recording metrics labels from the controller's transcription processor.

        Only updates when the controller is recording and has an active processor.
        """
        if self._controller is None:
            return

        try:
            if not self._controller.is_recording():
                self._metric_model.setText("Model: Not recording")
                self._metric_buffer.setText("Buffer: Not recording")
                self._metric_count.setText("Transcriptions: Not recording")
                self._metric_throughput.setText("Throughput: Not recording")
                return

            processor = getattr(self._controller, '_transcription_processor', None)
            if processor is None:
                return

            stats = processor.get_stats()

            # Model info
            model_size = stats.get('model_size', 'unknown')
            self._metric_model.setText(f"Model: {model_size}")

            # Buffer duration
            buffer_dur = stats.get('buffer_duration', 0)
            self._metric_buffer.setText(f"Buffer: {buffer_dur:.1f}s")

            # Transcription count
            count = stats.get('transcription_count', 0)
            self._metric_count.setText(f"Transcriptions: {count}")

            # Throughput (buffer duration / total audio processed)
            total_samples = stats.get('total_samples', 0)
            if total_samples > 0:
                audio_seconds = total_samples / 16000
                self._metric_throughput.setText(f"Throughput: {audio_seconds:.1f}s audio")
            else:
                self._metric_throughput.setText("Throughput: —")

        except Exception as exc:
            logger.debug("Error refreshing recording metrics: %s", exc)

    def update_wer_display(self, wer_value: Optional[float]) -> None:
        """Update the WER display label.

        Args:
            wer_value: WER as a float (0.0–1.0+), or None to reset.
        """
        p = current_palette()
        if wer_value is None:
            self._wer_label.setText("Last recording WER: —")
            self._wer_label.setStyleSheet(
                f"QLabel {{ color: {p.text}; font-size: 12px; font-weight: bold; padding: 2px; }}"
            )
        else:
            pct = wer_value * 100
            if wer_value <= 0.1:
                color = "#4CAF50"  # green — excellent
            elif wer_value <= 0.3:
                color = "#FFC107"  # yellow — acceptable
            else:
                color = "#F44336"  # red — poor
            self._wer_label.setText(f"Last recording WER: {pct:.1f}%")
            self._wer_label.setStyleSheet(
                f"QLabel {{ color: {color}; font-size: 12px; font-weight: bold; padding: 2px; }}"
            )

    def _on_benchmark_clicked(self) -> None:
        """Handle 'Run Benchmark' button click.

        Creates a BenchmarkRunner using the model selected in the
        benchmark model dropdown, and runs it asynchronously.
        """
        if self._benchmark_runner and self._benchmark_runner.is_running:
            logger.info("Benchmark already running, ignoring click")
            return

        # Disable button and show progress
        self._benchmark_btn.setEnabled(False)
        self._benchmark_btn.setText("Running...")

        # Read model selection from the benchmark model combo.
        # currentData() returns the plain model name (e.g. "base"),
        # but currentText() may include WER annotation — use data.
        model_size = self._benchmark_model_combo.currentData() or "tiny"

        # Create a fresh engine for the selected model.
        engine = None
        try:
            from meetandread.transcription.engine import WhisperTranscriptionEngine
            engine = WhisperTranscriptionEngine(model_size=model_size)
            engine.load_model()
        except Exception as exc:
            logger.warning("Could not create transcription engine for benchmark: %s", exc)

        self._benchmark_runner = BenchmarkRunner(
            engine=engine,
            on_progress=self._on_benchmark_progress,
            on_complete=self._on_benchmark_complete,
        )
        self._benchmark_runner.run_async()

    def _on_benchmark_progress(self, percent: int) -> None:
        """Update benchmark button text with progress.

        Args:
            percent: Progress percentage (0-100).
        """
        self._benchmark_btn.setText(f"Running... {percent}%")

    def _on_benchmark_complete(self, result: BenchmarkResult) -> None:
        """Handle benchmark completion — persist per-model result to config and update UI.

        Args:
            result: BenchmarkResult with WER, latency, and throughput data.
        """
        # Re-enable button
        self._benchmark_btn.setEnabled(True)
        self._benchmark_btn.setText("Run Benchmark")

        if result.error:
            self._benchmark_history_label.setText(f"Benchmark failed: {result.error}")
            p = current_palette()
            self._benchmark_history_label.setStyleSheet(
                f"QLabel {{ color: {p.danger}; font-size: 11px; padding: 2px; }}"
            )
            return

        # Extract model name from result
        model_name = result.model_info.get("model_size", "unknown") if result.model_info else "unknown"

        # Format result
        wer_pct = result.wer * 100
        result_text = (
            f"{model_name}: WER {wer_pct:.1f}% | "
            f"Latency: {result.total_latency_s:.2f}s | "
            f"Speed: {result.throughput_ratio:.1f}x realtime"
        )

        # Store in local history (keep last 5)
        self._benchmark_history.append({
            "wer": result.wer,
            "latency_s": result.total_latency_s,
            "throughput": result.throughput_ratio,
            "model_info": result.model_info,
        })
        if len(self._benchmark_history) > 5:
            self._benchmark_history = self._benchmark_history[-5:]

        # Persist per-model result to config
        try:
            from meetandread.config import get_config, set_config, save_config
            settings = get_config()
            history = dict(settings.transcription.benchmark_history)

            from datetime import datetime
            history[model_name] = {
                "wer": result.wer,
                "timestamp": datetime.now().isoformat(),
            }

            set_config("transcription.benchmark_history", history)
            save_config()
            logger.info("Persisted benchmark result for model '%s' to config", model_name)
        except Exception as exc:
            logger.warning("Failed to persist benchmark result to config: %s", exc)

        # Build per-model history display
        lines = []
        for i, entry in enumerate(reversed(self._benchmark_history), 1):
            w = entry["wer"] * 100
            t = entry["throughput"]
            m = entry.get("model_info", {}).get("model_size", "unknown")
            ts = ""
            # Show timestamp from config if available
            try:
                from meetandread.config import get_config
                _cfg = get_config()
                _hist_entry = _cfg.transcription.benchmark_history.get(m)
                if _hist_entry and "timestamp" in _hist_entry:
                    ts = f" ({_hist_entry['timestamp'][:16]})"
            except Exception:
                pass
            lines.append(f"#{i} {m}: WER {w:.1f}% | Speed {t:.1f}x{ts}")

        self._benchmark_history_label.setText("\n".join(lines))
        p = current_palette()
        self._benchmark_history_label.setStyleSheet(
            f"QLabel {{ color: {p.text_tertiary}; font-size: 11px; padding: 2px; }}"
        )

        # Update the benchmark model dropdown to reflect new WER
        self._refresh_benchmark_model_combo()

        # Refresh Settings dropdowns with updated WER data
        self._refresh_dropdown_wer()

        # Also update the WER display with benchmark result
        self.update_wer_display(result.wer)

        logger.info(
            "Benchmark complete: model=%s, WER=%.3f, throughput=%.1fx, latency=%.2fs",
            model_name, result.wer, result.throughput_ratio, result.total_latency_s,
        )

    # ------------------------------------------------------------------
    # Model dropdown helpers
    # ------------------------------------------------------------------

    def _populate_model_dropdown(self, combo: QComboBox, config_key: str) -> None:
        """Populate a model dropdown with all 5 models and WER annotations.

        Reads benchmark_history from config and MODEL_SPECS for model info.
        Sets the current selection from the config value.

        Args:
            combo: QComboBox to populate.
            config_key: Config key within transcription settings
                ('realtime_model_size' or 'postprocess_model_size').
        """
        combo.blockSignals(True)
        combo.clear()

        try:
            from meetandread.config import get_config
            settings = get_config()
            benchmark_history = settings.transcription.benchmark_history
            current_model = getattr(settings.transcription, config_key, "tiny")
        except Exception:
            benchmark_history = {}
            current_model = "tiny"

        model_order = ["tiny", "base", "small", "medium", "large"]
        select_index = 0

        for i, model_name in enumerate(model_order):
            entry = benchmark_history.get(model_name)
            if entry and "wer" in entry:
                wer_pct = entry["wer"] * 100
                item_text = f"{model_name} — WER: {wer_pct:.1f}%"
            else:
                item_text = f"{model_name} (not benchmarked)"

            combo.addItem(item_text, model_name)

            if model_name == current_model:
                select_index = i

        combo.setCurrentIndex(select_index)
        combo.blockSignals(False)

    def _on_live_model_changed(self, index: int) -> None:
        """Handle Live Model dropdown selection change.

        Updates config and emits model_changed signal.
        """
        model_size = self._live_model_combo.currentData()
        if model_size is None:
            return

        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.realtime_model_size", model_size)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save live model selection: %s", exc)

        self.model_changed.emit(model_size)
        logger.info("Live model changed to: %s", model_size)

    def _on_postprocess_model_changed(self, index: int) -> None:
        """Handle Post Process Model dropdown selection change.

        Updates config (no model_changed signal — that's for live model only).
        """
        model_size = self._postprocess_model_combo.currentData()
        if model_size is None:
            return

        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.postprocess_model_size", model_size)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save post-process model selection: %s", exc)

        logger.info("Post-process model changed to: %s", model_size)

    def _refresh_dropdown_wer(self) -> None:
        """Update all dropdown item texts with latest WER from config."""
        self._populate_model_dropdown(self._live_model_combo, "realtime_model_size")
        self._populate_model_dropdown(self._postprocess_model_combo, "postprocess_model_size")

    def _refresh_benchmark_model_combo(self) -> None:
        """Update benchmark model dropdown items with latest WER from config.

        Preserves the current model selection. On first call (empty combo),
        defaults to the current live model from config.
        """
        current_model = self._benchmark_model_combo.currentData()
        if current_model is None:
            # First call — default to current live model
            try:
                from meetandread.config import get_config
                current_model = get_config().transcription.realtime_model_size
            except Exception:
                current_model = "tiny"

        self._benchmark_model_combo.blockSignals(True)
        self._benchmark_model_combo.clear()

        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _bench_history = _cfg.transcription.benchmark_history
        except Exception:
            _bench_history = {}

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            self._benchmark_model_combo.addItem(_item_text, _mn)
            if _mn == current_model:
                _select_idx = _i
        self._benchmark_model_combo.setCurrentIndex(_select_idx)
        self._benchmark_model_combo.blockSignals(False)

    def update_benchmark_display(self, wer_by_model: dict) -> None:
        """Refresh both model dropdowns after benchmark completes.

        Writes WER results to config and refreshes dropdown text.

        Args:
            wer_by_model: Dict mapping model_size -> WER float (0.0-1.0).
        """
        try:
            from meetandread.config import get_config, set_config, save_config
            settings = get_config()
            history = dict(settings.transcription.benchmark_history)

            from datetime import datetime
            now = datetime.now().isoformat()
            for model_size, wer in wer_by_model.items():
                history[model_size] = {"wer": wer, "timestamp": now}

            set_config("transcription.benchmark_history", history)
            save_config()
        except Exception as exc:
            logger.warning("Failed to update benchmark history in config: %s", exc)

        self._refresh_dropdown_wer()

    def closeEvent(self, event):
        """Handle close event."""
        self.closed.emit()
        event.accept()
    
    def mousePressEvent(self, event):
        """Start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self._dragging and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    panel = FloatingTranscriptPanel()
    panel.show_panel()
    
    # Test adding some content
    panel.update_segment("Hello", 85, 0, is_final=False)
    panel.update_segment("world", 90, 1, is_final=False)
    panel.update_segment("this is", 75, 2, is_final=True)
    
    # New phrase
    panel.update_segment("New phrase", 80, 0, phrase_start=True, is_final=False)
    panel.update_segment("here", 85, 1, is_final=True)
    
    sys.exit(app.exec())
