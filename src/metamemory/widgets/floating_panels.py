"""
Floating transcript panel - separate window that docks to the main widget.

This solves the clipping issue by making the panel a separate QWidget
that floats outside the main widget bounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLabel, QFrame, QHBoxLayout, QPushButton,
    QInputDialog, QApplication, QTabWidget, QListWidget, QListWidgetItem,
    QSplitter, QTextBrowser,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

from metamemory.hardware.detector import HardwareDetector
from metamemory.hardware.recommender import ModelRecommender

import logging

logger = logging.getLogger(__name__)


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
        
        # Window settings
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Don't show in taskbar
        )
        
        # Size — slightly larger to accommodate history view
        self.setFixedSize(450, 400)
        
        # Style
        self.setStyleSheet("""
            FloatingTranscriptPanel {
                background-color: #1a1a1a;
                border: 2px solid #444;
                border-radius: 10px;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header with title and close button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(5)
        
        # Title bar (clickable for dragging)
        title = QLabel("Live Transcript")
        title.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #fff;
                border: 1px solid #555;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #F44336;
                border-color: #F44336;
            }
        """)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setToolTip("Close panel")
        close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # ------------------------------------------------------------------
        # Tab widget — Live and History tabs
        # ------------------------------------------------------------------
        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 5px;
                background-color: #1a1a1a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #aaa;
                padding: 6px 14px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #333;
                color: #4CAF50;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
        """)
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
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #fff;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        # Handle anchor clicks on speaker labels (signal only on QTextBrowser)
        self.text_edit.setMouseTracking(True)
        if hasattr(self.text_edit, "anchorClicked"):
            self.text_edit.anchorClicked.connect(self._on_anchor_clicked)
        live_layout.addWidget(self.text_edit)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 11px;
                padding: 3px;
            }
        """)
        live_layout.addWidget(self.status_label)

        self._tab_widget.addTab(live_tab, "Live")

        # ------------------------------------------------------------------
        # History tab — recording list and transcript viewer
        # ------------------------------------------------------------------
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #444;
                height: 3px;
            }
        """)

        # Top: recording list
        self._history_list = QListWidget()
        self._history_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                color: #ddd;
                border: none;
                border-radius: 5px;
                font-size: 12px;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background-color: #37474F;
                color: #fff;
            }
            QListWidget::item:hover {
                background-color: #2f3f3f;
            }
        """)
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        splitter.addWidget(self._history_list)

        # Bottom: transcript viewer (read-only, supports anchor clicks)
        self._history_viewer = QTextBrowser()
        self._history_viewer.setReadOnly(True)
        self._history_viewer.setStyleSheet("""
            QTextBrowser {
                background-color: #2a2a2a;
                color: #fff;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        self._history_viewer.setFrameShape(QFrame.Shape.NoFrame)
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._history_viewer.setOpenExternalLinks(False)
        self._history_viewer.setOpenLinks(False)
        self._history_viewer.anchorClicked.connect(self._on_history_anchor_clicked)
        splitter.addWidget(self._history_viewer)

        # 40% list / 60% viewer
        splitter.setSizes([160, 240])

        history_layout.addWidget(splitter)
        self._tab_widget.addTab(history_tab, "History")

        # Connect tab change to refresh history when switching to it
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Track the currently-viewed history transcript path (for rename)
        self._current_history_md_path: Optional[Path] = None
        
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
        
        # Connect to scrollbar value changed signal to detect manual scroll
        self.text_edit.verticalScrollBar().valueChanged.connect(self._on_scroll_value_changed)
    
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
    
    def show_panel(self) -> None:
        """Show the panel with a 150ms fade-in and start auto-scroll."""
        self._start_fade_in()
        self.scroll_timer.start(100)  # Scroll check every 100ms
        self.status_label.setText("Recording...")
    
    def hide_panel(self) -> None:
        """Hide the panel with a 150ms fade-out."""
        self.scroll_timer.stop()
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
            print(f"DEBUG Panel: Skipping [BLANK_AUDIO]")
            return
        
        # Start new phrase if needed
        if phrase_start or self.current_phrase_idx < 0:
            print(f"DEBUG Panel: Starting NEW PHRASE")
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
            print(f"DEBUG Panel: Updated segment {segment_index}: '{text}' [conf: {confidence}%]")

            # Find and replace just this segment in display
            self._replace_segment_in_display(self.current_phrase_idx, segment_index, text, confidence)
        else:
            # Add new segment - APPEND TO CURRENT LINE
            phrase.segments.append(text)
            phrase.confidences.append(confidence)
            print(f"DEBUG Panel: Added segment {segment_index}: '{text}' [conf: {confidence}%]")

            # Append segment to display with proper formatting
            self._append_segment_to_display(text, confidence)
        
        # Update status
        total_segments = sum(len(p.segments) for p in self.phrases)
        self.status_label.setText(f"Phrases: {len(self.phrases)} | Segments: {total_segments}")
        
        # Auto-scroll
        self._scroll_to_bottom()
    
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
            from metamemory.transcription.transcript_scanner import scan_recordings
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
        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        if not md_path.exists():
            self._current_history_md_path = None
            self._history_viewer.setPlainText(f"(File not found: {md_path})")
            return

        self._current_history_md_path = md_path
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
            from metamemory.speaker.signatures import VoiceSignatureStore
        except ImportError:
            logger.warning(
                "VoiceSignatureStore not available — skipping rename propagation"
            )
            return

        db_path = md_path.parent / "speaker_signatures.db"
        if not db_path.exists():
            # Try the default data directory
            from metamemory.audio.storage.paths import get_recordings_dir
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
        """Get color based on confidence score."""
        if confidence >= 85:
            return "#4CAF50"  # Green
        elif confidence >= 70:
            return "#FFC107"  # Yellow
        elif confidence >= 50:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red
    
    def _on_scroll_value_changed(self, value: int) -> None:
        """
        Detect manual scroll and pause auto-scroll.
        
        Called when scrollbar value changes. If user scrolls up from bottom,
        pause auto-scroll for 10 seconds to allow reading.
        """
        scrollbar = self.text_edit.verticalScrollBar()
        maximum = scrollbar.maximum()
        
        # Check if user scrolled up from bottom (not at maximum)
        if maximum > 0 and value < maximum - 10:  # 10 pixel threshold
            # User scrolled up - pause auto-scroll
            if not self._auto_scroll_paused:
                print(f"DEBUG Panel: Manual scroll detected (value={value}, max={maximum}), pausing auto-scroll")
                self._auto_scroll_paused = True
                self._pause_timer.start(10000)  # 10 seconds
                self.status_label.setText("Auto-scroll paused (10s)")
        
        # Update tracking
        self._last_scroll_value = value
        self._is_at_bottom = (value >= maximum - 5)  # Within 5 pixels of bottom
    
    def _resume_auto_scroll(self) -> None:
        """Resume auto-scroll after pause timer expires."""
        print("DEBUG Panel: Auto-scroll pause expired, resuming")
        self._auto_scroll_paused = False
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
class FloatingSettingsPanel(QWidget):
    """Floating settings panel for model selection."""
    
    closed = pyqtSignal()
    model_changed = pyqtSignal(str)  # Emit model name when changed
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Window settings
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Size
        self.setFixedSize(300, 420)
        
        # Style — aligned with FloatingTranscriptPanel dark theme
        self.setStyleSheet("""
            FloatingSettingsPanel {
                background-color: #1a1a1a;
                border: 2px solid #444;
                border-radius: 10px;
            }
            QLabel {
                color: #ddd;
                font-size: 12px;
            }
            QRadioButton {
                color: #ddd;
                font-size: 12px;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header with title and close button (matches transcript panel)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(5)
        
        title = QLabel("Settings")
        title.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #fff;
                border: 1px solid #555;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #F44336;
                border-color: #F44336;
            }
        """)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setToolTip("Close panel")
        close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("QFrame { background-color: #444; max-height: 1px; border: none; }")
        layout.addWidget(separator)
        
        # Model selection
        model_label = QLabel("Model Size:")
        model_label.setStyleSheet("QLabel { color: #888; font-size: 11px; padding: 3px; }")
        layout.addWidget(model_label)
        
        from PyQt6.QtWidgets import QButtonGroup, QRadioButton
        
        self.model_group = QButtonGroup(self)
        models = [("tiny", "Tiny (fastest)"), ("base", "Base (balanced)"), ("small", "Small (accurate)")]
        
        for model_id, model_name in models:
            btn = QRadioButton(model_name)
            btn.setStyleSheet("color: #ddd;")

            # Emit signal when button is checked (toggled True)
            btn.toggled.connect(
                lambda checked, m=model_id: checked and self.model_changed.emit(m)
            )

            self.model_group.addButton(btn)
            layout.addWidget(btn)

            if model_id == "tiny":
                btn.setChecked(True)

        # Hardware detection section
        hardware_label = QLabel("Hardware:")
        hardware_label.setStyleSheet("QLabel { color: #888; font-size: 11px; padding: 3px; }")
        layout.addWidget(hardware_label)

        self.hardware_detector = HardwareDetector()
        self.model_recommender = ModelRecommender()

        ram_value = self.hardware_detector.get_ram_gb()
        cpu_cores = self.hardware_detector.get_cpu_cores()
        cpu_freq = self.hardware_detector.get_cpu_frequency()
        recommended = self.model_recommender.get_recommendation()

        ram_label = QLabel(f"RAM: {ram_value:.1f} GB")
        cpu_label = QLabel(f"CPU: {cpu_cores} cores @ {cpu_freq:.1f} GHz")
        rec_label = QLabel(f"Recommended: {recommended}")
        rec_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

        layout.addWidget(ram_label)
        layout.addWidget(cpu_label)
        layout.addWidget(rec_label)

        layout.addStretch()

        # Dragging
        self._dragging = False
        self._drag_pos = None

    def show_panel(self):
        """Show the panel with a 150ms fade-in."""
        self._start_fade_in()
    
    def hide_panel(self):
        """Hide the panel with a 150ms fade-out."""
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
