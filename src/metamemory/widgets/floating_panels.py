"""
Floating transcript panel - separate window that docks to the main widget.

This solves the clipping issue by making the panel a separate QWidget
that floats outside the main widget bounds.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Phrase:
    """A phrase (line) of transcript with its segments."""
    segments: List[str]  # Text of each segment
    confidences: List[int]  # Confidence of each segment
    is_final: bool  # True if phrase is complete


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
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Window settings
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Don't show in taskbar
        )
        
        # Size
        self.setFixedSize(400, 300)
        
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
        layout.addWidget(title)
        
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
        layout.addWidget(self.text_edit)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 11px;
                padding: 3px;
            }
        """)
        layout.addWidget(self.status_label)
        
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
        """Show the panel and start auto-scroll."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.scroll_timer.start(100)  # Scroll check every 100ms
        self.status_label.setText("Recording...")
    
    def hide_panel(self) -> None:
        """Hide the panel."""
        self.scroll_timer.stop()
        self.hide()
    
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
    
    def update_segment(self, text: str, confidence: int, segment_index: int, is_final: bool = False, phrase_start: bool = False) -> None:
        """
        Update a single segment. Each segment is part of a phrase (line).
        
        Args:
            text: Transcribed text for this segment
            confidence: Confidence score (0-100)
            segment_index: Position of this segment in the current phrase
            is_final: If True, this phrase is complete
            phrase_start: If True, start a new phrase (new line)
        """
        print(f"DEBUG Panel: update_segment text='{text}', idx={segment_index}, phrase_start={phrase_start}")
        
        # Start new phrase if needed (BEFORE checking for blank audio)
        if phrase_start or self.current_phrase_idx < 0:
            print(f"DEBUG Panel: Starting NEW PHRASE")
            self.phrases.append(Phrase(segments=[], confidences=[], is_final=False))
            self.current_phrase_idx = len(self.phrases) - 1
        
        # Skip [BLANK_AUDIO] after creating the phrase structure
        if text.strip() == "[BLANK_AUDIO]":
            print(f"DEBUG Panel: Skipping [BLANK_AUDIO] but phrase {self.current_phrase_idx} exists")
            return
        
        # Get current phrase
        phrase = self.phrases[self.current_phrase_idx]
        phrase.is_final = is_final
        
        # Update or add segment
        if segment_index < len(phrase.segments):
            # Update existing segment
            phrase.segments[segment_index] = text
            phrase.confidences[segment_index] = confidence
            print(f"DEBUG Panel: Updated segment {segment_index}: '{text}' [conf: {confidence}%]")
        else:
            # Add new segment
            phrase.segments.append(text)
            phrase.confidences.append(confidence)
            print(f"DEBUG Panel: Added segment {segment_index}: '{text}' [conf: {confidence}%]")
        
        # Rebuild the entire display
        self._rebuild_display()
        
        # Update status
        total_segments = sum(len(p.segments) for p in self.phrases)
        self.status_label.setText(f"Phrases: {len(self.phrases)} | Segments: {total_segments}")
        
        # Auto-scroll
        self._scroll_to_bottom()
    
    def _rebuild_display(self) -> None:
        """Rebuild the entire transcript display from phrases."""
        self.text_edit.clear()
        cursor = self.text_edit.textCursor()
        
        for phrase_idx, phrase in enumerate(self.phrases):
            # Build the phrase text from segments
            for seg_idx, (text, conf) in enumerate(zip(phrase.segments, phrase.confidences)):
                # Determine color based on confidence
                color = self._get_confidence_color(conf)
                
                # Create format
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(color))
                fmt.setFontWeight(QFont.Weight.Bold if conf >= 80 else QFont.Weight.Normal)
                
                # Insert text
                cursor.insertText(text, fmt)
                
                # Add space between segments (but not after last)
                if seg_idx < len(phrase.segments) - 1:
                    cursor.insertText(" ")
            
            # Add newline after phrase (but not after last)
            if phrase_idx < len(self.phrases) - 1:
                cursor.insertBlock()
        
        print(f"DEBUG Panel: Rebuilt display with {len(self.phrases)} phrases")
    
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
    
    def _scroll_to_bottom(self) -> None:
        """Auto-scroll to show latest text."""
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
        self.setFixedSize(300, 400)
        
        # Style
        self.setStyleSheet("""
            FloatingSettingsPanel {
                background-color: #1a1a1a;
                border: 2px solid #444;
                border-radius: 10px;
            }
            QLabel {
                color: #fff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #333;
                color: #fff;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:pressed {
                background-color: #4CAF50;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #4CAF50;")
        layout.addWidget(title)
        
        # Model selection
        model_label = QLabel("Model Size:")
        layout.addWidget(model_label)
        
        from PyQt6.QtWidgets import QButtonGroup, QVBoxLayout as VBox
        
        self.model_group = QButtonGroup(self)
        models = [("tiny", "Tiny (fastest)"), ("base", "Base (balanced)"), ("small", "Small (accurate)")]
        
        for model_id, model_name in models:
            from PyQt6.QtWidgets import QRadioButton
            btn = QRadioButton(model_name)
            btn.setStyleSheet("color: #fff;")
            self.model_group.addButton(btn)
            layout.addWidget(btn)
            if model_id == "tiny":
                btn.setChecked(True)
        
        layout.addStretch()
        
        # Close button
        from PyQt6.QtWidgets import QPushButton
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide_panel)
        layout.addWidget(close_btn)
        
        # Dragging
        self._dragging = False
        self._drag_pos = None
    
    def show_panel(self):
        """Show the panel."""
        self.show()
        self.raise_()
        self.activateWindow()
    
    def hide_panel(self):
        """Hide the panel."""
        self.hide()
    
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
