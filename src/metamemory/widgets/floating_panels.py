"""
Floating transcript panel - separate window that docks to the main widget.

This solves the clipping issue by making the panel a separate QWidget
that floats outside the main widget bounds.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel, QFrame, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from typing import List, Optional
from dataclasses import dataclass

from metamemory.hardware.detector import HardwareDetector
from metamemory.hardware.recommender import ModelRecommender


@dataclass
class Phrase:
    """A phrase (line) of transcript with its segments and enhancement status."""
    segments: List[str]  # Text of each segment
    confidences: List[int]  # Confidence of each segment
    is_final: bool  # True if phrase is complete
    enhanced: List[bool]  # Whether each segment was enhanced


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
    segment_ready = pyqtSignal(str, int, int, bool, bool, bool)  # text, confidence, segment_index, is_final, phrase_start, enhanced
    
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

        # Enhancement status bar
        self.enhancement_status_label = QLabel("Enhancement: Idle")
        self.enhancement_status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 10px;
                padding: 3px;
                background-color: #222;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.enhancement_status_label)

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
        self.enhancement_status_label.setText("Enhancement: Idle")

    def update_enhancement_status(self, queue_size: int, workers_active: int, total_enhanced: int) -> None:
        """
        Update the enhancement status display.

        Args:
            queue_size: Number of segments currently in enhancement queue
            workers_active: Number of workers currently processing
            total_enhanced: Total number of segments enhanced
        """
        status = f"Enhancement: Queue {queue_size} | Workers {workers_active} | Enhanced {total_enhanced}"
        self.enhancement_status_label.setText(status)
    
    def update_segment(self, text: str, confidence: int, segment_index: int, is_final: bool = False, phrase_start: bool = False, enhanced: bool = False) -> None:
        """
        Update a single segment. Each segment is part of a phrase (line).

        Args:
            text: Transcribed text for this segment
            confidence: Confidence score (0-100)
            segment_index: Position of this segment in current phrase
            is_final: If True, this phrase is complete
            phrase_start: If True, start a new phrase (new line)
            enhanced: If True, this segment was enhanced by background processor
        """
        print(f"DEBUG Panel: update_segment text='{text}', idx={segment_index}, phrase_start={phrase_start}, enhanced={enhanced}")
        
        # Skip [BLANK_AUDIO] after creating a phrase structure
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
            
            self.phrases.append(Phrase(segments=[], confidences=[], is_final=False, enhanced=[]))
            self.current_phrase_idx = len(self.phrases) - 1
        
        # Get current phrase
        phrase = self.phrases[self.current_phrase_idx]
        phrase.is_final = is_final
        
        # Update or add segment to internal tracking
        if segment_index < len(phrase.segments):
            # Update existing segment - REPLACE IN PLACE
            phrase.segments[segment_index] = text
            phrase.confidences[segment_index] = confidence
            phrase.enhanced[segment_index] = enhanced
            print(f"DEBUG Panel: Updated segment {segment_index}: '{text}' [conf: {confidence}%] enhanced={enhanced}")
            
            # Find and replace just this segment in display
            self._replace_segment_in_display(self.current_phrase_idx, segment_index, text, confidence, enhanced)
        else:
            # Add new segment - APPEND TO CURRENT LINE
            phrase.segments.append(text)
            phrase.confidences.append(confidence)
            phrase.enhanced.append(enhanced)
            print(f"DEBUG Panel: Added segment {segment_index}: '{text}' [conf: {confidence}%] enhanced={enhanced}")
            
            # Append segment to display with proper formatting
            self._append_segment_to_display(text, confidence, enhanced)
        
        # Update status
        total_segments = sum(len(p.segments) for p in self.phrases)
        self.status_label.setText(f"Phrases: {len(self.phrases)} | Segments: {total_segments}")
        
        # Auto-scroll
        self._scroll_to_bottom()
    
    def _append_segment_to_display(self, text: str, confidence: int, enhanced: bool) -> None:
        """Append a segment to the current line with proper formatting."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add space between segments
        if self.phrases[self.current_phrase_idx].segments and len(self.phrases[self.current_phrase_idx].segments) > 0:
            cursor.insertText(" ")
        
        # Determine color and formatting
        color = self._get_confidence_color(confidence)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        # ONLY bold if enhanced, not based on confidence
        fmt.setFontWeight(QFont.Weight.Bold if enhanced else QFont.Weight.Normal)
        
        cursor.insertText(text, fmt)
    
    def _replace_segment_in_display(self, phrase_idx: int, segment_idx: int, text: str, confidence: int, enhanced: bool) -> None:
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
        # ONLY bold if enhanced
        fmt.setFontWeight(QFont.Weight.Bold if enhanced else QFont.Weight.Normal)
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
    enhancement_settings_changed = pyqtSignal(dict)  # Emit enhancement settings dict when changed
    
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

        # Enhancement settings section
        enhancement_label = QLabel("Enhancement Settings:")
        enhancement_label.setStyleSheet("font-weight: bold; color: #4CAF50; margin-top: 10px;")
        layout.addWidget(enhancement_label)

        # Confidence threshold slider
        threshold_label = QLabel("Confidence Threshold:")
        layout.addWidget(threshold_label)

        from PyQt6.QtWidgets import QSlider, QLabel as QLabel2
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)  # 50% to 95%
        self.threshold_slider.setValue(70)  # Default: 70%
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #fff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
        """)
        self.threshold_value_label = QLabel2("70%")
        self.threshold_value_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        layout.addLayout(threshold_layout)

        # Number of workers slider
        workers_label = QLabel("Enhancement Workers:")
        layout.addWidget(workers_label)

        self.workers_slider = QSlider(Qt.Orientation.Horizontal)
        self.workers_slider.setRange(1, 8)  # 1 to 8 workers
        self.workers_slider.setValue(4)  # Default: 4
        self.workers_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #fff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
        """)
        self.workers_value_label = QLabel2("4")
        self.workers_value_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        workers_layout = QHBoxLayout()
        workers_layout.addWidget(self.workers_slider)
        workers_layout.addWidget(self.workers_value_label)
        layout.addLayout(workers_layout)

        # Connect slider signals
        self.threshold_slider.valueChanged.connect(
            lambda value: (
                self.threshold_value_label.setText(f"{value}%"),
                self._emit_enhancement_settings()
            )
        )
        self.workers_slider.valueChanged.connect(
            lambda value: (
                self.workers_value_label.setText(str(value)),
                self._emit_enhancement_settings()
            )
        )

        # Enhancement model selection
        enhancement_model_label = QLabel("Enhancement Model:")
        layout.addWidget(enhancement_model_label)

        self.enhancement_model_group = QButtonGroup(self)
        enhancement_models = [("small", "Small"), ("medium", "Medium"), ("large", "Large")]

        for model_id, model_name in enhancement_models:
            from PyQt6.QtWidgets import QRadioButton
            btn = QRadioButton(model_name)
            btn.setStyleSheet("color: #fff;")
            self.enhancement_model_group.addButton(btn)
            layout.addWidget(btn)

            if model_id == "medium":
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

    def _emit_enhancement_settings(self) -> None:
        """Emit current enhancement settings."""
        settings = {
            'confidence_threshold': self.threshold_slider.value() / 100.0,  # Convert to 0-1 range
            'num_workers': self.workers_slider.value(),
        }
        self.enhancement_settings_changed.emit(settings)

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
    panel.update_segment("Hello", 85, 0, is_final=False, enhanced=False)
    panel.update_segment("world", 90, 1, is_final=False, enhanced=False)
    panel.update_segment("this is", 75, 2, is_final=True, enhanced=False)
    
    # New phrase
    panel.update_segment("New phrase", 80, 0, phrase_start=True, is_final=False, enhanced=False)
    panel.update_segment("here", 85, 1, is_final=True, enhanced=False)
    
    sys.exit(app.exec())
