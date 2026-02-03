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
class TranscriptLine:
    """A line of transcript text with metadata."""
    text: str
    confidence: int
    timestamp: float
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
        
        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 230);
                border: 2px solid rgba(100, 100, 100, 200);
                border-radius: 10px;
            }
            QTextEdit {
                background-color: transparent;
                color: white;
                border: none;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                padding: 10px;
            }
            QLabel {
                color: #aaaaaa;
                font-size: 12px;
                padding: 5px 10px;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header
        self.header = QLabel("📝 Live Transcript")
        self.header.setStyleSheet("font-weight: bold; color: white;")
        layout.addWidget(self.header)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: rgba(100, 100, 100, 100);")
        layout.addWidget(separator)
        
        # Text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        
        # Status
        self.status_label = QLabel("Waiting for audio...")
        layout.addWidget(self.status_label)
        
        # Track lines
        self.lines: List[TranscriptLine] = []
        self.current_line_idx = -1
        
        # Auto-scroll timer
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self._scroll_to_bottom)
    
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
        self.lines.clear()
        self.current_line_idx = -1
    
    def update_line(self, text: str, confidence: int, is_final: bool = False) -> None:
        """
        Update the current line (edit in place) or add new line.
        
        This matches the reference implementation pattern:
        - Edit current line while phrase is ongoing
        - Start new line when phrase is complete
        
        Args:
            text: Transcribed text
            confidence: Confidence score (0-100)
            is_final: If True, this phrase is complete, start new line next time
        """
        # Skip if same text as current line and not final (prevents flicker/duplicates)
        if (self.current_line_idx >= 0 and 
            self.lines and 
            len(self.lines) > self.current_line_idx and
            self.lines[self.current_line_idx].text == text and 
            not is_final):
            return  # Duplicate, skip
        
        # Determine color based on confidence
        color = self._get_confidence_color(confidence)
        
        # Create format
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Bold if confidence >= 80 else QFont.Weight.Normal)
        
        cursor = self.text_edit.textCursor()
        
        if is_final or self.current_line_idx < 0:
            # Start new line
            if self.current_line_idx >= 0:
                cursor.insertBlock()  # New paragraph
            
            self.current_line_idx += 1
            cursor.insertText(text, fmt)
        else:
            # Edit current line (replace it)
            # Select current block
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            for _ in range(self.current_line_idx):
                cursor.movePosition(QTextCursor.MoveOperation.NextBlock)
            
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(text, fmt)
        
        # Update status
        self.status_label.setText(f"Confidence: {confidence}% | Words: {len(self.lines)}")
        
        # Track
        line = TranscriptLine(
            text=text,
            confidence=confidence,
            timestamp=0,  # Would track actual time
            is_final=is_final
        )
        if len(self.lines) <= self.current_line_idx:
            self.lines.append(line)
        else:
            self.lines[self.current_line_idx] = line
    
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
            for i, line in enumerate(self.lines):
                f.write(f"{i+1}. [{line.confidence}%] {line.text}\n")
    
    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.closed.emit()
        event.accept()


# Settings panel (similar floating approach)
class FloatingSettingsPanel(QWidget):
    """Floating settings panel for model selection."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        self.setFixedSize(300, 200)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 240);
                border: 2px solid rgba(100, 100, 100, 200);
                border-radius: 10px;
            }
            QLabel {
                color: white;
                font-size: 12px;
                padding: 5px;
            }
            QComboBox {
                background-color: rgba(50, 50, 50, 200);
                color: white;
                border: 1px solid rgba(100, 100, 100, 200);
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(50, 50, 50, 240);
                color: white;
                selection-background-color: rgba(100, 100, 100, 200);
            }
        """)
        
        from PyQt6.QtWidgets import QVBoxLayout, QComboBox, QLabel
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("⚙️ Settings")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Model selection
        layout.addWidget(QLabel("Transcription Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small"])
        self.model_combo.setCurrentText("tiny")
        layout.addWidget(self.model_combo)
        
        # Info text
        info = QLabel("Tiny: Fastest, lowest accuracy\nBase: Balanced\nSmall: Best accuracy, slower")
        info.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(info)
        
        layout.addStretch()
    
    def dock_to_widget(self, widget: QWidget, position: str = "right") -> None:
        """Position panel next to widget."""
        widget_pos = widget.mapToGlobal(widget.rect().topLeft())
        widget_rect = widget.geometry()
        
        if position == "left":
            x = widget_pos.x() - self.width() - 10
            y = widget_pos.y()
        else:
            x = widget_pos.x() + widget_rect.width() + 10
            y = widget_pos.y()
        
        self.move(x, y)
    
    def show_panel(self) -> None:
        self.show()
        self.raise_()
    
    def hide_panel(self) -> None:
        self.hide()
    
    def get_selected_model(self) -> str:
        return self.model_combo.currentText()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QPushButton
    
    app = QApplication(sys.argv)
    
    # Create main button (simulating the widget)
    button = QPushButton("🎤 Record")
    button.setFixedSize(100, 50)
    button.move(500, 400)
    button.show()
    
    # Create floating transcript panel
    panel = FloatingTranscriptPanel()
    panel.dock_to_widget(button, "left")
    
    # Simulate transcription
    def add_text():
        panel.update_line("Hello this is a test", 85, is_final=False)
    
    def finalize_text():
        panel.update_line("Hello this is a test phrase", 82, is_final=True)
    
    # Add buttons to simulate
    from PyQt6.QtWidgets import QVBoxLayout
    layout = QVBoxLayout()
    
    btn_add = QPushButton("Add Text")
    btn_add.clicked.connect(add_text)
    btn_add.show()
    btn_add.move(500, 460)
    
    btn_final = QPushButton("Finalize")
    btn_final.clicked.connect(finalize_text)
    btn_final.show()
    btn_final.move(500, 490)
    
    # Show panel
    panel.show_panel()
    
    sys.exit(app.exec())
