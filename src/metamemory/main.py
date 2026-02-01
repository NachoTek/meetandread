"""
metamemory - Windows Desktop Audio Transcription Widget
Main application entry point.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from metamemory.widgets.main_widget import MeetAndReadWidget


def main():
    """Application entry point."""
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("metamemory")
    app.setApplicationDisplayName("metamemory")
    
    # Create and show the main widget
    widget = MeetAndReadWidget()
    widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()