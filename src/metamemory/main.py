"""
metamemory - Windows Desktop Audio Transcription Widget
Main application entry point.
"""

import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from metamemory.widgets.main_widget import MeetAndReadWidget
from metamemory.audio import has_partial_recordings, recover_part_files, get_recordings_dir


def check_and_offer_recovery(parent=None):
    """Check for partial recordings and offer to recover them.
    
    Args:
        parent: Parent widget for message boxes
    
    Returns:
        Tuple of (recovered_count, declined) where declined is True if user
        chose not to recover.
    """
    recordings_dir = get_recordings_dir()
    
    if not has_partial_recordings(recordings_dir):
        return 0, False
    
    # Show recovery offer dialog
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle("Recover Recordings")
    msg_box.setText("Unsaved recordings found")
    msg_box.setInformativeText(
        "Some recordings were not properly saved from a previous session. "
        "Would you like to recover them now?"
    )
    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
    msg_box.setIcon(QMessageBox.Icon.Question)
    
    reply = msg_box.exec()
    
    if reply == QMessageBox.StandardButton.No:
        return 0, True
    
    # Show progress dialog while recovering
    progress_msg = QMessageBox(parent)
    progress_msg.setWindowTitle("Recovering...")
    progress_msg.setText("Recovering partial recordings...")
    progress_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
    progress_msg.setIcon(QMessageBox.Icon.Information)
    progress_msg.show()
    
    # Process events to show the dialog
    QApplication.processEvents()
    
    # Do recovery in a thread to avoid blocking
    recovered_files = []
    recovery_error = None
    
    def do_recovery():
        nonlocal recovered_files, recovery_error
        try:
            recovered_files = recover_part_files(
                recordings_dir=recordings_dir,
                delete_original=False,  # Safer default - backup originals
            )
        except Exception as e:
            recovery_error = str(e)
    
    # Run recovery (in thread for UI responsiveness, but wait for completion)
    recovery_thread = threading.Thread(target=do_recovery)
    recovery_thread.start()
    recovery_thread.join(timeout=30.0)  # Wait up to 30 seconds
    
    progress_msg.close()
    
    # Show result
    if recovery_error:
        error_msg = QMessageBox(parent)
        error_msg.setWindowTitle("Recovery Error")
        error_msg.setText("Some recordings could not be recovered")
        error_msg.setInformativeText(f"Error: {recovery_error}")
        error_msg.setIcon(QMessageBox.Icon.Warning)
        error_msg.exec()
    elif recovered_files:
        success_msg = QMessageBox(parent)
        success_msg.setWindowTitle("Recovery Complete")
        success_msg.setText(f"Recovered {len(recovered_files)} recording(s)")
        success_msg.setInformativeText(
            f"Recovered files are in:\n{recordings_dir}\n\n"
            f"Original files have been backed up with .recovered.bak extension."
        )
        success_msg.setIcon(QMessageBox.Icon.Information)
        success_msg.exec()
    else:
        info_msg = QMessageBox(parent)
        info_msg.setWindowTitle("No Files Recovered")
        info_msg.setText("No recordings could be recovered")
        info_msg.setIcon(QMessageBox.Icon.Information)
        info_msg.exec()
    
    return len(recovered_files), False


def main():
    """Application entry point."""
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("metamemory")
    app.setApplicationDisplayName("metamemory")
    
    # Check for partial recordings and offer recovery before showing widget
    # This runs synchronously before the main event loop
    try:
        check_and_offer_recovery(parent=None)
    except Exception as e:
        # Log error but don't block startup
        print(f"Recovery check failed: {e}")
    
    # Create and show the main widget
    widget = MeetAndReadWidget()
    widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
