"""
metamemory - Windows Desktop Audio Transcription Widget
Main application entry point.
"""

import sys
import threading
import logging
import signal
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from metamemory.widgets.main_widget import MeetAndReadWidget
from metamemory.audio import has_partial_recordings, recover_part_files, get_recordings_dir
from metamemory.config import get_config
from metamemory.hardware.recommender import ModelRecommender


def check_critical_dlls():
    """Check that critical native DLLs can be loaded in frozen exe mode.

    Only runs when ``sys.frozen`` is True (i.e. inside a PyInstaller bundle).
    In development mode missing DLLs surface naturally as ImportError at import
    time, so the check is skipped.

    Each library is tested in its own try/except so the user sees *which*
    library failed, not a generic "something is missing" message.
    """
    if not getattr(sys, 'frozen', False):
        return

    critical_libs = [
        ('pywhispercpp', 'pywhispercpp'),
        ('sounddevice', 'sounddevice'),
    ]

    for name, module in critical_libs:
        try:
            __import__(module)
        except ImportError as exc:
            msg = (
                f"Required library '{name}' could not be loaded.\n\n"
                f"Error details: {exc}\n\n"
                f"The application cannot start without this component. "
                f"Please reinstall meetandread."
            )
            logging.error(f"DLL check failed for {name}: {exc}")
            QMessageBox.critical(
                None,
                "meetandread — Missing Component",
                msg,
            )
            sys.exit(1)


class TeeOutput:
    """Redirects stdout to both console and log file."""
    def __init__(self, logger):
        self.logger = logger
        self.stdout = sys.stdout
        
    def write(self, message):
        if self.stdout is not None:
            self.stdout.write(message)
            self.stdout.flush()
        if message.strip():
            self.logger.debug(message.rstrip())
    
    def flush(self):
        if self.stdout is not None:
            self.stdout.flush()


def setup_logging():
    """Setup logging to both console and file with timestamped filename."""
    # Determine logs directory: under user Documents for both frozen and dev
    logs_dir = Path.home() / "Documents" / "metamemory" / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"metamemory_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    
    # Redirect stdout to both console and file
    sys.stdout = TeeOutput(logging.getLogger())
    
    print(f"Logging to: {log_file}")
    print(f"Logs directory: {logs_dir}")
    return log_file


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


def check_hardware_requirements():
    """Check if the system meets minimum hardware requirements.
    
    Shows a warning dialog if the system is below minimum specs.
    The dialog is informational only — the app still starts.
    Only runs when auto_detect_on_startup is enabled in settings.
    Wrapped in try/except so it never blocks startup.
    """
    try:
        settings = get_config()
        if not settings.hardware.auto_detect_on_startup:
            return
        
        from metamemory.hardware.detector import HardwareDetector
        detector = HardwareDetector()
        specs = detector.detect()
        
        if not detector.has_minimum_requirements(specs, dual_mode=False):
            warning_msg = detector.get_warning_message(specs, dual_mode=False)
            
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Hardware Notice")
            msg_box.setText("Your system may not meet minimum requirements")
            msg_box.setInformativeText(warning_msg or "System resources may be insufficient for reliable recording.")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
            logging.info(f"Hardware warning shown: {warning_msg}")
    except Exception as e:
        logging.warning(f"Hardware requirements check failed: {e}")


def setup_signal_handlers(app):
    """Setup signal handlers for graceful shutdown.
    
    Args:
        app: QApplication instance to quit on signal
    """
    def sigint_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) gracefully."""
        print("\nReceived SIGINT, shutting down gracefully...")
        app.quit()
    
    # Register SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)
    
    # On Windows, also set up console control handler for Ctrl+C
    if sys.platform == 'win32':
        try:
            import win32api
            def win_handler(dwCtrlType):
                if dwCtrlType == 0:  # CTRL_C_EVENT
                    print("\nReceived CTRL+C event, shutting down gracefully...")
                    app.quit()
                    return True
                return False
            win32api.SetConsoleCtrlHandler(win_handler, True)
        except ImportError:
            # win32api not available, SIGINT handler should still work
            pass


def main():
    """Application entry point."""
    # Setup logging first
    log_file = setup_logging()
    logging.info("Starting metamemory")
    
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("metamemory")
    app.setApplicationDisplayName("metamemory")
    
    # Setup signal handlers for graceful Ctrl+C shutdown
    setup_signal_handlers(app)
    
    # Check critical native DLLs are loadable (frozen exe only)
    check_critical_dlls()
    
    # Run hardware detection on first startup (if auto-detect enabled)
    try:
        settings = get_config()
        if settings.hardware.auto_detect_on_startup and not settings.hardware.recommended_model:
            print("Running hardware detection...")
            recommender = ModelRecommender()
            recommended = recommender.detect_and_recommend()
            specs = recommender.get_detected_specs()
            print(f"  RAM: {specs.total_ram_gb:.1f} GB")
            print(f"  CPU: {specs.cpu_count_logical} cores")
            print(f"  Recommended model: {recommended}")
            print()
    except Exception as e:
        # Log error but don't block startup
        print(f"Hardware detection failed: {e}")
    
    # Check hardware requirements and warn if below minimum
    try:
        check_hardware_requirements()
    except Exception as e:
        print(f"Hardware requirements check failed: {e}")
    
    # Check for partial recordings and offer recovery before showing widget
    # This runs synchronously before the main event loop
    try:
        check_and_offer_recovery(parent=None)
    except Exception as e:
        # Log error but don't block startup
        print(f"Recovery check failed: {e}")
    
    # Create and show the main widget
    widget = MeetAndReadWidget()
    
    # Create and wire system tray icon manager
    from metamemory.widgets.tray_icon import TrayIconManager
    tray = TrayIconManager(widget=widget)
    tray.set_callbacks(
        on_toggle_recording=widget.toggle_recording,
        on_exit=widget._exit_application,
    )
    widget._tray_manager = tray
    tray.show()
    
    # Do NOT quit when last window is hidden — tray keeps the app alive
    app.setQuitOnLastWindowClosed(False)
    
    logging.info("Tray icon created and wired to main widget")
    
    widget.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
