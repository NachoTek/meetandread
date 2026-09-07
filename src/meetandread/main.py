"""
meetandread - Windows Desktop Audio Transcription Widget
Main application entry point.
"""

import sys
import queue
import threading
import logging
import signal
from pathlib import Path
from typing import List, NamedTuple, Optional
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from meetandread.widgets.main_widget import MeetAndReadWidget
from meetandread.audio import has_partial_recordings, recover_part_files, get_recordings_dir
from meetandread.config import get_config
from meetandread.hardware.recommender import ModelRecommender
from meetandread.logging_setup import configure_logging


class _RecoveryResult(NamedTuple):
    """Immutable envelope for recovery worker outcome."""
    recovered_paths: List[Path]
    error: Optional[str]


logger = logging.getLogger(__name__)


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


def setup_logging(capture_mode: bool = False, capture_dir: Optional[Path] = None):
    """Setup logging: root level INFO normally, full DEBUG in capture mode.

    Thin startup hook over ``meetandread.logging_setup`` (issue #98):
    level selection is all-or-nothing — INFO for a normal run, app-wide
    DEBUG when the Issue Reporter launches the app in capture mode.
    Keeps the per-run timestamped log file under Documents/meetandread/logs
    and the stdout console-mirroring tee; also runs the 30-day retention
    cleanup for normal-run logs (capture logs are never touched).

    Capture mode (issue #104): when *capture_dir* is given (entry at
    process start only, via ``--issue-capture``), the capture log
    streams into the capture directory itself under the
    prompt-flush durability contract, and retention never scans inside
    a capture directory.
    """
    if capture_dir is not None:
        from meetandread.capture_mode import configure_capture_logging

        return configure_capture_logging(capture_dir)
    return configure_logging(capture_mode=capture_mode)


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
    
    # Thread-safe result handoff via queue — no shared mutable state
    result_queue: queue.Queue[_RecoveryResult] = queue.Queue(maxsize=1)
    
    def do_recovery():
        try:
            recovered = recover_part_files(
                recordings_dir=recordings_dir,
                delete_original=False,  # Safer default - backup originals
            )
            result_queue.put(_RecoveryResult(recovered_paths=recovered, error=None))
        except Exception as e:
            result_queue.put(_RecoveryResult(recovered_paths=[], error=str(e)))
    
    # Run recovery in a thread to keep UI responsive
    recovery_thread = threading.Thread(target=do_recovery, daemon=True)
    recovery_thread.start()
    
    # Wait up to 30 seconds for the result envelope
    try:
        result = result_queue.get(timeout=30.0)
        recovered_files = result.recovered_paths
        recovery_error = result.error
        timed_out = False
    except queue.Empty:
        recovered_files = []
        recovery_error = None
        timed_out = True
    
    progress_msg.close()
    
    # Show result — three explicit outcomes: error, timeout, success/empty
    if recovery_error:
        error_msg = QMessageBox(parent)
        error_msg.setWindowTitle("Recovery Error")
        error_msg.setText("Some recordings could not be recovered")
        error_msg.setInformativeText(f"Error: {recovery_error}")
        error_msg.setIcon(QMessageBox.Icon.Warning)
        error_msg.exec()
    elif timed_out:
        timeout_msg = QMessageBox(parent)
        timeout_msg.setWindowTitle("Recovery Timed Out")
        timeout_msg.setText("Recording recovery is still in progress")
        timeout_msg.setInformativeText(
            "Recovery did not complete within 30 seconds. "
            "Partial recordings have been preserved and can be "
            "recovered on the next startup."
        )
        timeout_msg.setIcon(QMessageBox.Icon.Warning)
        timeout_msg.exec()
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
        
        from meetandread.hardware.detector import HardwareDetector
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


def setup_signal_handlers(app, widget_ref=None):
    """Setup signal handlers for graceful shutdown.

    When a widget reference is available, signal handlers invoke the
    widget's graceful exit path (controller shutdown).  Without a widget
    (e.g. signal arrives before widget is created), falls back to
    app.quit().

    Args:
        app: QApplication instance to quit on signal.
        widget_ref: Optional callable that returns the MeetAndReadWidget,
            or None.  Using a callable avoids referencing a widget that
            may not exist yet at signal-registration time.
    """
    def _graceful_exit():
        """Attempt graceful controller shutdown, then quit."""
        widget = None
        if widget_ref is not None:
            try:
                widget = widget_ref()
            except Exception:
                logger.debug("widget_ref() failed during shutdown", exc_info=True)
        if widget is not None:
            try:
                widget._exit_application()
            except Exception:
                logger.debug(
                    "_exit_application failed in signal handler, "
                    "falling back to app.quit()"
                )
                app.quit()
        else:
            app.quit()

    def _make_signal_handler(signal_name: str):
        """Build a graceful-exit handler for a named termination signal."""
        def handler(signum, frame):
            logger.info("Received %s, shutting down gracefully...", signal_name)
            _graceful_exit()
        return handler

    # Register SIGINT handler
    signal.signal(signal.SIGINT, _make_signal_handler("SIGINT"))

    # On Windows, also handle SIGBREAK (Ctrl+Break) through the same
    # graceful path. The Issue Reporter (#107) uses it for user-stop of
    # a capture run, and the capture-mode subprocess tests use it to
    # drive a genuine clean exit (marker contract, issue #104).
    if sys.platform == 'win32' and hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, _make_signal_handler("SIGBREAK"))

    # On Windows, also set up console control handler for Ctrl+C
    if sys.platform == 'win32':
        try:
            import win32api

            def win_handler(dwCtrlType):
                if dwCtrlType == 0:  # CTRL_C_EVENT
                    logger.info("Received CTRL+C event, shutting down gracefully...")
                    _graceful_exit()
                    return True
                return False
            win32api.SetConsoleCtrlHandler(win_handler, True)
        except ImportError:
            # win32api not available, SIGINT handler should still work
            pass


def main(capture_dir: Optional[Path] = None):
    """Application entry point.

    *capture_dir* is the already-parsed ``--issue-capture`` directory
    when the process started through the lightweight bootstrap
    (``meetandread.__main__``), which parsed the flag and configured
    capture logging BEFORE importing this module; None means parse the
    flag from the command line here (console-script entry point, or a
    direct ``main()`` call — the pre-bootstrap behavior).
    """
    # Issue Capture Mode entry (issue #104): the launch flag is parsed
    # BEFORE any other startup step, so a capture run owns the whole
    # process — logging included — from the first moment. Entry is at
    # process start only; there is deliberately no mid-process capture
    # API (ADR 0003).
    from meetandread.capture_mode import (
        ISSUE_CAPTURE_FLAG,
        CaptureModeError,
        capture_logging_configured,
        parse_capture_flag,
        write_completion_marker,
    )

    if capture_dir is None:
        try:
            capture_dir = parse_capture_flag()
        except CaptureModeError as exc:
            # Pre-logging path: the root logger's lastResort handler
            # routes ERROR to stderr (same as the single-instance
            # refusal below).
            logger.error("%s %s", ISSUE_CAPTURE_FLAG, exc)
            sys.exit(2)

    # Single-instance guard (issue #20): must run before QApplication so a
    # duplicate process exits before creating any UI or grabbing resources.
    from meetandread.single_instance import acquire_single_instance_lock

    if not acquire_single_instance_lock():
        # setup_logging() may not have run yet (bootstrap-configured
        # capture logging IS already live, per-record flushed); the root
        # logger or lastResort handler routes ERROR to stderr, so the
        # diagnostic path is preserved either way.
        logger.error(
            "meetandread is already running (or the single-instance lock could not be acquired) — exiting this instance."
        )
        sys.exit(1)

    # Setup logging first — capture mode (if requested) streams DEBUG into
    # the capture directory from this point on. Idempotence (fix round 1):
    # when the bootstrap already configured capture logging (a
    # _PromptFlushFileHandler is installed), do NOT reconfigure — a second
    # configure would tear down the run handler mid-stream and refuse the
    # same-second exclusive filename.
    if capture_dir is None or not capture_logging_configured():
        setup_logging(capture_dir=capture_dir)
    logging.getLogger(__name__).info(
        "Starting meetandread%s",
        " (Issue Capture Mode)" if capture_dir is not None else "",
    )
    
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("meetandread")
    app.setApplicationDisplayName("meetandread")

    # Widget placeholder — updated after widget creation.
    # Signal handlers read this via a lambda so they always get the
    # current widget (or None if the signal arrives too early).
    _widget_holder = [None]

    # Setup signal handlers for graceful Ctrl+C shutdown
    setup_signal_handlers(app, widget_ref=lambda: _widget_holder[0])
    
    # Check critical native DLLs are loadable (frozen exe only)
    check_critical_dlls()

    # Tier-2 feature dependencies (issue #61) — non-blocking degraded mode.
    # Runs before any Post-processing start so the Stalled/requeue scan
    # sees repaired dependencies; results are cached for the banner,
    # Diagnostics, and repair conversion.
    try:
        from meetandread.dependencies import unresolved_dependencies

        for status in unresolved_dependencies():
            logger.warning(
                "Optional dependency '%s' missing — %s degraded: %s",
                status.dependency.name,
                status.dependency.feature,
                status.dependency.resolution_text(),
            )
    except Exception as e:
        logger.warning("Feature dependency check failed: %s", e)
    
    # Run hardware detection on first startup (if auto-detect enabled)
    try:
        settings = get_config()
        if settings.hardware.auto_detect_on_startup and not settings.hardware.recommended_model:
            logger.info("Running hardware detection...")
            recommender = ModelRecommender()
            recommended = recommender.detect_and_recommend()
            specs = recommender.get_detected_specs()
            logger.info("RAM: %.1f GB, CPU: %d cores, Recommended model: %s",
                        specs.total_ram_gb, specs.cpu_count_logical, recommended)
    except Exception as e:
        # Log error but don't block startup
        logger.warning("Hardware detection failed: %s", e)
    
    # Check hardware requirements and warn if below minimum
    try:
        check_hardware_requirements()
    except Exception as e:
        logger.warning("Hardware requirements check failed: %s", e)
    
    # Check for partial recordings and offer recovery before showing widget
    # This runs synchronously before the main event loop
    try:
        check_and_offer_recovery(parent=None)
    except Exception as e:
        # Log error but don't block startup
        logger.warning("Recovery check failed: %s", e)

    # Process pending cleanup queue (orphaned files from prior sessions)
    try:
        from meetandread.recording.cleanup_queue import CleanupQueue
        cleanup_queue = CleanupQueue()
        result = cleanup_queue.process_pending()
        if result.processed > 0 or result.failed > 0:
            logging.info(
                "Startup cleanup: processed=%d, failed=%d",
                result.processed, result.failed,
            )
    except Exception as e:
        logging.warning("Startup cleanup queue processing failed: %s", e)
    
    # Create and show the main widget
    widget = MeetAndReadWidget()
    _widget_holder[0] = widget  # Enable signal handlers to reach the widget

    # Start Post-processing at app startup: pending-job recovery,
    # dependency-repair conversion, and the Stalled requeue scan run
    # while the app idles instead of waiting for the first record-start
    # (issues #61/#62 — 'Queued' rows must process automatically).
    try:
        widget.initialize_post_processing()
    except Exception as e:
        logger.warning("Post-processing startup initialization failed: %s", e)
    
    # Create and wire system tray icon manager
    from meetandread.widgets.tray_icon import TrayIconManager
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

    # Degraded-mode banner (issue #61): after the widget is visible so
    # the toast anchors against the shown widget.
    try:
        widget.maybe_show_dependency_banner()
    except Exception as e:
        logger.warning("Dependency banner failed: %s", e)

    # Clean-exit completion marker (issue #104): written only when the
    # event loop returns normally with exit code 0 — the marker is the
    # absence-proof of a crash ("no marker == crashed run", by
    # construction). Any hard exit (kill, native crash, sys.exit from a
    # dialog path) skips this point entirely.
    exit_code = app.exec()
    if capture_dir is not None and exit_code == 0:
        write_completion_marker(capture_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
