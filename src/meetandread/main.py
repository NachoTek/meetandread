"""
meetandread - Windows Desktop Audio Transcription Widget
Main application entry point.
"""

import os
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
            logging.getLogger(__name__).error(
                "critical_dll_check_failed: library=%s error_class=%s",
                name, type(exc).__name__,
            )
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


def check_and_offer_diagnostics_resume(
    parent=None,
    data_base: Optional[Path] = None,
    capture_mode: bool = False,
) -> Optional[Path]:
    """Detect an unsubmitted Diagnostics Bundle and offer to resume
    its submission (issue #110 — reporter-death resilience).

    If the Issue Reporter died mid-flow after a bundle exists on
    disk, the next normal application launch notices the
    assembled-but-unsubmitted bundle and offers to resume the
    submission — from the saved bundle, no re-reproduction — with a
    clear option to discard instead. Declining ("Not now") leaves
    normal startup otherwise unaffected: the bundle simply stays
    offerable on a later launch. Reporter startup remains the
    designated recovery path (ADR 0003, amended); this offer is the
    spec's optional "may also offer to resume" path.

    Resume REUSES the same review and Manual Submission
    implementation (no parallel submission path): the submission
    draft is built from the reviewed on-disk artifact exactly like
    the wizard's submit step, and the same IO seams (clipboard +
    browser) fire. Resolving the offer writes the submission-state
    record so it is never re-offered.

    Never raises: an internal failure logs and returns None — the
    offer must not break startup.
    """
    if capture_mode:
        return None
    try:
        from meetandread.manual_submission import (
            SubmissionState,
            find_unsubmitted_bundles,
            write_submission_state,
        )
        from meetandread.reporter_wizard import DEFAULT_DATA_BASE

        # The same env override the reporter program honors (#111
        # packaging / tests steer the data base): the app and the
        # reporter must agree on where capture runs live.
        env_base = os.environ.get("MAR_REPORTER_DATA_BASE")
        base = Path(
            data_base
            if data_base is not None
            else (Path(env_base) if env_base else DEFAULT_DATA_BASE)
        )
        pending = find_unsubmitted_bundles(base)
        if not pending:
            return None
        capture_dir = pending[0]

        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle("meetandread — Unsubmitted Issue Report")
        msg_box.setText("An issue report is waiting to be submitted")
        msg_box.setInformativeText(
            "A diagnostics capture run finished and its report was "
            "never submitted (the Issue Reporter closed before "
            "completion).\n\n"
            "Resume the submission now? You can also discard the "
            "report, or leave it for later."
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        msg_box.setIcon(QMessageBox.Icon.Question)

        reply = msg_box.exec()

        if reply == QMessageBox.StandardButton.Discard:
            write_submission_state(
                capture_dir, SubmissionState.DISCARDED
            )
            logger.info(
                "diagnostics_resume: action=discarded"
            )
            return None
        if reply != QMessageBox.StandardButton.Yes:
            logger.info("diagnostics_resume: action=postponed")
            return None

        # Resume: the same Manual Submission flow the wizard's
        # submit step runs, from the reviewed artifact on disk.
        import meetandread.manual_submission as ms

        draft = ms.build_submission(capture_dir)
        if isinstance(draft, ms.SubmissionUnavailable):
            # The artifact vanished since the scan (or review failed
            # closed on a prior resume): fail closed here too — no
            # browser, no clipboard, no state written.
            logger.warning(
                "diagnostics_resume_unavailable: detail=%s",
                draft.detail,
            )
            return None
        if ms.copy_to_clipboard(draft.clipboard_text):
            logger.info("diagnostics_resume: clipboard=path")
        if ms.open_new_issue_form(draft.url):
            write_submission_state(
                capture_dir, SubmissionState.SUBMITTED
            )
            logger.info("diagnostics_resume: action=submitted")
        else:
            QMessageBox.information(
                parent,
                "meetandread — Copy by hand",
                "Could not open a browser. Copy this address by "
                f"hand:\n\n{draft.url}",
            )
        return capture_dir
    except Exception as e:
        logger.warning(
            "diagnostics_resume_offer_failed: error_class=%s",
            type(e).__name__,
        )
        return None


def check_hardware_requirements():
    """Check if the system meets minimum hardware requirements.:
    
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

            logger.info("hardware_warning_shown: meets=0")
    except Exception as e:
        logger.warning(
            "hardware_requirements_check_failed: error_class=%s",
            type(e).__name__,
        )


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
            except Exception as exc:
                logger.debug(
                    "widget_ref_failed: error_class=%s",
                    type(exc).__name__,
                )
        if widget is not None:
            try:
                widget._exit_application()
            except Exception as exc:
                logger.debug(
                    "graceful_exit_failed: error_class=%s fallback=app_quit",
                    type(exc).__name__,
                )
                app.quit()
        else:
            app.quit()

    def _make_signal_handler(signal_name: str):
        """Build a graceful-exit handler for a named termination signal."""
        def handler(signum, frame):
            logger.info("signal_received: signal=%s action=graceful_exit", signal_name)
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
                    logger.info("signal_received: signal=CTRL_C action=graceful_exit")
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
            # refusal below). The exception message can embed the
            # user's capture path, so only the class travels (privacy
            # boundary at WARNING and above).
            logger.error(
                "%s rejected: error_class=%s",
                ISSUE_CAPTURE_FLAG, type(exc).__name__,
            )
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
            "single_instance_refused: exiting=1"
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
    # Named startup event: the suffix " (Issue Capture Mode)" is load-
    # bearing — the capture-mode subprocess tests wait for this exact
    # record to prove the startup sequence reached the logged stage.
    logging.getLogger(__name__).info(
        "app_startup: mode=%s banner=Starting meetandread%s",
        "issue_capture" if capture_dir is not None else "normal",
        " (Issue Capture Mode)" if capture_dir is not None else "",
    )

    # Interaction Trace (issue #105): capture runs only — installed at
    # process start, before the widget tree exists, so every semantic
    # user action from the first moment of the run lands in
    # <capture_dir>/interaction_trace.jsonl under the durability
    # contract. A normal run installs nothing and records no trace.
    # A conflicting install (a stale trace file from another run) is a
    # capture-mode startup failure: same refusal class as the #104
    # claim rejection (clear stderr, exit 2, no half-started run).
    if capture_dir is not None:
        from meetandread.interaction_trace import (
            InteractionTraceError,
            install_interaction_trace,
        )
        from meetandread.resource_snapshots import (
            SnapshotSeriesError,
            install_snapshot_series,
        )

        try:
            install_interaction_trace(capture_dir)
        except InteractionTraceError as exc:
            logger.error(
                "interaction_trace_install_failed: error_class=%s",
                type(exc).__name__,
            )
            sys.exit(2)
        # Resource Snapshot series (issue #106): same startup
        # discipline as the trace — the series FILE is created
        # exclusively at process start so a stale series from another
        # run is a capture-mode startup failure (exit 2), never a
        # half-started run writing into a foreign directory. The
        # driving ResourceMonitor starts below, once the QApplication
        # exists (its QTimer needs a Qt event loop host).
        try:
            install_snapshot_series(capture_dir)
        except SnapshotSeriesError as exc:
            logger.error(
                "snapshot_series_install_failed: error_class=%s",
                type(exc).__name__,
            )
            sys.exit(2)
    
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("meetandread")
    app.setApplicationDisplayName("meetandread")

    # Qt-side Interaction Trace wiring (issue #105): the application
    # filter (window focus changes, text-edit char counts, generic
    # buttons) installs only in a capture run — a normal run keeps a
    # zero-diagnostic widget path. Diagnostics are best-effort here: a
    # filter failure must not kill the diagnosed run.
    if capture_dir is not None:
        try:
            from meetandread.interaction_trace_qt import (
                install_interaction_trace_qt_filter,
            )

            install_interaction_trace_qt_filter(app)
        except Exception as e:
            logger.warning(
                "interaction_trace_qt_filter_failed: error_class=%s",
                type(e).__name__,
            )

        # Resource Snapshot series monitor (issue #106): start the
        # ResourceMonitor driving the series installed above. The
        # monitor needs a QTimer, so this runs after the QApplication
        # exists; its first poll fires immediately on start, landing
        # the series' first record at the first moment of the run.
        # Best-effort (a diagnostics failure must not kill the
        # diagnosed run): the capture DEBUG log and trace still cover
        # the run if the monitor cannot start.
        try:
            from meetandread.resource_snapshots import start_snapshot_series

            if not start_snapshot_series():
                logger.warning(
                    "snapshot_series_start_failed: reason=monitor_not_running"
                )
        except Exception as e:
            logger.warning(
                "snapshot_series_start_failed: error_class=%s",
                type(e).__name__,
            )

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
                "optional_dependency_missing: dependency=%s feature=%s",
                status.dependency.name,
                status.dependency.feature,
            )
    except Exception as e:
        logger.warning(
            "dependency_check_failed: error_class=%s",
            type(e).__name__,
        )
    
    # Run hardware detection on first startup (if auto-detect enabled)
    try:
        settings = get_config()
        if settings.hardware.auto_detect_on_startup and not settings.hardware.recommended_model:
            logger.info("hardware_detection_started:")
            recommender = ModelRecommender()
            recommended = recommender.detect_and_recommend()
            specs = recommender.get_detected_specs()
            logger.info(
                "hardware_detection_complete: ram_gb=%.1f cpu_cores=%d "
                "recommended_model=%s",
                specs.total_ram_gb, specs.cpu_count_logical, recommended,
            )
    except Exception as e:
        # Log error but don't block startup
        logger.warning(
            "hardware_detection_failed: error_class=%s",
            type(e).__name__,
        )

    # Environment info refinement (issue #108): the bootstrap wrote
    # the artifact with stdlib-only facts; now that the app side is
    # up, refine the hardware class from the REAL detected specs —
    # via environment_info's own prompt-flush discipline (the one
    # sanctioned rewrite; detection failure leaves the bootstrap's
    # "unknown" in place).
    if capture_dir is not None:
        try:
            from meetandread.environment_info import (
                refine_environment_info,
            )

            refine_environment_info(capture_dir)
            logger.debug("environment_info_refined:")
        except Exception as e:
            logger.warning(
                "environment_info_refine_failed: error_class=%s",
                type(e).__name__,
            )

    # Check hardware requirements and warn if below minimum
    try:
        check_hardware_requirements()
    except Exception as e:
        logger.warning(
            "hardware_requirements_check_failed: error_class=%s",
            type(e).__name__,
        )
    
    # Check for partial recordings and offer recovery before showing widget
    # This runs synchronously before the main event loop
    try:
        check_and_offer_recovery(parent=None)
    except Exception as e:
        # Log error but don't block startup
        logger.warning(
            "recovery_check_failed: error_class=%s",
            type(e).__name__,
        )

    # Reporter-death resilience (issue #110): on a NORMAL launch,
    # notice an assembled-but-unsubmitted Diagnostics Bundle left by
    # a dead Issue Reporter and offer to resume its submission from
    # the saved bundle (no re-reproduction). Never blocks startup;
    # capture runs skip the offer entirely (their submission flow is
    # supervised by the reporter's own wizard).
    try:
        check_and_offer_diagnostics_resume(
            parent=None, capture_mode=capture_dir is not None
        )
    except Exception as e:
        logger.warning(
            "diagnostics_resume_check_failed: error_class=%s",
            type(e).__name__,
        )

    # Process pending cleanup queue (orphaned files from prior sessions)
    try:
        from meetandread.recording.cleanup_queue import CleanupQueue
        cleanup_queue = CleanupQueue()
        result = cleanup_queue.process_pending()
        if result.processed > 0 or result.failed > 0:
            logger.info(
                "startup_cleanup_complete: processed=%d failed=%d",
                result.processed, result.failed,
            )
    except Exception as e:
        logger.warning(
            "startup_cleanup_failed: error_class=%s",
            type(e).__name__,
        )
    
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
        logger.warning(
            "post_processing_startup_init_failed: error_class=%s",
            type(e).__name__,
        )
    
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

    logger.info("tray_icon_ready:")
    
    widget.show()

    # Degraded-mode banner (issue #61): after the widget is visible so
    # the toast anchors against the shown widget.
    try:
        widget.maybe_show_dependency_banner()
    except Exception as e:
        logger.warning(
            "dependency_banner_failed: error_class=%s",
            type(e).__name__,
        )

    # Clean-exit completion marker (issue #104): written only when the
    # event loop returns normally with exit code 0 — the marker is the
    # absence-proof of a crash ("no marker == crashed run", by
    # construction). Any hard exit (kill, native crash, sys.exit from a
    # dialog path) skips this point entirely.
    exit_code = app.exec()

    # Interaction Trace teardown (issue #105, PR #123 fix round 3):
    # remove the application-level Qt filter and close the trace
    # writer BEFORE the QApplication object is destroyed. A Python-side
    # event filter left installed at app destruction crashes natively
    # (0xC0000005) when Qt delivers teardown events into the dying
    # filter — this ordering keeps the filter and every trace emission
    # strictly inside the live-application window. Capture runs only;
    # best-effort so the exit path is never compromised.
    if capture_dir is not None:
        try:
            from meetandread.interaction_trace_qt import (
                remove_interaction_trace_qt_filter,
            )

            remove_interaction_trace_qt_filter(app)
        except Exception as e:
            logger.warning(
                "interaction_trace_qt_teardown_failed: error_class=%s",
                type(e).__name__,
            )
        try:
            from meetandread.interaction_trace import (
                close_interaction_trace,
            )

            close_interaction_trace()
        except Exception as e:
            logger.warning(
                "interaction_trace_teardown_failed: error_class=%s",
                type(e).__name__,
            )
        # Resource Snapshot series teardown (issue #106): stop the
        # driving monitor and close the series writer BEFORE the
        # completion marker — every diagnostic emission stays strictly
        # inside the live-application window (same ordering rationale
        # as the trace teardown above; the monitor's parentless QTimer
        # must not fire into a dying QApplication).
        try:
            from meetandread.resource_snapshots import close_snapshot_series

            close_snapshot_series()
        except Exception as e:
            logger.warning(
                "snapshot_series_teardown_failed: error_class=%s",
                type(e).__name__,
            )
        # Transcript canary registry teardown (issue #108): close the
        # registry writer in the same teardown window — sampling stops
        # with the application it guards.
        try:
            from meetandread.transcript_canary import (
                close_canary_registry,
            )

            close_canary_registry()
        except Exception as e:
            logger.warning(
                "transcript_canary_teardown_failed: error_class=%s",
                type(e).__name__,
            )

    if capture_dir is not None and exit_code == 0:
        write_completion_marker(capture_dir)

    # GATED deterministic exit (issue #124, PR #125 fix round 2).
    #
    # CAPTURE RUNS (capture_dir is not None) end in a hard exit: do NOT
    # "simplify" this back to sys.exit(exit_code). Letting the
    # interpreter fall through into normal finalization tears down the
    # live Qt object graph (the whole widget tree dies in GC order) — on
    # the current sip/Qt stack (PyQt6-Qt6 6.11.2 / PyQt6-sip 13.12.0)
    # that teardown is a race which intermittently crashes natively with
    # 0xC0000005 during interpreter finalization AFTER sys.exit, in CI
    # and locally (nightly run 34574666396 on head 3f57eff; PR #123's
    # filter-removal ordering narrowed but cannot eliminate it).
    # os._exit skips finalization entirely, making the exit
    # deterministic. This is safe because:
    # - every capture log record, trace event, and snapshot record is
    #   already flushed+fsynced before its emit returns, and the completion
    #   marker is written above, BEFORE this point (#104/#105/#106
    #   durability contract; TestKillMidRun proves even taskkill /F loses
    #   nothing);
    # - the single-instance mutex is OS-released when the process dies
    #   (see single_instance.py module docstring);
    # - every test that runs main() does so as a subprocess; there are
    #   no in-process main() callers.
    #
    # NORMAL RUNS (capture_dir is None, issue #104): sys.exit with
    # normal interpreter finalization — the pre-#124 behavior. No
    # capture log or marker exists on this path, so the hard exit's
    # safety arguments do not apply; a normal run keeps its normal exit.
    #
    # Both cases fire ONLY on the real app exit path (app.exec()
    # returned) — the early sys.exit refusals above happen before any Qt
    # object graph exists and stay as-is.
    if capture_dir is not None:
        logger.debug("hard_exit: code=%d", exit_code)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except OSError:
            pass
        os._exit(exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
