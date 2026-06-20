"""Shared test fixtures for the meetandread test suite."""
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer


@pytest.fixture(autouse=True)
def _cleanup_qtimers():
    """Stop any leaked QTimers after each test.

    Tests that exercise the WASAPI retry flow create real single-shot QTimers
    (1s/2s/4s backoff schedule).  If a test finishes without calling
    ``_clear_retry_state()`` the timer keeps running and fires during the
    *next* test's event-processing phase (pytest-qt calls ``processEvents``
    in ``pytest_runtest_setup``).  The leaked callback chain eventually
    exhausts retries and opens a blocking ``QDialog.exec()`` which hangs
    forever in headless CI.

    This safety net walks every top-level widget, disconnects, stops, and
    deletes every ``QTimer`` child.
    """
    yield
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        for timer in widget.findChildren(QTimer):
            try:
                timer.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass
            timer.stop()
            timer.deleteLater()
