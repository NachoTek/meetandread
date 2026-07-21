"""Shared test fixtures for the meetandread test suite."""
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer


@pytest.fixture(autouse=True)
def _isolate_native_live_speaker_extraction(request, monkeypatch):
    """Keep ordinary tests from initializing sherpa-onnx against user data.

    The dedicated live-speaker module mocks the native boundary explicitly and
    remains exempt. Other tests may leave recording workers alive briefly;
    disabling lazy extractor initialization prevents those workers from reading
    the real user speaker store or crashing pytest inside native ONNX code.
    """
    if request.node.path.name in ("test_live_speaker_names.py", "test_speaker_identity_integration.py"):
        return

    from meetandread.recording.controller import RecordingController

    monkeypatch.setattr(
        RecordingController,
        "_ensure_live_extractor",
        lambda self: False,
    )


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
