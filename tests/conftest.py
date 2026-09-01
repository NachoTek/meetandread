"""Shared test fixtures for the meetandread test suite."""
import importlib
import sys
import types


def _ensure_stub_if_missing(name: str, **attrs) -> None:
    """Stub a native backend module when the real one cannot be imported.

    The app targets Windows-only native backends (WASAPI/PortAudio via
    ``sounddevice``). Importing the app eagerly imports ``sounddevice`` at
    module top-level, so on platforms where PortAudio is unavailable test
    collection fails before any fixture runs. This stubs only ``sounddevice``
    when it genuinely fails to import, keeping the pure-Python layers
    (timestamp producers, the highlight consumer, footer parsing, ...) testable
    everywhere. Other native backends (``pyaudiowpatch``, ``sherpa_onnx``,
    ``webrtcvad``, ``comtypes``, ``pywhispercpp``) are already imported
    gracefully by the app, so they are deliberately NOT stubbed here — stubbing
    them would mask their absence and flip feature flags in the tests that
    exercise them. On Windows (where the real library is installed) nothing is
    stubbed.
    """
    try:
        importlib.import_module(name)
    except Exception:
        stub = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(stub, key, value)
        sys.modules[name] = stub


_ensure_stub_if_missing(
    "sounddevice",
    InputStream=type("InputStream", (), {}),
    query_devices=lambda *a, **k: [],
    query_host_apis=lambda *a, **k: [],
    CallbackFlags=type("CallbackFlags", (), {}),
)


import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402


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

    Issue #86: CI runs with ``-p no:qt`` (issue #64), so pytest-qt's own
    ``pytest_runtest_teardown`` hook — which normally processes pending Qt
    events and closes tracked widgets between tests — never runs.  Without
    any ``processEvents()`` call, ``deleteLater()`` requests accumulate and
    widget/timer C++ objects are destroyed late (at gc or interpreter exit)
    while their native callbacks may still fire — a stochastic access
    violation under both single-process and xdist runs.  This teardown now
    applies pytest-qt's core hygiene even when the plugin is disabled, in
    the safe order: neutralize leaked timers first (retry timers and
    ResourceMonitor pollers), then close widgets, then a single event-loop
    flush to deliver the deferred deletions.  Flushing before the timers
    are stopped would let a due stale callback execute — the exact hazard
    this fixture exists to prevent.
    """
    yield
    app = QApplication.instance()
    if app is None:
        return
    from meetandread.performance.monitor import ResourceMonitor

    # 1. Neutralize leaked timers BEFORE any event processing.
    for widget in app.topLevelWidgets():
        for timer in widget.findChildren(QTimer):
            try:
                timer.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass
            timer.stop()
            timer.deleteLater()
    # ResourceMonitor poll timers are parentless — invisible to the
    # findChildren sweep — and keep firing into deleted panels otherwise.
    ResourceMonitor.stop_all()
    # 2. Close top-level widgets (deferred deletion via deleteLater).
    for widget in app.topLevelWidgets():
        try:
            widget.hide()
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass
    # 3. Single flush: deliver deferred deletions now that no stale
    # callback can run.
    app.processEvents()


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``windows``-marked tests when not running on Windows (ADR 0001).

    Tests flagged ``@pytest.mark.windows`` require the real Windows native
    audio stack (PortAudio / ``pyaudiowpatch``) and must run under the Windows
    interpreter (``.venv/Scripts/python.exe`` via WSL interop). Off-Windows they
    are skipped rather than erroring on a missing native library — the WSL Linux
    logic-layer run stays green. On Windows (``sys.platform == 'win32'``) they
    run normally.
    """
    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(
        reason="windows-only \u2014 run under .venv/Scripts/python.exe (ADR 0001)"
    )
    for item in items:
        if "windows" in item.keywords:
            item.add_marker(skip)
