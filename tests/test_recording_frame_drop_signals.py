"""Tests for frame-drop signal propagation through controller and Qt bridge.

Covers:
- RecordingController._on_session_frames_dropped sanitizes and forwards counts
- RecordingController.on_frames_dropped user callback error isolation
- RecordingController.start() wires on_frames_dropped into SessionConfig
- RecordingController.get_diagnostics() exposes aggregate frames_dropped
- _ControllerBridge.frames_dropped signal exists and emits int
- MeetAndReadWidget._on_frames_dropped forwards to record_button
- MeetAndReadWidget._on_frames_dropped handles malformed inputs
- Bridge wiring in MeetAndReadWidget.__init__

All tests use inline fakes/monkeypatches — no real audio devices.
"""

import threading
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from meetandread.recording.controller import RecordingController, ControllerState
from meetandread.audio.session import SessionConfig, SessionStats


# ---------------------------------------------------------------------------
# RecordingController frame-drop callback
# ---------------------------------------------------------------------------


class TestControllerFrameDropCallback:
    """Controller._on_session_frames_dropped sanitizes and forwards counts."""

    def test_positive_count_forwarded(self):
        """Positive count is forwarded to on_frames_dropped callback."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(5)
        assert received == [5]

    def test_zero_count_is_noop(self):
        """Count 0 should not invoke the callback."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(0)
        assert received == []

    def test_negative_count_coerced_to_zero_noop(self):
        """Negative counts are coerced to 0 and become no-ops."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(-3)
        assert received == []

    def test_float_count_coerced_to_int(self):
        """Float counts are truncated to int and forwarded."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(7.9)
        assert received == [7]

    def test_string_count_coerced_to_zero_noop(self):
        """Non-numeric string counts are treated as 0 (no-op)."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped("not-a-number")
        assert received == []

    def test_none_count_coerced_to_zero_noop(self):
        """None counts are treated as 0 (no-op)."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(None)
        assert received == []

    def test_no_callback_set_is_safe(self):
        """No crash when on_frames_dropped is None."""
        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_frames_dropped = None
        ctrl._on_session_frames_dropped(10)  # should not raise

    def test_callback_exception_does_not_propagate(self):
        """Exceptions in user callback are caught and don't break the call."""
        ctrl = RecordingController(enable_transcription=False)
        ctrl.on_frames_dropped = MagicMock(side_effect=RuntimeError("boom"))
        # Should not raise
        ctrl._on_session_frames_dropped(3)
        ctrl.on_frames_dropped.assert_called_once_with(3)

    def test_large_count_forwarded(self):
        """Large counts are forwarded without modification."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(99999)
        assert received == [99999]


class TestControllerSessionConfigWiring:
    """Controller.start() wires on_frames_dropped into SessionConfig."""

    def test_session_config_includes_drop_callback(self):
        """SessionConfig created by start() has on_frames_dropped set."""
        ctrl = RecordingController(enable_transcription=False)

        captured_configs = []

        from meetandread.audio.session import AudioSession

        original_init = AudioSession.__init__

        def patched_init(self_session):
            original_init(self_session)
            # Capture the config passed to session.start()
            real_start = self_session.start
            def capturing_start(config):
                captured_configs.append(config)
            self_session.start = capturing_start

        with patch.object(AudioSession, "__init__", patched_init), \
             patch("meetandread.recording.controller.AudioSession.__init__", patched_init):
            # Create a minimal fake WAV file
            import tempfile, os, wave
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                with wave.open(tmp_path, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

                ctrl.start({"fake"}, fake_path=tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.on_frames_dropped is not None
        # Verify the callback is the controller's internal handler
        assert cfg.on_frames_dropped == ctrl._on_session_frames_dropped


class TestControllerDiagnosticsExposeDrops:
    """Controller.get_diagnostics() exposes aggregate frames_dropped."""

    def test_diagnostics_shows_session_frames_dropped(self):
        """get_diagnostics()['session']['frames_dropped'] returns the count."""
        ctrl = RecordingController(enable_transcription=False)
        # Simulate session stats with drops
        ctrl._session._stats = SessionStats(frames_dropped=42)
        diag = ctrl.get_diagnostics()
        assert diag["session"]["frames_dropped"] == 42


# ---------------------------------------------------------------------------
# Qt bridge signal
# ---------------------------------------------------------------------------


class TestControllerBridgeFramesDroppedSignal:
    """_ControllerBridge.frames_dropped is a pyqtSignal(int)."""

    def test_signal_exists(self):
        """Bridge has a frames_dropped signal."""
        from meetandread.widgets.main_widget import _ControllerBridge
        bridge = _ControllerBridge()
        assert hasattr(bridge, "frames_dropped")

    def test_signal_emits_int(self, qapp):
        """frames_dropped signal can emit an integer."""
        from meetandread.widgets.main_widget import _ControllerBridge
        bridge = _ControllerBridge()
        received = []
        bridge.frames_dropped.connect(lambda c: received.append(c))
        bridge.frames_dropped.emit(5)
        # Process event loop to deliver queued signal
        qapp.processEvents()
        assert received == [5]


# ---------------------------------------------------------------------------
# Widget slot wiring
# ---------------------------------------------------------------------------


def _make_widget(qapp):
    """Create a MeetAndReadWidget with mocked Qt globals for testing."""
    from unittest.mock import patch, MagicMock
    from meetandread.widgets.main_widget import MeetAndReadWidget

    class _FakeScreenGeometry:
        def __init__(self, width=1920, height=1080):
            self._w = width
            self._h = height
        def width(self):
            return self._w
        def height(self):
            return self._h
        def contains(self, point):
            return 0 <= point.x() < self._w and 0 <= point.y() < self._h

    fake_geo = _FakeScreenGeometry()
    fake_screen = MagicMock()
    fake_screen.geometry.return_value = fake_geo
    with patch("meetandread.widgets.main_widget.QApplication.primaryScreen",
               return_value=fake_screen), \
         patch("meetandread.widgets.main_widget.QApplication.screens",
               return_value=[fake_screen]), \
         patch("meetandread.widgets.main_widget.get_config", return_value=None), \
         patch("meetandread.widgets.main_widget.save_config"):
        w = MeetAndReadWidget()
    w._floating_settings_panel = MagicMock()
    w._floating_settings_panel.isVisible.return_value = False
    w._cc_overlay = MagicMock()
    w._cc_overlay.isVisible.return_value = False
    return w


class TestWidgetFramesDroppedSlot:
    """MeetAndReadWidget._on_frames_dropped forwards to record_button."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_forwards_positive_count_to_record_button(self, widget):
        """Positive count is forwarded to record_button.on_frames_dropped."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped(5)
        mock_handler.assert_called_once_with(5)

    def test_zero_count_is_noop(self, widget):
        """Count 0 does not invoke record_button handler."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped(0)
        mock_handler.assert_not_called()

    def test_negative_count_coerced_to_zero_noop(self, widget):
        """Negative count is coerced to 0 and becomes a no-op."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped(-3)
        mock_handler.assert_not_called()

    def test_float_count_coerced_to_int(self, widget):
        """Float count is truncated to int and forwarded."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped(7.9)
        mock_handler.assert_called_once_with(7)

    def test_string_count_is_noop(self, widget):
        """Non-numeric string count is ignored."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped("bad")
        mock_handler.assert_not_called()

    def test_none_count_is_noop(self, widget):
        """None count is ignored."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._on_frames_dropped(None)
        mock_handler.assert_not_called()

    def test_no_handler_on_record_button_is_safe(self, widget):
        """Record button has on_frames_dropped (T3) and handles calls safely."""
        # T03 now provides on_frames_dropped — verify it exists and is callable
        assert callable(widget.record_button.on_frames_dropped)
        # Calling with None should not crash (handled by defensive coercion)
        widget.record_button.on_frames_dropped(None)

    def test_handler_exception_does_not_crash(self, widget):
        """Exception in record_button.on_frames_dropped is caught."""
        widget.record_button.on_frames_dropped = MagicMock(side_effect=RuntimeError("boom"))
        widget._on_frames_dropped(5)  # should not raise


class TestBridgeToWidgetWiring:
    """Bridge frames_dropped signal connects to widget _on_frames_dropped."""

    @pytest.fixture
    def widget(self, qapp):
        w = _make_widget(qapp)
        yield w
        w.close()

    def test_bridge_signal_reaches_slot(self, widget, qapp):
        """Emitting bridge.frames_dropped triggers _on_frames_dropped."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        widget._bridge.frames_dropped.emit(3)
        qapp.processEvents()
        mock_handler.assert_called_once_with(3)

    def test_controller_callback_emits_bridge_signal(self, widget, qapp):
        """Calling controller.on_frames_dropped emits via the bridge."""
        mock_handler = MagicMock()
        widget.record_button.on_frames_dropped = mock_handler
        # The controller callback is a lambda that emits the bridge signal
        widget._controller.on_frames_dropped(7)
        qapp.processEvents()
        mock_handler.assert_called_once_with(7)


class TestControllerCallbackErrorIsolation:
    """User-provided on_frames_dropped errors don't break recording state."""

    def test_broken_callback_does_not_change_recording_state(self):
        """After callback error, controller state is unchanged."""
        ctrl = RecordingController(enable_transcription=False)
        ctrl._state = ControllerState.RECORDING
        ctrl.on_frames_dropped = MagicMock(side_effect=RuntimeError("crash"))
        ctrl._on_session_frames_dropped(10)
        assert ctrl._state == ControllerState.RECORDING

    def test_broken_callback_allows_subsequent_calls(self):
        """After one callback failure, subsequent drops still invoke callback."""
        ctrl = RecordingController(enable_transcription=False)
        call_count = [0]
        def broken_then_ok(count):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first crash")

        ctrl.on_frames_dropped = broken_then_ok
        ctrl._on_session_frames_dropped(1)
        ctrl._on_session_frames_dropped(2)
        assert call_count[0] == 2  # both calls attempted


class TestBoundaryConditions:
    """Edge cases for count propagation."""

    def test_count_5_forwarded_exactly_once(self):
        """Count 5 is forwarded exactly once when emitted."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        ctrl._on_session_frames_dropped(5)
        assert received == [5]

    def test_sequential_drops_accumulate(self):
        """Multiple sequential drops each trigger the callback."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        ctrl.on_frames_dropped = lambda c: received.append(c)
        for i in range(1, 6):
            ctrl._on_session_frames_dropped(i)
        assert received == [1, 2, 3, 4, 5]

    def test_thread_safety_of_callback_invocation(self):
        """Callback invoked from multiple threads doesn't crash."""
        ctrl = RecordingController(enable_transcription=False)
        received = []
        lock = threading.Lock()

        def safe_append(c):
            with lock:
                received.append(c)

        ctrl.on_frames_dropped = safe_append
        threads = [
            threading.Thread(target=ctrl._on_session_frames_dropped, args=(i,))
            for i in range(1, 21)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert sorted(received) == list(range(1, 21))
