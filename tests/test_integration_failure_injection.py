"""Integration tests for WASAPI failure injection retry/fallback behavior.

These tests drive the widget retry flow with a real ``RecordingController``
instance backed by a fake session. They inject source-initialization failures
before capture starts and assert deterministic retry/fallback metadata without
real WASAPI devices or blocking sleeps.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog

from meetandread.audio.session import SessionStats
from meetandread.recording.controller import ControllerError, ControllerState, RecordingController
from meetandread.widgets.main_widget import MeetAndReadWidget


FORBIDDEN_DIAGNOSTIC_TOKENS = (
    "C:/Users/david.keymel/recordings/raw.wav",
    "transcript text should never appear",
    "embedding-vector",
    "sk-test-secret",
    "raw_audio",
)


class FakeRetrySession:
    """Fake AudioSession preserving retry stats across controller calls."""

    def __init__(self) -> None:
        self._state_val = ControllerState.IDLE
        self._stats = SessionStats()
        self.starts = []

    def start(self, config):
        self.starts.append(config)
        self._state_val = ControllerState.RECORDING

    def stop(self):
        self._state_val = ControllerState.FINALIZED
        return "recording.wav"

    def get_state(self):
        return self._state_val

    def get_stats(self):
        return self._stats

    def get_frame_drop_diagnostics(self):
        return {"total_dropped_frames": 0, "events": []}

    def get_error(self):
        return None

    def close(self):
        self._state_val = ControllerState.IDLE


class ScriptedFailureController(RecordingController):
    """RecordingController with injectable source-start outcomes."""

    def __init__(self, outcomes):
        super().__init__()
        self._session = FakeRetrySession()
        self._outcomes = list(outcomes)
        self.start_calls = []

    def start(self, sources={"mic"}, **kwargs):  # noqa: B006 - mirrors controller API for tests
        self.start_calls.append(set(sources))
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, ControllerError):
                return outcome
            if isinstance(outcome, Exception):
                return ControllerError(str(outcome), is_recoverable=True)
        self._set_state(ControllerState.RECORDING)
        return None


class FailureInjectionTest:
    """Integration coverage for WASAPI start failure retry/fallback semantics."""

    @staticmethod
    def _widget_with_controller(monkeypatch, controller):
        app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
        if app is None:
            app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication([])

        with monkeypatch.context() as m:
            from unittest.mock import Mock

            panel = Mock()
            panel.hide_panel = Mock()
            panel.show_panel = Mock()
            panel.update_transcription = Mock()
            panel.set_recording_state = Mock()
            m.setattr("meetandread.widgets.main_widget.FloatingSettingsPanel", Mock(return_value=panel))
            m.setattr("meetandread.widgets.main_widget.CCOverlayPanel", Mock(return_value=panel))
            toast_cls = Mock()
            toast_cls.return_value.show = Mock(return_value="wasapi-retry")
            toast_cls.return_value.dismiss = Mock()
            m.setattr("meetandread.widgets.main_widget.ToastManager", toast_cls)
            m.setattr("meetandread.widgets.main_widget.RecordingController", lambda *a, **k: controller)
            widget = MeetAndReadWidget()
        widget._controller = controller
        widget.toast_manager.show = getattr(widget.toast_manager, "show", lambda *a, **k: "wasapi-retry")
        widget._show_error = lambda *args, **kwargs: None
        return widget

    @staticmethod
    def _assert_sanitized_diagnostics(diagnostics):
        rendered = repr(diagnostics)
        for token in FORBIDDEN_DIAGNOSTIC_TOKENS:
            assert token not in rendered
        assert len(rendered) < 25000

    def test_wasapi_source_initialization_failure_retries_three_times_then_falls_back_with_consent(self, monkeypatch):
        """Endpoint unavailable before capture start yields 1s/2s/4s retry metadata and mic fallback."""
        controller = ScriptedFailureController(
            [
                ControllerError("AudioSourceError: WASAPI endpoint unavailable", is_recoverable=True),
                ControllerError("AudioSourceError: WASAPI endpoint unavailable", is_recoverable=True),
                ControllerError("AudioSourceError: WASAPI endpoint unavailable", is_recoverable=True),
                ControllerError("AudioSourceError: WASAPI endpoint unavailable", is_recoverable=True),
                None,
            ]
        )
        widget = self._widget_with_controller(monkeypatch, controller)
        scheduled_delays = []

        class ImmediateTimer:
            def __init__(self, parent=None):
                self._callback = None
                self.timeout = self

            def setSingleShot(self, value):
                self.single_shot = value

            def connect(self, callback):
                self._callback = callback

            def start(self, delay_ms):
                scheduled_delays.append(delay_ms)
                if self._callback:
                    self._callback()

            def stop(self):
                pass

            def deleteLater(self):
                pass

        monkeypatch.setattr("meetandread.widgets.main_widget.QTimer", ImmediateTimer)

        from unittest.mock import Mock

        dialog = Mock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted
        dialog.accepted_fallback.return_value = True
        dialog_cls = Mock(return_value=dialog)
        monkeypatch.setattr("meetandread.widgets.main_widget.FallbackConfirmationDialog", dialog_cls)

        widget._start_with_retry({"mic", "system"}, first_attempt=True)

        assert controller.start_calls == [
            {"mic", "system"},
            {"mic", "system"},
            {"mic", "system"},
            {"mic", "system"},
            {"mic"},
        ]
        assert scheduled_delays == [1000, 2000, 4000]
        assert dialog_cls.called

        diagnostics = controller.get_diagnostics()
        session = diagnostics["session"]
        retry_events = diagnostics["retry"]["events"]
        attempts = [event for event in retry_events if "attempt_number" in event]
        assert session["retry_attempts"] == 3
        assert [attempt["attempt_number"] for attempt in attempts] == [1, 2, 3]
        assert [attempt["backoff_seconds"] for attempt in attempts] == [1.0, 2.0, 4.0]
        assert session["failed_sources"] == ["system"]
        assert session["fallback_sources"] == ["mic"]
        assert session["retry_outcome"] == "fallback_to_mic_only"
        self._assert_sanitized_diagnostics(diagnostics)

    def test_mic_only_start_bypasses_wasapi_retry_path(self, monkeypatch):
        """Non-WASAPI mic-only starts do not begin retry or record retry metadata."""
        controller = ScriptedFailureController([ControllerError("microphone unavailable", is_recoverable=True)])
        widget = self._widget_with_controller(monkeypatch, controller)
        widget.mic_lobe.is_active = True
        widget.system_lobe.is_active = False

        widget.start_recording()

        assert controller.start_calls == [{"mic"}]
        diagnostics = controller.get_diagnostics()
        session = diagnostics["session"]
        assert session["retry_attempts"] == 0
        assert diagnostics["retry"]["events"] == []
        assert widget._retry_in_progress is False
        self._assert_sanitized_diagnostics(diagnostics)

    def test_retry_cancellation_does_not_start_fallback_silently(self, monkeypatch):
        """Cancelling an active retry clears retry state and does not open fallback or start mic-only."""
        from unittest.mock import Mock

        controller = ScriptedFailureController([])
        widget = self._widget_with_controller(monkeypatch, controller)
        widget._retry_in_progress = True
        widget._retry_sources = {"mic", "system"}
        widget._retry_attempt = 2
        widget._retry_timer = Mock()
        dialog_cls = Mock()
        monkeypatch.setattr("meetandread.widgets.main_widget.FallbackConfirmationDialog", dialog_cls)

        widget._cancel_retry()

        assert widget._retry_in_progress is False
        assert controller.start_calls == []
        assert not dialog_cls.called
        retry_session = controller.get_diagnostics()["session"]
        assert retry_session["fallback_sources"] == []

    def test_partial_system_failure_requires_explicit_fallback_consent(self, monkeypatch):
        """Partial source failure does not degrade to mic-only when fallback consent is rejected."""
        from unittest.mock import Mock

        controller = ScriptedFailureController(
            [ControllerError("AudioSourceError: system endpoint unavailable", is_recoverable=True)]
        )
        widget = self._widget_with_controller(monkeypatch, controller)
        dialog = Mock()
        dialog.exec.return_value = QDialog.DialogCode.Rejected
        dialog.accepted_fallback.return_value = False
        dialog_cls = Mock(return_value=dialog)
        monkeypatch.setattr("meetandread.widgets.main_widget.FallbackConfirmationDialog", dialog_cls)

        widget._retry_attempt = 3
        widget._start_with_retry({"mic", "system"}, first_attempt=False)

        assert controller.start_calls == [{"mic", "system"}]
        assert dialog_cls.called
        diagnostics = controller.get_diagnostics()
        session = diagnostics["session"]
        assert session["failed_sources"] == ["system"]
        assert session["fallback_sources"] == []
        assert session["retry_outcome"] == "failed"
        self._assert_sanitized_diagnostics(diagnostics)


def test_wasapi_source_initialization_failure_retries_three_times_then_falls_back_with_consent(monkeypatch):
    FailureInjectionTest().test_wasapi_source_initialization_failure_retries_three_times_then_falls_back_with_consent(monkeypatch)


def test_mic_only_start_bypasses_wasapi_retry_path(monkeypatch):
    FailureInjectionTest().test_mic_only_start_bypasses_wasapi_retry_path(monkeypatch)


def test_retry_cancellation_does_not_start_fallback_silently(monkeypatch):
    FailureInjectionTest().test_retry_cancellation_does_not_start_fallback_silently(monkeypatch)


def test_partial_system_failure_requires_explicit_fallback_consent(monkeypatch):
    FailureInjectionTest().test_partial_system_failure_requires_explicit_fallback_consent(monkeypatch)
