"""Final-assembly tests for complete hot-plug reconnection lifecycles.

These tests keep a real AudioSession consumer running while device events pass
through RecordingController and its callbacks cross the Qt bridge into the
production widget notification handlers. Only the physical audio device is
replaced by the deterministic FakeAudioModule.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from meetandread.audio.capture.fake_module import FakeAudioModule
from meetandread.audio.hotplug import DeviceEvent, DeviceEventType
from meetandread.audio.session import (
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SessionState,
    SourceConfig,
)
from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
    RecoveryOutcome,
)
from meetandread.widgets.main_widget import MeetAndReadWidget, _ControllerBridge


FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORIGINAL_WAV = str(FIXTURES / "SAMPLE-Audio1.wav")
REPLACEMENT_WAV = str(FIXTURES / "Harvard_AI_audio_16k.wav")
TOAST_ID = "recording-device-recovery"
DEVICE_ID = "fake-device-1"
DEVICE_NAME = "Test USB microphone"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class RecordingToastManager:
    """Record ToastManager replacements without creating visible widgets."""

    def __init__(self):
        self.shown = []

    def show(
        self,
        toast_id,
        title,
        message,
        *,
        duration_ms=8000,
        action_label=None,
        action_callback=None,
    ):
        self.shown.append(
            {
                "toast_id": toast_id,
                "title": title,
                "message": message,
                "duration_ms": duration_ms,
                "action_label": action_label,
                "action_callback": action_callback,
            }
        )


class ObservedFakeAudioModule(FakeAudioModule):
    """Fake hardware boundary that counts frames delivered to AudioSession."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delivered_frames = 0

    def read_frames(self, timeout=None):
        frames = super().read_frames(timeout=timeout)
        if frames is not None:
            self.delivered_frames += len(frames)
        return frames


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("Timed out waiting for replacement source frames")


def _event(event_type, state):
    return DeviceEvent(
        event_type,
        DEVICE_ID,
        friendly_name=DEVICE_NAME,
        state=state,
    )


def _replacement_wrapper():
    source = ObservedFakeAudioModule(
        wav_path=REPLACEMENT_WAV,
        blocksize=1024,
        queue_size=10,
        loop=True,
    )
    config = SourceConfig(type="mic")
    return AudioSourceWrapper(source, config, target_rate=16000, target_channels=1)


def _running_lifecycle(tmp_path):
    capture_config = SourceConfig(
        type="fake",
        fake_path=ORIGINAL_WAV,
        loop=True,
    )
    session = AudioSession()
    session.start(SessionConfig(sources=[capture_config], output_dir=tmp_path))
    # Model fake audio as a hot-pluggable mic after construction. AudioSession
    # still exercises the real consumer/swap path while no hardware is needed.
    session._sources[0].config.type = "mic"

    active_identity = SourceConfig(type="mic", device_id=DEVICE_ID)
    active_identity.friendly_name = DEVICE_NAME
    controller = RecordingController(enable_transcription=False)
    controller._session = session
    controller._state = ControllerState.RECORDING
    controller._snapshot_active_sources([active_identity])

    widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
    widget.toast_manager = RecordingToastManager()
    widget._recovery_toast_id = TOAST_ID
    widget._controller = controller

    bridge = _ControllerBridge()
    bridge.device_changed.connect(
        lambda event: MeetAndReadWidget._on_device_changed(widget, event)
    )
    bridge.recovery_attempted.connect(
        lambda result: MeetAndReadWidget._on_recovery_attempted(widget, result)
    )
    controller.on_device_change = lambda event: bridge.device_changed.emit(event)
    controller.on_recovery_attempt = lambda result: bridge.recovery_attempted.emit(result)

    return session, controller, widget, bridge


def _assert_replacement_frames(session, replacement):
    before = session.get_stats().frames_recorded
    _wait_until(lambda: replacement.source.delivered_frames > 0)
    _wait_until(lambda: session.get_stats().frames_recorded > before)
    assert session.get_state() is SessionState.RECORDING


def test_disconnect_auto_recovers_through_bridge_with_replacement_frames(tmp_path, qapp):
    session, controller, widget, bridge = _running_lifecycle(tmp_path)
    replacement = _replacement_wrapper()
    controller._rebuild_source_wrapper = lambda source_type, device_id=None: replacement

    try:
        lost = controller.handle_device_event(
            _event(DeviceEventType.REMOVED, "inactive"), now=100.0
        )
        recovered = controller.handle_device_event(
            _event(DeviceEventType.ADDED, "active"), now=103.0
        )
        qapp.processEvents()

        assert lost.outcome is RecoveryOutcome.TOTAL_LOSS
        assert recovered.outcome is RecoveryOutcome.AUTO_RECOVERED
        assert controller.get_state() is ControllerState.RECORDING
        _assert_replacement_frames(session, replacement)

        toasts = widget.toast_manager.shown
        assert [toast["title"] for toast in toasts] == [
            "Recording device disconnected",
            "Recording paused",
            "Recording device changed",
            "Recording recovered",
        ]
        assert all(toast["toast_id"] == TOAST_ID for toast in toasts)
        assert toasts[-1]["duration_ms"] > 0
        assert toasts[-1]["action_label"] is None
        assert "recovered" in toasts[-1]["message"]
    finally:
        session.stop()
        del bridge


def test_expired_recovery_uses_toast_action_without_starting_new_session(tmp_path, qapp):
    session, controller, widget, bridge = _running_lifecycle(tmp_path)
    replacement = _replacement_wrapper()
    controller._rebuild_source_wrapper = lambda source_type, device_id=None: replacement
    original_session = controller._session
    original_consumer = session._consumer_thread

    try:
        lost = controller.handle_device_event(
            _event(DeviceEventType.REMOVED, "inactive"), now=200.0
        )
        expired = controller.handle_device_event(
            _event(DeviceEventType.ADDED, "active"), now=206.0
        )
        qapp.processEvents()

        assert lost.outcome is RecoveryOutcome.TOTAL_LOSS
        assert expired.outcome is RecoveryOutcome.MANUAL_RETRY_REQUIRED
        paused = widget.toast_manager.shown[-1]
        assert paused["toast_id"] == TOAST_ID
        assert paused["title"] == "Recording paused"
        assert paused["duration_ms"] == 0
        assert paused["action_label"] == "Resume Recording"
        assert callable(paused["action_callback"])

        paused["action_callback"]()
        qapp.processEvents()

        assert controller._session is original_session
        assert session._consumer_thread is original_consumer
        assert original_consumer.is_alive()
        assert controller.get_state() is ControllerState.RECORDING
        assert controller._last_recovery_result.outcome is RecoveryOutcome.MANUAL_RECOVERED
        _assert_replacement_frames(session, replacement)

        resumed = widget.toast_manager.shown[-1]
        assert resumed["toast_id"] == TOAST_ID
        assert resumed["title"] == "Recording resumed"
        assert resumed["duration_ms"] > 0
        assert resumed["action_label"] is None
        assert "resumed" in resumed["message"]
    finally:
        session.stop()
        del bridge
