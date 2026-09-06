"""Full-audit batch A (issue #99): normalized instrumentation, capture stack.

Hardware-free audit tests for the audio capture stack's logging contract:
- named DEBUG events for significant internal steps,
- operational facts at INFO,
- problems at WARNING and above,
- raw OS device names never logged at ANY level (sha256-8 redaction only).

All flows are driven with patched backends (sounddevice.InputStream /
query_devices, the pyaudiowpatch module seam) or ``__new__`` construction
(prior art: tests/test_issue22_mic_diagnostics.py,
tests/test_audio_frame_drop_mitigation.py).
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.audio.capture.devices import (
    get_default_loopback_device,
    list_loopback_outputs,
    print_device_summary,
)
from meetandread.audio.capture.fake_module import FakeAudioModule
from meetandread.audio.capture.pyaudiowpatch_source import PyAudioWPatchSource
from meetandread.audio.capture.sounddevice_source import (
    SoundDeviceSource,
    SystemSource,
    _redact_device_name,
)
from meetandread.audio.hotplug.windows_device_monitor import (
    DeviceEvent,
    DeviceEventType,
    WindowsDeviceMonitor,
)
from meetandread.audio.session import (
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
)

FIXTURES_DIR = None  # populated in _fixture_wav helper via tmp_path

SD_LOG = "meetandread.audio.capture.sounddevice_source"
PAW_LOG = "meetandread.audio.capture.pyaudiowpatch_source"
DEV_LOG = "meetandread.audio.capture.devices"
FAKE_LOG = "meetandread.audio.capture.fake_module"
MON_LOG = "meetandread.audio.hotplug.windows_device_monitor"
SESS_LOG = "meetandread.audio.session"
CLI_LOG = "meetandread.audio.cli"

PRIVACY_NAME = "ACME Wire Tap 9000"
PRIVACY_ENDPOINT_ID = "{0.0.0.00000000}.{e7a1b2c3-4d5e-6f70-8a9b-cdef01234567}"


def _write_wav(tmp_path, name: str = "sine.wav", seconds: float = 0.4) -> str:
    """Write a short 16 kHz mono int16 WAV and return its string path."""
    rate = 16000
    n = int(rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    pcm = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    path = tmp_path / name
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return str(path)


def _assert_no_raw_name(records) -> None:
    joined = [r.getMessage() for r in records]
    for msg in joined:
        assert PRIVACY_NAME not in msg, f"raw device name leaked: {msg!r}"


def _messages_at(records, level: int):
    return [r.getMessage() for r in records if r.levelno == level]


# ---------------------------------------------------------------------------
# devices.py
# ---------------------------------------------------------------------------


class TestDevicesEnumeration:
    """Enumeration logs per-device details at DEBUG, counts at INFO."""

    def _fake_device(self, name: str = "Generic Speakers") -> dict:
        return {
            "index": 3,
            "name": name,
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        }

    def test_list_loopback_outputs_debug_details_and_info_count(self, caplog):
        devices = [self._fake_device("Speakers (WASAPI)")]
        with patch(
            "meetandread.audio.capture.devices._HAS_PYAUDIOWPATCH", False
        ), patch(
            "meetandread.audio.capture.devices.list_devices",
            return_value=devices,
        ), patch(
            "meetandread.audio.capture.devices.get_wasapi_hostapi_index",
            return_value=0,
        ):
            with caplog.at_level(logging.DEBUG, logger=DEV_LOG):
                out = list_loopback_outputs()

        assert len(out) == 1 and out[0]["loopback_ok"] is True
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any(
            "loopback_probe" in m and "<redacted:" in m for m in debug_msgs
        ), f"no redacted per-device DEBUG probe record: {debug_msgs}"

    def test_list_loopback_outputs_info_quiet_on_per_device_details(self, caplog):
        devices = [self._fake_device("Speakers (WASAPI)")]
        with patch(
            "meetandread.audio.capture.devices._HAS_PYAUDIOWPATCH", False
        ), patch(
            "meetandread.audio.capture.devices.list_devices",
            return_value=devices,
        ), patch(
            "meetandread.audio.capture.devices.get_wasapi_hostapi_index",
            return_value=0,
        ):
            with caplog.at_level(logging.INFO, logger=DEV_LOG):
                list_loopback_outputs()

        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert info_msgs, "expected operational INFO records from enumeration"
        assert not any(
            "<redacted:" in m for m in info_msgs
        ), f"per-device identifiers leaked to INFO: {info_msgs}"


class TestDevicesLoopbackDefaultsPrivacy:
    """PRIVACY: get_default_loopback_device never logs the raw device name."""

    def test_raw_name_never_logged_any_level(self, caplog):
        fake_pa = MagicMock()
        fake_pa_instance = MagicMock()
        fake_pa_instance.__enter__ = MagicMock(return_value=fake_pa_instance)
        fake_pa_instance.__exit__ = MagicMock(return_value=False)
        fake_pa_instance.get_default_wasapi_loopback.return_value = {
            "name": PRIVACY_NAME,
            "index": 11,
        }
        fake_pa.PyAudio.return_value = fake_pa_instance
        with patch(
            "meetandread.audio.capture.devices._HAS_PYAUDIOWPATCH", True
        ), patch("meetandread.audio.capture.devices._paw", fake_pa, create=True):
            with caplog.at_level(logging.DEBUG, logger=DEV_LOG):
                info = get_default_loopback_device()

        assert info is not None and info["name"] == PRIVACY_NAME
        _assert_no_raw_name(caplog.records)
        assert any(
            "<redacted:" in m for m in [r.getMessage() for r in caplog.records]
        ), "redacted identifier expected in loopback probe records"

    def test_summary_no_banner_lines(self, caplog):
        mic = {
            "index": 0,
            "name": "Some Mic",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
            "hostapi": 0,
        }
        output = {
            "index": 1,
            "name": "Some Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        }
        with patch(
            "meetandread.audio.capture.devices._HAS_PYAUDIOWPATCH", False
        ), patch(
            "meetandread.audio.capture.devices.list_devices",
            return_value=[mic, output],
        ), patch(
            "meetandread.audio.capture.devices.get_wasapi_hostapi_index",
            return_value=0,
        ):
            with caplog.at_level(logging.DEBUG, logger=DEV_LOG):
                print_device_summary()

        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            set(m.strip()) == {"="} or set(m.strip()) == {"-"} for m in msgs
        ), f"ad-hoc banner lines still present: {msgs}"
        _assert_no_raw_name(caplog.records)
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert any(
            "device_summary" in m for m in info_msgs
        ), f"expected single INFO summary event: {info_msgs}"


# ---------------------------------------------------------------------------
# sounddevice_source.py
# ---------------------------------------------------------------------------


def _make_sd_source(queue_size: int = 10, **overrides) -> SoundDeviceSource:
    src = SoundDeviceSource.__new__(SoundDeviceSource)
    src.device_id = 1
    src.channels = 1
    src.samplerate = 48000
    src.blocksize = 4096
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._stream = None
    src._running = False
    src._lock = threading.Lock()
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = "mic"
    for key, value in overrides.items():
        setattr(src, key, value)
    return src


class TestSoundDeviceSourceLifecycle:
    def test_lifecycle_debug_trail(self, caplog):
        src = _make_sd_source()
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            return_value={"name": "Test Mic (WASAPI)"},
        ):
            with caplog.at_level(logging.DEBUG, logger=SD_LOG):
                src.start()
                src.stop()

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("source_start" in m for m in debug_msgs), debug_msgs
        assert any("source_stop" in m for m in debug_msgs), debug_msgs

    def test_lifecycle_info_quiet(self, caplog):
        src = _make_sd_source()
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            return_value={"name": "Test Mic (WASAPI)"},
        ):
            with caplog.at_level(logging.INFO, logger=SD_LOG):
                src.start()
                src.stop()

        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert info_msgs, "expected the pinned stream-open INFO record"
        assert any("SoundDevice stream opened" in m for m in info_msgs)
        assert not any(
            "source_start" in m or "source_stop" in m for m in info_msgs
        ), f"lifecycle DEBUG events leaked to INFO: {info_msgs}"

    def test_callback_stats_debug_only(self, caplog):
        src = _make_sd_source()
        buf = np.zeros((1024, 1), dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger=SD_LOG):
            # Empty queue: exercises the enqueue path (stats at bucket 1).
            src._callback(buf, 1024, {}, 0)

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("callback_stats" in m for m in debug_msgs), (
            f"no per-callback/poll stats DEBUG record: {debug_msgs}"
        )

        caplog.clear()
        with caplog.at_level(logging.INFO, logger=SD_LOG):
            src._callback(buf, 1024, {}, 0)
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert not any(
            "callback_stats" in m for m in info_msgs
        ), f"callback stats leaked to INFO: {info_msgs}"

    def test_lifecycle_endpoint_id_redacted(self, caplog):
        src = _make_sd_source(device_id=PRIVACY_ENDPOINT_ID)
        with patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.InputStream",
            MagicMock(),
        ), patch(
            "meetandread.audio.capture.sounddevice_source.sounddevice.query_devices",
            return_value={"name": "Test Mic (WASAPI)"},
        ):
            with caplog.at_level(logging.DEBUG, logger=SD_LOG):
                src.start()
                src.stop()

        joined = [r.getMessage() for r in caplog.records]
        for msg in joined:
            assert PRIVACY_ENDPOINT_ID not in msg, (
                f"raw endpoint id leaked: {msg!r}"
            )
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        start_msgs = [m for m in debug_msgs if "source_start" in m]
        stop_msgs = [m for m in debug_msgs if "source_stop" in m]
        assert start_msgs and any("<redacted:" in m for m in start_msgs), (
            f"hashed id expected in source_start: {start_msgs}"
        )
        assert stop_msgs and any("<redacted:" in m for m in stop_msgs), (
            f"hashed id expected in source_stop: {stop_msgs}"
        )


# ---------------------------------------------------------------------------
# pyaudiowpatch_source.py
# ---------------------------------------------------------------------------


def _make_paw_source(queue_size: int = 10, **overrides) -> PyAudioWPatchSource:
    src = PyAudioWPatchSource.__new__(PyAudioWPatchSource)
    src.device_index = 5
    src.channels = 2
    src.samplerate = 48000
    src.blocksize = 1024
    src.dtype = "float32"
    src._queue = queue.Queue(maxsize=queue_size)
    src._stream = None
    src._running = False
    src._lock = threading.Lock()
    src._frames_dropped = 0
    src._frames_enqueued = 0
    src._consecutive_frames_dropped = 0
    src._max_consecutive_frames_dropped = 0
    src._on_frame_dropped = None
    src._source_label = "system"
    src._lossy_callbacks = 0
    src._pyaudio = MagicMock()
    src._pyaudio.get_device_info_by_index.return_value = {
        "name": "Loopback Test Device"
    }
    for key, value in overrides.items():
        setattr(src, key, value)
    return src


class TestPyAudioWPatchSourceLifecycle:
    def test_lifecycle_debug_trail(self, caplog):
        src = _make_paw_source()
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            src.start()
            src.stop()

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("source_start" in m for m in debug_msgs), debug_msgs
        assert any("source_stop" in m for m in debug_msgs), debug_msgs

    def test_lifecycle_info_quiet(self, caplog):
        src = _make_paw_source()
        with caplog.at_level(logging.INFO, logger=PAW_LOG):
            src.start()
            src.stop()

        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert not any(
            "source_start" in m or "source_stop" in m for m in info_msgs
        ), f"lifecycle DEBUG events leaked to INFO: {info_msgs}"

    def test_callback_data_loss_status_warns(self, caplog):
        src = _make_paw_source()
        src._running = True
        buf = np.zeros((1024, 2), dtype=np.float32).tobytes()
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            src._callback(buf, 1024, None, 2)  # paInputOverflow: samples discarded

        warn_msgs = _messages_at(caplog.records, logging.WARNING)
        assert any(
            "data-loss" in m.lower() for m in warn_msgs
        ), f"data-loss status flag not visible at WARNING: {warn_msgs}"

    def test_callback_data_loss_warning_rate_limited(self, caplog):
        src = _make_paw_source()
        src._running = True
        buf = np.zeros((1024, 2), dtype=np.float32).tobytes()
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            for _ in range(3):
                src._callback(buf, 1024, None, 2)

        warn_msgs = [
            m for m in _messages_at(caplog.records, logging.WARNING)
            if "data-loss" in m.lower()
        ]
        assert len(warn_msgs) == 2, (
            f"expected power-of-two bucketing (streak 1,2 of 3), got {warn_msgs}"
        )

    def test_callback_data_loss_streak_resets_on_clean(self, caplog):
        src = _make_paw_source()
        src._running = True
        buf = np.zeros((1024, 2), dtype=np.float32).tobytes()
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            src._callback(buf, 1024, None, 2)
            src._callback(buf, 1024, None, 2)
            src._callback(buf, 1024, None, 0)  # clean callback ends the streak
            src._callback(buf, 1024, None, 2)

        warn_msgs = [
            m for m in _messages_at(caplog.records, logging.WARNING)
            if "data-loss" in m.lower()
        ]
        assert len(warn_msgs) == 3, (
            f"expected warnings at streak 1, 2, then 1 after reset: {warn_msgs}"
        )

    def test_callback_non_loss_status_flag_stays_debug(self, caplog):
        src = _make_paw_source()
        src._running = True
        buf = np.zeros((1024, 2), dtype=np.float32).tobytes()
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            src._callback(buf, 1024, None, 16)  # paPrimingOutput: non-loss

        warn_msgs = _messages_at(caplog.records, logging.WARNING)
        assert not any(
            "status" in m.lower() for m in warn_msgs
        ), f"non-loss status flag escalated to WARNING: {warn_msgs}"
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("status flag" in m.lower() for m in debug_msgs), debug_msgs

    def test_open_failure_stays_error(self, caplog):
        src = _make_paw_source()
        src._pyaudio.open.side_effect = OSError("device gone")
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            with pytest.raises(OSError):
                src.start()

        error_msgs = _messages_at(caplog.records, logging.ERROR)
        assert any("loopback stream" in m.lower() for m in error_msgs), error_msgs

    def test_lifecycle_endpoint_id_redacted(self, caplog):
        src = _make_paw_source(device_index=PRIVACY_ENDPOINT_ID)
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            src.start()
            src.stop()
            src.close()

        joined = [r.getMessage() for r in caplog.records]
        for msg in joined:
            assert PRIVACY_ENDPOINT_ID not in msg, (
                f"raw endpoint id leaked: {msg!r}"
            )
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        start_msgs = [m for m in debug_msgs if "source_start" in m]
        assert start_msgs and any("<redacted:" in m for m in start_msgs), (
            f"hashed id expected in source_start: {start_msgs}"
        )

    def test_open_failure_error_redacts_endpoint_id(self, caplog):
        src = _make_paw_source(device_index=PRIVACY_ENDPOINT_ID)
        src._pyaudio.open.side_effect = OSError("device gone")
        with caplog.at_level(logging.DEBUG, logger=PAW_LOG):
            with pytest.raises(OSError):
                src.start()

        joined = [r.getMessage() for r in caplog.records]
        for msg in joined:
            assert PRIVACY_ENDPOINT_ID not in msg, (
                f"raw endpoint id leaked: {msg!r}"
            )
        error_msgs = _messages_at(caplog.records, logging.ERROR)
        assert any("loopback stream" in m.lower() for m in error_msgs), error_msgs
        assert any("<redacted:" in m for m in error_msgs), error_msgs


# ---------------------------------------------------------------------------
# fake_module.py
# ---------------------------------------------------------------------------


class TestFakeModuleLifecycle:
    def test_fake_source_lifecycle_debug_only(self, tmp_path, caplog):
        wav = _write_wav(tmp_path)
        src = FakeAudioModule(wav_path=wav, blocksize=1024, queue_size=10)
        with caplog.at_level(logging.DEBUG, logger=FAKE_LOG):
            src.start()
            src.read_frames(timeout=0.2)
            src.stop()

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("fake_source_start" in m for m in debug_msgs), debug_msgs
        assert any("fake_source_stop" in m for m in debug_msgs), debug_msgs

    def test_fake_source_no_info_events(self, tmp_path, caplog):
        wav = _write_wav(tmp_path)
        src = FakeAudioModule(wav_path=wav, blocksize=1024, queue_size=10)
        with caplog.at_level(logging.INFO, logger=FAKE_LOG):
            src.start()
            src.read_frames(timeout=0.2)
            src.stop()

        assert _messages_at(caplog.records, logging.INFO) == [], (
            "fake source (test double) must not emit INFO events"
        )


# ---------------------------------------------------------------------------
# windows_device_monitor.py
# ---------------------------------------------------------------------------


class TestWindowsDeviceMonitorEvents:
    def _monitor(self) -> WindowsDeviceMonitor:
        return WindowsDeviceMonitor(platform_name="Windows", comtypes_backend=object())

    def test_injected_events_log_info_with_redaction(self, caplog):
        monitor = self._monitor()
        event = DeviceEvent(
            event_type=DeviceEventType.ADDED,
            device_id="dev-9",
            friendly_name=PRIVACY_NAME,
        )
        with caplog.at_level(logging.INFO, logger=MON_LOG):
            monitor.inject_event(event)

        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert any(
            "device_event" in m and "added" in m for m in info_msgs
        ), f"hot-plug device add not logged at INFO: {info_msgs}"
        _assert_no_raw_name(caplog.records)

    def test_poll_internals_debug_not_info(self, caplog):
        monitor = self._monitor()
        with caplog.at_level(logging.DEBUG, logger=MON_LOG):
            monitor.inject_event(
                DeviceEvent(event_type="added", device_id="dev-a")
            )
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("event_queued" in m for m in debug_msgs), debug_msgs

        caplog.clear()
        with caplog.at_level(logging.INFO, logger=MON_LOG):
            monitor.inject_event(
                DeviceEvent(event_type="added", device_id="dev-b")
            )
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert not any(
            "event_queued" in m for m in info_msgs
        ), f"queue/poll internal leaked to INFO: {info_msgs}"


# ---------------------------------------------------------------------------
# session.py
# ---------------------------------------------------------------------------


class TestSessionLifecycleEvents:
    def _start_stop(self, caplog, tmp_path, level, seconds=0.3):
        wav = _write_wav(tmp_path, seconds=seconds)
        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=wav, loop=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            max_frames=int(seconds * 16000),
        )
        session = AudioSession()
        with caplog.at_level(level, logger=SESS_LOG):
            session.start(config)
            session.stop()

    def test_session_lifecycle_debug_trail(self, tmp_path, caplog):
        self._start_stop(caplog, tmp_path, logging.DEBUG)
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("source_start" in m for m in debug_msgs), debug_msgs
        assert any("source_stop" in m for m in debug_msgs), debug_msgs
        assert any("stream_open" in m for m in debug_msgs), debug_msgs
        assert any("stream_close" in m for m in debug_msgs), debug_msgs
        assert any("consumer_loop_started" in m for m in debug_msgs), debug_msgs

    def test_session_frame_accounting_debug_firehose(self, tmp_path, caplog):
        """Frame accounting fires periodically under a fast-emitting source."""
        import time as _time

        wav = _write_wav(tmp_path, seconds=2.0)
        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=wav, loop=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
            max_frames=16000 * 30,  # 30s cap: never reached, firehose runs
        )
        session = AudioSession()
        try:
            with caplog.at_level(logging.DEBUG, logger=SESS_LOG):
                session.start(config)
                # The fake source emits faster than real-time; 200 emitting
                # rounds accumulate well inside a couple of seconds.
                deadline = _time.monotonic() + 8.0
                while _time.monotonic() < deadline:
                    if any(
                        "session_frame_accounting" in r.getMessage()
                        for r in caplog.records
                    ):
                        break
                    _time.sleep(0.05)
        finally:
            session.stop()

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("session_frame_accounting" in m for m in debug_msgs), (
            f"periodic frame accounting never fired: {debug_msgs[-8:]}"
        )

    def test_session_start_stop_operational_info(self, tmp_path, caplog):
        self._start_stop(caplog, tmp_path, logging.INFO)
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert any("session_start" in m for m in info_msgs), (
            f"session start is an operational fact and must appear at INFO: {info_msgs}"
        )
        assert any("session_stop" in m for m in info_msgs), info_msgs

    def test_session_info_quiet_on_debug_spam(self, tmp_path, caplog):
        self._start_stop(caplog, tmp_path, logging.INFO)
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert not any(
            "session_frame_accounting" in m for m in info_msgs
        ), f"periodic frame accounting leaked to INFO: {info_msgs}"
        assert not any(
            "stream_open" in m or "stream_close" in m for m in info_msgs
        ), f"per-step events leaked to INFO: {info_msgs}"


class TestSessionSwapEvent:
    def test_source_swap_logged_at_debug(self, tmp_path, caplog):
        wav = _write_wav(tmp_path)
        config = SessionConfig(
            sources=[SourceConfig(type="fake", fake_path=wav, loop=True)],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
        )
        session = AudioSession()
        session.start(config)
        new_source = FakeAudioModule(wav_path=wav, loop=True)
        new_wrapper = AudioSourceWrapper(
            new_source,
            SourceConfig(type="fake", fake_path=wav, loop=True),
            target_rate=16000,
            target_channels=1,
        )
        try:
            with caplog.at_level(logging.DEBUG, logger=SESS_LOG):
                session.swap_source("fake", new_wrapper)
        finally:
            session.stop()

        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("source_swap" in m for m in debug_msgs), debug_msgs


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------


class TestCliDebugDetail:
    def test_fake_record_debug_steps(self, tmp_path, caplog):
        from meetandread.audio import cli

        wav = _write_wav(tmp_path)
        args = cli.create_parser().parse_args(
            ["record", "--fake", wav, "--seconds", "0.3", "--output-dir", str(tmp_path)]
        )
        with caplog.at_level(logging.DEBUG, logger=CLI_LOG):
            rc = cli.cmd_record(args)

        assert rc == 0
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("source_resolution" in m for m in debug_msgs), debug_msgs
        assert any("finalization" in m for m in debug_msgs), debug_msgs

    def test_mic_record_debug_enumeration(self, tmp_path, caplog):
        from meetandread.audio import cli

        args = cli.create_parser().parse_args(
            ["record", "--mic", "--seconds", "0.3", "--output-dir", str(tmp_path)]
        )
        with patch(
            "meetandread.audio.cli.list_mic_inputs",
            return_value=[{"index": 0, "name": "Audit Mic"}],
        ), patch(
            "meetandread.audio.cli.list_loopback_outputs",
            return_value=[{"index": 1, "name": "Audit Speakers", "loopback_ok": True}],
        ), patch(
            "meetandread.audio.session.MicSource"
        ) as mock_mic_cls:
            fake_source = MagicMock()
            fake_source.get_metadata.return_value = {
                "sample_rate": 16000,
                "channels": 1,
            }
            mock_mic_cls.return_value = fake_source
            with caplog.at_level(logging.DEBUG, logger=CLI_LOG):
                rc = cli.cmd_record(args)

        assert rc == 0
        debug_msgs = _messages_at(caplog.records, logging.DEBUG)
        assert any("device_enumeration" in m for m in debug_msgs), debug_msgs

    def test_fake_record_info_progress_intact(self, tmp_path, caplog):
        from meetandread.audio import cli

        wav = _write_wav(tmp_path)
        args = cli.create_parser().parse_args(
            ["record", "--fake", wav, "--seconds", "0.3", "--output-dir", str(tmp_path)]
        )
        with caplog.at_level(logging.INFO, logger=CLI_LOG):
            rc = cli.cmd_record(args)

        assert rc == 0
        info_msgs = _messages_at(caplog.records, logging.INFO)
        assert any("Recording from fake source" in m for m in info_msgs)
        assert any("Recording complete" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Cross-module privacy sweep
# ---------------------------------------------------------------------------


class TestPrivacySweep:
    def test_selection_path_never_logs_raw_name(self, tmp_path, caplog):
        """Distinctive device name flows through selection; never logged raw."""
        mic = {
            "index": 2,
            "name": PRIVACY_NAME,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
            "hostapi": 0,
        }
        with patch(
            "meetandread.audio.capture.devices._HAS_PYAUDIOWPATCH", False
        ), patch(
            "meetandread.audio.capture.devices.list_devices",
            return_value=[mic],
        ), patch(
            "meetandread.audio.capture.devices.get_wasapi_hostapi_index",
            return_value=0,
        ):
            with caplog.at_level(logging.DEBUG, logger=DEV_LOG):
                print_device_summary()
                list_loopback_outputs()

        _assert_no_raw_name(caplog.records)
        assert any(
            "<redacted:" in r.getMessage() for r in caplog.records
        ), "redacted identifier form expected somewhere in the records"

    def test_system_source_init_never_logs_raw_name(self, caplog):
        """PRIVACY: SystemSource.__init__ success path redacts the loopback name."""
        mock_loopback_info = {
            "index": 42,
            "name": PRIVACY_NAME,
            "maxInputChannels": 2,
            "defaultSampleRate": 48000.0,
        }
        fake_pa_instance = MagicMock()
        fake_pa_instance.__enter__ = MagicMock(return_value=fake_pa_instance)
        fake_pa_instance.__exit__ = MagicMock(return_value=False)
        fake_pa_instance.get_default_wasapi_loopback.return_value = mock_loopback_info
        fake_paw = MagicMock()
        fake_paw.PyAudio.return_value = fake_pa_instance

        with patch.dict("sys.modules", {"pyaudiowpatch": fake_paw}), patch(
            "meetandread.audio.capture.pyaudiowpatch_source._HAS_PYAUDIOWPATCH",
            True,
        ), patch(
            "meetandread.audio.capture.pyaudiowpatch_source.PyAudioWPatchSource",
            MagicMock(),
        ):
            with caplog.at_level(logging.DEBUG, logger=SD_LOG):
                src = SystemSource()

        assert src.available is True
        _assert_no_raw_name(caplog.records)
        assert any(
            "<redacted:" in r.getMessage() for r in caplog.records
        ), "redacted identifier expected in SystemSource init records"

    def test_redaction_helper_is_stable_and_short(self):
        redacted = _redact_device_name(PRIVACY_NAME)
        assert redacted.startswith("<redacted:")
        assert re.fullmatch(r"<redacted:[0-9a-f]{8}>", redacted), redacted
