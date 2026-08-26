"""Tests for the mic open-probe (issue #71).

``can_open_mic()`` must answer the question CI actually needs: can a
microphone input stream be *opened* on this machine? Enumeration
(``list_mic_inputs()``) succeeds on GitHub-hosted runners while opening
the enumerated device fails, so hardware-gated tests must probe by
opening, not by enumerating.
"""
from __future__ import annotations

from unittest import mock

import pytest

from meetandread.audio.capture import devices


class TestCanOpenMic:
    """can_open_mic probes the device-open path, not enumeration."""

    def test_returns_false_when_no_devices_enumerated(self):
        with mock.patch.object(devices, "list_mic_inputs", return_value=[]):
            assert devices.can_open_mic() is False

    def test_returns_false_when_open_fails(self):
        """Enumeration succeeds but the stream cannot open (CI runner)."""
        fake_device = {"index": 0, "max_input_channels": 1,
                       "default_samplerate": 48000}
        with mock.patch.object(devices, "list_mic_inputs",
                               return_value=[fake_device]), \
             mock.patch.object(
                 devices.sounddevice, "InputStream",
                 side_effect=OSError("Device unavailable")):
            assert devices.can_open_mic() is False

    def test_returns_true_when_stream_opens(self):
        fake_device = {"index": 0, "max_input_channels": 1,
                       "default_samplerate": 48000}
        stream = mock.MagicMock()
        with mock.patch.object(devices, "list_mic_inputs",
                               return_value=[fake_device]), \
             mock.patch.object(devices.sounddevice, "InputStream",
                               return_value=stream):
            assert devices.can_open_mic() is True
        stream.start.assert_called_once()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_returns_true_stops_after_first_openable_device(self):
        """A broken first device must not mask a working second one."""
        broken = {"index": 0, "max_input_channels": 1,
                  "default_samplerate": 48000}
        working = {"index": 1, "max_input_channels": 2,
                   "default_samplerate": 44100}
        stream = mock.MagicMock()
        real_input_stream = devices.sounddevice.InputStream

        def _input_stream(*args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if device == 0:
                raise OSError("Device unavailable")
            return stream

        with mock.patch.object(devices, "list_mic_inputs",
                               return_value=[broken, working]), \
             mock.patch.object(devices.sounddevice, "InputStream",
                               side_effect=_input_stream):
            assert devices.can_open_mic() is True
        assert real_input_stream is devices.sounddevice.InputStream

    def test_never_raises(self):
        """Unexpected backend errors degrade to False, never propagate."""
        with mock.patch.object(devices, "list_mic_inputs",
                               side_effect=RuntimeError("backend died")):
            assert devices.can_open_mic() is False
