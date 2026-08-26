"""Audio capture module for meetandread.

Provides device enumeration and audio capture sources (mic, system, fake).
"""

from .devices import (
    list_devices,
    get_wasapi_hostapi_index,
    get_default_loopback_device,
    list_mic_inputs,
    list_loopback_outputs,
    can_open_mic,
)

from .sounddevice_source import (
    SoundDeviceSource,
    MicSource,
    SystemSource,
    AudioSourceError,
    NonWasapiDeviceError,
)

from .fake_module import (
    FakeAudioModule,
    FakeAudioSource,  # Compatibility alias
)

from .pyaudiowpatch_source import PyAudioWPatchSource

__all__ = [
    # Device enumeration
    "list_devices",
    "get_wasapi_hostapi_index",
    "get_default_loopback_device",
    "list_mic_inputs",
    "list_loopback_outputs",
    "can_open_mic",
    # Capture sources
    "SoundDeviceSource",
    "MicSource",
    "SystemSource",
    "PyAudioWPatchSource",
    "FakeAudioModule",
    "FakeAudioSource",
    # Exceptions
    "AudioSourceError",
    "NonWasapiDeviceError",
]
