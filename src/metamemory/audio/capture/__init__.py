"""Audio capture module for metamemory.

Provides device enumeration and audio capture sources (mic, system, fake).
"""

from .devices import (
    list_devices,
    get_wasapi_hostapi_index,
    list_mic_inputs,
    list_loopback_outputs,
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
    "list_mic_inputs",
    "list_loopback_outputs",
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
