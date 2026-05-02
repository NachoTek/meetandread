"""Audio capture and storage module for meetandread.

Provides high-level recording session management, low-level capture sources,
and crash-safe storage primitives.

High-level API (AudioSession):
    from meetandread.audio import AudioSession, SessionConfig, SourceConfig
    
    config = SessionConfig(sources=[SourceConfig(type='mic')])
    session = AudioSession()
    session.start(config)
    wav_path = session.stop()

Low-level capture:
    from meetandread.audio import MicSource, SystemSource, FakeAudioModule
    
    source = MicSource()
    source.start()
    frames = source.read_frames()
    source.stop()

Storage:
    from meetandread.audio import PcmPartWriter, finalize_part_to_wav
    
    writer = PcmPartWriter.create(stem="test")
    writer.write_frames_i16(audio_bytes)
    writer.close()
    wav_path = finalize_part_to_wav(writer.part_path, writer.metadata_path)
"""

# Session management (high-level API)
from meetandread.audio.session import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionStats,
    DenoisingStats,
    SessionState,
    SessionError,
    NoSourcesError,
)

# Capture sources (low-level API)
from meetandread.audio.capture import (
    MicSource,
    SystemSource,
    FakeAudioModule,
    FakeAudioSource,  # Compatibility alias
    AudioSourceError,
    NonWasapiDeviceError,
    list_devices,
    get_wasapi_hostapi_index,
    list_mic_inputs,
    list_loopback_outputs,
)

# Storage primitives
from meetandread.audio.storage import (
    PcmPartWriter,
    PcmMetadata,
    load_metadata,
    finalize_part_to_wav,
    finalize_stem,
    get_recordings_dir,
    get_transcripts_dir,
    new_recording_stem,
    find_part_files,
    recover_part_files,
    has_partial_recordings,
)

__all__ = [
    # Session management
    "AudioSession",
    "SessionConfig",
    "SourceConfig",
    "SessionStats",
    "DenoisingStats",
    "SessionState",
    "SessionError",
    "NoSourcesError",
    # Capture sources
    "MicSource",
    "SystemSource",
    "FakeAudioModule",
    "FakeAudioSource",
    "AudioSourceError",
    "NonWasapiDeviceError",
    "list_devices",
    "get_wasapi_hostapi_index",
    "list_mic_inputs",
    "list_loopback_outputs",
    # Storage
    "PcmPartWriter",
    "PcmMetadata",
    "load_metadata",
    "finalize_part_to_wav",
    "finalize_stem",
    "get_recordings_dir",
    "get_transcripts_dir",
    "new_recording_stem",
    "find_part_files",
    "recover_part_files",
    "has_partial_recordings",
]
