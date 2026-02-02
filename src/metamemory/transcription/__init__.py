"""Real-time transcription engine with faster-whisper.

Provides the core transcription pipeline:
- AudioRingBuffer: Thread-safe audio buffering
- VADChunkingProcessor: Intelligent audio segmentation
- LocalAgreementBuffer: Prevents text flickering
- WhisperTranscriptionEngine: Whisper model inference with confidence
- Confidence scoring and color coding
"""

from metamemory.transcription.audio_buffer import AudioRingBuffer
from metamemory.transcription.vad_processor import VADChunkingProcessor
from metamemory.transcription.local_agreement import LocalAgreementBuffer
from metamemory.transcription.engine import (
    WhisperTranscriptionEngine,
    TranscriptionSegment,
    WordInfo,
)
from metamemory.transcription.confidence import (
    normalize_confidence,
    get_confidence_level,
    get_confidence_color,
    get_distortion_intensity,
    get_confidence_legend,
    format_confidence_for_display,
    ConfidenceLevel,
    ConfidenceLegendItem,
)

__all__ = [
    "AudioRingBuffer",
    "VADChunkingProcessor",
    "LocalAgreementBuffer",
    "WhisperTranscriptionEngine",
    "TranscriptionSegment",
    "WordInfo",
    # Confidence scoring
    "normalize_confidence",
    "get_confidence_level",
    "get_confidence_color",
    "get_distortion_intensity",
    "get_confidence_legend",
    "format_confidence_for_display",
    "ConfidenceLevel",
    "ConfidenceLegendItem",
]
