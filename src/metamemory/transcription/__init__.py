"""Hybrid transcription engine with whisper.cpp.

HYBRID TRANSCRIPTION DESIGN:
- Real-time: Tiny model for immediate display (no agreement buffer blocking)
- Post-process: Stronger model for enhanced accuracy after recording stops

Provides the core transcription pipeline:
- AudioRingBuffer: Thread-safe audio buffering
- VADChunkingProcessor: Intelligent audio segmentation
- WhisperTranscriptionEngine: Whisper model inference with confidence (whisper.cpp backend)
- PostProcessingQueue: Background enhancement after recording stops
- TranscriptStore: Word-level storage with confidence for UI color coding
- Confidence scoring and color coding for visual feedback
- AccumulatingTranscriptionProcessor: Accumulating audio processor for meetings

Uses whisper.cpp via pywhispercpp for CPU-only operation without PyTorch DLL dependencies.
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
from metamemory.transcription.post_processor import (
    PostProcessingQueue,
    PostProcessJob,
    PostProcessStatus,
)
from metamemory.transcription.transcript_store import (
    TranscriptStore,
    Word,
    Segment,
)
from metamemory.transcription.accumulating_processor import (
    AccumulatingTranscriptionProcessor,
    SegmentResult,
)

__all__ = [
    # Core components
    "AudioRingBuffer",
    "VADChunkingProcessor",
    "LocalAgreementBuffer",
    "WhisperTranscriptionEngine",
    "TranscriptionSegment",
    "WordInfo",
    # Hybrid transcription
    "PostProcessingQueue",
    "PostProcessJob",
    "PostProcessStatus",
    "TranscriptStore",
    "Word",
    "Segment",
    # Accumulating processor (for meetings)
    "AccumulatingTranscriptionProcessor",
    "SegmentResult",
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
