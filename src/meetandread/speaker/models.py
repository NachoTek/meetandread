"""Data models for the speaker diarization and identification pipeline.

Immutable dataclasses representing segments, embeddings, profiles, and
diarization results. These models decouple the sherpa-onnx API from the
rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SpeakerSegment:
    """A single speaker-labeled time segment from diarization.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Raw speaker label from diarization (e.g. "spk0", "spk1").
    """

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class VoiceSignature:
    """A speaker embedding vector with metadata.

    Attributes:
        embedding: Float32 embedding vector (typically 256-dim for eres2net).
        speaker_label: The diarization label this was extracted from (e.g. "spk0").
        num_segments: Number of audio segments averaged to produce this embedding.
    """

    embedding: np.ndarray
    speaker_label: str
    num_segments: int = 1


@dataclass
class SpeakerProfile:
    """A known speaker with a name and stored voice signatures.

    Attributes:
        name: Human-readable speaker name (e.g. "Alice").
        embedding: Averaged embedding vector for this speaker.
        num_samples: Number of segments that contributed to the average.
    """

    name: str
    embedding: np.ndarray
    num_samples: int = 1


@dataclass(frozen=True)
class SpeakerMatch:
    """Result of matching an embedding against known speakers.

    Attributes:
        name: Matched speaker name.
        score: Cosine similarity score (0.0 to 1.0).
        confidence: "high", "medium", or "low" based on score thresholds.
    """

    name: str
    score: float
    confidence: str = "medium"

    def __post_init__(self) -> None:
        valid = ("high", "medium", "low")
        if self.confidence not in valid:
            raise ValueError(f"confidence must be one of {valid}, got '{self.confidence}'")


@dataclass
class DiarizationResult:
    """Complete output from speaker diarization.

    Attributes:
        segments: Time-ordered speaker segments.
        signatures: Per-speaker voice signatures (keyed by raw label, e.g. "spk0").
        matches: Speaker identification results (keyed by raw label). May be empty
                 if no known speakers exist in the voice signature store.
        duration_seconds: Total audio duration processed.
        num_speakers: Number of distinct speakers detected.
        error: If diarization failed, a descriptive error message. None on success.
    """

    segments: List[SpeakerSegment] = field(default_factory=list)
    signatures: Dict[str, VoiceSignature] = field(default_factory=dict)
    matches: Dict[str, SpeakerMatch] = field(default_factory=dict)
    duration_seconds: float = 0.0
    num_speakers: int = 0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if diarization completed without error."""
        return self.error is None

    def speaker_label_for(self, raw_label) -> str:
        """Return the display label for a raw speaker label.

        If the speaker was identified via voice matching, returns the known name.
        Otherwise returns the raw label formatted as "SPK_0", "SPK_1", etc.

        Note: raw_label may be an int (from sherpa-onnx) or a string (from
        JSON deserialization after subprocess round-trip).  Both int 1 and
        string "1" should resolve to the same match entry.
        """
        # Normalize: try exact match first, then string coercion
        if raw_label in self.matches:
            return self.matches[raw_label].name
        str_label = str(raw_label)
        if str_label in self.matches:
            return self.matches[str_label].name
        # speaker may be an int from sherpa-onnx, or a string like "spk0"
        if isinstance(raw_label, int):
            return f"SPK_{raw_label}"
        # Convert "spk0" -> "SPK_0", "spk1" -> "SPK_1"
        try:
            idx = int("".join(c for c in str(raw_label) if c.isdigit()))
            return f"SPK_{idx}"
        except (ValueError, StopIteration):
            return str(raw_label).upper()
