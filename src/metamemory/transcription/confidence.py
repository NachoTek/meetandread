"""Confidence scoring and color coding for transcription segments.

Provides confidence normalization from Whisper log probabilities to 0-100 scale,
color coding by confidence level, and visual distortion effects for low confidence text.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class ConfidenceLevel(Enum):
    """Confidence level categories with associated color coding.
    
    Levels:
        HIGH: 80-100% (green) - High confidence, clear audio
        MEDIUM: 70-80% (yellow) - Medium confidence
        LOW: 50-70% (orange) - Low confidence, potential issues
        VERY_LOW: 0-50% (red) - Very low confidence, may be inaccurate
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ConfidenceLegendItem:
    """Item for confidence legend display.
    
    Attributes:
        level: The confidence level enum
        color: Hex color code for this level
        range_str: Human-readable range (e.g., "80-100%")
        description: Description of what this confidence means
    """
    level: ConfidenceLevel
    color: str
    range_str: str
    description: str


# Color codes for confidence levels
CONFIDENCE_COLORS = {
    ConfidenceLevel.HIGH: "#4CAF50",      # Green
    ConfidenceLevel.MEDIUM: "#FFC107",    # Yellow
    ConfidenceLevel.LOW: "#FF9800",       # Orange
    ConfidenceLevel.VERY_LOW: "#F44336",  # Red
}

# Confidence level thresholds (inclusive lower bound)
CONFIDENCE_THRESHOLDS = [
    (80, ConfidenceLevel.HIGH),
    (70, ConfidenceLevel.MEDIUM),
    (50, ConfidenceLevel.LOW),
    (0, ConfidenceLevel.VERY_LOW),
]

# Visual distortion parameters
DISTORTION_NO_EFFECT_THRESHOLD = 85  # No distortion at 85% or above
DISTORTION_MAX_CONFIDENCE = 0        # 0% confidence = max distortion
DISTORTION_MAX_INTENSITY = 0.7       # Cap at 0.7 to keep text readable


def normalize_confidence(avg_log_prob: float) -> int:
    """Normalize Whisper's avg_log_prob to 0-100 scale.
    
    Whisper log probabilities:
    - -1.0: High confidence (best case)
    - -2.0: Medium confidence
    - -3.0: Low confidence (worst case)
    
    Mapping:
    - log_prob > -1.0 -> 95 (clamped to max)
    - log_prob = -1.0 -> 95
    - log_prob = -2.0 -> 62 (midpoint)
    - log_prob = -3.0 -> 30
    - log_prob < -3.0 -> 30 (clamped to min)
    
    Args:
        avg_log_prob: Average log probability from Whisper segment
        
    Returns:
        Confidence score 0-100 (integer)
        
    Example:
        >>> normalize_confidence(-1.0)
        95
        >>> normalize_confidence(-2.0)
        62
        >>> normalize_confidence(-3.0)
        30
    """
    # Whisper log_prob range: [-3.0, -1.0]
    LOGPROB_LOW = -3.0
    LOGPROB_HIGH = -1.0
    SCORE_LOW = 30
    SCORE_HIGH = 95
    
    # Clamp to range
    if avg_log_prob > LOGPROB_HIGH:
        return SCORE_HIGH
    elif avg_log_prob < LOGPROB_LOW:
        return SCORE_LOW
    
    # Linear interpolation: map [LOGPROB_LOW, LOGPROB_HIGH] to [SCORE_LOW, SCORE_HIGH]
    # normalized = (avg_log_prob - LOGPROB_LOW) / (LOGPROB_HIGH - LOGPROB_LOW)
    # result = SCORE_LOW + normalized * (SCORE_HIGH - SCORE_LOW)
    normalized = (avg_log_prob - LOGPROB_LOW) / (LOGPROB_HIGH - LOGPROB_LOW)
    return int(SCORE_LOW + normalized * (SCORE_HIGH - SCORE_LOW))


def get_confidence_level(confidence: int) -> ConfidenceLevel:
    """Get the confidence level category for a confidence score.
    
    Args:
        confidence: Confidence score 0-100
        
    Returns:
        ConfidenceLevel enum value
        
    Example:
        >>> get_confidence_level(85)
        ConfidenceLevel.HIGH
        >>> get_confidence_level(75)
        ConfidenceLevel.MEDIUM
        >>> get_confidence_level(42)
        ConfidenceLevel.VERY_LOW
    """
    for threshold, level in CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return level
    return ConfidenceLevel.VERY_LOW


def get_confidence_color(confidence: int) -> str:
    """Get the hex color code for a confidence score.
    
    Args:
        confidence: Confidence score 0-100
        
    Returns:
        Hex color code string (e.g., "#4CAF50")
        
    Example:
        >>> get_confidence_color(85)
        '#4CAF50'
        >>> get_confidence_color(75)
        '#FFC107'
        >>> get_confidence_color(42)
        '#F44336'
    """
    level = get_confidence_level(confidence)
    return CONFIDENCE_COLORS[level]


def get_distortion_intensity(confidence: int) -> float:
    """Calculate visual distortion intensity for low confidence text.
    
    Distortion is used to visually indicate uncertainty in transcription.
    Higher distortion = more uncertainty = more wavy/blur effect.
    
    Parameters:
        - No effect at 85% or above (0.0 intensity)
        - Maximum effect at 0% confidence (capped at 0.7 for readability)
        - Linear interpolation between 85% and 0%
    
    Args:
        confidence: Confidence score 0-100
        
    Returns:
        Distortion intensity 0.0 to 1.0 (capped at 0.7 for readability)
        
    Example:
        >>> get_distortion_intensity(85)
        0.0
        >>> get_distortion_intensity(42)
        0.51
        >>> get_distortion_intensity(0)
        0.7
    """
    if confidence >= DISTORTION_NO_EFFECT_THRESHOLD:
        return 0.0
    
    # Linear interpolation from [0, 85] to [DISTORTION_MAX_INTENSITY, 0]
    # At confidence=85: 0.0
    # At confidence=0: DISTORTION_MAX_INTENSITY (0.7)
    intensity = DISTORTION_MAX_INTENSITY * (1.0 - (confidence / DISTORTION_NO_EFFECT_THRESHOLD))
    return min(intensity, DISTORTION_MAX_INTENSITY)


def get_confidence_legend() -> List[ConfidenceLegendItem]:
    """Get the full confidence legend for UI display.
    
    Returns list of legend items ordered from highest to lowest confidence.
    Each item includes the confidence level, color, range string, and description.
    
    Returns:
        List of ConfidenceLegendItem objects, ordered high to low
        
    Example:
        >>> legend = get_confidence_legend()
        >>> legend[0].description
        'High confidence - clear audio'
        >>> len(legend)
        4
    """
    return [
        ConfidenceLegendItem(
            level=ConfidenceLevel.HIGH,
            color=CONFIDENCE_COLORS[ConfidenceLevel.HIGH],
            range_str="80-100%",
            description="High confidence - clear audio",
        ),
        ConfidenceLegendItem(
            level=ConfidenceLevel.MEDIUM,
            color=CONFIDENCE_COLORS[ConfidenceLevel.MEDIUM],
            range_str="70-80%",
            description="Medium confidence",
        ),
        ConfidenceLegendItem(
            level=ConfidenceLevel.LOW,
            color=CONFIDENCE_COLORS[ConfidenceLevel.LOW],
            range_str="50-70%",
            description="Low confidence - potential issues",
        ),
        ConfidenceLegendItem(
            level=ConfidenceLevel.VERY_LOW,
            color=CONFIDENCE_COLORS[ConfidenceLevel.VERY_LOW],
            range_str="0-50%",
            description="Very low confidence - may be inaccurate",
        ),
    ]


def format_confidence_for_display(confidence: int, text: str) -> dict:
    """Format confidence information for UI display.
    
    Combines all confidence-related data into a single dict for easy
    consumption by UI components.
    
    Args:
        confidence: Confidence score 0-100
        text: The transcribed text
        
    Returns:
        Dictionary with keys:
        - text: Original text
        - confidence: Score 0-100
        - color: Hex color code
        - level: ConfidenceLevel name (e.g., "high")
        - distortion: Distortion intensity 0.0-1.0
        
    Example:
        >>> result = format_confidence_for_display(85, "hello world")
        >>> result['color']
        '#4CAF50'
        >>> result['level']
        'high'
        >>> result['distortion']
        0.0
    """
    level = get_confidence_level(confidence)
    return {
        "text": text,
        "confidence": confidence,
        "color": get_confidence_color(confidence),
        "level": level.value,
        "distortion": get_distortion_intensity(confidence),
    }
