"""Tests for confidence scoring module.

Covers: normalize_confidence, get_confidence_level, get_confidence_color,
get_distortion_intensity, get_confidence_legend, format_confidence_for_display
"""

import pytest
import sys
sys.path.insert(0, 'src')

from metamemory.transcription.confidence import (
    normalize_confidence,
    get_confidence_level,
    get_confidence_color,
    get_distortion_intensity,
    get_confidence_legend,
    format_confidence_for_display,
    ConfidenceLevel,
    CONFIDENCE_COLORS,
)


class TestNormalizeConfidence:
    """Test confidence normalization from Whisper log_probs."""
    
    def test_high_confidence_max(self):
        """Test -1.0 returns 95 (max)."""
        assert normalize_confidence(-1.0) == 95
    
    def test_low_confidence_min(self):
        """Test -3.0 returns 30 (min)."""
        assert normalize_confidence(-3.0) == 30
    
    def test_midpoint(self):
        """Test -2.0 returns midpoint (~62)."""
        result = normalize_confidence(-2.0)
        assert 60 <= result <= 65
    
    def test_clamp_above_high(self):
        """Test values above -1.0 clamp to 95."""
        assert normalize_confidence(-0.5) == 95
        assert normalize_confidence(0.0) == 95
    
    def test_clamp_below_low(self):
        """Test values below -3.0 clamp to 30."""
        assert normalize_confidence(-3.5) == 30
        assert normalize_confidence(-5.0) == 30
    
    def test_linear_interpolation(self):
        """Test linear interpolation between bounds."""
        # At -2.5 (75% of the way from -3.0 to -1.0)
        # Expected: 30 + 0.75 * 65 = 78.75 -> 78 or 79
        result = normalize_confidence(-2.5)
        assert 45 <= result <= 50
        
        # At -1.5 (25% of the way from -3.0 to -1.0)
        # Expected: 30 + 0.25 * 65 = 46.25 -> 46
        result = normalize_confidence(-1.5)
        assert 75 <= result <= 80


class TestConfidenceLevel:
    """Test confidence level categorization."""
    
    def test_high_level(self):
        """Test 80-100% is HIGH."""
        assert get_confidence_level(100) == ConfidenceLevel.HIGH
        assert get_confidence_level(95) == ConfidenceLevel.HIGH
        assert get_confidence_level(80) == ConfidenceLevel.HIGH
    
    def test_medium_level(self):
        """Test 70-80% is MEDIUM."""
        assert get_confidence_level(79) == ConfidenceLevel.MEDIUM
        assert get_confidence_level(75) == ConfidenceLevel.MEDIUM
        assert get_confidence_level(70) == ConfidenceLevel.MEDIUM
    
    def test_low_level(self):
        """Test 50-70% is LOW."""
        assert get_confidence_level(69) == ConfidenceLevel.LOW
        assert get_confidence_level(60) == ConfidenceLevel.LOW
        assert get_confidence_level(50) == ConfidenceLevel.LOW
    
    def test_very_low_level(self):
        """Test 0-50% is VERY_LOW."""
        assert get_confidence_level(49) == ConfidenceLevel.VERY_LOW
        assert get_confidence_level(25) == ConfidenceLevel.VERY_LOW
        assert get_confidence_level(0) == ConfidenceLevel.VERY_LOW


class TestConfidenceColor:
    """Test color coding by confidence."""
    
    def test_high_color(self):
        """Test HIGH is green (#4CAF50)."""
        assert get_confidence_color(85) == "#4CAF50"
        assert get_confidence_color(100) == "#4CAF50"
    
    def test_medium_color(self):
        """Test MEDIUM is yellow (#FFC107)."""
        assert get_confidence_color(75) == "#FFC107"
        assert get_confidence_color(70) == "#FFC107"
    
    def test_low_color(self):
        """Test LOW is orange (#FF9800)."""
        assert get_confidence_color(60) == "#FF9800"
        assert get_confidence_color(50) == "#FF9800"
    
    def test_very_low_color(self):
        """Test VERY_LOW is red (#F44336)."""
        assert get_confidence_color(40) == "#F44336"
        assert get_confidence_color(0) == "#F44336"


class TestDistortionIntensity:
    """Test visual distortion calculation."""
    
    def test_no_distortion_at_85(self):
        """Test 85% has 0 distortion."""
        assert get_distortion_intensity(85) == 0.0
    
    def test_no_distortion_above_85(self):
        """Test >85% has 0 distortion."""
        assert get_distortion_intensity(90) == 0.0
        assert get_distortion_intensity(100) == 0.0
    
    def test_max_distortion_at_0(self):
        """Test 0% has max distortion (0.7)."""
        assert get_distortion_intensity(0) == 0.7
    
    def test_linear_interpolation(self):
        """Test linear interpolation between 85% and 0%."""
        # At 42.5 (50% of the way from 0 to 85)
        # Expected: 0.7 * 0.5 = 0.35
        result = get_distortion_intensity(42)
        assert 0.3 <= result <= 0.4
        
        # At 85% threshold, should be 0
        result = get_distortion_intensity(85)
        assert result == 0.0
    
    def test_distortion_increases_as_confidence_decreases(self):
        """Test distortion increases as confidence decreases."""
        assert get_distortion_intensity(80) < get_distortion_intensity(60)
        assert get_distortion_intensity(60) < get_distortion_intensity(30)
        assert get_distortion_intensity(30) < get_distortion_intensity(0)


class TestConfidenceLegend:
    """Test confidence legend generation."""
    
    def test_legend_has_four_items(self):
        """Test legend has 4 confidence levels."""
        legend = get_confidence_legend()
        assert len(legend) == 4
    
    def test_legend_ordered_high_to_low(self):
        """Test legend is ordered from HIGH to VERY_LOW."""
        legend = get_confidence_legend()
        assert legend[0].level == ConfidenceLevel.HIGH
        assert legend[1].level == ConfidenceLevel.MEDIUM
        assert legend[2].level == ConfidenceLevel.LOW
        assert legend[3].level == ConfidenceLevel.VERY_LOW
    
    def test_legend_has_correct_colors(self):
        """Test legend items have correct colors."""
        legend = get_confidence_legend()
        assert legend[0].color == CONFIDENCE_COLORS[ConfidenceLevel.HIGH]
        assert legend[1].color == CONFIDENCE_COLORS[ConfidenceLevel.MEDIUM]
        assert legend[2].color == CONFIDENCE_COLORS[ConfidenceLevel.LOW]
        assert legend[3].color == CONFIDENCE_COLORS[ConfidenceLevel.VERY_LOW]
    
    def test_legend_has_range_strings(self):
        """Test legend items have range strings."""
        legend = get_confidence_legend()
        assert "80-100%" in legend[0].range_str
        assert "70-80%" in legend[1].range_str
        assert "50-70%" in legend[2].range_str
        assert "0-50%" in legend[3].range_str


class TestFormatConfidenceForDisplay:
    """Test confidence formatting for UI."""
    
    def test_format_returns_dict(self):
        """Test function returns dictionary."""
        result = format_confidence_for_display(85, "hello")
        assert isinstance(result, dict)
    
    def test_format_includes_all_fields(self):
        """Test result includes all required fields."""
        result = format_confidence_for_display(85, "test text")
        assert "text" in result
        assert "confidence" in result
        assert "color" in result
        assert "level" in result
        assert "distortion" in result
    
    def test_format_preserves_text(self):
        """Test text is preserved in output."""
        result = format_confidence_for_display(85, "original text")
        assert result["text"] == "original text"
    
    def test_format_includes_correct_values(self):
        """Test values are calculated correctly."""
        result = format_confidence_for_display(85, "test")
        assert result["confidence"] == 85
        assert result["color"] == "#4CAF50"
        assert result["level"] == "high"
        assert result["distortion"] == 0.0
    
    def test_format_very_low_confidence(self):
        """Test formatting for very low confidence."""
        result = format_confidence_for_display(40, "uncertain")
        assert result["color"] == "#F44336"
        assert result["level"] == "very_low"
        assert result["distortion"] > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
