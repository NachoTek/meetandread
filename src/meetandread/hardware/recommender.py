"""Model recommendation engine based on hardware specifications.

Provides intelligent model size recommendations (tiny/base/small/medium/large)
based on detected system specs. Integrates with ConfigManager to save recommendations.
"""

from dataclasses import dataclass
from typing import List, Optional

from meetandread.hardware.detector import HardwareDetector, SystemSpecs


@dataclass
class ModelInfo:
    """Information about a Whisper model size.
    
    Attributes:
        size: Model size identifier ('tiny', 'base', 'small', etc.)
        ram_required_gb: Estimated RAM required in GB
        description: Human-readable description
        accuracy_rating: Accuracy description ('Basic', 'Good', 'Best')
        latency_profile: Latency description ('Fastest', 'Fast', 'Moderate')
    """
    size: str
    ram_required_gb: float
    description: str
    accuracy_rating: str
    latency_profile: str


# Model specifications from RESEARCH.md
MODEL_SPECS = {
    "tiny": ModelInfo(
        size="tiny",
        ram_required_gb=0.2,  # ~200MB
        description="Tiny model - fastest, lowest accuracy, good for low-end hardware",
        accuracy_rating="Basic",
        latency_profile="Fastest",
    ),
    "base": ModelInfo(
        size="base",
        ram_required_gb=0.5,  # ~500MB
        description="Base model - good balance of speed and accuracy",
        accuracy_rating="Good",
        latency_profile="Fast",
    ),
    "small": ModelInfo(
        size="small",
        ram_required_gb=1.5,  # ~1.5GB
        description="Small model - recommended for < 2s latency with best accuracy",
        accuracy_rating="Very Good",
        latency_profile="Moderate",
    ),
    "medium": ModelInfo(
        size="medium",
        ram_required_gb=2.0,  # ~2GB
        description="Medium model — high accuracy, slower processing",
        accuracy_rating="High",
        latency_profile="Slow",
    ),
    "large": ModelInfo(
        size="large",
        ram_required_gb=3.0,  # ~3GB
        description="Large model — best accuracy, slowest processing",
        accuracy_rating="Best",
        latency_profile="Slowest",
    ),
}


def recommend_model_size(specs: SystemSpecs, prefer_accuracy: bool = False) -> str:
    """Recommend Whisper model size based on hardware specs.
    
    Recommendation algorithm optimized for real-time transcription (< 2s latency):
    - For real-time: Always recommend "tiny" (fastest, ~0.5-1s latency)
    - For post-processing: Can use larger models (base/small/medium/large)
    
    Current approach prioritizes real-time performance over accuracy:
    - If RAM < 6GB OR CPU < 4 cores: recommend "tiny" (required for low-end)
    - Otherwise: recommend "tiny" (for < 2s latency target)
    
    NOTE: Post-processing architecture uses tiny for real-time
    and larger models for post-recording transcription.
    
    Args:
        specs: System hardware specifications
        prefer_accuracy: If True, may recommend larger model if borderline
        
    Returns:
        Recommended model size: 'tiny', 'base', 'small', 'medium', or 'large'
        
    Example:
        >>> specs = SystemSpecs(
        ...     total_ram_gb=16, available_ram_gb=8,
        ...     cpu_count_logical=8, cpu_count_physical=4,
        ...     cpu_freq_mhz=2400, is_64bit=True, platform='Windows'
        ... )
        >>> recommend_model_size(specs)
        'small'
        >>> specs2 = SystemSpecs(
        ...     total_ram_gb=4, available_ram_gb=2,
        ...     cpu_count_logical=2, cpu_count_physical=2,
        ...     cpu_freq_mhz=2000, is_64bit=True, platform='Windows'
        ... )
        >>> recommend_model_size(specs2)
        'tiny'
    """
    # For real-time transcription, always use tiny for < 2s latency
    # Larger models (base/small) can be used for post-processing enhancement
    if specs.total_ram_gb < 6.0 or specs.cpu_count_logical < 4:
        return "tiny"
    
    # Even on high-end hardware, use tiny for real-time
    # Post-processing will use larger models for post-recording transcription
    return "tiny"


def get_model_info(size: str) -> ModelInfo:
    """Get specifications for a model size.
    
    Args:
        size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        
    Returns:
        ModelInfo for the specified size
        
    Raises:
        ValueError: If size is not recognized
    """
    if size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size: {size}. Valid: {list(MODEL_SPECS.keys())}")
    return MODEL_SPECS[size]


def get_all_model_info() -> List[ModelInfo]:
    """Get information for all available model sizes.
    
    Returns:
        List of ModelInfo objects for all models
    """
    return list(MODEL_SPECS.values())


class ModelRecommender:
    """Intelligent model size recommendation with config integration.
    
    Provides model recommendations based on hardware detection and manages
    saving recommendations to the configuration system. Handles user overrides
    and supports re-detection for hardware upgrades.
    
    Example:
        >>> recommender = ModelRecommender()
        >>> rec = recommender.detect_and_recommend()
        >>> print(f"Recommended: {rec}")
        Recommended: small
        >>>
        >>> # Get recommendation info
        >>> info = recommender.get_recommended_info()
        >>> print(info.description)
        Small model - recommended for < 2s latency with best accuracy
    """
    
    def __init__(
        self,
        hardware_detector: Optional[HardwareDetector] = None,
    ):
        """Initialize the model recommender.
        
        Args:
            hardware_detector: Optional HardwareDetector instance.
                If None, creates a new one.
        """
        self._detector = hardware_detector or HardwareDetector()
        self._recommended_size: Optional[str] = None
        self._detected_specs: Optional[SystemSpecs] = None
    
    def detect_and_recommend(self, prefer_accuracy: bool = False) -> str:
        """Detect hardware and generate recommendation.
        
        This method re-runs hardware detection and updates the recommendation.
        Use this when hardware may have changed (e.g., after RAM upgrade).
        
        Args:
            prefer_accuracy: If True, may recommend larger model if borderline
            
        Returns:
            Recommended model size: 'tiny', 'base', or 'small'
        """
        # Force fresh detection
        self._detected_specs = self._detector.refresh()
        self._recommended_size = recommend_model_size(self._detected_specs, prefer_accuracy)
        
        return self._recommended_size
    
    def get_recommendation(self) -> str:
        """Get current model recommendation.
        
        If no recommendation has been made yet, runs detection first.
        
        Returns:
            Recommended model size: 'tiny', 'base', or 'small'
        """
        if self._recommended_size is None:
            return self.detect_and_recommend()
        return self._recommended_size
    
    def get_recommended_info(self) -> ModelInfo:
        """Get ModelInfo for the current recommendation.
        
        Returns:
            ModelInfo for the recommended model size
        """
        size = self.get_recommendation()
        return get_model_info(size)
    
    def get_detected_specs(self) -> SystemSpecs:
        """Get the hardware specs from last detection.
        
        Returns:
            SystemSpecs from most recent detection
            
        Raises:
            RuntimeError: If detection has not been run yet
        """
        if self._detected_specs is None:
            self._detected_specs = self._detector.detect()
        return self._detected_specs
    
    def save_recommendation_to_config(self) -> bool:
        """Save the current recommendation to configuration.
        
        Saves:
        - hardware.recommended_model: The recommended size
        - hardware.last_detected_ram_gb: Detected RAM
        - hardware.last_detected_cpu_count: Detected CPU count
        
        Returns:
            True if saved successfully, False otherwise
            
        Raises:
            RuntimeError: If config system is not available
        """
        try:
            from meetandread.config import set_config, save_config
            
            if self._recommended_size is None:
                self.detect_and_recommend()
            
            # Save recommendation
            set_config("hardware.recommended_model", self._recommended_size)
            
            # Save detected specs
            if self._detected_specs:
                set_config("hardware.last_detected_ram_gb", self._detected_specs.total_ram_gb)
                set_config("hardware.last_detected_cpu_count", self._detected_specs.cpu_count_logical)
            
            # Persist
            return save_config()
        except ImportError:
            raise RuntimeError("Config system not available - cannot save recommendation")
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.getLogger(__name__).error(f"Failed to save recommendation: {e}")
            return False
    
    def check_user_override(self) -> Optional[str]:
        """Check if user has overridden the recommendation.
        
        Returns:
            User's override model size if set, None otherwise
        """
        try:
            from meetandread.config import get_config
            override = get_config("hardware.user_override_model")
            return override
        except (ImportError, ValueError):
            return None
    
    def set_user_override(self, size: Optional[str]) -> bool:
        """Set or clear user override for model size.
        
        Args:
            size: Model size to override with, or None to clear override
            
        Returns:
            True if saved successfully
            
        Raises:
            ValueError: If size is not a valid model size
        """
        if size is not None and size not in MODEL_SPECS:
            raise ValueError(f"Invalid model size: {size}. Valid: {list(MODEL_SPECS.keys())}")
        
        try:
            from meetandread.config import set_config, save_config
            set_config("hardware.user_override_model", size)
            return save_config()
        except ImportError:
            raise RuntimeError("Config system not available")
    
    def get_effective_model_size(self) -> str:
        """Get the effective model size considering user override.
        
        If user has set an override, returns that. Otherwise returns
        the hardware-based recommendation.
        
        Returns:
            Effective model size to use: 'tiny', 'base', or 'small'
        """
        override = self.check_user_override()
        if override is not None:
            return override
        return self.get_recommendation()
    
    def get_recommendation_summary(self) -> str:
        """Get a human-readable summary of the recommendation.
        
        Returns:
            Summary string explaining the recommendation
        """
        specs = self.get_detected_specs()
        recommended = self.get_recommendation()
        info = get_model_info(recommended)
        override = self.check_user_override()
        
        lines = [
            f"Hardware: {specs.total_ram_gb:.1f}GB RAM, {specs.cpu_count_logical} cores",
            f"Recommended model: {recommended}",
            f"  - {info.description}",
            f"  - Accuracy: {info.accuracy_rating}, Speed: {info.latency_profile}",
        ]
        
        if override:
            lines.append(f"User override: {override} (recommended: {recommended})")
        
        return "\n".join(lines)
