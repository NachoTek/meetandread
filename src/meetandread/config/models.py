"""Settings dataclasses for meetandread configuration.

Provides type-safe configuration models with serialization support.
All settings use dataclasses with from_dict/to_dict methods for JSON persistence.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict, List


@dataclass
class ModelSettings:
    """Configuration for Whisper model selection.
    
    Attributes:
        realtime_model_size: Model size for real-time transcription.
            Options: "tiny", "base", "small", "auto"
            Default: "auto" (triggers hardware detection on first run)
    """
    realtime_model_size: str = field(
        default="auto",
        metadata={"description": "Model size for real-time transcription: tiny, base, small, or auto"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSettings":
        """Create from dictionary, using defaults for missing fields."""
        return cls(
            realtime_model_size=data.get("realtime_model_size", cls.realtime_model_size)
        )


@dataclass
class TranscriptionSettings:
    """Configuration for transcription behavior.
    
    HYBRID TRANSCRIPTION SETTINGS:
    - Real-time: Uses tiny model for immediate display (no agreement buffer)
    - Post-processing: Uses stronger model after recording stops
    
    Attributes:
        enabled: Whether transcription is enabled.
            Default: True
        confidence_threshold: Confidence threshold for display coloring.
            Range: 0.0 to 1.0
            Default: 0.7
        min_chunk_size_sec: Minimum audio chunk size for VAD/processing.
            Default: 0.5 (seconds) - reduced for lower latency
        agreement_threshold: DEPRECATED - no longer used (real-time commits immediately)
            Default: 1
        
        # Model selection for hybrid transcription
        realtime_model_size: Model size for real-time transcription.
            Options: "tiny", "base", "small"
            Default: "tiny" (fastest for real-time)
        postprocess_model_size: Model size for post-processing.
            Options: "base", "small", "medium", "large"
            Default: "base" (better accuracy for archive)
        enable_postprocessing: Whether to run post-processing after recording.
            Default: True
    """
    enabled: bool = field(
        default=True,
        metadata={"description": "Whether transcription is enabled"}
    )
    confidence_threshold: float = field(
        default=0.7,
        metadata={"description": "Confidence threshold for display coloring (0.0-1.0)"}
    )
    min_chunk_size_sec: float = field(
        default=0.5,  # Reduced from 1.0s for lower latency (target < 2s total)
        metadata={"description": "Minimum audio chunk size in seconds"}
    )
    agreement_threshold: int = field(
        default=1,  # DEPRECATED - real-time commits immediately
        metadata={"description": "DEPRECATED: No longer used in hybrid transcription"}
    )
    
    # HYBRID TRANSCRIPTION: Model selection
    realtime_model_size: str = field(
        default="tiny",
        metadata={"description": "Model size for real-time transcription: tiny, base, or small"}
    )
    postprocess_model_size: str = field(
        default="base",
        metadata={"description": "Model size for post-processing: base, small, medium, or large"}
    )
    enable_postprocessing: bool = field(
        default=True,
        metadata={"description": "Enable post-processing with stronger model after recording"}
    )
    benchmark_history: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"description": "Per-model benchmark results: {model_size: {wer: float, timestamp: str}}"}
    )
    
    # MICROPHONE DENOISING SETTINGS
    microphone_denoising_enabled: bool = field(
        default=False,
        metadata={"description": "Whether microphone denoising is enabled"}
    )
    microphone_denoising_provider: str = field(
        default="spectral_gate",
        metadata={"description": "Denoising provider name (e.g. 'spectral_gate'). Invalid names fall back to default at runtime."}
    )
    microphone_denoising_latency_budget_ms: int = field(
        default=200,
        metadata={"description": "Maximum allowed latency in ms per denoising call (diagnostic, not enforced)"}
    )

    # CC OVERLAY SETTINGS
    cc_font_size: int = field(
        default=48,
        metadata={"description": "CC overlay font size in pixels (range: 16-96)"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionSettings":
        """Create from dictionary, using defaults for missing fields."""
        return cls(
            enabled=data.get("enabled", cls.enabled),
            confidence_threshold=data.get("confidence_threshold", cls.confidence_threshold),
            min_chunk_size_sec=data.get("min_chunk_size_sec", cls.min_chunk_size_sec),
            agreement_threshold=data.get("agreement_threshold", cls.agreement_threshold),
            realtime_model_size=data.get("realtime_model_size", cls.realtime_model_size),
            postprocess_model_size=data.get("postprocess_model_size", cls.postprocess_model_size),
            enable_postprocessing=data.get("enable_postprocessing", cls.enable_postprocessing),
            benchmark_history=data.get("benchmark_history", {}),
            microphone_denoising_enabled=data.get("microphone_denoising_enabled", False),
            microphone_denoising_provider=data.get("microphone_denoising_provider", "spectral_gate"),
            microphone_denoising_latency_budget_ms=data.get("microphone_denoising_latency_budget_ms", 200),
            cc_font_size=data.get("cc_font_size", 48),
        )


@dataclass
class HardwareSettings:
    """Configuration for hardware detection and model recommendations.
    
    Attributes:
        auto_detect_on_startup: Whether to auto-detect hardware on first run.
            Default: True
        last_detected_ram_gb: Last detected RAM in GB (cached).
            Default: None
        last_detected_cpu_count: Last detected CPU core count (cached).
            Default: None
        recommended_model: Cached model recommendation from hardware detection.
            Default: None
        user_override_model: User's explicit model override (if set, ignores recommendation).
            Default: None
    """
    auto_detect_on_startup: bool = field(
        default=True,
        metadata={"description": "Auto-detect hardware capabilities on first run"}
    )
    last_detected_ram_gb: Optional[float] = field(
        default=None,
        metadata={"description": "Cached RAM detection result in GB"}
    )
    last_detected_cpu_count: Optional[int] = field(
        default=None,
        metadata={"description": "Cached CPU core count detection result"}
    )
    recommended_model: Optional[str] = field(
        default=None,
        metadata={"description": "Cached model recommendation from hardware detection"}
    )
    user_override_model: Optional[str] = field(
        default=None,
        metadata={"description": "User's explicit model size override"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareSettings":
        """Create from dictionary, using defaults for missing fields."""
        return cls(
            auto_detect_on_startup=data.get("auto_detect_on_startup", cls.auto_detect_on_startup),
            last_detected_ram_gb=data.get("last_detected_ram_gb"),
            last_detected_cpu_count=data.get("last_detected_cpu_count"),
            recommended_model=data.get("recommended_model"),
            user_override_model=data.get("user_override_model")
        )


@dataclass
class UISettings:
    """Configuration for UI behavior and appearance.
    
    Attributes:
        show_confidence_legend: Whether to show the confidence color legend.
            Default: True
        transcript_auto_scroll: Whether transcript panel auto-scrolls.
            Default: True
        widget_position: Last known widget position as (x, y) tuple.
            Default: None (will use default positioning)
        widget_dock_edge: Which edge the widget is docked to.
            Options: "left", "right", "top", "bottom", None
            Default: None
    """
    show_confidence_legend: bool = field(
        default=True,
        metadata={"description": "Show confidence color coding legend in UI"}
    )
    transcript_auto_scroll: bool = field(
        default=True,
        metadata={"description": "Auto-scroll transcript panel to show new content"}
    )
    widget_position: Optional[Tuple[int, int]] = field(
        default=None,
        metadata={"description": "Last widget position as (x, y) tuple"}
    )
    widget_dock_edge: Optional[str] = field(
        default=None,
        metadata={"description": "Widget dock edge: left, right, top, bottom, or None"}
    )
    audio_sources: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Persisted audio source selection: list of 'mic' and/or 'system'"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Handle tuple serialization
        if result.get("widget_position") is not None:
            result["widget_position"] = list(result["widget_position"])
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UISettings":
        """Create from dictionary, using defaults for missing fields."""
        pos = data.get("widget_position")
        if pos is not None and isinstance(pos, list):
            pos = tuple(pos)
        
        return cls(
            show_confidence_legend=data.get("show_confidence_legend", cls.show_confidence_legend),
            transcript_auto_scroll=data.get("transcript_auto_scroll", cls.transcript_auto_scroll),
            widget_position=pos,
            widget_dock_edge=data.get("widget_dock_edge"),
            audio_sources=data.get("audio_sources"),
        )


@dataclass
class SpeakerSettings:
    """Configuration for speaker diarization and identification.

    Attributes:
        enabled: Whether speaker diarization is enabled.
            Default: True
        confidence_threshold: Minimum cosine similarity to identify a
            known speaker (0.0–1.0). Default: 0.6
        clustering_threshold: Threshold for fast clustering in diarization
            (0–1). Higher values produce more speakers. Default: 0.5
        min_duration_on: Minimum speech segment duration in seconds.
            Shorter segments are discarded. Default: 0.3
        min_duration_off: Minimum silence gap between speakers in seconds.
            Shorter gaps are merged, reducing false splits in noisy rooms.
            Default: 0.8 (tuned for noisy meeting environments)
    """
    enabled: bool = field(
        default=True,
        metadata={"description": "Whether speaker diarization is enabled"}
    )
    confidence_threshold: float = field(
        default=0.6,
        metadata={"description": "Min cosine similarity for speaker identification (0.0-1.0)"}
    )
    clustering_threshold: float = field(
        default=0.5,
        metadata={"description": "Clustering threshold for diarization (higher = more speakers)"}
    )
    min_duration_on: float = field(
        default=0.3,
        metadata={"description": "Minimum speech segment duration in seconds (shorter segments discarded)"}
    )
    min_duration_off: float = field(
        default=0.8,
        metadata={"description": "Minimum silence gap between speakers in seconds (reduces false splits in noisy rooms)"}
    )

    @staticmethod
    def _coerce_float(value: object, fallback: float, min_val: float = 0.0, max_val: float = 60.0) -> float:
        """Coerce a value to float, clamping to [min_val, max_val] with fallback."""
        if not isinstance(value, (int, float)):
            return fallback
        try:
            f = float(value)
        except (TypeError, ValueError):
            return fallback
        if f < min_val or f > max_val:
            return fallback
        return f

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerSettings":
        """Create from dictionary, using defaults for missing fields.

        Non-numeric, negative, or out-of-range min_duration values are
        silently replaced with safe defaults rather than raising.
        """
        raw_on = data.get("min_duration_on", cls.min_duration_on)
        raw_off = data.get("min_duration_off", cls.min_duration_off)
        safe_on = cls._coerce_float(raw_on, cls.min_duration_on, min_val=0.0, max_val=30.0)
        safe_off = cls._coerce_float(raw_off, cls.min_duration_off, min_val=0.0, max_val=30.0)

        return cls(
            enabled=data.get("enabled", cls.enabled),
            confidence_threshold=data.get("confidence_threshold", cls.confidence_threshold),
            clustering_threshold=data.get("clustering_threshold", cls.clustering_threshold),
            min_duration_on=safe_on,
            min_duration_off=safe_off,
        )


@dataclass
class AppSettings:
    """Root container for all application settings.
    
    This is the top-level settings object that contains all configuration
    categories. It handles versioning for migration support.
    
    Attributes:
        config_version: Configuration schema version for migrations.
            Current: 1
        model: Model selection settings.
        transcription: Transcription behavior settings.
        hardware: Hardware detection and recommendation settings.
        ui: UI behavior and appearance settings.
    """
    config_version: int = field(
        default=6,
        metadata={"description": "Configuration schema version for migrations"}
    )
    model: ModelSettings = field(default_factory=ModelSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    ui: UISettings = field(default_factory=UISettings)
    speaker: SpeakerSettings = field(default_factory=SpeakerSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_version": self.config_version,
            "model": self.model.to_dict(),
            "transcription": self.transcription.to_dict(),
            "hardware": self.hardware.to_dict(),
            "ui": self.ui.to_dict(),
            "speaker": self.speaker.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        """Create from dictionary, using defaults for missing/invalid fields."""
        # Handle nested settings with defaults if missing
        model_data = data.get("model", {})
        transcription_data = data.get("transcription", {})
        hardware_data = data.get("hardware", {})
        ui_data = data.get("ui", {})
        speaker_data = data.get("speaker", {})
        
        return cls(
            config_version=data.get("config_version", cls.config_version),
            model=ModelSettings.from_dict(model_data) if isinstance(model_data, dict) else ModelSettings(),
            transcription=TranscriptionSettings.from_dict(transcription_data) if isinstance(transcription_data, dict) else TranscriptionSettings(),
            hardware=HardwareSettings.from_dict(hardware_data) if isinstance(hardware_data, dict) else HardwareSettings(),
            ui=UISettings.from_dict(ui_data) if isinstance(ui_data, dict) else UISettings(),
            speaker=SpeakerSettings.from_dict(speaker_data) if isinstance(speaker_data, dict) else SpeakerSettings(),
        )

    @classmethod
    def get_defaults(cls) -> "AppSettings":
        """Get a fresh instance with all default values."""
        return cls()
